import sys
import os
import time
import glob
from flask import Flask, render_template, jsonify, request
import threading
# Add path to core
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

import numpy as np
from core.mcts import MCTSEngine


app = Flask(__name__)

# Global Game State
class GameState:
    def __init__(self):
        self.board = np.zeros((19, 19), dtype=int)
        self.current_player = 1 # 1: Black, -1: White
        self.move_history = []
        self.engine = None
        self.current_engine_path = None
        self.lock = threading.Lock()
        
    def init_engine(self, engine_path):
        if self.current_engine_path != engine_path:
            print(f"Loading Engine: {engine_path}")
            # Force device to cuda:0 for now
            with self.lock:
                self.engine = MCTSEngine(engine_path, device='cuda:0')
                # Sync state
                self.engine.reset()
                for r, c in self.move_history:
                    idx = r * 19 + c
                    self.engine.update_state(idx)
            self.current_engine_path = engine_path
            
    def reset(self):
        with self.lock:
            self.board = np.zeros((19, 19), dtype=int)
            self.current_player = 1
            self.move_history = []
            if self.engine:
                self.engine.reset()
            
    def check_win(self, r, c, color):
        # Simple 6-in-a-row check
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        for dr, dc in directions:
            count = 1
            # Forward
            nr, nc = r + dr, c + dc
            while 0 <= nr < 19 and 0 <= nc < 19 and self.board[nr][nc] == color:
                count += 1
                nr += dr
                nc += dc
            # Backward
            nr, nc = r - dr, c - dc
            while 0 <= nr < 19 and 0 <= nc < 19 and self.board[nr][nc] == color:
                count += 1
                nr -= dr
                nc -= dc
            if count >= 6:
                return True
        return False

    def play(self, r, c):
        with self.lock:
            if self.board[r, c] != 0:
                return None # Invalid
                
            color = self.current_player
            self.board[r, c] = color
            self.move_history.append((r, c))
            
            # Update Engine
            move_idx = r * 19 + c
            if self.engine:
                self.engine.update_state(move_idx)
                
            # Check Win
            if self.check_win(r, c, color):
                return "win"
                
            # Switch player logic (Connect6)
            total = len(self.move_history)
            if total == 1:
                self.current_player = -1
            else:
                if (total - 1) % 2 == 0:
                    self.current_player *= -1
                    
            return "continue"

game = GameState()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/reset', methods=['POST'])
def reset():
    game.reset()
    return jsonify({'status': 'ok'})

@app.route('/state')
def get_state():
    return jsonify({
        'board': game.board.tolist(),
        'current_player': game.current_player,
        'history': game.move_history
    })

@app.route('/models')
def list_models():
    # List engines in checkpoints
    ckpt_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../reinforcement_learning/checkpoints'))
    models = []
    
    # Engines
    engines = glob.glob(os.path.join(ckpt_dir, "*.engine"))
    # Sort by mod time
    engines.sort(key=os.path.getmtime, reverse=True)
    
    for e in engines:
        name = os.path.basename(e)
        models.append({'name': name, 'path': e})
        
    return jsonify({'models': models})

@app.route('/move', methods=['POST'])
def human_move():
    data = request.json
    r, c = data['r'], data['c']
    
    res = game.play(r, c)
    if res:
        winner = None
        if res == "win":
            winner = game.current_player # The player who just moved
            
        return jsonify({
            'status': 'ok', 
            'board': game.board.tolist(), 
            'current_player': game.current_player,
            'game_over': res == "win",
            'winner': winner
        })
    else:
        return jsonify({'status': 'error', 'msg': 'Invalid move'}), 400

@app.route('/ai_move', methods=['POST'])
def ai_move():
    data = request.json or {}
    
    # Params
    sims = int(data.get('sims', 800))
    temp = float(data.get('temp', 0.1)) # Default low temp for "best" move but allowing some exploration
    model_path = data.get('model_path')
    batch_size = int(data.get('batch_size', 32))
    num_threads = int(data.get('num_threads', 4))
    
    # Load Engine if needed
    if model_path and os.path.exists(model_path):
        game.init_engine(model_path)
    elif not game.engine:
        # Default
        ckpt_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../reinforcement_learning/checkpoints/current_model.engine'))
        if not os.path.exists(ckpt_path):
             return jsonify({'status': 'error', 'msg': 'Engine not found'}), 500
        game.init_engine(ckpt_path)
        
    # Set Params
    if game.engine:
        game.engine.set_params(batch_size=batch_size, num_threads=num_threads)
        
    # Run MCTS
    # Note: MCTS engine call is blocking and thread-safe internally (if compiled with OpenMP/atomics),
    # but we need to protect game state (board, engine status).
    # However, locking for 10 seconds blocks 'reset'.
    # We accept that 'reset' waits.
    
    best_move = -1
    with game.lock:
        # Check if engine still valid (reset might have happened if we didn't lock above?)
        # Actually, since we lock here, 'reset' can't run.
        # But we didn't lock during 'get params'.
        # Ideally, we lock around the whole critical section.
        if not game.engine: return jsonify({'status': 'error', 'msg': 'Engine lost'}), 500
        
        # Release GIL for C++? Ctypes releases GIL.
        # But we hold game.lock.
        start = time.time()
        best_move = game.engine.get_mcts_move(simulations=sims, temperature=temp)
        duration = time.time() - start
        
        # Get stats
        win_rate = game.engine.get_win_rate()
        
        # Get Policy (Search Probabilities)
        policy_array = game.engine.get_policy() # array of 361
    
    # Get Top N moves
    top_n = 5
    # Sort indices by prob descending
    top_indices = np.argsort(policy_array)[::-1][:top_n]
    
    debug_moves = []
    for idx in top_indices:
        prob = policy_array[idx]
        if prob < 0.001: continue
        r_m, c_m = idx // 19, idx % 19
        coord = f"{chr(ord('A') + c_m)}{19 - r_m}"
        debug_moves.append({'coord': coord, 'prob': float(prob)})
    
    # Apply move
    r = best_move // 19
    c = best_move % 19
    
    res = game.play(r, c) # play acquires lock internally. Safe.
    
    winner = None
    if res == "win":
        winner = game.current_player # AI just played, so AI (which was current) won
    
    return jsonify({
        'status': 'ok',
        'move': [int(r), int(c)],
        'board': game.board.tolist(), # Added missing board field
        'win_rate': float(win_rate),
        'duration': duration,
        'current_player': game.current_player,
        'debug_moves': debug_moves,
        'game_over': res == "win",
        'winner': winner
    })

if __name__ == '__main__':
    # Determine Engine Path
    ckpt = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../reinforcement_learning/checkpoints/current_model.engine'))
    if os.path.exists(ckpt):
        game.init_engine(ckpt)
    
    app.run(host='0.0.0.0', port=5000, debug=False)
