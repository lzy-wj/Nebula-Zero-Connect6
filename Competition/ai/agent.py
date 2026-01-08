import time
import threading
import numpy as np
import torch
import os
import sys

# Add parent path to import core
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../vit_resnet/phrase4')))

from core.mcts import MCTSEngine
# We assume Connect6Game logic is handled by the GUI/Main logic, 
# but the Agent needs to track state too.

class C6Agent:
    def __init__(self, engine_path, mode="mcts", device="cuda"):
        """
        mode: 'mcts' or 'policy'
        """
        self.mode = mode
        self.engine_path = engine_path
        self.device = device
        self.mcts = None
        self.thinking = False
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        
        # Initialize Engine
        try:
            self.mcts = MCTSEngine(engine_path, device=device)
            # Warmup
            with self.lock:
                self.mcts.run_mcts_simulations(1)
        except Exception as e:
            print(f"Failed to load AI Engine: {e}")
            
    def reset(self):
        if self.mcts:
            with self.lock:
                self.mcts.reset()
            
    def update_opponent_move(self, move_idx):
        """
        Update internal MCTS state with opponent's move.
        """
        # Must stop pondering before updating state to avoid race condition
        self.stop_pondering()
        
        if self.mcts:
            with self.lock:
                self.mcts.update_state(move_idx)
            
    def get_best_move(self, time_limit=5.0, simulations=None, heuristic=True):
        """
        Main thinking function.
        time_limit: seconds
        simulations: if set, ignores time_limit (fixed simulations)
        """
        if not self.mcts:
            return -1
            
        self.stop_event.clear()
        self.thinking = True
        
        start_time = time.time()
        
        # Mode 1: Pure Policy (Fast)
        if self.mode == 'policy':
            with self.lock:
                self.mcts.run_mcts_simulations(1)
                policy = self.mcts.get_policy()
            
            move = np.argmax(policy)
            self.thinking = False
            
            with self.lock:
                self.mcts.update_state(move)
            return move
            
        # Mode 2: MCTS
        # Loop for time
        step_sims = 50 # Run in small batches to allow interruption
        total_sims = 0
        
        while True:
            if self.stop_event.is_set():
                break
            
            with self.lock:
                self.mcts.run_mcts_simulations(step_sims)
            total_sims += step_sims
            
            # Check termination conditions
            if simulations is not None:
                if total_sims >= simulations:
                    break
            else:
                if (time.time() - start_time) >= time_limit:
                    break
        
        # Get result
        with self.lock:
            best_move = self.mcts.get_best_move(temperature=0.0)
            self.mcts.update_state(best_move)
            
        self.thinking = False
        return best_move
        
    def start_pondering(self):
        """
        Start thinking in background (Infinite loop until stop_pondering called).
        """
        if self.thinking: return
        
        def ponder_loop():
            self.stop_event.clear()
            self.thinking = True
            while not self.stop_event.is_set():
                with self.lock:
                    self.mcts.run_mcts_simulations(50)
                time.sleep(0.001) # Yield slightly
            self.thinking = False
            
        self.ponder_thread = threading.Thread(target=ponder_loop, daemon=True)
        self.ponder_thread.start()
        
    def stop_pondering(self):
        self.stop_event.set()
        if hasattr(self, 'ponder_thread') and self.ponder_thread.is_alive():
            # We don't want to join if we are in the thread itself (unlikely but possible in some designs)
            if threading.current_thread() != self.ponder_thread:
                self.ponder_thread.join(timeout=1.0)
