import sys
import os
import time
import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
import signal
import argparse

# === 1. Environment Limits (Prevent CPU Preemption) ===
os.environ["OMP_NUM_THREADS"] = "1" 
os.environ["OPENBLAS_NUM_THREADS"] = "1" 
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# Add paths (Local phrase4 context)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.connect6_game import Connect6Game
from core.mcts import MCTSEngine
import config
from data_refiner import DataRefiner  # Import Refiner

def encode_policy(policy_array):
    """
    Compress policy array to sparse string: idx:prob;idx:prob...
    Only keep probs > 0.001 to save space.
    """
    items = []
    for idx, prob in enumerate(policy_array):
        if prob > 0.001:
            items.append(f"{idx}:{prob:.4f}")
    return ";".join(items)

def worker_process(gpu_id, engine_path, sims, games_to_play, out_file, worker_id, lock, opponent_engine_path=None, mix_ratio=0.0):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    device = torch.device('cuda:0')
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    
    # Init Refiner
    refiner = DataRefiner()

    # Limit MCTS engine CPU threads (read from config)
    # os.environ['OMP_NUM_THREADS'] = str(config.MCTS_THREADS) # REMOVED: Do not override global OMP settings
    
    try:
        mcts = MCTSEngine(engine_path, device=device)
        # Use config for MCTS parameters
        if hasattr(config, 'MCTS_THREADS'):
            print(f"[Worker {worker_id}] Setting MCTS threads: {config.MCTS_THREADS} | Batch: {config.MCTS_BATCH_SIZE}")
            mcts.set_params(batch_size=config.MCTS_BATCH_SIZE, num_threads=config.MCTS_THREADS)
            
        seed = int(time.time()) + worker_id * 10000
        mcts.set_random_seed(seed)
        np.random.seed(seed)
    except Exception as e:
        print(f"[Worker {worker_id}] Init Error: {e}")
        return

    mcts_opponent = None
    if opponent_engine_path and os.path.exists(opponent_engine_path):
        try:
            print(f"[Worker {worker_id}] Loading Opponent Engine: {opponent_engine_path}")
            mcts_opponent = MCTSEngine(opponent_engine_path, device=device)
            # Use same params for opponent? Or weaker? Default to same for now.
            if hasattr(config, 'MCTS_THREADS'):
                mcts_opponent.set_params(batch_size=config.MCTS_BATCH_SIZE, num_threads=config.MCTS_THREADS)
            mcts_opponent.set_random_seed(seed + 1)
        except Exception as e:
            print(f"[Worker {worker_id}] Opponent Init Error: {e}")
            mcts_opponent = None

    local_data = []
    
    for i in range(games_to_play):
        game = Connect6Game()
    for i in range(games_to_play):
        game = Connect6Game()
        mcts.reset()
        if mcts_opponent:
            mcts_opponent.reset()
            
        # Determine Game Mode: Self-Play or Asymmetric
        is_asymmetric = False
        if mcts_opponent and np.random.random() < mix_ratio:
            is_asymmetric = True
            # Randomly assign Current Model to Black(1) or White(-1)
            # If current=Black, Opponent=White
            current_is_black = np.random.random() > 0.5
            
        game_moves = []
        game_policies = []
        game_bonuses = [] # Store bonus rewards
        
        while True:
            stones_to_place = 1 if game.move_count == 0 else 2
            
            if stones_to_place > 0:
                for _ in range(stones_to_place):
                    if game.winner != 0: break
                    
                    temp = config.TEMP_FINAL
                    if len(game.moves) < config.OPENING_MOVES:
                        # Asymmetric Temperature
                        if hasattr(config, 'TEMP_OPENING_BLACK') and hasattr(config, 'TEMP_OPENING_WHITE'):
                             temp = config.TEMP_OPENING_BLACK if game.current_player == 1 else config.TEMP_OPENING_WHITE
                        else:
                             temp = config.TEMP_OPENING
                    
                    # 1. Run MCTS
                    # Dynamic Simulations for Balancing
                    current_sims = sims
                    if hasattr(config, 'SIMULATIONS_BLACK') and hasattr(config, 'SIMULATIONS_WHITE'):
                        if game.current_player == 1: # Black
                            current_sims = config.SIMULATIONS_BLACK
                        else: # White
                            current_sims = config.SIMULATIONS_WHITE
                    
                    # Late Game Dynamic Scaling (>100 moves)
                    # Increase simulations by 50% to catch blunders in complex positions
                    if len(game.moves) > 75:
                        current_sims = int(current_sims * 1.5)
                            
                    if len(game.moves) > 75:
                        current_sims = int(current_sims * 1.5)
                    
                    active_mcts = mcts
                    if is_asymmetric:
                        # logical mapping
                        # if current_is_black (1): Black->mcts, White->mcts_opp
                        # if not current_is_black (-1): Black->mcts_opp, White->mcts
                        if current_is_black:
                            if game.current_player == 1: active_mcts = mcts
                            else: active_mcts = mcts_opponent
                        else:
                            if game.current_player == 1: active_mcts = mcts_opponent
                            else: active_mcts = mcts
                    
                    # === Dynamic Simulation Loop ===
                    sims_done = 0
                    check_interval = getattr(config, 'DYNAMIC_CHECK_INTERVAL', 400)
                    fuse_ratio = getattr(config, 'DYNAMIC_FUSE_RATIO', 10.0)
                    
                    # Only apply dynamic to High Sims (> check_interval) to avoid overhead on small sims
                    if current_sims > check_interval:
                         while sims_done < current_sims:
                             # Run a chunk
                             chunk = min(check_interval, current_sims - sims_done)
                             active_mcts.run_simulations(chunk)
                             sims_done += chunk
                             
                             # Check for Early Stopping (Fuse)
                             if sims_done >= check_interval: # Ensure at least one check's worth
                                 policy = active_mcts.get_policy()
                                 # Get Top 2 indices
                                 top_indices = np.argsort(policy)[-2:][::-1]
                                 p1 = policy[top_indices[0]]
                                 p2 = policy[top_indices[1]] if len(top_indices) > 1 else 0.0
                                 
                                 if p2 == 0 or p1 > p2 * fuse_ratio:
                                     # print(f"Fuse Triggered! {sims_done}/{current_sims} Top1={p1:.2f} Top2={p2:.2f}")
                                     break
                    else:
                         # Run all at once if small budget
                         active_mcts.run_simulations(current_sims)

                    move_idx = active_mcts.get_mcts_move(simulations=0, temperature=temp)
                    
                    # 2. Get Policy Target (Always from active_mcts)
                    policy = active_mcts.get_policy()
                    policy_str = encode_policy(policy)
                    game_policies.append(policy_str)
                    
                    # 3. Apply Move & Calculate Bonus
                    r, c = move_idx // 19, move_idx % 19
                    coord = game._to_coord(r, c)
                    
                    # Calculate Dense Reward for this move
                    # Note: analyze_move_quality needs the board state BEFORE the move? 
                    # Or AFTER? The function logic implies checking lines formed by the move.
                    # It usually checks AFTER the piece is placed.
                    # BUT our refiner.check_line logic iterates. 
                    # Let's pass the board state BEFORE move, but tell it where we put the stone.
                    # Wait, refiner.analyze_move_quality assumes board has the stone?
                    # Let's look at refiner logic:
                    # check_line(board, r, c...) checks board[r][c].
                    # So we must place the stone FIRST.
                    
                    game.board[r, c] = game.current_player
                    
                    # Calculate Bonus!
                    # bonus = refiner.analyze_move_quality(game.board, r, c, game.current_player)
                    bonus = 0.0 # Sparse Reward Only
                    game_bonuses.append(f"{bonus:.2f}")

                    game.moves.append(coord)
                    game_moves.append(coord)
                    
                    mcts.update_state(move_idx)
                    if mcts_opponent:
                        mcts_opponent.update_state(move_idx)
                    
                    if game.check_win_at(r, c, game.current_player):
                        game.winner = game.current_player
                        break
            
            if game.winner != 0: break
            if len(game.moves) >= 361: break
            
            game.current_player = -game.current_player
            game.move_count += 1
            
        winner_str = 'draw'
        if game.winner == 1: winner_str = 'black'
        elif game.winner == -1: winner_str = 'white'
        
        # Format Data
        full_policy_str = "|".join(game_policies)
        moves_str = ",".join(game_moves)
        bonuses_str = ",".join(game_bonuses)
        
        with lock:
            # Append directly to file
            # Format: moves, winner, policies, bonuses
            with open(out_file, 'a') as f:
                f.write(f'"{moves_str}",{winner_str},"{full_policy_str}","{bonuses_str}"\n')
                
        print(f"[Worker {worker_id}] Game {i+1}/{games_to_play} Winner: {winner_str}")

def main():
    config.ensure_dirs()
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=str, default=os.path.join(config.RAW_DATA_DIR, 'gen_data.csv'))
    parser.add_argument('--total', type=int, default=config.GAMES_PER_LOOP)
    parser.add_argument('--engine', type=str, default=config.INITIAL_MODEL_PATH, help='Path to TensorRT Engine')
    parser.add_argument('--opponent', type=str, default=None, help='Path to Opponent Engine for Asymmetric Play')
    parser.add_argument('--mix_ratio', type=float, default=0.0, help='Ratio of asymmetric games')
    args = parser.parse_args()
    
    # Init CSV header if new
    if not os.path.exists(args.out):
        with open(args.out, 'w') as f:
            f.write("moves,winner,policies,bonuses\n")
            
    gpu_list = [int(x) for x in config.GPUS.split(',')]
    games_per_worker = args.total // config.NUM_WORKERS
    
    lock = mp.Lock()
    processes = []
    
    print(f"Generating {args.total} games using Engine: {args.engine}")
    
    for i in range(config.NUM_WORKERS):
        gpu_id = gpu_list[i % len(gpu_list)]
    for i in range(config.NUM_WORKERS):
        gpu_id = gpu_list[i % len(gpu_list)]
        p = mp.Process(target=worker_process, args=(gpu_id, args.engine, config.SIMULATIONS, games_per_worker, args.out, i, lock, args.opponent, args.mix_ratio))
        p.start()
        processes.append(p)
        
    try:
        for p in processes:
            p.join()
    except KeyboardInterrupt:
        print("Interrupted. Terminating...")
        for p in processes:
            p.terminate()
            
if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()
