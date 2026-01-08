import sys
import os
import time
import argparse
import shutil
import glob
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import subprocess

# Add paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.connect6_game import Connect6Game
from core.mcts import MCTSEngine
import config

# Cross-platform Python executable
PYTHON = sys.executable

def run_command(cmd, env_vars=None):
    print(f"Running: {cmd}")
    env = os.environ.copy()
    if env_vars:
        env.update(env_vars)
    subprocess.check_call(cmd, shell=True, env=env)

def convert_to_engine(pth_path, engine_path, gpu_id):
    """
    Convert .pth -> .onnx -> .engine
    """
    if os.path.exists(engine_path):
        print(f"Engine found at {engine_path}, skipping conversion.")
        return
    
    print(f"Converting {pth_path} to {engine_path}...")
    
    # 1. Export ONNX
    onnx_path = engine_path.replace('.engine', '.onnx')
    script_export = os.path.join(config.BASE_DIR, 'pipeline/export_onnx.py')
    cmd_export = f"\"{PYTHON}\" {script_export} {pth_path} {onnx_path}"
    run_command(cmd_export, env_vars={'CUDA_VISIBLE_DEVICES': str(gpu_id)})
    
    # 2. Build Engine
    script_build = os.path.join(config.BASE_DIR, 'pipeline/build_engine.py')
    cmd_build = f"\"{PYTHON}\" {script_build} {onnx_path} {engine_path}"
    run_command(cmd_build, env_vars={'CUDA_VISIBLE_DEVICES': str(gpu_id)})
    
    # Cleanup ONNX
    if os.path.exists(onnx_path):
        os.remove(onnx_path)

def plot_results(results, generation, save_path):
    """
    Generate a summary bar chart for the evaluation results.
    """
    opponents = list(results.keys())
    if not opponents:
        return

    # Metrics to plot
    win_rates = [results[o]['win_rate'] for o in opponents]
    loss_rates = [results[o]['loss_rate'] for o in opponents]
    draw_rates = [results[o]['draw_rate'] for o in opponents]
    avg_steps = [results[o]['avg_game_steps'] for o in opponents]
    
    x = np.arange(len(opponents))
    width = 0.25
    
    # Use a style
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Win/Loss/Draw Rates
    rects1 = ax1.bar(x - width, win_rates, width, label='Win', color='#4CAF50', alpha=0.8)
    rects2 = ax1.bar(x, draw_rates, width, label='Draw', color='#FFC107', alpha=0.8)
    rects3 = ax1.bar(x + width, loss_rates, width, label='Loss', color='#F44336', alpha=0.8)
    
    ax1.set_ylabel('Rate')
    ax1.set_title(f'Generation {generation} Win Rates')
    ax1.set_xticks(x)
    ax1.set_xticklabels(opponents)
    ax1.set_ylim(0, 1.1)
    ax1.legend()
    
    # Add labels
    def autolabel(rects, ax):
        for rect in rects:
            height = rect.get_height()
            if height > 0:
                ax.annotate(f'{height:.1%}',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    autolabel(rects1, ax1)
    autolabel(rects2, ax1)
    autolabel(rects3, ax1)
    
    # Plot 2: Average Steps
    rects4 = ax2.bar(x, avg_steps, width*1.5, color='#2196F3', alpha=0.8)
    ax2.set_ylabel('Average Steps')
    ax2.set_title('Average Game Length')
    ax2.set_xticks(x)
    ax2.set_xticklabels(opponents)
    
    def autolabel_steps(rects, ax):
        for rect in rects:
            height = rect.get_height()
            if height > 0:
                ax.annotate(f'{int(height)}',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=9)
    autolabel_steps(rects4, ax2)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    ax2.set_xticks(x)
    ax2.set_xticklabels(opponents)
    
    def autolabel_steps(rects, ax):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)
    autolabel_steps(rects4, ax2)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.close()

def plot_static_results(results, generation, save_path):
    """
    Generate chart ONLY for static benchmarks (gen_350, gen_450, etc.)
    Excludes relative history (gen_X where X is recent).
    """
    # Filter keys: Keep only hardcoded benchmarks
    static_keys = [k for k in results.keys() if k in ['gen_350', 'gen_450']]
    if not static_keys:
        return

    # Extract subset
    subset = {k: results[k] for k in static_keys}
    plot_results(subset, generation, save_path)

def play_match(engine1_path, engine2_path, games=30, simulations=1200, gpu_id=0):
    """
    Play a match between two engines.
    engine1: Current Model
    engine2: Opponent
    Returns: dict with detailed stats
    """
    save_data = False # Hardcoded for now, or passed via args?
    # Let's make it an argument if we can, but simpler to just hardcode list accumulation
    # and return it.
    collected_data = [] # List of (moves_str, winner, policy_str, bonus_str)
    
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    device = torch.device('cuda:0')
    
    # 2. Add encode_policy helper (copied from generate.py)
    def encode_policy(policy):
        # policy is [19*19+1] or similar
        # We need to sparsify it: "idx:prob;idx:prob"
        items = []
        for idx, prob in enumerate(policy):
            if prob > 0.001:
                items.append(f"{idx}:{prob:.4f}")
        return ";".join(items)

    try:
        # Instantiate two MCTS wrappers
        # They share the same C++ lib state, but have different TensorRT contexts
        mcts1 = MCTSEngine(engine1_path, device=device)
        mcts2 = MCTSEngine(engine2_path, device=device)
        
        # Set params
        mcts1.set_params(batch_size=config.MCTS_BATCH_SIZE, num_threads=config.MCTS_THREADS)
        mcts2.set_params(batch_size=config.MCTS_BATCH_SIZE, num_threads=config.MCTS_THREADS)
        
    except Exception as e:
        print(f"Error initializing engines: {e}")
        return {}, []

    stats = {
        'wins': 0, 'losses': 0, 'draws': 0,
        'win_steps': [], 'loss_steps': [], 'draw_steps': [],
        'black_wins': 0, 'black_games': 0,
        'white_wins': 0, 'white_games': 0
    }
    
    # Alternate colors
    # Game i: P1=engine1 (Black), P2=engine2 (White)
    # Game i+1: P1=engine2 (Black), P2=engine1 (White)
    
    for i in range(games):
        game = Connect6Game()
        
        # Determine who is Black (Player 1)
        if i % 2 == 0:
            black_player = mcts1
            white_player = mcts2
            p1_is_engine1 = True
            stats['black_games'] += 1
        else:
            black_player = mcts2
            white_player = mcts1
            p1_is_engine1 = False
            stats['white_games'] += 1
            
        print(f"Game {i+1}/{games} | Black: {'Current' if p1_is_engine1 else 'Opponent'}")
        
        game_policies = [] # List of policy strings for this game
        
        while True:
            # Determine current player object
            if game.current_player == 1: # Black
                current_mcts = black_player
            else: # White
                current_mcts = white_player
            
            # 1. Set Callback
            # This redirects the C++ engine to use the correct neural network
            from core.mcts import mcts_lib
            mcts_lib.set_eval_callback(current_mcts.c_callback)
            
            # 2. Reset and Replay
            # We must clear the tree because it contains nodes from the other player's network
            mcts_lib.init_game()
            for m_str in game.moves:
                coord = game._parse_coord(m_str)
                if coord:
                    r, c = coord
                    idx = r * 19 + c
                    mcts_lib.play_move(idx)
                
            # 3. Search
            move = current_mcts.get_mcts_move(simulations=simulations, temperature=0.0) # Deterministic for eval
            
            # Capture Policy
            policy_dist = current_mcts.get_policy()
            policy_str = encode_policy(policy_dist)
            game_policies.append(policy_str)
            
            # 4. Apply Move
            game.play(move)
            
            if game.winner != 0:
                steps = len(game.moves)
                
                if game.winner == 2: # Draw
                    stats['draws'] += 1
                    stats['draw_steps'].append(steps)
                    print(f"Game {i+1} Result: Draw ({steps} moves)")
                elif game.winner == 1: # Black Wins
                    if p1_is_engine1:
                        stats['wins'] += 1
                        stats['win_steps'].append(steps)
                        stats['black_wins'] += 1
                        print(f"Game {i+1} Result: Current Model (Black) Wins ({steps} moves)")
                    else:
                        stats['losses'] += 1
                        stats['loss_steps'].append(steps)
                        print(f"Game {i+1} Result: Opponent (Black) Wins ({steps} moves)")
                else: # White Wins (-1)
                    if not p1_is_engine1: # Engine1 is White
                        stats['wins'] += 1
                        stats['win_steps'].append(steps)
                        stats['white_wins'] += 1
                        print(f"Game {i+1} Result: Current Model (White) Wins ({steps} moves)")
                    else:
                        stats['losses'] += 1
                        stats['loss_steps'].append(steps)
                        print(f"Game {i+1} Result: Opponent (White) Wins ({steps} moves)")
                
                # --- Post-Game Data Collection ---
                # --- Post-Game Data Collection ---
                # Save standard format: "moves,winner,policies,bonuses"
                # Matches generate.py exactly
                if len(game.moves) == len(game_policies):
                    winner_str = 'draw'
                    if game.winner == 1: winner_str = 'black'
                    elif game.winner == -1: winner_str = 'white'
                    
                    moves_str = ",".join(game.moves)
                    full_policy_str = "|".join(game_policies) # Policies separated by pipe
                    
                    # Quote fields to handle commas/separators safely
                    line = f'"{moves_str}",{winner_str},"{full_policy_str}","0"'
                    collected_data.append(line)
                else:
                    print(f"Warning: Moves/Policies mismatch {len(game.moves)} vs {len(game_policies)}")

                break
                
    return stats, collected_data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--current_engine', type=str, required=True)
    parser.add_argument('--generation', type=int, required=True)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--save_data', action='store_true', help='Save evaluation games for training')
    parser.add_argument('--games', type=int, default=30, help='Number of games to play per opponent')
    args = parser.parse_args()
    
    # Opponents - 使用本地 checkpoints 目录，避免硬编码路径
    opponents = []
    
    # 添加静态基准模型（从 config 读取）
    for bench in config.STATIC_BENCHMARKS:
        bench_path = os.path.join(config.CHECKPOINT_DIR, bench)
        if os.path.exists(bench_path):
            name = bench.replace('model_', '').replace('.pth', '')
            opponents.append((bench_path, name))
    
    # 添加相对代差模型（从 config 读取）
    for offset in config.EVAL_GENERATION_OFFSETS:
        past_gen = args.generation - offset
        if past_gen > 0:
            past_gen_path = os.path.join(config.CHECKPOINT_DIR, f'model_gen_{past_gen}.pth')
            if os.path.exists(past_gen_path):
                opponents.append((past_gen_path, f"gen_{past_gen}"))
    
    # 如果没有任何对手模型，打印提示并返回
    if not opponents:
        print("=" * 50)
        print("提示: 当前没有对手模型可用于评估")
        print("      这在训练初期是正常的")
        print("      等到有更多 generation 模型后会自动开始评估")
        print("=" * 50)
        # 写入空结果以便 run_loop 不会出错
        json_path = os.path.join(config.LOG_DIR, 'eval_results.json')
        with open(json_path, 'w') as f:
            json.dump({}, f)
        return
        
    # Initialize Results Container
    results_summary = {}
    
    for pth_path, name in opponents:
        if not os.path.exists(pth_path):
            print(f"Opponent {name} not found at {pth_path}. Skipping.")
            continue
            
        # Convert to Engine
        engine_path = pth_path.replace('.pth', '.engine')
        convert_to_engine(pth_path, engine_path, args.gpu)
        
        print(f"\n>>> Evaluating against {name}...")
        stats, data_lines = play_match(args.current_engine, engine_path, games=args.games, simulations=1200, gpu_id=int(args.gpu))
        
        if args.save_data and data_lines:
            save_csv = os.path.join(config.RAW_DATA_DIR, f'eval_data_{args.generation}.csv')
            mode = 'a' if os.path.exists(save_csv) else 'w'
            with open(save_csv, mode) as f:
                if mode == 'w':
                    f.write("moves,winner,policies,bonuses\n") # Header standard format
                for line in data_lines:
                    # Line is already formatted as CSV string
                    f.write(line + "\n")
            print(f"Saved {len(data_lines)} positions to {save_csv}")
        
        if not stats:
            print("Match failed.")
            continue

        wins = stats['wins']
        losses = stats['losses']
        draws = stats['draws']
        total = wins + losses + draws
        
        if total == 0:
            continue
            
        results_summary[name] = {
            'wins': wins,
            'losses': losses,
            'draws': draws,
            'win_rate': wins / total,
            'loss_rate': losses / total,
            'draw_rate': draws / total,
            'avg_win_steps': np.mean(stats['win_steps']) if stats['win_steps'] else 0,
            'avg_loss_steps': np.mean(stats['loss_steps']) if stats['loss_steps'] else 0,
            'avg_game_steps': np.mean(stats['win_steps'] + stats['loss_steps'] + stats['draw_steps']) if (stats['win_steps'] + stats['loss_steps'] + stats['draw_steps']) else 0,
            'black_win_rate': stats['black_wins'] / stats['black_games'] if stats['black_games'] > 0 else 0,
            'white_win_rate': stats['white_wins'] / stats['white_games'] if stats['white_games'] > 0 else 0
        }
        
        print(f"Vs {name}: Win Rate {results_summary[name]['win_rate']:.2%}")

    # Save Results
    json_path = os.path.join(config.LOG_DIR, 'eval_results.json')
    with open(json_path, 'w') as f:
        json.dump(results_summary, f, indent=4)
        
    # Generate Chart
    chart_path = os.path.join(config.LOG_DIR, 'eval_chart.png')
    # Generate Charts
    # 1. Full Chart (Maybe kept locally for debug, but not uploaded if user hates it)
    chart_path = os.path.join(config.LOG_DIR, 'eval_chart.png')
    plot_results(results_summary, args.generation, chart_path)
    
    # 2. Static Only Chart (User Request)
    static_chart_path = os.path.join(config.LOG_DIR, 'eval_static_chart.png')
    plot_static_results(results_summary, args.generation, static_chart_path)
    
    print(f"Evaluation complete. Results saved to {json_path} and {chart_path}")

if __name__ == "__main__":
    main()
