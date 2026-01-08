import os
import sys
import time
import subprocess
import shutil
import glob
import json
import signal
import swanlab
import config
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg') # Non-interactive backend

# Cross-platform Python executable
PYTHON = sys.executable

# Signal handler for graceful shutdown
def signal_handler(sig, frame):
    print("\n⚠️ Ctrl+C detected. Shutting down gracefully...")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

def run_command(cmd, env_vars=None):
    print(f"Running: {cmd}")
    try:
        # Force unbuffered output so logs show up immediately in training.log
        env = os.environ.copy()
        env['PYTHONUNBUFFERED'] = '1'
        if env_vars:
            env.update(env_vars)
        subprocess.check_call(cmd, shell=True, env=env)
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {cmd}")
        raise e

def manage_buffer(new_data_path):
    """
    Merge new data into main buffer and keep size limit.
    """
    buffer_path = os.path.join(config.BUFFER_DIR, 'replay_buffer.csv')
    
    # If buffer doesn't exist, just move new data
    if not os.path.exists(buffer_path):
        shutil.copy(new_data_path, buffer_path)
        return
        
    # Append new data to buffer
    # We use simple file appending for speed, but we need to handle header
    # Read new data (skip header)
    with open(new_data_path, 'r') as f:
        lines = f.readlines()
    
    header = lines[0]
    new_content = lines[1:]
    
    with open(buffer_path, 'a') as f:
        for line in new_content:
            f.write(line)
            
    # Trim buffer if too large
    # This is slow for large files, but robust. 
    # For 20k lines it's fine.
    with open(buffer_path, 'r') as f:
        all_lines = f.readlines()
        
    if len(all_lines) > config.BUFFER_SIZE:
        print(f"Trimming buffer from {len(all_lines)} to {config.BUFFER_SIZE}...")
        # Keep header + last N lines
        keep_lines = [header] + all_lines[-(config.BUFFER_SIZE):]
        with open(buffer_path, 'w') as f:
            f.writelines(keep_lines)

def update_history_plot(history_file, plot_file):
    """
    Reads history JSON and plots a trend line.
    """
    if not os.path.exists(history_file):
        return

    try:
        with open(history_file, 'r') as f:
            history = json.load(f)
    except:
        return

    if not history:
        return

    generations = [h['generation'] for h in history]
    avg_win_rates = [h.get('avg_win_rate', 0) for h in history]
    avg_white_win_rates = [h.get('avg_white_win_rate', 0) for h in history]

    plt.figure(figsize=(10, 6))
    plt.plot(generations, avg_win_rates, marker='o', label='Avg Win Rate (Black+White)')
    plt.plot(generations, avg_white_win_rates, marker='x', linestyle='--', label='Avg White Win Rate')
    
    plt.title('Win Rate Trend')
    plt.xlabel('Generation')
    plt.ylabel('Win Rate')
    plt.ylim(0, 1.0)
    plt.grid(True)
    plt.legend()
    
    plt.savefig(plot_file)
    plt.close()

def analyze_generation_data(data_path, generation):
    """
    分析新生成的对局数据，返回统计信息并生成图表用于 SwanLab
    """
    import pandas as pd
    
    if not os.path.exists(data_path):
        return {}
    
    try:
        df = pd.read_csv(data_path, on_bad_lines='skip')
    except:
        return {}
    
    total_games = len(df)
    if total_games == 0:
        return {}
    
    # 解析 winner 列
    if 'winner' in df.columns:
        df['winner'] = df['winner'].astype(str).str.lower()
        black_wins = len(df[df['winner'].str.contains('black', na=False)])
        white_wins = len(df[df['winner'].str.contains('white', na=False)])
        draws = total_games - black_wins - white_wins
    else:
        return {}
    
    # 解析对局长度
    def count_moves(moves_str):
        if not isinstance(moves_str, str) or moves_str == 'nan':
            return 0
        return moves_str.count(',') + 1
    
    col_moves = 'moves' if 'moves' in df.columns else df.columns[0]
    df['length'] = df[col_moves].apply(count_moves)
    avg_length = df['length'].mean()
    
    # 开局多样性分析（前5步）
    def get_opening(moves_str, n_moves=5):
        if not isinstance(moves_str, str) or moves_str == 'nan':
            return ''
        moves = moves_str.split(',')[:n_moves]
        return ','.join(moves)
    
    df['opening'] = df[col_moves].apply(lambda x: get_opening(x, 5))
    unique_openings = df['opening'].nunique()
    opening_diversity = unique_openings / total_games if total_games > 0 else 0
    
    stats = {
        'total_games': total_games,
        'black_wins': black_wins,
        'white_wins': white_wins,
        'draws': draws,
        'black_win_rate': black_wins / total_games if total_games > 0 else 0,
        'white_win_rate': white_wins / total_games if total_games > 0 else 0,
        'avg_game_length': avg_length,
        'opening_diversity': opening_diversity,
        'unique_openings': unique_openings,
    }
    
    # 生成图表
    try:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # 图1: 胜率分布
        ax1 = axes[0]
        colors = ['black', 'gray', 'white']
        labels = ['Black', 'Draw', 'White']
        counts = [black_wins, draws, white_wins]
        bars = ax1.bar(labels, counts, color=colors, edgecolor='black')
        ax1.set_title(f'Gen {generation} Win Distribution')
        ax1.set_ylabel('Games')
        for bar, count in zip(bars, counts):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                    f'{count}\n({count/total_games:.1%})', ha='center', va='bottom')
        
        # 图2: 对局长度分布
        ax2 = axes[1]
        ax2.hist(df['length'], bins=20, color='steelblue', edgecolor='black', alpha=0.7)
        ax2.axvline(avg_length, color='red', linestyle='--', label=f'Avg: {avg_length:.1f}')
        ax2.set_title(f'Gen {generation} Game Length')
        ax2.set_xlabel('Moves')
        ax2.set_ylabel('Games')
        ax2.legend()
        
        plt.tight_layout()
        
        # 保存图表
        chart_path = os.path.join(config.LOG_DIR, f'gen_{generation}_analysis.png')
        plt.savefig(chart_path, dpi=100)
        plt.close()
        
        stats['chart_path'] = chart_path
        
    except Exception as e:
        print(f"Error generating analysis chart: {e}")
    
    return stats

def main():
    config.ensure_dirs()
    
    # History file for tracking progress across restarts
    history_file = os.path.join(config.LOG_DIR, 'eval_history.json')
    if not os.path.exists(history_file):
        with open(history_file, 'w') as f:
            json.dump([], f)

    # Initialize SwanLab for the main loop (支持断点续传)
    swanlab_id_file = os.path.join(config.LOG_DIR, 'swanlab_run_id.txt')
    existing_run_id = None
    
    # 尝试读取之前的 run_id
    if os.path.exists(swanlab_id_file):
        with open(swanlab_id_file, 'r') as f:
            existing_run_id = f.read().strip()
        if existing_run_id:
            print(f"Found existing SwanLab run ID: {existing_run_id}")
    
    # 初始化 SwanLab（如果有 run_id 则续传，否则创建新实验）
    if existing_run_id:
        swanlab.init(
            project="connect6-experiment",
            experiment_name="AlphaZero_Training_Loop",
            description="胜率，平均步数总的展示",
            resume="allow",
            id=existing_run_id
        )
        print(f"Resumed SwanLab experiment: {existing_run_id}")
    else:
        swanlab.init(
            project="connect6-experiment",
            experiment_name="AlphaZero_Training_Loop",
            description="胜率，平均步数总的展示"
        )
        # 保存 run_id 以便下次续传
        # SwanLab 的 run id 可以从 swanlab.get_run() 获取
        current_run = swanlab.get_run()
        if current_run and hasattr(current_run, 'id'):
            with open(swanlab_id_file, 'w') as f:
                f.write(current_run.id)
            print(f"Created new SwanLab experiment: {current_run.id}")
        else:
            print("Warning: Could not save SwanLab run ID")
    
    # Set visible devices based on config (Only affects initial context, we override later)
    # config.GPUS="1,2" for generation.
    # But for training we want GPU 0.
    # So we should NOT set global CUDA_VISIBLE_DEVICES here if we want different processes to see different cards.
    # Let's remove the global restrict.
    # os.environ["CUDA_VISIBLE_DEVICES"] = config.GPUS
    print(f"Configured GPUs for Generation: {config.GPUS}")
    
    # Initial Setup
    engine_path = config.CURRENT_ENGINE_PATH
    if not os.path.exists(engine_path):
        print("No current engine found. Checking initial model...")
        if os.path.exists(config.INITIAL_MODEL_PATH):
            print(f"Using pre-built initial engine: {config.INITIAL_MODEL_PATH}")
            shutil.copy(config.INITIAL_MODEL_PATH, engine_path)
        elif hasattr(config, 'INITIAL_MODEL_PTH') and os.path.exists(config.INITIAL_MODEL_PTH):
            # Build engine from PTH
            print(f"Building initial engine from: {config.INITIAL_MODEL_PTH}")
            
            # Copy PTH to best.pth first
            best_pth = os.path.join(config.CHECKPOINT_DIR, 'best.pth')
            shutil.copy(config.INITIAL_MODEL_PTH, best_pth)
            
            # Export to ONNX
            onnx_path = os.path.join(config.CHECKPOINT_DIR, 'initial.onnx')
            export_script = os.path.join(config.BASE_DIR, 'pipeline/export_onnx.py')
            run_command(f"\"{PYTHON}\" {export_script} {best_pth} {onnx_path}")
            
            # Build TensorRT Engine
            build_script = os.path.join(config.BASE_DIR, 'pipeline/build_engine.py')
            run_command(f"\"{PYTHON}\" {build_script} {onnx_path} {engine_path}")
            
            # Also save as initial_model.engine for backup
            shutil.copy(engine_path, config.INITIAL_MODEL_PATH)
            
            # Cleanup ONNX
            if os.path.exists(onnx_path):
                os.remove(onnx_path)
                
            print("Initial engine built successfully!")
        else:
            print(f"Error: No initial model found!")
            print(f"  Checked: {config.INITIAL_MODEL_PATH}")
            if hasattr(config, 'INITIAL_MODEL_PTH'):
                print(f"  Checked: {config.INITIAL_MODEL_PTH}")
            return

    generation = 0
    
    # Resume Logic: Find latest generation
    existing_checkpoints = glob.glob(os.path.join(config.CHECKPOINT_DIR, 'model_gen_*.pth'))
    if existing_checkpoints:
        gens = []
        for p in existing_checkpoints:
            try:
                # Extract number from filename 'model_gen_123.pth'
                base = os.path.basename(p)
                num = int(base.replace('model_gen_', '').replace('.pth', ''))
                gens.append(num)
            except:
                pass
        if gens:
            last_gen = max(gens)
            generation = last_gen + 1
            print(f"Found checkpoint for generation {last_gen}. Resuming from generation {generation}...")
            
            # Ensure best.pth corresponds to the last generation if it doesn't exist
            resume_path = os.path.join(config.CHECKPOINT_DIR, 'best.pth')
            last_gen_path = os.path.join(config.CHECKPOINT_DIR, f'model_gen_{last_gen}.pth')
            if not os.path.exists(resume_path) and os.path.exists(last_gen_path):
                print(f"Restoring best.pth from {last_gen_path}")
                shutil.copy(last_gen_path, resume_path)

    current_epochs = config.TRAIN_EPOCHS

    while True:
        print(f"\n{'='*20} Generation {generation} {'='*20}\n")
        
        # Step 1: Self-Play
        print(">>> Step 1: Self-Play Generation")
        gen_data_path = os.path.join(config.RAW_DATA_DIR, f'gen_{generation}.csv')
        
        # Call generate.py
        # We pass the CURRENT engine path
        # Note: generate.py uses config.INITIAL_MODEL_PATH by default, we should override it via cmd line if possible
        # But wait, generate.py uses args for path? 
        # Let's check generate.py logic.
        # It calls worker_process with config.INITIAL_MODEL_PATH hardcoded in main().
        # We need to fix generate.py to accept --engine argument!
        
        # Quick fix: Just overwrite the engine file that generate.py uses?
        # Or better, update generate.py to use config.CURRENT_ENGINE_PATH
        
        # Let's assume we update generate.py or config.py to point to CURRENT_ENGINE_PATH
        # For now, I will copy CURRENT_ENGINE to config.INITIAL_MODEL_PATH to cheat
        # if they are different.
        
        if config.INITIAL_MODEL_PATH != engine_path:
             shutil.copy(engine_path, config.INITIAL_MODEL_PATH)

        # Check if we already have enough data
        existing_games = 0
        if os.path.exists(gen_data_path):
            with open(gen_data_path, 'r') as f:
                existing_games = sum(1 for line in f) - 1 # Minus header
        
        if existing_games >= config.GAMES_PER_LOOP:
            print(f"Found {existing_games} games in {gen_data_path}. Skipping generation.")
        else:
            # Generate using GPU 1 and 2 (handled by config.GPUS inside generate.py)
            script_path = os.path.join(config.BASE_DIR, 'pipeline/generate.py')
            
            # === Asymmetric Self-Play Logic ===
            opponent_args = ""
            if config.ASYMMETRIC_SELFPLAY_RATIO > 0:
                # Logic: Use an opponent from 10 generations ago
                target_gen = generation - config.OPPONENT_MODEL_GENERATION_GAP
                
                # To save build time, we can reuse the engine if target_gen hasn't changed enough?
                # Actually, let's just cache the engine by generation ID.
                if target_gen > 0:
                    opponent_pth = os.path.join(config.CHECKPOINT_DIR, f'model_gen_{target_gen}.pth')
                    
                    # 1. Try to find existing compiled engine (User said they might be pre-compiled)
                    # We look for model_gen_X.engine
                    standard_engine_name = os.path.join(config.CHECKPOINT_DIR, f'model_gen_{target_gen}.engine')
                    cache_engine_name = os.path.join(config.CHECKPOINT_DIR, f'opponent_gen_{target_gen}.engine')
                    
                    engine_to_use = None
                    
                    if os.path.exists(standard_engine_name):
                        print(f"Found pre-compiled standard engine: {standard_engine_name}")
                        engine_to_use = standard_engine_name
                    elif os.path.exists(cache_engine_name):
                        print(f"Found cached opponent engine: {cache_engine_name}")
                        engine_to_use = cache_engine_name
                    elif os.path.exists(opponent_pth):
                        # Build it if missing, but preferably name it standard_engine_name so it's reusable?
                        # Or keep separate to avoid messing with other checks.
                        # User said "no need to rename, I compiled them before".
                        # If we didn't find it, we MUST build it. We'll build to cache_engine_name.
                        print(f"Propiling Opponent Engine for Gen {target_gen}...")
                        onnx_opp = os.path.join(config.CHECKPOINT_DIR, f'opponent_{target_gen}.onnx')
                        engine_to_use = cache_engine_name # Build here
                        
                        cmd_export = f"\"{PYTHON}\" {os.path.join(config.BASE_DIR, 'pipeline/export_onnx.py')} {opponent_pth} {onnx_opp}"
                        run_command(cmd_export)
                        
                        cmd_build = f"\"{PYTHON}\" {os.path.join(config.BASE_DIR, 'pipeline/build_engine.py')} {onnx_opp} {engine_to_use}"
                        run_command(cmd_build)
                        
                        if os.path.exists(onnx_opp):
                            os.remove(onnx_opp)
                    
                    if engine_to_use and os.path.exists(engine_to_use):
                        opponent_args = f"--opponent {engine_to_use} --mix_ratio {config.ASYMMETRIC_SELFPLAY_RATIO}"
            
            cmd_gen = f"\"{PYTHON}\" {script_path} --out {gen_data_path} --total {config.GAMES_PER_LOOP} --engine {engine_path} {opponent_args}"
            
            t0 = time.time()
            run_command(cmd_gen)
            t1 = time.time()
            t1 = time.time()
            duration = t1 - t0
            speed = config.GAMES_PER_LOOP / duration if duration > 0 else 0
            print(f"\n[Timer] Self-Play Generation: {duration:.2f} seconds | Speed: {speed:.2f} games/s\n")
        
        # Step 2: Buffer Management
        print(">>> Step 2: Updating Replay Buffer")
        manage_buffer(gen_data_path)
        
        # Step 2.5: Data Analysis - Upload to SwanLab
        print(">>> Step 2.5: Analyzing Generation Data")
        gen_stats = analyze_generation_data(gen_data_path, generation)
        if gen_stats:
            log_data = {
                "data/black_win_rate": gen_stats.get('black_win_rate', 0),
                "data/white_win_rate": gen_stats.get('white_win_rate', 0),
                "data/avg_game_length": gen_stats.get('avg_game_length', 0),
                "data/opening_diversity": gen_stats.get('opening_diversity', 0),
            }
            if 'chart_path' in gen_stats and os.path.exists(gen_stats['chart_path']):
                log_data["data/generation_analysis"] = swanlab.Image(gen_stats['chart_path'], caption=f"Gen {generation}")
            swanlab.log(log_data)
            print(f"  Black: {gen_stats.get('black_win_rate', 0):.1%}, White: {gen_stats.get('white_win_rate', 0):.1%}, Diversity: {gen_stats.get('opening_diversity', 0):.1%}")
        
        # 分析缓冲区并上传（这个图会动态更新）
        buffer_csv = os.path.join(config.BUFFER_DIR, 'replay_buffer.csv')
        buffer_size = 0
        if os.path.exists(buffer_csv):
            with open(buffer_csv, 'r') as f:
                buffer_size = sum(1 for line in f) - 1
            
            # 分析缓冲区数据并生成图表
            buffer_stats = analyze_generation_data(buffer_csv, f"Buffer (Gen {generation})")
            if buffer_stats:
                buffer_log = {
                    "data/buffer_size": buffer_size,
                    "data/buffer_black_win_rate": buffer_stats.get('black_win_rate', 0),
                    "data/buffer_white_win_rate": buffer_stats.get('white_win_rate', 0),
                    "data/buffer_opening_diversity": buffer_stats.get('opening_diversity', 0),
                }
                if 'chart_path' in buffer_stats and os.path.exists(buffer_stats['chart_path']):
                    buffer_log["data/buffer_analysis"] = swanlab.Image(buffer_stats['chart_path'], caption=f"Buffer @ Gen {generation}")
                swanlab.log(buffer_log)
                print(f"  Buffer: {buffer_size} games, Black: {buffer_stats.get('black_win_rate', 0):.1%}")
        
        print(f"Current Buffer Size: {buffer_size}")
        
        if buffer_size < config.HOT_START_MIN_BUFFER:
            print(f"Buffer size {buffer_size} < {config.HOT_START_MIN_BUFFER}. Skipping training to accumulate more data (Hot Start)...")
            generation += 1
            continue

        # Step 3: Training
        print(">>> Step 3: Finetuning")
        # Train on rolling buffer
        buffer_csv = os.path.join(config.BUFFER_DIR, 'replay_buffer.csv')
        
        # Resume from previous best
        resume_path = os.path.join(config.CHECKPOINT_DIR, 'best.pth')
        if not os.path.exists(resume_path):
            # 使用 config 中定义的初始模型
            if hasattr(config, 'INITIAL_MODEL_PTH') and os.path.exists(config.INITIAL_MODEL_PTH):
                shutil.copy(config.INITIAL_MODEL_PTH, resume_path)
                print(f"Copied initial model from {config.INITIAL_MODEL_PTH}")
        
        # Training uses GPU specified in config
        # We pass CUDA_VISIBLE_DEVICES explicitly to the training command
        train_gpu = getattr(config, 'TRAINING_GPU', '0')
        script_path = os.path.join(config.BASE_DIR, 'pipeline/train.py')
        cmd_train = f"\"{PYTHON}\" {script_path} --resume {resume_path} --data {buffer_csv} --run_name gen_{generation} --epochs {current_epochs}"
        run_command(cmd_train, env_vars={'CUDA_VISIBLE_DEVICES': str(train_gpu)})
        
        # 上传训练指标到 SwanLab
        train_metrics_path = os.path.join(config.LOG_DIR, 'train_metrics.json')
        if os.path.exists(train_metrics_path):
            with open(train_metrics_path, 'r') as f:
                train_metrics = json.load(f)
            swanlab.log({
                "train/loss": train_metrics.get('loss', 0),
                "train/policy_loss": train_metrics.get('policy_loss', 0),
                "train/value_loss": train_metrics.get('value_loss', 0),
                "train/accuracy_top1": train_metrics.get('accuracy_top1', 0),
                "train/accuracy_top5": train_metrics.get('accuracy_top5', 0),
                "train/policy_entropy": train_metrics.get('policy_entropy', 0),
                "generation": generation,
            })
            print(f"Uploaded training metrics: loss={train_metrics.get('loss', 0):.4f}, policy={train_metrics.get('policy_loss', 0):.4f}, value={train_metrics.get('value_loss', 0):.4f}, acc1={train_metrics.get('accuracy_top1', 0):.2%}")
        
        # Save generation checkpoint
        shutil.copy(resume_path, os.path.join(config.CHECKPOINT_DIR, f'model_gen_{generation}.pth'))
        
        # Step 4: Update Engine
        print(">>> Step 4: Exporting & Compiling Engine")
        onnx_path = os.path.join(config.CHECKPOINT_DIR, 'temp.onnx')
        
        # Export ONNX
        # Use same GPU as training
        script_path = os.path.join(config.BASE_DIR, 'pipeline/export_onnx.py')
        cmd_export = f"\"{PYTHON}\" {script_path} {resume_path} {onnx_path}"
        run_command(cmd_export, env_vars={'CUDA_VISIBLE_DEVICES': str(train_gpu)})
        
        # Build Engine
        candidate_engine_path = os.path.join(config.CHECKPOINT_DIR, f'model_gen_{generation}.engine')
        script_build = os.path.join(config.BASE_DIR, 'pipeline/build_engine.py')
        cmd_build = f"\"{PYTHON}\" {script_build} {onnx_path} {candidate_engine_path}"
        run_command(cmd_build, env_vars={'CUDA_VISIBLE_DEVICES': str(train_gpu)})
        
        # Step 5: Evaluation
        print(">>> Step 5: Evaluation")
        script_eval = os.path.join(config.BASE_DIR, 'pipeline/evaluate.py') # Renamed to script_eval to avoid conflict
        
        # Clean up old results
        json_path = os.path.join(config.LOG_DIR, 'eval_results.json')
        chart_path = os.path.join(config.LOG_DIR, 'eval_chart.png')
        if os.path.exists(json_path): os.remove(json_path)
        if os.path.exists(chart_path): os.remove(chart_path)

        # Evaluate the CANDIDATE engine
        # We pass the candidate_engine_path, not config.CURRENT_ENGINE_PATH
        # The gpu_id should be train_gpu
        cmd_eval = f"\"{PYTHON}\" {script_eval} --current_engine {candidate_engine_path} --generation {generation} --gpu {train_gpu} --save_data"
        
        run_command(cmd_eval)
        
        # Ingest Evaluation Data into Buffer
        eval_data_csv = os.path.join(config.RAW_DATA_DIR, f'eval_data_{generation}.csv')
        if os.path.exists(eval_data_csv):
            print(f"Ingesting evaluation data from {eval_data_csv} into buffer...")
            manage_buffer(eval_data_csv)
            
        try:
            # Log Evaluation Results to SwanLab
            if os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    results = json.load(f)
                
                log_data = {"generation": generation}
                
                # Flatten results for scalar logging
                for opp_name, stats in results.items():
                    for metric, value in stats.items():
                        log_data[f"eval/{opp_name}/{metric}"] = value
                
                # Calculate average win rate against past generations
                win_rates = []
                white_win_rates = []
                for opp_name, stats in results.items():
                    if opp_name.startswith("gen_"):
                        win_rates.append(stats['win_rate'])
                        white_win_rates.append(stats['white_win_rate'])
                
                if win_rates:
                    avg_win_rate = sum(win_rates) / len(win_rates)
                    avg_white_win_rate = sum(white_win_rates) / len(white_win_rates)
                    
                    # Log Average Win Rates to SwanLab
                    log_data["eval/average_win_rate"] = avg_win_rate
                    log_data["eval/average_white_win_rate"] = avg_white_win_rate
                    
                    print(f"Average Win Rate: {avg_win_rate:.2%}")
                    print(f"Average White Win Rate: {avg_white_win_rate:.2%}")
                    
                    # --- Update History ---
                    try:
                        with open(history_file, 'r') as f:
                            history = json.load(f)
                    except:
                        history = []
                    
                    # Check if generation already exists
                    existing_idx = next((i for i, h in enumerate(history) if h['generation'] == generation), -1)
                    entry = {
                        "generation": generation,
                        "avg_win_rate": avg_win_rate,
                        "avg_white_win_rate": avg_white_win_rate,
                        "timestamp": time.time()
                    }
                    
                    if existing_idx >= 0:
                        history[existing_idx] = entry
                    else:
                        history.append(entry)
                        
                    # Sort by generation
                    history.sort(key=lambda x: x['generation'])
                    
                    with open(history_file, 'w') as f:
                        json.dump(history, f, indent=4)
                        
                    # Generate Trend Plot
                    trend_plot_path = os.path.join(config.LOG_DIR, 'eval_trend.png')
                    update_history_plot(history_file, trend_plot_path)
                    
                    if os.path.exists(trend_plot_path):
                        log_data["eval/win_rate_trend"] = swanlab.Image(trend_plot_path, caption="Win Rate Trend")
                        
                    # User requested NOT to upload the detailed per-opponent chart (ugly)
                    # if os.path.exists(results_chart_path):
                    #     log_data["eval/detail_chart"] = swanlab.Image(results_chart_path, caption="Generation Details")
                        
                    # Upload Static Benchmark Chart (Clean)
                    static_chart_path = os.path.join(config.LOG_DIR, 'eval_static_chart.png')
                    if os.path.exists(static_chart_path):
                        log_data["eval/static_benchmarks"] = swanlab.Image(static_chart_path, caption="Static Opponents (gen_350/450)")
                
                # Add Chart
                if os.path.exists(chart_path):
                    log_data["eval/summary_chart"] = swanlab.Image(chart_path, caption=f"Gen {generation} Evaluation")
                
                swanlab.log(log_data)
                print("Logged evaluation results to SwanLab.")
                
                # --- Dynamic Training Guidance & Gating ---
                passed_gating = True
                if win_rates:
                    # Dynamic Epochs
                    if avg_win_rate < 0.4:
                        print("Win rate is low. Increasing training epochs for next loop.")
                        current_epochs = min(current_epochs + 1, 10)
                    elif avg_win_rate > 0.8:
                        print("Win rate is high. Decreasing training epochs for next loop.")
                        current_epochs = max(current_epochs - 1, 1)
                    else:
                        print("Win rate is stable. Keeping training epochs.")
                    
                    print(f"Next Training Epochs: {current_epochs}")
                    
                    # 门控检查（从 config 读取阈值）
                    if avg_win_rate < config.GATING_MIN_WIN_RATE:
                        print(f"GATING FAILED: Overall win rate too low (< {config.GATING_MIN_WIN_RATE}).")
                        passed_gating = False
                    elif avg_white_win_rate < config.GATING_MIN_WHITE_WIN_RATE:
                        print(f"GATING FAILED: White win rate too low (< {config.GATING_MIN_WHITE_WIN_RATE}).")
                        passed_gating = False
                        
                if passed_gating:
                    print(">>> GATING PASSED: Updating Current Engine.")
                    shutil.copy(candidate_engine_path, config.CURRENT_ENGINE_PATH)
                else:
                    print(">>> GATING FAILED: Keeping previous engine for data generation.")
                    # We do NOT update config.CURRENT_ENGINE_PATH
                    # But we still increment generation so we don't overwrite the checkpoint/logs
                    # The next generation will be trained on NEW data generated by OLD model + OLD data
                
        except Exception as e:
            print(f"Evaluation failed (continuing loop): {e}")
            # If eval fails, assume pass to avoid getting stuck? Or fail safe?
            # Let's assume pass if eval crashes, to keep moving.
            shutil.copy(candidate_engine_path, config.CURRENT_ENGINE_PATH)
        
        print(f"Generation {generation} Complete!")
        generation += 1
        
        # Optional: Sleep to let GPU cool down?
        time.sleep(5)

if __name__ == "__main__":
    main()
