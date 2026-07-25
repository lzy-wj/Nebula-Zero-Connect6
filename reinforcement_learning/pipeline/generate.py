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


def normalize_seed(seed):
    """把任意代号派生的种子限制在 NumPy 和 C++ int 的共同安全范围。"""
    return int(seed) % 2_147_483_647


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


def build_forced_opening(game_seed):
    """Build a deterministic, legal central opening for a configured game subset."""

    ratio = min(1.0, max(0.0, float(getattr(config, 'FORCED_OPENING_RATIO', 0.0))))
    stones = max(0, int(getattr(config, 'FORCED_OPENING_STONES', 0)))
    if ratio <= 0.0 or stones <= 0:
        return []
    rng = np.random.default_rng(normalize_seed(game_seed + 7919))
    if float(rng.random()) >= ratio:
        return []
    radius = min(9, max(0, int(getattr(config, 'FORCED_OPENING_RADIUS', 4))))
    candidates = [
        row * 19 + column
        for row in range(9 - radius, 10 + radius)
        for column in range(9 - radius, 10 + radius)
    ]
    count = min(stones, len(candidates))
    return [int(value) for value in rng.choice(candidates, size=count, replace=False)]

def worker_process(
    gpu_id,
    engine_path,
    sims,
    games_to_play,
    out_file,
    worker_id,
    lock,
    seed,
    opponent_engine_path=None,
    mix_ratio=0.0,
    game_counter=None,
    counter_lock=None,
    total_games=None,
):
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
            mcts.set_search_params(
                cpuct=config.MCTS_CPUCT,
                widening_base=config.MCTS_WIDENING_BASE,
                widening_scale=config.MCTS_WIDENING_SCALE,
            )
            mcts.set_deterministic_selection(config.MCTS_DETERMINISTIC_SELECTION)
            
        worker_seed = normalize_seed(seed + worker_id * 10000)
        mcts.set_random_seed(worker_seed)
        np.random.seed(worker_seed)
    except Exception as e:
        print(f"[Worker {worker_id}] Init Error: {e}")
        raise RuntimeError(f"Worker {worker_id} 初始化失败") from e

    mcts_opponent = None
    if opponent_engine_path and os.path.exists(opponent_engine_path):
        try:
            print(f"[Worker {worker_id}] Loading Opponent Engine: {opponent_engine_path}")
            mcts_opponent = MCTSEngine(opponent_engine_path, device=device)
            # Use same params for opponent? Or weaker? Default to same for now.
            if hasattr(config, 'MCTS_THREADS'):
                mcts_opponent.set_params(batch_size=config.MCTS_BATCH_SIZE, num_threads=config.MCTS_THREADS)
                mcts_opponent.set_search_params(
                    cpuct=config.MCTS_CPUCT,
                    widening_base=config.MCTS_WIDENING_BASE,
                    widening_scale=config.MCTS_WIDENING_SCALE,
                )
            mcts_opponent.set_random_seed(worker_seed + 1)
        except Exception as e:
            print(f"[Worker {worker_id}] Opponent Init Error: {e}")
            mcts_opponent = None

    worker_started_at = time.perf_counter()
    worker_leaf_requests = 0
    worker_unique_evaluations = 0
    completed_games = 0
    claimed_games = 0

    def claim_game_index():
        """双卡动态领取下一局，避免长短局造成一张卡提前空等。"""

        nonlocal claimed_games
        if game_counter is None:
            if claimed_games >= games_to_play:
                return None
            game_index = claimed_games
            claimed_games += 1
            return game_index
        with counter_lock:
            if game_counter.value >= total_games:
                return None
            game_index = int(game_counter.value)
            game_counter.value += 1
            return game_index

    while True:
        game_index = claim_game_index()
        if game_index is None:
            break

        # 随机性绑定全局对局编号，不依赖这局最终被卡 6 还是卡 7 领取。
        game_seed = normalize_seed(seed + game_index * 1009)
        mcts.set_random_seed(game_seed)
        if mcts_opponent is not None:
            mcts_opponent.set_random_seed(normalize_seed(game_seed + 1))
        np.random.seed(game_seed)
        game = Connect6Game()
        # C++ 搜索状态是进程内单例；每局开始时明确切回当前模型。
        from core.mcts import mcts_lib
        mcts_lib.set_eval_callback(mcts.c_callback)
        mcts.reset()
            
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
                    
                    # 残局只放大一次搜索预算，避免旧代码重复乘成 2.25 倍。
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

                        # 两个 Python 包装器共享同一个 C++ 全局棋盘和回调。
                        # 切换模型时必须重新注册回调并重放局面，否则会串树、串模型。
                        mcts_lib.set_eval_callback(active_mcts.c_callback)
                        mcts_lib.init_game()
                        for historical_move in game.moves:
                            parsed = game._parse_coord(historical_move)
                            if parsed is None:
                                continue
                            hr, hc = parsed
                            mcts_lib.play_move(hr * 19 + hc)
                    
                    # === 可选的动态搜索循环 ===
                    sims_done = 0
                    check_interval = getattr(config, 'DYNAMIC_CHECK_INTERVAL', 400)
                    fuse_ratio = getattr(config, 'DYNAMIC_FUSE_RATIO', 10.0)
                    early_stop = getattr(config, 'DYNAMIC_EARLY_STOP', False)
                    
                    # 默认严格执行黑/白双方配置的完整模拟次数。只有显式开启
                    # DYNAMIC_EARLY_STOP 时，才允许优势明显的节点提前结束。
                    if early_stop and current_sims > check_interval:
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
                    if move_idx < 0 or move_idx >= 361:
                        raise RuntimeError(f"Worker {worker_id} 搜索返回非法落点: {move_idx}")
                    
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
                    
                    # 非对称模式每步都会重放；普通自对弈则复用搜索树。
                    if not is_asymmetric:
                        mcts.update_state(move_idx)
                    
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

        if not is_asymmetric:
            search_stats = mcts.get_search_stats()
            worker_leaf_requests += search_stats.get('leaf_requests', 0)
            worker_unique_evaluations += search_stats.get('unique_evaluations', 0)
                
        completed_games += 1
        display_total = total_games if total_games is not None else games_to_play
        print(
            f"[Worker {worker_id}] 全局第 {game_index + 1}/{display_total} 局 "
            f"Winner: {winner_str}"
        )

    elapsed = time.perf_counter() - worker_started_at
    speed = completed_games / elapsed if elapsed > 0 else 0.0
    print(
        f"[Worker {worker_id}] 完成 {completed_games} 局，耗时 {elapsed:.1f}s，"
        f"速度 {speed:.3f} 局/s"
    )
    if worker_leaf_requests:
        saved_ratio = 1.0 - worker_unique_evaluations / worker_leaf_requests
        print(
            f"[Worker {worker_id}] 叶子请求 {worker_leaf_requests}，"
            f"实际网络评估 {worker_unique_evaluations}，去重节省 {saved_ratio:.1%}"
        )


def batched_worker_process(
    gpu_id,
    engine_path,
    sims,
    games_to_play,
    out_file,
    worker_id,
    lock,
    seed,
    opponent_engine_path=None,
    mix_ratio=0.0,
    game_counter=None,
    counter_lock=None,
    total_games=None,
):
    """单卡单 TensorRT context，同时推进多盘自对弈以填满网络 batch。"""

    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    device = torch.device('cuda:0')
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    if opponent_engine_path and mix_ratio > 0.0:
        raise ValueError("多棋局合批暂不支持非对称双模型，请将并发棋局数设为 1")

    try:
        mcts = MCTSEngine(engine_path, device=device)
        if not mcts.supports_multi_context:
            raise RuntimeError("当前 libmcts.so 不支持多棋局上下文")
        mcts.set_params(
            batch_size=config.MCTS_BATCH_SIZE,
            num_threads=config.MCTS_THREADS,
        )
        mcts.set_search_params(
            cpuct=config.MCTS_CPUCT,
            widening_base=config.MCTS_WIDENING_BASE,
            widening_scale=config.MCTS_WIDENING_SCALE,
        )
        mcts.set_deterministic_selection(config.MCTS_DETERMINISTIC_SELECTION)
        cache_capacity = getattr(config, 'MCTS_EVAL_CACHE_SIZE', 32768)
        mcts.set_eval_cache_capacity(cache_capacity)
        mcts.reset_search_stats()
    except Exception as exc:
        raise RuntimeError(f"Worker {worker_id} 初始化多棋局 MCTS 失败") from exc

    concurrent_games = max(1, int(getattr(config, 'MCTS_CONCURRENT_GAMES', 1)))
    completed_games = 0
    claimed_games = 0
    worker_started_at = time.perf_counter()

    def claim_game_index():
        """动态领取全局对局编号，让各张卡始终有棋局可推进。"""

        nonlocal claimed_games
        if game_counter is None:
            if claimed_games >= games_to_play:
                return None
            game_index = claimed_games
            claimed_games += 1
            return game_index
        with counter_lock:
            if game_counter.value >= total_games:
                return None
            game_index = int(game_counter.value)
            game_counter.value += 1
            return game_index

    def create_slot():
        game_index = claim_game_index()
        if game_index is None:
            return None
        game_seed = normalize_seed(seed + game_index * 1009)
        game = Connect6Game()
        context = mcts.create_game_context(seed=game_seed)
        opening = build_forced_opening(game_seed)
        for move_index in opening:
            game.play(move_index)
            context.update_state(move_index)
        return {
            'index': game_index,
            'game': game,
            'context': context,
            # 空策略让联合训练跳过随机注入着法，但保持逐手字段严格对齐。
            'policies': [''] * len(opening),
            'bonuses': ['0.00'] * len(opening),
            'forced_opening': bool(opening),
        }

    def search_budget(game):
        budget = sims
        if hasattr(config, 'SIMULATIONS_BLACK') and hasattr(config, 'SIMULATIONS_WHITE'):
            budget = (
                config.SIMULATIONS_BLACK
                if game.current_player == 1
                else config.SIMULATIONS_WHITE
            )
        # 与旧生成器完全一致：残局只放大一次，不降低任何一方的搜索次数。
        if len(game.moves) > 75:
            budget = int(budget * 1.5)
        return budget

    def move_temperature(game):
        if len(game.moves) >= config.OPENING_MOVES:
            return config.TEMP_FINAL
        if hasattr(config, 'TEMP_OPENING_BLACK') and hasattr(config, 'TEMP_OPENING_WHITE'):
            return (
                config.TEMP_OPENING_BLACK
                if game.current_player == 1
                else config.TEMP_OPENING_WHITE
            )
        return config.TEMP_OPENING

    def finish_slot(slot):
        nonlocal completed_games
        game = slot['game']
        winner = 'draw'
        if game.winner == 1:
            winner = 'black'
        elif game.winner == -1:
            winner = 'white'

        moves_str = ",".join(game.moves)
        policies_str = "|".join(slot['policies'])
        bonuses_str = ",".join(slot['bonuses'])
        with lock:
            with open(out_file, 'a') as output_file:
                output_file.write(
                    f'"{moves_str}",{winner},"{policies_str}","{bonuses_str}"\n'
                )

        slot['context'].close()
        completed_games += 1
        display_total = total_games if total_games is not None else games_to_play
        print(
            f"[Worker {worker_id}] 全局第 {slot['index'] + 1}/{display_total} 局 "
            f"Winner: {winner} | 活跃棋局: {len(active_slots)}"
        )

    active_slots = []
    for _ in range(concurrent_games):
        slot = create_slot()
        if slot is None:
            break
        active_slots.append(slot)

    print(
        f"[Worker {worker_id}] GPU {gpu_id} 启动单 context 多棋局合批："
        f"并发 {len(active_slots)}，MCTS batch {config.MCTS_BATCH_SIZE}，"
        f"线程 {config.MCTS_THREADS}，缓存 {getattr(config, 'MCTS_EVAL_CACHE_SIZE', 32768)}，"
        f"随机开局比例 {getattr(config, 'FORCED_OPENING_RATIO', 0.0):.0%}"
    )

    while active_slots:
        contexts = [slot['context'] for slot in active_slots]
        budgets = [search_budget(slot['game']) for slot in active_slots]
        # 即使旧配置打开了动态熔断，合批路径仍执行完整预算，保证搜索深度不降。
        mcts.run_simulations_multi(contexts, budgets)

        next_slots = []
        for slot in active_slots:
            game = slot['game']
            context = slot['context']
            temperature = move_temperature(game)
            move_index = context.get_mcts_move(simulations=0, temperature=temperature)
            if move_index < 0 or move_index >= 361:
                raise RuntimeError(
                    f"Worker {worker_id} 第 {slot['index']} 局返回非法落点: {move_index}"
                )
            row, column = divmod(move_index, 19)
            if game.board[row, column] != 0:
                raise RuntimeError(
                    f"Worker {worker_id} 第 {slot['index']} 局搜索落在已有棋子: {move_index}"
                )

            slot['policies'].append(encode_policy(context.get_policy()))
            slot['bonuses'].append("0.00")
            game.play(move_index)
            context.update_state(move_index)

            if game.winner != 0 or len(game.moves) >= 361:
                finish_slot(slot)
                replacement = create_slot()
                if replacement is not None:
                    next_slots.append(replacement)
            else:
                next_slots.append(slot)
        active_slots = next_slots

    elapsed = time.perf_counter() - worker_started_at
    stats = mcts.get_search_stats()
    speed = completed_games / max(elapsed, 1e-9)
    print(
        f"[Worker {worker_id}] 合批完成 {completed_games} 局，耗时 {elapsed:.1f}s，"
        f"速度 {speed:.3f} 局/s，统计: {stats}"
    )

def main():
    config.ensure_dirs()
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=str, default=os.path.join(config.RAW_DATA_DIR, 'gen_data.csv'))
    parser.add_argument('--total', type=int, default=config.GAMES_PER_LOOP)
    parser.add_argument('--engine', type=str, default=config.INITIAL_MODEL_PATH, help='Path to TensorRT Engine')
    parser.add_argument('--opponent', type=str, default=None, help='Path to Opponent Engine for Asymmetric Play')
    parser.add_argument('--mix_ratio', type=float, default=0.0, help='Ratio of asymmetric games')
    parser.add_argument('--seed', type=int, default=config.SEED, help='Base random seed')
    args = parser.parse_args()
    
    # Init CSV header if new
    if not os.path.exists(args.out):
        with open(args.out, 'w') as f:
            f.write("moves,winner,policies,bonuses\n")

    # 断点续跑时从现有行数继续编号，使每局随机种子不会与已生成数据重复。
    with open(args.out, 'r') as existing_file:
        start_index = max(0, sum(1 for _ in existing_file) - 1)
            
    gpu_list = [int(x) for x in config.GPUS.split(',')]
    lock = mp.Lock()
    processes = []
    
    print(f"Generating {args.total} games using Engine: {args.engine}")
    
    worker_count = min(config.NUM_WORKERS, args.total)
    game_counter = mp.Value('i', start_index)
    counter_lock = mp.Lock()
    final_index = start_index + args.total

    use_multigame_batching = (
        getattr(config, 'MCTS_CONCURRENT_GAMES', 1) > 1
        and not (args.opponent and args.mix_ratio > 0.0)
    )
    worker_target = batched_worker_process if use_multigame_batching else worker_process
    print(
        "Self-play mode: "
        + (
            f"multi-game batching ({config.MCTS_CONCURRENT_GAMES} games/GPU)"
            if use_multigame_batching
            else "legacy single-game search"
        )
    )

    for i in range(worker_count):
        gpu_id = gpu_list[i % len(gpu_list)]
        p = mp.Process(
            target=worker_target,
            args=(
                gpu_id,
                args.engine,
                config.SIMULATIONS,
                0,
                args.out,
                i,
                lock,
                args.seed,
                args.opponent,
                args.mix_ratio,
                game_counter,
                counter_lock,
                final_index,
            ),
        )
        p.start()
        processes.append(p)
        
    try:
        for p in processes:
            p.join()
        failed = [p.pid for p in processes if p.exitcode != 0]
        if failed:
            raise RuntimeError(f"自对弈 worker 异常退出: {failed}")
    except KeyboardInterrupt:
        print("Interrupted. Terminating...")
        for p in processes:
            p.terminate()
            
if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()
