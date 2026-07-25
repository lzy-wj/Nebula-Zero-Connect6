"""用独立进程让同一模型的两组 MCTS 参数进行配对换色对局。"""

import argparse
import multiprocessing as mp
import os
import sys
import time
import traceback

import numpy as np


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
RL_DIR = os.path.join(ROOT, 'reinforcement_learning')
sys.path.insert(0, RL_DIR)

from core.connect6_game import Connect6Game


def agent_process(
    connection,
    engine_path,
    gpu_id,
    batch_size,
    threads,
    mcts_library=None,
    cpuct=1.5,
    widening_base=20,
    widening_scale=0.25,
    deterministic=False,
):
    """每个进程拥有独立的 C++ 树、TensorRT 上下文和 CUDA Graph。"""

    try:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        if mcts_library:
            os.environ['NEBULA_MCTS_LIBRARY'] = os.path.abspath(mcts_library)
        import torch
        from core.mcts import MCTSEngine

        engine = MCTSEngine(engine_path, torch.device('cuda:0'))
        engine.set_params(batch_size=batch_size, num_threads=threads)
        engine.set_search_params(cpuct, widening_base, widening_scale)
        engine.set_deterministic_selection(deterministic)
        connection.send(('ready', None))

        while True:
            command, payload = connection.recv()
            if command == 'stop':
                break
            if command == 'reset':
                engine.reset()
                connection.send(('ok', None))
            elif command == 'play':
                engine.update_state(int(payload))
                connection.send(('ok', None))
            elif command == 'search':
                simulations = int(payload)
                started_at = time.perf_counter()
                engine.run_simulations(simulations)
                move = engine.get_mcts_move(simulations=0, temperature=0.0)
                connection.send(
                    (
                        'move',
                        {
                            'move': move,
                            'seconds': time.perf_counter() - started_at,
                            'stats': engine.get_search_stats(),
                        },
                    )
                )
            else:
                raise ValueError(f'未知命令: {command}')
    except Exception:
        connection.send(('error', traceback.format_exc()))
    finally:
        connection.close()


def expect(connection, expected):
    status, payload = connection.recv()
    if status == 'error':
        raise RuntimeError(payload)
    if status != expected:
        raise RuntimeError(f'期望 {expected}，实际收到 {status}: {payload}')
    return payload


def build_paired_openings(games, seed, opening_stones):
    rng = np.random.default_rng(seed)
    center = 9 * 19 + 9
    candidates = [
        row * 19 + col
        for row in range(5, 14)
        for col in range(5, 14)
        if row * 19 + col != center
    ]
    pairs = []
    for _ in range((games + 1) // 2):
        opening = [center]
        if opening_stones > 1:
            opening.extend(
                int(move)
                for move in rng.choice(
                    candidates,
                    size=opening_stones - 1,
                    replace=False,
                )
            )
        pairs.append(opening)
    return [pairs[index // 2] for index in range(games)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine', required=True)
    parser.add_argument('--engine-b', default=None)
    parser.add_argument('--gpu', default='6', help='兼容旧命令：同时指定 A/B 使用的卡')
    parser.add_argument('--gpu-a', default=None, help='A 引擎使用的物理 GPU')
    parser.add_argument('--gpu-b', default=None, help='B 引擎使用的物理 GPU')
    parser.add_argument('--games', type=int, default=20)
    parser.add_argument('--batch-size-a', type=int, default=64)
    parser.add_argument('--batch-size-b', type=int, default=64)
    parser.add_argument('--threads-a', type=int, default=5)
    parser.add_argument('--threads-b', type=int, default=24)
    parser.add_argument('--mcts-library-a', default=None)
    parser.add_argument('--mcts-library-b', default=None)
    parser.add_argument('--cpuct-a', type=float, default=1.5)
    parser.add_argument('--cpuct-b', type=float, default=1.5)
    parser.add_argument('--widening-base-a', type=int, default=20)
    parser.add_argument('--widening-base-b', type=int, default=20)
    parser.add_argument('--widening-scale-a', type=float, default=0.25)
    parser.add_argument('--widening-scale-b', type=float, default=0.25)
    parser.add_argument('--deterministic-a', action='store_true')
    parser.add_argument('--deterministic-b', action='store_true')
    parser.add_argument('--black-simulations', type=int, default=400)
    parser.add_argument('--white-simulations', type=int, default=1200)
    parser.add_argument('--opening-stones', type=int, default=5)
    parser.add_argument('--seed', type=int, default=2026)
    args = parser.parse_args()
    engine_b = args.engine_b or args.engine
    gpu_a = args.gpu_a or args.gpu
    gpu_b = args.gpu_b or args.gpu

    context = mp.get_context('spawn')
    parent_a, child_a = context.Pipe()
    parent_b, child_b = context.Pipe()
    processes = [
        context.Process(
            target=agent_process,
            args=(
                child_a,
                args.engine,
                gpu_a,
                args.batch_size_a,
                args.threads_a,
                args.mcts_library_a,
                args.cpuct_a,
                args.widening_base_a,
                args.widening_scale_a,
                args.deterministic_a,
            ),
        ),
        context.Process(
            target=agent_process,
            args=(
                child_b,
                engine_b,
                gpu_b,
                args.batch_size_b,
                args.threads_b,
                args.mcts_library_b,
                args.cpuct_b,
                args.widening_base_b,
                args.widening_scale_b,
                args.deterministic_b,
            ),
        ),
    ]
    for process in processes:
        process.start()
    expect(parent_a, 'ready')
    expect(parent_b, 'ready')

    openings = build_paired_openings(args.games, args.seed, args.opening_stones)
    results = {'a_wins': 0, 'b_wins': 0, 'draws': 0}
    search_seconds = {'a': 0.0, 'b': 0.0}
    searched_simulations = {'a': 0, 'b': 0}

    try:
        for game_index, opening in enumerate(openings):
            game = Connect6Game()
            for connection in (parent_a, parent_b):
                connection.send(('reset', None))
                expect(connection, 'ok')

            for move in opening:
                game.play(move)
                for connection in (parent_a, parent_b):
                    connection.send(('play', move))
                    expect(connection, 'ok')

            # 每两局共享开局并交换颜色。
            a_color = 1 if game_index % 2 == 0 else -1
            while game.winner == 0:
                is_a = game.current_player == a_color
                connection = parent_a if is_a else parent_b
                label = 'a' if is_a else 'b'
                simulations = (
                    args.black_simulations
                    if game.current_player == 1
                    else args.white_simulations
                )
                connection.send(('search', simulations))
                result = expect(connection, 'move')
                move = int(result['move'])
                if not 0 <= move < 361 or game.board.flat[move] != 0:
                    raise RuntimeError(f'第 {game_index} 局出现非法落点: {move}')

                search_seconds[label] += float(result['seconds'])
                searched_simulations[label] += simulations
                game.play(move)
                for target in (parent_a, parent_b):
                    target.send(('play', move))
                    expect(target, 'ok')

            if game.winner == 2:
                results['draws'] += 1
            elif game.winner == a_color:
                results['a_wins'] += 1
            else:
                results['b_wins'] += 1
            print(
                f"对局 {game_index + 1}/{args.games}: "
                f"winner={game.winner}, moves={len(game.moves)}, A颜色={a_color}"
            )
    finally:
        for connection in (parent_a, parent_b):
            try:
                connection.send(('stop', None))
            except (BrokenPipeError, EOFError):
                pass
        for process in processes:
            process.join(timeout=10)
            if process.is_alive():
                process.terminate()

    results.update(
        {
            'threads_a': args.threads_a,
            'threads_b': args.threads_b,
            'gpu_a': gpu_a,
            'gpu_b': gpu_b,
            'batch_size_a': args.batch_size_a,
            'batch_size_b': args.batch_size_b,
            'mcts_library_a': args.mcts_library_a,
            'mcts_library_b': args.mcts_library_b,
            'search_seconds_a': round(search_seconds['a'], 3),
            'search_seconds_b': round(search_seconds['b'], 3),
            'simulations_per_second_a': round(
                searched_simulations['a'] / max(search_seconds['a'], 1e-9)
            ),
            'simulations_per_second_b': round(
                searched_simulations['b'] / max(search_seconds['b'], 1e-9)
            ),
        }
    )
    print(results)


if __name__ == '__main__':
    main()
