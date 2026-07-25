"""测量单 TensorRT context 跨多盘棋合批后的固定搜索吞吐。"""

import argparse
import json
import os
import statistics
import sys
import time

import numpy as np


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
RL_DIR = os.path.join(ROOT, 'reinforcement_learning')
sys.path.insert(0, RL_DIR)


def build_contexts(engine, count, opening_stones, seed):
    """构造不同但合法的开局，避免基准被相同空棋盘的缓存命中主导。"""

    rng = np.random.default_rng(seed)
    contexts = []
    center = 9 * 19 + 9
    all_moves = np.arange(361, dtype=np.int32)
    for context_index in range(count):
        context = engine.create_game_context(seed=seed + context_index * 1009)
        opening = [center]
        if opening_stones > 1:
            candidates = all_moves[all_moves != center]
            opening.extend(
                int(move)
                for move in rng.choice(
                    candidates,
                    size=opening_stones - 1,
                    replace=False,
                )
            )
        for move in opening:
            context.update_state(move)
        contexts.append(context)
    return contexts


def close_contexts(contexts):
    for context in contexts:
        context.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine', required=True)
    parser.add_argument('--mcts-library', default=None)
    parser.add_argument('--concurrency', default='1,4,8,16,32')
    parser.add_argument('--simulations', type=int, default=1200)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--threads', type=int, default=32)
    parser.add_argument('--opening-stones', type=int, default=9)
    parser.add_argument('--warmup-simulations', type=int, default=256)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--cache-capacity', type=int, default=32768)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--deterministic', action='store_true')
    args = parser.parse_args()

    if args.mcts_library:
        os.environ['NEBULA_MCTS_LIBRARY'] = os.path.abspath(args.mcts_library)

    import torch
    from core.mcts import MCTSEngine

    engine = MCTSEngine(args.engine, device=torch.device('cuda:0'))
    if not engine.supports_multi_context:
        raise RuntimeError('当前 MCTS 动态库不支持多棋局基准')
    engine.set_params(batch_size=args.batch_size, num_threads=args.threads)
    engine.set_deterministic_selection(args.deterministic)
    engine.set_eval_cache_capacity(args.cache_capacity)

    warmup_contexts = build_contexts(engine, 2, args.opening_stones, args.seed - 1)
    engine.run_simulations_multi(
        warmup_contexts,
        [args.warmup_simulations] * len(warmup_contexts),
    )
    torch.cuda.synchronize()
    close_contexts(warmup_contexts)

    concurrency_values = [
        int(value)
        for value in args.concurrency.split(',')
        if value.strip()
    ]
    results = []
    for concurrency in concurrency_values:
        durations = []
        repeat_stats = []
        for repeat in range(args.repeats):
            # 各并发档使用同一组嵌套开局，并在每轮前清缓存，保证横向比较
            # 不会被前一个档位留下的命中或不同棋局难度干扰。
            engine.clear_eval_cache()
            contexts = build_contexts(
                engine,
                concurrency,
                args.opening_stones,
                args.seed + repeat * 100000,
            )
            engine.reset_search_stats()
            started_at = time.perf_counter()
            engine.run_simulations_multi(
                contexts,
                [args.simulations] * concurrency,
            )
            torch.cuda.synchronize()
            durations.append(time.perf_counter() - started_at)
            stats = engine.get_search_stats()
            visit_counts = {threshold: 0 for threshold in (1, 2, 4, 8, 16, 32)}
            for context in contexts:
                for threshold, count in context.get_second_stone_visit_counts().items():
                    visit_counts[threshold] += count
            stats['second_stone_nodes_by_min_visits'] = visit_counts
            repeat_stats.append(stats)
            close_contexts(contexts)

        total_simulations = args.simulations * concurrency
        median_duration = statistics.median(durations)
        median_index = min(
            range(len(durations)),
            key=lambda index: abs(durations[index] - median_duration),
        )
        results.append({
            'concurrency': concurrency,
            'simulations_per_repeat': total_simulations,
            'seconds': [round(value, 5) for value in durations],
            'median_simulations_per_second': round(
                total_simulations / max(median_duration, 1e-9)
            ),
            'median_stats': repeat_stats[median_index],
        })

    print(json.dumps({
        'engine': args.engine,
        'mcts_library': args.mcts_library,
        'batch_size': args.batch_size,
        'threads': args.threads,
        'simulations_per_context': args.simulations,
        'cache_capacity': args.cache_capacity,
        'results': results,
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
