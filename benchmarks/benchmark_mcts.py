"""固定搜索次数的 MCTS 吞吐基准，避免用棋局长度比较造成误判。"""

import argparse
import multiprocessing as mp
import os
import statistics
import sys
import time


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
RL_DIR = os.path.join(ROOT, 'reinforcement_learning')
sys.path.insert(0, RL_DIR)


def run_worker(
    worker_id,
    engine_path,
    simulations,
    batch_size,
    threads,
    warmup_simulations,
    repeats,
    gpu_id,
    result_queue,
):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    import torch
    from core.mcts import MCTSEngine

    engine = MCTSEngine(engine_path, device=torch.device('cuda:0'))
    engine.set_params(batch_size=batch_size, num_threads=threads)
    if warmup_simulations > 0:
        engine.reset()
        engine.run_simulations(warmup_simulations)
        torch.cuda.synchronize()

    worker_times = []
    worker_stats = []
    for repeat in range(repeats):
        engine.reset()
        engine.set_random_seed(2026 + repeat)
        started_at = time.perf_counter()
        engine.run_simulations(simulations)
        torch.cuda.synchronize()
        worker_times.append(time.perf_counter() - started_at)
        worker_stats.append(engine.get_search_stats())
    result_queue.put((worker_id, worker_times, worker_stats))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine', required=True)
    parser.add_argument('--simulations', type=int, default=12000)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--threads', type=int, default=5)
    parser.add_argument('--processes', type=int, default=1)
    parser.add_argument('--warmup-simulations', type=int, default=512)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--gpus', default='6', help='物理 GPU 编号，如 6 或 6,7')
    args = parser.parse_args()

    if args.repeats <= 0:
        raise ValueError('--repeats 必须大于 0')

    context = mp.get_context('spawn')
    result_queue = context.Queue()
    gpu_ids = [value.strip() for value in args.gpus.split(',') if value.strip()]
    if not gpu_ids:
        raise ValueError('--gpus 至少需要一个物理 GPU 编号')
    processes = [
        context.Process(
            target=run_worker,
            args=(
                worker_id,
                args.engine,
                args.simulations,
                args.batch_size,
                args.threads,
                args.warmup_simulations,
                args.repeats,
                gpu_ids[worker_id % len(gpu_ids)],
                result_queue,
            ),
        )
        for worker_id in range(args.processes)
    ]

    wall_started_at = time.perf_counter()
    for process in processes:
        process.start()
    for process in processes:
        process.join()
    wall_seconds = time.perf_counter() - wall_started_at

    failed = [process.exitcode for process in processes if process.exitcode != 0]
    if failed:
        raise RuntimeError(f'基准子进程失败: {failed}')
    results = [result_queue.get(timeout=5) for _ in processes]

    results.sort(key=lambda item: item[0])
    worker_times = [times for _, times, _ in results]
    worker_stats = [stats for _, _, stats in results]
    flat_times = [duration for times in worker_times for duration in times]
    # 多进程同时搜索时，最慢 worker 的累计时间决定稳态吞吐。
    search_makespan = max(sum(times) for times in worker_times)
    total_simulations = args.simulations * args.processes * args.repeats
    payload = {
            'processes': args.processes,
            'gpus': gpu_ids,
            'batch_size': args.batch_size,
            'simulations_per_process': args.simulations,
            'repeats': args.repeats,
            'startup_included_wall_seconds': round(wall_seconds, 4),
            'worker_seconds': [
                [round(value, 4) for value in times]
                for times in worker_times
            ],
            'median_search_seconds': round(statistics.median(flat_times), 4),
            'steady_simulations_per_second': round(
                total_simulations / search_makespan
            ),
    }
    valid_stats = [item for worker in worker_stats for item in worker if item]
    if valid_stats:
        leaf_requests = sum(item['leaf_requests'] for item in valid_stats)
        unique_evaluations = sum(item['unique_evaluations'] for item in valid_stats)
        payload['leaf_requests'] = leaf_requests
        payload['unique_evaluations'] = unique_evaluations
        payload['deduplication_ratio'] = round(
            1.0 - unique_evaluations / max(leaf_requests, 1),
            4,
        )
    print(payload)


if __name__ == '__main__':
    main()
