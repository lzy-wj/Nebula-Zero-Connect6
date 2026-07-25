"""按代启动强化学习子进程，保证配置和 SwanLab 认证状态逐代刷新。"""

import argparse
import glob
import json
import os
import subprocess
import sys

import swanlab

import config


def load_json(path, default=None):
    try:
        with open(path, 'r', encoding='utf-8') as file:
            return json.load(file)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return default


def atomic_text_write(path, value):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    temp_path = f'{path}.tmp'
    with open(temp_path, 'w', encoding='utf-8') as file:
        file.write(str(value))
    os.replace(temp_path, path)


def project_slug():
    return ''.join(
        char if char.isalnum() or char in '-_' else '_'
        for char in config.SWANLAB_PROJECT
    )


def initialize_loop_swanlab(mode):
    """初始化跨代总览实验；每代详细信息由子进程单独记录。"""
    slug = project_slug()
    run_id_dir = os.path.join(config.LOG_DIR, 'swanlab_runs', slug)
    os.makedirs(run_id_dir, exist_ok=True)
    run_id_path = os.path.join(run_id_dir, 'loop.txt')
    legacy_id_path = os.path.join(config.LOG_DIR, f'swanlab_run_id_{slug}.txt')

    existing_run_id = None
    for candidate_path in (run_id_path, legacy_id_path):
        try:
            with open(candidate_path, 'r', encoding='utf-8') as file:
                existing_run_id = file.read().strip() or None
        except FileNotFoundError:
            continue
        if existing_run_id:
            break

    if config.SELFPLAY_BACKEND == 'remote':
        selfplay_backend = 'remote'
        selfplay_host = config.REMOTE_SELFPLAY_HOST
        selfplay_gpus = config.REMOTE_SELFPLAY_GPUS
    else:
        selfplay_backend = 'local'
        selfplay_host = ''
        selfplay_gpus = config.GPUS

    init_kwargs = {
        'project': config.SWANLAB_PROJECT,
        'name': config.SWANLAB_LOOP_RUN_NAME,
        'description': 'Nebula Zero 跨代训练总览与 incumbent 趋势',
        'config': {
            'role': 'generation_loop',
            'selfplay_backend': selfplay_backend,
            'selfplay_host': selfplay_host,
            'selfplay_gpus': selfplay_gpus,
            'training_gpus': config.TRAINING_GPUS,
            'train_precision': config.TRAIN_PRECISION,
        },
    }
    if mode:
        init_kwargs['mode'] = mode
    if existing_run_id:
        init_kwargs.update({'id': existing_run_id, 'resume': 'allow'})

    try:
        run = swanlab.init(**init_kwargs)
        run_id = getattr(run, 'id', None)
        if run_id:
            atomic_text_write(run_id_path, run_id)
        return True
    except Exception as error:
        print(f'最外层 SwanLab loop 初始化失败，将继续本地训练: {error}', flush=True)
        return False


def log_loop_summary(summary):
    """把一代的关键标量写入跨代总览，不混入本代详细图表。"""
    generation = int(summary['generation'])
    payload = {
        'generation': generation,
        'loop/candidate_accepted': int(bool(summary.get('accepted'))),
    }
    accepted_generation = summary.get('accepted_generation')
    if accepted_generation is not None:
        payload['loop/accepted_generation'] = accepted_generation
    if 'buffer_size' in summary:
        payload['loop/buffer_size'] = summary['buffer_size']
    if 'current_epochs' in summary:
        payload['loop/current_epochs'] = summary['current_epochs']
    if 'selfplay_games_per_second' in summary:
        payload['performance/selfplay_games_per_second'] = summary[
            'selfplay_games_per_second'
        ]
    for key, value in summary.get('train', {}).items():
        payload[f'train/{key}'] = value
    for key, value in summary.get('eval', {}).items():
        payload[f'eval/{key}'] = value
    for key, value in summary.get('phase_times', {}).items():
        payload[f'timing/{key}'] = value

    trend_path = os.path.join(config.LOG_DIR, 'eval_trend.png')
    if os.path.exists(trend_path):
        payload['loop/gating_trend'] = swanlab.Image(trend_path)
    swanlab.log(payload, step=generation * 4 + 3)


def sync_loop_summaries(active):
    """补传尚未进入 loop 的代摘要，支持监督器异常后的断点恢复。"""
    if not active:
        return
    slug = project_slug()
    marker_path = os.path.join(
        config.LOG_DIR,
        'swanlab_runs',
        slug,
        'loop_last_generation.txt',
    )
    marker_exists = True
    try:
        with open(marker_path, 'r', encoding='utf-8') as file:
            last_generation = int(file.read().strip())
    except (FileNotFoundError, ValueError, OSError):
        marker_exists = False
        # 从每代独立实验开始的位置推断旧 loop 已记录到哪一代，避免重复上传。
        generation_run_pattern = os.path.join(
            config.LOG_DIR,
            'swanlab_runs',
            slug,
            'gen_*.txt',
        )
        generation_runs = []
        for path in glob.glob(generation_run_pattern):
            name = os.path.splitext(os.path.basename(path))[0]
            try:
                generation_runs.append(int(name.rsplit('_', 1)[1]))
            except (IndexError, ValueError):
                continue
        last_generation = min(generation_runs) - 1 if generation_runs else -1

    pattern = os.path.join(config.LOG_DIR, 'generation_summaries', 'gen_*.json')
    summaries_by_generation = {}

    # 切换到逐代实验之前的少量历史代，使用本地门控历史补齐 loop 标量。
    history_path = os.path.join(config.LOG_DIR, 'eval_history.json')
    history = load_json(history_path, [])
    accepted_generation = None
    if isinstance(history, list):
        for entry in sorted(history, key=lambda item: int(item.get('generation', -1))):
            generation = int(entry.get('generation', -1))
            if generation < 0:
                continue
            if entry.get('accepted'):
                accepted_generation = generation
            summaries_by_generation[generation] = {
                'generation': generation,
                'accepted': bool(entry.get('accepted')),
                'accepted_generation': accepted_generation,
                'eval': {
                    'incumbent_score_rate': entry.get('incumbent_score_rate', 0),
                    'incumbent_white_win_rate': entry.get('white_win_rate', 0),
                },
            }

    for path in glob.glob(pattern):
        summary = load_json(path)
        if isinstance(summary, dict) and 'generation' in summary:
            summaries_by_generation[int(summary['generation'])] = summary
    summaries = [
        summaries_by_generation[generation]
        for generation in sorted(summaries_by_generation)
    ]

    for summary in summaries:
        generation = int(summary['generation'])
        if generation <= last_generation:
            continue
        try:
            log_loop_summary(summary)
        except Exception as error:
            print(f'最外层 loop 记录第 {generation} 代失败: {error}', flush=True)
            return
        atomic_text_write(marker_path, generation)
        last_generation = generation

    if not marker_exists and last_generation >= 0 and not os.path.exists(marker_path):
        atomic_text_write(marker_path, last_generation)


def main():
    parser = argparse.ArgumentParser(description='Nebula Zero 按代常驻监督器')
    parser.add_argument(
        '--swanlab-mode',
        choices=['online', 'offline', 'local', 'disabled'],
        default=os.environ.get('NEBULA_SWANLAB_MODE', 'online'),
    )
    args = parser.parse_args()

    loop_active = initialize_loop_swanlab(args.swanlab_mode)
    loop_finish_state = 'aborted'
    sync_loop_summaries(loop_active)

    loop_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'run_loop.py')
    command = [
        sys.executable,
        loop_script,
        '--max-generations',
        '1',
        '--swanlab-mode',
        args.swanlab_mode,
    ]

    try:
        while True:
            print('\n按代监督器：启动下一代主循环', flush=True)
            try:
                result = subprocess.run(command, check=False)
            except KeyboardInterrupt:
                print('\n按代监督器收到退出信号，停止启动新一代。', flush=True)
                return

            if result.returncode != 0:
                loop_finish_state = 'crashed'
                print(
                    f'本代主循环异常退出（状态码 {result.returncode}），'
                    '为保护数据，不自动重试。',
                    flush=True,
                )
                raise SystemExit(result.returncode)
            sync_loop_summaries(loop_active)
    finally:
        if loop_active:
            try:
                swanlab.finish(
                    state=loop_finish_state,
                    async_log_timeout=config.SWANLAB_FINISH_TIMEOUT,
                )
            except Exception as error:
                print(f'最外层 SwanLab loop 收尾失败: {error}', flush=True)


if __name__ == '__main__':
    main()
