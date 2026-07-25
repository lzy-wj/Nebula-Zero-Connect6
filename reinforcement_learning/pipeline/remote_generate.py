"""把本代自对弈安全地交给远端 GPU 服务器，并把棋谱取回本地主循环。"""

import argparse
import hashlib
import os
import shlex
import subprocess
import sys
import time


RL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
PROJECT_ROOT = os.path.abspath(os.path.join(RL_DIR, '..'))
sys.path.insert(0, RL_DIR)

import config


CSV_HEADER = 'moves,winner,policies,bonuses'


def display_command(command):
    """只显示命令结构；项目路径和参数均使用列表传递。"""
    return shlex.join(str(part) for part in command)


def run_command(command, capture_output=False):
    print(f"远端执行器: {display_command(command)}", flush=True)
    return subprocess.run(
        command,
        check=True,
        text=True,
        capture_output=capture_output,
    )


def ssh_base(host):
    return [
        'ssh',
        '-o', 'BatchMode=yes',
        '-o', 'ConnectTimeout=15',
        '-o', 'ServerAliveInterval=30',
        '-o', 'ServerAliveCountMax=3',
        '-o', 'ControlMaster=auto',
        '-o', 'ControlPersist=300',
        '-o', f'ControlPath={config.REMOTE_SSH_CONTROL_PATH}',
        host,
    ]


def ssh_run(host, remote_command, capture_output=False):
    return run_command(
        [*ssh_base(host), remote_command],
        capture_output=capture_output,
    )


def quote_join(parts):
    return shlex.join(str(part) for part in parts)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def csv_rows(path):
    if not os.path.exists(path):
        return 0
    with open(path, 'r', encoding='utf-8', errors='replace') as file:
        first_line = file.readline().strip()
        if first_line != CSV_HEADER:
            raise ValueError(f'棋谱表头异常: {path}: {first_line!r}')
        return sum(1 for line in file if line.strip())


def validate_csv(path, expected_rows=None):
    rows = csv_rows(path)
    if expected_rows is not None and rows != expected_rows:
        raise RuntimeError(f'棋谱数量异常: {path} 有 {rows} 局，期望 {expected_rows} 局')
    return rows


def rsync_base():
    ssh_transport = quote_join([
        'ssh',
        '-o', 'BatchMode=yes',
        '-o', 'ConnectTimeout=15',
        '-o', 'ControlMaster=auto',
        '-o', 'ControlPersist=300',
        '-o', f'ControlPath={config.REMOTE_SSH_CONTROL_PATH}',
    ])
    return [
        'rsync',
        '-az',
        '--partial',
        '--human-readable',
        '-e',
        ssh_transport,
    ]


def sync_project_source(host, remote_root):
    """同步小体积源码，保留远端数据、权重、引擎和隔离环境。"""
    remote_parent = os.path.dirname(remote_root.rstrip('/'))
    ssh_run(host, quote_join(['mkdir', '-p', remote_parent, remote_root]))
    exclusions = [
        '.git/',
        '__pycache__/',
        'swanlog/',
        'reinforcement_learning/data/',
        'reinforcement_learning/logs/',
        'reinforcement_learning/checkpoints/',
        'supervised_learning/checkpoints/',
        '*.so',
        '*.dll',
        '*.pth',
        '*.engine',
        '*.onnx',
    ]
    command = rsync_base()
    for pattern in exclusions:
        command.extend(['--exclude', pattern])
    command.extend([f'{PROJECT_ROOT}/', f'{host}:{remote_root}/'])
    run_command(command)


def remote_text(host, command):
    return ssh_run(host, command, capture_output=True).stdout.strip()


def remote_file_exists(host, path):
    command = f"test -f {shlex.quote(path)} && printf yes || true"
    return remote_text(host, command) == 'yes'


def remote_sha256(host, path):
    if not remote_file_exists(host, path):
        return None
    output = remote_text(host, quote_join(['sha256sum', path]))
    return output.split()[0] if output else None


def remote_csv_rows(host, path):
    """在远端用 Python 校验表头并计数，避免 wc 把空行算成棋局。"""
    script = (
        "import pathlib,sys; p=pathlib.Path(sys.argv[1]); "
        "lines=p.read_text(encoding='utf-8',errors='replace').splitlines() if p.exists() else []; "
        f"assert not lines or lines[0].strip()=={CSV_HEADER!r}, 'bad csv header'; "
        "print(sum(bool(line.strip()) for line in lines[1:]))"
    )
    output = remote_text(host, quote_join(['python3', '-c', script, path]))
    return int(output or 0)


def atomic_push(host, local_path, remote_path):
    remote_temp = f'{remote_path}.uploading'
    ssh_run(host, quote_join(['mkdir', '-p', os.path.dirname(remote_path)]))
    run_command([*rsync_base(), local_path, f'{host}:{remote_temp}'])
    ssh_run(host, quote_join(['mv', '-f', remote_temp, remote_path]))


def atomic_pull(host, remote_path, local_path):
    os.makedirs(os.path.dirname(os.path.abspath(local_path)), exist_ok=True)
    local_temp = f'{local_path}.downloading'
    try:
        run_command([*rsync_base(), f'{host}:{remote_path}', local_temp])
        validate_csv(local_temp)
        os.replace(local_temp, local_path)
    finally:
        if os.path.exists(local_temp):
            os.remove(local_temp)


def ensure_remote_mcts(host, remote_root, source_digest, build_cpuset):
    """源码变化时才重编译 C++ MCTS，避免远端继续加载旧动态库。"""
    core_dir = os.path.join(remote_root, 'reinforcement_learning', 'core')
    marker_path = os.path.join(core_dir, 'libmcts.sources.sha256')
    library_path = os.path.join(core_dir, 'libmcts.so')
    marker = remote_text(
        host,
        f"test -f {shlex.quote(marker_path)} && cat {shlex.quote(marker_path)} || true",
    )
    if marker == source_digest and remote_file_exists(host, library_path):
        return False

    marker_temp = f'{marker_path}.tmp'
    command = (
        f"cd {shlex.quote(core_dir)} && "
        f"taskset -c {shlex.quote(build_cpuset)} "
        f"python3 compile_mcts.py && "
        f"printf %s {shlex.quote(source_digest)} > {shlex.quote(marker_temp)} && "
        f"mv -f {shlex.quote(marker_temp)} {shlex.quote(marker_path)}"
    )
    ssh_run(host, command)
    return True


def mcts_source_digest():
    digest = hashlib.sha256()
    core_dir = os.path.join(RL_DIR, 'core')
    for name in ('mcts_engine.cpp', 'c6_logic.h', 'compile_mcts.py'):
        path = os.path.join(core_dir, name)
        digest.update(name.encode('utf-8'))
        with open(path, 'rb') as file:
            digest.update(file.read())
    return digest.hexdigest()


def remote_environment():
    """远端参数固定为实测吞吐点，搜索预算与本地主循环保持一致。"""
    return {
        'PYTHONUNBUFFERED': '1',
        'PYTHONNOUSERSITE': '1',
        'NEBULA_SELFPLAY_GPUS': config.REMOTE_SELFPLAY_GPUS,
        'NEBULA_NUM_WORKERS': str(config.REMOTE_NUM_WORKERS),
        'NEBULA_MCTS_THREADS': str(config.REMOTE_MCTS_THREADS),
        'NEBULA_MCTS_BATCH_SIZE': str(config.REMOTE_MCTS_BATCH_SIZE),
        'NEBULA_MCTS_CONCURRENT_GAMES': str(config.REMOTE_MCTS_CONCURRENT_GAMES),
        'NEBULA_MCTS_EVAL_CACHE_SIZE': str(config.MCTS_EVAL_CACHE_SIZE),
        'NEBULA_SIMULATIONS': str(config.SIMULATIONS),
        'NEBULA_SIMULATIONS_BLACK': str(config.SIMULATIONS_BLACK),
        'NEBULA_SIMULATIONS_WHITE': str(config.SIMULATIONS_WHITE),
        'NEBULA_DYNAMIC_EARLY_STOP': '0',
        'NEBULA_TRT_PRECISION': config.TRT_PRECISION,
        'NEBULA_TRT_FUSED_SELFPLAY': '1' if config.TRT_FUSED_SELFPLAY else '0',
        'NEBULA_TRT_CUSTOM_ATTENTION': '1' if config.TRT_CUSTOM_ATTENTION else '0',
        'NEBULA_CUDA_GRAPH': os.environ.get('NEBULA_CUDA_GRAPH', '1'),
    }


def environment_parts(values):
    return ['env', *(f'{key}={value}' for key, value in values.items())]


def ensure_remote_engine(host, remote_root, local_weight):
    """仅当 best.pth 改变时原子替换 H20 专用 TensorRT 引擎。"""
    checkpoint_dir = os.path.join(remote_root, 'reinforcement_learning', 'checkpoints')
    remote_weight = os.path.join(checkpoint_dir, 'best.pth')
    remote_engine = os.path.join(checkpoint_dir, 'current_model.engine')
    marker_path = os.path.join(checkpoint_dir, 'current_model.sha256')
    local_digest = sha256_file(local_weight)
    marker = remote_text(
        host,
        f"test -f {shlex.quote(marker_path)} && cat {shlex.quote(marker_path)} || true",
    )
    if marker == local_digest and remote_file_exists(host, remote_engine):
        print('远端 H20 引擎已对应当前 incumbent，跳过重建。', flush=True)
        return remote_engine, False

    atomic_push(host, local_weight, remote_weight)
    onnx_temp = os.path.join(checkpoint_dir, 'current_model.building.onnx')
    engine_temp = os.path.join(checkpoint_dir, 'current_model.building.engine')
    marker_temp = f'{marker_path}.tmp'
    env = remote_environment()
    env.update({
        'CUDA_VISIBLE_DEVICES': config.REMOTE_BUILD_GPU,
        'NEBULA_BUILD_GPU': config.REMOTE_BUILD_GPU,
    })
    export_script = os.path.join(remote_root, 'reinforcement_learning', 'pipeline', 'export_onnx.py')
    build_script = os.path.join(remote_root, 'reinforcement_learning', 'pipeline', 'build_engine.py')
    lock_path = os.path.join(checkpoint_dir, 'remote_selfplay.lock')
    inner = (
        f"set -e; rm -f {shlex.quote(onnx_temp)} {shlex.quote(engine_temp)}; "
        f"{quote_join(environment_parts(env))} {shlex.quote(config.REMOTE_PYTHON)} "
        f"{shlex.quote(export_script)} {shlex.quote(remote_weight)} {shlex.quote(onnx_temp)}; "
        f"{quote_join(environment_parts(env))} {shlex.quote(config.REMOTE_PYTHON)} "
        f"{shlex.quote(build_script)} {shlex.quote(onnx_temp)} {shlex.quote(engine_temp)}; "
        f"test -s {shlex.quote(engine_temp)}; "
        f"mv -f {shlex.quote(engine_temp)} {shlex.quote(remote_engine)}; "
        f"printf %s {shlex.quote(local_digest)} > {shlex.quote(marker_temp)}; "
        f"mv -f {shlex.quote(marker_temp)} {shlex.quote(marker_path)}; "
        f"rm -f {shlex.quote(onnx_temp)}"
    )
    command = (
        f"mkdir -p {shlex.quote(checkpoint_dir)} && "
        f"flock -w 30 {shlex.quote(lock_path)} "
        f"taskset -c {shlex.quote(config.REMOTE_BUILD_CPUSET)} "
        f"bash -lc {shlex.quote(inner)}"
    )
    ssh_run(host, command)
    return remote_engine, True


def reconcile_generation_file(host, local_path, remote_path):
    """选择局数更多的一侧续跑；同样多但不一致时以本地主循环为准。"""
    local_count = csv_rows(local_path) if os.path.exists(local_path) else 0
    remote_count = remote_csv_rows(host, remote_path)
    if remote_count > local_count:
        print(f'恢复远端已完成的 {remote_count} 局棋谱。', flush=True)
        atomic_pull(host, remote_path, local_path)
        return remote_count
    if local_count > remote_count:
        print(f'把本地已有的 {local_count} 局棋谱同步到远端。', flush=True)
        atomic_push(host, local_path, remote_path)
        return local_count
    if local_count == 0:
        return 0

    local_digest = sha256_file(local_path)
    if remote_sha256(host, remote_path) != local_digest:
        print('本地与远端棋谱行数相同但内容不同，以本地主循环副本为准。', flush=True)
        atomic_push(host, local_path, remote_path)
    return local_count


def generate_remote_games(host, remote_root, engine_path, remote_out, total, seed):
    current = remote_csv_rows(host, remote_out)
    remaining = total - current
    if remaining <= 0:
        return 0

    env = remote_environment()
    script = os.path.join(remote_root, 'reinforcement_learning', 'pipeline', 'generate.py')
    lock_path = os.path.join(
        remote_root,
        'reinforcement_learning',
        'checkpoints',
        'remote_selfplay.lock',
    )
    command_parts = [
        'taskset', '-c', config.REMOTE_CPUSET,
        *environment_parts(env),
        config.REMOTE_PYTHON,
        script,
        '--out', remote_out,
        '--total', str(remaining),
        '--engine', engine_path,
        '--seed', str(seed),
    ]
    remote_command = (
        f"flock -w 30 {shlex.quote(lock_path)} {quote_join(command_parts)}"
    )
    ssh_run(host, remote_command)
    final_count = remote_csv_rows(host, remote_out)
    if final_count != total:
        raise RuntimeError(f'远端只生成 {final_count}/{total} 局')
    return remaining


def run_remote_generation(generation, local_out, target_total, seed, local_weight):
    if not config.REMOTE_SELFPLAY_HOST:
        raise ValueError('远端自对弈已启用，但 NEBULA_REMOTE_SELFPLAY_HOST 为空')
    if config.REMOTE_MCTS_BATCH_SIZE != config.MCTS_BATCH_SIZE:
        raise ValueError(
            '远端 MCTS batch 必须与当前引擎构建 batch 一致；'
            '请统一 REMOTE_MCTS_BATCH_SIZE 和 MCTS_BATCH_SIZE'
        )
    if config.DYNAMIC_EARLY_STOP:
        raise ValueError('远端生产自对弈禁止提前熔断，必须完整执行搜索预算')

    host = config.REMOTE_SELFPLAY_HOST
    remote_root = config.REMOTE_PROJECT_DIR.rstrip('/')
    started_at = time.perf_counter()
    if config.REMOTE_SYNC_CODE:
        sync_project_source(host, remote_root)
    ensure_remote_mcts(host, remote_root, mcts_source_digest(), config.REMOTE_BUILD_CPUSET)
    engine_path, rebuilt = ensure_remote_engine(host, remote_root, local_weight)

    remote_out = os.path.join(
        remote_root,
        'reinforcement_learning',
        'data',
        'remote_raw',
        f'gen_{generation}.csv',
    )
    ssh_run(host, quote_join(['mkdir', '-p', os.path.dirname(remote_out)]))
    reconcile_generation_file(host, local_out, remote_out)
    generated = generate_remote_games(
        host,
        remote_root,
        engine_path,
        remote_out,
        target_total,
        seed,
    )
    atomic_pull(host, remote_out, local_out)
    validate_csv(local_out, target_total)
    elapsed = time.perf_counter() - started_at
    print(
        f'远端第 {generation} 代自对弈完成：新生成 {generated} 局，'
        f'引擎重建={rebuilt}，端到端耗时 {elapsed:.1f}s',
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description='Nebula Zero 远端自对弈执行器')
    parser.add_argument('--generation', type=int, required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--target-total', type=int, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--weight', required=True)
    args = parser.parse_args()
    run_remote_generation(
        generation=args.generation,
        local_out=os.path.abspath(args.out),
        target_total=args.target_total,
        seed=args.seed,
        local_weight=os.path.abspath(args.weight),
    )


if __name__ == '__main__':
    main()
