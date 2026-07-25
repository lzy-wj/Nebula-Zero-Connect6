"""NebulaNet V3 的双卡 DDP 预训练入口。"""

import argparse
import json
import math
import os
import random
import shutil
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Sampler


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
sys.path.insert(0, SCRIPT_DIR)

from dataset import SelfPlayPositionDataset, load_game_records, seed_data_worker
from model import NebulaNetV3
from model_fast import FastC6NetV4

RL_DIR = os.path.join(PROJECT_ROOT, 'reinforcement_learning')
if RL_DIR not in sys.path:
    sys.path.insert(0, RL_DIR)
from core.model import C6TransNet


class WeightedDistributedSampler(Sampler):
    """各 rank 共享一次加权抽样序列，再按步长切片，保证样本数完全相同。"""

    def __init__(self, weights, global_num_samples, rank, world_size, seed):
        self.weights = weights
        self.rank = rank
        self.world_size = world_size
        self.seed = seed
        self.epoch = 0
        self.samples_per_rank = math.ceil(global_num_samples / world_size)
        self.total_size = self.samples_per_rank * world_size

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        indices = torch.multinomial(
            self.weights,
            self.total_size,
            replacement=True,
            generator=generator,
        )
        return iter(indices[self.rank:self.total_size:self.world_size].tolist())

    def __len__(self):
        return self.samples_per_rank


class DistributedShardSampler(Sampler):
    """验证集不补齐、不重复，只把自然索引分片给各 rank。"""

    def __init__(self, dataset_size, rank, world_size):
        self.indices = range(rank, dataset_size, world_size)

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


def parse_args():
    default_data = os.path.abspath(
        os.path.join(PROJECT_ROOT, '..', 'datasets', 'selfplay_data_v1', 'selfplay_data')
    )
    default_output = os.path.join(SCRIPT_DIR, 'output', 'network_v3_pretrain_a800')
    parser = argparse.ArgumentParser(description='NebulaNet V3 双卡预训练')
    parser.add_argument('--data-root', default=os.environ.get('NEBULA_V3_DATA', default_data))
    parser.add_argument('--output-dir', default=default_output)
    parser.add_argument('--run-name', default='network_v3_pretrain_a800')
    parser.add_argument('--project', default='Nebula-zero-one')
    parser.add_argument(
        '--swanlab-mode',
        choices=['online', 'offline', 'local', 'disabled'],
        default='online',
    )
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--batch-size', type=int, default=256, help='每张 GPU 的 batch')
    parser.add_argument('--workers', type=int, default=8, help='每个 DDP rank 的加载进程数')
    parser.add_argument('--samples-per-epoch', type=int, default=0, help='0 表示训练集局数')
    parser.add_argument('--validation-start-generation', type=int, default=796)
    parser.add_argument('--validation-positions', type=int, default=2)
    parser.add_argument('--learning-rate', type=float, default=3e-4)
    parser.add_argument('--min-lr-ratio', type=float, default=0.05)
    parser.add_argument('--warmup-ratio', type=float, default=0.05)
    parser.add_argument('--weight-decay', type=float, default=0.05)
    parser.add_argument('--value-loss-weight', type=float, default=0.5)
    parser.add_argument('--grad-clip', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--log-interval', type=int, default=50)
    parser.add_argument('--resume', default=None)
    parser.add_argument('--limit-files', type=int, default=None, help='仅供冒烟测试')
    parser.add_argument('--limit-games', type=int, default=None, help='仅供冒烟测试')
    parser.add_argument('--channels', type=int, default=320)
    parser.add_argument('--conv-depth', type=int, default=8)
    parser.add_argument('--transformer-depth', type=int, default=4)
    parser.add_argument('--num-heads', type=int, default=10)
    parser.add_argument('--drop-path-rate', type=float, default=0.08)
    parser.add_argument(
        '--architecture',
        choices=['nebula_v3', 'fast_v4'],
        default='nebula_v3',
    )
    parser.add_argument(
        '--teacher-checkpoint',
        default=None,
        help='Fast V4 的 V2 初始化与策略蒸馏教师 checkpoint',
    )
    parser.add_argument(
        '--source-block-indices',
        default='0,1,2,3,5',
        help='Fast V4 从 V2 保留的注意力层编号',
    )
    parser.add_argument(
        '--bridge-attention-source',
        type=int,
        default=None,
        help='仅恢复指定 V2 层的注意力半层，不恢复该层 FFN',
    )
    parser.add_argument(
        '--distill-policy-weight',
        type=float,
        default=0.0,
        help='教师策略交叉熵在策略目标中的权重',
    )
    parser.add_argument(
        '--distill-hard-weight',
        type=float,
        default=0.0,
        help='教师 Top-1 硬标签在策略目标中的权重',
    )
    parser.add_argument(
        '--feature-distill-weight',
        type=float,
        default=0.0,
        help='学生与教师最终棋盘特征对齐损失权重',
    )
    parser.add_argument(
        '--init-checkpoint',
        default=None,
        help='只加载候选模型权重并重置优化器，用于新阶段精修',
    )
    return parser.parse_args()


def distributed_context():
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    rank = int(os.environ.get('RANK', '0'))
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl')
    return rank, local_rank, world_size


def is_main_process(rank):
    return rank == 0


def seed_everything(seed, rank):
    effective_seed = seed + rank
    random.seed(effective_seed)
    np.random.seed(effective_seed)
    torch.manual_seed(effective_seed)
    torch.cuda.manual_seed_all(effective_seed)


def atomic_torch_save(payload, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    temporary_path = f'{path}.tmp'
    torch.save(payload, temporary_path)
    os.replace(temporary_path, path)


def atomic_json_dump(payload, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    temporary_path = f'{path}.tmp'
    with open(temporary_path, 'w', encoding='utf-8') as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)
    os.replace(temporary_path, path)


def strip_module_prefix(state_dict):
    return {
        key.removeprefix('module.'): value
        for key, value in state_dict.items()
    }


def checkpoint_state(checkpoint):
    state = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint))
    return strip_module_prefix(state)


def parse_block_indices(value):
    indices = tuple(int(item.strip()) for item in value.split(',') if item.strip())
    if not indices:
        raise ValueError('--source-block-indices 不能为空')
    return indices


def load_v2_teacher(checkpoint_path, device=None):
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    teacher = C6TransNet(input_planes=17)
    teacher.load_state_dict(checkpoint_state(checkpoint))
    teacher.eval().requires_grad_(False)
    if device is not None:
        teacher = teacher.to(device, memory_format=torch.channels_last)
    return teacher


def adapt_v2_features(features):
    """把 5 个真实平面映射回教师实际消费的三个旧平面。"""

    old_features = torch.zeros(
        (features.shape[0], 17, 19, 19),
        dtype=features.dtype,
        device=features.device,
    )
    old_features[:, 0] = features[:, 0]
    old_features[:, 1] = features[:, 1]
    old_features[:, 16] = features[:, 3]
    return old_features


def forward_v2_with_features(model, features):
    """执行 V2 教师并额外返回最终棋盘 token，不改变生产模型接口。"""

    outputs = model.relu(model.bn_in(model.conv_in(features)))
    outputs = model.res_stack(outputs)
    tokens = model.forward_transformer(outputs.flatten(2).transpose(1, 2))
    policy_logits = model.head_move1(tokens).squeeze(-1)
    value = model.value_head(tokens.mean(dim=1))
    return policy_logits, value, tokens


def cosine_warmup_lambda(step, total_steps, warmup_steps, min_ratio):
    if warmup_steps > 0 and step < warmup_steps:
        return max(1e-3, (step + 1) / warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(1.0, max(0.0, progress))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_ratio + (1.0 - min_ratio) * cosine


def batch_metrics(policy_logits, value_prediction, policy_target, value_target, features):
    occupied = ((features[:, 0] + features[:, 1]) > 0.5).flatten(1)
    float_logits = policy_logits.float()
    raw_probability = torch.softmax(float_logits, dim=1)
    illegal_mass = (raw_probability * occupied.float()).sum(dim=1)
    masked_logits = float_logits.masked_fill(occupied, -10_000.0)
    log_probability = torch.log_softmax(masked_logits, dim=1)

    policy_loss_per_sample = -(policy_target.float() * log_probability).sum(dim=1)
    value_prediction = value_prediction.float().flatten()
    value_target = value_target.float().flatten()
    value_loss_per_sample = F.smooth_l1_loss(
        value_prediction,
        value_target,
        beta=0.5,
        reduction='none',
    )

    target_move = policy_target.argmax(dim=1)
    top1 = masked_logits.argmax(dim=1).eq(target_move).float()
    top5 = masked_logits.topk(5, dim=1).indices.eq(target_move[:, None]).any(dim=1).float()
    value_mae = (value_prediction - value_target).abs()
    return {
        'policy_loss': policy_loss_per_sample,
        'value_loss': value_loss_per_sample,
        'top1': top1,
        'top5': top5,
        'value_mae': value_mae,
        'illegal_mass': illegal_mass,
    }


def reduce_epoch_sums(metric_sums, device, world_size):
    names = sorted(metric_sums)
    values = torch.tensor(
        [metric_sums[name] for name in names],
        dtype=torch.float64,
        device=device,
    )
    if world_size > 1:
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
    reduced = {name: float(value) for name, value in zip(names, values.cpu())}
    count = max(1.0, reduced.pop('count'))
    return {name: value / count for name, value in reduced.items()}


def run_epoch(
    model,
    loader,
    device,
    world_size,
    value_loss_weight,
    optimizer=None,
    scheduler=None,
    grad_clip=1.0,
    log_interval=50,
    global_step=0,
    swan_active=False,
    rank=0,
    teacher_model=None,
    distill_policy_weight=0.0,
    distill_hard_weight=0.0,
    feature_distill_weight=0.0,
):
    training = optimizer is not None
    model.train(training)
    if teacher_model is not None:
        teacher_model.eval()
    metric_sums = {
        'count': 0.0,
        'loss': 0.0,
        'policy_loss': 0.0,
        'value_loss': 0.0,
        'top1': 0.0,
        'top5': 0.0,
        'value_mae': 0.0,
        'illegal_mass': 0.0,
        'distill_policy_loss': 0.0,
        'distill_hard_loss': 0.0,
        'feature_distill_loss': 0.0,
    }
    started_at = time.perf_counter()

    for batch_index, (features, policy_target, value_target) in enumerate(loader, start=1):
        features = features.to(
            device,
            dtype=torch.float32,
            non_blocking=True,
            memory_format=torch.channels_last,
        )
        policy_target = policy_target.to(device, non_blocking=True)
        value_target = value_target.to(device, non_blocking=True)

        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training), torch.autocast(
            device_type='cuda',
            dtype=torch.bfloat16,
        ):
            needs_teacher_features = (
                training
                and teacher_model is not None
                and feature_distill_weight > 0
            )
            if needs_teacher_features:
                policy_logits, _, value_prediction, student_features = model(
                    features,
                    return_features=True,
                )
            else:
                policy_logits, _, value_prediction = model(features)
                student_features = None
            metrics = batch_metrics(
                policy_logits,
                value_prediction,
                policy_target,
                value_target,
                features,
            )
            policy_loss = metrics['policy_loss'].mean()
            distill_policy_loss_per_sample = torch.zeros_like(
                metrics['policy_loss']
            )
            distill_hard_loss_per_sample = torch.zeros_like(metrics['policy_loss'])
            feature_distill_loss_per_sample = torch.zeros_like(metrics['policy_loss'])
            needs_teacher = (
                training
                and teacher_model is not None
                and (
                    distill_policy_weight > 0
                    or distill_hard_weight > 0
                    or feature_distill_weight > 0
                )
            )
            if needs_teacher:
                # 教师只提供平滑策略分布；价值继续由真实胜负监督，避免继承
                # V2 已知较弱的价值误差。
                with torch.no_grad():
                    teacher_logits, _, teacher_features = forward_v2_with_features(
                        teacher_model,
                        adapt_v2_features(features)
                    )
                    occupied = (
                        (features[:, 0] + features[:, 1]) > 0.5
                    ).flatten(1)
                    teacher_masked_logits = teacher_logits.float().masked_fill(
                        occupied,
                        -10_000.0,
                    )
                    teacher_probability = torch.softmax(
                        teacher_masked_logits,
                        dim=1,
                    )
                student_log_probability = torch.log_softmax(
                    policy_logits.float().masked_fill(occupied, -10_000.0),
                    dim=1,
                )
                distill_policy_loss_per_sample = -(
                    teacher_probability * student_log_probability
                ).sum(dim=1)
                distill_hard_loss_per_sample = -student_log_probability.gather(
                    1,
                    teacher_masked_logits.argmax(dim=1, keepdim=True),
                ).squeeze(1)
                if needs_teacher_features:
                    student_normalized = F.layer_norm(
                        student_features.float(),
                        (student_features.shape[-1],),
                    )
                    teacher_normalized = F.layer_norm(
                        teacher_features.float(),
                        (teacher_features.shape[-1],),
                    )
                    feature_distill_loss_per_sample = (
                        student_normalized - teacher_normalized
                    ).square().mean(dim=(1, 2))
            metrics['distill_policy_loss'] = distill_policy_loss_per_sample
            metrics['distill_hard_loss'] = distill_hard_loss_per_sample
            metrics['feature_distill_loss'] = feature_distill_loss_per_sample
            distill_policy_loss = distill_policy_loss_per_sample.mean()
            distill_hard_loss = distill_hard_loss_per_sample.mean()
            feature_distill_loss = feature_distill_loss_per_sample.mean()
            optimized_policy_loss = (
                (1.0 - distill_policy_weight - distill_hard_weight) * policy_loss
                + distill_policy_weight * distill_policy_loss
                + distill_hard_weight * distill_hard_loss
                if needs_teacher
                else policy_loss
            )
            value_loss = metrics['value_loss'].mean()
            loss = (
                optimized_policy_loss
                + value_loss_weight * value_loss
                + feature_distill_weight * feature_distill_loss
            )

        if training:
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            scheduler.step()
            global_step += 1
        else:
            grad_norm = torch.zeros((), device=device)

        batch_size = features.shape[0]
        metric_sums['count'] += batch_size
        metric_sums['loss'] += float(loss.detach()) * batch_size
        for name, values in metrics.items():
            metric_sums[name] += float(values.detach().sum())

        if training and rank == 0 and batch_index % log_interval == 0:
            elapsed = time.perf_counter() - started_at
            samples_per_second = metric_sums['count'] * world_size / max(elapsed, 1e-9)
            payload = {
                'train/step_loss': float(loss.detach()),
                'train/step_policy_loss': float(policy_loss.detach()),
                'train/step_distill_policy_loss': float(
                    distill_policy_loss.detach()
                ),
                'train/step_distill_hard_loss': float(
                    distill_hard_loss.detach()
                ),
                'train/step_feature_distill_loss': float(
                    feature_distill_loss.detach()
                ),
                'train/step_value_loss': float(value_loss.detach()),
                'train/grad_norm': float(grad_norm.detach()),
                'train/learning_rate': scheduler.get_last_lr()[0],
                'performance/samples_per_second': samples_per_second,
            }
            print(
                f"step={global_step} loss={payload['train/step_loss']:.4f} "
                f"lr={payload['train/learning_rate']:.2e} "
                f"throughput={samples_per_second:.0f} samples/s",
                flush=True,
            )
            if swan_active:
                import swanlab

                swanlab.log(payload, step=global_step)

    elapsed = time.perf_counter() - started_at
    reduced = reduce_epoch_sums(metric_sums, device, world_size)
    reduced['seconds'] = elapsed
    reduced['samples_per_second'] = (
        metric_sums['count'] * world_size / max(elapsed, 1e-9)
    )
    return reduced, global_step


def initialize_swanlab(args, model, train_size, validation_size, global_batch):
    if args.swanlab_mode == 'disabled':
        return False
    import swanlab

    run_id_path = os.path.join(args.output_dir, 'swanlab_run_id.txt')
    run_id = None
    try:
        with open(run_id_path, 'r', encoding='utf-8') as file:
            run_id = file.read().strip() or None
    except FileNotFoundError:
        pass

    init_kwargs = {
        'reinit': True,
        'project': args.project,
        'name': args.run_name,
        'description': 'Connect6 候选网络：合法点训练、固定最近代验证与 BF16 双卡预训练',
        'mode': args.swanlab_mode,
        'config': {
            'model': type(model).__name__,
            'parameters': model.parameter_count,
            'model_config': model.model_config,
            'train_games': train_size,
            'validation_games': validation_size,
            'validation_start_generation': args.validation_start_generation,
            'epochs': args.epochs,
            'global_batch_size': global_batch,
            'learning_rate': args.learning_rate,
            'weight_decay': args.weight_decay,
            'value_loss_weight': args.value_loss_weight,
            'teacher_checkpoint': args.teacher_checkpoint,
            'distill_policy_weight': args.distill_policy_weight,
            'distill_hard_weight': args.distill_hard_weight,
            'feature_distill_weight': args.feature_distill_weight,
            'init_checkpoint': args.init_checkpoint,
            'precision': 'bf16',
            'seed': args.seed,
        },
    }
    if run_id:
        init_kwargs.update({'id': run_id, 'resume': 'allow'})
    run = swanlab.init(**init_kwargs)
    current_id = getattr(run, 'id', None)
    if current_id and not run_id:
        os.makedirs(args.output_dir, exist_ok=True)
        temporary_path = f'{run_id_path}.tmp'
        with open(temporary_path, 'w', encoding='utf-8') as file:
            file.write(current_id)
        os.replace(temporary_path, run_id_path)
    return True


def main():
    args = parse_args()
    if args.resume and args.init_checkpoint:
        raise ValueError('--resume 与 --init-checkpoint 不能同时使用')
    if args.distill_policy_weight < 0 or args.distill_hard_weight < 0:
        raise ValueError('蒸馏权重不能为负数')
    if args.distill_policy_weight + args.distill_hard_weight > 1.0:
        raise ValueError('软/硬策略蒸馏权重之和不能超过 1')
    if args.feature_distill_weight < 0:
        raise ValueError('--feature-distill-weight 不能为负数')
    if args.feature_distill_weight > 0 and args.architecture != 'fast_v4':
        raise ValueError('最终特征蒸馏目前只支持 Fast V4')
    rank, local_rank, world_size = distributed_context()
    main_process = is_main_process(rank)
    seed_everything(args.seed, rank)
    device = torch.device('cuda', local_rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision('high')

    training_records, validation_records, data_statistics = load_game_records(
        args.data_root,
        validation_start_generation=args.validation_start_generation,
        max_files=args.limit_files,
        max_games=args.limit_games,
    )
    if not training_records and validation_records:
        # 极小冒烟集可能全部落在验证代，仍需切出一小部分训练样本。
        split = max(1, len(validation_records) // 10)
        training_records = validation_records[split:]
        validation_records = validation_records[:split]
    elif not validation_records and training_records:
        # 反之固定留最近读取到的 10%，保证训练入口始终有独立验证集。
        split = max(1, len(training_records) // 10)
        validation_records = training_records[:split]
        training_records = training_records[split:]
    if main_process:
        print(
            f'数据加载完成：训练 {len(training_records):,} 局，'
            f'验证 {len(validation_records):,} 局，统计={data_statistics}',
            flush=True,
        )

    training_dataset = SelfPlayPositionDataset(training_records, training=True)
    validation_dataset = SelfPlayPositionDataset(
        validation_records,
        training=False,
        positions_per_game=args.validation_positions,
    )
    global_samples = args.samples_per_epoch or len(training_dataset)
    training_sampler = WeightedDistributedSampler(
        training_dataset.record_weights(),
        global_samples,
        rank,
        world_size,
        args.seed,
    )
    validation_sampler = DistributedShardSampler(
        len(validation_dataset),
        rank,
        world_size,
    )
    loader_kwargs = {
        'batch_size': args.batch_size,
        'num_workers': args.workers,
        'pin_memory': True,
        'worker_init_fn': seed_data_worker,
    }
    if args.workers > 0:
        loader_kwargs.update({'persistent_workers': True, 'prefetch_factor': 3})
    training_loader = DataLoader(
        training_dataset,
        sampler=training_sampler,
        drop_last=True,
        **loader_kwargs,
    )
    validation_loader = DataLoader(
        validation_dataset,
        sampler=validation_sampler,
        drop_last=False,
        **loader_kwargs,
    )

    teacher_model = None
    if args.architecture == 'fast_v4':
        source_blocks = parse_block_indices(args.source_block_indices)
        model = FastC6NetV4(
            source_block_indices=source_blocks,
            bridge_attention_source=args.bridge_attention_source,
        )
        if not args.resume:
            if args.init_checkpoint:
                initial = torch.load(
                    args.init_checkpoint,
                    map_location='cpu',
                    weights_only=False,
                )
                model.load_state_dict(checkpoint_state(initial))
            else:
                if not args.teacher_checkpoint:
                    raise ValueError('Fast V4 首次训练必须提供 --teacher-checkpoint')
                source_model = load_v2_teacher(args.teacher_checkpoint)
                model.initialize_from_v2(source_model)
                del source_model
        if (
            args.distill_policy_weight > 0
            or args.distill_hard_weight > 0
            or args.feature_distill_weight > 0
        ):
            if not args.teacher_checkpoint:
                raise ValueError('启用策略蒸馏时必须提供 --teacher-checkpoint')
            teacher_model = load_v2_teacher(args.teacher_checkpoint, device=device)
    else:
        model = NebulaNetV3(
            channels=args.channels,
            conv_depth=args.conv_depth,
            transformer_depth=args.transformer_depth,
            num_heads=args.num_heads,
            drop_path_rate=args.drop_path_rate,
        )
        if args.init_checkpoint and not args.resume:
            initial = torch.load(
                args.init_checkpoint,
                map_location='cpu',
                weights_only=False,
            )
            model.load_state_dict(checkpoint_state(initial))
    model = model.to(device, memory_format=torch.channels_last)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        fused=True,
    )
    total_steps = max(1, len(training_loader) * args.epochs)
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: cosine_warmup_lambda(
            step,
            total_steps,
            warmup_steps,
            args.min_lr_ratio,
        ),
    )

    start_epoch = 0
    global_step = 0
    best_validation_loss = float('inf')
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(strip_module_prefix(checkpoint['model_state_dict']))
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = int(checkpoint['epoch']) + 1
        global_step = int(checkpoint.get('global_step', 0))
        best_validation_loss = float(checkpoint.get('best_validation_loss', float('inf')))

    if world_size > 1:
        model = DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
            gradient_as_bucket_view=True,
            static_graph=True,
        )
    raw_model = model.module if isinstance(model, DistributedDataParallel) else model
    if main_process:
        print(
            f'模型参数量 {raw_model.parameter_count:,} | world_size={world_size} | '
            f'全局 batch={args.batch_size * world_size}',
            flush=True,
        )

    swan_active = False
    finish_state = 'crashed'
    if main_process:
        swan_active = initialize_swanlab(
            args,
            raw_model,
            len(training_records),
            len(validation_records),
            args.batch_size * world_size,
        )

    try:
        for epoch in range(start_epoch, args.epochs):
            training_sampler.set_epoch(epoch)
            train_metrics, global_step = run_epoch(
                model,
                training_loader,
                device,
                world_size,
                args.value_loss_weight,
                optimizer=optimizer,
                scheduler=scheduler,
                grad_clip=args.grad_clip,
                log_interval=args.log_interval,
                global_step=global_step,
                swan_active=swan_active,
                rank=rank,
                teacher_model=teacher_model,
                distill_policy_weight=args.distill_policy_weight,
                distill_hard_weight=args.distill_hard_weight,
                feature_distill_weight=args.feature_distill_weight,
            )
            validation_metrics, _ = run_epoch(
                model,
                validation_loader,
                device,
                world_size,
                args.value_loss_weight,
                rank=rank,
            )
            validation_score = validation_metrics['loss']

            if main_process:
                print(
                    f"epoch={epoch + 1}/{args.epochs} "
                    f"train_loss={train_metrics['loss']:.4f} "
                    f"val_loss={validation_score:.4f} "
                    f"val_top1={validation_metrics['top1']:.2%} "
                    f"val_value_mae={validation_metrics['value_mae']:.4f}",
                    flush=True,
                )
                epoch_payload = {
                    'epoch': epoch + 1,
                    **{f'train/{key}': value for key, value in train_metrics.items()},
                    **{f'validation/{key}': value for key, value in validation_metrics.items()},
                    'train/learning_rate': scheduler.get_last_lr()[0],
                }
                if swan_active:
                    import swanlab

                    swanlab.log(epoch_payload, step=global_step)

                checkpoint = {
                    'format_version': 1,
                    'architecture': type(raw_model).__name__,
                    'model_config': raw_model.model_config,
                    'model_state_dict': raw_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'epoch': epoch,
                    'global_step': global_step,
                    'best_validation_loss': min(best_validation_loss, validation_score),
                    'train_metrics': train_metrics,
                    'validation_metrics': validation_metrics,
                    'args': vars(args),
                }
                latest_path = os.path.join(args.output_dir, 'checkpoint_latest.pth')
                atomic_torch_save(checkpoint, latest_path)
                if validation_score < best_validation_loss:
                    best_validation_loss = validation_score
                    best_path = os.path.join(args.output_dir, 'checkpoint_best.pth')
                    temporary_path = f'{best_path}.tmp'
                    shutil.copy2(latest_path, temporary_path)
                    os.replace(temporary_path, best_path)
                atomic_json_dump(
                    {
                        'epoch': epoch + 1,
                        'global_step': global_step,
                        'best_validation_loss': best_validation_loss,
                        'train': train_metrics,
                        'validation': validation_metrics,
                    },
                    os.path.join(args.output_dir, 'metrics_latest.json'),
                )

        finish_state = 'success'
    finally:
        if main_process and swan_active:
            import swanlab

            swanlab.finish(state=finish_state, async_log_timeout=30)
        if world_size > 1:
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
