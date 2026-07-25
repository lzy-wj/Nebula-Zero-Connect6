"""在同一最近代验证集上比较旧生产网络与 NebulaNet V3。"""

import argparse
import json
import os
import sys
import time

import torch
from torch.utils.data import DataLoader


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
RL_DIR = os.path.join(PROJECT_ROOT, 'reinforcement_learning')
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, RL_DIR)

from dataset import SelfPlayPositionDataset, load_game_records, seed_data_worker
from model import NebulaNetV3
from model_fast import FastC6NetV4
from core.model import C6TransNet


def parse_args():
    default_data = os.path.abspath(
        os.path.join(PROJECT_ROOT, '..', 'datasets', 'selfplay_data_v1', 'selfplay_data')
    )
    parser = argparse.ArgumentParser(description='新旧网络离线验证对比')
    parser.add_argument('--new-checkpoint', required=True)
    parser.add_argument(
        '--old-checkpoint',
        default=os.path.join(RL_DIR, 'checkpoints', 'best.pth'),
    )
    parser.add_argument('--data-root', default=default_data)
    parser.add_argument('--validation-start-generation', type=int, default=796)
    parser.add_argument('--positions-per-game', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument(
        '--output',
        default=os.path.join(SCRIPT_DIR, 'output', 'checkpoint_comparison.json'),
    )
    return parser.parse_args()


def checkpoint_state(checkpoint):
    state = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint))
    return {key.removeprefix('module.'): value for key, value in state.items()}


def build_candidate_model(checkpoint):
    architecture = checkpoint.get('architecture', 'NebulaNetV3')
    model_classes = {
        'NebulaNetV3': NebulaNetV3,
        'FastC6NetV4': FastC6NetV4,
    }
    if architecture not in model_classes:
        raise ValueError(f'不支持评估的网络架构: {architecture}')
    model = model_classes[architecture](**checkpoint['model_config'])
    model.load_state_dict(checkpoint_state(checkpoint))
    return model


def adapt_old_features(features):
    """旧网络只真正消费己方、对方和颜色三个平面。"""
    old_features = torch.zeros(
        (features.shape[0], 17, 19, 19),
        dtype=features.dtype,
        device=features.device,
    )
    old_features[:, 0] = features[:, 0]
    old_features[:, 1] = features[:, 1]
    old_features[:, 16] = features[:, 3]
    return old_features


@torch.no_grad()
def evaluate(model, loader, device, old_model=False):
    model.eval()
    sums = {
        'count': 0.0,
        'policy_loss': 0.0,
        'value_loss': 0.0,
        'top1': 0.0,
        'top5': 0.0,
        'value_mae': 0.0,
        'illegal_mass': 0.0,
    }
    started_at = time.perf_counter()
    for features, policy_target, value_target in loader:
        features = features.to(
            device,
            non_blocking=True,
            memory_format=torch.channels_last,
        )
        policy_target = policy_target.to(device, non_blocking=True).float()
        value_target = value_target.to(device, non_blocking=True).float()
        occupied = ((features[:, 0] + features[:, 1]) > 0.5).flatten(1)
        model_input = adapt_old_features(features) if old_model else features

        with torch.autocast('cuda', dtype=torch.bfloat16):
            policy_logits, _, value_prediction = model(model_input)
        policy_logits = policy_logits.float()
        raw_probability = torch.softmax(policy_logits, dim=1)
        masked_logits = policy_logits.masked_fill(occupied, -10_000.0)
        log_probability = torch.log_softmax(masked_logits, dim=1)
        policy_loss = -(policy_target * log_probability).sum(dim=1)
        value_prediction = value_prediction.float().flatten()
        value_loss = torch.nn.functional.smooth_l1_loss(
            value_prediction,
            value_target,
            beta=0.5,
            reduction='none',
        )
        target_move = policy_target.argmax(dim=1)
        batch = features.shape[0]
        sums['count'] += batch
        sums['policy_loss'] += float(policy_loss.sum())
        sums['value_loss'] += float(value_loss.sum())
        sums['top1'] += float(masked_logits.argmax(1).eq(target_move).sum())
        sums['top5'] += float(
            masked_logits.topk(5, dim=1).indices.eq(target_move[:, None]).any(1).sum()
        )
        sums['value_mae'] += float((value_prediction - value_target).abs().sum())
        sums['illegal_mass'] += float((raw_probability * occupied.float()).sum())

    elapsed = time.perf_counter() - started_at
    count = sums.pop('count')
    metrics = {key: value / count for key, value in sums.items()}
    metrics['loss'] = metrics['policy_loss'] + 0.5 * metrics['value_loss']
    metrics['seconds'] = elapsed
    metrics['samples_per_second'] = count / max(elapsed, 1e-9)
    metrics['positions'] = int(count)
    return metrics


def main():
    args = parse_args()
    _, validation_records, statistics = load_game_records(
        args.data_root,
        validation_start_generation=args.validation_start_generation,
    )
    dataset = SelfPlayPositionDataset(
        validation_records,
        training=False,
        positions_per_game=args.positions_per_game,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
        worker_init_fn=seed_data_worker,
    )
    device = torch.device('cuda:0')

    new_checkpoint = torch.load(args.new_checkpoint, map_location='cpu', weights_only=False)
    new_model = build_candidate_model(new_checkpoint)
    new_model = new_model.to(device, memory_format=torch.channels_last)
    new_metrics = evaluate(new_model, loader, device, old_model=False)
    del new_model
    torch.cuda.empty_cache()

    old_checkpoint = torch.load(args.old_checkpoint, map_location='cpu', weights_only=False)
    old_model = C6TransNet(input_planes=17)
    old_model.load_state_dict(checkpoint_state(old_checkpoint))
    old_model = old_model.to(device, memory_format=torch.channels_last)
    old_metrics = evaluate(old_model, loader, device, old_model=True)

    result = {
        'candidate_architecture': new_checkpoint.get('architecture', 'NebulaNetV3'),
        'validation_games': len(validation_records),
        'positions_per_game': args.positions_per_game,
        'data_statistics': statistics,
        'candidate': new_metrics,
        'production_v2': old_metrics,
        'delta': {
            key: new_metrics[key] - old_metrics[key]
            for key in ('loss', 'policy_loss', 'value_loss', 'top1', 'top5', 'value_mae')
        },
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    temporary_path = f'{args.output}.tmp'
    with open(temporary_path, 'w', encoding='utf-8') as file:
        json.dump(result, file, indent=2, ensure_ascii=False)
    os.replace(temporary_path, args.output)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
