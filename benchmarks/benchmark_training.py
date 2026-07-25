"""只使用指定 GPU，测量 C6TransNet BF16 前向、反向和优化器吞吐。"""

import argparse
import json
import os
import statistics
import sys
import time


parser = argparse.ArgumentParser()
parser.add_argument('--gpus', default='6')
parser.add_argument('--batch-size', type=int, default=386)
parser.add_argument('--warmup', type=int, default=2)
parser.add_argument('--rounds', type=int, default=5)
args = parser.parse_args()

# 必须在导入 torch 之前限制设备，保证不会意外初始化其他人的 GPU。
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
RL_DIR = os.path.join(ROOT, 'reinforcement_learning')
sys.path.insert(0, RL_DIR)

from core.model import C6TransNet


def main():
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA 不可用')
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError('当前 GPU 不支持 BF16')

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device('cuda:0')
    model = C6TransNet(input_planes=17).to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.train()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-5,
        weight_decay=1e-4,
        fused=True,
    )
    features = torch.randn(
        args.batch_size,
        17,
        19,
        19,
        device=device,
    )
    policy_target = torch.randint(0, 361, (args.batch_size,), device=device)
    value_target = torch.empty(args.batch_size, 1, device=device).uniform_(-1, 1)

    def training_step():
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            policy, _, value = model(features)
            loss = F.cross_entropy(policy, policy_target) + F.mse_loss(
                value,
                value_target,
            )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        return loss

    for _ in range(args.warmup):
        training_step()
    torch.cuda.synchronize()

    durations = []
    losses = []
    for _ in range(args.rounds):
        started_at = time.perf_counter()
        loss = training_step()
        torch.cuda.synchronize()
        durations.append(time.perf_counter() - started_at)
        losses.append(float(loss.detach()))

    median_seconds = statistics.median(durations)
    peak_memory = []
    for device_index in range(torch.cuda.device_count()):
        peak_memory.append(
            round(torch.cuda.max_memory_allocated(device_index) / 2**30, 3)
        )
    print(
        json.dumps(
            {
                'physical_gpus': args.gpus.split(','),
                'visible_gpu_count': torch.cuda.device_count(),
                'precision': 'bf16',
                'batch_size': args.batch_size,
                'step_seconds': [round(value, 4) for value in durations],
                'median_step_seconds': round(median_seconds, 4),
                'samples_per_second': round(args.batch_size / median_seconds),
                'peak_memory_gib': peak_memory,
                'final_loss': round(losses[-1], 6),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == '__main__':
    main()
