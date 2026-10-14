"""Check whether whole-position MoE routing is stable under D4 symmetry."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from experiments.nebula_v4.model import build_model
from experiments.pair_native.pair_replay import unpack_points
from experiments.pair_native.train import load_data, transform_board


def features_for(row, transform: int) -> np.ndarray:
    black = unpack_points(row["board_black"])
    white = unpack_points(row["board_white"])
    board = transform_board(black.astype(np.int8) - white.astype(np.int8), transform)
    player = int(row["player"])
    return np.stack(
        (
            board == player,
            board == -player,
            board == 0,
            np.full_like(board, player == 1, dtype=bool),
            np.zeros_like(board, dtype=bool),
        ),
        axis=0,
    ).astype(np.float32)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--samples", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda:0")
    arguments = parser.parse_args()

    checkpoint = torch.load(arguments.checkpoint, map_location="cpu", weights_only=False)
    model = build_model(checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.eval().to(arguments.device)
    if model.moe is None:
        raise ValueError("router analysis requires an MoE checkpoint")

    _, _, _, validation_rows, _ = load_data(
        Path(arguments.data_dir),
        validation_fraction=0.1,
        single_pass_audit_load=True,
        replay_audit_workers=2,
    )
    generator = np.random.default_rng(2026)
    count = min(arguments.samples, len(validation_rows))
    indices = generator.choice(len(validation_rows), count, replace=False)
    rows = [validation_rows[int(index)] for index in indices]
    features = torch.from_numpy(np.stack([
        features_for(row, transform)
        for row in rows
        for transform in range(8)
    ]))

    assignments = []
    with torch.inference_mode():
        for start in range(0, len(features), arguments.batch_size):
            batch = features[start:start + arguments.batch_size].to(arguments.device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                *_, auxiliary = model(batch, return_aux=True)
            assignments.extend(auxiliary["assignments"].cpu().tolist())
    routes = np.asarray(assignments, dtype=np.int64).reshape(count, 8)
    modal_counts = np.stack([
        np.bincount(route, minlength=model.moe.num_experts)
        for route in routes
    ]).max(axis=1)
    transform_counts = {
        str(transform): {
            str(int(expert)): int(total)
            for expert, total in sorted(Counter(routes[:, transform]).items())
        }
        for transform in range(8)
    }
    phase_counts = {}
    for name, condition in (
        ("opening", lambda stones: stones < 32),
        ("middle", lambda stones: 32 <= stones <= 64),
        ("late", lambda stones: stones > 64),
    ):
        selected = routes[[condition(int(row["stones_before"])) for row in rows], 0]
        phase_counts[name] = {
            str(int(expert)): int(total)
            for expert, total in sorted(Counter(selected).items())
        }
    result = {
        "samples": count,
        "all_eight_same_fraction": float(np.mean(modal_counts == 8)),
        "mean_modal_agreement": float(np.mean(modal_counts / 8.0)),
        "transform_expert_counts": transform_counts,
        "untransformed_phase_counts": phase_counts,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
