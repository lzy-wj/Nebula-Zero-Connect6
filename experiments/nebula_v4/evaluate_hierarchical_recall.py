"""Measure hierarchical Pair candidate recall on audited validation targets."""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from experiments.nebula_v4.mmap_dataset import MMapTurnDataset, collate_mmap_turns
from experiments.pair_native.model import load_checkpoint
from experiments.pair_native.pair_replay import PASS_MOVE


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--samples", type=int, default=20_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--first-candidates", type=int, default=16)
    parser.add_argument("--seconds-per-first", type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda:0")
    model, checkpoint = load_checkpoint(args.checkpoint)
    model = torch.compile(model.eval().to(device), fullgraph=True, mode="reduce-overhead")
    score_model = getattr(model, "_orig_mod", model)
    dataset = MMapTurnDataset(args.cache_dir, "validation", training=False)
    if args.samples > 0 and args.samples < len(dataset):
        generator = np.random.default_rng(2026)
        indices = np.sort(generator.choice(len(dataset), args.samples, replace=False))
        dataset = Subset(dataset, indices.tolist())
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        collate_fn=collate_mmap_turns,
    )

    samples = 0
    pair_samples = 0
    top_target_hits = 0.0
    support_mass = 0.0
    candidate_top1_hits = 0.0
    singleton_hits = 0.0
    singleton_samples = 0
    with torch.inference_mode():
        for batch in loader:
            features = batch["features"].to(
                device, non_blocking=True, memory_format=torch.channels_last
            )
            target_a = batch["policy_a"].to(device, non_blocking=True)
            target_b = batch["policy_b"].to(device, non_blocking=True)
            probability = batch["policy_probability"].to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16):
                unary, embeddings, _, occupied = model(features)
                first_logits = unary.masked_fill(
                    occupied, torch.finfo(unary.dtype).min
                )
                selected_first = first_logits.topk(
                    args.first_candidates, dim=1
                ).indices
                rows = score_model.pair_row_scores(
                    unary, embeddings, occupied, selected_first
                )
                second_values, selected_second = rows.topk(
                    args.seconds_per_first, dim=2
                )

            batch_size = features.shape[0]
            candidate_first = selected_first[:, :, None].expand_as(selected_second)
            candidate_a = torch.minimum(candidate_first, selected_second).flatten(1)
            candidate_b = torch.maximum(candidate_first, selected_second).flatten(1)
            valid = probability.gt(0)
            singleton = target_b.eq(PASS_MOVE)
            pair_valid = valid & ~singleton
            safe_b = target_b.clamp_min(0)
            matches = (
                target_a[:, :, None].eq(candidate_a[:, None, :])
                & safe_b[:, :, None].eq(candidate_b[:, None, :])
            ).any(dim=2)
            support_mass += float((probability * matches * pair_valid).sum())

            target_index = probability.argmax(dim=1, keepdim=True)
            target_top_a = target_a.gather(1, target_index).squeeze(1)
            target_top_b = target_b.gather(1, target_index).squeeze(1)
            top_singleton = target_top_b.eq(PASS_MOVE)
            pair_mask = ~top_singleton
            top_pair_hit = (
                target_top_a[:, None].eq(candidate_a)
                & target_top_b[:, None].eq(candidate_b)
            ).any(dim=1)
            top_target_hits += float((top_pair_hit & pair_mask).sum())
            pair_samples += int(pair_mask.sum())

            best = second_values.flatten(1).argmax(dim=1)
            predicted_a = candidate_a.gather(1, best[:, None]).squeeze(1)
            predicted_b = candidate_b.gather(1, best[:, None]).squeeze(1)
            candidate_top1_hits += float((
                predicted_a.eq(target_top_a)
                & predicted_b.eq(target_top_b)
                & pair_mask
            ).sum())

            unary_prediction = first_logits.argmax(dim=1)
            singleton_hits += float((
                unary_prediction.eq(target_top_a) & top_singleton
            ).sum())
            singleton_samples += int(top_singleton.sum())
            samples += batch_size

    print(json.dumps({
        "checkpoint_model_config": checkpoint["model_config"],
        "samples": samples,
        "pair_samples": pair_samples,
        "first_candidates": args.first_candidates,
        "seconds_per_first": args.seconds_per_first,
        "returned_pairs": args.first_candidates * args.seconds_per_first,
        "top_target_candidate_recall": top_target_hits / max(pair_samples, 1),
        "target_support_mass_recall": support_mass / max(pair_samples, 1),
        "candidate_policy_top1": candidate_top1_hits / max(pair_samples, 1),
        "singleton_policy_top1": singleton_hits / max(singleton_samples, 1),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
