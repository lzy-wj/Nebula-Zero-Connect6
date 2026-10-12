"""Audit and summarize native-Pair replay quality without trusting manifests."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

from experiments.pair_native.pair_replay import audit_shard, load_shard, unpack_points


def d4_keys(board: np.ndarray, player: int):
    for reflected in (False, True):
        transformed = np.fliplr(board) if reflected else board
        for rotations in range(4):
            candidate = np.rot90(transformed, rotations) if rotations else transformed
            yield candidate.tobytes() + bytes((player + 1,))


def percentile_summary(values) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def analyze(data_dir: Path) -> dict:
    manifests = sorted(data_dir.glob("*.manifest.json"))
    if not manifests:
        raise ValueError(f"no Pair replay manifests under {data_dir}")
    audits = [audit_shard(manifest) for manifest in manifests]
    invalid = [audit for audit in audits if not audit["valid"]]
    if invalid:
        raise ValueError(f"invalid Pair replay shards: {invalid}")

    games = []
    turns = []
    metadata = []
    for manifest_path in manifests:
        manifest, shard_games, shard_turns = load_shard(manifest_path)
        games.extend(shard_games)
        turns.extend(shard_turns)
        metadata.append(manifest.get("metadata", {}))

    exact_states = set()
    canonical_states = set()
    searched_exact_states = set()
    searched_canonical_states = set()
    entropies = []
    effective_actions = []
    support_sizes = []
    chosen_masses = []
    root_values = []
    search_turns = []
    phase_counts = Counter()
    terminal_singletons = 0
    forced_openings = 0
    for turn in turns:
        player = int(turn["player"])
        exact_states.add(turn["state_hash"] + bytes((player + 1,)))
        black = unpack_points(turn["board_black"])
        white = unpack_points(turn["board_white"])
        board = black.astype(np.int8) - white.astype(np.int8)
        canonical_states.add(min(d4_keys(board, player)))

        probabilities = np.asarray(turn["policy_probability"], dtype=np.float64)
        positive = probabilities[probabilities > 0]
        entropy = float(-np.sum(positive * np.log(positive)))
        entropies.append(entropy)
        effective_actions.append(math.exp(entropy))
        support_sizes.append(len(positive))
        chosen = (int(turn["chosen_a"]), int(turn["chosen_b"]))
        chosen_masses.append(sum(
            float(probability)
            for first, second, probability in zip(
                turn["policy_a"],
                turn["policy_b"],
                probabilities,
            )
            if (int(first), int(second)) == chosen
        ))
        root_values.append(float(turn["root_value"]))
        stones = int(turn["stones_before"])
        phase_counts["early" if stones < 32 else "mid" if stones <= 64 else "late"] += 1
        if int(turn["simulations"]) == 0:
            forced_openings += 1
        else:
            search_turns.append(turn)
            searched_exact_states.add(turn["state_hash"] + bytes((player + 1,)))
            searched_canonical_states.add(min(d4_keys(board, player)))
        if stones > 0 and int(turn["chosen_b"]) == -1:
            terminal_singletons += 1

    lengths = [len(game["moves"]) for game in games]
    winners = Counter(int(game["winner"]) for game in games)
    root_array = np.asarray(root_values, dtype=np.float64)
    game_signatures = [tuple(int(move) for move in game["moves"]) for game in games]
    total = max(len(turns), 1)
    searched = max(len(search_turns), 1)
    return {
        "data_dir": str(data_dir),
        "shards": len(manifests),
        "all_shards_valid": True,
        "games": len(games),
        "turns": len(turns),
        "stones": int(sum(lengths)),
        "winner": {
            "black": winners[1],
            "white": winners[-1],
            "draw": winners[0],
            "black_rate": winners[1] / max(len(games), 1),
        },
        "game_length_stones": percentile_summary(lengths),
        "duplicate_complete_games": len(games) - len(set(game_signatures)),
        "state_diversity": {
            "exact_unique": len(exact_states),
            "exact_unique_rate": len(exact_states) / total,
            "d4_unique": len(canonical_states),
            "d4_unique_rate": len(canonical_states) / total,
            "searched_exact_unique": len(searched_exact_states),
            "searched_exact_unique_rate": len(searched_exact_states) / searched,
            "searched_d4_unique": len(searched_canonical_states),
            "searched_d4_unique_rate": len(searched_canonical_states) / searched,
        },
        "turn_phases": dict(phase_counts),
        "forced_openings": forced_openings,
        "terminal_singleton_turns": terminal_singletons,
        "policy": {
            "support": percentile_summary(support_sizes),
            "entropy": percentile_summary(entropies),
            "effective_actions": percentile_summary(effective_actions),
            "chosen_mass": percentile_summary(chosen_masses),
        },
        "value": {
            "mean_absolute_root_value": float(np.mean(np.abs(root_array))),
            "nonzero_root_values": int(np.count_nonzero(root_array)),
            "decisive_root_values": int(np.sum(np.abs(root_array) > 0.25)),
            "decisive_rate": float(np.mean(np.abs(root_array) > 0.25)),
        },
        "search": {
            "searched_turns": len(search_turns),
            "mean_simulations": sum(int(turn["simulations"]) for turn in search_turns)
            / searched,
            "mean_neural_evaluations": sum(
                int(turn["neural_evaluations"]) for turn in search_turns
            ) / searched,
            "model_hashes": sorted({game["model_sha256"] for game in games}),
            "engine_hashes": sorted({game["engine_sha256"] for game in games}),
            "search_config_hashes": sorted({
                game["search_config_sha256"] for game in games
            }),
        },
        "audits": audits,
        "metadata": metadata,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output", default="")
    arguments = parser.parse_args()
    result = analyze(Path(arguments.data_dir).expanduser().resolve())
    payload = json.dumps(result, indent=2)
    if arguments.output:
        output = Path(arguments.output).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(payload + "\n", encoding="utf-8")
        temporary.replace(output)
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
