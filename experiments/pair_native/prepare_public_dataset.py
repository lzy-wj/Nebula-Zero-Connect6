"""Export the exact audited replay window behind a promoted Pair checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path

from experiments.pair_native.pair_replay import audit_shard


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def promoted_entry(state: dict, through_round: int | None) -> dict:
    entries = [entry for entry in state.get("promotions", []) if entry.get("promote")]
    if through_round is not None:
        entries = [
            entry for entry in entries
            if int(entry.get("through_round", -1)) == through_round
        ]
    if not entries:
        label = "latest" if through_round is None else str(through_round)
        raise ValueError(f"no promoted checkpoint found for through-round {label}")
    return entries[-1]


def public_metadata(metadata: dict) -> dict:
    """Keep reproducibility fields while removing host paths and CPU topology."""
    execution = dict(metadata.get("execution", {}))
    execution.pop("affinity", None)
    result = {
        "model_config": metadata.get("model_config", {}),
        "search": metadata.get("search", {}),
        "execution": execution,
    }
    for key in ("worker", "gpu"):
        if key in metadata:
            result[key] = metadata[key]
    result["checkpoint"] = "pair_native_v3_inference.pth"
    return result


def length_bucket(stones: int) -> str:
    if stones <= 40:
        return "short"
    if stones <= 80:
        return "medium"
    if stones <= 140:
        return "long"
    return "very_long"


def dataset_card(stats: dict) -> str:
    winners = stats["winner_counts"]
    lengths = stats["length_buckets"]
    return f"""---
license: mit
task_categories:
- reinforcement-learning
tags:
- connect6
- alphazero
- mcts
- self-play
pretty_name: Nebula Zero Connect6 Pair Replay V3
---

# Nebula Zero Connect6 — Pair Replay V3

这是 Pair-native V3 晋级模型实际使用的、经过逐局规则与策略标签审计的 replay
窗口。数据保留联合双落子访问分布、价值、搜索统计和模型/引擎哈希。

This is the exact audited replay window used by the promoted Pair-native V3
checkpoint. It preserves joint two-stone visit distributions, values, search
statistics, and model/engine provenance.

## Snapshot

- Games: `{stats['games']:,}`
- Turn samples: `{stats['turns']:,}`
- Parquet shards: `{stats['shards']}`
- Black / white / draw: `{winners['black']:,}` / `{winners['white']:,}` / `{winners['draw']:,}`
- Short / medium / long / very long games: `{lengths['short']:,}` / `{lengths['medium']:,}` / `{lengths['long']:,}` / `{lengths['very_long']:,}`
- MCTS budget: 400–1200 simulations per complete turn

Each shard consists of `*.games.parquet`, `*.turns.parquet`, and a manifest.
The schema and strict loader are implemented in
`experiments/pair_native/pair_replay.py` of
[Nebula-Zero-Connect6](https://github.com/lzy-wj/Nebula-Zero-Connect6).

```python
from experiments.pair_native.pair_replay import load_and_audit_shard

audit, manifest, games, turns = load_and_audit_shard(
    "data/gen-1554-attempt-00/gen-1554-worker-00.manifest.json"
)
assert audit["valid"]
```

`dataset_manifest.json` records the selected promotion and aggregate statistics;
`SHA256SUMS` covers every public file.
The dataset is intended for Connect6 research; published internal promotion
results are not an external rating or proof of tournament strength.
"""


def export_dataset(
    state_path: Path,
    output_dir: Path,
    through_round: int | None = None,
    overwrite: bool = False,
) -> dict:
    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        raise RuntimeError("dataset export requires pyarrow") from error

    state_path = state_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    state = json.loads(state_path.read_text(encoding="utf-8"))
    entry = promoted_entry(state, through_round)
    selected_round = int(entry["through_round"])
    source_dirs = [Path(path).expanduser().resolve() for path in entry["replay_directories"]]
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(f"output directory is not empty: {output_dir}")
        shutil.rmtree(output_dir)
    data_root = output_dir / "data"
    data_root.mkdir(parents=True, exist_ok=True)

    winner_counts = Counter()
    length_counts = Counter()
    generations = set()
    model_hashes = set()
    total_games = 0
    total_turns = 0
    shard_records = []
    sample_games = []

    for source_dir in source_dirs:
        if not source_dir.is_dir():
            raise FileNotFoundError(source_dir)
        target_dir = data_root / source_dir.name
        target_dir.mkdir(parents=True, exist_ok=False)
        manifests = sorted(source_dir.glob("*.manifest.json"))
        if not manifests:
            raise ValueError(f"no replay manifests under {source_dir}")
        for source_manifest_path in manifests:
            source_audit = audit_shard(source_manifest_path)
            if not source_audit["valid"]:
                raise ValueError(f"invalid source shard: {source_manifest_path}")
            source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
            public_manifest = dict(source_manifest)
            public_manifest["metadata"] = public_metadata(source_manifest.get("metadata", {}))
            target_manifest_path = target_dir / source_manifest_path.name
            for table_name in ("games", "turns"):
                filename = source_manifest["files"][table_name]["name"]
                source_file = source_dir / filename
                if sha256_file(source_file) != source_manifest["files"][table_name]["sha256"]:
                    raise ValueError(f"source hash mismatch: {source_file}")
                shutil.copy2(source_file, target_dir / filename)
            target_manifest_path.write_text(
                json.dumps(public_manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            target_audit = audit_shard(target_manifest_path)
            if not target_audit["valid"]:
                raise ValueError(f"invalid exported shard: {target_manifest_path}")

            games_path = target_dir / public_manifest["files"]["games"]["name"]
            game_rows = pq.read_table(
                games_path,
                columns=["generation", "model_sha256", "winner", "moves"],
            ).to_pylist()
            for game in game_rows:
                generations.add(int(game["generation"]))
                model_hashes.add(game["model_sha256"])
                winner_counts[{1: "black", -1: "white", 0: "draw"}[int(game["winner"])]] += 1
                length_counts[length_bucket(len(game["moves"]))] += 1
                if len(sample_games) < 8:
                    sample_games.append({
                        "generation": int(game["generation"]),
                        "winner": int(game["winner"]),
                        "moves": [int(move) for move in game["moves"]],
                    })
            total_games += int(public_manifest["games"])
            total_turns += int(public_manifest["turns"])
            relative_manifest = target_manifest_path.relative_to(output_dir).as_posix()
            shard_records.append({
                "manifest": relative_manifest,
                "games": int(public_manifest["games"]),
                "turns": int(public_manifest["turns"]),
            })

    stats = {
        "games": total_games,
        "turns": total_turns,
        "shards": len(shard_records),
        "generations": sorted(generations),
        "winner_counts": {
            name: winner_counts[name] for name in ("black", "white", "draw")
        },
        "length_buckets": {
            name: length_counts[name]
            for name in ("short", "medium", "long", "very_long")
        },
    }
    (output_dir / "sample_games.json").write_text(
        json.dumps(sample_games, indent=2) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "schema_version": 1,
        "release": "pair-native-v3-exact-replay",
        "through_round": selected_round,
        "checkpoint_sha256": entry.get("candidate_sha256"),
        "model_hashes_in_replay": sorted(model_hashes),
        "stats": stats,
        "shards": shard_records,
    }
    (output_dir / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "README.md").write_text(dataset_card(stats), encoding="utf-8")

    files = sorted(
        path for path in output_dir.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS"
    )
    (output_dir / "SHA256SUMS").write_text(
        "".join(
            f"{sha256_file(path)}  {path.relative_to(output_dir).as_posix()}\n"
            for path in files
        ),
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--through-round", type=int)
    parser.add_argument("--overwrite", action="store_true")
    arguments = parser.parse_args()
    manifest = export_dataset(
        Path(arguments.state),
        Path(arguments.output_dir),
        through_round=arguments.through_round,
        overwrite=arguments.overwrite,
    )
    print(json.dumps(manifest["stats"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
