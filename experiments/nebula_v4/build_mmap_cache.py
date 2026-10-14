"""Audit once and pack a deterministic V4 supervision window into mmap arrays."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import multiprocessing as mp
import os
import shutil
import time
from pathlib import Path

import numpy as np

from experiments.nebula_v4.launch_compare import discover_generations
from experiments.pair_native.pair_replay import PASS_MOVE, audit_shard, load_shard


DEFAULT_POLICY_WIDTH = 64
BOARD_BYTES = 46


def manifests_for(generations: list[tuple[int, Path]]) -> list[Path]:
    manifests: list[Path] = []
    for _, directory in generations:
        current = sorted(directory.glob("*.manifest.json"))
        if not current:
            raise ValueError(f"no manifests under {directory}")
        manifests.extend(current)
    return manifests


def manifest_policy_width(manifests: list[Path]) -> int:
    widths = []
    for path in manifests:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        width = manifest.get("metadata", {}).get("search", {}).get("top_k")
        if width is not None:
            widths.append(int(width))
    return max(widths, default=DEFAULT_POLICY_WIDTH)


def create_arrays(
    directory: Path,
    rows: int,
    policy_width: int = DEFAULT_POLICY_WIDTH,
) -> dict[str, np.memmap]:
    if policy_width < 1:
        raise ValueError("policy width must be positive")
    directory.mkdir(parents=True)
    specifications = {
        "board_black": (np.uint8, (rows, BOARD_BYTES)),
        "board_white": (np.uint8, (rows, BOARD_BYTES)),
        "player": (np.int8, (rows,)),
        "stones_before": (np.uint16, (rows,)),
        "chosen_a": (np.int16, (rows,)),
        "chosen_b": (np.int16, (rows,)),
        "policy_a": (np.int16, (rows, policy_width)),
        "policy_b": (np.int16, (rows, policy_width)),
        "policy_probability": (np.float32, (rows, policy_width)),
        "policy_length": (np.uint16, (rows,)),
        "value_outcome": (np.int8, (rows,)),
        "root_value": (np.float32, (rows,)),
        "remaining_turns": (np.uint16, (rows,)),
    }
    return {
        name: np.lib.format.open_memmap(
            directory / f"{name}.npy", mode="w+", dtype=dtype, shape=shape
        )
        for name, (dtype, shape) in specifications.items()
    }


def write_rows(
    arrays: dict[str, np.memmap],
    offset: int,
    rows: list[dict],
    games: list[dict],
    last_turns: dict[str, int] | None = None,
) -> None:
    count = len(rows)
    target = slice(offset, offset + count)
    winners = {game["game_id"]: int(game["winner"]) for game in games}
    if last_turns is None:
        last_turns = {}
        for row in rows:
            game_id = row["game_id"]
            last_turns[game_id] = max(
                last_turns.get(game_id, -1),
                int(row["turn_index"]),
            )
    black = b"".join(row["board_black"] for row in rows)
    white = b"".join(row["board_white"] for row in rows)
    if len(black) != count * BOARD_BYTES or len(white) != count * BOARD_BYTES:
        raise ValueError("unexpected packed board width")
    arrays["board_black"][target] = np.frombuffer(black, np.uint8).reshape(
        count, BOARD_BYTES
    )
    arrays["board_white"][target] = np.frombuffer(white, np.uint8).reshape(
        count, BOARD_BYTES
    )
    for name, dtype in (
        ("player", np.int8),
        ("stones_before", np.uint16),
        ("chosen_a", np.int16),
        ("chosen_b", np.int16),
        ("root_value", np.float32),
    ):
        arrays[name][target] = np.fromiter(
            (row[name] for row in rows), dtype=dtype, count=count
        )
    lengths = np.fromiter(
        (len(row["policy_a"]) for row in rows), dtype=np.uint16, count=count
    )
    policy_width = arrays["policy_a"].shape[1]
    if count and int(lengths.max()) > policy_width:
        raise ValueError(f"policy width exceeds {policy_width}")
    policy_a = np.full((count, policy_width), -1, dtype=np.int16)
    policy_b = np.full((count, policy_width), -2, dtype=np.int16)
    probability = np.zeros((count, policy_width), dtype=np.float32)
    for index, row in enumerate(rows):
        length = int(lengths[index])
        policy_a[index, :length] = row["policy_a"]
        policy_b[index, :length] = row["policy_b"]
        probability[index, :length] = row["policy_probability"]
    arrays["policy_a"][target] = policy_a
    arrays["policy_b"][target] = policy_b
    arrays["policy_probability"][target] = probability
    arrays["policy_length"][target] = lengths
    arrays["value_outcome"][target] = np.fromiter(
        (winners[row["game_id"]] * int(row["player"]) for row in rows),
        dtype=np.int8,
        count=count,
    )
    arrays["remaining_turns"][target] = np.fromiter(
        (last_turns[row["game_id"]] - int(row["turn_index"]) for row in rows),
        dtype=np.uint16,
        count=count,
    )


_PACK_ARRAYS: dict[str, np.memmap] = {}


def initialize_pack_worker(output: str) -> None:
    global _PACK_ARRAYS
    directory = Path(output)
    _PACK_ARRAYS = {
        path.stem: np.load(path, mmap_mode="r+")
        for path in directory.glob("*.npy")
    }


def pack_shard_worker(task: tuple[str, int, int, np.ndarray | None]) -> int:
    manifest_name, expected_turns, output_offset, local_indices = task
    _, games, shard_rows = load_shard(manifest_name)
    if len(shard_rows) != expected_turns:
        raise ValueError(f"turn count changed after audit: {manifest_name}")
    last_turns: dict[str, int] = {}
    for row in shard_rows:
        game_id = row["game_id"]
        last_turns[game_id] = max(
            last_turns.get(game_id, -1), int(row["turn_index"])
        )
    rows = (
        shard_rows
        if local_indices is None
        else [shard_rows[int(index)] for index in local_indices]
    )
    write_rows(_PACK_ARRAYS, output_offset, rows, games, last_turns)
    return len(rows)


def pack_split(
    manifests: list[Path],
    audits: dict[Path, dict],
    output: Path,
    selected: np.ndarray | None,
    workers: int,
    policy_width: int,
) -> int:
    total = sum(int(audits[path]["turns"]) for path in manifests)
    output_rows = total if selected is None else len(selected)
    arrays = create_arrays(output, output_rows, policy_width)
    for array in arrays.values():
        array.flush()
    del arrays
    source_offset = 0
    output_offset = 0
    started = time.time()
    tasks = []
    for manifest in manifests:
        turns = int(audits[manifest]["turns"])
        if selected is None:
            local_indices = None
        else:
            left = int(np.searchsorted(selected, source_offset, side="left"))
            right = int(np.searchsorted(selected, source_offset + turns, side="left"))
            local_indices = selected[left:right] - source_offset
        count = turns if local_indices is None else len(local_indices)
        tasks.append(
            (str(manifest), turns, output_offset, local_indices)
        )
        output_offset += count
        source_offset += turns
    if source_offset != total or output_offset != output_rows:
        raise RuntimeError("cache row accounting mismatch")
    completed_rows = 0
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=min(workers, len(tasks)),
        mp_context=mp.get_context("spawn"),
        initializer=initialize_pack_worker,
        initargs=(str(output),),
    ) as executor:
        results = executor.map(pack_shard_worker, tasks)
        for shard_index, count in enumerate(results, 1):
            completed_rows += count
            if shard_index % 25 == 0 or shard_index == len(manifests):
                print(
                    json.dumps(
                        {
                            "event": "pack_progress",
                            "split": output.name,
                            "shards": shard_index,
                            "total_shards": len(manifests),
                            "rows": completed_rows,
                            "seconds": time.time() - started,
                        }
                    ),
                    flush=True,
                )
    if completed_rows != output_rows:
        raise RuntimeError("parallel cache packing lost rows")
    for path in output.glob("*.npy"):
        array = np.load(path, mmap_mode="r+")
        array.flush()
        del array
    return output_rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--generations", type=int, default=200)
    parser.add_argument("--validation-generations", type=int, default=10)
    parser.add_argument("--train-samples", type=int, default=12_000_000)
    parser.add_argument("--validation-samples", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=20261013)
    parser.add_argument("--audit-workers", type=int, default=32)
    parser.add_argument("--pack-workers", type=int, default=16)
    parser.add_argument(
        "--policy-width",
        type=int,
        default=0,
        help="fixed cache width; 0 derives the maximum top-k from manifests",
    )
    arguments = parser.parse_args()

    data_root = Path(arguments.data_root).expanduser().resolve()
    selected_generations = discover_generations(data_root)[-arguments.generations :]
    if len(selected_generations) != arguments.generations:
        raise ValueError("not enough generations")
    split = len(selected_generations) - arguments.validation_generations
    training_generations = selected_generations[:split]
    validation_generations = selected_generations[split:]
    training_manifests = manifests_for(training_generations)
    validation_manifests = manifests_for(validation_generations)
    all_manifests = training_manifests + validation_manifests

    output = Path(arguments.output).expanduser().resolve()
    temporary = output.with_name(output.name + ".building")
    if output.exists() or temporary.exists():
        raise FileExistsError(output if output.exists() else temporary)
    temporary.mkdir(parents=True)
    started = time.time()
    try:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=arguments.audit_workers,
            mp_context=mp.get_context("spawn"),
        ) as executor:
            reports = list(executor.map(audit_shard, all_manifests))
        audits = dict(zip(all_manifests, reports))
        invalid = [str(path) for path, report in audits.items() if not report["valid"]]
        if invalid:
            raise ValueError(f"invalid shards: {invalid[:10]}")
        discovered_policy_width = manifest_policy_width(all_manifests)
        policy_width = arguments.policy_width or discovered_policy_width
        if policy_width < discovered_policy_width:
            raise ValueError(
                f"policy width {policy_width} is below replay width "
                f"{discovered_policy_width}"
            )
        print(
            json.dumps(
                {
                    "event": "audit_complete",
                    "shards": len(all_manifests),
                    "seconds": time.time() - started,
                }
            ),
            flush=True,
        )

        training_turns = sum(int(audits[path]["turns"]) for path in training_manifests)
        validation_turns = sum(
            int(audits[path]["turns"]) for path in validation_manifests
        )
        generator = np.random.default_rng(arguments.seed)
        train_count = min(arguments.train_samples, training_turns)
        train_indices = np.sort(
            generator.choice(training_turns, size=train_count, replace=False)
        )
        validation_generator = np.random.default_rng(arguments.seed + 1)
        validation_count = min(arguments.validation_samples, validation_turns)
        validation_indices = np.sort(
            validation_generator.choice(
                validation_turns, size=validation_count, replace=False
            )
        )
        packed_training = pack_split(
            training_manifests,
            audits,
            temporary / "train",
            train_indices,
            arguments.pack_workers,
            policy_width,
        )
        packed_validation = pack_split(
            validation_manifests,
            audits,
            temporary / "validation",
            None,
            arguments.pack_workers,
            policy_width,
        )
        np.save(temporary / "training_indices.npy", train_indices)
        np.save(temporary / "validation_indices.npy", validation_indices)
        source_digest = hashlib.sha256()
        for manifest in all_manifests:
            source_digest.update(manifest.read_bytes())
        metadata = {
            "schema_version": 2,
            "created_unix": time.time(),
            "data_root": str(data_root),
            "training_generations": [
                training_generations[0][0],
                training_generations[-1][0],
            ],
            "validation_generations": [
                validation_generations[0][0],
                validation_generations[-1][0],
            ],
            "source_shards": len(all_manifests),
            "source_games": sum(int(report["games"]) for report in reports),
            "source_manifest_sha256": source_digest.hexdigest(),
            "source_training_turns": training_turns,
            "source_validation_turns": validation_turns,
            "training_turns": packed_training,
            "validation_turns": packed_validation,
            "validation_sample_turns": validation_count,
            "policy_width": policy_width,
            "seed": arguments.seed,
            "strict_audit": True,
            "seconds": time.time() - started,
        }
        (temporary / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.rename(temporary, output)
        print(json.dumps({"event": "complete", **metadata}), flush=True)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
