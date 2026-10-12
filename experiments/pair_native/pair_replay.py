"""Versioned Parquet replay shards for native unordered-pair self-play.

Each shard has a game table, a turn table, and a manifest written last.  A turn
stores the exact board, joint pair visit distribution, search statistics, and
provenance needed to decide whether it is a real training sample.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from experiments.pair_native.game_audit import (
    BOARD_POINTS,
    BOARD_SIZE,
    check_win,
    player_at_move,
)


SCHEMA_VERSION = 1
PASS_MOVE = -1
BOARD_BYTES = (BOARD_POINTS + 7) // 8


@dataclass(slots=True)
class PairGameRecord:
    game_id: str
    generation: int
    seed: int
    model_sha256: str
    engine_sha256: str
    search_config_sha256: str
    winner: int
    termination: str
    moves: list[int]


@dataclass(slots=True)
class PairTurnRecord:
    game_id: str
    turn_index: int
    player: int
    stones_before: int
    board_black: bytes
    board_white: bytes
    state_hash: bytes
    chosen_a: int
    chosen_b: int
    policy_a: list[int]
    policy_b: list[int]
    policy_probability: list[float]
    root_value: float
    simulations: int
    unique_nodes: int
    neural_evaluations: int
    search_ms: float


def require_pyarrow():
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as error:
        raise RuntimeError("pair replay shards require pyarrow") from error
    return pa, pq


def pack_points(mask: np.ndarray) -> bytes:
    flat = np.asarray(mask, dtype=np.uint8).reshape(-1)
    if flat.size != BOARD_POINTS:
        raise ValueError(f"expected {BOARD_POINTS} board points, got {flat.size}")
    return np.packbits(flat, bitorder="little").tobytes()


def unpack_points(payload: bytes) -> np.ndarray:
    if len(payload) != BOARD_BYTES:
        raise ValueError(f"expected {BOARD_BYTES} board bytes, got {len(payload)}")
    values = np.unpackbits(
        np.frombuffer(payload, dtype=np.uint8),
        bitorder="little",
    )[:BOARD_POINTS]
    return values.astype(bool).reshape(BOARD_SIZE, BOARD_SIZE)


def board_payloads(board: np.ndarray) -> tuple[bytes, bytes, bytes]:
    board = np.asarray(board, dtype=np.int8).reshape(BOARD_SIZE, BOARD_SIZE)
    black = pack_points(board == 1)
    white = pack_points(board == -1)
    digest = hashlib.blake2b(black + white, digest_size=16).digest()
    return black, white, digest


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def game_schema(pa):
    return pa.schema([
        ("schema_version", pa.int16()),
        ("game_id", pa.string()),
        ("generation", pa.int32()),
        ("seed", pa.int64()),
        ("model_sha256", pa.string()),
        ("engine_sha256", pa.string()),
        ("search_config_sha256", pa.string()),
        ("winner", pa.int8()),
        ("termination", pa.string()),
        ("moves", pa.list_(pa.int16())),
    ])


def turn_schema(pa):
    return pa.schema([
        ("schema_version", pa.int16()),
        ("game_id", pa.string()),
        ("turn_index", pa.int16()),
        ("player", pa.int8()),
        ("stones_before", pa.int16()),
        ("board_black", pa.binary(BOARD_BYTES)),
        ("board_white", pa.binary(BOARD_BYTES)),
        ("state_hash", pa.binary(16)),
        ("chosen_a", pa.int16()),
        ("chosen_b", pa.int16()),
        ("policy_a", pa.list_(pa.int16())),
        ("policy_b", pa.list_(pa.int16())),
        ("policy_probability", pa.list_(pa.float32())),
        ("root_value", pa.float32()),
        ("simulations", pa.int32()),
        ("unique_nodes", pa.int32()),
        ("neural_evaluations", pa.int32()),
        ("search_ms", pa.float32()),
    ])


def records_to_columns(records, fields: list[str]) -> dict[str, list[Any]]:
    columns = {field: [] for field in fields}
    for record in records:
        values = asdict(record)
        values["schema_version"] = SCHEMA_VERSION
        for field in fields:
            columns[field].append(values[field])
    return columns


def atomic_write_parquet(table, path: Path, pq):
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    pq.write_table(
        table,
        temporary,
        compression="zstd",
        use_dictionary=True,
        write_statistics=True,
    )
    os.replace(temporary, path)


def write_shard(
    output_dir: str | os.PathLike,
    shard_name: str,
    games: list[PairGameRecord],
    turns: list[PairTurnRecord],
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Atomically write data files, then publish the manifest as commit marker."""

    if not games:
        raise ValueError("a replay shard must contain at least one game")
    if not turns:
        raise ValueError("a replay shard must contain at least one turn")
    pa, pq = require_pyarrow()
    directory = Path(output_dir).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    games_path = directory / f"{shard_name}.games.parquet"
    turns_path = directory / f"{shard_name}.turns.parquet"
    manifest_path = directory / f"{shard_name}.manifest.json"

    game_fields = [field.name for field in game_schema(pa)]
    turn_fields = [field.name for field in turn_schema(pa)]
    game_table = pa.Table.from_pydict(
        records_to_columns(games, game_fields),
        schema=game_schema(pa),
    )
    turn_table = pa.Table.from_pydict(
        records_to_columns(turns, turn_fields),
        schema=turn_schema(pa),
    )
    atomic_write_parquet(game_table, games_path, pq)
    atomic_write_parquet(turn_table, turns_path, pq)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "shard": shard_name,
        "games": len(games),
        "turns": len(turns),
        "files": {
            "games": {"name": games_path.name, "sha256": sha256_file(games_path)},
            "turns": {"name": turns_path.name, "sha256": sha256_file(turns_path)},
        },
        "metadata": metadata or {},
    }
    temporary_manifest = manifest_path.with_suffix(
        manifest_path.suffix + f".tmp.{os.getpid()}"
    )
    temporary_manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_manifest, manifest_path)
    return manifest_path


def load_shard(manifest_path: str | os.PathLike):
    pa, pq = require_pyarrow()
    path = Path(manifest_path).expanduser().resolve()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported schema version: {manifest.get('schema_version')}")
    tables = {}
    for name in ("games", "turns"):
        entry = manifest["files"][name]
        data_path = path.parent / entry["name"]
        if sha256_file(data_path) != entry["sha256"]:
            raise ValueError(f"{name} parquet SHA256 mismatch")
        tables[name] = pq.read_table(data_path)
    if not tables["games"].schema.equals(game_schema(pa), check_metadata=False):
        raise ValueError("game parquet schema mismatch")
    if not tables["turns"].schema.equals(turn_schema(pa), check_metadata=False):
        raise ValueError("turn parquet schema mismatch")
    return manifest, tables["games"].to_pylist(), tables["turns"].to_pylist()


def load_verified_games(manifest_path: str | os.PathLike):
    """Reverify both committed files while decoding only the small game table.

    Callers use this only after a strict turn-level audit.  Rehashing the turn
    parquet closes the mutation window without paying for a second conversion
    of every turn into Python objects.
    """

    pa, pq = require_pyarrow()
    path = Path(manifest_path).expanduser().resolve()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported schema version: {manifest.get('schema_version')}")
    paths = {}
    for name in ("games", "turns"):
        entry = manifest["files"][name]
        data_path = path.parent / entry["name"]
        if sha256_file(data_path) != entry["sha256"]:
            raise ValueError(f"{name} parquet SHA256 mismatch")
        paths[name] = data_path
    games = pq.read_table(paths["games"])
    if not games.schema.equals(game_schema(pa), check_metadata=False):
        raise ValueError("game parquet schema mismatch")
    return manifest, games.to_pylist()


def replay_game(moves: list[int]):
    board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
    states = {}
    winner = None
    for move_index, raw_move in enumerate(moves):
        move = int(raw_move)
        if move_index == 0 or move_index % 2 == 1:
            states[move_index] = board.copy()
        if winner is not None:
            return board, states, winner, "moves_after_terminal"
        if not 0 <= move < BOARD_POINTS or board.flat[move] != 0:
            return board, states, winner, "illegal_move"
        player = player_at_move(move_index)
        board.flat[move] = player
        if check_win(board, move, player):
            winner = player
    if winner is None and len(moves) == BOARD_POINTS:
        winner = 0
    if winner is None:
        return board, states, winner, "nonterminal"
    return board, states, winner, None


def immediate_win(board: np.ndarray, move: int, player: int) -> bool:
    if not 0 <= move < BOARD_POINTS or board.flat[move] != 0:
        return False
    board.flat[move] = player
    won = check_win(board, move, player)
    board.flat[move] = 0
    return won


def validate_action(
    board: np.ndarray,
    player: int,
    first: int,
    second: int,
    opening: bool,
) -> str | None:
    if not 0 <= first < BOARD_POINTS or board.flat[first] != 0:
        return "illegal_first_move"
    if second == PASS_MOVE:
        if opening or immediate_win(board, first, player):
            return None
        return "illegal_singleton_action"
    if not first < second < BOARD_POINTS:
        return "noncanonical_pair"
    if board.flat[second] != 0:
        return "illegal_second_move"
    if immediate_win(board, first, player):
        return "pair_continues_after_first_stone_win"
    return None


def _audit_loaded_shard(
    manifest_path: str | os.PathLike,
    manifest: dict,
    games: list[dict],
    turns: list[dict],
    probability_tolerance: float = 0.02,
) -> dict:
    """Audit already loaded records without reading the shard a second time."""

    errors = Counter()
    if int(manifest.get("games", -1)) != len(games):
        errors["manifest_game_count_mismatch"] += 1
    if int(manifest.get("turns", -1)) != len(turns):
        errors["manifest_turn_count_mismatch"] += 1
    game_by_id = {}
    replayed = {}
    generations = set()
    model_hashes = set()
    engine_hashes = set()
    search_config_hashes = set()
    for game in games:
        if int(game["schema_version"]) != SCHEMA_VERSION:
            errors["game_schema_version_mismatch"] += 1
        game_id = game["game_id"]
        if game_id in game_by_id:
            errors["duplicate_game_id"] += 1
            continue
        game_by_id[game_id] = game
        generations.add(int(game["generation"]))
        model_hashes.add(game["model_sha256"])
        engine_hashes.add(game["engine_sha256"])
        search_config_hashes.add(game["search_config_sha256"])
        for field in ("model_sha256", "engine_sha256", "search_config_sha256"):
            value = game[field]
            if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
                errors[f"invalid_{field}"] += 1
        _, states, winner, replay_error = replay_game(game["moves"])
        if replay_error:
            errors[replay_error] += 1
        if winner != game["winner"]:
            errors["winner_mismatch"] += 1
        replayed[game_id] = (states, winner)
    if len(generations) != 1:
        errors["mixed_generation_shard"] += 1
    if len(model_hashes) != 1:
        errors["mixed_model_shard"] += 1
    if len(engine_hashes) != 1:
        errors["mixed_engine_shard"] += 1
    if len(search_config_hashes) != 1:
        errors["mixed_search_config_shard"] += 1

    turns_by_game = defaultdict(list)
    valid_turns = 0
    unique_states = set()
    search_metadata = manifest.get("metadata", {}).get("search", {})
    expected_simulations = search_metadata.get("simulations")
    if expected_simulations is not None:
        expected_simulations = int(expected_simulations)
        if expected_simulations <= 0:
            errors["invalid_expected_simulations"] += 1
    expected_black_simulations = search_metadata.get(
        "black_simulations",
        expected_simulations,
    )
    expected_white_simulations = search_metadata.get(
        "white_simulations",
        expected_simulations,
    )
    if expected_black_simulations is not None:
        expected_black_simulations = int(expected_black_simulations)
    if expected_white_simulations is not None:
        expected_white_simulations = int(expected_white_simulations)
    if (
        expected_black_simulations is not None
        and expected_black_simulations <= 0
    ) or (
        expected_white_simulations is not None
        and expected_white_simulations <= 0
    ):
        errors["invalid_expected_color_simulations"] += 1
    adaptive_simulations = bool(search_metadata.get("adaptive_simulations", False))
    adaptive_min_simulations = int(
        search_metadata.get("adaptive_min_simulations", expected_simulations or 0)
    )
    adaptive_mid_simulations = int(
        search_metadata.get("adaptive_mid_simulations", adaptive_min_simulations)
    )
    if adaptive_simulations and not (
        0 < adaptive_min_simulations <= adaptive_mid_simulations
    ):
        errors["invalid_adaptive_simulations"] += 1
    for turn in turns:
        if int(turn["schema_version"]) != SCHEMA_VERSION:
            errors["turn_schema_version_mismatch"] += 1
        game_id = turn["game_id"]
        turns_by_game[game_id].append(turn)
        turn_errors_before = sum(errors.values())
        game = game_by_id.get(game_id)
        if game is None:
            errors["orphan_turn"] += 1
            continue
        states, _ = replayed[game_id]
        turn_index = int(turn["turn_index"])
        start = 0 if turn_index == 0 else 2 * turn_index - 1
        board = states.get(start)
        if board is None:
            errors["turn_index_out_of_range"] += 1
            continue
        player = player_at_move(start)
        if turn["player"] != player or turn["stones_before"] != start:
            errors["turn_metadata_mismatch"] += 1
        black, white, digest = board_payloads(board)
        if turn["board_black"] != black or turn["board_white"] != white:
            errors["board_payload_mismatch"] += 1
        if turn["state_hash"] != digest:
            errors["state_hash_mismatch"] += 1
        unique_states.add(digest + bytes((player + 1,)))

        moves = game["moves"]
        actual_first = int(moves[start])
        actual_second = (
            PASS_MOVE
            if start == 0 or start + 1 >= len(moves)
            else int(moves[start + 1])
        )
        if actual_second != PASS_MOVE:
            actual_first, actual_second = sorted((actual_first, actual_second))
        if turn["chosen_a"] != actual_first or turn["chosen_b"] != actual_second:
            errors["chosen_action_mismatch"] += 1

        action_error = validate_action(
            board.copy(),
            player,
            int(turn["chosen_a"]),
            int(turn["chosen_b"]),
            opening=start == 0,
        )
        if action_error:
            errors[action_error] += 1

        policy_a = turn["policy_a"]
        policy_b = turn["policy_b"]
        probabilities = turn["policy_probability"]
        if not policy_a or not (len(policy_a) == len(policy_b) == len(probabilities)):
            errors["policy_length_mismatch"] += 1
        else:
            total = 0.0
            chosen_mass = 0.0
            seen_actions = set()
            for first, second, probability in zip(policy_a, policy_b, probabilities):
                action = (int(first), int(second))
                if action in seen_actions:
                    errors["duplicate_policy_action"] += 1
                seen_actions.add(action)
                if not math.isfinite(probability) or probability < 0:
                    errors["invalid_policy_probability"] += 1
                    continue
                total += float(probability)
                candidate_error = validate_action(
                    board.copy(),
                    player,
                    action[0],
                    action[1],
                    opening=start == 0,
                )
                if candidate_error:
                    errors[f"policy_{candidate_error}"] += 1
                if action == (actual_first, actual_second):
                    chosen_mass += float(probability)
            if abs(total - 1.0) > probability_tolerance:
                errors["policy_not_normalized"] += 1
            if chosen_mass <= 0:
                errors["chosen_action_missing_from_policy"] += 1

        if not math.isfinite(turn["root_value"]) or not -1.0 <= turn["root_value"] <= 1.0:
            errors["invalid_root_value"] += 1
        forced_radius = manifest.get("metadata", {}).get("search", {}).get(
            "forced_opening_radius",
            -1,
        )
        forced_opening = turn_index == 0 and forced_radius >= 0
        valid_forced_statistics = (
            forced_opening
            and turn["simulations"] == 0
            and turn["unique_nodes"] == 0
            and turn["neural_evaluations"] == 0
            and turn["search_ms"] == 0
        )
        valid_search_statistics = (
            not forced_opening
            and turn["simulations"] > 0
            and turn["unique_nodes"] >= 0
            and turn["neural_evaluations"] >= 0
            and math.isfinite(turn["search_ms"])
            and turn["search_ms"] > 0
        )
        expected_turn_simulations = (
            expected_black_simulations
            if int(turn["player"]) == 1
            else expected_white_simulations
        )
        if expected_turn_simulations is not None and not forced_opening:
            if adaptive_simulations:
                allowed_simulations = {
                    min(adaptive_min_simulations, expected_turn_simulations),
                    min(adaptive_mid_simulations, expected_turn_simulations),
                    expected_turn_simulations,
                }
                if int(turn["simulations"]) not in allowed_simulations:
                    errors["simulation_budget_mismatch"] += 1
            elif int(turn["simulations"]) != expected_turn_simulations:
                errors["simulation_budget_mismatch"] += 1
        if forced_opening:
            center = BOARD_SIZE // 2
            expected_points = {
                row * BOARD_SIZE + column
                for row in range(center - forced_radius, center + forced_radius + 1)
                for column in range(center - forced_radius, center + forced_radius + 1)
            }
            policy_actions = {
                (int(first), int(second))
                for first, second in zip(policy_a, policy_b)
            }
            expected_actions = {(point, PASS_MOVE) for point in expected_points}
            if int(turn["chosen_a"]) not in expected_points:
                errors["forced_opening_outside_region"] += 1
            if policy_actions != expected_actions:
                errors["forced_opening_policy_mismatch"] += 1
            expected_probability = 1.0 / len(expected_points)
            if any(
                abs(float(probability) - expected_probability) > 1.0e-6
                for probability in probabilities
            ):
                errors["forced_opening_probability_mismatch"] += 1
        if not valid_forced_statistics and not valid_search_statistics:
            errors["invalid_search_statistics"] += 1
        if sum(errors.values()) == turn_errors_before:
            valid_turns += 1

    for game_id, game in game_by_id.items():
        game_turns = sorted(turns_by_game.get(game_id, []), key=lambda item: item["turn_index"])
        expected_count = (len(game["moves"]) + 2) // 2
        if len(game_turns) != expected_count:
            errors["game_turn_count_mismatch"] += 1
        if [turn["turn_index"] for turn in game_turns] != list(range(len(game_turns))):
            errors["nonconsecutive_turn_indices"] += 1

    return {
        "schema_version": SCHEMA_VERSION,
        "manifest": str(Path(manifest_path).expanduser().resolve()),
        "games": len(games),
        "turns": len(turns),
        "valid_turns": valid_turns,
        "unique_states": len(unique_states),
        "errors": {name: int(value) for name, value in sorted(errors.items())},
        "valid": not errors,
        "metadata": manifest.get("metadata", {}),
        "provenance": {
            "generations": sorted(generations),
            "model_sha256": sorted(model_hashes),
            "engine_sha256": sorted(engine_hashes),
            "search_config_sha256": sorted(search_config_hashes),
        },
    }


def load_and_audit_shard(
    manifest_path: str | os.PathLike,
    probability_tolerance: float = 0.02,
) -> tuple[dict, dict, list[dict], list[dict]]:
    """Load a shard once and return its strict audit beside the decoded rows."""

    manifest, games, turns = load_shard(manifest_path)
    audit = _audit_loaded_shard(
        manifest_path,
        manifest,
        games,
        turns,
        probability_tolerance,
    )
    return audit, manifest, games, turns


def audit_shard(
    manifest_path: str | os.PathLike,
    probability_tolerance: float = 0.02,
) -> dict:
    """Validate provenance, games, boards, joint policies, and search metrics."""

    audit, _, _, _ = load_and_audit_shard(
        manifest_path,
        probability_tolerance,
    )
    return audit
