"""Strict, streaming audit for Connect6 self-play CSV files.

The existing training format stores one sparse policy per placed stone.  This
module does not try to repair records.  It replays every move under the game
rules, verifies the recorded result, checks policy legality/alignment, and
reports how many turn-boundary states are genuinely usable for training.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import Counter
from pathlib import Path

import numpy as np


BOARD_SIZE = 19
BOARD_POINTS = BOARD_SIZE * BOARD_SIZE
WINNER_TO_PLAYER = {"black": 1, "white": -1, "draw": 0}
PLAYER_TO_WINNER = {1: "black", -1: "white", 0: "draw"}
DIRECTIONS = ((0, 1), (1, 0), (1, 1), (1, -1))


def coordinate_to_index(text: str) -> int | None:
    value = str(text).strip().lower()
    if len(value) < 2 or not "a" <= value[0] <= "s":
        return None
    try:
        row = int(value[1:]) - 1
    except ValueError:
        return None
    column = ord(value[0]) - ord("a")
    if not 0 <= row < BOARD_SIZE:
        return None
    return row * BOARD_SIZE + column


def player_at_move(move_index: int) -> int:
    if move_index == 0:
        return 1
    return -1 if ((move_index + 1) // 2) % 2 else 1


def is_turn_start(move_index: int) -> bool:
    return move_index == 0 or move_index % 2 == 1


def check_win(board: np.ndarray, move: int, player: int) -> bool:
    row, column = divmod(int(move), BOARD_SIZE)
    for row_step, column_step in DIRECTIONS:
        count = 1
        for sign in (-1, 1):
            test_row = row + sign * row_step
            test_column = column + sign * column_step
            while (
                0 <= test_row < BOARD_SIZE
                and 0 <= test_column < BOARD_SIZE
                and board[test_row, test_column] == player
            ):
                count += 1
                test_row += sign * row_step
                test_column += sign * column_step
        if count >= 6:
            return True
    return False


def parse_sparse_policy(text: str) -> tuple[dict[int, float], Counter]:
    values: dict[int, float] = {}
    errors = Counter()
    if not text.strip():
        return values, errors
    for raw_item in text.split(";"):
        item = raw_item.strip()
        if not item:
            continue
        if ":" not in item:
            errors["malformed_policy_item"] += 1
            continue
        index_text, probability_text = item.split(":", 1)
        try:
            index = int(index_text)
            probability = float(probability_text)
        except ValueError:
            errors["malformed_policy_item"] += 1
            continue
        if not 0 <= index < BOARD_POINTS:
            errors["policy_index_out_of_range"] += 1
            continue
        if not math.isfinite(probability) or probability < 0:
            errors["invalid_policy_probability"] += 1
            continue
        if index in values:
            errors["duplicate_policy_index"] += 1
            values[index] += probability
        else:
            values[index] = probability
    return values, errors


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = round((len(ordered) - 1) * fraction)
    return float(ordered[index])


def state_key(board: np.ndarray, player: int, canonical_d4: bool) -> bytes:
    if canonical_d4:
        transforms = []
        for reflected in (False, True):
            candidate = np.fliplr(board) if reflected else board
            for rotation in range(4):
                transforms.append(np.rot90(candidate, rotation).tobytes())
        board_bytes = min(transforms)
    else:
        board_bytes = board.tobytes()
    payload = board_bytes + bytes((player + 1,))
    return hashlib.blake2b(payload, digest_size=16).digest()


class GameAudit:
    def __init__(self, policy_tolerance: float = 0.02, max_examples: int = 20):
        self.policy_tolerance = float(policy_tolerance)
        self.max_examples = int(max_examples)
        self.counts = Counter()
        self.winners = Counter()
        self.valid_winners = Counter()
        self.errors = Counter()
        self.error_examples: dict[str, list[dict]] = {}
        self.game_lengths: list[float] = []
        self.valid_game_lengths: list[float] = []
        self.policy_entropies: list[float] = []
        self.policy_supports: list[float] = []
        self.policy_top1: list[float] = []
        self.chosen_policy_mass: list[float] = []
        self.game_fingerprints: set[tuple[int, ...]] = set()
        self.turn_states: set[bytes] = set()
        self.turn_states_d4: set[bytes] = set()
        self.labeled_turn_states: set[bytes] = set()
        self.openings: set[tuple[int, ...]] = set()
        self.file_rows = Counter()

    def record_error(self, name: str, path: Path, row_number: int, detail: str):
        self.errors[name] += 1
        examples = self.error_examples.setdefault(name, [])
        if len(examples) < self.max_examples:
            examples.append({
                "file": str(path),
                "row": int(row_number),
                "detail": str(detail),
            })

    def audit_row(self, row: dict[str, str], path: Path, row_number: int):
        self.counts["rows"] += 1
        self.file_rows[str(path)] += 1
        row_hard_valid = True
        winner_name = row.get("winner", "").strip().lower()
        self.winners[winner_name or "missing"] += 1
        if winner_name not in WINNER_TO_PLAYER:
            self.record_error("invalid_recorded_winner", path, row_number, winner_name)
            row_hard_valid = False

        move_texts = [item.strip() for item in row.get("moves", "").split(",") if item.strip()]
        policies = row.get("policies", "").split("|") if row.get("policies", "") else []
        self.game_lengths.append(float(len(move_texts)))
        row_labels_usable = True
        row_all_moves_labeled = True
        row_valid_policy_labels = 0
        if not move_texts:
            self.record_error("empty_game", path, row_number, "no moves")
            return
        if len(policies) != len(move_texts):
            self.record_error(
                "policy_move_length_mismatch",
                path,
                row_number,
                f"moves={len(move_texts)} policies={len(policies)}",
            )
            row_labels_usable = False
            row_all_moves_labeled = False

        board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
        parsed_moves: list[int] = []
        terminal_player: int | None = None
        search_started = False
        current_turn_policy_valid: list[bool] = []
        turn_start_key: bytes | None = None

        for move_index, coordinate in enumerate(move_texts):
            if terminal_player is not None:
                self.record_error(
                    "moves_after_terminal",
                    path,
                    row_number,
                    f"move_index={move_index} coordinate={coordinate}",
                )
                row_hard_valid = False
                break
            move = coordinate_to_index(coordinate)
            if move is None:
                self.record_error(
                    "invalid_coordinate",
                    path,
                    row_number,
                    f"move_index={move_index} coordinate={coordinate}",
                )
                row_hard_valid = False
                break
            row_index, column_index = divmod(move, BOARD_SIZE)
            if board[row_index, column_index] != 0:
                self.record_error(
                    "occupied_move",
                    path,
                    row_number,
                    f"move_index={move_index} coordinate={coordinate}",
                )
                row_hard_valid = False
                break

            player = player_at_move(move_index)
            if is_turn_start(move_index):
                if current_turn_policy_valid:
                    self._finish_turn_labels(current_turn_policy_valid, turn_start_key)
                current_turn_policy_valid = []
                turn_start_key = state_key(board, player, canonical_d4=False)
                self.turn_states.add(turn_start_key)
                self.turn_states_d4.add(state_key(board, player, canonical_d4=True))
                self.counts["turn_states"] += 1
                for threshold in (32, 64, 96):
                    if move_index <= threshold:
                        self.counts[f"turn_states_at_most_{threshold}_stones"] += 1

            policy_text = policies[move_index] if move_index < len(policies) else ""
            policy_values, policy_errors = parse_sparse_policy(policy_text)
            for error_name, amount in policy_errors.items():
                self.errors[error_name] += amount
            policy_valid = False
            if not policy_values:
                self.counts["missing_policy_labels"] += 1
                row_all_moves_labeled = False
                if search_started:
                    self.record_error(
                        "missing_policy_after_search_started",
                        path,
                        row_number,
                        f"move_index={move_index}",
                    )
                    self.counts["invalid_policy_labels"] += 1
                    row_labels_usable = False
            else:
                search_started = True
                total = sum(policy_values.values())
                occupied_mass = sum(
                    probability
                    for index, probability in policy_values.items()
                    if board.flat[index] != 0
                )
                chosen_mass = policy_values.get(move, 0.0)
                normalized = abs(total - 1.0) <= self.policy_tolerance
                legal = occupied_mass <= self.policy_tolerance
                chosen_supported = chosen_mass > 0
                parsed_cleanly = not policy_errors
                policy_valid = (
                    parsed_cleanly
                    and total > 0
                    and normalized
                    and legal
                    and chosen_supported
                )
                if not normalized:
                    self.record_error(
                        "policy_not_normalized",
                        path,
                        row_number,
                        f"move_index={move_index} total={total:.6f}",
                    )
                if not legal:
                    self.record_error(
                        "policy_mass_on_occupied",
                        path,
                        row_number,
                        f"move_index={move_index} occupied_mass={occupied_mass:.6f}",
                    )
                if not chosen_supported:
                    self.record_error(
                        "chosen_move_missing_from_policy",
                        path,
                        row_number,
                        f"move_index={move_index} move={move}",
                    )
                if not policy_valid:
                    self.counts["invalid_policy_labels"] += 1
                    row_labels_usable = False
                else:
                    probabilities = [value / total for value in policy_values.values() if value > 0]
                    entropy = -sum(value * math.log(value) for value in probabilities)
                    self.policy_entropies.append(entropy)
                    self.policy_supports.append(float(len(probabilities)))
                    self.policy_top1.append(max(probabilities))
                    self.chosen_policy_mass.append(chosen_mass / total)
                    self.counts["valid_policy_labels"] += 1
                    row_valid_policy_labels += 1
            current_turn_policy_valid.append(policy_valid)

            board[row_index, column_index] = player
            parsed_moves.append(move)
            if check_win(board, move, player):
                terminal_player = player
                self.counts["terminal_games"] += 1
                if move_index > 0 and move_index % 2 == 1:
                    self.counts["terminal_singleton_turns"] += 1

        if current_turn_policy_valid:
            self._finish_turn_labels(current_turn_policy_valid, turn_start_key)

        if parsed_moves:
            fingerprint = tuple(parsed_moves)
            if fingerprint in self.game_fingerprints:
                self.counts["duplicate_games"] += 1
            else:
                self.game_fingerprints.add(fingerprint)
            self.openings.add(fingerprint[:5])

        if terminal_player is None and len(parsed_moves) == BOARD_POINTS:
            terminal_player = 0
        if terminal_player is None:
            self.record_error(
                "nonterminal_game",
                path,
                row_number,
                f"parsed_moves={len(parsed_moves)}",
            )
            row_hard_valid = False
        expected_player = WINNER_TO_PLAYER.get(winner_name)
        if terminal_player is not None and expected_player is not None and terminal_player != expected_player:
            self.record_error(
                "recorded_winner_mismatch",
                path,
                row_number,
                f"recorded={winner_name} replayed={PLAYER_TO_WINNER[terminal_player]}",
            )
            row_hard_valid = False
        row_hard_valid = (
            row_hard_valid
            and len(parsed_moves) == len(move_texts)
            and terminal_player is not None
            and expected_player is not None
            and terminal_player == expected_player
        )
        if row_hard_valid:
            self.counts["valid_games"] += 1
            self.valid_winners[winner_name] += 1
            self.valid_game_lengths.append(float(len(parsed_moves)))
        if row_hard_valid and row_labels_usable and row_valid_policy_labels > 0:
            self.counts["training_usable_games"] += 1
        if row_hard_valid and row_labels_usable and row_all_moves_labeled:
            self.counts["fully_labeled_valid_games"] += 1

    def _finish_turn_labels(self, valid_labels: list[bool], turn_key: bytes | None):
        if not valid_labels:
            return
        if len(valid_labels) == 2:
            self.counts["completed_pair_turns"] += 1
        if all(valid_labels):
            self.counts["fully_labeled_turns"] += 1
            if turn_key is not None:
                self.labeled_turn_states.add(turn_key)
            if len(valid_labels) == 2:
                self.counts["legacy_pair_labeled_turns"] += 1

    def report(self) -> dict:
        games = int(self.counts["rows"])
        valid_games = int(self.counts["valid_games"])
        turn_states = int(self.counts["turn_states"])

        def mean(values):
            return float(statistics.fmean(values)) if values else 0.0

        def rate(value, total):
            return float(value / total) if total else 0.0

        winner_counts = {
            name: int(self.winners[name]) for name in ("black", "white", "draw")
        }
        valid_winner_counts = {
            name: int(self.valid_winners[name]) for name in ("black", "white", "draw")
        }
        return {
            "schema_version": 1,
            "files": {name: int(value) for name, value in sorted(self.file_rows.items())},
            "games": games,
            "valid_games": valid_games,
            "valid_game_rate": rate(valid_games, games),
            "fully_labeled_valid_games": int(self.counts["fully_labeled_valid_games"]),
            "training_usable_games": int(self.counts["training_usable_games"]),
            "winners": winner_counts,
            "valid_winners": valid_winner_counts,
            "black_win_rate_valid": rate(valid_winner_counts["black"], valid_games),
            "white_win_rate_valid": rate(valid_winner_counts["white"], valid_games),
            "draw_rate_valid": rate(valid_winner_counts["draw"], valid_games),
            "mean_stones_valid": mean(self.valid_game_lengths),
            "median_stones_valid": percentile(self.valid_game_lengths, 0.5),
            "p90_stones_valid": percentile(self.valid_game_lengths, 0.9),
            "p95_stones_valid": percentile(self.valid_game_lengths, 0.95),
            "duplicate_games": int(self.counts["duplicate_games"]),
            "opening_unique_ratio": rate(len(self.openings), games),
            "turn_states": turn_states,
            "unique_turn_states": len(self.turn_states),
            "unique_turn_state_rate": rate(len(self.turn_states), turn_states),
            "d4_unique_turn_states": len(self.turn_states_d4),
            "d4_unique_turn_state_rate": rate(len(self.turn_states_d4), turn_states),
            "stone_capacity_coverage": {
                str(threshold): {
                    "turn_states": int(
                        self.counts[f"turn_states_at_most_{threshold}_stones"]
                    ),
                    "rate": rate(
                        self.counts[f"turn_states_at_most_{threshold}_stones"],
                        turn_states,
                    ),
                }
                for threshold in (32, 64, 96)
            },
            "completed_pair_turns": int(self.counts["completed_pair_turns"]),
            "terminal_singleton_turns": int(self.counts["terminal_singleton_turns"]),
            "fully_labeled_turns": int(self.counts["fully_labeled_turns"]),
            "unique_fully_labeled_turn_states": len(self.labeled_turn_states),
            "legacy_pair_labeled_turns": int(self.counts["legacy_pair_labeled_turns"]),
            "native_pair_visit_targets": 0,
            "native_pair_target_note": (
                "Legacy files contain sequential first/second policies, not a joint "
                "unordered-pair visit distribution. They can audit games but cannot "
                "serve as exact native-pair policy targets."
            ),
            "valid_policy_labels": int(self.counts["valid_policy_labels"]),
            "missing_policy_labels": int(self.counts["missing_policy_labels"]),
            "invalid_policy_labels": int(self.counts["invalid_policy_labels"]),
            "policy_entropy_mean": mean(self.policy_entropies),
            "policy_effective_actions": (
                math.exp(mean(self.policy_entropies)) if self.policy_entropies else 0.0
            ),
            "policy_support_mean": mean(self.policy_supports),
            "policy_top1_mass_mean": mean(self.policy_top1),
            "chosen_move_mass_mean": mean(self.chosen_policy_mass),
            "errors": {name: int(value) for name, value in sorted(self.errors.items())},
            "error_examples": self.error_examples,
        }


def expand_inputs(inputs: list[str]) -> list[Path]:
    paths: list[Path] = []
    for raw_path in inputs:
        path = Path(raw_path).expanduser().resolve()
        if path.is_dir():
            paths.extend(sorted(path.glob("*.csv")))
        elif path.is_file():
            paths.append(path)
        else:
            raise FileNotFoundError(path)
    unique_paths = []
    seen = set()
    for path in paths:
        if path not in seen:
            seen.add(path)
            unique_paths.append(path)
    if not unique_paths:
        raise ValueError("no CSV files found")
    return unique_paths


def audit_paths(
    inputs: list[str] | tuple[str, ...],
    policy_tolerance: float = 0.02,
    max_examples: int = 20,
) -> dict:
    csv.field_size_limit(2**31 - 1)
    audit = GameAudit(policy_tolerance=policy_tolerance, max_examples=max_examples)
    per_file = {}
    for path in expand_inputs(list(inputs)):
        file_audit = GameAudit(
            policy_tolerance=policy_tolerance,
            max_examples=max_examples,
        )
        with path.open(newline="", encoding="utf-8", errors="replace") as source:
            reader = csv.DictReader(source)
            required = {"moves", "winner", "policies"}
            if not required.issubset(reader.fieldnames or ()):
                audit.record_error(
                    "invalid_header",
                    path,
                    1,
                    f"fields={reader.fieldnames}",
                )
                continue
            for row_number, row in enumerate(reader, start=2):
                audit.audit_row(row, path, row_number)
                file_audit.audit_row(row, path, row_number)
        per_file[str(path)] = file_audit.report()
    report = audit.report()
    report["per_file"] = per_file
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Strictly audit Connect6 self-play CSVs")
    parser.add_argument("inputs", nargs="+", help="CSV files or directories")
    parser.add_argument("--output", default="", help="optional JSON report path")
    parser.add_argument("--policy-tolerance", type=float, default=0.02)
    parser.add_argument("--max-examples", type=int, default=20)
    parser.add_argument("--fail-on-invalid", action="store_true")
    arguments = parser.parse_args()
    report = audit_paths(
        arguments.inputs,
        policy_tolerance=arguments.policy_tolerance,
        max_examples=arguments.max_examples,
    )
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    if arguments.output:
        output_path = Path(arguments.output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = output_path.with_suffix(output_path.suffix + ".tmp")
        temporary.write_text(payload + "\n", encoding="utf-8")
        temporary.replace(output_path)
    print(payload)
    if arguments.fail_on_invalid and (
        report["valid_games"] != report["games"]
        or report["training_usable_games"] != report["games"]
    ):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
