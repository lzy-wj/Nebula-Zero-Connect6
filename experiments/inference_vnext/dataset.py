"""Replay datasets for inference-first exact and native-pair training."""

import csv
import os
import random
import re
import sys
from collections import Counter
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset


BOARD_SIZE = 19
BOARD_POINTS = BOARD_SIZE * BOARD_SIZE
REPLAY_PATTERN = re.compile(r"^gen_(\d+)_(train|validation)\.csv$")


@dataclass(frozen=True, slots=True)
class ReplayGame:
    moves: tuple
    winner: int
    policies: tuple
    generation: int


def coordinate_to_index(coordinate):
    coordinate = coordinate.strip().lower()
    if len(coordinate) < 2 or not "a" <= coordinate[0] <= "s":
        return None
    try:
        row = int(coordinate[1:]) - 1
    except ValueError:
        return None
    column = ord(coordinate[0]) - ord("a")
    if not 0 <= row < BOARD_SIZE:
        return None
    return row * BOARD_SIZE + column


def player_at_move(move_index):
    if move_index == 0:
        return 1
    return -1 if ((move_index + 1) // 2) % 2 else 1


def _normalize_data_roots(data_roots):
    if isinstance(data_roots, (str, bytes, os.PathLike)):
        return [os.fspath(data_roots)]
    roots = [os.fspath(root) for root in data_roots]
    if not roots:
        raise ValueError("at least one replay root is required")
    return roots


def discover_replay_files(data_root, recent_generations=0):
    partitions = {"train": [], "validation": []}
    generations = set()
    for root in _normalize_data_roots(data_root):
        for name in os.listdir(root):
            match = REPLAY_PATTERN.match(name)
            if not match:
                continue
            generation = int(match.group(1))
            partition = match.group(2)
            partitions[partition].append(
                (generation, os.path.join(root, name))
            )
            generations.add(generation)

    if recent_generations > 0:
        selected = set(sorted(generations)[-recent_generations:])
        for partition in partitions:
            partitions[partition] = [
                item for item in partitions[partition] if item[0] in selected
            ]
    for partition in partitions:
        partitions[partition].sort(reverse=True)
    return partitions


def _load_partition(files, seen_games, max_games, statistics, partition):
    records = []
    winner_map = {"black": 1, "white": -1, "draw": 0}
    for generation, path in files:
        with open(path, newline="", encoding="utf-8", errors="replace") as source:
            reader = csv.reader(source)
            header = next(reader, None)
            if not header or header[:3] != ["moves", "winner", "policies"]:
                statistics[f"{partition}_bad_header_files"] += 1
                continue
            for row in reader:
                statistics[f"{partition}_rows_seen"] += 1
                if len(row) < 3:
                    statistics[f"{partition}_short_rows"] += 1
                    continue
                winner = winner_map.get(row[1].strip().lower())
                moves = tuple(
                    coordinate_to_index(value)
                    for value in row[0].split(",")
                    if value.strip()
                )
                policies = tuple(row[2].split("|")) if row[2] else ()
                if winner is None:
                    statistics[f"{partition}_bad_winner"] += 1
                    continue
                if not moves or any(move is None for move in moves):
                    statistics[f"{partition}_bad_moves"] += 1
                    continue
                if len(moves) != len(set(moves)):
                    statistics[f"{partition}_repeated_moves"] += 1
                    continue
                if len(policies) != len(moves):
                    statistics[f"{partition}_policy_length"] += 1
                    continue
                if moves in seen_games:
                    statistics[f"{partition}_duplicates"] += 1
                    continue
                seen_games.add(moves)
                records.append(ReplayGame(
                    moves=moves,
                    winner=winner,
                    policies=policies,
                    generation=generation,
                ))
                if max_games and len(records) >= max_games:
                    return records
    return records


def load_replay_records(
    data_root,
    recent_generations=0,
    max_train_games=0,
    max_validation_games=0,
):
    csv.field_size_limit(sys.maxsize)
    files = discover_replay_files(data_root, recent_generations)
    statistics = Counter()
    seen_games = set()
    # Validation wins duplicate ownership so a repeated game cannot leak into train.
    validation = _load_partition(
        files["validation"],
        seen_games,
        max_validation_games,
        statistics,
        "validation",
    )
    training = _load_partition(
        files["train"],
        seen_games,
        max_train_games,
        statistics,
        "train",
    )
    statistics["train_games"] = len(training)
    statistics["validation_games"] = len(validation)
    return training, validation, dict(statistics)


def parse_sparse_policy(policy_string, fallback_move, occupied=None):
    policy = np.zeros(BOARD_POINTS, dtype=np.float32)
    for item in policy_string.split(";"):
        if ":" not in item:
            continue
        index_string, probability_string = item.split(":", 1)
        try:
            index = int(index_string)
            probability = float(probability_string)
        except ValueError:
            continue
        if 0 <= index < BOARD_POINTS and np.isfinite(probability) and probability >= 0:
            policy[index] = probability
    if occupied is not None:
        policy[occupied] = 0.0
    total = float(policy.sum())
    if total <= 0:
        policy[int(fallback_move)] = 1.0
    else:
        policy /= total
    return policy


def make_features(board, current_player, second_stone):
    features = np.empty((5, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    features[0] = board == current_player
    features[1] = board == -current_player
    features[2] = board == 0
    features[3].fill(1.0 if current_player == 1 else 0.0)
    features[4].fill(1.0 if second_stone else 0.0)
    return features


def transform_board_policy(board, policies, moves, transform):
    rotation = transform % 4
    reflect = transform >= 4
    board = board.reshape(BOARD_SIZE, BOARD_SIZE)
    policy_boards = [policy.reshape(BOARD_SIZE, BOARD_SIZE) for policy in policies]
    move_boards = []
    for move in moves:
        one_hot = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.uint8)
        one_hot.flat[int(move)] = 1
        move_boards.append(one_hot)
    if reflect:
        board = np.fliplr(board)
        policy_boards = [np.fliplr(policy) for policy in policy_boards]
        move_boards = [np.fliplr(move) for move in move_boards]
    if rotation:
        board = np.rot90(board, rotation)
        policy_boards = [np.rot90(policy, rotation) for policy in policy_boards]
        move_boards = [np.rot90(move, rotation) for move in move_boards]
    return (
        np.ascontiguousarray(board),
        [np.ascontiguousarray(policy).reshape(-1) for policy in policy_boards],
        [int(np.argmax(move)) for move in move_boards],
    )


class _ReplayPositionDataset(Dataset):
    def __init__(self, records, training, positions_per_game, maximum_stones):
        if not records:
            raise ValueError("replay records must not be empty")
        self.records = records
        self.training = bool(training)
        self.positions_per_game = int(positions_per_game)
        self.maximum_stones = min(int(maximum_stones), BOARD_POINTS - 1)
        self.max_generation = max(record.generation for record in records)
        self.eligible = [self._eligible_indices(record) for record in records]
        if any(not indices for indices in self.eligible):
            filtered = [
                (record, indices)
                for record, indices in zip(self.records, self.eligible)
                if indices
            ]
            self.records = [item[0] for item in filtered]
            self.eligible = [item[1] for item in filtered]
        if not self.records:
            raise ValueError("no replay records contain eligible positions")

    def _eligible_indices(self, record):
        raise NotImplementedError

    def __len__(self):
        return len(self.records) * self.positions_per_game

    def record_weights(self):
        weights = []
        generation_scale = max(1, self.max_generation)
        for record in self.records:
            progress = record.generation / generation_scale
            weight = 0.25 + 0.75 * progress * progress
            if record.winner == -1:
                weight *= 1.15
            elif record.winner == 0:
                weight *= 2.0
            weights.extend([weight] * self.positions_per_game)
        weights = torch.tensor(weights, dtype=torch.double)
        return weights / weights.mean()

    def select_move_index(self, record_index, position_slot):
        candidates = self.eligible[record_index]
        if self.training:
            return candidates[int(np.random.randint(0, len(candidates)))]
        fraction = (position_slot + 1) / (self.positions_per_game + 1)
        return candidates[min(len(candidates) - 1, int(fraction * len(candidates)))]


class ExactPositionDataset(_ReplayPositionDataset):
    def _eligible_indices(self, record):
        limit = min(len(record.moves), self.maximum_stones + 1)
        return tuple(index for index in range(limit) if record.policies[index])

    def __getitem__(self, index):
        record_index = index // self.positions_per_game
        position_slot = index % self.positions_per_game
        record = self.records[record_index]
        move_index = self.select_move_index(record_index, position_slot)
        current_player = player_at_move(move_index)
        board = np.zeros(BOARD_POINTS, dtype=np.int8)
        for previous in range(move_index):
            board[record.moves[previous]] = player_at_move(previous)
        occupied = board != 0
        policy = parse_sparse_policy(
            record.policies[move_index],
            record.moves[move_index],
            occupied,
        )
        if self.training:
            board, (policy,), _ = transform_board_policy(
                board,
                (policy,),
                (),
                int(np.random.randint(0, 8)),
            )
        board = board.reshape(BOARD_SIZE, BOARD_SIZE)
        features = make_features(
            board,
            current_player,
            second_stone=move_index > 0 and move_index % 2 == 0,
        )
        value = np.float32(record.winner * current_player)
        return (
            torch.from_numpy(features),
            torch.from_numpy(np.ascontiguousarray(policy, dtype=np.float32)),
            torch.tensor(value, dtype=torch.float32),
        )


class PairPositionDataset(_ReplayPositionDataset):
    def _eligible_indices(self, record):
        limit = min(len(record.moves) - 1, self.maximum_stones + 1)
        return tuple(
            index
            for index in range(1, limit, 2)
            if record.policies[index] and record.policies[index + 1]
        )

    def __getitem__(self, index):
        record_index = index // self.positions_per_game
        position_slot = index % self.positions_per_game
        record = self.records[record_index]
        move_index = self.select_move_index(record_index, position_slot)
        current_player = player_at_move(move_index)
        first_move = record.moves[move_index]
        second_move = record.moves[move_index + 1]
        board = np.zeros(BOARD_POINTS, dtype=np.int8)
        for previous in range(move_index):
            board[record.moves[previous]] = player_at_move(previous)
        occupied = board != 0
        first_policy = parse_sparse_policy(
            record.policies[move_index],
            first_move,
            occupied,
        )
        occupied_after_first = occupied.copy()
        occupied_after_first[first_move] = True
        second_policy = parse_sparse_policy(
            record.policies[move_index + 1],
            second_move,
            occupied_after_first,
        )
        if self.training:
            board, (first_policy, second_policy), (first_move,) = transform_board_policy(
                board,
                (first_policy, second_policy),
                (first_move,),
                int(np.random.randint(0, 8)),
            )
        board = board.reshape(BOARD_SIZE, BOARD_SIZE)
        features = make_features(board, current_player, second_stone=False)
        value = np.float32(record.winner * current_player)
        return (
            torch.from_numpy(features),
            torch.from_numpy(np.ascontiguousarray(first_policy, dtype=np.float32)),
            torch.tensor(first_move, dtype=torch.long),
            torch.from_numpy(np.ascontiguousarray(second_policy, dtype=np.float32)),
            torch.tensor(value, dtype=torch.float32),
        )


def seed_data_worker(worker_id):
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)
