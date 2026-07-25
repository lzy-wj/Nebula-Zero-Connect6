"""旧自对弈棋谱到 NebulaNet V3 训练位置的可靠数据管线。"""

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
GENERATION_PATTERN = re.compile(r'^(?:eval_data_|gen_)(\d+)\.csv$')


@dataclass(frozen=True, slots=True)
class GameRecord:
    moves: tuple
    winner: int
    policies: tuple
    generation: int
    is_evaluation: bool


def coordinate_to_index(coordinate):
    coordinate = coordinate.strip().lower()
    if len(coordinate) < 2 or not ('a' <= coordinate[0] <= 's'):
        return None
    if not coordinate[1:].isdigit():
        return None
    row = int(coordinate[1:]) - 1
    column = ord(coordinate[0]) - ord('a')
    if not (0 <= row < BOARD_SIZE and 0 <= column < BOARD_SIZE):
        return None
    return row * BOARD_SIZE + column


def player_at_move(move_index):
    """返回逐子序列在 move_index 时轮到的颜色：黑 1，白 -1。"""
    if move_index == 0:
        return 1
    turn_index = (move_index + 1) // 2
    return -1 if turn_index % 2 == 1 else 1


def discover_data_files(data_root):
    candidates = []
    for name in os.listdir(data_root):
        match = GENERATION_PATTERN.match(name)
        if not match:
            continue
        generation = int(match.group(1))
        is_evaluation = name.startswith('eval_data_')
        candidates.append((generation, is_evaluation, os.path.join(data_root, name)))

    # 最近代和门控对局优先；跨文件重复时保留信息质量更高的新副本。
    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return candidates


def load_game_records(
    data_root,
    validation_start_generation=796,
    max_files=None,
    max_games=None,
):
    """加载并去重棋谱；按代切分可避免同一演化阶段泄漏到验证集。"""
    csv.field_size_limit(sys.maxsize)
    files = discover_data_files(data_root)
    if max_files is not None:
        files = files[:max_files]

    training_records = []
    validation_records = []
    seen_games = set()
    statistics = Counter()
    winner_map = {'black': 1, 'white': -1, 'draw': 0}

    for generation, is_evaluation, path in files:
        with open(path, 'r', encoding='utf-8', errors='replace', newline='') as file:
            reader = csv.reader(file)
            header = next(reader, None)
            if not header or header[:3] != ['moves', 'winner', 'policies']:
                statistics['bad_header_files'] += 1
                continue

            for row in reader:
                statistics['rows_seen'] += 1
                if len(row) < 3:
                    statistics['short_rows'] += 1
                    continue
                winner = winner_map.get(row[1].strip().lower())
                if winner is None:
                    statistics['bad_winner_rows'] += 1
                    continue

                moves = tuple(
                    coordinate_to_index(value)
                    for value in row[0].split(',')
                    if value.strip()
                )
                if not moves or any(move is None for move in moves):
                    statistics['bad_move_rows'] += 1
                    continue
                if len(moves) != len(set(moves)):
                    statistics['repeated_move_rows'] += 1
                    continue

                policies = tuple(row[2].split('|')) if row[2] else ()
                if len(policies) != len(moves) or any(not value for value in policies):
                    statistics['policy_length_rows'] += 1
                    continue
                if moves in seen_games:
                    statistics['duplicate_rows'] += 1
                    continue
                seen_games.add(moves)

                record = GameRecord(
                    moves=moves,
                    winner=winner,
                    policies=policies,
                    generation=generation,
                    is_evaluation=is_evaluation,
                )
                if generation >= validation_start_generation:
                    validation_records.append(record)
                    statistics['validation_games'] += 1
                else:
                    training_records.append(record)
                    statistics['training_games'] += 1
                statistics[f'winner_{row[1].strip().lower()}'] += 1

                if max_games is not None and len(seen_games) >= max_games:
                    return training_records, validation_records, dict(statistics)

    return training_records, validation_records, dict(statistics)


def parse_sparse_policy(policy_string):
    policy = np.zeros(BOARD_POINTS, dtype=np.float32)
    for item in policy_string.split(';'):
        if ':' not in item:
            continue
        index_string, probability_string = item.split(':', 1)
        try:
            index = int(index_string)
            probability = float(probability_string)
        except ValueError:
            continue
        if 0 <= index < BOARD_POINTS and np.isfinite(probability) and probability >= 0:
            policy[index] = probability
    return policy


class SelfPlayPositionDataset(Dataset):
    """每局每轮采样少量位置，避免把 700 万局面全部驻留内存。"""

    def __init__(self, records, training=True, positions_per_game=1):
        if not records:
            raise ValueError('棋谱列表不能为空')
        if positions_per_game <= 0:
            raise ValueError('positions_per_game 必须大于 0')
        self.records = records
        self.training = bool(training)
        self.positions_per_game = int(positions_per_game)
        self.max_generation = max(record.generation for record in records)

    def __len__(self):
        return len(self.records) * self.positions_per_game

    def record_weights(self):
        """温和偏向新代与稀有结果，不复制整局数据、不二次加权 loss。"""
        weights = []
        generation_scale = max(1, self.max_generation)
        for record in self.records:
            progress = record.generation / generation_scale
            weight = 0.25 + 0.75 * progress * progress
            if record.is_evaluation:
                weight *= 1.10
            if record.winner == -1:
                weight *= 1.15
            elif record.winner == 0:
                weight *= 2.0
            weights.extend([weight] * self.positions_per_game)
        weights = torch.tensor(weights, dtype=torch.double)
        return weights / weights.mean()

    def _select_move_index(self, record, position_slot):
        if self.training:
            return int(np.random.randint(0, len(record.moves)))
        # 验证时固定取均匀分布的位置，保证跨 epoch 指标可比较。
        fraction = (position_slot + 1) / (self.positions_per_game + 1)
        return min(len(record.moves) - 1, int(fraction * len(record.moves)))

    def __getitem__(self, index):
        record_index = index // self.positions_per_game
        position_slot = index % self.positions_per_game
        record = self.records[record_index]
        move_index = self._select_move_index(record, position_slot)
        current_player = player_at_move(move_index)

        board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
        if move_index:
            prefix = np.asarray(record.moves[:move_index], dtype=np.int64)
            players = np.fromiter(
                (player_at_move(index) for index in range(move_index)),
                dtype=np.int8,
                count=move_index,
            )
            board.reshape(-1)[prefix] = players

        policy = parse_sparse_policy(record.policies[move_index])
        occupied = board.reshape(-1) != 0
        policy[occupied] = 0.0
        policy_sum = float(policy.sum())
        if policy_sum <= 0:
            policy[record.moves[move_index]] = 1.0
        else:
            policy /= policy_sum

        if self.training:
            transform = int(np.random.randint(0, 8))
            rotation = transform % 4
            if transform >= 4:
                board = np.fliplr(board)
                policy = np.fliplr(policy.reshape(BOARD_SIZE, BOARD_SIZE)).reshape(-1)
            if rotation:
                board = np.rot90(board, rotation).copy()
                policy = np.rot90(
                    policy.reshape(BOARD_SIZE, BOARD_SIZE),
                    rotation,
                ).reshape(-1).copy()

        board = np.ascontiguousarray(board)
        policy = np.ascontiguousarray(policy, dtype=np.float32)
        features = np.empty((5, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
        features[0] = board == current_player
        features[1] = board == -current_player
        features[2] = board == 0
        features[3].fill(1.0 if current_player == 1 else 0.0)
        is_second_stone = move_index > 0 and move_index % 2 == 0
        features[4].fill(1.0 if is_second_stone else 0.0)

        value = np.float32(record.winner * current_player)
        return (
            torch.from_numpy(features),
            torch.from_numpy(policy),
            torch.tensor(value, dtype=torch.float32),
        )


def seed_data_worker(worker_id):
    worker_seed = torch.initial_seed() % (2 ** 32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)
