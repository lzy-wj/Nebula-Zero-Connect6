"""Memory-mapped fixed-width supervision data for V4 training."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from experiments.pair_native.train import (
    collate_turns,
    transform_board,
    transform_index,
)


ARRAY_NAMES = (
    "board_black",
    "board_white",
    "player",
    "stones_before",
    "chosen_a",
    "chosen_b",
    "policy_a",
    "policy_b",
    "policy_probability",
    "policy_length",
    "value_outcome",
    "root_value",
    "remaining_turns",
)


class MMapTurnDataset(Dataset):
    def __init__(
        self,
        cache_dir: str | Path,
        split: str,
        training: bool,
        indices: np.ndarray | None = None,
    ):
        self.cache_dir = Path(cache_dir).expanduser().resolve()
        self.metadata = json.loads(
            (self.cache_dir / "metadata.json").read_text(encoding="utf-8")
        )
        self.arrays = {
            name: np.load(
                self.cache_dir / split / f"{name}.npy",
                mmap_mode="r",
            )
            for name in ARRAY_NAMES
        }
        lengths = {len(array) for array in self.arrays.values()}
        if len(lengths) != 1:
            raise ValueError("mmap cache arrays have inconsistent lengths")
        policy_width = int(self.metadata.get("policy_width", 0))
        policy_shapes = {
            self.arrays[name].shape[1]
            for name in ("policy_a", "policy_b", "policy_probability")
        }
        if (
            policy_width < 1
            or policy_shapes != {policy_width}
            or np.any(self.arrays["policy_length"] > policy_width)
        ):
            raise ValueError("mmap cache policy width does not match metadata")
        self.rows = lengths.pop()
        self.training = bool(training)
        self.indices = None if indices is None else np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        return self.rows if self.indices is None else len(self.indices)

    def __getitem__(self, item: int):
        index = item if self.indices is None else int(self.indices[item])
        black = np.unpackbits(
            self.arrays["board_black"][index], bitorder="little"
        )[:361].reshape(19, 19)
        white = np.unpackbits(
            self.arrays["board_white"][index], bitorder="little"
        )[:361].reshape(19, 19)
        board = black.astype(np.int8) - white.astype(np.int8)
        transform = random.randrange(8) if self.training else 0
        board = transform_board(board, transform)
        player = int(self.arrays["player"][index])
        features = np.stack(
            (
                board == player,
                board == -player,
                board == 0,
                np.full_like(board, player == 1, dtype=bool),
                np.zeros_like(board, dtype=bool),
            ),
            axis=0,
        ).astype(np.float32)
        length = int(self.arrays["policy_length"][index])
        policy_a = self.arrays["policy_a"][index].copy()
        policy_b = self.arrays["policy_b"][index].copy()
        for position in range(length):
            policy_a[position] = transform_index(int(policy_a[position]), transform)
            policy_b[position] = transform_index(int(policy_b[position]), transform)
            if policy_b[position] >= 0 and policy_a[position] > policy_b[position]:
                policy_a[position], policy_b[position] = (
                    policy_b[position],
                    policy_a[position],
                )
        chosen_a = transform_index(int(self.arrays["chosen_a"][index]), transform)
        chosen_b = transform_index(int(self.arrays["chosen_b"][index]), transform)
        if chosen_b >= 0 and chosen_a > chosen_b:
            chosen_a, chosen_b = chosen_b, chosen_a
        return {
            "features": torch.from_numpy(features),
            "policy_a": torch.from_numpy(policy_a[:length]).long(),
            "policy_b": torch.from_numpy(policy_b[:length]).long(),
            "policy_probability": torch.from_numpy(
                self.arrays["policy_probability"][index, :length].copy()
            ),
            "chosen_a": chosen_a,
            "chosen_b": chosen_b,
            "value": float(self.arrays["value_outcome"][index]),
            "remaining_turns": int(self.arrays["remaining_turns"][index]),
            "stones_before": int(self.arrays["stones_before"][index]),
        }


def collate_mmap_turns(samples: list[dict]) -> dict[str, torch.Tensor]:
    return collate_turns(samples)
