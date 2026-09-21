import csv

import torch

from experiments.inference_vnext.dataset import (
    ExactPositionDataset,
    PairPositionDataset,
    ReplayGame,
    load_replay_records,
)


def make_record(offset=0):
    moves = tuple(range(offset, offset + 8))
    policies = tuple(f"{move}:1.0" for move in moves)
    return ReplayGame(
        moves=moves,
        winner=1,
        policies=policies,
        generation=10,
    )


def test_exact_dataset_respects_maximum_stones():
    dataset = ExactPositionDataset(
        [make_record()],
        training=False,
        positions_per_game=1,
        maximum_stones=4,
    )

    features, policy, value = dataset[0]

    assert features.shape == (5, 19, 19)
    assert policy.shape == (361,)
    assert torch.isclose(policy.sum(), torch.tensor(1.0))
    assert int((features[0] + features[1]).sum()) <= 4
    assert value.item() in (-1.0, 1.0)


def test_pair_dataset_returns_aligned_second_target():
    dataset = PairPositionDataset(
        [make_record()],
        training=False,
        positions_per_game=1,
        maximum_stones=6,
    )

    features, first_policy, first_move, second_policy, value = dataset[0]

    occupied = (features[0] + features[1]).flatten().bool()
    assert first_move.item() % 2 == 1
    assert not occupied[first_move]
    assert first_policy[first_move].item() == 1.0
    assert second_policy[first_move + 1].item() == 1.0
    assert value.item() == 1.0


def write_game(path, moves):
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(("moves", "winner", "policies", "bonuses"))
        coordinates = [f"{chr(ord('a') + move)}1" for move in moves]
        policies = [f"{move}:1.0" for move in moves]
        writer.writerow((
            ",".join(coordinates),
            "black",
            "|".join(policies),
            "0.00",
        ))


def test_replay_loader_keeps_validation_partition_independent(tmp_path):
    write_game(tmp_path / "gen_0010_train.csv", (0, 1, 2, 3))
    write_game(tmp_path / "gen_0010_validation.csv", (4, 5, 6, 7))

    training, validation, statistics = load_replay_records(tmp_path)

    assert len(training) == 1
    assert len(validation) == 1
    assert statistics["train_games"] == 1
    assert statistics["validation_games"] == 1
