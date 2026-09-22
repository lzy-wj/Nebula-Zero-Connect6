"""Split an isolated self-play CSV into deterministic train/validation replay files."""

import argparse
import csv
import json
import os
import random


def read_rows(path):
    with open(path, newline="", encoding="utf-8", errors="replace") as source:
        reader = csv.reader(source)
        header = next(reader, None)
        if not header or header[:3] != ["moves", "winner", "policies"]:
            raise ValueError("input must use the Connect6 replay CSV schema")
        rows = [row for row in reader if len(row) >= 3]
    if not rows:
        raise ValueError("input contains no replay games")
    return header, rows


def atomic_write(path, header, rows):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    temporary = f"{path}.tmp"
    with open(temporary, "w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(header)
        writer.writerows(rows)
    os.replace(temporary, path)


def split_replay(input_path, output_dir, generation, validation_games, seed):
    header, rows = read_rows(input_path)
    if not 0 < validation_games < len(rows):
        raise ValueError("validation games must be between zero and total games")
    indices = list(range(len(rows)))
    random.Random(seed).shuffle(indices)
    validation_indices = set(indices[:validation_games])
    training = [row for index, row in enumerate(rows) if index not in validation_indices]
    validation = [row for index, row in enumerate(rows) if index in validation_indices]
    prefix = f"gen_{generation:04d}"
    train_path = os.path.join(output_dir, f"{prefix}_train.csv")
    validation_path = os.path.join(output_dir, f"{prefix}_validation.csv")
    atomic_write(train_path, header, training)
    atomic_write(validation_path, header, validation)
    return {
        "input": os.path.abspath(input_path),
        "train": os.path.abspath(train_path),
        "validation": os.path.abspath(validation_path),
        "train_games": len(training),
        "validation_games": len(validation),
        "seed": seed,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Prepare isolated self-play for vNext training",
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--generation", type=int, required=True)
    parser.add_argument("--validation-games", type=int, default=400)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    if args.generation < 0:
        parser.error("--generation must not be negative")
    result = split_replay(
        args.input,
        args.output_dir,
        args.generation,
        args.validation_games,
        args.seed,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
