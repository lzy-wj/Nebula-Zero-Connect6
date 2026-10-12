"""Build the isolated turn-level Pair MCTS shared library."""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="")
    parser.add_argument("--debug", action="store_true")
    arguments = parser.parse_args()
    source = Path(__file__).with_name("pair_mcts.cpp")
    output = (
        Path(arguments.output).expanduser().resolve()
        if arguments.output
        else source.with_name("libpair_mcts.so")
    )
    temporary = output.with_suffix(output.suffix + f".tmp.{os.getpid()}")
    command = [
        "g++",
        "-shared",
        "-fPIC",
        "-std=c++17",
        "-Wall",
        "-Wextra",
        "-Wpedantic",
        "-O0" if arguments.debug else "-O3",
        "-march=native",
        "-fopenmp",
        str(source),
        "-o",
        str(temporary),
    ]
    subprocess.run(command, check=True)
    os.replace(temporary, output)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
