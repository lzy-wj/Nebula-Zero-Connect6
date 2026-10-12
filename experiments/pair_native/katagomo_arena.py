"""Color-swapped arena between native Pair MCTS and public KataGomo Connect6.

KataGomo's Connect6 GTP protocol searches one stone at a time.  The native
engine searches one unordered two-stone action per full turn.  The adapter
keeps those semantics explicit and checks the Python, native C++, and GTP
states after every completed turn.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import queue
import re
import subprocess
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

from experiments.pair_native.arena import paired_standard_error, sample_unique_openings
from experiments.pair_native.model import load_checkpoint
from experiments.pair_native.pair_mcts import PASS_MOVE, PairMCTSEngine, TorchPairEvaluator
from experiments.pair_native.pair_replay import sha256_file
from experiments.pair_native.train import parse_cpu_affinity
from reinforcement_learning.core.connect6_game import Connect6Game


BOARD_SIZE = 19
BOARD_POINTS = BOARD_SIZE * BOARD_SIZE
GTP_COLUMNS = "ABCDEFGHJKLMNOPQRST"


def to_gtp_vertex(move: int) -> str:
    if isinstance(move, (bool, np.bool_)) or not isinstance(move, (int, np.integer)):
        raise ValueError(f"invalid move index: {move!r}")
    move = int(move)
    if not 0 <= move < BOARD_POINTS:
        raise ValueError(f"move outside 19x19 board: {move}")
    row, column = divmod(move, BOARD_SIZE)
    return f"{GTP_COLUMNS[column]}{BOARD_SIZE - row}"


def from_gtp_vertex(vertex: str) -> int:
    match = re.fullmatch(r"([A-HJ-T])(\d{1,2})", str(vertex).strip().upper())
    if match is None:
        raise ValueError(f"invalid KataGomo GTP vertex: {vertex!r}")
    column = GTP_COLUMNS.find(match.group(1))
    row = BOARD_SIZE - int(match.group(2))
    if column < 0 or not 0 <= row < BOARD_SIZE:
        raise ValueError(f"KataGomo vertex outside 19x19 board: {vertex!r}")
    return row * BOARD_SIZE + column


def normalized_winner(game: Connect6Game) -> int:
    return 0 if game.winner == 2 else int(game.winner)


def verify_synchronized(game: Connect6Game, native_context) -> None:
    if native_context.total_stones != len(game.moves):
        raise RuntimeError("Python and native contexts disagree on stone count")
    if native_context.current_player != game.current_player and not game.winner:
        raise RuntimeError("Python and native contexts disagree on player to move")
    if bool(native_context.terminal) != bool(game.winner):
        raise RuntimeError("Python and native contexts disagree on terminal state")
    if game.winner and native_context.winner != normalized_winner(game):
        raise RuntimeError("Python and native contexts disagree on winner")


def verify_terminal_result(game: Connect6Game, native_context) -> None:
    """Check terminal agreement when canonical Pair order omits a last stone."""
    if not game.winner or not native_context.terminal:
        raise RuntimeError("Python and native contexts disagree on terminal state")
    if native_context.winner != normalized_winner(game):
        raise RuntimeError("Python and native contexts disagree on winner")


def _runtime_library_dirs() -> list[str]:
    """Find CUDA wheel libraries needed by the independently built binary."""
    candidates = []
    site_packages = Path(torch.__file__).resolve().parent.parent
    for relative in ("nvidia/cudnn/lib", "nvidia/cublas/lib"):
        path = site_packages / relative
        if path.is_dir():
            candidates.append(str(path))
    cuda = Path("/usr/local/cuda/lib64")
    if cuda.is_dir():
        candidates.append(str(cuda))
    return candidates


def build_override_config(
    visits: int,
    threads: int,
    batch_size: int,
    cache_power: int,
    seed: int,
) -> str:
    values = {
        "allowResignation": "false",
        "cudaDeviceToUse": 0,
        "logAllGTPCommunication": "false",
        "logSearchInfo": "false",
        "maxVisits": int(visits),
        "nnCacheSizePowerOfTwo": int(cache_power),
        "nnMaxBatchSize": int(batch_size),
        "numSearchThreads": int(threads),
        "ponderingEnabled": "false",
        "searchRandSeed": f"nebula-{int(seed)}",
    }
    return ",".join(f"{key}={value}" for key, value in values.items())


class KataGomoGTP:
    """One persistent, serialized GTP subprocess."""

    def __init__(
        self,
        engine: Path,
        model: Path,
        config: Path,
        gpu: int,
        visits: int,
        threads: int,
        batch_size: int,
        cache_power: int,
        seed: int,
        timeout_seconds: float,
        library_path: str = "",
    ):
        self.engine = engine
        self.model = model
        self.config = config
        self.gpu = int(gpu)
        self.visits = int(visits)
        self.timeout_seconds = float(timeout_seconds)
        self.override = build_override_config(
            visits,
            threads,
            batch_size,
            cache_power,
            seed,
        )
        self.library_path = library_path
        self.process: subprocess.Popen[str] | None = None
        self.responses: queue.Queue[tuple[bool, int | None, str]] = queue.Queue()
        self.stderr_tail: deque[str] = deque(maxlen=200)
        self.next_id = 1
        self.reader_threads: list[threading.Thread] = []
        self.name = ""
        self.version = ""

    def _stdout_loop(self) -> None:
        assert self.process is not None and self.process.stdout is not None
        message: list[str] = []
        for raw_line in self.process.stdout:
            line = raw_line.rstrip("\r\n")
            if line:
                message.append(line)
                continue
            if not message:
                continue
            payload = "\n".join(message).strip()
            message.clear()
            match = re.match(r"^([=?])(\d+)?\s?(.*)$", payload, re.DOTALL)
            if match is None:
                self.responses.put((False, None, f"malformed GTP response: {payload}"))
                continue
            identifier = int(match.group(2)) if match.group(2) else None
            self.responses.put((match.group(1) == "=", identifier, match.group(3).strip()))
        if message:
            self.responses.put((False, None, "unterminated GTP response: " + "\n".join(message)))
        self.responses.put((False, None, "KataGomo GTP stdout closed"))

    def _stderr_loop(self) -> None:
        assert self.process is not None and self.process.stderr is not None
        for raw_line in self.process.stderr:
            self.stderr_tail.append(raw_line.rstrip())

    def start(self) -> None:
        if self.process is not None:
            return
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = str(self.gpu)
        library_dirs = []
        if self.library_path:
            library_dirs.extend(value for value in self.library_path.split(":") if value)
        library_dirs.extend(_runtime_library_dirs())
        if environment.get("LD_LIBRARY_PATH"):
            library_dirs.extend(environment["LD_LIBRARY_PATH"].split(":"))
        environment["LD_LIBRARY_PATH"] = ":".join(dict.fromkeys(library_dirs))
        command = [
            str(self.engine),
            "gtp",
            "-model",
            str(self.model),
            "-config",
            str(self.config),
            "-override-config",
            self.override,
        ]
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=environment,
            start_new_session=True,
        )
        self.reader_threads = [
            threading.Thread(target=self._stdout_loop, daemon=True),
            threading.Thread(target=self._stderr_loop, daemon=True),
        ]
        for thread in self.reader_threads:
            thread.start()
        self.name = self.request("name")
        self.version = self.request("version")
        self.request(f"boardsize {BOARD_SIZE}")
        self.request("freestyle")
        self.request("novc")
        self.request(f"maxmoves {BOARD_POINTS}")
        self.request("firstpasswin off")
        self.request("clear_board")

    def request(self, command: str) -> str:
        if self.process is None or self.process.stdin is None:
            raise RuntimeError("KataGomo GTP process is not running")
        if self.process.poll() is not None:
            raise RuntimeError(self._failure(f"KataGomo exited with {self.process.returncode}"))
        identifier = self.next_id
        self.next_id += 1
        self.process.stdin.write(f"{identifier} {command}\n")
        self.process.stdin.flush()
        try:
            success, response_id, payload = self.responses.get(timeout=self.timeout_seconds)
        except queue.Empty as error:
            raise TimeoutError(self._failure(f"KataGomo timed out on {command!r}")) from error
        if response_id != identifier:
            raise RuntimeError(
                self._failure(
                    f"KataGomo response id mismatch: expected {identifier}, got {response_id}"
                )
            )
        if not success:
            raise RuntimeError(self._failure(f"KataGomo rejected {command!r}: {payload}"))
        return payload

    def _failure(self, message: str) -> str:
        tail = "\n".join(self.stderr_tail)
        return message if not tail else f"{message}\nKataGomo stderr tail:\n{tail}"

    def clear_board(self) -> None:
        self.request("clear_board")

    def play(self, move: int) -> None:
        self.request(f"play B {to_gtp_vertex(move)}")

    def genmove(self) -> tuple[int, float]:
        started = time.perf_counter()
        response = self.request("genmove B")
        elapsed = time.perf_counter() - started
        if response.strip().lower() in {"pass", "resign"}:
            raise RuntimeError(f"KataGomo unexpectedly returned {response!r}")
        return from_gtp_vertex(response), elapsed

    def close(self) -> None:
        process = self.process
        if process is None:
            return
        try:
            if process.poll() is None:
                self.request("quit")
        except BaseException:
            pass
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
            for stream in (process.stdin, process.stdout, process.stderr):
                if stream is not None:
                    stream.close()
            self.process = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *_):
        self.close()


def _play_native_turn(
    game: Connect6Game,
    history: list[int],
    native_engine: PairMCTSEngine,
    native_context,
    katagomo: KataGomoGTP,
    simulations: int,
    device: torch.device,
) -> tuple[float, dict[str, int]]:
    opening = not history
    started = time.perf_counter()
    native_engine.run([native_context], simulations)
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    search_stats = native_context.stats()
    first, second = native_context.choose_action(temperature=0.0)
    if opening and second != PASS_MOVE:
        raise RuntimeError("native opening search returned two stones")
    native_context.play(first, second)
    game.play(first)
    history.append(first)
    katagomo.play(first)
    if not opening:
        if game.winner:
            if second != PASS_MOVE:
                raise RuntimeError("native search played after a winning first stone")
        else:
            if second == PASS_MOVE:
                raise RuntimeError("native search returned a non-terminal singleton")
            game.play(second)
            history.append(second)
            katagomo.play(second)
    verify_synchronized(game, native_context)
    return elapsed, search_stats


def _play_katagomo_turn(
    game: Connect6Game,
    history: list[int],
    native_context,
    katagomo: KataGomoGTP,
) -> tuple[float, int]:
    opening = not history
    first, elapsed = katagomo.genmove()
    if first in history:
        raise RuntimeError(f"KataGomo returned occupied point {first}")
    game.play(first)
    history.append(first)
    searches = 1
    if opening or game.winner:
        native_context.play(first, PASS_MOVE)
        verify_synchronized(game, native_context)
        return elapsed, searches

    second, second_elapsed = katagomo.genmove()
    elapsed += second_elapsed
    searches += 1
    if second in history:
        raise RuntimeError(f"KataGomo returned occupied point {second}")
    game.play(second)
    history.append(second)
    native_context.play(*sorted((first, second)))
    if game.winner:
        verify_terminal_result(game, native_context)
    else:
        verify_synchronized(game, native_context)
    return elapsed, searches


def _confidence_interval(score: float, standard_error: float) -> list[float]:
    return [
        max(0.0, score - 1.96 * standard_error),
        min(1.0, score + 1.96 * standard_error),
    ]


def wilson_score_interval(score: float, units: int, z_score: float = 1.96) -> list[float]:
    """Non-degenerate score interval for all-win or all-loss screening runs."""
    if units < 1:
        raise ValueError("Wilson interval requires at least one independent unit")
    denominator = 1.0 + z_score * z_score / units
    center = (score + z_score * z_score / (2.0 * units)) / denominator
    radius = (
        z_score
        * math.sqrt(
            max(score * (1.0 - score) / units, 0.0)
            + z_score * z_score / (4.0 * units * units)
        )
        / denominator
    )
    return [max(0.0, center - radius), min(1.0, center + radius)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--native-checkpoint", required=True)
    parser.add_argument("--katagomo-engine", required=True)
    parser.add_argument("--katagomo-model", required=True)
    parser.add_argument("--katagomo-config", required=True)
    parser.add_argument("--pairs", type=int, default=8)
    parser.add_argument("--native-simulations", type=int, default=400)
    parser.add_argument("--katagomo-visits", type=int, default=400)
    parser.add_argument("--top-k", type=int, default=64)
    parser.add_argument("--opening-mode", choices=("natural", "forced"), default="natural")
    parser.add_argument("--opening-radius", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260924)
    parser.add_argument("--value-scale", type=float, default=1.0)
    parser.add_argument("--mcts-cpu-threads", type=int, default=24)
    parser.add_argument("--batch-buckets", default="1,2,4,8,16,32,64")
    parser.add_argument("--native-device", default="cuda:0")
    parser.add_argument("--katagomo-gpu", type=int, default=1)
    parser.add_argument("--katagomo-threads", type=int, default=32)
    parser.add_argument("--katagomo-batch-size", type=int, default=32)
    parser.add_argument("--katagomo-cache-power", type=int, default=20)
    parser.add_argument("--katagomo-timeout-seconds", type=float, default=300.0)
    parser.add_argument("--katagomo-library-path", default="")
    parser.add_argument("--cpu-affinity", default="")
    parser.add_argument("--output", default="")
    arguments = parser.parse_args()

    positive = {
        "pairs": arguments.pairs,
        "native simulations": arguments.native_simulations,
        "KataGomo visits": arguments.katagomo_visits,
        "top-k": arguments.top_k,
        "MCTS CPU threads": arguments.mcts_cpu_threads,
        "KataGomo threads": arguments.katagomo_threads,
        "KataGomo batch size": arguments.katagomo_batch_size,
        "KataGomo timeout": arguments.katagomo_timeout_seconds,
    }
    if arguments.pairs < 2:
        raise ValueError("pairs must be at least two")
    if any(value <= 0 for value in positive.values()):
        raise ValueError(f"positive arguments required: {positive}")
    if not 10 <= arguments.katagomo_cache_power <= 30:
        raise ValueError("KataGomo cache power must be in [10, 30]")
    if not 0 <= arguments.opening_radius <= 9:
        raise ValueError("opening radius must be in [0, 9]")

    native_checkpoint = Path(arguments.native_checkpoint).expanduser().resolve()
    katagomo_engine_path = Path(arguments.katagomo_engine).expanduser().resolve()
    katagomo_model_path = Path(arguments.katagomo_model).expanduser().resolve()
    katagomo_config_path = Path(arguments.katagomo_config).expanduser().resolve()
    for path in (
        native_checkpoint,
        katagomo_engine_path,
        katagomo_model_path,
        katagomo_config_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    affinity = parse_cpu_affinity(arguments.cpu_affinity)
    if affinity:
        os.sched_setaffinity(0, affinity)
    batch_buckets = [
        int(value) for value in arguments.batch_buckets.split(",") if value.strip()
    ]
    if any(value < 1 for value in batch_buckets):
        raise ValueError("batch buckets must be positive")

    device = torch.device(arguments.native_device)
    native_model, _ = load_checkpoint(native_checkpoint)
    native_evaluator = TorchPairEvaluator(
        native_model,
        device,
        top_k=arguments.top_k,
        candidate_gumbel=0.0,
        compile_model=True,
        deduplicate=False,
        batch_buckets=batch_buckets,
        value_scale=arguments.value_scale,
        seed=arguments.seed,
    )
    native_engine = PairMCTSEngine(
        native_evaluator,
        top_k=arguments.top_k,
        root_noise_epsilon=0.0,
        cpu_threads=arguments.mcts_cpu_threads,
    )
    native_engine.reset_global_stats()

    rng = np.random.default_rng(arguments.seed)
    openings = (
        sample_unique_openings(rng, arguments.pairs, arguments.opening_radius)
        if arguments.opening_mode == "forced"
        else [None] * arguments.pairs
    )
    games = []
    started = time.perf_counter()
    gtp = KataGomoGTP(
        katagomo_engine_path,
        katagomo_model_path,
        katagomo_config_path,
        arguments.katagomo_gpu,
        arguments.katagomo_visits,
        arguments.katagomo_threads,
        arguments.katagomo_batch_size,
        arguments.katagomo_cache_power,
        arguments.seed,
        arguments.katagomo_timeout_seconds,
        arguments.katagomo_library_path,
    )
    try:
        gtp.start()
        for pair_index, opening in enumerate(openings):
            for native_black in (True, False):
                gtp.clear_board()
                game = Connect6Game()
                history: list[int] = []
                native_context = native_engine.create_context(
                    arguments.seed + pair_index * 2 + int(native_black)
                )
                native_seconds = 0.0
                katagomo_seconds = 0.0
                native_searches = 0
                katagomo_searches = 0
                native_simulations_actual = 0
                try:
                    if opening is not None:
                        game.play(opening)
                        history.append(opening)
                        native_context.play(opening, PASS_MOVE)
                        gtp.play(opening)
                        verify_synchronized(game, native_context)
                    complete_turns = int(opening is not None)
                    while not game.winner:
                        native_to_move = (
                            (game.current_player == 1 and native_black)
                            or (game.current_player == -1 and not native_black)
                        )
                        if native_to_move:
                            elapsed, search_stats = _play_native_turn(
                                game,
                                history,
                                native_engine,
                                native_context,
                                gtp,
                                arguments.native_simulations,
                                device,
                            )
                            native_seconds += elapsed
                            native_searches += 1
                            native_simulations_actual += int(search_stats["simulations"])
                        else:
                            elapsed, searches = _play_katagomo_turn(
                                game,
                                history,
                                native_context,
                                gtp,
                            )
                            katagomo_seconds += elapsed
                            katagomo_searches += searches
                        complete_turns += 1
                        if complete_turns > 181:
                            raise RuntimeError("arena exceeded the legal maximum number of turns")
                    winner = normalized_winner(game)
                    native_color = 1 if native_black else -1
                    games.append({
                        "pair": pair_index,
                        "native_black": native_black,
                        "winner": winner,
                        "winner_model": (
                            "draw"
                            if winner == 0
                            else "native"
                            if winner == native_color
                            else "katagomo"
                        ),
                        "opening": history[0],
                        "moves": history,
                        "stones": len(history),
                        "complete_turns": complete_turns,
                        "native_searches": native_searches,
                        "native_simulations_actual": native_simulations_actual,
                        "native_seconds": native_seconds,
                        "katagomo_searches": katagomo_searches,
                        "katagomo_seconds": katagomo_seconds,
                    })
                    print(
                        f"{len(games)}/{2 * arguments.pairs} "
                        f"pair={pair_index} native_black={native_black} "
                        f"winner={games[-1]['winner_model']} stones={len(history)}",
                        flush=True,
                    )
                finally:
                    native_context.close()
        torch.cuda.synchronize(device)
    finally:
        gtp.close()
    elapsed = time.perf_counter() - started

    native_wins = sum(item["winner_model"] == "native" for item in games)
    katagomo_wins = sum(item["winner_model"] == "katagomo" for item in games)
    draws = sum(item["winner_model"] == "draw" for item in games)
    black_wins = sum(item["winner"] == 1 for item in games)
    white_wins = sum(item["winner"] == -1 for item in games)
    pair_scores = []
    for pair_index in range(arguments.pairs):
        pair_games = [item for item in games if item["pair"] == pair_index]
        pair_scores.append(
            sum(
                1.0 if item["winner_model"] == "native" else
                0.5 if item["winner_model"] == "draw" else 0.0
                for item in pair_games
            ) / 2.0
        )
    total = len(games)
    native_score = (native_wins + 0.5 * draws) / total
    binomial_standard_error = math.sqrt(
        max(native_score * (1.0 - native_score), 0.0) / total
    )
    natural_opening = arguments.opening_mode == "natural"
    standard_error = (
        binomial_standard_error
        if natural_opening
        else paired_standard_error(pair_scores)
    )
    standard_error_method = (
        "binomial_independent_games"
        if natural_opening
        else "paired_common_opening"
    )
    normal_interval = _confidence_interval(native_score, standard_error)
    wilson_units = total if natural_opening else arguments.pairs
    wilson_interval = wilson_score_interval(native_score, wilson_units)
    boundary_score = native_score in (0.0, 1.0)
    native_seconds = sum(item["native_seconds"] for item in games)
    katagomo_seconds = sum(item["katagomo_seconds"] for item in games)
    native_searches = sum(item["native_searches"] for item in games)
    katagomo_searches = sum(item["katagomo_searches"] for item in games)
    result = {
        "schema_version": 1,
        "protocol": "native-pair-vs-katagomo-single-stone-gtp",
        "pairs": arguments.pairs,
        "games": total,
        "opening_mode": arguments.opening_mode,
        "opening_radius": arguments.opening_radius,
        "unique_actual_openings": len({item["opening"] for item in games}),
        "seed": arguments.seed,
        "native_checkpoint": str(native_checkpoint),
        "native_sha256": sha256_file(native_checkpoint),
        "native_simulations_per_full_turn": arguments.native_simulations,
        "native_top_k": arguments.top_k,
        "native_value_scale": arguments.value_scale,
        "native_device": arguments.native_device,
        "katagomo_engine": str(katagomo_engine_path),
        "katagomo_engine_sha256": sha256_file(katagomo_engine_path),
        "katagomo_model": str(katagomo_model_path),
        "katagomo_model_sha256": sha256_file(katagomo_model_path),
        "katagomo_config": str(katagomo_config_path),
        "katagomo_config_sha256": sha256_file(katagomo_config_path),
        "katagomo_name": gtp.name,
        "katagomo_version": gtp.version,
        "katagomo_gpu": arguments.katagomo_gpu,
        "katagomo_visits_per_stone": arguments.katagomo_visits,
        "katagomo_nominal_visits_per_full_turn": 2 * arguments.katagomo_visits,
        "katagomo_threads": arguments.katagomo_threads,
        "katagomo_batch_size": arguments.katagomo_batch_size,
        "native_wins": native_wins,
        "katagomo_wins": katagomo_wins,
        "draws": draws,
        "native_score": native_score,
        "native_score_standard_error": standard_error,
        "native_score_confidence_interval_95": (
            wilson_interval if boundary_score else normal_interval
        ),
        "native_score_confidence_interval_method": (
            f"wilson_boundary_{standard_error_method}"
            if boundary_score
            else f"normal_{standard_error_method}"
        ),
        "native_score_normal_interval_95": normal_interval,
        "native_score_wilson_interval_95": wilson_interval,
        "native_score_standard_error_method": standard_error_method,
        "black_wins": black_wins,
        "white_wins": white_wins,
        "mean_stones": float(np.mean([item["stones"] for item in games])),
        "seconds": elapsed,
        "games_per_second": total / elapsed,
        "native_total_search_seconds": native_seconds,
        "native_mean_search_seconds": native_seconds / max(native_searches, 1),
        "katagomo_total_search_seconds": katagomo_seconds,
        "katagomo_mean_search_seconds_per_stone": (
            katagomo_seconds / max(katagomo_searches, 1)
        ),
        "native_evaluator": native_evaluator.stats(),
        "native_search": native_engine.global_stats(),
        "games_detail": games,
    }
    payload = json.dumps(result, indent=2)
    if arguments.output:
        output = Path(arguments.output).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(payload + "\n", encoding="utf-8")
        temporary.replace(output)
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
