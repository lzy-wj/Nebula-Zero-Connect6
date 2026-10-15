"""ctypes bridge and batched evaluator for the isolated native-Pair MCTS."""

from __future__ import annotations

import ctypes
import os
import subprocess
from pathlib import Path

import numpy as np
import torch

from experiments.pair_native.model import BOARD_POINTS, PairNativeSparseNet


PASS_MOVE = -1
EVAL_CALLBACK = ctypes.CFUNCTYPE(
    ctypes.c_int,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_int8),
    ctypes.POINTER(ctypes.c_int8),
    ctypes.POINTER(ctypes.c_int16),
    ctypes.POINTER(ctypes.c_int16),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
)


def default_library_path() -> Path:
    return Path(__file__).with_name("libpair_mcts.so")


def ensure_library(path: str | os.PathLike | None = None) -> Path:
    output = Path(path).expanduser().resolve() if path else default_library_path()
    source = Path(__file__).with_name("pair_mcts.cpp")
    if not output.exists() or output.stat().st_mtime_ns < source.stat().st_mtime_ns:
        subprocess.run(
            [
                os.environ.get("PYTHON", "python"),
                "-m",
                "experiments.pair_native.compile_pair_mcts",
                "--output",
                str(output),
            ],
            check=True,
            cwd=Path(__file__).resolve().parents[2],
        )
    return output


def load_library(path: str | os.PathLike | None = None):
    library = ctypes.CDLL(str(ensure_library(path)))
    library.pair_mcts_abi_version.restype = ctypes.c_int
    if library.pair_mcts_abi_version() != 1:
        raise RuntimeError("unsupported Pair MCTS ABI")
    library.pair_mcts_set_eval_callback.argtypes = [EVAL_CALLBACK]
    library.pair_mcts_set_params.argtypes = [
        ctypes.c_int,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_float,
    ]
    library.pair_mcts_set_random_candidates.argtypes = [ctypes.c_int]
    library.pair_mcts_set_cpu_threads.argtypes = [ctypes.c_int]
    library.pair_mcts_create_context.argtypes = [ctypes.c_uint32]
    library.pair_mcts_create_context.restype = ctypes.c_void_p
    library.pair_mcts_destroy_context.argtypes = [ctypes.c_void_p]
    library.pair_mcts_run_multi.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_int,
    ]
    library.pair_mcts_run_multi.restype = ctypes.c_int
    library.pair_mcts_choose_action.argtypes = [
        ctypes.c_void_p,
        ctypes.c_float,
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
    ]
    library.pair_mcts_choose_action.restype = ctypes.c_int
    library.pair_mcts_get_policy.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int16),
        ctypes.POINTER(ctypes.c_int16),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
    ]
    library.pair_mcts_get_policy.restype = ctypes.c_int
    library.pair_mcts_get_root_value.argtypes = [ctypes.c_void_p]
    library.pair_mcts_get_root_value.restype = ctypes.c_float
    library.pair_mcts_play_action.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
    library.pair_mcts_play_action.restype = ctypes.c_int
    library.pair_mcts_reset_tree.argtypes = [ctypes.c_void_p]
    for name in (
        "pair_mcts_get_winner",
        "pair_mcts_is_terminal",
        "pair_mcts_get_total_stones",
        "pair_mcts_get_current_player",
    ):
        function = getattr(library, name)
        function.argtypes = [ctypes.c_void_p]
        function.restype = ctypes.c_int
    library.pair_mcts_get_last_stats.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_longlong),
        ctypes.POINTER(ctypes.c_longlong),
        ctypes.POINTER(ctypes.c_longlong),
    ]
    library.pair_mcts_reset_global_stats.argtypes = []
    library.pair_mcts_get_global_stats.argtypes = [
        ctypes.POINTER(ctypes.c_longlong),
        ctypes.POINTER(ctypes.c_longlong),
        ctypes.POINTER(ctypes.c_longlong),
        ctypes.POINTER(ctypes.c_longlong),
    ]
    return library


class PairTopKInference(torch.nn.Module):
    """Compile the network, joint pair scoring, top-k and normalization together."""

    def __init__(self, model: PairNativeSparseNet, top_k: int):
        super().__init__()
        self.model = model
        self.top_k = int(top_k)

    def forward(self, features: torch.Tensor):
        unary, pair_embeddings, values, occupied = self.model(features)
        pair_scores = self.model.pair_scores(unary, pair_embeddings, occupied)
        pair_score, flat_pair = torch.topk(
            pair_scores.flatten(1),
            k=self.top_k,
            dim=1,
        )
        first = flat_pair.div(BOARD_POINTS, rounding_mode="floor")
        second = flat_pair % BOARD_POINTS

        opening = occupied.sum(dim=1).eq(0)
        opening_logits = unary.masked_fill(occupied, torch.finfo(unary.dtype).min)
        opening_score, opening_moves = torch.topk(
            opening_logits,
            k=self.top_k,
            dim=1,
        )
        first = torch.where(opening[:, None], opening_moves, first)
        second = torch.where(
            opening[:, None],
            torch.full_like(second, PASS_MOVE),
            second,
        )
        pair_score = torch.where(opening[:, None], opening_score, pair_score)
        priors = torch.softmax(pair_score.float(), dim=1)
        return first, second, priors, values


class TorchPairEvaluator:
    def __init__(
        self,
        model: PairNativeSparseNet,
        device: torch.device,
        top_k: int = 64,
        dtype: torch.dtype = torch.float16,
        candidate_gumbel: float = 0.0,
        seed: int = 2026,
        compile_model: bool = False,
        deduplicate: bool = True,
        pad_batch_to: int = 0,
        batch_buckets: tuple[int, ...] | list[int] = (),
        value_scale: float = 1.0,
    ):
        self.model = model.eval().to(device)
        self.inference = None
        if compile_model and candidate_gumbel <= 0:
            self.inference = torch.compile(
                PairTopKInference(self.model, top_k).eval(),
                fullgraph=True,
                mode="reduce-overhead",
            )
        elif compile_model:
            self.model = torch.compile(
                self.model,
                fullgraph=True,
                mode="reduce-overhead",
            )
        self.device = device
        self.top_k = int(top_k)
        self.dtype = dtype
        self.candidate_gumbel = float(candidate_gumbel)
        self.deduplicate = bool(deduplicate)
        self.pad_batch_to = int(pad_batch_to)
        self.batch_buckets = tuple(sorted({int(value) for value in batch_buckets}))
        self.value_scale = float(value_scale)
        if self.pad_batch_to < 0:
            raise ValueError("pad_batch_to must be non-negative")
        if any(value <= 0 for value in self.batch_buckets):
            raise ValueError("batch_buckets must contain positive integers")
        self.requested_positions = 0
        self.unique_positions = 0
        self.eval_batches = 0
        self.generator = torch.Generator(device=device)
        self.generator.manual_seed(seed)
        self.last_error: Exception | None = None
        self.callback = EVAL_CALLBACK(self._callback)

    def _features(self, boards: torch.Tensor, players: torch.Tensor) -> torch.Tensor:
        board = boards.reshape(-1, 19, 19)
        player = players.reshape(-1, 1, 1)
        own = board.eq(player)
        opponent = board.eq(-player)
        empty = board.eq(0)
        black = player.eq(1).expand_as(board)
        turn_boundary = torch.zeros_like(own)
        return torch.stack((own, opponent, empty, black, turn_boundary), dim=1).float()

    def _callback(
        self,
        batch_size,
        boards_pointer,
        players_pointer,
        first_pointer,
        second_pointer,
        priors_pointer,
        values_pointer,
    ):
        try:
            boards_numpy = np.ctypeslib.as_array(
                boards_pointer,
                shape=(batch_size * BOARD_POINTS,),
            ).reshape(batch_size, BOARD_POINTS)
            players_numpy = np.ctypeslib.as_array(
                players_pointer,
                shape=(batch_size,),
            )
            self.requested_positions += batch_size
            self.eval_batches += 1
            inverse = None
            if self.deduplicate and batch_size > 1:
                keys = np.concatenate(
                    (boards_numpy, players_numpy[:, None]),
                    axis=1,
                )
                _, unique_indices, inverse = np.unique(
                    keys,
                    axis=0,
                    return_index=True,
                    return_inverse=True,
                )
                boards_input = boards_numpy[unique_indices]
                players_input = players_numpy[unique_indices]
            else:
                boards_input = boards_numpy
                players_input = players_numpy
            self.unique_positions += len(boards_input)
            input_count = len(boards_input)
            padded_count = self.pad_batch_to
            if self.batch_buckets:
                padded_count = next(
                    (value for value in self.batch_buckets if input_count <= value),
                    input_count,
                )
            if padded_count and input_count < padded_count:
                padding = padded_count - input_count
                boards_input = np.concatenate(
                    (boards_input, np.repeat(boards_input[:1], padding, axis=0)),
                    axis=0,
                )
                players_input = np.concatenate(
                    (players_input, np.repeat(players_input[:1], padding, axis=0)),
                    axis=0,
                )
            boards = torch.from_numpy(boards_input).to(self.device)
            players = torch.from_numpy(players_input).to(self.device)
            features = self._features(boards, players)
            with torch.inference_mode(), torch.autocast(
                device_type="cuda",
                dtype=self.dtype,
            ):
                if self.inference is not None:
                    first, second, priors, values = self.inference(features)
                else:
                    unary, pair_embeddings, values, occupied = self.model(features)
                    opening = occupied.sum(dim=1).eq(0)
                    if opening.all():
                        opening_logits = unary.masked_fill(
                            occupied,
                            torch.finfo(unary.dtype).min,
                        )
                        pair_score, first = torch.topk(
                            opening_logits,
                            k=self.top_k,
                            dim=1,
                        )
                        second = torch.full_like(first, PASS_MOVE)
                    else:
                        score_model = getattr(self.model, "_orig_mod", self.model)
                        pair_scores = score_model.pair_scores(
                            unary,
                            pair_embeddings,
                            occupied,
                        )
                        if self.candidate_gumbel > 0:
                            uniform = torch.rand(
                                pair_scores.shape,
                                device=self.device,
                                dtype=torch.float32,
                                generator=self.generator,
                            ).clamp_(1.0e-6, 1.0 - 1.0e-6)
                            gumbel = -torch.log(-torch.log(uniform))
                            pair_scores = pair_scores + self.candidate_gumbel * gumbel.to(
                                pair_scores.dtype
                            )
                        pair_score, flat_pair = torch.topk(
                            pair_scores.flatten(1),
                            k=self.top_k,
                            dim=1,
                        )
                        first = flat_pair.div(BOARD_POINTS, rounding_mode="floor")
                        second = flat_pair % BOARD_POINTS
                        if opening.any():
                            opening_logits = unary.masked_fill(
                                occupied,
                                torch.finfo(unary.dtype).min,
                            )
                            opening_score, opening_moves = torch.topk(
                                opening_logits,
                                k=self.top_k,
                                dim=1,
                            )
                            first = torch.where(opening[:, None], opening_moves, first)
                            second = torch.where(
                                opening[:, None],
                                torch.full_like(second, PASS_MOVE),
                                second,
                            )
                            pair_score = torch.where(
                                opening[:, None],
                                opening_score,
                                pair_score,
                            )
                    priors = torch.softmax(pair_score.float(), dim=1)

            first = first[:input_count]
            second = second[:input_count]
            priors = priors[:input_count]
            values = values[:input_count] * self.value_scale

            if inverse is not None:
                inverse_tensor = torch.from_numpy(inverse).to(self.device)
                first = first[inverse_tensor]
                second = second[inverse_tensor]
                priors = priors[inverse_tensor]
                values = values[inverse_tensor]

            first_numpy = first.to(dtype=torch.int16).cpu().numpy()
            second_numpy = second.to(dtype=torch.int16).cpu().numpy()
            priors_numpy = priors.cpu().numpy()
            values_numpy = values.float().cpu().numpy()
            np.copyto(
                np.ctypeslib.as_array(
                    first_pointer,
                    shape=(batch_size * self.top_k,),
                ).reshape(batch_size, self.top_k),
                first_numpy,
            )
            np.copyto(
                np.ctypeslib.as_array(
                    second_pointer,
                    shape=(batch_size * self.top_k,),
                ).reshape(batch_size, self.top_k),
                second_numpy,
            )
            np.copyto(
                np.ctypeslib.as_array(
                    priors_pointer,
                    shape=(batch_size * self.top_k,),
                ).reshape(batch_size, self.top_k),
                priors_numpy,
            )
            np.copyto(
                np.ctypeslib.as_array(values_pointer, shape=(batch_size,)),
                values_numpy,
            )
            return 0
        except Exception as error:  # ctypes callbacks must not leak exceptions.
            self.last_error = error
            return 1

    def stats(self) -> dict[str, int | float]:
        return {
            "requested_positions": self.requested_positions,
            "unique_positions": self.unique_positions,
            "unique_position_rate": (
                self.unique_positions / self.requested_positions
                if self.requested_positions
                else 0.0
            ),
            "eval_batches": self.eval_batches,
        }


class PairMCTSContext:
    def __init__(self, engine: "PairMCTSEngine", seed: int):
        self.engine = engine
        self.handle = ctypes.c_void_p(engine.library.pair_mcts_create_context(seed))
        if not self.handle:
            raise RuntimeError("failed to create Pair MCTS context")

    def close(self):
        if self.handle:
            self.engine.library.pair_mcts_destroy_context(self.handle)
            self.handle = None

    def choose_action(self, temperature: float = 0.0) -> tuple[int, int]:
        first = ctypes.c_int()
        second = ctypes.c_int()
        ok = self.engine.library.pair_mcts_choose_action(
            self.handle,
            float(temperature),
            ctypes.byref(first),
            ctypes.byref(second),
        )
        if not ok:
            raise RuntimeError("Pair MCTS root has no action")
        return first.value, second.value

    def policy(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        capacity = self.engine.top_k
        first = np.empty(capacity, dtype=np.int16)
        second = np.empty(capacity, dtype=np.int16)
        probabilities = np.empty(capacity, dtype=np.float32)
        count = self.engine.library.pair_mcts_get_policy(
            self.handle,
            first.ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
            second.ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
            probabilities.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            capacity,
        )
        return first[:count].copy(), second[:count].copy(), probabilities[:count].copy()

    def root_value(self) -> float:
        return float(self.engine.library.pair_mcts_get_root_value(self.handle))

    def play(self, first: int, second: int):
        if not self.engine.library.pair_mcts_play_action(self.handle, first, second):
            raise ValueError(f"illegal Pair action: {(first, second)}")

    def reset_tree(self):
        self.engine.library.pair_mcts_reset_tree(self.handle)

    def stats(self) -> dict[str, int]:
        simulations = ctypes.c_longlong()
        evaluations = ctypes.c_longlong()
        nodes = ctypes.c_longlong()
        self.engine.library.pair_mcts_get_last_stats(
            self.handle,
            ctypes.byref(simulations),
            ctypes.byref(evaluations),
            ctypes.byref(nodes),
        )
        return {
            "simulations": simulations.value,
            "neural_evaluations": evaluations.value,
            "unique_nodes": nodes.value,
        }

    @property
    def terminal(self) -> bool:
        return bool(self.engine.library.pair_mcts_is_terminal(self.handle))

    @property
    def winner(self) -> int:
        return int(self.engine.library.pair_mcts_get_winner(self.handle))

    @property
    def total_stones(self) -> int:
        return int(self.engine.library.pair_mcts_get_total_stones(self.handle))

    @property
    def current_player(self) -> int:
        return int(self.engine.library.pair_mcts_get_current_player(self.handle))

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class PairMCTSEngine:
    def __init__(
        self,
        evaluator,
        library_path: str | os.PathLike | None = None,
        top_k: int = 64,
        cpuct: float = 1.5,
        root_noise_epsilon: float = 0.25,
        root_dirichlet_alpha: float = 0.15,
        random_candidates: int = 0,
        cpu_threads: int = 1,
    ):
        self.library = load_library(library_path)
        self.evaluator = evaluator
        self.top_k = int(top_k)
        self.library.pair_mcts_set_eval_callback(evaluator.callback)
        self.library.pair_mcts_set_params(
            self.top_k,
            float(cpuct),
            float(root_noise_epsilon),
            float(root_dirichlet_alpha),
        )
        self.library.pair_mcts_set_random_candidates(int(random_candidates))
        self.library.pair_mcts_set_cpu_threads(int(cpu_threads))

    def create_context(self, seed: int) -> PairMCTSContext:
        return PairMCTSContext(self, seed)

    def set_evaluator(self, evaluator):
        self.evaluator = evaluator
        self.library.pair_mcts_set_eval_callback(evaluator.callback)

    def run(self, contexts: list[PairMCTSContext], budgets: int | list[int]):
        if isinstance(budgets, int):
            budgets = [budgets] * len(contexts)
        if len(budgets) != len(contexts):
            raise ValueError("budget/context count mismatch")
        handles = (ctypes.c_void_p * len(contexts))(
            *(context.handle.value for context in contexts)
        )
        budget_array = (ctypes.c_int * len(contexts))(*(int(value) for value in budgets))
        status = self.library.pair_mcts_run_multi(handles, budget_array, len(contexts))
        if status != 0:
            if getattr(self.evaluator, "last_error", None) is not None:
                error = self.evaluator.last_error
                self.evaluator.last_error = None
                raise RuntimeError("Pair evaluator callback failed") from error
            raise RuntimeError(f"Pair MCTS failed with status {status}")

    def global_stats(self) -> dict[str, int | float]:
        values = [ctypes.c_longlong() for _ in range(4)]
        self.library.pair_mcts_get_global_stats(
            *(ctypes.byref(value) for value in values)
        )
        simulations, evaluations, batches, tactical = (value.value for value in values)
        return {
            "simulations": simulations,
            "neural_evaluations": evaluations,
            "eval_batches": batches,
            "tactical_expansions": tactical,
            "average_batch": evaluations / batches if batches else 0.0,
        }

    def reset_global_stats(self):
        self.library.pair_mcts_reset_global_stats()
