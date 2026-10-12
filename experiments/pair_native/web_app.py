"""Interactive Web arena for native-Pair and historical Connect6 models.

The browser submits individual stones, while this module only advances the
native-Pair search context after a complete Connect6 turn is available.  The
historical TensorRT models retain their original one-search-per-stone
semantics.  Keeping those two contracts explicit is important: a numeric MCTS
budget is per full turn for a native model and per stone for a legacy model.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import secrets
import threading
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from flask import Flask, jsonify, render_template, request

from experiments.pair_native.model import load_checkpoint
from experiments.pair_native.pair_mcts import (
    PASS_MOVE,
    PairMCTSEngine,
    TorchPairEvaluator,
)


BOARD_SIZE = 19
BOARD_POINTS = BOARD_SIZE * BOARD_SIZE
BLACK = 1
WHITE = -1
DRAW = 0
MAX_SIMULATIONS = 10_000
MAX_CPU_THREADS = 24
NATIVE_TOP_K = (16, 32, 64, 128)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACT_ROOT = REPOSITORY_ROOT / "artifacts"
DEFAULT_TRAINING_ROOT = DEFAULT_ARTIFACT_ROOT / "pair_native"
DEFAULT_LEGACY_ROOT = DEFAULT_ARTIFACT_ROOT / "legacy"


def _number(value, default, minimum, maximum, cast):
    try:
        parsed = cast(value)
    except (TypeError, ValueError, OverflowError):
        parsed = cast(default)
    return max(minimum, min(maximum, parsed))


def _checkpoint_id(kind: str, path: Path) -> str:
    digest = hashlib.sha256(f"{kind}:{path.resolve()}".encode()).hexdigest()[:20]
    return f"{kind}-{digest}"


def _read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


@dataclass(frozen=True)
class ModelRecord:
    id: str
    kind: str
    name: str
    group: str
    path: str
    size_mb: float
    modified_at: int
    accepted: bool = False
    current: bool = False
    description: str = ""

    def public(self) -> dict:
        payload = asdict(self)
        payload.pop("path")
        payload["budget_semantics"] = (
            "每完整回合" if self.kind == "native_pair" else "每颗棋子"
        )
        return payload


class ModelRegistry:
    """Discover only checkpoints below administrator-configured roots."""

    def __init__(
        self,
        training_root: Path | None = None,
        legacy_root: Path | None = None,
    ):
        self.training_root = Path(
            training_root
            or os.environ.get("NEBULA_PAIR_WEB_TRAINING_ROOT", DEFAULT_TRAINING_ROOT)
        ).expanduser().resolve()
        self.legacy_root = Path(
            legacy_root
            or os.environ.get("NEBULA_PAIR_WEB_LEGACY_ROOT", DEFAULT_LEGACY_ROOT)
        ).expanduser().resolve()

    def _states(self) -> list[tuple[Path, dict]]:
        if not self.training_root.is_dir():
            return []
        return [
            (path, payload)
            for path in self.training_root.glob("*/state.json")
            if (payload := _read_json(path))
        ]

    def records(self) -> tuple[list[ModelRecord], str | None]:
        states = self._states()
        accepted_paths: set[Path] = set()
        current_path: Path | None = None
        newest_running = -1.0
        for state_path, state in states:
            accepted = state.get("accepted_checkpoint")
            if accepted:
                accepted_paths.add(Path(accepted).expanduser().resolve())
            for promotion in state.get("promotions", []):
                if promotion.get("promote") and promotion.get("candidate_checkpoint"):
                    accepted_paths.add(
                        Path(promotion["candidate_checkpoint"]).expanduser().resolve()
                    )
            updated = float(state.get("updated_unix") or state_path.stat().st_mtime)
            if state.get("status") == "running" and accepted and updated > newest_running:
                newest_running = updated
                current_path = Path(accepted).expanduser().resolve()

        if current_path is None and states:
            _, newest_state = max(
                states,
                key=lambda item: float(
                    item[1].get("updated_unix") or item[0].stat().st_mtime
                ),
            )
            accepted = newest_state.get("accepted_checkpoint")
            if accepted:
                current_path = Path(accepted).expanduser().resolve()

        native_paths: set[Path] = set()
        if self.training_root.is_dir():
            native_paths.update(self.training_root.glob("**/best.pth"))
            native_paths.update(self.training_root.glob("**/initial.pth"))
        native_paths.update(path for path in accepted_paths if path.is_file())

        records: list[ModelRecord] = []
        for path in native_paths:
            path = path.resolve()
            if not path.is_file():
                continue
            try:
                relative = path.relative_to(self.training_root)
            except ValueError:
                continue
            run = relative.parts[0] if relative.parts else "pair_native"
            stage = "/".join(relative.parts[1:-1]) or path.stem
            accepted = path in accepted_paths
            current = path == current_path
            prefix = "★ 当前已晋级" if current else "✓ 已晋级" if accepted else "候选"
            records.append(
                ModelRecord(
                    id=_checkpoint_id("native", path),
                    kind="native_pair",
                    name=f"{prefix} · {run} / {stage}",
                    group=f"Pair-native · {run}",
                    path=str(path),
                    size_mb=round(path.stat().st_size / 2**20, 1),
                    modified_at=int(path.stat().st_mtime),
                    accepted=accepted,
                    current=current,
                    description="稀疏 Pair 模型；一次搜索选择完整两子动作",
                )
            )

        if self.legacy_root.is_dir():
            preferred_order = {
                "best.engine": 0,
                "model_gen_801.engine": 1,
                "model_gen_800.engine": 2,
                "model_gen_400.engine": 3,
                "model_gen_0.engine": 4,
            }
            engines = sorted(
                self.legacy_root.glob("*.engine"),
                key=lambda path: (preferred_order.get(path.name, 99), path.name),
            )
            for path in engines:
                records.append(
                    ModelRecord(
                        id=_checkpoint_id("legacy", path),
                        kind="legacy",
                        name=f"第一代 · {path.stem}",
                        group="第一代 ModelScope · TensorRT",
                        path=str(path.resolve()),
                        size_mb=round(path.stat().st_size / 2**20, 1),
                        modified_at=int(path.stat().st_mtime),
                        description="历史逐子模型；每颗棋子独立执行一次 MCTS",
                    )
                )

        records.sort(
            key=lambda item: (
                not item.current,
                not item.accepted,
                item.kind != "native_pair",
                -item.modified_at,
                item.name,
            )
        )
        maximum = int(os.environ.get("NEBULA_PAIR_WEB_MAX_MODELS", "220"))
        records = records[: max(maximum, 1)]
        default_id = next((item.id for item in records if item.current), None)
        if default_id is None:
            default_id = next(
                (item.id for item in records if item.accepted),
                records[0].id if records else None,
            )
        return records, default_id

    def resolve(self, model_id: str) -> ModelRecord | None:
        records, _ = self.records()
        return next((record for record in records if record.id == model_id), None)


@dataclass(frozen=True)
class RunRecord:
    id: str
    name: str
    path: str
    status: str
    updated_unix: float
    completed: int
    target: int

    def public(self) -> dict:
        payload = asdict(self)
        payload.pop("path")
        return payload


def _ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


class TrainingMonitor:
    """Read-only view over atomic async-loop state and learner histories."""

    CANDIDATE_PATTERN = re.compile(r"through-(\d+)-attempt-(\d+)")

    def __init__(self, training_root: Path):
        self.training_root = Path(training_root).expanduser().resolve()
        self._diversity_cache: dict[str, dict] = {}
        self._diversity_lock = threading.Lock()

    def runs(self) -> tuple[list[RunRecord], str | None]:
        records: list[RunRecord] = []
        if self.training_root.is_dir():
            for state_path in self.training_root.glob("*/state.json"):
                state = _read_json(state_path)
                if not state:
                    continue
                path = state_path.parent.resolve()
                records.append(
                    RunRecord(
                        id=_checkpoint_id("run", path),
                        name=path.name,
                        path=str(path),
                        status=str(state.get("status", "unknown")),
                        updated_unix=float(
                            state.get("updated_unix") or state_path.stat().st_mtime
                        ),
                        completed=len(state.get("completed_rounds", [])),
                        target=int(state.get("config", {}).get("rounds", 0)),
                    )
                )
        priority = {"running": 0, "complete": 1, "paused": 2, "failed": 3}
        records.sort(
            key=lambda item: (priority.get(item.status, 3), -item.updated_unix)
        )
        running = [record for record in records if record.status == "running"]
        if running:
            default_id = running[0].id
        else:
            # Prefer the most recently updated terminal run over an older
            # paused/failed run; other runs remain explicitly selectable.
            default_id = (
                max(records, key=lambda item: item.updated_unix).id
                if records
                else None
            )
        return records, default_id

    def resolve(self, run_id: str) -> RunRecord | None:
        records, _ = self.runs()
        return next((record for record in records if record.id == run_id), None)

    def _replay_diversity(self, directory: Path) -> dict:
        key = str(directory.resolve())
        with self._diversity_lock:
            cached = self._diversity_cache.get(key)
        if cached is not None:
            return cached
        import pyarrow.parquet as parquet

        opening_counts = Counter()
        signatures = Counter()
        for path in sorted(directory.glob("*.games.parquet")):
            table = parquet.read_table(path, columns=["moves"])
            for raw_moves in table["moves"].to_pylist():
                moves = tuple(int(move) for move in raw_moves)
                if moves:
                    opening_counts[moves[0]] += 1
                    signatures[moves] += 1
        games = sum(opening_counts.values())
        if not games:
            result = {"opening": {}, "diversity": {}}
        else:
            top_move, top_count = opening_counts.most_common(1)[0]
            probabilities = [count / games for count in opening_counts.values()]
            entropy = -sum(value * math.log2(value) for value in probabilities)
            result = {
                "opening": {
                    "unique": len(opening_counts),
                    "entropy_bits": entropy,
                    "effective_openings": 2.0**entropy,
                    "top_move": top_move,
                    "top_share": top_count / games,
                },
                "diversity": {
                    "unique_complete_game_rate": len(signatures) / games,
                    "duplicate_complete_games": games - len(signatures),
                    "maximum_complete_game_repetitions": max(signatures.values()),
                },
            }
        with self._diversity_lock:
            self._diversity_cache[key] = result
        return result

    @staticmethod
    def _round_payload(item: dict, previous: list[dict]) -> dict:
        summary = item.get("summary", {})
        winners = summary.get("winner_counts", {})
        lengths = summary.get("length_buckets", {})
        games = int(summary.get("games", 0))
        black = int(winners.get("black", 0))
        white = int(winners.get("white", 0))
        draw = int(winners.get("draw", 0))
        workers = summary.get("workers", [])
        opening = summary.get("opening", {})
        diversity = summary.get("diversity", {})
        worker_search = next(
            (
                worker.get("audit", {}).get("metadata", {}).get("search", {})
                for worker in workers
                if worker.get("audit", {}).get("metadata", {}).get("search")
            ),
            {},
        )
        opening_temperature = item.get("opening_control", {}).get(
            "opening_temperature"
        )
        if opening_temperature is None:
            opening_temperature = worker_search.get("opening_temperature")
        if opening_temperature is None:
            opening_temperature = (
                worker_search.get("early_temperature", 0.0)
                if int(worker_search.get("temperature_turns", 0)) > 0
                else 0.0
            )
        invalid_audits = sum(
            1 for worker in workers if not worker.get("audit", {}).get("valid", False)
        )
        valid_turns = sum(
            int(worker.get("audit", {}).get("valid_turns", 0)) for worker in workers
        )
        unique_states = sum(
            int(worker.get("audit", {}).get("unique_states", 0)) for worker in workers
        )
        payload = {
            "round_index": int(item.get("round_index", 0)),
            "generation": int(item.get("generation", 0)),
            "games": games,
            "turns": int(summary.get("turns", 0)),
            "stones": int(summary.get("stones", 0)),
            "mean_stones": _ratio(summary.get("stones", 0), games),
            "mean_turns": _ratio(summary.get("turns", 0), games),
            "black": black,
            "white": white,
            "draw": draw,
            "black_decisive": _ratio(black, black + white),
            "white_decisive": _ratio(white, black + white),
            "draw_rate": _ratio(draw, games),
            "length": {
                name: int(lengths.get(name, 0))
                for name in ("short", "medium", "long", "very_long")
            },
            "length_rate": {
                name: _ratio(lengths.get(name, 0), games)
                for name in ("short", "medium", "long", "very_long")
            },
            "seconds": float(summary.get("seconds", 0.0)),
            "games_per_second": float(summary.get("games_per_second", 0.0)),
            "turns_per_second": float(summary.get("turns_per_second", 0.0)),
            "black_simulations": int(
                item.get("color_control", {}).get("black_simulations", 0)
            ),
            "white_simulations": int(
                item.get("color_control", {}).get("white_simulations", 0)
            ),
            "boosted_color": item.get("color_control", {}).get(
                "boosted_color", "none"
            ),
            "controller_black_decisive": item.get("color_control", {}).get(
                "observed_black_decisive_rate"
            ),
            "checkpoint_sha": str(item.get("checkpoint_sha256", ""))[:12],
            "opening_temperature": float(opening_temperature or 0.0),
            "opening_top_move": opening.get("top_move"),
            "opening_top_share": opening.get("top_share"),
            "opening_entropy_bits": opening.get("entropy_bits"),
            "effective_openings": opening.get("effective_openings"),
            "unique_complete_game_rate": diversity.get("unique_complete_game_rate"),
            "duplicate_complete_games": diversity.get("duplicate_complete_games"),
            "maximum_complete_game_repetitions": diversity.get(
                "maximum_complete_game_repetitions"
            ),
            "invalid_audits": invalid_audits,
            "unique_state_rate": _ratio(unique_states, valid_turns),
        }
        window = (previous + [payload])[-3:]
        window_black = sum(value["black"] for value in window)
        window_white = sum(value["white"] for value in window)
        payload["rolling_black_decisive"] = _ratio(
            window_black, window_black + window_white
        )
        return payload

    @classmethod
    def _training_payloads(cls, run_path: Path, promotions: list[dict]) -> list[dict]:
        promotion_by_round = {
            int(item.get("through_round", -1)): item for item in promotions
        }
        selected: dict[int, tuple[int, Path]] = {}
        for history_path in run_path.glob("candidates/through-*-attempt-*/history.json"):
            match = cls.CANDIDATE_PATTERN.search(history_path.parent.name)
            if not match:
                continue
            through, attempt = (int(value) for value in match.groups())
            if through not in selected or attempt > selected[through][0]:
                selected[through] = (attempt, history_path)
        payloads = []
        for through, (attempt, history_path) in sorted(selected.items()):
            history = _read_json(history_path)
            epochs = history.get("epochs", [])
            initial = history.get("initial_validation", {})
            final = epochs[-1].get("validation", {}) if epochs else {}
            training = epochs[-1].get("training", {}) if epochs else {}
            sampling = history.get("sampling", {})
            promotion = promotion_by_round.get(through, {})
            payloads.append(
                {
                    "through_round": through,
                    "attempt": attempt,
                    "games": int(history.get("games", 0)),
                    "training_turns": int(history.get("training_turns", 0)),
                    "validation_turns": int(history.get("validation_turns", 0)),
                    "initial_validation": {
                        name: initial.get(name)
                        for name in (
                            "loss",
                            "policy_loss",
                            "value_loss",
                            "policy_top1",
                            "value_mae",
                            "value_decisive_sign_accuracy",
                        )
                    },
                    "final_validation": {
                        name: final.get(name)
                        for name in (
                            "loss",
                            "policy_loss",
                            "value_loss",
                            "policy_top1",
                            "value_mae",
                            "value_decisive_sign_accuracy",
                        )
                    },
                    "train_samples_per_second": training.get("samples_per_second"),
                    "sampling": {
                        name: sampling.get(name)
                        for name in (
                            "phase_source",
                            "phase_sampled",
                            "phase_target",
                            "length_source",
                            "length_sampled",
                            "length_target",
                            "winner_source",
                            "winner_sampled",
                            "winner_target",
                            "minimum_weight",
                            "maximum_weight",
                            "maximum_to_minimum_weight_ratio",
                            "effective_sample_size",
                            "rows",
                        )
                    },
                    "promoted": bool(promotion.get("promote", False)),
                }
            )
        return payloads

    @staticmethod
    def _alerts(
        state: dict,
        rounds: list[dict],
        promotions: list[dict],
        training: list[dict],
    ) -> list[dict]:
        alerts: list[dict] = []

        def add(level: str, title: str, detail: str):
            alerts.append({"level": level, "title": title, "detail": detail})

        if state.get("status") == "failed":
            add("critical", "训练已经失败", "编排器状态为 failed，请先检查日志。")
        invalid = sum(item["invalid_audits"] for item in rounds)
        if invalid:
            add("critical", "数据审计失败", f"发现 {invalid} 个无效 replay shard。")
        if rounds:
            latest = rounds[-1]
            rate = latest["rolling_black_decisive"]
            if rate > 0.55 or rate < 0.45:
                weak = "白方" if rate > 0.55 else "黑方"
                add(
                    "warning" if 0.40 <= rate <= 0.60 else "critical",
                    "黑白胜率离开目标区间",
                    f"最近三代黑方决定局胜率 {rate:.1%}；{weak}正在处于劣势。",
                )
            base = int(state.get("config", {}).get("simulations", 0))
            if latest["black_simulations"] > base or latest["white_simulations"] > base:
                add(
                    "info",
                    "颜色补偿已触发",
                    f"当前预算 黑 {latest['black_simulations']} / 白 {latest['white_simulations']}。",
                )
            if latest.get("opening_top_share") is not None:
                share = float(latest["opening_top_share"])
                if share > 0.40:
                    add(
                        "critical" if share > 0.75 else "warning",
                        "自然开局过度集中",
                        f"最常见首手占 {share:.1%}；建议提高首手温度。",
                    )
            if latest.get("unique_complete_game_rate") is not None:
                unique_rate = float(latest["unique_complete_game_rate"])
                if unique_rate < 0.85:
                    add(
                        "critical" if unique_rate < 0.70 else "warning",
                        "完整棋局重复偏多",
                        f"本代完整棋局唯一率仅 {unique_rate:.1%}。",
                    )
            recent = rounds[-3:]
            games = sum(item["games"] for item in recent)
            short = sum(item["length"]["short"] for item in recent)
            if games and short / games < 0.05:
                add(
                    "warning",
                    "短局来源偏少",
                    f"最近三代短局仅 {short / games:.1%}，重采样无法替代源数据。",
                )
            if len(rounds) >= 8:
                reference = sum(item["games_per_second"] for item in rounds[:5]) / 5
                recent_speed = sum(item["games_per_second"] for item in recent) / len(recent)
                if reference and recent_speed < 0.8 * reference:
                    add(
                        "warning",
                        "自对弈吞吐下降",
                        f"最近 {recent_speed:.2f} 局/秒，早期基线 {reference:.2f} 局/秒。",
                    )
        promoted_rounds = [
            int(item.get("through_round", -1))
            for item in promotions
            if item.get("promote")
        ]
        if promotions:
            gap = int(promotions[-1].get("through_round", 0)) - (
                max(promoted_rounds) if promoted_rounds else -1
            )
            if gap >= 15:
                add("warning", "较长时间没有晋级", f"已经连续 {gap} 个候选未通过 gate。")
            recent_scores = [
                float(item.get("score", 0.0)) for item in promotions[-3:]
            ]
            if len(recent_scores) == 3 and sum(recent_scores) / 3 < 0.40:
                add(
                    "critical",
                    "近期候选明显退化",
                    f"最近三个 gate 平均得分 {sum(recent_scores) / 3:.1%}。",
                )
        if training:
            sampling = training[-1].get("sampling", {})
            sampled = sampling.get("length_sampled") or []
            target = sampling.get("length_target") or []
            if len(sampled) == 4 and len(target) == 4 and target[0] - sampled[0] > 0.10:
                add(
                    "warning",
                    "短局训练占比达不到目标",
                    f"实际采样 {sampled[0]:.1%}，目标 {target[0]:.1%}；当前受源数据和权重上限约束。",
                )
            initial = training[-1].get("initial_validation", {}).get("loss")
            final = training[-1].get("final_validation", {}).get("loss")
            if initial and final and final > initial * 1.02:
                add(
                    "warning",
                    "最新候选验证损失上升",
                    f"validation loss 从 {initial:.3f} 上升到 {final:.3f}。",
                )
        updated = float(state.get("updated_unix", 0.0))
        if state.get("status") == "running" and updated and time.time() - updated > 600:
            add("critical", "状态长时间未更新", "运行状态超过 10 分钟没有推进。")
        if not alerts:
            add("ok", "关键指标正常", "暂未发现需要立即处理的异常。")
        return alerts

    def snapshot(self, run_id: str) -> dict:
        record = self.resolve(run_id)
        if record is None:
            raise KeyError("训练任务不存在")
        run_path = Path(record.path)
        state = _read_json(run_path / "state.json")
        if not state:
            raise RuntimeError("state.json 正在更新或不可读")
        rounds: list[dict] = []
        for item in state.get("completed_rounds", []):
            enriched = item
            summary = item.get("summary", {})
            replay_directory = item.get("directory")
            if replay_directory and (
                "opening" not in summary or "diversity" not in summary
            ):
                enriched = dict(item)
                enriched["summary"] = dict(summary) | self._replay_diversity(
                    Path(replay_directory)
                )
            rounds.append(self._round_payload(enriched, rounds))
        promotions = [
            {
                "through_round": int(item.get("through_round", 0)),
                "promoted": bool(item.get("promote", False)),
                "score": float(item.get("score", 0.0)),
                "standard_error": float(item.get("standard_error", 0.0)),
                "lower_bound": float(item.get("lower_bound", 0.0)),
                "required_lower_bound": float(item.get("required_lower_bound", 0.5)),
                "seconds": max(
                    0.0,
                    float(item.get("gate_finished_unix", 0.0))
                    - float(item.get("gate_started_unix", 0.0)),
                ),
                "candidate_sha": str(item.get("candidate_sha256", ""))[:12],
                "baseline_sha": str(item.get("baseline_sha256", ""))[:12],
            }
            for item in state.get("promotions", [])
        ]
        training = self._training_payloads(run_path, state.get("promotions", []))

        games = sum(item["games"] for item in rounds)
        turns = sum(item["turns"] for item in rounds)
        stones = sum(item["stones"] for item in rounds)
        black = sum(item["black"] for item in rounds)
        white = sum(item["white"] for item in rounds)
        draw = sum(item["draw"] for item in rounds)
        lengths = {
            name: sum(item["length"][name] for item in rounds)
            for name in ("short", "medium", "long", "very_long")
        }
        first_started = (
            float(state.get("completed_rounds", [{}])[0].get("actor_started_unix", 0.0))
            if rounds
            else 0.0
        )
        last_finished = (
            float(state.get("completed_rounds", [{}])[-1].get("actor_finished_unix", 0.0))
            if rounds
            else 0.0
        )
        target_rounds = int(state.get("config", {}).get("rounds", 0))
        pace = _ratio(last_finished - first_started, len(rounds)) if rounds else 0.0
        eta_seconds = max(target_rounds - len(rounds), 0) * pace
        promoted_rounds = [item["through_round"] for item in promotions if item["promoted"]]
        config = {
            key: value
            for key, value in state.get("config", {}).items()
            if key not in {"output_dir", "initial_checkpoint"}
        }
        alerts = self._alerts(state, rounds, state.get("promotions", []), training)
        return {
            "run": {
                **record.public(),
                "status": str(state.get("status", record.status)),
                "completed": len(rounds),
                "target": target_rounds,
                "trained_through": int(state.get("trained_through", -1)),
                "promotion_decisions": len(promotions),
                "accepted_sha": str(state.get("accepted_sha256", ""))[:12],
                "last_promoted_round": max(promoted_rounds) if promoted_rounds else None,
                "updated_unix": float(state.get("updated_unix", record.updated_unix)),
                "seconds_per_round": pace,
                "eta_seconds": eta_seconds,
                "eta_unix": time.time() + eta_seconds if pace else None,
            },
            "cumulative": {
                "games": games,
                "turns": turns,
                "stones": stones,
                "mean_stones": _ratio(stones, games),
                "mean_turns": _ratio(turns, games),
                "black": black,
                "white": white,
                "draw": draw,
                "black_decisive": _ratio(black, black + white),
                "white_decisive": _ratio(white, black + white),
                "draw_rate": _ratio(draw, games),
                "length": lengths,
                "length_rate": {
                    name: _ratio(count, games) for name, count in lengths.items()
                },
                "invalid_audits": sum(item["invalid_audits"] for item in rounds),
                "mean_games_per_second": _ratio(
                    sum(item["games_per_second"] for item in rounds), len(rounds)
                ),
            },
            "rounds": rounds,
            "promotions": promotions,
            "training": training,
            "latest_training": training[-1] if training else None,
            "alerts": alerts,
            "config": config,
            "definitions": {
                "length": {
                    "short": "≤40 子",
                    "medium": "41–80 子",
                    "long": "81–140 子",
                    "very_long": ">140 子",
                },
                "color_target": [0.45, 0.55],
            },
        }


@dataclass(frozen=True)
class SearchSettings:
    simulations: int
    temperature: float
    top_k: int
    cpu_threads: int
    batch_size: int
    cpuct: float
    value_scale: float
    root_noise: float
    dirichlet_alpha: float
    compile_model: bool
    seed: int

    @classmethod
    def from_payload(cls, payload: dict) -> "SearchSettings":
        top_k = _number(payload.get("top_k"), 64, 1, 128, int)
        if top_k not in NATIVE_TOP_K:
            top_k = min(NATIVE_TOP_K, key=lambda value: abs(value - top_k))
        return cls(
            simulations=_number(
                payload.get("simulations", payload.get("sims")),
                400,
                1,
                MAX_SIMULATIONS,
                int,
            ),
            temperature=_number(payload.get("temperature"), 0.0, 0.0, 2.0, float),
            top_k=top_k,
            cpu_threads=_number(payload.get("cpu_threads"), 24, 1, MAX_CPU_THREADS, int),
            batch_size=_number(payload.get("batch_size"), 64, 1, 128, int),
            cpuct=_number(payload.get("cpuct"), 1.5, 0.1, 5.0, float),
            value_scale=_number(payload.get("value_scale"), 1.0, 0.0, 2.0, float),
            root_noise=_number(payload.get("root_noise"), 0.0, 0.0, 1.0, float),
            dirichlet_alpha=_number(
                payload.get("dirichlet_alpha"), 0.15, 0.01, 5.0, float
            ),
            compile_model=bool(payload.get("compile_model", False)),
            seed=_number(payload.get("seed"), secrets.randbits(31), 0, 2**31 - 1, int),
        )

    def runtime_key(self, kind: str) -> tuple:
        if kind == "native_pair":
            return (
                self.top_k,
                self.cpu_threads,
                self.cpuct,
                self.value_scale,
                self.root_noise,
                self.dirichlet_alpha,
                self.compile_model,
            )
        return (self.cpu_threads, self.batch_size, self.cpuct)


def _coord(move: int) -> str:
    if move == PASS_MOVE:
        return "PASS"
    row, column = divmod(int(move), BOARD_SIZE)
    return f"{'ABCDEFGHJKLMNOPQRST'[column]}{BOARD_SIZE - row}"


class NativeRuntime:
    kind = "native_pair"

    def __init__(self, record: ModelRecord, settings: SearchSettings, device: torch.device):
        self.record = record
        self.settings = settings
        self.device = device
        model, metadata = load_checkpoint(record.path, map_location="cpu")
        self.metadata = metadata
        self.evaluator = TorchPairEvaluator(
            model,
            device,
            top_k=settings.top_k,
            candidate_gumbel=0.0,
            seed=settings.seed,
            compile_model=settings.compile_model,
            deduplicate=False,
            value_scale=settings.value_scale,
        )
        self.engine = PairMCTSEngine(
            self.evaluator,
            top_k=settings.top_k,
            cpuct=settings.cpuct,
            root_noise_epsilon=settings.root_noise,
            root_dirichlet_alpha=settings.dirichlet_alpha,
            cpu_threads=settings.cpu_threads,
        )
        self.context = self.engine.create_context(settings.seed)

    def replay(self, actions: list[tuple[int, int]], moves: list[int]):
        del moves
        for first, second in actions:
            self.context.play(first, second)

    def search(
        self,
        simulations: int,
        temperature: float,
        progress: Callable[[int], None],
    ) -> dict:
        completed = 0
        total_evaluations = 0
        total_nodes = 0
        chunk_size = min(128, simulations)
        while completed < simulations:
            chunk = min(chunk_size, simulations - completed)
            self.engine.run([self.context], chunk)
            stats = self.context.stats()
            total_evaluations += int(stats["neural_evaluations"])
            total_nodes += int(stats["unique_nodes"])
            completed += chunk
            progress(completed)
        root_value = self.context.root_value()
        first, second, probabilities = self.context.policy()
        action = self.context.choose_action(temperature=temperature)
        candidates = [
            {
                "moves": [int(a)] if int(b) == PASS_MOVE else [int(a), int(b)],
                "coords": [_coord(int(a))]
                if int(b) == PASS_MOVE
                else [_coord(int(a)), _coord(int(b))],
                "probability": float(probability),
            }
            for a, b, probability in zip(first[:8], second[:8], probabilities[:8])
        ]
        return {
            "action": (int(action[0]), int(action[1])),
            "root_value": float(root_value),
            "candidates": candidates,
            "neural_evaluations": total_evaluations,
            "unique_nodes": total_nodes,
        }

    def play_action(self, first: int, second: int):
        self.context.play(first, second)

    def close(self):
        if getattr(self, "context", None) is not None:
            self.context.close()
            self.context = None
        self.engine = None
        self.evaluator = None
        if self.device.type == "cuda":
            torch.cuda.empty_cache()


class LegacyRuntime:
    kind = "legacy"

    def __init__(self, record: ModelRecord, settings: SearchSettings, device: torch.device):
        # Importing TensorRT is relatively expensive and unnecessary for a native-only visit.
        from reinforcement_learning.core.mcts import MCTSEngine

        self.record = record
        self.settings = settings
        self.device = device
        self.engine = MCTSEngine(record.path, device=device)
        self.engine.set_params(
            batch_size=settings.batch_size,
            num_threads=settings.cpu_threads,
        )
        self.engine.set_tree_batch_size(0)
        self.engine.set_unique_leaf_batching(False)
        self.engine.set_search_params(
            cpuct=settings.cpuct,
            widening_base=20,
            widening_scale=0.25,
        )
        self.engine.set_deterministic_selection(True)
        self.engine.set_eval_cache_capacity(32768)
        self.context = self.engine.create_game_context(settings.seed)

    def replay(self, actions: list[tuple[int, int]], moves: list[int]):
        del actions
        for move in moves:
            self.context.update_state(move)

    def search_one(
        self,
        simulations: int,
        temperature: float,
        progress: Callable[[int], None],
    ) -> dict:
        completed = 0
        chunk_size = min(256, simulations)
        while completed < simulations:
            chunk = min(chunk_size, simulations - completed)
            self.engine.run_simulations_multi([self.context], chunk)
            completed += chunk
            progress(completed)
        root_value = float(self.context.get_win_rate())
        policy = np.asarray(self.context.get_policy(), dtype=np.float32)
        ranked = np.argsort(policy)[::-1][:8]
        candidates = [
            {
                "moves": [int(move)],
                "coords": [_coord(int(move))],
                "probability": float(policy[move]),
            }
            for move in ranked
            if float(policy[move]) > 0.0
        ]
        move = int(self.context.get_mcts_move(simulations=0, temperature=temperature))
        return {
            "move": move,
            "root_value": root_value,
            "candidates": candidates,
        }

    def play_stone(self, move: int):
        self.context.update_state(move)

    def close(self):
        if getattr(self, "context", None) is not None:
            self.context.close()
            self.context = None
        self.engine = None
        if self.device.type == "cuda":
            torch.cuda.empty_cache()


class PairWebGame:
    def __init__(self, registry: ModelRegistry):
        self.registry = registry
        self.lock = threading.RLock()
        self.ai_lock = threading.Lock()
        self.search_lock = threading.Lock()
        self.device = torch.device(
            os.environ.get("NEBULA_PAIR_WEB_DEVICE", "cuda:0")
        )
        self.runtime: NativeRuntime | LegacyRuntime | None = None
        self.runtime_key: tuple | None = None
        self.loaded_model_id: str | None = None
        self.loaded_model_name: str | None = None
        self._clear_position()
        self._set_search_idle()

    def _clear_position(self):
        self.board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
        self.current_player = BLACK
        self.history: list[int] = []
        self.actions: list[tuple[int, int]] = []
        self.pending: list[int] = []
        self.game_over = False
        self.winner: int | None = None

    def _set_search_idle(self):
        with self.search_lock:
            self.search_info = {
                "status": "idle",
                "phase": "idle",
                "completed": 0,
                "total": 0,
                "started_at": None,
                "finished_at": None,
                "model": None,
                "error": None,
            }

    def _begin_search(self, total: int, model: str, phase: str = "loading"):
        with self.search_lock:
            self.search_info = {
                "status": "running",
                "phase": phase,
                "completed": 0,
                "total": int(total),
                "started_at": time.time(),
                "finished_at": None,
                "model": model,
                "error": None,
            }

    def _search_phase(self, phase: str):
        with self.search_lock:
            self.search_info["phase"] = phase

    def _search_progress(self, completed: int):
        with self.search_lock:
            self.search_info["completed"] = min(
                int(completed), int(self.search_info["total"])
            )

    def _finish_search(self, error: str | None = None):
        with self.search_lock:
            self.search_info["status"] = "error" if error else "done"
            self.search_info["phase"] = "error" if error else "done"
            if not error:
                self.search_info["completed"] = self.search_info["total"]
            self.search_info["finished_at"] = time.time()
            self.search_info["error"] = error

    def search_snapshot(self) -> dict:
        with self.search_lock:
            payload = dict(self.search_info)
        started = payload["started_at"]
        finished = payload["finished_at"]
        payload["elapsed"] = (
            max(0.0, (finished or time.time()) - started) if started else 0.0
        )
        total = int(payload["total"])
        payload["progress"] = payload["completed"] / total if total else 0.0
        return payload

    @staticmethod
    def _check_win(board: np.ndarray, move: int, player: int) -> bool:
        row, column = divmod(move, BOARD_SIZE)
        for delta_row, delta_column in ((0, 1), (1, 0), (1, 1), (1, -1)):
            count = 1
            for sign in (-1, 1):
                test_row = row + sign * delta_row
                test_column = column + sign * delta_column
                while (
                    0 <= test_row < BOARD_SIZE
                    and 0 <= test_column < BOARD_SIZE
                    and board[test_row, test_column] == player
                ):
                    count += 1
                    test_row += sign * delta_row
                    test_column += sign * delta_column
            if count >= 6:
                return True
        return False

    def _place(self, move: int, player: int) -> str:
        if not 0 <= move < BOARD_POINTS:
            return "invalid"
        row, column = divmod(move, BOARD_SIZE)
        if self.board[row, column] != 0:
            return "occupied"
        self.board[row, column] = player
        self.history.append(move)
        if self._check_win(self.board, move, player):
            self.game_over = True
            self.winner = player
            return "win"
        if len(self.history) == BOARD_POINTS:
            self.game_over = True
            self.winner = DRAW
            return "draw"
        return "continue"

    def _record_complete_action(self, first: int, second: int):
        self.actions.append((int(first), int(second)))
        self.pending.clear()
        self.current_player = -self.current_player

    def _runtime_play_native(self, first: int, second: int):
        if isinstance(self.runtime, NativeRuntime):
            self.runtime.play_action(first, second)

    def _runtime_play_stone(self, move: int):
        if isinstance(self.runtime, LegacyRuntime):
            self.runtime.play_stone(move)

    def play_human_stone(self, row: int, column: int) -> str:
        with self.lock:
            if self.game_over:
                return "game_over"
            move = row * BOARD_SIZE + column
            opening = len(self.history) == 0
            player = self.current_player
            result = self._place(move, player)
            if result in {"invalid", "occupied"}:
                return result
            self._runtime_play_stone(move)

            if opening:
                self._runtime_play_native(move, PASS_MOVE)
                self._record_complete_action(move, PASS_MOVE)
                return result

            if not self.pending:
                if self.game_over:
                    self._runtime_play_native(move, PASS_MOVE)
                    self._record_complete_action(move, PASS_MOVE)
                else:
                    self.pending.append(move)
                return result

            first_click = self.pending[0]
            first, second = sorted((first_click, move))
            self._runtime_play_native(first, second)
            self._record_complete_action(first, second)
            return result

    def _play_native_ai_action(self, first: int, second: int) -> list[int]:
        opening = len(self.history) == 0
        if opening != (second == PASS_MOVE):
            raise RuntimeError("Pair 模型返回了与当前回合不匹配的动作")
        player = self.current_player
        placed: list[int] = []
        result = self._place(first, player)
        if result in {"invalid", "occupied"}:
            raise RuntimeError(f"Pair 模型返回非法第一子：{first}")
        placed.append(first)
        if not self.game_over and second != PASS_MOVE:
            result = self._place(second, player)
            if result in {"invalid", "occupied"}:
                raise RuntimeError(f"Pair 模型返回非法第二子：{second}")
            placed.append(second)
        self._runtime_play_native(first, second)
        self._record_complete_action(first, second)
        return placed

    def _play_legacy_ai_stone(self, move: int) -> str:
        result = self._place(move, self.current_player)
        if result in {"invalid", "occupied"}:
            raise RuntimeError(f"旧模型返回非法落点：{move}")
        self._runtime_play_stone(move)
        return result

    def _finish_legacy_turn(self, ordered_moves: list[int]):
        if len(ordered_moves) == 1:
            action = (ordered_moves[0], PASS_MOVE)
        else:
            action = tuple(sorted(ordered_moves))
        self.actions.append((int(action[0]), int(action[1])))
        self.pending.clear()
        self.current_player = -self.current_player

    def _close_runtime(self):
        if self.runtime is not None:
            self.runtime.close()
        self.runtime = None
        self.runtime_key = None
        self.loaded_model_id = None
        self.loaded_model_name = None

    def _ensure_runtime(self, record: ModelRecord, settings: SearchSettings):
        key = (record.id, *settings.runtime_key(record.kind))
        if self.runtime is not None and self.runtime_key == key:
            return
        self._close_runtime()
        runtime_class = NativeRuntime if record.kind == "native_pair" else LegacyRuntime
        runtime = runtime_class(record, settings, self.device)
        try:
            runtime.replay(self.actions, self.history)
        except Exception:
            runtime.close()
            raise
        self.runtime = runtime
        self.runtime_key = key
        self.loaded_model_id = record.id
        self.loaded_model_name = record.name

    def ai_move(self, record: ModelRecord, settings: SearchSettings) -> dict:
        if not self.ai_lock.acquire(blocking=False):
            raise ValueError("已有一个 AI 搜索正在进行")
        try:
            return self._ai_move(record, settings)
        finally:
            self.ai_lock.release()

    def _ai_move(self, record: ModelRecord, settings: SearchSettings) -> dict:
        with self.lock:
            if self.game_over:
                raise ValueError("本局已经结束")
            if self.pending:
                raise ValueError("请先完成当前回合的第二颗棋子")

            stones_to_search = 1 if not self.history else 2
            total_budget = settings.simulations * (
                stones_to_search if record.kind == "legacy" else 1
            )
            self._begin_search(total_budget, record.name)
            started = time.perf_counter()
            try:
                self._ensure_runtime(record, settings)
                self._search_phase("searching")
                if isinstance(self.runtime, NativeRuntime):
                    result = self.runtime.search(
                        settings.simulations,
                        settings.temperature,
                        self._search_progress,
                    )
                    placed = self._play_native_ai_action(*result["action"])
                    search_details = result
                else:
                    placed = []
                    analyses = []
                    for stone_index in range(stones_to_search):
                        offset = stone_index * settings.simulations
                        result = self.runtime.search_one(
                            settings.simulations,
                            settings.temperature,
                            lambda value, base=offset: self._search_progress(base + value),
                        )
                        analyses.append(result)
                        placed.append(result["move"])
                        self._play_legacy_ai_stone(result["move"])
                        if self.game_over:
                            break
                    self._finish_legacy_turn(placed)
                    search_details = {
                        "root_value": analyses[0]["root_value"],
                        "candidates": analyses[0]["candidates"],
                        "stone_analyses": analyses,
                        "neural_evaluations": None,
                        "unique_nodes": None,
                    }
                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
                duration = time.perf_counter() - started
                self._finish_search()
            except Exception as error:
                self._finish_search(str(error))
                raise

            return {
                **self.snapshot(),
                "moves": [[move // BOARD_SIZE, move % BOARD_SIZE] for move in placed],
                "move_indices": placed,
                "duration": duration,
                "root_value": search_details["root_value"],
                "candidates": search_details["candidates"],
                "stone_analyses": search_details.get("stone_analyses"),
                "neural_evaluations": search_details.get("neural_evaluations"),
                "unique_nodes": search_details.get("unique_nodes"),
                "search": {
                    **asdict(settings),
                    "total_effective_simulations": total_budget,
                    "budget_semantics": (
                        "per_turn" if record.kind == "native_pair" else "per_stone"
                    ),
                    "model_kind": record.kind,
                },
            }

    def _rebuild(self, moves: list[int]):
        runtime_record = None
        runtime_settings = None
        if self.runtime is not None:
            runtime_record = self.runtime.record
            runtime_settings = self.runtime.settings
        self._close_runtime()
        self._clear_position()
        for move in moves:
            row, column = divmod(move, BOARD_SIZE)
            result = self.play_human_stone(row, column)
            if result in {"invalid", "occupied", "game_over"}:
                raise ValueError(f"非法棋谱落点：{_coord(move)}")
            if self.game_over and move != moves[-1]:
                raise ValueError("棋谱在终局后仍有落子")
        if runtime_record is not None and runtime_settings is not None:
            self._ensure_runtime(runtime_record, runtime_settings)

    def reset(self):
        with self.lock:
            record = self.runtime.record if self.runtime is not None else None
            settings = self.runtime.settings if self.runtime is not None else None
            self._close_runtime()
            self._clear_position()
            if record is not None and settings is not None:
                self._ensure_runtime(record, settings)
            self._set_search_idle()

    def undo_turns(self, turns: int) -> int:
        with self.lock:
            if not self.history:
                return 0
            remaining = list(self.history)
            removed = 0
            for _ in range(max(1, turns)):
                if not remaining:
                    break
                # Complete positions contain 1 + 2n stones.  Even lengths are
                # an unfinished human turn and therefore lose one pending stone.
                count = 1 if len(remaining) % 2 == 0 or len(remaining) == 1 else 2
                count = min(count, len(remaining))
                del remaining[-count:]
                removed += count
            self._rebuild(remaining)
            self._set_search_idle()
            return removed

    def load_moves(self, moves: list[int]):
        with self.lock:
            previous = list(self.history)
            try:
                self._rebuild(moves)
            except Exception:
                self._rebuild(previous)
                raise
            self._set_search_idle()

    def snapshot(self) -> dict:
        with self.lock:
            history = []
            for index, move in enumerate(self.history):
                if index == 0:
                    player = BLACK
                    turn = 1
                    stone_in_turn = 1
                else:
                    turn = 2 + (index - 1) // 2
                    player = WHITE if turn % 2 == 0 else BLACK
                    stone_in_turn = 1 + (index - 1) % 2
                row, column = divmod(move, BOARD_SIZE)
                history.append(
                    {
                        "index": index + 1,
                        "row": row,
                        "col": column,
                        "move": move,
                        "coord": _coord(move),
                        "player": player,
                        "turn": turn,
                        "stone_in_turn": stone_in_turn,
                    }
                )
            stones_remaining = 1 if not self.history else 1 if self.pending else 2
            if self.game_over:
                stones_remaining = 0
            return {
                "board": self.board.tolist(),
                "current_player": self.current_player,
                "history": [[item["row"], item["col"]] for item in history],
                "history_detail": history,
                "pending": [
                    [move // BOARD_SIZE, move % BOARD_SIZE] for move in self.pending
                ],
                "stones_remaining": stones_remaining,
                "game_over": self.game_over,
                "winner": self.winner,
                "loaded_model_id": self.loaded_model_id,
                "loaded_model_name": self.loaded_model_name,
                "loaded_model_kind": self.runtime.kind if self.runtime else None,
            }


TEMPLATE_DIR = Path(__file__).with_name("web") / "templates"
app = Flask(__name__, template_folder=str(TEMPLATE_DIR))
registry = ModelRegistry()
monitor = TrainingMonitor(registry.training_root)
game = PairWebGame(registry)


@app.after_request
def no_cache(response):
    response.headers["Cache-Control"] = "no-store"
    return response


@app.get("/")
def index():
    return render_template("monitor.html")


@app.get("/monitor")
def monitor_page():
    return render_template("monitor.html")


@app.get("/play")
def play_page():
    return render_template("index.html")


@app.get("/api/health")
def health():
    return jsonify(
        {
            "status": "ok",
            "device": str(game.device),
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": (
                torch.cuda.get_device_name(game.device)
                if game.device.type == "cuda" and torch.cuda.is_available()
                else None
            ),
        }
    )


@app.get("/api/models")
def models():
    records, default_id = registry.records()
    return jsonify(
        {
            "status": "ok",
            "models": [record.public() for record in records],
            "default_id": default_id,
        }
    )


@app.get("/api/runs")
def training_runs():
    records, default_id = monitor.runs()
    return jsonify(
        {
            "status": "ok",
            "runs": [record.public() for record in records],
            "default_id": default_id,
        }
    )


@app.get("/api/training")
def training_snapshot():
    records, default_id = monitor.runs()
    del records
    run_id = request.args.get("run_id") or default_id
    if not run_id:
        return jsonify({"status": "error", "message": "没有发现训练任务"}), 404
    try:
        payload = monitor.snapshot(run_id)
    except KeyError as error:
        return jsonify({"status": "error", "message": str(error)}), 404
    except Exception as error:
        app.logger.exception("training monitor failed")
        return jsonify({"status": "error", "message": f"读取训练状态失败：{error}"}), 503
    return jsonify({"status": "ok", **payload})


@app.get("/api/state")
def state():
    return jsonify({"status": "ok", **game.snapshot()})


@app.get("/api/search")
def search_status():
    return jsonify({"status": "ok", **game.search_snapshot()})


@app.post("/api/move")
def human_move():
    payload = request.get_json(silent=True) or {}
    try:
        row = int(payload["row"])
        column = int(payload["col"])
    except (KeyError, TypeError, ValueError):
        return jsonify({"status": "error", "message": "落子坐标格式错误"}), 400
    result = game.play_human_stone(row, column)
    if result in {"continue", "win", "draw"}:
        return jsonify({"status": "ok", "result": result, **game.snapshot()})
    messages = {
        "occupied": "这个交叉点已经有棋子",
        "invalid": "落点超出棋盘",
        "game_over": "本局已经结束",
    }
    return jsonify({"status": "error", "message": messages[result]}), 409


@app.post("/api/ai-move")
def ai_move():
    payload = request.get_json(silent=True) or {}
    record = registry.resolve(str(payload.get("model_id", "")))
    if record is None:
        return jsonify({"status": "error", "message": "所选模型不存在，请刷新列表"}), 400
    settings = SearchSettings.from_payload(payload)
    try:
        result = game.ai_move(record, settings)
    except ValueError as error:
        return jsonify({"status": "error", "message": str(error)}), 409
    except Exception as error:
        app.logger.exception("AI search failed")
        return jsonify({"status": "error", "message": f"搜索失败：{error}"}), 500
    return jsonify({"status": "ok", **result})


@app.post("/api/reset")
def reset():
    game.reset()
    return jsonify({"status": "ok", **game.snapshot()})


@app.post("/api/undo")
def undo():
    payload = request.get_json(silent=True) or {}
    turns = _number(payload.get("turns"), 1, 1, 4, int)
    removed = game.undo_turns(turns)
    if not removed:
        return jsonify({"status": "error", "message": "当前没有可撤销的着法"}), 409
    return jsonify({"status": "ok", "removed": removed, **game.snapshot()})


@app.post("/api/load")
def load_record():
    payload = request.get_json(silent=True) or {}
    raw_moves = payload.get("moves")
    if not isinstance(raw_moves, list) or len(raw_moves) > BOARD_POINTS:
        return jsonify({"status": "error", "message": "moves 必须是合法数组"}), 400
    parsed: list[int] = []
    try:
        for item in raw_moves:
            if isinstance(item, dict):
                row = int(item.get("row", item.get("r")))
                column = int(item.get("col", item.get("c")))
            else:
                row, column = int(item[0]), int(item[1])
            if not (0 <= row < BOARD_SIZE and 0 <= column < BOARD_SIZE):
                raise ValueError("坐标越界")
            parsed.append(row * BOARD_SIZE + column)
        game.load_moves(parsed)
    except (TypeError, ValueError, IndexError) as error:
        return jsonify({"status": "error", "message": f"棋谱无效：{error}"}), 400
    return jsonify({"status": "ok", **game.snapshot()})


def main() -> int:
    if game.device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("NEBULA_PAIR_WEB_DEVICE requested CUDA but CUDA is unavailable")
    host = os.environ.get("NEBULA_PAIR_WEB_HOST", "127.0.0.1")
    port = int(os.environ.get("NEBULA_PAIR_WEB_PORT", "18473"))
    app.run(host=host, port=port, debug=False, threaded=True, use_reloader=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
