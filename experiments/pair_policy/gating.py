"""Backward-compatible import path for the shared production gating rules."""

from reinforcement_learning.gating import (  # noqa: F401
    approximate_improvement_probability,
    gate_passes,
    paired_score_statistics,
)


__all__ = [
    "approximate_improvement_probability",
    "gate_passes",
    "paired_score_statistics",
]
