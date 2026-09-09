"""Trajectory scorers derived from MuJoCo Playground trainer metrics."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import fmean

from inspect_robots import Score, Target
from inspect_robots.rollout import TrialRecord


@dataclass(frozen=True)
class EpisodeReturn:
    """Sum the exact clipped per-step reward returned by the trainer environment."""

    name: str = "episode_return"

    def __call__(self, record: TrialRecord, target: Target | None) -> Score:
        values = [float(step.result.reward or 0.0) for step in record.steps]
        return Score(value=sum(values), metadata={"steps": len(values)})


@dataclass(frozen=True)
class Survived:
    """Report whether the robot avoided the upstream fall termination."""

    name: str = "survived"

    def __call__(self, record: TrialRecord, target: Target | None) -> Score:
        survived = record.termination_reason != "fell"
        return Score(value=survived, explanation="no fall" if survived else "fell")


@dataclass(frozen=True)
class MeanTrainerTerm:
    """Average one unscaled trainer reward or cost term over recorded steps."""

    term: str

    @property
    def name(self) -> str:
        """Use a stable metric key that retains the upstream term name."""
        return f"trainer_mean/{self.term}"

    def __call__(self, record: TrialRecord, target: Target | None) -> Score:
        values: list[float] = []
        for step in record.steps:
            trainer = step.result.info.get("trainer", {})
            raw = trainer.get("raw_terms", {}) if isinstance(trainer, dict) else {}
            if self.term in raw:
                values.append(float(raw[self.term]))
        if not values:
            return Score(value=0.0, explanation=f"trainer term {self.term!r} was not recorded")
        return Score(value=fmean(values), metadata={"samples": len(values)})
