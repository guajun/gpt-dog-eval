"""Deterministic Unitree Go1 velocity-tracking benchmark tasks."""

from __future__ import annotations

from typing import Any

from inspect_robots import Scene, Target, Task

from gpt_dog_eval.scorers import EpisodeReturn, MeanTrainerTerm, Survived

_SCENES = {
    "stand": ("Stand still while keeping the body level.", (0.0, 0.0, 0.0), 11),
    "forward": ("Walk forward at 0.5 metres per second.", (0.5, 0.0, 0.0), 22),
    "left": ("Walk left at 0.3 metres per second.", (0.0, 0.3, 0.0), 33),
    "turn": ("Turn left in place at 0.5 radians per second.", (0.0, 0.0, 0.5), 44),
}


def _scene(name: str) -> Scene:
    instruction, command, seed = _SCENES[name]
    return Scene(
        id=name,
        instruction=instruction,
        target=Target(kind="velocity_tracking", spec={"command": command}),
        init_seed=seed,
        metadata={"command": command},
    )


def go1_velocity_smoke(
    *,
    steps: int = 250,
    scenes: str = "stand,forward,left,turn",
    epochs: int = 1,
    **kwargs: Any,
) -> Task:
    """Create a fixed-command suite scored with the upstream trainer terms."""
    if kwargs:
        raise TypeError(f"unknown go1-velocity-smoke options: {sorted(kwargs)}")
    if steps < 1:
        raise ValueError("steps must be >= 1")
    requested = tuple(item.strip() for item in scenes.split(",") if item.strip())
    unknown = sorted(set(requested) - set(_SCENES))
    if not requested or unknown:
        raise ValueError(f"scenes must select from {sorted(_SCENES)}; unknown={unknown}")

    scorers = (
        EpisodeReturn(),
        Survived(),
        MeanTrainerTerm("tracking_lin_vel"),
        MeanTrainerTerm("tracking_ang_vel"),
        MeanTrainerTerm("orientation"),
        MeanTrainerTerm("torques"),
        MeanTrainerTerm("action_rate"),
        MeanTrainerTerm("energy"),
        MeanTrainerTerm("feet_slip"),
        MeanTrainerTerm("dof_pos_limits"),
    )
    return Task(
        name="go1-velocity-smoke",
        scenes=tuple(_scene(name) for name in requested),
        scorer=scorers,
        max_steps=steps,
        epochs=epochs,
        metadata={
            "source_environment": "MuJoCo Playground Go1JoystickFlatTerrain",
            "trainer_costs": True,
        },
    )
