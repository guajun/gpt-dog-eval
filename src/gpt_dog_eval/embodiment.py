"""MuJoCo Playground embodiment for the Unitree Go1 joystick task."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from inspect_robots import (
    Action,
    EmbodimentInfo,
    Observation,
    Scene,
    StepResult,
)
from inspect_robots.embodiment import AUTO_RESET, RESETTABLE, SEEDABLE

from gpt_dog_eval.constants import (
    ACTION_SPACE,
    EMBODIMENT_DOCS,
    OBSERVATION_SPACE,
    split_policy_observation,
)


class Go1PlaygroundEmbodiment:
    """Adapt Playground's exact Go1 trainer environment to Inspect Robots."""

    def __init__(
        self,
        *,
        env_name: str = "Go1JoystickFlatTerrain",
        noise_level: float = 0.0,
        perturbations: bool = False,
        jit: bool = True,
    ) -> None:
        if env_name not in {"Go1JoystickFlatTerrain", "Go1JoystickRoughTerrain"}:
            raise ValueError("env_name must be a supported Go1 joystick environment")
        if not 0.0 <= noise_level <= 1.0:
            raise ValueError("noise_level must be between 0 and 1")

        self.env_name = env_name
        self.noise_level = float(noise_level)
        self.perturbations = bool(perturbations)
        self.jit = bool(jit)
        self.info = EmbodimentInfo(
            name="go1-playground",
            action_space=ACTION_SPACE,
            observation_space=OBSERVATION_SPACE,
            control_hz=50.0,
            is_simulated=True,
            capabilities=frozenset({SEEDABLE, RESETTABLE, AUTO_RESET}),
            supported_target_kinds=frozenset({"velocity_tracking"}),
            docs=EMBODIMENT_DOCS,
        )

        self._env: Any | None = None
        self._state: Any | None = None
        self._jax: Any | None = None
        self._jp: Any | None = None
        self._reset_fn: Any | None = None
        self._step_fn: Any | None = None
        self._instruction: str | None = None
        self._step_index = 0
        self._scales: dict[str, float] = {}

    def _ensure_env(self) -> Any:
        if self._env is not None:
            return self._env

        import jax
        import jax.numpy as jp
        from mujoco_playground import locomotion

        config = locomotion.get_default_config(self.env_name)
        config.impl = "jax"
        config.noise_config.level = self.noise_level
        config.pert_config.enable = self.perturbations
        env = locomotion.load(self.env_name, config=config)

        self._env = env
        self._jax = jax
        self._jp = jp
        self._reset_fn = jax.jit(env.reset) if self.jit else env.reset
        self._step_fn = jax.jit(env.step) if self.jit else env.step
        self._scales = {
            str(key): float(value) for key, value in config.reward_config.scales.items()
        }
        return env

    @staticmethod
    def _command(scene: Scene) -> np.ndarray:
        raw = scene.metadata.get("command", (0.0, 0.0, 0.0))
        command = np.asarray(raw, dtype=np.float32).reshape(-1)
        if command.shape != (3,) or not np.all(np.isfinite(command)):
            raise ValueError("scene metadata command must contain three finite values")
        return command

    def reset(self, scene: Scene, *, seed: int | None = None) -> Observation:
        """Reset deterministically and replace the sampled command with the scene command."""
        env = self._ensure_env()
        assert self._jax is not None
        assert self._jp is not None
        assert self._reset_fn is not None

        state = self._reset_fn(self._jax.random.PRNGKey(0 if seed is None else seed))
        info = dict(state.info)
        info["command"] = self._jp.asarray(self._command(scene))
        info["steps_until_next_cmd"] = self._jp.asarray(2**30, dtype=self._jp.int32)

        # The upstream reset observation contains its randomly sampled command.
        # Recompute it after pinning the benchmark command so policies see the
        # same command that the trainer reward evaluates.
        obs = env._get_obs(state.data, info)
        state = state.replace(obs=obs, info=info)

        self._state = state
        self._instruction = scene.instruction
        self._step_index = 0
        return self._observation(state)

    def step(self, action: Action) -> StepResult:
        """Advance one trainer control step and preserve every trainer term."""
        if self._state is None or self._step_fn is None or self._jp is None:
            raise RuntimeError("reset() must be called before step()")

        command = np.asarray(action.data, dtype=np.float32).reshape(-1)
        if command.shape != (12,):
            raise ValueError(f"action must have shape (12,), got {command.shape}")
        state = self._step_fn(self._state, self._jp.asarray(command))
        self._state = state
        self._step_index += 1

        weighted = {
            key.removeprefix("reward/"): float(np.asarray(value))
            for key, value in state.metrics.items()
            if key.startswith("reward/")
        }
        raw = {
            key: value / self._scales[key]
            for key, value in weighted.items()
            if self._scales.get(key, 0.0) != 0.0
        }
        costs = {key: raw[key] for key, scale in self._scales.items() if scale < 0 and key in raw}
        rewards = {key: raw[key] for key, scale in self._scales.items() if scale > 0 and key in raw}
        done = bool(np.asarray(state.done))

        return StepResult(
            observation=self._observation(state, trainer_terms=raw),
            reward=float(np.asarray(state.reward)),
            terminated=done,
            termination_reason="fell" if done else None,
            info={
                "fell": done,
                "step": self._step_index,
                "command": np.asarray(state.info["command"], dtype=np.float32).tolist(),
                "trainer": {
                    "weighted_terms": weighted,
                    "raw_terms": raw,
                    "cost_terms": costs,
                    "reward_terms": rewards,
                },
            },
        )

    def _observation(
        self,
        state: Any,
        *,
        trainer_terms: Mapping[str, float] | None = None,
    ) -> Observation:
        assert self._env is not None
        raw_obs = state.obs["state"] if isinstance(state.obs, Mapping) else state.obs
        fields = split_policy_observation(np.asarray(raw_obs))
        fields.update(
            {
                "base_pos": np.asarray(state.data.qpos[:3], dtype=np.float32),
                "base_quat": np.asarray(state.data.qpos[3:7], dtype=np.float32),
                "joint_pos": np.asarray(state.data.qpos[7:], dtype=np.float32),
                "actuator_force": np.asarray(state.data.actuator_force, dtype=np.float32),
            }
        )
        extra: dict[str, Any] = {"sim_step": self._step_index}
        if trainer_terms is not None:
            extra["last_trainer_terms"] = dict(trainer_terms)
        return Observation(
            state=fields,
            instruction=self._instruction,
            state_time=self._step_index / 50.0,
            extra=extra,
        )

    def close(self) -> None:
        """Release references to compiled environment state."""
        self._state = None
        self._env = None
        self._reset_fn = None
        self._step_fn = None
        self._jax = None
        self._jp = None


def go1_playground(**kwargs: Any) -> Go1PlaygroundEmbodiment:
    """Create the registered CPU-capable Go1 Playground embodiment."""
    return Go1PlaygroundEmbodiment(**kwargs)
