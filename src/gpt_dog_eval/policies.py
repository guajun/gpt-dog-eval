"""Non-LLM policies used to validate the Go1 evaluation stack."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
from inspect_robots import Action, ActionChunk, Observation, PolicyConfig, PolicyInfo, Scene

from gpt_dog_eval.constants import ACTION_SPACE, POLICY_OBSERVATION_SPACE


class ZeroGo1Policy:
    """Request the nominal joint pose at every step."""

    def __init__(self) -> None:
        self.info = PolicyInfo(
            name="go1-zero",
            action_space=ACTION_SPACE,
            observation_space=POLICY_OBSERVATION_SPACE,
            control_hz=50.0,
        )
        self.config = PolicyConfig(action_horizon=1)

    def reset(self, scene: Scene) -> None:
        """Reset the stateless baseline."""

    def act(self, observation: Observation) -> ActionChunk:
        """Return one zero residual action."""
        return ActionChunk(
            actions=(Action(data=np.zeros(12, dtype=np.float32)),),
            control_hz=50.0,
            inference_latency_s=0.0,
        )


class OnnxGo1Policy:
    """Run MuJoCo Playground's bundled pretrained Go1 ONNX policy on CPU."""

    def __init__(self, *, model_path: str | None = None) -> None:
        self.info = PolicyInfo(
            name="go1-onnx",
            action_space=ACTION_SPACE,
            observation_space=POLICY_OBSERVATION_SPACE,
            control_hz=50.0,
        )
        self.config = PolicyConfig(action_horizon=1)
        self.model_path = model_path
        self._session: Any | None = None
        self._input_name: str | None = None

    def reset(self, scene: Scene) -> None:
        """Keep the immutable ONNX session across scenes."""

    def _ensure_session(self) -> Any:
        if self._session is not None:
            return self._session

        import mujoco_playground
        import onnxruntime as ort

        if self.model_path is None:
            package_root = Path(mujoco_playground.__file__).resolve().parent
            path = package_root / "experimental" / "sim2sim" / "onnx" / "go1_policy.onnx"
        else:
            path = Path(self.model_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Go1 ONNX policy not found: {path}")

        self._session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        self._input_name = self._session.get_inputs()[0].name
        return self._session

    def act(self, observation: Observation) -> ActionChunk:
        """Map the exact upstream actor observation to one normalized action."""
        obs = np.asarray(observation.state["policy_obs"], dtype=np.float32).reshape(1, 48)
        session = self._ensure_session()
        assert self._input_name is not None
        started = time.perf_counter()
        output = session.run(None, {self._input_name: obs})[0]
        latency = time.perf_counter() - started
        action = np.asarray(output[0], dtype=np.float32).reshape(12)
        return ActionChunk(
            actions=(Action(data=action),),
            control_hz=50.0,
            inference_latency_s=latency,
        )


def go1_zero(**kwargs: Any) -> ZeroGo1Policy:
    """Create the deterministic nominal-pose baseline."""
    if kwargs:
        raise TypeError(f"go1-zero does not accept options: {sorted(kwargs)}")
    return ZeroGo1Policy()


def go1_onnx(**kwargs: Any) -> OnnxGo1Policy:
    """Create the upstream pretrained ONNX locomotion baseline."""
    return OnnxGo1Policy(**kwargs)
