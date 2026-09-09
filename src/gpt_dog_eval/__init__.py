"""Inspect Robots components for headless Unitree Go1 evaluation."""

from gpt_dog_eval.embodiment import Go1PlaygroundEmbodiment, go1_playground
from gpt_dog_eval.policies import OnnxGo1Policy, ZeroGo1Policy, go1_onnx, go1_zero
from gpt_dog_eval.tasks import go1_velocity_smoke

__all__ = [
    "Go1PlaygroundEmbodiment",
    "OnnxGo1Policy",
    "ZeroGo1Policy",
    "go1_onnx",
    "go1_playground",
    "go1_velocity_smoke",
    "go1_zero",
]
