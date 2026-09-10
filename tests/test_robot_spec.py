from types import SimpleNamespace

import numpy as np
from inspect_robots import Observation

from gpt_dog_eval.agent import JointChunkAgentPolicy
from gpt_dog_eval.inference import FakeInferenceProvider, InferenceRequest, InferenceResponse
from gpt_dog_eval.robot_spec import robot_spec_context, robot_spec_sha256


def _observation(*, step: int = 0, with_trainer_terms: bool = False) -> Observation:
    return Observation(
        state={"policy_obs": np.zeros(48), "last_applied_action": np.zeros(12)},
        state_time=step / 50.0,
        extra={"last_trainer_terms": {"energy": 123.0}} if with_trainer_terms else {},
    )


class CaptureProvider(FakeInferenceProvider):
    def __init__(self) -> None:
        super().__init__(chunk_steps=1)
        self.requests: list[InferenceRequest] = []

    def complete(self, request: InferenceRequest) -> InferenceResponse:
        self.requests.append(request)
        return super().complete(request)


def test_static_spec_reaches_every_request_and_survives_scene_reset() -> None:
    spec = {"schema_version": 1, "nominal_joint_angles": [0.1, 0.9, -1.8] * 4}
    digest = robot_spec_sha256(spec)
    context = robot_spec_context(spec)
    provider = CaptureProvider()
    policy = JointChunkAgentPolicy(config_path=None, inference_provider=provider, robot_spec=spec)
    spec["nominal_joint_angles"] = [99.0] * 12
    policy.reset(SimpleNamespace(instruction="first scene"))  # type: ignore[arg-type]
    policy.act(_observation(with_trainer_terms=True))
    policy.act(_observation(step=1))
    record = SimpleNamespace(steps=[], metadata={})
    policy.on_trial_end(record, "logs", "test")
    assert record.metadata["robot_spec"]["sha256"] == digest
    policy.reset(SimpleNamespace(instruction="second scene"))  # type: ignore[arg-type]
    chunk = policy.act(_observation())
    for request in provider.requests:
        assert request.instructions.endswith(context)
        assert "last_trainer_terms" not in str(request)
    assert "first scene" not in str(provider.requests[-1].input_items)
    assert policy.transcript()[0]["content"] == provider.requests[-1].instructions
    np.testing.assert_array_equal(chunk.actions[0].data, np.zeros(12))
