from __future__ import annotations

import threading
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from inspect_robots import Observation

from gpt_dog_eval.agent import JointChunkAgentPolicy, ToolValidationError, _compile_joint_chunk
from gpt_dog_eval.fake_server import make_fake_server
from gpt_dog_eval.inference import (
    FakeInferenceProvider,
    InferenceConfig,
    ResponsesInferenceProvider,
    create_inference_provider,
    register_inference_provider,
)


def _observation(
    *, step: int = 0, last_action: float = 0.0, with_trainer_terms: bool = False
) -> Observation:
    state = np.zeros(48, dtype=np.float32)
    state[33:45] = last_action
    extra = {"last_trainer_terms": {"energy": 123.0}} if with_trainer_terms else {}
    return Observation(state={"policy_obs": state}, state_time=step / 50.0, extra=extra)


def test_fake_provider_drives_joint_chunk_policy_without_network() -> None:
    fake = FakeInferenceProvider(chunk_steps=4)
    policy = JointChunkAgentPolicy(
        config_path=None,
        max_chunk_steps=5,
        inference_provider=fake,
    )
    policy.reset(SimpleNamespace(instruction="Stand still."))  # type: ignore[arg-type]
    chunk = policy.act(_observation())
    assert len(chunk) == 4
    assert fake.calls == 1
    np.testing.assert_array_equal(chunk.actions[-1].data, np.zeros(12, dtype=np.float32))

    policy.act(_observation(step=4))
    assert any(item.get("role") == "tool" for item in policy.transcript())


def test_agent_context_does_not_leak_trainer_costs() -> None:
    policy = JointChunkAgentPolicy(
        config_path=None,
        max_chunk_steps=5,
        inference_provider=FakeInferenceProvider(chunk_steps=2),
    )
    policy.reset(SimpleNamespace(instruction="Stand still."))  # type: ignore[arg-type]
    policy.act(_observation(with_trainer_terms=True))
    observations = [
        item["content"]
        for item in policy.transcript()
        if item.get("role") == "user"
        and isinstance(item.get("content"), str)
        and item["content"].startswith("Observation ")
    ]
    assert len(observations) == 1
    assert "trainer" not in observations[0].lower()
    assert "energy" not in observations[0].lower()
    assert "123.000" not in observations[0]


def test_joint_chunk_interpolates_and_enforces_slew_limit() -> None:
    target = [0.2] * 12
    chunk = _compile_joint_chunk(
        {
            "keyframes": [{"step": 2, "target": target}],
            "total_steps": 2,
            "note": "Move smoothly.",
        },
        current=np.zeros(12),
        max_chunk_steps=5,
        max_keyframes=4,
        max_action_delta=0.1,
    )
    np.testing.assert_allclose(chunk[0], np.full(12, 0.1))
    np.testing.assert_allclose(chunk[1], np.full(12, 0.2))

    with pytest.raises(ToolValidationError, match="maximum"):
        _compile_joint_chunk(
            {
                "keyframes": [{"step": 1, "target": [1.0] * 12}],
                "total_steps": 1,
                "note": "Too fast.",
            },
            current=np.zeros(12),
            max_chunk_steps=5,
            max_keyframes=4,
            max_action_delta=0.1,
        )


def test_custom_provider_can_be_registered() -> None:
    marker = FakeInferenceProvider(chunk_steps=2)

    def factory(config: InferenceConfig) -> FakeInferenceProvider:
        return marker

    register_inference_provider("unit-test-provider", factory)
    config = InferenceConfig(provider="unit-test-provider")
    assert create_inference_provider(config) is marker


def test_responses_provider_uses_common_base_url_and_api_key() -> None:
    server = make_fake_server("127.0.0.1", 0, api_key="test-key", chunk_steps=3)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.host_port
    provider = ResponsesInferenceProvider(
        InferenceConfig(
            provider="responses",
            base_url=f"http://{host}:{port}/v1",
            api_key="test-key",
            model="fake-go1",
            reasoning_effort=None,
        )
    )
    try:
        policy = JointChunkAgentPolicy(
            config_path=None,
            max_chunk_steps=5,
            inference_provider=provider,
        )
        policy.reset(SimpleNamespace(instruction="Stand still."))  # type: ignore[arg-type]
        chunk = policy.act(_observation())
        assert len(chunk) == 3
    finally:
        provider.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_secret_toml_loads_without_exposing_key(tmp_path: Any) -> None:
    path = tmp_path / "secret.toml"
    path.write_text(
        """[inference]
provider = "fake"
base_url = "http://localhost/v1"
api_key = "private"
model = "fake-go1"
timeout_s = 5.0
""",
        encoding="utf-8",
    )
    config = InferenceConfig.from_toml(path)
    assert config.api_key == "private"
    assert config.base_url == "http://localhost/v1"
