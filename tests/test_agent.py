from __future__ import annotations

import threading
from dataclasses import replace
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
    InferenceRequest,
    InferenceResponse,
    ResponsesInferenceProvider,
    ToolCall,
    create_inference_provider,
    register_inference_provider,
)


def _observation(
    *,
    step: int = 0,
    last_action: float = 0.0,
    applied_action: float = 0.0,
    with_trainer_terms: bool = False,
) -> Observation:
    state = np.zeros(48, dtype=np.float32)
    state[33:45] = last_action
    extra = {"last_trainer_terms": {"energy": 123.0}} if with_trainer_terms else {}
    return Observation(
        state={"policy_obs": state, "last_applied_action": np.full(12, applied_action)},
        state_time=step / 50.0,
        extra=extra,
    )


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


def test_trial_end_persists_interrupted_chunk_execution() -> None:
    policy = JointChunkAgentPolicy(
        config_path=None,
        max_chunk_steps=5,
        inference_provider=FakeInferenceProvider(chunk_steps=4),
    )
    policy.reset(SimpleNamespace(instruction="Stand still."))  # type: ignore[arg-type]
    chunk = policy.act(_observation())
    record = SimpleNamespace(
        steps=[SimpleNamespace(action=action) for action in chunk.actions[:2]],
        termination_reason="max_steps",
        error=None,
        status="success",
        metadata={},
        policy_transcript=policy.transcript(),
    )

    policy.on_trial_end(record, "logs", "run-id")

    expected = {
        "status": "interrupted",
        "requested_steps": 4,
        "executed_steps": 2,
        "stop_reason": "max_steps",
        "execution_feedback_complete": True,
        "last_applied_action": [0.0] * 12,
        "modified_steps": 0,
        "max_action_error": 0.0,
        "modifications": [],
    }
    assert record.policy_transcript[-1] == {
        "role": "tool",
        "tool_call_id": "fake_call_1",
        "content": expected,
    }
    assert record.metadata["chunk_executions"] == [
        {
            "tool_call_id": "fake_call_1",
            "tool_name": "run_joint_chunk",
            **expected,
        }
    ]


def test_trial_end_records_completed_chunk_at_exact_horizon() -> None:
    policy = JointChunkAgentPolicy(
        config_path=None,
        max_chunk_steps=5,
        inference_provider=FakeInferenceProvider(chunk_steps=2),
    )
    policy.reset(SimpleNamespace(instruction="Stand still."))  # type: ignore[arg-type]
    chunk = policy.act(_observation())
    record = SimpleNamespace(
        steps=[SimpleNamespace(action=action) for action in chunk.actions],
        termination_reason="max_steps",
        error=None,
        status="success",
        metadata={},
        policy_transcript=policy.transcript(),
    )

    policy.on_trial_end(record, "logs", "run-id")

    assert record.metadata["chunk_executions"][0] == {
        "tool_call_id": "fake_call_1",
        "tool_name": "run_joint_chunk",
        "status": "completed",
        "requested_steps": 2,
        "executed_steps": 2,
        "stop_reason": "max_steps",
        "execution_feedback_complete": True,
        "last_applied_action": [0.0] * 12,
        "modified_steps": 0,
        "max_action_error": 0.0,
        "modifications": [],
    }


def test_fake_provider_default_chunk_fits_episode_call_budget() -> None:
    fake = FakeInferenceProvider()
    assert fake.chunk_steps == 15
    assert 60 * fake.chunk_steps >= 250


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


class ScriptedProvider:
    def __init__(self, *calls: ToolCall) -> None:
        self.calls = iter(calls)
        self.requests: list[InferenceRequest] = []

    def complete(self, request: InferenceRequest) -> InferenceResponse:
        self.requests.append(request)
        return InferenceResponse(tool_calls=(next(self.calls),), output_items=(), usage={})

    def close(self) -> None:
        pass


def _motion(call_id: str, target: float, steps: int) -> ToolCall:
    return ToolCall(
        call_id,
        "run_joint_chunk",
        {
            "total_steps": steps,
            "keyframes": [{"step": steps, "target": [target] * 12}],
            "note": "Regression trajectory.",
        },
    )


def test_reversal_and_hold_use_applied_target_and_report_guardrail_changes() -> None:
    provider = ScriptedProvider(
        _motion("reverse", 0.0, 2),
        ToolCall("hold", "hold", {"steps": 2, "note": "Keep the applied target."}),
    )
    policy = JointChunkAgentPolicy(config_path=None, inference_provider=provider)
    policy.reset(SimpleNamespace(instruction="Stand still."))  # type: ignore[arg-type]
    observation = _observation(step=4, last_action=0.1, applied_action=0.2)
    chunk = policy.act(observation)
    np.testing.assert_allclose(chunk.actions[0].data, 0.1)
    np.testing.assert_allclose(chunk.actions[1].data, 0.0)
    np.testing.assert_allclose(observation.state["policy_obs"][33:45], 0.1)

    # A stricter external guardrail actually stops at .05, not the requested 0.
    feedback = {
        "tool_call_id": "reverse",
        "steps": [
            {"sim_step": 5, "chunk_index": 0, "applied_action": [0.1] * 12},
            {"sim_step": 6, "chunk_index": 1, "applied_action": [0.05] * 12},
        ],
    }
    next_obs = replace(
        _observation(step=6, last_action=0.1, applied_action=0.05),
        extra={"action_execution": feedback},
    )
    hold = policy.act(next_obs)
    np.testing.assert_allclose(hold.actions[0].data, 0.05)
    result = next(item["content"] for item in policy.transcript() if item["role"] == "tool")
    assert result["execution_feedback_complete"] is True
    assert result["modified_steps"] == 1
    assert result["last_applied_action"] == [0.05] * 12
    assert result["modifications"][0]["chunk_step"] == 2
    assert result["modifications"][0]["joints"][0] == "FR_hip"
    assert any(
        item.get("type") == "function_call_output" for item in provider.requests[1].input_items
    )


def test_slew_validation_uses_applied_target_and_retries() -> None:
    provider = ScriptedProvider(
        _motion("unsafe", -0.1, 1),  # Valid from stale zero; invalid from actual .2.
        ToolCall("hold", "hold", {"steps": 1, "note": "Hold after rejection."}),
    )
    policy = JointChunkAgentPolicy(config_path=None, inference_provider=provider)
    policy.reset(SimpleNamespace(instruction="Stand still."))  # type: ignore[arg-type]
    chunk = policy.act(_observation(last_action=0.0, applied_action=0.2))
    assert len(provider.requests) == 2
    np.testing.assert_allclose(chunk.actions[0].data, 0.2)


@pytest.mark.parametrize("budget_stop", [False, True])
def test_stop_holds_actual_target(budget_stop: bool) -> None:
    call = (
        _motion("invalid", -1.0, 1)
        if budget_stop
        else ToolCall("stop", "give_up", {"reason": "End this test."})
    )
    policy = JointChunkAgentPolicy(
        config_path=None, inference_provider=ScriptedProvider(call), max_llm_calls=1
    )
    policy.reset(SimpleNamespace(instruction="Stand still."))  # type: ignore[arg-type]
    chunk = policy.act(_observation(last_action=0.1, applied_action=0.2))
    assert chunk.actions[0].meta["request_stop"] is True
    np.testing.assert_allclose(chunk.actions[0].data, 0.2)


@pytest.mark.parametrize("reason", ["fell", "max_steps", "simulator fault"])
def test_fifteen_requested_ten_executed_survives_trial_end(reason: str) -> None:
    policy = JointChunkAgentPolicy(
        config_path=None, inference_provider=ScriptedProvider(_motion("motion", 0.3, 15))
    )
    policy.reset(SimpleNamespace(instruction="Stand still."))  # type: ignore[arg-type]
    chunk = policy.act(_observation())
    actions = list(chunk.actions[:10])
    actions[-1] = replace(actions[-1], data=np.full(12, 0.15, dtype=np.float32))
    record = SimpleNamespace(
        steps=[SimpleNamespace(action=a) for a in actions],
        termination_reason=reason if reason != "simulator fault" else None,
        error=reason if reason == "simulator fault" else None,
        status="error" if reason == "simulator fault" else "success",
        metadata={},
    )
    policy.on_trial_end(record, "logs", "test")
    result = record.metadata["chunk_executions"][0]
    assert (result["requested_steps"], result["executed_steps"]) == (15, 10)
    assert result["status"] == "interrupted"
    assert result["modified_steps"] == 1
    assert result["stop_reason"] == reason
    np.testing.assert_allclose(result["last_applied_action"], 0.15)
    assert record.policy_transcript[-1]["content"] == {
        key: value for key, value in result.items() if key not in {"tool_call_id", "tool_name"}
    }
    policy.reset(SimpleNamespace(instruction="New scene."))  # type: ignore[arg-type]
    assert not any(item["role"] == "tool" for item in policy.transcript())


def test_agent_rejects_missing_execution_feedback_instead_of_stale_fallback() -> None:
    policy = JointChunkAgentPolicy(config_path=None, inference_provider=FakeInferenceProvider())
    with pytest.raises(ValueError, match="last_applied_action"):
        policy.act(Observation(state={"policy_obs": np.zeros(48)}))
