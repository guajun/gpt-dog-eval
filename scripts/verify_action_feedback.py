"""Verify the action-feedback fix in real CPU MuJoCo, without paid inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from inspect_robots import eval as evaluate
from inspect_robots.approver import AutoApprover, DeltaLimitApprover
from inspect_robots.logging.json_log import JsonLogSink
from inspect_robots.logging.sink import NullSink

from gpt_dog_eval.agent import JointChunkAgentPolicy
from gpt_dog_eval.constants import ACTION_SPACE
from gpt_dog_eval.embodiment import Go1PlaygroundEmbodiment
from gpt_dog_eval.inference import FakeInferenceProvider, InferenceResponse, ToolCall
from gpt_dog_eval.policies import OnnxGo1Policy, ZeroGo1Policy
from gpt_dog_eval.tasks import go1_velocity_smoke


class ScriptedProvider:
    """Nonzero ramps, reversal, hold, and a chunk cut short at the horizon."""

    def __init__(self):
        self.calls = 0

    def complete(self, request):
        self.calls += 1
        call_id = f"scripted_{self.calls}"
        if self.calls == 3:
            name, arguments = "hold", {"steps": 2, "note": "Hold the actual applied target."}
        else:
            target, steps = {1: (0.1, 2), 2: (0.0, 2), 4: (0.12, 15)}[self.calls]
            name = "run_joint_chunk"
            arguments = {
                "keyframes": [{"step": steps, "target": [target] * 12}],
                "total_steps": steps,
                "note": "Free action-feedback regression probe, not a locomotion policy.",
            }
        item = {
            "type": "function_call",
            "call_id": call_id,
            "name": name,
            "arguments": json.dumps(arguments),
        }
        return InferenceResponse(
            tool_calls=(ToolCall(call_id, name, arguments),),
            output_items=(item,),
            usage={"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
        )

    def close(self):
        pass


class FeedbackChecks(NullSink):
    def __init__(self):
        self.steps = 0
        self.nonzero_lag_steps = 0

    def log_step(self, t, observation, action, result):
        actual = np.asarray(action.data, dtype=np.float32)
        state = result.observation.state
        np.testing.assert_array_equal(state["last_applied_action"], actual)
        # Prove the legacy ONNX input remains unchanged, including its lag.
        np.testing.assert_array_equal(
            state["policy_obs"][33:45], observation.state["last_applied_action"]
        )
        self.nonzero_lag_steps += int(np.max(np.abs(state["policy_obs"][33:45] - actual)) > 1e-7)
        self.steps += 1


class FaultProbeEmbodiment(Go1PlaygroundEmbodiment):
    """Inject one controlled fault to verify the final partial execution log."""

    fault_after = None

    def step(self, action):
        if self._step_index == self.fault_after:
            raise RuntimeError("Intentional action-feedback regression fault")
        return super().step(action)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/action-feedback-v2"))
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    env = FaultProbeEmbodiment()
    checks = FeedbackChecks()
    results = {}
    zero_returns = [4.57472528424114, 6.430928820744157, 8.056614292785525, 7.441093302331865]
    onnx_returns = [8.885102009400725, 8.201437903568149, 7.445327070658095, 8.317601774819195]

    def run(label, policy, task, approver, *, expected_status="success"):
        print(f"Running {label}", flush=True)
        json_sink = JsonLogSink(str(output / label))
        log = evaluate(
            task,
            policy,
            env,
            log_dir=str(output / label),
            approver=approver,
            sinks=[checks, json_sink],
            fail_on_error=False,
        )[0]
        assert log.status == expected_status
        results[label] = {
            "log": str(json_sink.path.relative_to(output)),
            "status": log.status,
            "returns": {
                sample.scene_id: sample.reduced.get("episode_return") for sample in log.samples
            },
            "steps": log.stats.total_steps,
            "termination": {sample.scene_id: sample.termination_reasons for sample in log.samples},
        }
        return log

    try:
        for label, policy, expected, approver in (
            ("zero", ZeroGo1Policy(), zero_returns, AutoApprover()),
            ("onnx-native", OnnxGo1Policy(), onnx_returns, AutoApprover()),
            (
                "fake",
                JointChunkAgentPolicy(config_path=None, inference_provider=FakeInferenceProvider()),
                zero_returns,
                DeltaLimitApprover(ACTION_SPACE),
            ),
        ):
            log = run(label, policy, go1_velocity_smoke(), approver)
            np.testing.assert_allclose(
                [sample.reduced["episode_return"] for sample in log.samples],
                expected,
                atol=2e-5,
                rtol=0,
            )
            assert log.stats.total_steps == 1000
            if label == "fake":
                for sample in log.samples:
                    metadata = sample.trial_metadata[0]
                    chunks = metadata["chunk_executions"]
                    assert metadata["action_feedback_version"] == 2
                    assert all(c["execution_feedback_complete"] for c in chunks)
                    assert all(c["modified_steps"] == 0 for c in chunks)
                    assert chunks[-1]["requested_steps"] == 15
                    assert chunks[-1]["executed_steps"] == 10

        for label, delta in (("scripted", 0.1), ("scripted-strict-guardrail", 0.025)):
            provider = ScriptedProvider()
            policy = JointChunkAgentPolicy(config_path=None, inference_provider=provider)
            log = run(
                label,
                policy,
                go1_velocity_smoke(steps=16, scenes="stand"),
                DeltaLimitApprover(ACTION_SPACE, max_delta=delta),
            )
            assert provider.calls == 4
            chunks = log.samples[0].trial_metadata[0]["chunk_executions"]
            assert all(c["execution_feedback_complete"] for c in chunks)
            assert (chunks[-1]["requested_steps"], chunks[-1]["executed_steps"]) == (15, 10)
            assert chunks[-1]["status"] == "interrupted"
            np.testing.assert_allclose(
                chunks[2]["last_applied_action"], chunks[1]["last_applied_action"]
            )
            modifications = sum(c["modified_steps"] for c in chunks)
            assert modifications == 0 if delta == 0.1 else modifications > 0
            results[label]["modified_steps"] = modifications
            results[label]["chunk_executions"] = chunks
        env.fault_after = 10
        log = run(
            "intentional-fault",
            JointChunkAgentPolicy(config_path=None, inference_provider=FakeInferenceProvider()),
            go1_velocity_smoke(steps=20, scenes="stand"),
            DeltaLimitApprover(ACTION_SPACE),
            expected_status="error",
        )
        chunks = log.samples[0].trial_metadata[0]["chunk_executions"]
        assert (chunks[0]["requested_steps"], chunks[0]["executed_steps"]) == (15, 10)
        assert chunks[0]["execution_feedback_complete"]
        assert chunks[0]["status"] == "interrupted"
        assert "Intentional action-feedback regression fault" in chunks[0]["stop_reason"]
        results["intentional-fault"]["chunk_executions"] = chunks
        results["checks"] = {"steps": checks.steps, "nonzero_lag_steps": checks.nonzero_lag_steps}
        assert checks.nonzero_lag_steps > 0
        (output / "verification.json").write_text(json.dumps(results, indent=2) + "\n")
        print(json.dumps(results["checks"]), flush=True)
        print(f"PASS: {output / 'verification.json'}", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
