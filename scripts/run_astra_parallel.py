"""Run one independent Astra context per scene and retain live/final evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from inspect_robots import eval as evaluate
from inspect_robots.approver import ChainApprover, ClampApprover, DeltaLimitApprover
from inspect_robots.logging.json_log import JsonLogSink
from inspect_robots.logging.sink import NullSink

from gpt_dog_eval.agent import JointChunkAgentPolicy
from gpt_dog_eval.constants import ACTION_SPACE
from gpt_dog_eval.embodiment import Go1PlaygroundEmbodiment
from gpt_dog_eval.inference import resolve_inference_config
from gpt_dog_eval.robot_spec import build_robot_spec, robot_spec_sha256, source_mjcf
from gpt_dog_eval.tasks import go1_velocity_smoke

SCENES = ("stand", "forward", "left", "turn")
ROOT = Path(__file__).resolve().parents[1]


def now():
    return datetime.now(UTC).isoformat()


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


class ProgressSink(NullSink):
    def __init__(self, directory, scene, policy):
        self.directory = directory
        self.policy = policy
        self.progress = {"scene": scene, "status": "starting", "steps": 0, "return": 0.0}
        self.telemetry = (directory / "telemetry.jsonl").open("w", encoding="utf-8")
        self.flush()

    def flush(self):
        self.progress.update(
            updated_at=now(),
            calls=self.policy._calls_used,
            usage=dict(self.policy._usage),
            finalized_chunks=len(self.policy._chunk_executions),
            modified_steps=sum(c.get("modified_steps", 0) for c in self.policy._chunk_executions),
        )
        write_json(self.directory / "progress.json", self.progress)
        write_json(self.directory / "transcript.json", self.policy.transcript())

    def log_step(self, t, observation, action, result):
        obs = result.observation.state["policy_obs"]
        gravity = obs[6:9]
        tilt = float(np.degrees(np.arccos(np.clip(-gravity[2], -1, 1))))
        reward = float(result.reward)
        self.progress.update(
            status="running",
            steps=t + 1,
            tilt_deg=tilt,
            velocity=obs[:3].tolist(),
            yaw_rate=float(obs[5]),
        )
        self.progress["return"] += reward
        self.telemetry.write(
            json.dumps(
                {
                    "t": t,
                    "reward": reward,
                    "policy_obs": obs.tolist(),
                    "action": np.asarray(action.data).tolist(),
                    "last_applied_action": result.observation.state["last_applied_action"].tolist(),
                    "weighted_terms": result.info["trainer"]["weighted_terms"],
                    "tilt_deg": tilt,
                    "request_stop": bool(action.meta.get("request_stop")),
                    "stop_detail": action.meta.get("stop_detail"),
                }
            )
            + "\n"
        )
        self.telemetry.flush()
        if action.meta.get("chunk_final") or result.terminated or action.meta.get("request_stop"):
            self.flush()

    def on_trial_end(self, record):
        self.progress.update(
            status=record.status,
            termination_reason=record.termination_reason,
            error=record.error,
        )
        self.flush()


def worker(output, scene, *, robot_spec=False, fake=False):
    directory = output / scene
    env = Go1PlaygroundEmbodiment(noise_level=0.0, perturbations=False)
    spec = build_robot_spec(env._ensure_env()) if robot_spec else None
    policy = JointChunkAgentPolicy(
        config_path=None if fake else str(ROOT / "secret.toml"),
        provider="fake" if fake else "responses",
        model="fake-go1" if fake else "gpt-6-astra",
        reasoning_effort="xhigh",
        max_llm_calls=60,
        max_chunk_steps=15,
        max_keyframes=4,
        max_action_delta=0.1,
        episode_steps=250,
        robot_spec=spec,
    )
    if spec is not None:
        write_json(directory / "robot-spec.json", spec)
        (directory / "system-prompt.txt").write_text(policy._instructions, encoding="utf-8")
        model_directory = directory / "model-sources"
        model_directory.mkdir()
        for name, content in source_mjcf(env._ensure_env()).items():
            (model_directory / name).write_bytes(content)
    monitor = ProgressSink(directory, scene, policy)
    json_sink = JsonLogSink(str(directory))
    try:
        log = evaluate(
            go1_velocity_smoke(steps=250, scenes=scene),
            policy,
            env,
            seed=0,
            log_dir=str(directory),
            sinks=[json_sink, monitor],
            approver=ChainApprover(ClampApprover(ACTION_SPACE), DeltaLimitApprover(ACTION_SPACE)),
        )[0]
        write_json(
            directory / "finished.json",
            {
                "scene": scene,
                "status": log.status,
                "finished_at": now(),
                "log": str(json_sink.path.relative_to(output)),
                "steps": log.stats.total_steps,
                "metrics": log.results.metrics,
                "termination": log.samples[0].termination_reasons,
                "usage": log.samples[0].trial_metadata[0].get("llm_usage", {}),
                "robot_spec_sha256": robot_spec_sha256(spec) if spec is not None else None,
            },
        )
    except BaseException as exc:
        write_json(directory / "worker-error.json", {"type": type(exc).__name__, "time": now()})
        raise
    finally:
        monitor.telemetry.close()
        env.close()
        policy.close()


def launch(output, *, robot_spec=False, fake=False, implementation_commit=None):
    os.chdir(ROOT)
    config = resolve_inference_config(
        config_path=None if fake else "secret.toml",
        provider="fake" if fake else "responses",
        model="fake-go1" if fake else "gpt-6-astra",
        reasoning_effort="xhigh",
    )
    if not fake and (config.api_key == "fake-local-key" or "127.0.0.1" in config.base_url):
        raise ValueError("A real inference endpoint must be configured before launching Astra")
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "started_at": now(),
        "batch": output.name,
        "provider": config.provider,
        "model": config.model,
        "reasoning_effort": config.reasoning_effort,
        "timeout_s": config.timeout_s,
        "eval_seed": 0,
        "epoch": 0,
        "steps": 250,
        "max_llm_calls": 60,
        "max_chunk_steps": 15,
        "max_keyframes": 4,
        "max_action_delta": 0.1,
        "action_feedback_version": 2,
        "robot_spec_enabled": robot_spec,
        "experiment_condition": "static-robot-spec-v3" if robot_spec else "action-feedback-v2",
        "independent_contexts": True,
        "implementation_commit": implementation_commit,
        "checkout_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "checkout_dirty": bool(
            subprocess.check_output(["git", "status", "--porcelain"], text=True)
        ),
        "source_sha256": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [*sorted((ROOT / "src/gpt_dog_eval").glob("*.py")), Path(__file__).resolve()]
        },
        "workers": {},
    }
    write_json(output / "manifest.json", manifest)
    for scene in SCENES:
        directory = output / scene
        directory.mkdir()
        with (directory / "worker.stdout.log").open("wb") as stream:
            process = subprocess.Popen(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--output",
                    str(output),
                    "--worker",
                    scene,
                    *(["--robot-spec"] if robot_spec else []),
                    *(["--fake"] if fake else []),
                ],
                cwd=ROOT,
                stdin=subprocess.DEVNULL,
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                env={**os.environ, "JAX_PLATFORM_NAME": "cpu", "PYTHONUNBUFFERED": "1"},
            )
        manifest["workers"][scene] = {"pid": process.pid}
        write_json(output / "manifest.json", manifest)
    print(json.dumps({"output": str(output), **manifest}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker", choices=SCENES)
    parser.add_argument("--robot-spec", action="store_true")
    parser.add_argument("--fake", action="store_true", help="Validate all four workers for free")
    parser.add_argument("--implementation-commit")
    args = parser.parse_args()
    output = (
        args.output
        or ROOT
        / "outputs"
        / (
            f"{'fake' if args.fake else 'astra'}-"
            f"{'robot-spec-v3' if args.robot_spec else 'feedback-v2'}-"
            f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}"
        )
    ).resolve()
    if args.worker:
        worker(output, args.worker, robot_spec=args.robot_spec, fake=args.fake)
    else:
        launch(
            output,
            robot_spec=args.robot_spec,
            fake=args.fake,
            implementation_commit=args.implementation_commit,
        )
