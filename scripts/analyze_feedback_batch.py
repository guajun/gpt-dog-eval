"""Check and compare a feedback-v2 batch using saved logs; no inference calls."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCENES = ("stand", "forward", "left", "turn")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def status(batch):
    summaries = []
    for scene in SCENES:
        path = batch / scene / "progress.json"
        if not path.exists():
            summaries.append({"scene": scene, "status": "starting"})
            continue
        progress = read(path)
        summaries.append(
            {
                key: progress.get(key)
                for key in (
                    "scene",
                    "status",
                    "steps",
                    "return",
                    "calls",
                    "modified_steps",
                    "tilt_deg",
                    "termination_reason",
                    "updated_at",
                )
            }
        )
        summaries[-1]["tokens"] = progress.get("usage", {}).get("total_tokens", 0)
        summaries[-1]["finished"] = (batch / scene / "finished.json").exists()
        summaries[-1]["worker_error"] = (batch / scene / "worker-error.json").exists()
    print(json.dumps(summaries, ensure_ascii=False))


def analyze(batch, reference):
    previous = {(r["policy"], r["scene"]): r for r in read(reference / "analysis.json")["runs"]}
    manifest = read(batch / "manifest.json")
    reports = []
    trajectories = {}
    for scene in SCENES:
        done = read(batch / scene / "finished.json")
        payload = read(batch / done["log"])
        sample = payload["samples"][0]
        metadata = sample["trial_metadata"][0]
        rows = [
            json.loads(line)
            for line in (batch / scene / "telemetry.jsonl").read_text().splitlines()
        ]
        assert [row["t"] for row in rows] == list(range(len(rows)))
        assert len(rows) == payload["stats"]["total_steps"]
        observations = np.asarray([row["policy_obs"] for row in rows])
        actions = np.asarray([row["action"] for row in rows])
        rewards = np.asarray([row["reward"] for row in rows])
        tilts = np.asarray([row["tilt_deg"] for row in rows])
        np.testing.assert_allclose(
            [row["last_applied_action"] for row in rows], actions, atol=1e-7, rtol=0
        )
        np.testing.assert_allclose(
            observations[:, 33:45], np.vstack((np.zeros(12), actions[:-1])), atol=1e-7, rtol=0
        )
        scored = sample["reduced"].get("episode_return")
        if scored is not None:
            np.testing.assert_allclose(rewards.sum(), scored, atol=1e-7, rtol=0)
        command = np.asarray(sample["scene_metadata"]["command"])
        transcript = sample["policy_transcripts"][0]
        context_checks = 0
        tool_errors = []
        notes = []
        calls_left_at_reset = None
        step = 0
        for item in transcript:
            content = item.get("content")
            if item.get("role") == "user" and isinstance(content, str):
                match = re.match(r"Observation step=(\d+)/250; calls_left=(\d+)", content)
                if match:
                    step = int(match[1])
                    if step == 0:
                        calls_left_at_reset = int(match[2])
                    current_line = next(
                        line
                        for line in content.splitlines()
                        if line.startswith("last_applied_action=")
                    )
                    current = json.loads(current_line.split("=", 1)[1])
                    np.testing.assert_allclose(
                        current, actions[step - 1] if step else np.zeros(12), atol=1e-7, rtol=0
                    )
                    context_checks += 1
            if item.get("role") == "tool" and isinstance(content, dict) and "error" in content:
                tool_errors.append({"step": step, "error": content["error"]})
            for call in item.get("tool_calls", []):
                notes.append(
                    {
                        "step": step,
                        "tool": call["name"],
                        "note": call["arguments"].get("note", call["arguments"].get("reason")),
                    }
                )
        assert calls_left_at_reset == 60
        assert metadata["action_feedback_version"] == 2
        chunks = metadata["chunk_executions"]
        assert all(c["execution_feedback_complete"] for c in chunks)
        old = previous["Astra-60", scene]
        old_data = np.load(reference / old["trajectory"])
        common_steps = min(len(rows), old["steps"])
        terms = sorted(rows[0]["weighted_terms"])
        integrated = {
            term: sum(row["weighted_terms"][term] for row in rows) * 0.02 for term in terms
        }
        report = {
            "scene": scene,
            "log": done["log"],
            "status": payload["status"],
            "steps": len(rows),
            "termination": sample["termination_reasons"][0],
            "return": scored,
            "recorded_return": float(rewards.sum()),
            "previous_return": old["return"],
            "return_change": None if scored is None else scored - old["return"],
            "previous_steps": old["steps"],
            "previous_termination": old["termination"],
            "common_steps": common_steps,
            "common_v1_return": float(old_data["rewards"][:common_steps].sum()),
            "common_v2_return": float(rewards[:common_steps].sum()),
            "command": command.tolist(),
            "mean_velocity": observations[:, [0, 1, 5]].mean(axis=0).tolist(),
            "lin_rmse": float(
                np.sqrt(np.mean(np.sum((observations[:, :2] - command[:2]) ** 2, axis=1)))
            ),
            "yaw_rmse": float(np.sqrt(np.mean((observations[:, 5] - command[2]) ** 2))),
            "max_tilt_deg": float(tilts.max()),
            "final_tilt_deg": float(tilts[-1]),
            "zero_reward_steps": int(np.sum(rewards == 0)),
            "llm_usage": metadata["llm_usage"],
            "chunk_count": len(chunks),
            "modified_steps": sum(c["modified_steps"] for c in chunks),
            "modified_chunks": sum(c["modified_steps"] > 0 for c in chunks),
            "max_action_error": max((c["max_action_error"] for c in chunks), default=0),
            "context_action_checks": context_checks,
            "tool_errors": tool_errors,
            "notes": notes,
            "integrated_terms": integrated,
            "clip_adjustment": float(rewards.sum() - sum(integrated.values())),
            "policy_stop_requested": any(row["request_stop"] for row in rows),
            "stop_detail": rows[-1]["stop_detail"],
            "duration_s": payload["stats"]["duration_s"],
            "final_chunk": chunks[-1] if chunks else None,
        }
        reports.append(report)
        trajectories[scene] = {"observations": observations, "rewards": rewards, "tilts": tilts}
    output = batch / "comparison"
    output.mkdir(exist_ok=True)
    (output / "summary.json").write_text(
        json.dumps({"batch": manifest["batch"], "runs": reports}, indent=2) + "\n"
    )
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(11, 5.2))
    x = np.arange(4)
    width = 0.19
    for offset, label, color, values, steps in (
        (
            -1.5,
            "Zero / Fake",
            "#85909c",
            [previous["Zero", s]["return"] for s in SCENES],
            [250] * 4,
        ),
        (
            -0.5,
            "ONNX-PPO native",
            "#d97706",
            [previous["ONNX-PPO", s]["return"] for s in SCENES],
            [250] * 4,
        ),
        (
            0.5,
            "Astra v1 / 60 calls",
            "#7294c0",
            [r["previous_return"] for r in reports],
            [r["previous_steps"] for r in reports],
        ),
        (
            1.5,
            "Astra v2 / 60 calls",
            "#007f66",
            [np.nan if r["return"] is None else r["return"] for r in reports],
            [r["steps"] for r in reports],
        ),
    ):
        bars = ax.bar(x + offset * width, values, width, label=label, color=color)
        for bar, value, count in zip(bars, values, steps, strict=True):
            if np.isfinite(value):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    value + 0.12,
                    f"{value:.2f}",
                    ha="center",
                    fontsize=9,
                )
                if count < 250:
                    bar.set_hatch("//")
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        max(value * 0.5, 0.18),
                        f"{count}",
                        ha="center",
                        color="white",
                        fontsize=9,
                    )
    ax.set(xticks=x, xticklabels=SCENES, ylabel="Episode return", ylim=(0, 10.3))
    ax.set_title("Go1: actual-action feedback fix, four independent Astra contexts")
    ax.legend(ncol=2, loc="upper center", frameon=False)
    fig.text(
        0.08,
        0.02,
        "One trial per scene; 250-step horizon. Hatched bars: early termination; "
        "number inside: executed steps.\nSame scene seeds; inference sampling is uncontrolled. "
        "Results do not isolate a causal treatment effect.",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    for extension in ("png", "svg"):
        fig.savefig(output / f"return-comparison.{extension}", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(3, 4, figsize=(14, 8), sharex=True)
    for column, scene in enumerate(SCENES):
        for policy, label, color in (
            ("ONNX-PPO", "ONNX", "#d97706"),
            ("Astra-60", "Astra v1", "#7294c0"),
            (None, "Astra v2", "#007f66"),
        ):
            if policy:
                old_data = np.load(reference / previous[policy, scene]["trajectory"])
                obs, rewards, tilts = (
                    old_data["observations"][1:],
                    old_data["rewards"],
                    old_data["tilt"],
                )
            else:
                obs, rewards, tilts = (
                    trajectories[scene][key] for key in ("observations", "rewards", "tilts")
                )
            t = (np.arange(len(rewards)) + 1) * 0.02
            component = {"stand": None, "forward": 0, "left": 1, "turn": 5}[scene]
            value = np.linalg.norm(obs[:, :2], axis=1) if component is None else obs[:, component]
            axes[0, column].plot(t, value, color=color, label=label)
            axes[1, column].plot(t, tilts, color=color)
            axes[2, column].plot(t, np.cumsum(rewards), color=color)
        axes[0, column].axhline(
            {"stand": 0, "forward": 0.5, "left": 0.3, "turn": 0.5}[scene],
            ls="--",
            color="black",
            lw=0.8,
        )
        axes[0, column].set_title(scene)
        axes[1, column].set_ylim(0, 100)
        axes[2, column].set(xlabel="Simulation time (s)", xlim=(0, 5))
    axes[0, 0].set_ylabel("Commanded component\n(m/s; turn: rad/s)")
    axes[1, 0].set_ylabel("Body tilt (degrees)")
    axes[2, 0].set_ylabel("Cumulative return")
    axes[0, 0].legend(frameon=False)
    fig.suptitle("Feedback v2 comparison: trajectories end at actual termination")
    fig.tight_layout()
    for extension in ("png", "svg"):
        fig.savefig(output / f"trajectories.{extension}", dpi=180)
    plt.close(fig)
    print(
        json.dumps(
            [
                {
                    k: r[k]
                    for k in (
                        "scene",
                        "steps",
                        "termination",
                        "return",
                        "return_change",
                        "modified_steps",
                        "llm_usage",
                    )
                }
                for r in reports
            ],
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch", type=Path)
    parser.add_argument(
        "--reference", type=Path, default=Path("outputs/analysis-20260910-parallel")
    )
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args()
    if args.status:
        status(args.batch)
    else:
        analyze(args.batch, args.reference)
