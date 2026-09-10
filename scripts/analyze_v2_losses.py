"""Attribute v2 score gaps and replay the failed turn; no inference or policy changes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch", type=Path)
    parser.add_argument(
        "--reference", type=Path, default=Path("outputs/analysis-20260910-parallel")
    )
    parser.add_argument("--replay-turn", action="store_true")
    args = parser.parse_args()
    batch = args.batch.resolve()
    output = batch / "comparison"
    runs = read(output / "summary.json")["runs"]
    baselines = {
        (r["policy"], r["scene"]): r for r in read(args.reference / "analysis.json")["runs"]
    }
    reports = []
    for run in runs:
        scene, n = run["scene"], run["steps"]
        baseline = baselines["ONNX-PPO", scene]
        trajectory = np.load(args.reference / baseline["trajectory"])
        baseline_terms = dict(
            zip(
                trajectory["terms"].tolist(),
                (trajectory["weighted_terms"][:n].sum(axis=0) * 0.02).tolist(),
                strict=True,
            )
        )
        gaps = {
            term: baseline_terms[term] - value for term, value in run["integrated_terms"].items()
        }
        prefix_return = float(trajectory["rewards"][:n].sum())
        clip_gap = prefix_return - sum(baseline_terms.values()) - run["clip_adjustment"]
        tail = float(trajectory["rewards"][n:].sum())
        full_gap = baseline["return"] - run["return"]
        np.testing.assert_allclose(
            sum(gaps.values()) + clip_gap + tail, full_gap, atol=1e-7, rtol=0
        )
        ordered = [
            {
                "term": term,
                "gap": value,
                "astra": run["integrated_terms"][term],
                "onnx_same_steps": baseline_terms[term],
            }
            for term, value in sorted(gaps.items(), key=lambda pair: -pair[1])
        ]
        reports.append(
            {
                "scene": scene,
                "matched_steps": n,
                "astra_return": run["return"],
                "onnx_prefix_return": prefix_return,
                "same_time_gap": prefix_return - run["return"],
                "onnx_remaining_return": tail,
                "full_gap": full_gap,
                "term_gaps": ordered,
                "clipping_gap": clip_gap,
                "absolute_negative_contributions": {
                    term: value for term, value in run["integrated_terms"].items() if value < 0
                },
            }
        )
    report = {"reference": "ONNX-PPO native, same seed and same executed time", "runs": reports}
    turn = next(r for r in runs if r["scene"] == "turn")
    rows = [json.loads(line) for line in (batch / "turn/telemetry.jsonl").read_text().splitlines()]
    observations = np.asarray([row["policy_obs"] for row in rows])
    actions = np.asarray([row["action"] for row in rows])
    tilts = np.asarray([row["tilt_deg"] for row in rows])
    sample = read(batch / turn["log"])["samples"][0]
    chunks = sample["trial_metadata"][0]["chunk_executions"]
    notes = [note for note in turn["notes"] if note["step"] >= 61]
    checkpoints = []
    for note in notes:
        step = note["step"]
        data = observations[step - 1]
        checkpoints.append(
            {
                **note,
                "time_s": step * 0.02,
                "tilt_deg": float(tilts[step - 1]),
                "gyro_xyz": data[3:6].tolist(),
                "velocity_xyz": data[:3].tolist(),
                "gravity": data[6:9].tolist(),
                "action": actions[step - 1].reshape(4, 3).tolist(),
            }
        )
    report["turn"] = {
        "checkpoints": checkpoints,
        "yaw_range": [float(observations[:, 5].min()), float(observations[:, 5].max())],
        "first_zero_reward_step": next(
            (i + 1 for i, row in enumerate(rows) if row["reward"] == 0), None
        ),
        "zero_reward_steps": turn["zero_reward_steps"],
        "max_action_abs": float(np.max(np.abs(actions))),
        "saturated_steps": int(np.sum(np.any(np.abs(actions) >= 0.99999, axis=1))),
        "motion_chunks": len(chunks),
        "control_feedback_hz": len(chunks) / (106 * 0.02),
    }
    if args.replay_turn:
        from inspect_robots import Action
        from inspect_robots.rollout import derive_seed

        from gpt_dog_eval.embodiment import Go1PlaygroundEmbodiment
        from gpt_dog_eval.tasks import go1_velocity_smoke

        env = Go1PlaygroundEmbodiment()
        scene = go1_velocity_smoke(scenes="turn").scenes[0]
        contacts, feet, qpos, qvel, rewards = [], [], [], [], []
        try:
            env.reset(scene, seed=derive_seed(0, scene.init_seed, 0))
            for index, action in enumerate(actions):
                result = env.step(Action(data=action))
                np.testing.assert_allclose(
                    result.observation.state["policy_obs"], observations[index], atol=1e-7, rtol=0
                )
                np.testing.assert_allclose(result.reward, rows[index]["reward"], atol=1e-9, rtol=0)
                data = env._state.data
                contacts.append(
                    np.asarray(
                        [
                            data.sensordata[env._env.mj_model.sensor_adr[sid]] > 0
                            for sid in env._env._feet_floor_found_sensor
                        ]
                    )
                )
                feet.append(np.asarray(data.site_xpos[env._env._feet_site_id]).copy())
                qpos.append(np.asarray(data.qpos).copy())
                qvel.append(np.asarray(data.qvel).copy())
                rewards.append(float(result.reward))
            contacts, feet = np.asarray(contacts), np.asarray(feet)
            assert sum(rewards) == turn["return"]
            np.savez_compressed(
                output / "turn-diagnostic-replay.npz",
                contacts=contacts,
                feet=feet,
                qpos=qpos,
                qvel=qvel,
                observations=observations,
                actions=actions,
                rewards=rewards,
            )
            for index, checkpoint in enumerate(checkpoints):
                start = checkpoint["step"]
                end = checkpoints[index + 1]["step"] if index + 1 < len(checkpoints) else len(rows)
                checkpoint.update(
                    next_observation_step=end,
                    next_tilt_deg=float(tilts[end - 1]),
                    next_roll_rate=float(observations[end - 1, 3]),
                    contact_at_start=contacts[start - 1].astype(int).tolist(),
                    contact_fraction_during_chunk=contacts[start:end].mean(axis=0).tolist(),
                    feet_z_at_start=feet[start - 1, :, 2].tolist(),
                    next_action=actions[start].reshape(4, 3).tolist(),
                    end_action=actions[end - 1].reshape(4, 3).tolist(),
                    base_height=float(qpos[start - 1][2]),
                    world_velocity=np.asarray(qvel[start - 1][:3]).tolist(),
                )
            report["turn"]["replay_return_error"] = sum(rewards) - turn["return"]
            report["turn"]["first_no_foot_contact_step"] = next(
                (i + 1 for i, contact in enumerate(contacts) if i >= 60 and not contact.any()),
                None,
            )
            report["turn"]["first_single_side_support_step"] = next(
                (
                    i + 1
                    for i, c in enumerate(contacts)
                    if i >= 60 and (not c[0] and not c[2] and (c[1] or c[3]))
                ),
                None,
            )
        finally:
            env.close()

        t = (np.arange(len(rows)) + 1) * 0.02
        fig, axes = plt.subplots(
            3, 1, figsize=(10, 7), sharex=True, gridspec_kw={"height_ratios": [2, 2, 1]}
        )
        axes[0].plot(t, tilts, color="#d44f42", label="Body tilt")
        axes[0].axhline(90, ls=":", color="black", label="Physical fall threshold")
        axes[0].set(ylabel="Tilt (degrees)", ylim=(0, 100))
        axes[0].legend(loc="upper left", bbox_to_anchor=(0, 0.88), frameon=False)
        axes[1].plot(t, observations[:, 3], label="Body roll rate", color="#81549b")
        axes[1].plot(t, observations[:, 5], label="Body yaw rate", color="#007f66")
        axes[1].axhline(0.5, ls="--", color="#007f66", label="Yaw target")
        axes[1].set_ylabel("Angular velocity (rad/s)")
        axes[1].legend(loc="upper left", frameon=False, ncol=3)
        axes[2].imshow(
            contacts.T,
            cmap="Greys",
            aspect="auto",
            interpolation="nearest",
            extent=(0, len(rows) * 0.02, 3.5, -0.5),
            vmin=0,
            vmax=1,
        )
        axes[2].set(
            yticks=range(4),
            yticklabels=["FR", "FL", "RR", "RL"],
            xlabel="Simulation time (s)",
            ylabel="Contact\nblack=yes",
        )
        for step in (72, 82, 88, 95, 100, 106):
            for axis in axes:
                axis.axvline(step * 0.02, color="#777777", alpha=0.4, lw=0.8)
            axes[0].text(step * 0.02, 98, str(step), ha="center", va="top", fontsize=8)
        fig.suptitle("Astra v2 turn: decision steps, angular motion and actual foot contacts")
        fig.tight_layout()
        for extension in ("png", "svg"):
            fig.savefig(output / f"turn-failure.{extension}", dpi=180)
        plt.close(fig)
    (output / "loss-attribution.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
