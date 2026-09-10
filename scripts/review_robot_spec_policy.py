"""Review saved v2/v3 policy behavior and replay v3 forward; no inference calls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from inspect_robots import Action
from inspect_robots.rollout import derive_seed

from gpt_dog_eval.embodiment import Go1PlaygroundEmbodiment
from gpt_dog_eval.robot_spec import build_robot_spec, robot_spec_sha256
from gpt_dog_eval.tasks import go1_velocity_smoke


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def telemetry(root, scene):
    return [
        json.loads(line) for line in (root / scene / "telemetry.jsonl").read_text().splitlines()
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch", type=Path)
    parser.add_argument(
        "--previous", type=Path, default=Path("outputs/astra-feedback-v2-20260910T083830Z")
    )
    args = parser.parse_args()
    output = args.batch / "comparison"
    env = Go1PlaygroundEmbodiment()
    robot = env._ensure_env()
    model = robot.mj_model
    spec = read(args.batch / "stand/robot-spec.json")
    assert robot_spec_sha256(build_robot_spec(robot)) == robot_spec_sha256(spec)
    foot_ids = [model.site(name).id for name in ("FR", "FL", "RR", "RL")]
    nominal = np.asarray(spec["nominal_joint_angles"])
    report = {"method": "Saved telemetry review; exact forward replay; no inference", "runs": []}
    for label, root in (("v2", args.previous), ("v3", args.batch)):
        for scene in ("stand", "forward", "left", "turn"):
            rows = telemetry(root, scene)
            observations = np.asarray([r["policy_obs"] for r in rows])
            actions = np.asarray([r["action"] for r in rows])
            done = read(root / scene / "finished.json")
            sample = read(root / done["log"])["samples"][0]
            chunks = sample["trial_metadata"][0]["chunk_executions"]
            durations = [c["executed_steps"] for c in chunks]
            component = {"stand": 0, "forward": 0, "left": 1, "turn": 5}[scene]
            v = observations[:, component]
            result = {
                "policy": label,
                "scene": scene,
                "log": done["log"],
                "chunk_median_steps": float(np.median(durations)),
                "chunk_mean_steps": float(np.mean(durations)),
                "motion_feedback_hz": len(chunks) / (sum(durations) * 0.02),
                "negative_main_velocity_fraction": float((v < 0).mean()),
                "negative_main_velocity_below_minus_0_02_fraction": float((v < -0.02).mean()),
                "saturated_action_steps": int((np.abs(actions) >= 0.999).any(axis=1).sum()),
                "last100_mean_velocity": float(v[-100:].mean()),
                "last100_velocity_std": float(v[-100:].std()),
                "last100_joint_abs_delta_sum_mean": float(
                    np.abs(observations[-100:, 9:21]).sum(axis=1).mean()
                ),
                "last100_return": sum(r["reward"] for r in rows[-100:]),
                "first100_return": sum(r["reward"] for r in rows[:100]),
            }
            if scene == "stand":
                widths = []
                for obs in observations[-100:]:
                    data = mujoco.MjData(model)
                    data.qpos[:7] = [0, 0, 0, 1, 0, 0, 0]
                    data.qpos[7:] = nominal + obs[9:21]
                    mujoco.mj_kinematics(model, data)
                    feet = data.site_xpos[foot_ids]
                    widths.append(float(feet[[1, 3], 1].mean() - feet[[0, 2], 1].mean()))
                result["last100_mean_foot_center_width_body_frame_m"] = float(np.mean(widths))
                result["last100_mean_action"] = actions[-100:].mean(axis=0).tolist()
            report["runs"].append(result)

    scene = go1_velocity_smoke(scenes="forward").scenes[0]
    env.reset(scene, seed=derive_seed(0, scene.init_seed, 0))
    rows = telemetry(args.batch, "forward")
    contacts, feet, qpos, qvel, rewards = [], [], [], [], []
    for row in rows:
        result = env.step(Action(data=np.asarray(row["action"], dtype=np.float32)))
        np.testing.assert_allclose(
            result.observation.state["policy_obs"], row["policy_obs"], atol=1e-7, rtol=0
        )
        np.testing.assert_allclose(result.reward, row["reward"], atol=1e-9, rtol=0)
        data = env._state.data
        contacts.append(
            [
                bool(data.sensordata[model.sensor_adr[sid]] > 0)
                for sid in robot._feet_floor_found_sensor
            ]
        )
        feet.append(np.asarray(data.site_xpos)[foot_ids].copy())
        qpos.append(np.asarray(data.qpos).copy())
        qvel.append(np.asarray(data.qvel).copy())
        rewards.append(float(result.reward))
    contacts = np.asarray(contacts)
    np.savez_compressed(
        output / "forward-diagnostic-replay.npz",
        contacts=contacts,
        feet=feet,
        qpos=qpos,
        qvel=qvel,
        rewards=rewards,
    )
    checkpoints = []
    for step in (70, 74, 80, 85, 90, 95, 100, 105, 110, 111, 115, 116):
        row = rows[step - 1]
        checkpoints.append(
            {
                "step": step,
                "tilt_deg": row["tilt_deg"],
                "gyro": row["policy_obs"][3:6],
                "contacts_FR_FL_RR_RL": contacts[step - 1].astype(int).tolist(),
                "feet_world_z": np.asarray(feet)[step - 1, :, 2].tolist(),
                "base_world_z": float(qpos[step - 1][2]),
            }
        )
    counts = contacts.sum(axis=1)
    report["forward_replay"] = {
        "steps": len(rows),
        "return_error": sum(rewards) - sum(r["reward"] for r in rows),
        "first_no_contact_after_step70": next(
            (i + 1 for i in range(70, len(rows)) if counts[i] == 0), None
        ),
        "checkpoints": checkpoints,
        "contact_patterns_after70": [
            {
                "steps": [start + 1, end],
                "contact_fraction_FR_FL_RR_RL": contacts[start:end].mean(axis=0).tolist(),
            }
            for start, end in zip(
                (70, 74, 80, 85, 90, 95, 100, 105, 111, 115),
                (74, 80, 85, 90, 95, 100, 105, 111, 115, 116),
                strict=True,
            )
        ],
    }
    (output / "policy-review.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))
    env.close()


if __name__ == "__main__":
    main()
