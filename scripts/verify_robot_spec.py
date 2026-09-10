"""Validate exported geometry against MuJoCo FK, with no inference requests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from gpt_dog_eval.embodiment import Go1PlaygroundEmbodiment
from gpt_dog_eval.robot_spec import build_robot_spec, robot_spec_sha256


def rotation(q):
    w, x, y, z = np.asarray(q) / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    env = Go1PlaygroundEmbodiment()._ensure_env()
    m = env.mj_model
    spec = build_robot_spec(env)
    original = {
        name: getattr(m, name).copy()
        for name in (
            "body_pos",
            "body_mass",
            "body_inertia",
            "jnt_pos",
            "jnt_axis",
            "actuator_gainprm",
        )
    }
    assert build_robot_spec(env) == spec
    for name, values in original.items():
        np.testing.assert_array_equal(values, getattr(m, name))
    np.testing.assert_allclose(sum(b["mass"] for b in spec["bodies"]), m.body_mass.sum())
    joints = {j["body"]: (i, j) for i, j in enumerate(spec["joints_in_action_order"])}
    rng = np.random.default_rng(20260910)
    largest_error = 0.0
    for _ in range(20):
        q = np.asarray(spec["nominal_joint_angles"]) + rng.uniform(-0.5, 0.5, 12)
        data = mujoco.MjData(m)
        data.qpos[:3] = rng.uniform(-0.3, 0.3, 3)
        quat = rng.normal(size=4)
        data.qpos[3:7] = quat / np.linalg.norm(quat)
        data.qpos[7:] = q
        mujoco.mj_forward(m, data)
        frames = {"world": (np.zeros(3), np.eye(3))}
        for body in spec["bodies"]:
            if body["name"] == "trunk":
                frames["trunk"] = (data.qpos[:3].copy(), rotation(data.qpos[3:7]))
                continue
            parent_pos, parent_rot = frames[body["parent"]]
            origin_rot = parent_rot @ rotation(body["origin_quat_wxyz"])
            origin_pos = parent_pos + parent_rot @ np.asarray(body["origin_pos"])
            i, joint = joints[body["name"]]
            angle = q[i] - joint["zero_reference"]
            joint_rot = rotation(
                [np.cos(angle / 2), *(np.asarray(joint["axis"]) * np.sin(angle / 2))]
            )
            child_rot = origin_rot @ joint_rot
            anchor = np.asarray(joint["anchor_pos"])
            child_pos = origin_pos + origin_rot @ anchor - child_rot @ anchor
            frames[body["name"]] = (child_pos, child_rot)
            bid = m.body(body["name"]).id
            np.testing.assert_allclose(child_pos, data.xpos[bid], atol=1e-12)
            com = child_pos + child_rot @ np.asarray(body["com_pos"])
            np.testing.assert_allclose(com, data.xipos[bid], atol=1e-12)
            ir = child_rot @ rotation(body["inertia_quat_wxyz"])
            inertia = ir @ np.diag(body["principal_inertia"]) @ ir.T
            expected_rot = data.ximat[bid].reshape(3, 3)
            expected = expected_rot @ np.diag(m.body_inertia[bid]) @ expected_rot.T
            np.testing.assert_allclose(inertia, expected, atol=1e-12)
        for site in spec["sites"]:
            pos, rot = frames[site["body"]]
            expected = data.site_xpos[m.site(site["name"]).id]
            error = float(np.max(np.abs(pos + rot @ np.asarray(site["pos"]) - expected)))
            largest_error = max(largest_error, error)
            assert error < 1e-12
    result = {
        "random_poses": 20,
        "sites_per_pose": 5,
        "max_fk_error_m": largest_error,
        "com_and_inertia_checks": "passed",
        "model_unchanged": True,
        "robot_spec_sha256": robot_spec_sha256(spec),
        "inference_calls": 0,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
