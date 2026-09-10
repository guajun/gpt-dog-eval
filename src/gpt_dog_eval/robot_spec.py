"""Static robot description extracted from the exact, configured MuJoCo model."""

from __future__ import annotations

import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np

from gpt_dog_eval.constants import JOINT_LABELS


def source_mjcf(env: Any) -> dict[str, bytes]:
    """Collect only the entry scene and its recursive XML includes for provenance."""
    path = Path(env.xml_path)
    files = {path.name: path.read_bytes()}
    pending = [path.name]
    while pending:
        for include in ET.fromstring(files[pending.pop()]).iter("include"):
            name = include.attrib["file"]
            if name not in files:
                files[name] = bytes(env._model_assets[name])
                pending.append(name)
    return files


def build_robot_spec(env: Any) -> dict[str, Any]:
    """Export model constants, never rollout state, contacts, reward, or a policy."""
    model = env.mj_model
    names = tuple(model.actuator(i).name for i in range(model.nu))
    if names != JOINT_LABELS:
        raise ValueError("robot description requires the Go1 action joint order")
    bodies = []
    for i in range(1, model.nbody):
        bodies.append(
            {
                "name": model.body(i).name,
                "parent": model.body(int(model.body_parentid[i])).name or "world",
                "origin_pos": model.body_pos[i].tolist(),
                "origin_quat_wxyz": model.body_quat[i].tolist(),
                "mass": float(model.body_mass[i]),
                "com_pos": model.body_ipos[i].tolist(),
                "inertia_quat_wxyz": model.body_iquat[i].tolist(),
                "principal_inertia": model.body_inertia[i].tolist(),
            }
        )
    joints = []
    for i, name in enumerate(names):
        j = int(model.actuator_trnid[i, 0])
        dof = int(model.jnt_dofadr[j])
        joints.append(
            {
                "action_name": name,
                "joint_name": model.joint(j).name,
                "body": model.body(int(model.jnt_bodyid[j])).name,
                "anchor_pos": model.jnt_pos[j].tolist(),
                "axis": model.jnt_axis[j].tolist(),
                "zero_reference": float(model.qpos0[int(model.jnt_qposadr[j])]),
                "range": model.jnt_range[j].tolist(),
                "armature": float(model.dof_armature[dof]),
                "damping": float(model.dof_damping[dof]),
                "frictionloss": float(model.dof_frictionloss[dof]),
                "gear": model.actuator_gear[i].tolist(),
                "position_gain": float(model.actuator_gainprm[i, 0]),
                "bias_q": float(model.actuator_biasprm[i, 1]),
                "bias_qvel": float(model.actuator_biasprm[i, 2]),
                "ctrl_range": model.actuator_ctrlrange[i].tolist(),
                "force_range": model.actuator_forcerange[i].tolist(),
            }
        )
    sites = []
    for name in ("imu", "FR", "FL", "RR", "RL"):
        site = model.site(name)
        sites.append(
            {
                "name": name,
                "body": model.body(int(model.site_bodyid[site.id])).name,
                "pos": model.site_pos[site.id].tolist(),
                "quat_wxyz": model.site_quat[site.id].tolist(),
            }
        )
    collisions = []
    for i in range(model.ngeom):
        if model.geom_contype[i] or model.geom_conaffinity[i]:
            collisions.append(
                {
                    "name": model.geom(i).name,
                    "body": model.body(int(model.geom_bodyid[i])).name or "world",
                    "type": int(model.geom_type[i]),
                    "pos": model.geom_pos[i].tolist(),
                    "quat_wxyz": model.geom_quat[i].tolist(),
                    "size": model.geom_size[i].tolist(),
                    "friction": model.geom_friction[i].tolist(),
                    "solref": model.geom_solref[i].tolist(),
                    "solimp": model.geom_solimp[i].tolist(),
                    "priority": int(model.geom_priority[i]),
                    "condim": int(model.geom_condim[i]),
                    "contype": int(model.geom_contype[i]),
                    "conaffinity": int(model.geom_conaffinity[i]),
                }
            )
    return {
        "schema_version": 1,
        "robot": "Unitree Go1",
        "scene_xml": Path(env.xml_path).name,
        "source_xml_sha256": {
            name: hashlib.sha256(value).hexdigest() for name, value in source_mjcf(env).items()
        },
        "units": "SI: m, kg, s, rad, N*m; inertia in kg*m^2",
        "body_axes": "+X forward, +Y left, +Z up; positive rotation is right-hand rule",
        "frame_semantics": (
            "body origin is relative to parent at joint zero_reference; joint anchor/axis "
            "are in the child body frame; COM and inertia orientation are relative to body. "
            "Quaternions are wxyz, mapping local coordinates into parent coordinates. "
            "I_body_at_COM=R(inertia_quat)*diag(principal_inertia)*R.T. "
            "The trunk is a free-floating body, not fixed at its model origin. "
            "Sites are fixed in their body; foot sites are sphere centers, not ground surfaces."
        ),
        "observation_semantics": (
            "body_linear_velocity and body_gyro are measured at the imu site in imu axes; "
            "projected_gravity is world unit down expressed in imu axes. "
            "joint_position_delta=q-nominal_joint_angles, joint_velocity=dq/dt. "
            "Nominal joint angles are distinct from kinematic zero_reference."
        ),
        "action_order": list(names),
        "nominal_joint_angles": np.asarray(env._default_pose).tolist(),
        "action_scale": float(env._config.action_scale),
        "actuation_semantics": (
            "q_target=nominal_joint_angles+action_scale*action. Each position actuator "
            "has force=clip(position_gain*q_target+bias_q*q+bias_qvel*qvel, force_range). "
            "Joint damping and frictionloss are separate passive generalized forces; "
            "the actuator force limit is not a limit on their sum. Gear is unity. "
            "Actuators have no activation dynamics; armature is equivalent joint inertia."
        ),
        "physics": {
            "gravity": model.opt.gravity.tolist(),
            "sim_dt": float(model.opt.timestep),
            "ctrl_dt": float(env._config.ctrl_dt),
            "total_mass": float(model.body_mass.sum()),
            "collision_scope": "Only feet collide with floor; no body-ground or self collisions",
            "geom_type_codes": {"0": "plane", "2": "sphere"},
        },
        "bodies": bodies,
        "joints_in_action_order": joints,
        "sites": sites,
        "active_collision_geoms": collisions,
    }


def robot_spec_json(spec: dict[str, Any]) -> str:
    return json.dumps(spec, sort_keys=True, separators=(",", ":"), allow_nan=False)


def robot_spec_sha256(spec: dict[str, Any]) -> str:
    return hashlib.sha256(robot_spec_json(spec).encode("utf-8")).hexdigest()


def robot_spec_context(spec: dict[str, Any]) -> str:
    return (
        "\n\n<static_robot_spec>\n"
        "The following static specification is extracted from this simulator's configured "
        "robot model. It describes geometry, coordinate frames, inertia, joints and actuation; "
        "it contains no current contact state, reward, demonstration or motion policy.\n"
        + robot_spec_json(spec)
        + "\n</static_robot_spec>"
    )
