# Go1 static robot specification experiment

This condition adds the simulator's static robot description to the Astra system
context. It leaves the v2 observation, action execution, tools, reward and physics
unchanged. There is no training, trajectory demonstration, contact observation,
FK/IK tool or stabilizing controller. The experiment tests whether explicit robot
geometry and dynamics improve zero-shot joint-chunk control.

`build_robot_spec()` reads the configured `env.mj_model`, including runtime PD
overrides, and the environment's actual nominal joint pose. The JSON contains:

- The 13 rigid bodies' parent transforms, masses, centers of mass, principal
  inertias and inertia-frame rotations.
- The 12 actuated joints' anchors, axes, zero references, limits, reflected
  inertia, damping and friction; actuator gains, gear and force limits.
- The nominal joint angles and action scaling, IMU and foot-site transforms,
  SI units, quaternion convention and observation coordinate meanings.
- Gravity, physics/control timestep and the active collision geometry. The
  existing `feetonly` model only supports foot-floor contact, not body-ground or
  self contact. This is a description of the current simulator, not a claim of
  identified real-hardware accuracy.

The agent receives compact JSON and explanatory text on every request via the
existing `instructions` field. The complete prompt has 14,327 characters for the
current model. Source XML is archived beside each trial for auditing; meshes and
duplicate XML text are not added to the language-model context. The static JSON
contains no reward coefficients, rollout states or previous experimental results.

Each scene starts a distinct process, policy and history. Static descriptions are
identical across scenes; dynamic histories are independent. This continues the
application-managed conversation state pattern in the [official Responses
documentation](https://developers.openai.com/api/docs/guides/conversation-state).

```bash
JAX_PLATFORM_NAME=cpu uv run python scripts/verify_robot_spec.py \
  --output outputs/robot-spec-verification/geometry.json
uv run python scripts/run_astra_parallel.py --robot-spec --fake \
  --output outputs/fake-robot-spec-v3-validation
uv run python scripts/run_astra_parallel.py --robot-spec \
  --implementation-commit <source-commit>
uv run python scripts/analyze_feedback_batch.py outputs/<new-batch> \
  --previous-batch outputs/astra-feedback-v2-20260910T083830Z
```

Omitting `--robot-spec` reproduces the v2 prompt condition. Real runs retain
`gpt-6-astra`, xhigh reasoning, 60 calls per scene, a 250-step horizon, at most
15 steps and 4 keyframes per chunk, and a maximum action delta of 0.1 per control
step. Seed, noise, perturbation and termination settings are unchanged.

New comparisons include Zero, native ONNX-PPO, the previous Astra v2 batch, and
Astra with the static specification. All pre-v2 Astra runs are excluded from new
comparisons at the user's request; their historical logs and reports are retained.
The analyzer includes per-scene returns, trajectories and weighted return changes
with an explicit clipping adjustment. Early termination is labeled and shared
time-window returns are reported separately.

Verification before real inference: 30 tests pass locally and remotely, with
Ruff/mypy checks passing. Exported FK agrees with MuJoCo at 100 site positions
across 20 random body/joint poses (maximum error 2.50e-16 m); link COM and rotated
inertia checks pass. Four independent Fake-provider trials complete 1,000 steps,
reproduce historical Zero returns exactly, retain the identical specification in
every transcript and preserve final 15-requested/10-executed receipts. No paid
inference is used for these checks. Initial test collection encountered a missing
test-module import; the fixture was made self-contained before verification.

Current specification SHA-256:
`9fe71127ced2957c048a9e454a716360d2e0f3d4f334dd72696812cbb4449cf9`.

Results and complete trial IDs are recorded in the [experiment journal](experiment-journal.md)
and [issue #1](https://github.com/guajun/gpt-dog-eval/issues/1).
