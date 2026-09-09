# GPT Dog Eval

Headless evaluation of arbitrary policies on the Unitree Go1 velocity-tracking
task from Google DeepMind's MuJoCo Playground. The simulator owns the physics,
observations, reward terms, and termination logic used by the upstream trainer.
This package only adapts that environment to Inspect Robots and records the
trainer's individual reward and cost terms.

The first milestone deliberately contains no LLM integration. It proves the
simulation, scoring, logging, and policy boundary with deterministic and ONNX
baselines. An agent-loop policy can be added after those invariants are stable.

## Setup

The target machine does not need a GPU. `uv` installs Python 3.12 and a CPU JAX
runtime into the project environment.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
cd ~/gpt-dog-eval
~/.local/bin/uv sync --extra dev
```

The first environment reset downloads the MuJoCo Menagerie assets pinned by
MuJoCo Playground and compiles the MJX step function for CPU. That first reset
is much slower than later steps.

## Run

List the registered components:

```bash
uv run inspect-robots list
```

Run a 50-step headless smoke test with a zero-action policy:

```bash
JAX_PLATFORM_NAME=cpu uv run inspect-robots run \
  --task go1-velocity-smoke -T steps=50 -T scenes=stand \
  --policy go1-zero --embodiment go1-playground --no-prompt
```

Run the upstream pretrained ONNX locomotion policy on all four commands:

```bash
JAX_PLATFORM_NAME=cpu uv run inspect-robots run \
  --task go1-velocity-smoke -T steps=250 \
  --policy go1-onnx --embodiment go1-playground --no-prompt
```

Each step records the exact weighted trainer terms plus their unscaled values
under `StepResult.info["trainer"]`. Task scorers report episode return, survival,
tracking rewards, and the main regularization costs.

## Components

- `go1-playground`: `Go1JoystickFlatTerrain`, CPU MJX, no window or rendering.
- `go1-zero`: emits twelve zeros, which requests the nominal standing pose.
- `go1-onnx`: upstream pretrained Go1 locomotion policy, CPU ONNX Runtime.
- `go1-velocity-smoke`: stand, forward, lateral, and left-turn commands.

Actions are normalized joint-position residuals in this order:

```text
FR_hip FR_thigh FR_calf FL_hip FL_thigh FL_calf
RR_hip RR_thigh RR_calf RL_hip RL_thigh RL_calf
```

MuJoCo Playground converts each value `a` to
`nominal_joint_angle + 0.5 * a` radians before its PD controller.

