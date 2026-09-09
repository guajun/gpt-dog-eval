from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from inspect_robots import Observation

from gpt_dog_eval.embodiment import Go1PlaygroundEmbodiment
from gpt_dog_eval.policies import ZeroGo1Policy, go1_zero
from gpt_dog_eval.scorers import EpisodeReturn, MeanTrainerTerm, Survived
from gpt_dog_eval.tasks import go1_velocity_smoke


def test_embodiment_constructs_without_heavy_runtime() -> None:
    embodiment = Go1PlaygroundEmbodiment()
    assert embodiment.info.action_space.shape == (12,)
    assert embodiment.info.control_hz == 50.0


def test_embodiment_validates_configuration() -> None:
    with pytest.raises(ValueError, match="env_name"):
        Go1PlaygroundEmbodiment(env_name="Ant")
    with pytest.raises(ValueError, match="noise_level"):
        Go1PlaygroundEmbodiment(noise_level=2.0)


def test_zero_policy_emits_nominal_residual() -> None:
    policy = ZeroGo1Policy()
    chunk = policy.act(Observation(state={"policy_obs": np.zeros(48, dtype=np.float32)}))
    np.testing.assert_array_equal(chunk.actions[0].data, np.zeros(12, dtype=np.float32))
    assert chunk.control_hz == 50.0
    with pytest.raises(TypeError, match="does not accept"):
        go1_zero(unexpected=True)


def test_task_selects_scenes_and_trainer_metrics() -> None:
    task = go1_velocity_smoke(steps=10, scenes="stand,turn", epochs=2)
    assert task.max_steps == 10
    assert task.epochs == 2
    assert [scene.id for scene in task.scenes] == ["stand", "turn"]
    assert any(scorer.name == "trainer_mean/energy" for scorer in task.scorers)


def test_task_rejects_invalid_selection() -> None:
    with pytest.raises(ValueError, match="scenes"):
        go1_velocity_smoke(scenes="moonwalk")


def _record(*results: object, termination_reason: str = "max_steps") -> Any:
    steps = [SimpleNamespace(result=result) for result in results]
    return SimpleNamespace(steps=steps, termination_reason=termination_reason)


def test_scorers_read_recorded_trainer_values() -> None:
    result_a = SimpleNamespace(
        reward=0.25,
        info={"trainer": {"raw_terms": {"energy": 2.0}}},
    )
    result_b = SimpleNamespace(
        reward=0.5,
        info={"trainer": {"raw_terms": {"energy": 4.0}}},
    )
    record = _record(result_a, result_b)
    assert EpisodeReturn()(record, None).value == pytest.approx(0.75)
    assert MeanTrainerTerm("energy")(record, None).value == pytest.approx(3.0)
    assert Survived()(record, None).value is True
    assert Survived()(_record(termination_reason="fell"), None).value is False
