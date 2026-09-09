import numpy as np
import pytest

from gpt_dog_eval.constants import ACTION_SPACE, JOINT_LABELS, split_policy_observation


def test_action_contract() -> None:
    assert ACTION_SPACE.shape == (12,)
    assert ACTION_SPACE.semantics is not None
    assert ACTION_SPACE.semantics.dim_labels == JOINT_LABELS


def test_split_policy_observation_preserves_layout() -> None:
    source = np.arange(48, dtype=np.float32)
    fields = split_policy_observation(source)
    np.testing.assert_array_equal(fields["base_lin_vel"], source[0:3])
    np.testing.assert_array_equal(fields["last_action"], source[33:45])
    np.testing.assert_array_equal(fields["command"], source[45:48])


def test_split_policy_observation_rejects_wrong_shape() -> None:
    with pytest.raises(ValueError, match="shape"):
        split_policy_observation(np.zeros(47, dtype=np.float32))
