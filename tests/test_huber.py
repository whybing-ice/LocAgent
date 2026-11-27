import numpy as np
import pytest

from locagent.algorithms.huber import estimate_position_huber
from locagent.algorithms.ols import estimate_position_ols
from locagent.simulation.data_generator import generate_noiseless_ranges


ANCHORS = [(0, 0), (10, 0), (0, 10), (10, 10), (5, 0), (5, 10)]


@pytest.mark.parametrize("target", [(2, 3), (0, 0), (12, -4)])
def test_recovers_noiseless_position(target):
    ranges = generate_noiseless_ranges(ANCHORS, target)
    np.testing.assert_allclose(estimate_position_huber(ANCHORS, ranges), target, atol=1e-10)


def test_reduces_error_from_single_outlier():
    target = (2, 3)
    ranges = generate_noiseless_ranges(ANCHORS, target)
    ranges[3] += 4
    huber_error = np.linalg.norm(estimate_position_huber(ANCHORS, ranges) - target)
    ols_error = np.linalg.norm(estimate_position_ols(ANCHORS, ranges) - target)
    assert huber_error < ols_error
    assert huber_error <= 0.8


@pytest.mark.parametrize("delta", [0, -1, np.inf, np.nan])
def test_rejects_invalid_delta(delta):
    with pytest.raises(ValueError, match="delta"):
        estimate_position_huber(ANCHORS, [5] * len(ANCHORS), delta=delta)


def test_rejects_negative_range():
    with pytest.raises(ValueError, match="nonnegative"):
        estimate_position_huber(ANCHORS, [-1, 5, 5, 5, 5, 5])


def test_rejects_collinear_anchors():
    with pytest.raises(ValueError, match="collinear"):
        estimate_position_huber([(0, 0), (1, 0), (2, 0)], [1, 2, 3])
