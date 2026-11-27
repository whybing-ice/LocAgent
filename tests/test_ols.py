import numpy as np
import pytest

from locagent.algorithms.ols import estimate_position_ols
from locagent.simulation.data_generator import generate_noiseless_ranges


@pytest.mark.parametrize("target", [(2.0, 3.0), (0.0, 0.0), (12.0, -4.0)])
def test_recovers_noiseless_position(target):
    anchors = [(0, 0), (10, 0), (0, 10)]
    ranges = generate_noiseless_ranges(anchors, target)
    np.testing.assert_allclose(estimate_position_ols(anchors, ranges), target, atol=1e-12)


def test_fits_inconsistent_ranges_independently_of_anchor_order():
    anchors = np.array([(0, 0), (6, 0), (0, 8), (6, 8)])
    ranges = np.array([5, 5, 5, 7])
    expected = [2, 3.25]
    np.testing.assert_allclose(estimate_position_ols(anchors, ranges), expected)
    np.testing.assert_allclose(estimate_position_ols(anchors[::-1], ranges[::-1]), expected)


@pytest.mark.parametrize(
    ("anchors", "ranges", "message"),
    [
        ([(0, 0), (1, 0)], [1, 1], "three anchors"),
        ([(0, 0), (1, 1), (2, 2)], [1, 1, 1], "collinear"),
        ([0, 1, 2], [1, 1, 1], "anchor_positions"),
        ([(0, 0), (1, 0), (0, 1)], [1, 1], "ranges must have shape"),
        ([(0, 0), (1, 0), (0, 1)], [1, -1, 1], "nonnegative"),
        ([(0, 0), (1, 0), (0, 1)], [1, np.nan, 1], "finite"),
    ],
)
def test_rejects_invalid_inputs(anchors, ranges, message):
    with pytest.raises(ValueError, match=message):
        estimate_position_ols(anchors, ranges)
