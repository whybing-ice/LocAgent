import numpy as np
import pytest

from locagent.simulation.data_generator import generate_noiseless_ranges


def test_noiseless_ranges_follow_anchor_order():
    ranges = generate_noiseless_ranges([(5, 3), (2, -1), (-4, -9)], (2, -1))
    np.testing.assert_array_equal(ranges, [5.0, 0.0, 10.0])


def test_empty_anchor_array_returns_empty_ranges():
    assert generate_noiseless_ranges(np.empty((0, 2)), (0, 0)).shape == (0,)


@pytest.mark.parametrize(
    ("anchors", "target", "message"),
    [
        ([0, 1], (0, 0), "anchor_positions"),
        ([(0, 1)], (0, 0, 0), "target_position"),
        ([(np.nan, 0)], (0, 0), "finite"),
        ([(0, 0)], (0, np.inf), "finite"),
    ],
)
def test_rejects_invalid_coordinates(anchors, target, message):
    with pytest.raises(ValueError, match=message):
        generate_noiseless_ranges(anchors, target)

