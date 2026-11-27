import numpy as np
import pytest

from locagent.algorithms.ols import estimate_position_ols
from locagent.algorithms.wls import estimate_position_calibrated_wls, estimate_position_wls
from locagent.simulation.data_generator import generate_noiseless_ranges


ANCHORS = [(0, 0), (10, 0), (0, 10), (10, 10)]
TARGET = (2, 3)


def test_recovers_noiseless_position():
    ranges = generate_noiseless_ranges(ANCHORS, TARGET)
    estimate = estimate_position_wls(ANCHORS, ranges, [1, 2, 3, 4])
    np.testing.assert_allclose(estimate, TARGET, atol=1e-12)


def test_equal_weights_match_ols_for_inconsistent_ranges():
    ranges = [4, 8, 7, 14]
    np.testing.assert_allclose(
        estimate_position_wls(ANCHORS, ranges, [5] * 4),
        estimate_position_ols(ANCHORS, ranges),
    )


def test_downweighting_biased_range_reduces_position_error():
    ranges = generate_noiseless_ranges(ANCHORS, TARGET)
    ranges[-1] += 4
    ols = estimate_position_ols(ANCHORS, ranges)
    wls = estimate_position_wls(ANCHORS, ranges, [1, 1, 1, 0.05])
    assert np.linalg.norm(wls - TARGET) < np.linalg.norm(ols - TARGET)


@pytest.mark.parametrize("weights", [[1, 1], [1, 1, 1, 0], [1, -1, 1, 1], [1, 1, np.inf, 1]])
def test_rejects_invalid_weights(weights):
    with pytest.raises(ValueError, match="weights"):
        estimate_position_wls(ANCHORS, [5] * 4, weights)


def test_rejects_collinear_anchors():
    with pytest.raises(ValueError, match="collinear"):
        estimate_position_wls([(0, 0), (1, 0), (2, 0)], [1, 2, 3], [1, 2, 3])


def test_calibrated_wls_removes_constant_range_bias():
    bias = np.array([0.2, 0.5, 0.8, 2.0])
    ranges = generate_noiseless_ranges(ANCHORS, TARGET) + bias
    errors = np.repeat(bias[:, None], 3, axis=1)
    estimate = estimate_position_calibrated_wls(ANCHORS, ranges, errors)
    np.testing.assert_allclose(estimate, TARGET, atol=1e-12)


def test_calibrated_wls_downweights_noisy_anchors():
    ranges = generate_noiseless_ranges(ANCHORS, TARGET)
    ranges[-1] += 2
    errors = [[-0.1, 0.1]] * 3 + [[-2, 2]]
    calibrated = estimate_position_calibrated_wls(ANCHORS, ranges, errors)
    baseline = estimate_position_wls(ANCHORS, ranges, [1] * 4)
    assert np.linalg.norm(calibrated - TARGET) < np.linalg.norm(baseline - TARGET)


@pytest.mark.parametrize("errors", [[[0]] * 4, [[0, 0]] * 3, [[0, np.nan]] * 4])
def test_calibrated_wls_rejects_invalid_calibration(errors):
    with pytest.raises(ValueError, match="calibration_errors"):
        estimate_position_calibrated_wls(ANCHORS, [5] * 4, errors)
