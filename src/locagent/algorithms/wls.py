"""Linearized weighted least squares for 2D range localization."""

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .ols import _solve_position, _validate_observations


def estimate_position_wls(
    anchor_positions: ArrayLike, ranges: ArrayLike, weights: ArrayLike
) -> NDArray[np.float64]:
    """Estimate (x, y) using one finite, strictly positive weight per anchor."""
    anchors, distances = _validate_observations(anchor_positions, ranges)
    weights = np.asarray(weights, dtype=np.float64)
    if weights.shape != (len(anchors),):
        raise ValueError("weights must have shape (n,)")
    if not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError("weights must be finite and strictly positive")

    relative_weights = weights / weights.max()
    origin = np.average(anchors, axis=0, weights=relative_weights)
    offsets = anchors - origin
    rhs = np.sum(offsets**2, axis=1) - distances**2
    # Weighted centering eliminates the intercept of the squared-range equations.
    rhs -= np.average(rhs, weights=relative_weights)
    row_scales = np.sqrt(weights) / np.sqrt(weights.max())
    return origin + _solve_position(2 * offsets * row_scales[:, None], rhs * row_scales)


def estimate_position_calibrated_wls(
    anchor_positions: ArrayLike,
    ranges: ArrayLike,
    calibration_errors: ArrayLike,
    *,
    min_range_std: float = 0.05,
) -> NDArray[np.float64]:
    """Estimate position using calibration errors (measured minus true range).

    Use shape (n_anchors, n_samples) with at least two samples per anchor.
    min_range_std is a positive noise floor in the same units as ranges.
    """
    anchors, distances = _validate_observations(anchor_positions, ranges)
    errors = np.asarray(calibration_errors, dtype=np.float64)
    if errors.ndim != 2 or errors.shape[0] != len(anchors) or errors.shape[1] < 2:
        raise ValueError("calibration_errors must have at least two samples per anchor")
    if not np.isfinite(errors).all():
        raise ValueError("calibration_errors must be finite")
    if not np.isfinite(min_range_std) or min_range_std <= 0:
        raise ValueError("min_range_std must be finite and positive")

    corrected_ranges = np.maximum(distances - errors.mean(axis=1), 0)
    range_variance = np.maximum(errors.std(axis=1, ddof=1), min_range_std) ** 2
    # Approximate squared-range variance using centered Gaussian range noise.
    squared_range_variance = range_variance * (4 * corrected_ranges**2 + 2 * range_variance)
    weights = squared_range_variance.min() / squared_range_variance
    return estimate_position_wls(anchors, corrected_ranges, weights)
