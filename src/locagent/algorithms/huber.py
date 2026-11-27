"""Robust 2D localization using Huber loss on range residuals."""

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import least_squares

from .ols import _validate_observations, estimate_position_ols


def estimate_position_huber(
    anchor_positions: ArrayLike, ranges: ArrayLike, *, delta: float = 0.5
) -> NDArray[np.float64]:
    """Estimate (x, y), with delta in the same units as the measured ranges."""
    anchors, distances = _validate_observations(anchor_positions, ranges)
    if not np.isfinite(delta) or delta <= 0:
        raise ValueError("delta must be finite and positive")
    initial = estimate_position_ols(anchors, distances)

    def residuals(position: NDArray[np.float64]) -> NDArray[np.float64]:
        offsets = position - anchors
        return np.hypot(offsets[:, 0], offsets[:, 1]) - distances

    result = least_squares(residuals, initial, loss="huber", f_scale=delta)
    if not result.success:
        raise RuntimeError(f"Huber optimization failed: {result.message}")
    return result.x
