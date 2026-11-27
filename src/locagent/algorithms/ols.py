"""Linearized ordinary least squares for 2D range localization."""

import numpy as np
from numpy.typing import ArrayLike, NDArray


def estimate_position_ols(
    anchor_positions: ArrayLike, ranges: ArrayLike
) -> NDArray[np.float64]:
    """Estimate (x, y) from at least three non-collinear anchors and their ranges."""
    anchors, distances = _validate_observations(anchor_positions, ranges)
    origin = anchors.mean(axis=0)
    offsets = anchors - origin
    # Center the squared-range equations to eliminate the common target norm.
    rhs = np.sum(offsets**2, axis=1) - distances**2
    return origin + _solve_position(2 * offsets, rhs - rhs.mean())


def _validate_observations(
    anchor_positions: ArrayLike, ranges: ArrayLike
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    anchors = np.asarray(anchor_positions, dtype=np.float64)
    distances = np.asarray(ranges, dtype=np.float64)
    if anchors.ndim != 2 or anchors.shape[1] != 2:
        raise ValueError("anchor_positions must have shape (n, 2)")
    if len(anchors) < 3:
        raise ValueError("at least three anchors are required")
    if distances.shape != (len(anchors),):
        raise ValueError("ranges must have shape (n,)")
    if not (np.isfinite(anchors).all() and np.isfinite(distances).all()):
        raise ValueError("anchors and ranges must be finite")
    if np.any(distances < 0):
        raise ValueError("ranges must be nonnegative")

    return anchors, distances


def _solve_position(
    matrix: NDArray[np.float64], rhs: NDArray[np.float64]
) -> NDArray[np.float64]:
    position, _, rank, _ = np.linalg.lstsq(matrix, rhs, rcond=None)
    if rank < 2:
        raise ValueError("anchor positions must not be collinear")
    return position
