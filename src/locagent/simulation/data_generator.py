"""TODO: Generate anchor positions and noisy range observations."""



import numpy as np
from numpy.typing import ArrayLike, NDArray


def generate_noiseless_ranges(
    anchor_positions: ArrayLike, target_position: ArrayLike
) -> NDArray[np.float64]:
    """Return the Euclidean distance from each 2D anchor to the target."""
    anchors = np.asarray(anchor_positions, dtype=np.float64)
    target = np.asarray(target_position, dtype=np.float64)
    if anchors.ndim != 2 or anchors.shape[1] != 2:
        raise ValueError("anchor_positions must have shape (n, 2)")
    if target.shape != (2,):
        raise ValueError("target_position must have shape (2,)")
    if not (np.isfinite(anchors).all() and np.isfinite(target).all()):
        raise ValueError("positions must contain only finite coordinates")

    offsets = anchors - target
    return np.hypot(offsets[:, 0], offsets[:, 1])

