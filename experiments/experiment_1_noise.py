"""Compare equal-weight and calibrated WLS on held-out simulated observations."""

import numpy as np

from locagent.algorithms.wls import estimate_position_calibrated_wls, estimate_position_wls
from locagent.simulation.data_generator import generate_noiseless_ranges


def main() -> None:
    rng = np.random.default_rng(42)
    anchors = [(0, 0), (10, 0), (0, 10), (10, 10)]
    bias = np.array([0.1, -0.2, 0.2, 1.5])
    noise_std = np.array([0.2, 0.3, 0.25, 1.2])
    calibration_errors = rng.normal(bias[:, None], noise_std[:, None], size=(4, 100))
    targets = rng.uniform(1, 9, size=(1000, 2))
    test_errors = rng.normal(bias, noise_std, size=(len(targets), len(anchors)))
    equal_errors = []
    calibrated_errors = []
    for target, measurement_error in zip(targets, test_errors, strict=True):
        ranges = np.maximum(generate_noiseless_ranges(anchors, target) + measurement_error, 0)
        equal = estimate_position_wls(anchors, ranges, [1] * len(anchors))
        calibrated = estimate_position_calibrated_wls(anchors, ranges, calibration_errors)
        equal_errors.append(np.linalg.norm(equal - target))
        calibrated_errors.append(np.linalg.norm(calibrated - target))

    for label, errors in [("Equal weights", equal_errors), ("Calibrated", calibrated_errors)]:
        errors = np.asarray(errors)
        print(
            f"{label}: <= 0.8 m {np.count_nonzero(errors <= 0.8)}/{len(errors)}, "
            f"median {np.median(errors):.3f} m, P90 {np.quantile(errors, 0.9):.3f} m"
        )

if __name__ == "__main__":
    main()
