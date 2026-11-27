"""Compare localization methods with and without a single range outlier."""

import numpy as np

from locagent.algorithms.huber import estimate_position_huber
from locagent.algorithms.ols import estimate_position_ols
from locagent.algorithms.wls import estimate_position_calibrated_wls, estimate_position_wls
from locagent.simulation.data_generator import generate_noiseless_ranges


def main() -> None:
    rng = np.random.default_rng(42)
    anchors = [(0, 0), (10, 0), (0, 10), (10, 10), (5, 0), (5, 10)]
    noise_std = np.array([0.2, 0.3, 0.25, 0.35, 0.2, 0.25])
    calibration = rng.normal(0, noise_std[:, None], size=(len(anchors), 100))
    targets = rng.uniform(1, 9, size=(1000, 2))
    noise = rng.normal(0, noise_std, size=(len(targets), len(anchors)))
    outliers = rng.integers(len(anchors), size=len(targets))
    weights = 1 / noise_std**2

    for label, outlier_size in [("Noise only", 0), ("One outlier", 4)]:
        errors = {name: [] for name in ("OLS", "WLS", "Calibrated WLS", "Huber")}
        for target, offsets, outlier in zip(targets, noise, outliers, strict=True):
            ranges = generate_noiseless_ranges(anchors, target) + offsets
            ranges[outlier] += outlier_size
            ranges = np.maximum(ranges, 0)
            estimates = (
                estimate_position_ols(anchors, ranges),
                estimate_position_wls(anchors, ranges, weights),
                estimate_position_calibrated_wls(anchors, ranges, calibration),
                estimate_position_huber(anchors, ranges),
            )
            for values, position in zip(errors.values(), estimates, strict=True):
                values.append(np.linalg.norm(position - target))

        print(label)
        for name, values in errors.items():
            rate = np.mean(np.asarray(values) <= 0.8)
            print(f"  {name}: {rate:.1%} within 0.8 m, P90 {np.quantile(values, 0.9):.3f} m")

if __name__ == "__main__":
    main()
