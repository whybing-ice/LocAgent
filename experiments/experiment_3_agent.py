"""Compare fixed and adaptive windows on reproducible simulated BLE reports."""

import argparse
import csv
from pathlib import Path

import numpy as np

from locagent.agent.loc_agent import RangeReport, locate


def main(log_path: Path) -> None:
    rng = np.random.default_rng(42)
    anchors = [(0, 0), (14, 0), (0, 12), (14, 12), (7, 0), (7, 12)]
    # Fit the log-distance model on independent simulated calibration samples.
    calibration_distances = np.repeat([1, 2, 4, 8, 16], 30)
    calibration_rssi = -55 - 20 * np.log10(calibration_distances) + rng.normal(0, 1.2, 150)
    slope, intercept = np.polyfit(np.log10(calibration_distances), calibration_rssi, 1)
    errors = {(scene, method): [] for scene in ("stationary", "moving") for method in ("fixed", "adaptive")}

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["run", "second", "scene", "method", "motion_suspected", "window_s", "anchors", "reason", "x", "y", "error_m"])
        for run in range(20):
            reports = []
            previous_short = None
            origin = rng.uniform([3, 3], [5, 7])
            phase = rng.integers(0, 7, size=len(anchors))
            for second in range(30):
                scene = "moving" if second >= 15 else "stationary"
                target = origin + [max(second - 14, 0) * 0.45, 0]
                for index, anchor in enumerate(anchors):
                    period = 2 if index in (0, 3, 4) else 7
                    if (second + phase[index]) % period == 0:
                        distance = np.linalg.norm(target - anchor)
                        rssi = -55 - 20 * np.log10(distance) + rng.normal(0, 1.2)
                        estimated_range = 10 ** ((rssi - intercept) / slope)
                        reports.append(RangeReport(second, str(index), anchor, estimated_range))

                recent = locate(reports, second, long_window=3).position
                moving = (
                    recent is not None
                    and previous_short is not None
                    and np.linalg.norm(np.subtract(recent, previous_short)) >= 0.35
                )
                previous_short = recent
                fixed = locate(reports, second)
                adaptive = locate(reports, second, moving=moving) if moving else fixed
                for name, decision in [("fixed", fixed), ("adaptive", adaptive)]:
                    position = decision.position
                    error = None if position is None else np.linalg.norm(position - target)
                    errors[scene, name].append(error)
                    writer.writerow([
                        run, second, scene, name, moving if name == "adaptive" else "",
                        decision.window, decision.anchor_count, decision.reason,
                        *(position if position is not None else ("", "")), error,
                    ])

    for (scene, method), values in errors.items():
        solved = [value for value in values if value is not None]
        print(
            f"{scene} {method}: solved {len(solved)}/{len(values)}, "
            f"median {np.median(solved):.3f} m, CEP90 {np.quantile(solved, 0.9):.3f} m"
        )
    print(f"Decisions saved to {log_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, default=Path("results/tables/agent_decisions.csv"))
    main(parser.parse_args().log)
