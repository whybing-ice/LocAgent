"""Replay Pettorru BLE CSVs using disjoint calibration and evaluation days."""

import argparse
import csv
from datetime import datetime
from itertools import groupby
from pathlib import Path

import numpy as np

from locagent.agent.loc_agent import RangeReport, locate


# Anchor coordinates from the source repository's deployment table.
ANCHORS = {
    "01": (18.15, 24.90), "02": (19.30, 30.95), "03": (23.00, 27.65),
    "04": (20.40, 22.85), "05": (17.70, 18.55), "06": (17.80, 24.65),
    "07": (20.45, 16.15), "08": (18.35, 18.50), "09": (17.20, 14.60),
    "0a": (22.45, 20.70), "0b": (20.50, 11.90), "0c": (25.10, 12.50),
    "0d": (23.70, 18.50), "0e": (20.40, 12.45), "0f": (18.70, 14.55),
    "10": (19.20, 9.95),
}


def read_rows(path):
    rows = []
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream, delimiter=";"):
            mac = row["MAC_ADDRESS"].lower()
            anchor = mac[-2:]
            if not mac.startswith("50:50:50:aa:00:") or anchor not in ANCHORS:
                continue
            rows.append((
                datetime.fromisoformat(row["TIMESTAMP"]), anchor, float(row["RSSI"]),
                (float(row["POSITION_X"]), float(row["POSITION_Y"])),
                abs(float(row["VELOCITY_LINEAR"])),
            ))
    return sorted(rows, key=lambda row: row[0])


def main(calibration_path, test_path, log_path, *, motion_confirmations=None):
    calibration, test = read_rows(calibration_path), read_rows(test_path)
    if not calibration or not test:
        raise ValueError("both files must contain matching BLE anchors")
    if {row[0].date() for row in calibration} & {row[0].date() for row in test}:
        raise ValueError("calibration and test dates must be disjoint")
    distances = [max(np.linalg.norm(np.subtract(target, ANCHORS[aid])), 1)
                 for _, aid, _, target, _ in calibration]
    slope, intercept = np.polyfit(np.log10(distances), [row[2] for row in calibration], 1)
    if slope >= 0:
        raise ValueError("calibration did not produce a decreasing RSSI model")
    print(f"Matched BLE rows: calibration {len(calibration)}, evaluation {len(test)}")
    print(f"RSSI slope {slope:.2f} dB/decade, intercept {intercept:.2f} dBm")
    reports = []
    previous_probe = None
    motion_streak = 0
    start = test[0][0]
    summary = {}
    coverage = []
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["timestamp", "scene", "method", "motion_suspected", "window_s",
                         "anchors", "reason", "x", "y", "error_m"])
        for timestamp, group in groupby(test, key=lambda row: row[0]):
            batch = list(group)
            now = (timestamp - start).total_seconds()
            reports = [report for report in reports if report.timestamp >= now - 45]
            for _, aid, rssi, _, _ in batch:
                distance = 10 ** ((rssi - intercept) / slope)
                reports.append(RangeReport(now, aid, ANCHORS[aid], distance))
            coverage.append((
                len({r.anchor_id for r in reports if r.timestamp >= now - 15}),
                len({r.anchor_id for r in reports}),
            ))
            short = locate(reports, now, long_window=15, delta=2)
            probe = short.position
            moving = False
            if probe is not None:
                if previous_probe is not None:
                    previous_time, previous_position = previous_probe
                    elapsed = now - previous_time
                    moving = elapsed <= 15 and (
                        np.linalg.norm(np.subtract(probe, previous_position)) / elapsed >= 0.10
                    )
                previous_probe = (now, probe)
            motion_streak = motion_streak + 1 if moving else 0
            fixed = locate(reports, now, long_window=45, delta=2)
            adaptive = locate(
                reports, now, moving=True, short_window=15, long_window=45, delta=2
            ) if moving else fixed
            target = np.mean([row[3] for row in batch], axis=0)
            scene = "moving" if np.median([row[4] for row in batch]) >= 0.02 else "stationary"
            decisions = [("fixed", fixed, None), ("fixed_short", short, None),
                         ("adaptive", adaptive, moving)]
            if motion_confirmations is not None:
                confirmed = motion_streak >= motion_confirmations
                decisions.append(("candidate", adaptive if confirmed else fixed, confirmed))
            for name, decision, motion in decisions:
                position = decision.position
                error = None if position is None else np.linalg.norm(position - target)
                summary.setdefault((scene, name), []).append(error)
                writer.writerow([
                    timestamp, scene, name, motion if motion is not None else "",
                    decision.window, decision.anchor_count, decision.reason,
                    *(position if position is not None else ("", "")), error,
                ])
    print(f"Median distinct anchors (15 s, 45 s): {np.median(coverage, axis=0)}")
    metrics_by_scene = {}
    for (scene, name), errors in sorted(summary.items()):
        solved = [error for error in errors if error is not None]
        cep90 = float(np.quantile(solved, 0.9)) if solved else None
        metrics_by_scene.setdefault(scene, {})[name] = {
            "solved": len(solved), "total": len(errors), "cep90_m": cep90,
        }
        metrics = f"CEP90 {cep90:.3f} m" if solved else "CEP90 unavailable"
        print(f"{scene} {name}: solved {len(solved)}/{len(errors)}, {metrics}")
    print(f"Decisions saved to {log_path}")
    return metrics_by_scene


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("calibration", type=Path)
    parser.add_argument("test", type=Path)
    parser.add_argument("--log", type=Path, default=Path("results/tables/public_ble_decisions.csv"))
    parser.add_argument("--motion-confirmations", type=int, choices=range(2, 6))
    args = parser.parse_args()
    main(args.calibration, args.test, args.log, motion_confirmations=args.motion_confirmations)
