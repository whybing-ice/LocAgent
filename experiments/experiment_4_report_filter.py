"""Measure the effect of screening one inconsistent report under coordinate faults."""

import argparse
import csv
from pathlib import Path

import numpy as np

from locagent.agent.loc_agent import RangeReport, locate


def main(seed, audit=None):
    rng = np.random.default_rng(seed)
    angles = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    anchors = 15 * np.column_stack((np.cos(angles), np.sin(angles)))
    trials = 300
    targets = rng.uniform(-7, 7, (trials, 2))
    noise = rng.normal(0, 0.35, (trials, len(anchors)))
    fault_indices = rng.integers(len(anchors), size=trials)
    fault_angles = rng.uniform(0, 2 * np.pi, trials)
    if audit is not None:
        audit.writerow(["seed", "shift_m", "trial", "anchor_id", "timestamp", "injected",
                        "status", "reason", "conflict_m", "improvement_m"])
    print("shift_m,correct,wrong,baseline_solved,filtered_solved,baseline_cep90,filtered_cep90,"
          "faulty_flagged,faulty_total,healthy_flagged,healthy_total")
    for shift in (0, 5, 10, 20):
        baseline_errors, filtered_errors = [], []
        correct = wrong = 0
        faulty_flagged = healthy_flagged = 0
        scenarios = zip(targets, noise, fault_indices, fault_angles, strict=True)
        for trial, (target, errors, index, angle) in enumerate(scenarios):
            distances = np.linalg.norm(anchors - target, axis=1) + errors
            reported = anchors.copy()
            reported[index] += shift * np.array([np.cos(angle), np.sin(angle)])
            reports = [
                RangeReport(0, str(i), tuple(position), float(distance))
                for i, (position, distance) in enumerate(zip(reported, distances))
            ]
            baseline = locate(reports, 0, delta=1)
            filtered = locate(reports, 0, delta=1, rejection_threshold=3)
            for assessment in filtered.assessments:
                injected = shift > 0 and assessment.anchor_id == str(index)
                if assessment.status in ("suspect_kept", "rejected"):
                    if injected:
                        faulty_flagged += 1
                    else:
                        healthy_flagged += 1
                if audit is not None:
                    audit.writerow([seed, shift, trial, assessment.anchor_id, assessment.timestamp,
                                    injected, assessment.status, assessment.reason,
                                    assessment.conflict, assessment.improvement])
            for decision, values in [(baseline, baseline_errors), (filtered, filtered_errors)]:
                if decision.position is not None:
                    values.append(np.linalg.norm(np.subtract(decision.position, target)))
            rejected = filtered.rejected_anchor_id
            if rejected is not None:
                if shift and rejected == str(index):
                    correct += 1
                else:
                    wrong += 1
        faulty_total = trials if shift else 0
        healthy_total = trials * len(anchors) - faulty_total
        print(
            f"{shift},{correct},{wrong},{len(baseline_errors)},{len(filtered_errors)},"
            f"{np.quantile(baseline_errors, 0.9):.3f},{np.quantile(filtered_errors, 0.9):.3f},"
            f"{faulty_flagged},{faulty_total},{healthy_flagged},{healthy_total}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log", type=Path, help="Save one assessment per window report")
    args = parser.parse_args()
    if args.log is None:
        main(args.seed)
    else:
        args.log.parent.mkdir(parents=True, exist_ok=True)
        with args.log.open("w", encoding="utf-8", newline="") as stream:
            main(args.seed, csv.writer(stream))
