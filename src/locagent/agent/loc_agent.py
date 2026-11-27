"""Choose an observation window and estimate the current position."""

from dataclasses import dataclass, replace
from typing import Literal

import numpy as np

from locagent.algorithms.huber import estimate_position_huber


@dataclass(frozen=True)
class RangeReport:
    timestamp: float
    anchor_id: str
    position: tuple[float, float]
    distance: float


@dataclass(frozen=True)
class ReportAssessment:
    anchor_id: str
    timestamp: float
    status: Literal["normal", "suspect_kept", "rejected"] | None
    reason: str
    conflict: float | None = None
    improvement: float | None = None


@dataclass(frozen=True)
class Decision:
    position: tuple[float, float] | None
    window: float
    anchor_count: int
    reason: str
    rejected_anchor_id: str | None = None
    assessments: tuple[ReportAssessment, ...] = ()


def _latest_reports(reports, now, window):
    latest = {}
    for report in reports:
        if now - window <= report.timestamp <= now:
            previous = latest.get(report.anchor_id)
            if previous is None or report.timestamp >= previous.timestamp:
                latest[report.anchor_id] = report
    return list(latest.values())


def _has_geometry(reports):
    if len(reports) < 3:
        return False
    anchors = np.asarray([report.position for report in reports])
    return np.linalg.matrix_rank(anchors - anchors.mean(axis=0)) == 2


def _solve(reports, delta):
    return estimate_position_huber(
        [report.position for report in reports],
        [report.distance for report in reports],
        delta=delta,
    )


def _absolute_residuals(reports, position):
    offsets = np.asarray([report.position for report in reports]) - position
    return np.abs(np.hypot(offsets[:, 0], offsets[:, 1]) - [r.distance for r in reports])


def _unchecked(reports, reason):
    return tuple(ReportAssessment(r.anchor_id, r.timestamp, None, reason) for r in reports)


def _screen_reports(reports, initial, delta, threshold):
    if len(reports) < 5:
        return initial, None, _unchecked(reports, "too_few_anchors")

    initial_residuals = _absolute_residuals(reports, initial)
    assessments = []
    suspects = []
    for index, report in enumerate(reports):
        remaining = reports[:index] + reports[index + 1:]
        if not _has_geometry(remaining):
            assessments.append(ReportAssessment(report.anchor_id, report.timestamp, None,
                                                "insufficient_remaining_geometry"))
            continue
        try:
            position = _solve(remaining, delta)
        except RuntimeError:
            assessments.append(ReportAssessment(report.anchor_id, report.timestamp, None,
                                                "leave_one_out_failed"))
            continue
        conflict = _absolute_residuals([report], position)[0]
        before = np.delete(initial_residuals, index).mean()
        improvement = before - _absolute_residuals(remaining, position).mean()
        if conflict >= threshold:
            suspects.append((index, position))
            status, reason = "suspect_kept", "large_conflict"
        else:
            status, reason = "normal", "within_threshold"
        assessments.append(ReportAssessment(report.anchor_id, report.timestamp, status, reason,
                                            float(conflict), float(improvement)))

    if len(suspects) > 1:
        for index, _ in suspects:
            assessments[index] = replace(assessments[index], reason="multiple_conflicts")
    if len(suspects) != 1:
        return initial, None, tuple(assessments)
    index, position = suspects[0]
    suspect = assessments[index]
    if any(item.status is None for item in assessments):
        assessments[index] = replace(suspect, reason="incomplete_screening")
    elif suspect.improvement < threshold / 100:
        assessments[index] = replace(suspect, reason="small_improvement")
    else:
        assessments[index] = replace(suspect, status="rejected", reason="unique_conflict")
        return position, suspect.anchor_id, tuple(assessments)
    return initial, None, tuple(assessments)


def locate(
    reports: list[RangeReport],
    now: float,
    *,
    moving: bool = False,
    short_window: float = 3,
    long_window: float = 10,
    delta: float = 0.5,
    rejection_threshold: float | None = None,
) -> Decision:
    """Choose a window and optionally screen one inconsistent report."""
    if rejection_threshold is not None and (
        not np.isfinite(rejection_threshold) or rejection_threshold <= 0
    ):
        raise ValueError("rejection_threshold must be finite and positive")
    window = long_window
    reason = "long_window"
    selected = _latest_reports(reports, now, long_window)
    if moving:
        recent = _latest_reports(selected, now, short_window)
        if _has_geometry(recent):
            selected, window, reason = recent, short_window, "motion"
        else:
            reason = "short_window_insufficient"
    if not _has_geometry(selected):
        reason = "insufficient_geometry"
        assessments = _unchecked(selected, reason) if rejection_threshold is not None else ()
        return Decision(None, window, len(selected), reason, assessments=assessments)
    try:
        position = _solve(selected, delta)
    except RuntimeError:
        assessments = _unchecked(selected, "solver_failed") if rejection_threshold is not None else ()
        return Decision(None, window, len(selected), "solver_failed", assessments=assessments)
    rejected = None
    assessments = ()
    if rejection_threshold is not None:
        position, rejected, assessments = _screen_reports(selected, position, delta, rejection_threshold)
    x, y = position
    return Decision((float(x), float(y)), window, len(selected) - int(rejected is not None),
                    reason, rejected, assessments)
