import numpy as np

from locagent.agent.loc_agent import RangeReport, locate


def test_motion_prefers_recent_anchors_and_excludes_future_reports():
    anchors = [(0, 0), (6, 0), (0, 8)]
    reports = [RangeReport(9, str(i), anchor, 5) for i, anchor in enumerate(anchors)]
    reports.append(RangeReport(11, "0", anchors[0], 50))
    decision = locate(reports, 10, moving=True)
    assert (decision.window, decision.anchor_count, decision.reason) == (3, 3, "motion")
    np.testing.assert_allclose(decision.position, (3, 4), atol=1e-10)


def test_motion_falls_back_when_recent_geometry_is_insufficient():
    reports = [
        RangeReport(9, "a", (0, 0), 5),
        RangeReport(9, "b", (6, 0), 5),
        RangeReport(5, "c", (0, 8), 5),
        RangeReport(8, "a", (0, 0), 50),
    ]
    decision = locate(reports, 10, moving=True)
    assert (decision.window, decision.anchor_count) == (10, 3)
    assert decision.reason == "short_window_insufficient"
    np.testing.assert_allclose(decision.position, (3, 4), atol=1e-10)


def test_reports_insufficient_geometry_without_position():
    decision = locate([], 10)
    assert decision.position is None
    assert decision.reason == "insufficient_geometry"


def test_screening_rejects_a_single_conflicting_coordinate():
    anchors = [(0, 0), (6, 0), (0, 8), (6, 8), (3, 9), (8, 4)]
    target = (3, 4)
    reports = [
        RangeReport(0, str(i), anchor, np.linalg.norm(np.subtract(anchor, target)))
        for i, anchor in enumerate(anchors)
    ]
    clean = locate(reports, 0, rejection_threshold=3)
    assert all(item.status == "normal" for item in clean.assessments)
    reports[-1] = RangeReport(0, "5", (20, 4), 5)
    decision = locate(reports, 0, delta=1, rejection_threshold=3)
    assert decision.rejected_anchor_id == "5"
    assert decision.anchor_count == 5
    verdict = decision.assessments[-1]
    assert (verdict.anchor_id, verdict.timestamp, verdict.status) == ("5", 0, "rejected")
    np.testing.assert_allclose(verdict.conflict, 12)
    np.testing.assert_allclose(decision.position, target, atol=1e-10)


def test_conflict_with_little_effect_is_flagged_and_kept():
    angles = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    anchors = 5 * np.column_stack((np.cos(angles), np.sin(angles)))
    anchors[0] = (12, 0)
    reports = [RangeReport(2, str(i), tuple(anchor), 5) for i, anchor in enumerate(anchors)]
    baseline = locate(reports, 2, delta=0.1)
    screened = locate(reports, 2, delta=0.1, rejection_threshold=3)
    assert baseline.assessments == ()
    assert screened.position == baseline.position
    assert screened.anchor_count == len(anchors)
    suspect = screened.assessments[0]
    assert (suspect.status, suspect.reason) == ("suspect_kept", "small_improvement")
    assert suspect.conflict > 3 and suspect.improvement < 0.03


def test_multiple_conflicts_are_all_recorded_and_kept():
    anchors = [(0, 0), (6, 0), (0, 8), (6, 8), (3, 9), (20, 4), (-14, 4)]
    reports = [RangeReport(2, str(i), a, 5) for i, a in enumerate(anchors)]
    decision = locate(reports, 2, delta=1, rejection_threshold=3)
    flagged = [item for item in decision.assessments if item.status == "suspect_kept"]
    assert [item.anchor_id for item in flagged] == ["5", "6"]
    assert all(item.reason == "multiple_conflicts" for item in flagged)
    assert decision.rejected_anchor_id is None
    assert decision.anchor_count == len(anchors)


def test_too_few_anchors_are_marked_unchecked():
    reports = [RangeReport(1, str(i), a, 5) for i, a in enumerate([(0, 0), (6, 0), (0, 8)])]
    decision = locate(reports, 2, rejection_threshold=3)
    assert decision.position is not None
    assert len(decision.assessments) == 3
    assert all(item.status is None and item.conflict is None for item in decision.assessments)
