"""Failure-rate arithmetic, the published-table check, Kaplan-Meier with delayed entry and AFR by age."""

from datetime import date

import numpy as np
import pandas as pd

from drivefail import reliability as rel


def test_poisson_ci_known_values() -> None:
    lo, hi = rel.poisson_rate_ci(np.array([0, 10]), np.array([1.0, 1.0]))
    assert lo[0] == 0 and np.isclose(hi[0], 3.6889, atol=1e-3)  # the "rule of 3.7" for zero events
    assert np.isclose(lo[1], 4.7954, atol=1e-3) and np.isclose(hi[1], 18.3904, atol=1e-3)


def test_afr_table_follows_backblaze_definition() -> None:
    days = pd.date_range("2025-10-01", "2025-10-10").date
    counts = pd.DataFrame({"date": np.repeat(days, 2), "model": ["A", "B"] * len(days), "drives": [1000, 500] * len(days),
                           "failures": [1, 0] * len(days), "hours_sum": [1000 * 8766.0, 500 * 2 * 8766.0] * len(days),
                           "hours_n": [1000, 500] * len(days), "capacity_bytes": [1, 1] * len(days)})
    drives = pd.DataFrame({"model": ["A", "A", "B"],
                           "first_date": [date(2025, 9, 1), date(2025, 10, 5), date(2025, 10, 1)],
                           "last_date": [date(2025, 10, 3), date(2025, 11, 1), date(2025, 10, 10)]})
    t = rel.afr_table(counts, drives, [("p", date(2025, 10, 1), date(2025, 10, 10))]).set_index("model")
    assert t.loc["A", "drive_days"] == 10_000 and t.loc["A", "failures"] == 10
    assert np.isclose(t.loc["A", "afr_pct"], 100 * 10 / (10_000 / 365))
    assert t.loc["A", "drive_count"] == 2  # distinct drives seen in the period
    assert np.isclose(t.loc["B", "avg_age_months"], 24.0, atol=0.05)
    assert t.loc["B", "afr_pct"] == 0 and t.loc["B", "afr_ci_high"] > 0


def test_crosscheck_flags_exact_and_inexact_models() -> None:
    pub = pd.DataFrame({"manufacturer": ["X", "X"], "model": ["M1", "M2"], "size_tb": [8, 8], "drive_count": [10, 20],
                        "avg_age_months": [1.0, 2.0], "drive_days": [900, 1800], "failures": [1, 2],
                        "afr_pct": [40.56, 40.56]})
    ours = pd.DataFrame({"model": ["M1", "M2"], "drive_count": [10, 20], "drive_days": [900, 1801], "failures": [1, 2],
                         "afr_pct": [40.555, 40.53], "afr_ci_low": [1, 1], "afr_ci_high": [2, 2], "avg_age_months": [1, 2]})
    c = rel.crosscheck(ours, pub).set_index("model")
    assert c.loc["M1", "exact_match"] and not c.loc["M2", "exact_match"]
    assert c.loc["M2", "drive_days_diff"] == 1


def test_kaplan_meier_with_delayed_entry_matches_hand_computation() -> None:
    # Five drives entering at different ages; failures at ages 3 and 5.
    surv = pd.DataFrame({"entry": [0.0, 1.0, 2.0, 2.5, 4.0], "exit": [3.0, 6.0, 5.0, 7.0, 8.0], "event": [1, 0, 1, 0, 0]})
    km = rel.kaplan_meier(surv, np.array([0.5, 3.5, 5.5]), min_at_risk=1).set_index("age_years")
    # At age 3: at risk = drives with entry < 3 <= exit -> 4 (the drive entering at 4 is not yet observed).
    # At age 5: at risk = {1.0-6.0, 2.0-5.0, 2.5-7.0, 4.0-8.0} -> 4.
    assert np.isclose(km.loc[3.5, "survival"], 1 - 1 / 4)
    assert np.isclose(km.loc[5.5, "survival"], (1 - 1 / 4) * (1 - 1 / 4))
    assert km.loc[3.5, "at_risk"] == 3  # the drive that failed at 3 has left


def test_afr_by_age_splits_exposure_across_bins() -> None:
    surv = pd.DataFrame({"entry": [0.5, 1.5], "exit": [1.5, 1.75], "event": [0, 1]})
    t = rel.afr_by_age(surv, np.array([0.0, 1.0, 2.0])).set_index("age_from")
    assert np.isclose(t.loc[0.0, "drive_years"], 0.5) and np.isclose(t.loc[1.0, "drive_years"], 0.75)
    assert t.loc[1.0, "failures"] == 1 and t.loc[0.0, "failures"] == 0
