"""Weighted ranking metrics, bootstrap, drive-level alerting and the cost-based threshold."""

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from drivefail import evaluation as ev


def _toy(n: int = 3000, seed: int = 0):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.05).astype(int)
    s = np.round(rng.normal(y * 1.2, 1.0), 1)  # rounding creates ties
    w = rng.choice([1.0, 5.0], n)
    return y, s, w


def test_weighted_metrics_match_scikit_learn() -> None:
    y, s, w = _toy()
    m = ev.row_metrics(y, s, w)
    assert np.isclose(m["ap"], average_precision_score(y, s, sample_weight=w))
    assert np.isclose(m["roc_auc"], roc_auc_score(y, s, sample_weight=w))


def test_integer_weights_equal_duplicated_rows() -> None:
    y, s, _ = _toy(800, seed=1)
    w = np.random.default_rng(2).integers(1, 4, len(y)).astype(float)
    rep = np.repeat(np.arange(len(y)), w.astype(int))
    a, b = ev.row_metrics(y, s, w), ev.row_metrics(y[rep], s[rep], None)
    assert all(np.isclose(a[k], b[k]) for k in a)


def test_recall_at_fpr_respects_the_cap() -> None:
    y = np.array([1, 1, 0, 1, 0, 0, 0, 0, 0, 0])
    s = np.array([10, 9, 8, 7, 6, 5, 4, 3, 2, 1], dtype=float)
    m = ev.row_metrics(y, s)
    assert m["recall_at_fpr_0.1pct"] == 2 / 3  # stop before the first negative
    assert m["recall_at_fpr_1pct"] == 2 / 3


def test_clustered_bootstrap_is_paired_and_centred() -> None:
    y, s, w = _toy(2000, seed=3)
    codes = np.arange(len(y)) // 4  # four rows per drive
    b = ev.clustered_bootstrap(y, w, codes, {"a": s, "b": s + 0.0}, n_boot=50, seed=0)
    pa, pb = b[b.model == "a"].ap.to_numpy(), b[b.model == "b"].ap.to_numpy()
    assert np.allclose(pa, pb)  # identical scores on identical resamples
    assert abs(np.median(pa) - ev.row_metrics(y, s, w)["ap"]) < 0.05


def _drives() -> pd.DataFrame:
    return pd.DataFrame({"serial_number": list("abcdef"), "positive": [1, 1, 0, 0, 0, 0],
                         "weight": [1.0, 1.0, 2.0, 2.0, 2.0, 2.0], "max_score": [0.9, 0.3, 0.8, 0.2, 0.1, 0.05]})


def test_alert_outcomes_counts_and_saving() -> None:
    o = ev.alert_outcomes(_drives(), np.array([0.85, 0.5, 0.25]), cost_ratio=10).set_index("threshold")
    assert (o.loc[0.85, "tp"], o.loc[0.85, "fp"]) == (1.0, 0.0)
    assert (o.loc[0.5, "tp"], o.loc[0.5, "fp"]) == (1.0, 2.0)  # weighted false alarm
    assert (o.loc[0.25, "tp"], o.loc[0.25, "fp"]) == (2.0, 2.0)
    assert o.loc[0.25, "net_saving"] == 9 * 2 - 2
    assert ev.best_threshold(_drives(), cost_ratio=10) <= 0.3  # catching both failures is worth one false alarm
    assert ev.best_threshold(_drives(), cost_ratio=2) > 0.8  # at 2:1 a false alarm (weight 2) is not worth it


def test_drive_level_drops_censored_healthy_drives() -> None:
    rows = pd.DataFrame({"serial_number": ["a", "a", "b", "b", "c"], "label": [0, 1, 0, 0, 0],
                         "censored": [False, False, False, True, False], "weight": [1, 1, 5, 5, 5],
                         "s": [0.1, 0.7, 0.2, 0.3, 0.4]})
    d = ev.drive_level(rows, "s").set_index("serial_number")
    assert list(d.index) == ["a", "c"]
    assert d.loc["a", "max_score"] == 0.7 and d.loc["a", "positive"] == 1


def test_lead_times_and_holm() -> None:
    rows = pd.DataFrame({"serial_number": ["a"] * 4 + ["b"] * 2, "date": [1, 2, 3, 4, 1, 2],
                         "label": [0, 1, 1, 1, 1, 1], "days_to_failure": [40, 25, 10, 0, 1, 0],
                         "s": [0.1, 0.9, 0.2, 0.8, 0.1, 0.1]})
    lt = ev.lead_times(rows, "s", 0.5).set_index("serial_number")
    assert lt.loc["a", "lead_days"] == 25 and not lt.loc["b", "caught"]
    assert np.allclose(ev.holm(np.array([0.01, 0.04, 0.03])), [0.03, 0.06, 0.06])
