"""Evaluation: weighted row-level metrics, drive-clustered bootstrap, drive-level alerting and cost-based thresholds.

Rows carry inverse-probability weights from the drive sampling (healthy drives are subsampled), so every metric is
computed with weights and estimates the value on the full fleet.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class SortedScores:
    """Rows sorted by descending score, with the positions where a run of tied scores ends."""

    order: np.ndarray
    ends: np.ndarray  # index (in sorted order) of the last row of every distinct score

    @classmethod
    def build(cls, score: np.ndarray) -> "SortedScores":
        order = np.argsort(-score, kind="mergesort")
        s = score[order]
        ends = np.flatnonzero(np.r_[s[1:] != s[:-1], True])
        return cls(order, ends)


def _curve(ss: SortedScores, y: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Weighted cumulative true- and false-positive mass at every distinct threshold."""
    yw = (y * w)[ss.order]
    nw = ((1 - y) * w)[ss.order]
    tp = np.cumsum(yw)[ss.ends]
    fp = np.cumsum(nw)[ss.ends]
    return tp, fp, float(yw.sum()), float(nw.sum())


def _metrics_from_curve(tp: np.ndarray, fp: np.ndarray, p: float, n: float) -> dict[str, float]:
    if p == 0 or n == 0:
        return {"ap": float("nan"), "roc_auc": float("nan"), "recall_at_fpr_0.1pct": float("nan"),
                "recall_at_fpr_1pct": float("nan")}
    precision = tp / np.maximum(tp + fp, 1e-300)
    recall, fpr = tp / p, fp / n
    tpr0, fpr0 = np.r_[0.0, recall], np.r_[0.0, fpr]
    out = {"ap": float(np.sum(np.diff(tpr0) * precision)),
           "roc_auc": float(np.sum(np.diff(fpr0) * (tpr0[1:] + tpr0[:-1]) / 2))}
    for name, cap in (("recall_at_fpr_0.1pct", 0.001), ("recall_at_fpr_1pct", 0.01)):
        ok = fpr <= cap
        out[name] = float(recall[ok].max()) if ok.any() else 0.0
    return out


def row_metrics(y: np.ndarray, score: np.ndarray, w: np.ndarray | None = None,
                ss: SortedScores | None = None) -> dict[str, float]:
    """Weighted average precision (sum_k (R_k - R_{k-1}) P_k, as in scikit-learn), ROC AUC (ties count half) and
    recall at 0.1% and 1% false-positive rate."""
    w = np.ones(len(y)) if w is None else w
    ss = ss or SortedScores.build(score)
    return _metrics_from_curve(*_curve(ss, y, w))


def average_precision(y: np.ndarray, score: np.ndarray, w: np.ndarray | None = None) -> float:
    """Weighted average precision."""
    return row_metrics(y, score, w)["ap"]


def clustered_bootstrap(y: np.ndarray, w: np.ndarray, drive_codes: np.ndarray, scores: dict[str, np.ndarray],
                        n_boot: int, seed: int = 0) -> pd.DataFrame:
    """Row-level metrics of several scores on the same bootstrap resamples of drives (paired).

    Drives are resampled with Poisson(1) multiplicities (the Poisson bootstrap), which keeps all rows of a drive
    together; this matters because the daily rows of one failing drive are strongly correlated.
    """
    rng = np.random.default_rng(seed)
    n_drives = int(drive_codes.max()) + 1
    sorted_ = {k: SortedScores.build(s) for k, s in scores.items()}
    rows = []
    for b in range(n_boot):
        wb = w * rng.poisson(1.0, n_drives)[drive_codes]
        for k in scores:
            rows.append({"boot": b, "model": k, **row_metrics(y, scores[k], wb, sorted_[k])})
    return pd.DataFrame(rows)


def curve_points(y: np.ndarray, score: np.ndarray, w: np.ndarray, n_points: int = 400) -> pd.DataFrame:
    """Weighted precision, recall and false-positive rate at up to ``n_points`` thresholds (for plotting)."""
    tp, fp, p, n = _curve(SortedScores.build(score), y, w)
    recall = tp / p
    spaced = np.searchsorted(recall, np.linspace(0, 1, n_points), side="left")
    keep = np.unique(np.clip(np.r_[np.arange(min(50, len(tp))), spaced, len(tp) - 1], 0, len(tp) - 1))
    return pd.DataFrame({"recall": tp[keep] / p, "precision": tp[keep] / np.maximum(tp[keep] + fp[keep], 1e-12),
                         "fpr": fp[keep] / n})


def summarize_bootstrap(point: pd.DataFrame, boot: pd.DataFrame, baseline: str, alpha: float = 0.05) -> pd.DataFrame:
    """Point estimates with percentile CIs, and paired differences to ``baseline`` with a two-sided p-value."""
    metrics = [c for c in boot.columns if c not in ("boot", "model")]
    out = []
    for _, r in point.iterrows():
        b = boot[boot.model == r.model].set_index("boot")
        base = boot[boot.model == baseline].set_index("boot")
        for m in metrics:
            d = (b[m] - base[m]).dropna()
            rec = {"model": r.model, "metric": m, "value": r[m],
                   "ci_low": b[m].quantile(alpha / 2), "ci_high": b[m].quantile(1 - alpha / 2)}
            if r.model != baseline:
                base_value = point.loc[point.model == baseline, m].iloc[0]
                rec.update({"versus": baseline, "diff": r[m] - base_value, "diff_ci_low": d.quantile(alpha / 2),
                            "diff_ci_high": d.quantile(1 - alpha / 2),
                            "p_value": float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())))})
            out.append(rec)
    return pd.DataFrame(out)


def holm(p: np.ndarray) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values."""
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for i, j in enumerate(order):
        running = max(running, (len(p) - i) * p[j])
        adj[j] = min(1.0, running)
    return adj


def drive_level(rows: pd.DataFrame, score_col: str) -> pd.DataFrame:
    """One record per drive over an evaluation window: did it fail, its weight and its highest daily score.

    Drives whose window ends in censoring (they leave the fleet without a recorded failure) are dropped because
    their outcome is unknown.
    """
    g = rows.groupby("serial_number", sort=False)
    d = pd.DataFrame({"positive": g["label"].max().astype(int), "weight": g["weight"].first(),
                      "censored": g["censored"].max().astype(bool), "max_score": g[score_col].max()})
    d = d[~(d.censored & (d.positive == 0))]
    return d.reset_index()


def alert_outcomes(drives: pd.DataFrame, thresholds: np.ndarray, cost_ratio: float, n_fleet: float | None = None
                   ) -> pd.DataFrame:
    """Weighted confusion counts and net saving of flagging every drive whose max score reaches each threshold.

    Cost model: a proactive swap costs 1 unit; a failure that was not flagged costs ``cost_ratio`` units. Net saving
    against no model = (cost_ratio - 1) * TP - FP, reported in swap units per 10,000 drives in the window.
    """
    s = drives.max_score.to_numpy()
    pos = drives.positive.to_numpy() == 1
    w = drives.weight.to_numpy()
    n_fleet = float(w.sum()) if n_fleet is None else n_fleet
    order = np.argsort(-s, kind="mergesort")
    s_sorted = s[order]
    tp_c = np.cumsum((w * pos)[order])
    fp_c = np.cumsum((w * ~pos)[order])
    k = np.searchsorted(-s_sorted, -np.asarray(thresholds), side="right")  # number of drives with score >= t
    tp = np.where(k > 0, tp_c[np.maximum(k - 1, 0)], 0.0)
    fp = np.where(k > 0, fp_c[np.maximum(k - 1, 0)], 0.0)
    p = float((w * pos).sum())
    saving = (cost_ratio - 1) * tp - fp
    return pd.DataFrame({"threshold": thresholds, "tp": tp, "fp": fp, "fn": p - tp,
                         "recall": tp / p if p else np.nan,
                         "precision": np.where(tp + fp > 0, tp / np.maximum(tp + fp, 1e-12), np.nan),
                         "flagged_per_10k": 1e4 * (tp + fp) / n_fleet,
                         "net_saving": saving, "net_saving_per_10k": 1e4 * saving / n_fleet,
                         "saving_share": saving / (cost_ratio * p) if p else np.nan})


def threshold_grid(scores: np.ndarray, n: int = 400) -> np.ndarray:
    """Candidate thresholds: quantiles of the scores, concentrated in the upper tail."""
    q = 1 - np.geomspace(1e-5, 1, n)
    return np.unique(np.r_[np.quantile(scores, np.clip(q, 0, 1)), np.inf])[::-1]


def best_threshold(drives: pd.DataFrame, cost_ratio: float) -> float:
    """Threshold with the highest net saving (ties: the higher threshold)."""
    grid = threshold_grid(drives.max_score.to_numpy())
    out = alert_outcomes(drives, grid, cost_ratio)
    return float(out.threshold.iloc[int(np.argmax(out.net_saving.to_numpy()))])


def lead_times(rows: pd.DataFrame, score_col: str, threshold: float) -> pd.DataFrame:
    """For every failing drive, days between its first alert (score >= threshold) and its failure."""
    pos_serials = rows.loc[rows.label == 1, "serial_number"].unique()
    r = rows[rows.serial_number.isin(pos_serials)].sort_values(["serial_number", "date"])
    hit = r[r[score_col] >= threshold].groupby("serial_number")["days_to_failure"].max()  # earliest alert
    out = pd.DataFrame({"serial_number": pos_serials})
    out["lead_days"] = out.serial_number.map(hit)
    out["caught"] = out.lead_days.notna()
    return out
