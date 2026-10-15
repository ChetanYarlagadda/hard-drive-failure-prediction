"""Fleet reliability: annualized failure rates with exact Poisson intervals, the check against Backblaze's published
table, Kaplan-Meier survival by power-on age with delayed entry, and failure rates by age."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from scipy.stats import chi2

HOURS_PER_YEAR = 24 * 365.25
PUBLISHED_SOURCE = ("Backblaze, 'Backblaze Drive Stats for 2025' (blog post, February 2026), table 'Q4 2025 "
                    "quarterly hard drive failure rates'")


def poisson_rate_ci(k: np.ndarray, exposure: np.ndarray, alpha: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """Exact (Garwood) confidence interval of a Poisson rate k / exposure."""
    k = np.asarray(k, dtype=float)
    exposure = np.asarray(exposure, dtype=float)
    lo = np.where(k > 0, chi2.ppf(alpha / 2, 2 * k) / 2, 0.0) / exposure
    hi = chi2.ppf(1 - alpha / 2, 2 * k + 2) / 2 / exposure
    return lo, hi


def distinct_drives(drives: pd.DataFrame, a: date, b: date) -> pd.Series:
    """Number of distinct drives per model reporting at least once between ``a`` and ``b``."""
    first, last = pd.to_datetime(drives.first_date).dt.date, pd.to_datetime(drives.last_date).dt.date
    return drives[(first <= b) & (last >= a)].groupby("model").size()


def afr_table(counts: pd.DataFrame, drives: pd.DataFrame, periods: list[tuple[str, date, date]]) -> pd.DataFrame:
    """AFR per model and period from daily (date, model) counts and the per-drive table.

    AFR = failures / (drive days / 365) x 100%, as Backblaze defines it. The drive count is the number of distinct
    drives seen in the period (the definition that reproduces Backblaze's published counts); the mean age is taken
    on the last day of the period. Only models present in ``drives`` are reported.
    """
    c = counts[counts.model.isin(drives.model.unique())].copy()
    c["date"] = pd.to_datetime(c["date"]).dt.date
    out = []
    for name, a, b in periods:
        p = c[(c.date >= a) & (c.date <= b)]
        g = p.groupby("model").agg(drive_days=("drives", "sum"), failures=("failures", "sum"),
                                   capacity_bytes=("capacity_bytes", "max"))
        last = p[p.date == b].set_index("model")
        g["drive_count"] = distinct_drives(drives, a, b).reindex(g.index).fillna(0).astype(int)
        g["avg_age_months"] = (last.hours_sum / last.hours_n / HOURS_PER_YEAR * 12).reindex(g.index)
        g["afr_pct"] = 100 * g.failures / (g.drive_days / 365)
        lo, hi = poisson_rate_ci(g.failures.to_numpy(), g.drive_days.to_numpy() / 365)
        g["afr_ci_low"], g["afr_ci_high"] = 100 * lo, 100 * hi
        g.insert(0, "period", name)
        g["days"] = (b - a).days + 1
        out.append(g.reset_index())
    return pd.concat(out, ignore_index=True)


def load_published(path: Path) -> pd.DataFrame:
    """The published Q4 2025 table, transcribed from the Backblaze blog."""
    return pd.read_csv(path)


def crosscheck(ours: pd.DataFrame, published: pd.DataFrame) -> pd.DataFrame:
    """Side-by-side comparison of recomputed and published drive counts, drive days, failures and AFR."""
    ours = ours[["model", "drive_count", "drive_days", "failures", "afr_pct", "afr_ci_low", "afr_ci_high", "avg_age_months"]]
    m = published.merge(ours, on="model", how="left", suffixes=("_published", "_recomputed"))
    for c in ("drive_count", "drive_days", "failures"):
        m[f"{c}_recomputed"] = m[f"{c}_recomputed"].fillna(0).astype(int)
        m[f"{c}_diff"] = m[f"{c}_recomputed"] - m[f"{c}_published"]
    m["afr_pct_recomputed"] = m["afr_pct_recomputed"].round(2)
    m["afr_diff_pct_points"] = (m.afr_pct_recomputed - m.afr_pct_published).round(2)
    m["avg_age_months_recomputed"] = m["avg_age_months_recomputed"].round(1)
    m["exact_match"] = (m.drive_days_diff == 0) & (m.failures_diff == 0) & (m.drive_count_diff == 0)
    cols = ["manufacturer", "model", "size_tb", "drive_count_published", "drive_count_recomputed", "drive_days_published",
            "drive_days_recomputed", "failures_published", "failures_recomputed", "afr_pct_published",
            "afr_pct_recomputed", "afr_ci_low", "afr_ci_high", "avg_age_months_published", "avg_age_months_recomputed",
            "drive_count_diff", "drive_days_diff", "failures_diff", "afr_diff_pct_points", "exact_match"]
    return m[cols]


def survival_frame(drives: pd.DataFrame, max_age_years: float = 15.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Entry age, exit age (years of power-on time) and event flag per drive, plus the drives dropped as invalid.

    A drive is kept when its power-on hours are present, non-decreasing, below ``max_age_years`` and grow by no more
    than the calendar time it was observed (plus two days of slack).
    """
    d = drives.copy()
    d["entry"] = d.hours_first / HOURS_PER_YEAR
    d["exit"] = d.hours_last / HOURS_PER_YEAR
    span_days = (pd.to_datetime(d.last_date) - pd.to_datetime(d.first_date)).dt.days
    ok = (d.hours_first.notna() & d.hours_last.notna() & (d.hours_first >= 0) & (d.hours_last >= d.hours_first)
          & (d.exit < max_age_years) & ((d.hours_last - d.hours_first) <= (span_days + 2) * 24))
    d["exit"] = np.maximum(d.exit, d.entry + 1 / HOURS_PER_YEAR)  # one hour minimum so entry < exit
    d["event"] = d.failed.astype(int)
    return d[ok].reset_index(drop=True), d[~ok].reset_index(drop=True)


def at_risk(surv: pd.DataFrame, ages: np.ndarray) -> np.ndarray:
    """Number of drives under observation at each age (entered before it, not yet exited)."""
    entry, exit_ = np.sort(surv.entry.to_numpy()), np.sort(surv.exit.to_numpy())
    return np.searchsorted(entry, ages, side="left") - np.searchsorted(exit_, ages, side="left")


def truncate(surv: pd.DataFrame, age: float) -> pd.DataFrame:
    """Drives still under observation after ``age``, with entry moved up to ``age`` (conditioning on T > age)."""
    s = surv[surv.exit > age].copy()
    s["entry"] = np.maximum(s.entry, age)
    return s


def kaplan_meier(surv: pd.DataFrame, grid: np.ndarray, min_at_risk: int = 200) -> pd.DataFrame:
    """Kaplan-Meier survival against power-on age with delayed entry (left truncation), on an age grid.

    Drives enter the risk set at the age they had on their first day in the data. Where few drives are at risk, one
    failure moves the estimate a lot, so the curve covers only the longest run of grid ages with at least
    ``min_at_risk`` drives at risk: it starts at that run's first age a0, S(t) = P(T > t | T > a0), and ends with
    the run.
    """
    n = at_risk(surv, grid)
    ok = n >= min_at_risk
    if not ok.any():
        return pd.DataFrame(columns=["age_years", "survival", "ci_low", "ci_high", "at_risk", "start_age"])
    edges = np.flatnonzero(np.diff(np.r_[0, ok.astype(int), 0]))  # run starts and (exclusive) ends
    starts, ends = edges[::2], edges[1::2]
    i = int(np.argmax(ends - starts))
    lo_i, hi_i = starts[i], ends[i] - 1
    a0 = float(grid[lo_i])
    s = truncate(surv, a0)
    kmf = KaplanMeierFitter().fit(s.exit, event_observed=s.event, entry=s.entry)
    sf = kmf.survival_function_at_times(grid).to_numpy()
    ci = kmf.confidence_interval_survival_function_
    lo = np.interp(grid, ci.index.to_numpy(), ci.iloc[:, 0].to_numpy())
    hi = np.interp(grid, ci.index.to_numpy(), ci.iloc[:, 1].to_numpy())
    out = pd.DataFrame({"age_years": grid, "survival": sf, "ci_low": lo, "ci_high": hi, "at_risk": n, "start_age": a0})
    return out.iloc[lo_i:hi_i + 1].reset_index(drop=True)


def km_one_year_failure(surv: pd.DataFrame, age: float) -> float:
    """KM probability of failing within one year for a drive alive at ``age`` years: 1 - S(age + 1 | T > age)."""
    s = truncate(surv, age)
    kmf = KaplanMeierFitter().fit(s.exit, event_observed=s.event, entry=s.entry)
    return float(1 - kmf.survival_function_at_times([age + 1.0]).to_numpy()[0])


def afr_by_age(surv: pd.DataFrame, edges: np.ndarray, by: str | None = None) -> pd.DataFrame:
    """Failures per drive-year of exposure in power-on age bins (each drive's observed span split across bins)."""
    groups = [(None, surv)] if by is None else list(surv.groupby(by))
    out = []
    for key, g in groups:
        for a, b in zip(edges[:-1], edges[1:]):
            exposure = np.clip(np.minimum(g.exit, b) - np.maximum(g.entry, a), 0, None).sum()
            fails = int(((g.event == 1) & (g.exit >= a) & (g.exit < b)).sum())
            lo, hi = poisson_rate_ci(np.array([fails]), np.array([exposure])) if exposure > 0 else ([np.nan], [np.nan])
            out.append({"group": key if key is not None else "all", "age_from": a, "age_to": b,
                        "drive_years": exposure, "failures": fails,
                        "afr_pct": 100 * fails / exposure if exposure > 0 else np.nan,
                        "afr_ci_low": 100 * lo[0], "afr_ci_high": 100 * hi[0]})
    return pd.DataFrame(out)
