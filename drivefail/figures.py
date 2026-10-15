"""All report figures, drawn from the result tables in reports/results/."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from drivefail import plotting
from drivefail.features import label as feature_label
from drivefail.plotting import INK, INK_2, MFG_COLOR, MODEL_COLOR, MUTED, SERIES

RULE = "SMART rule (critical-counter sum)"
ORDER = [RULE, "Logistic regression", "LightGBM", "XGBoost", "CatBoost"]


def _suptitle(fig: plt.Figure, text: str, gap_in: float = 0.3) -> None:
    """Left-aligned takeaway title placed ``gap_in`` inches above the top of the axes."""
    fig.suptitle(text, x=0.01, y=fig.subplotpars.top + gap_in / fig.get_figheight(), ha="left", va="bottom",
                 fontsize=12, fontweight="bold", color=INK)


def _short(model: str) -> str:
    return "SMART rule" if model == RULE else model


def fig_data_window(daily: pd.DataFrame, meta: dict, path: Path, title: str) -> Path:
    """Drives reporting and failures per day, with the warm-up, train, gap and test periods shaded."""
    s = meta["settings"]
    d = daily.copy()
    d["date"] = pd.to_datetime(d["date"])
    fig, ax = plt.subplots(figsize=(12, 4.2))
    ax.bar(d.date, d.failures, color=SERIES[7], width=0.8, label="Failures per day (modelled drives)")
    ax.set_ylabel("Failures per day")
    ax2 = ax.twinx()
    ax2.plot(d.date, d.drives / 1e3, color=SERIES[0], lw=2, label="Drives reporting (thousands)")
    ax2.set_ylabel("Drives reporting (thousands)", color=INK_2)
    ax2.set_ylim(0, d.drives.max() / 1e3 * 1.25)
    ax2.grid(False)
    ax2.spines["right"].set_visible(True)
    h = pd.Timedelta(days=s["horizon"] - 1)
    one = pd.Timedelta(days=1)
    spans = [(pd.Timestamp(s["start"]), pd.Timestamp(s["train_start"]) - one, "Feature\nwarm-up", MUTED, 0.10),
             (pd.Timestamp(s["train_start"]), pd.Timestamp(s["train_end"]), "Train (A) and\nvalidation (B) rows",
              SERIES[0], 0.16),
             (pd.Timestamp(s["train_end"]) + one, pd.Timestamp(s["test_start"]) - one, "Gap: training\nlabels resolve",
              MUTED, 0.06),
             (pd.Timestamp(s["test_start"]), pd.Timestamp(s["test_end"]), "Test rows\n(unseen drives C)", SERIES[1], 0.16),
             (pd.Timestamp(s["test_end"]) + one, pd.Timestamp(s["test_end"]) + h, "Test labels\nresolve", MUTED, 0.06)]
    top = d.failures.max() * 1.5
    for a, b, lab, col, alpha in spans:
        b = b + pd.Timedelta(hours=23)
        ax.axvspan(a, b, color=col, alpha=alpha, lw=0)
        ax.text(a + (b - a) / 2, top * 0.97, lab, ha="center", va="top", fontsize=8.5, color=INK_2)
    ax.set_ylim(0, top)
    ax.set_xlim(d.date.min() - one, d.date.max() + one)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=8.5, ncol=2, bbox_to_anchor=(0.0, -0.1))
    ax.set_title(f"{len(d)} daily snapshots, {d.drives.sum() / 1e6:.1f}M drive-days of the 33 modelled drive models; "
                 f"label = the drive fails within the next {s['horizon']} days", loc="left", fontsize=9.5)
    _suptitle(fig, title, gap_in=0.35)
    plotting.save(fig, path)
    return path


def fig_afr_check(check: pd.DataFrame, path: Path, title: str) -> Path:
    """Recomputed Q4 2025 AFR per model with 95% Poisson CIs against Backblaze's published values."""
    c = check.sort_values("afr_pct_published").reset_index(drop=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 7.6), gridspec_kw={"width_ratios": [1.6, 1], "wspace": 0.3})
    ax = axes[0]
    y = np.arange(len(c))
    ax.hlines(y, c.afr_ci_low, c.afr_ci_high, color=SERIES[0], lw=1.5, alpha=0.6)
    ax.plot(c.afr_pct_recomputed, y, "o", color=SERIES[0], ms=5, label="Recomputed from the daily files (95% CI)")
    ax.plot(c.afr_pct_published, y, "|", color=INK, ms=11, mew=2, label="Published by Backblaze")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{m} ({s} TB)" for m, s in zip(c.model, c.size_tb)], fontsize=8)
    ax.set_xscale("symlog", linthresh=0.5)
    ax.set_xlim(0, 25)
    ax.set_xticks([0, 0.5, 1, 2, 5, 10, 20])
    ax.set_xticklabels(["0", "0.5", "1", "2", "5", "10", "20"])
    ax.set_xlabel("Annualized failure rate, Q4 2025 (%, symlog scale)")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right", fontsize=8.5)
    ax.set_title("AFR = failures / (drive days / 365), one row per drive model", loc="left")
    ax = axes[1]
    ax.scatter(c.drive_days_published / 1e6, c.drive_days_recomputed / 1e6, s=18, color=SERIES[0])
    lim = max(c.drive_days_published.max(), c.drive_days_recomputed.max()) / 1e6 * 1.05
    ax.plot([0, lim], [0, lim], color=MUTED, lw=1, ls=":")
    ax.set_xlabel("Published drive days (millions)")
    ax.set_ylabel("Recomputed drive days (millions)")
    n_exact = int(c.exact_match.sum())
    fail_diff = int(np.abs(c.failures_diff).sum())
    ax.set_title(f"Drive days, failures and drive counts identical\nfor {n_exact} of {len(c)} models "
                 f"(total absolute failure difference: {fail_diff})", loc="left", fontsize=9.5)
    _suptitle(fig, title)
    plotting.save(fig, path)
    return path


def fig_survival(km: pd.DataFrame, by_age: pd.DataFrame, summary: pd.DataFrame, path: Path, title: str) -> Path:
    """Kaplan-Meier survival by power-on age per model and AFR by age per manufacturer."""
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5), gridspec_kw={"wspace": 0.25})
    ax = axes[0]
    sm = summary.set_index("model")
    for i, m in enumerate(summary.sort_values("drives", ascending=False).model):
        g = km[km.model == m]
        if g.empty:
            continue
        col = SERIES[i % len(SERIES)]
        ax.plot(g.age_years, g.survival, color=col, lw=1.8, ls="-" if i < 4 else "--",
                label=f"{m} ({sm.drives[m]:,} drives)")
        ax.fill_between(g.age_years, g.ci_low, g.ci_high, color=col, alpha=0.12, lw=0)
    ax.set_xlabel("Power-on age (years)")
    ax.set_ylabel("Survival probability, S(t | alive at curve start)")
    ax.set_title("Kaplan-Meier with delayed entry, 8 largest drive models\n(ages with at least 1,000 drives at risk)",
                 loc="left", fontsize=9.5)
    ax.legend(fontsize=7.8, loc="lower left", ncol=2)
    ax = axes[1]
    for mfg in ["HGST", "Seagate", "Toshiba", "WDC"]:
        g = by_age[(by_age.group == mfg) & (by_age.drive_years >= 500)]
        mid = (g.age_from + g.age_to) / 2
        ax.errorbar(mid, g.afr_pct, yerr=[g.afr_pct - g.afr_ci_low, g.afr_ci_high - g.afr_pct], color=MFG_COLOR[mfg],
                    marker="o", ms=4, lw=1.6, capsize=2, label=mfg)
    g = by_age[(by_age.group == "all") & (by_age.drive_years >= 500)]
    ax.plot((g.age_from + g.age_to) / 2, g.afr_pct, color=INK, lw=1.2, ls="--", label="All modelled drives")
    ax.set_xlabel("Power-on age (years; 1-year bins with at least 500 drive-years)")
    ax.set_ylabel("Failures per 100 drive-years (%, 95% CI)")
    ax.set_title("Failure rate by age and manufacturer, Oct 2025 - Feb 2026\n(ST8000DM002 left out: its power-on "
                 "hours wrap around)", loc="left", fontsize=9.5)
    ax.set_ylim(0, None)
    ax.legend(fontsize=8.5, loc="upper left")
    _suptitle(fig, title, gap_in=0.6)
    plotting.save(fig, path)
    return path


def fig_models(pr: pd.DataFrame, res: pd.DataFrame, curves: pd.DataFrame, cost: pd.DataFrame, fixed: pd.DataFrame,
               ratio: float, path: Path, title: str) -> Path:
    """Row-level precision-recall on the test month and drive-level recall against drives flagged."""
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), gridspec_kw={"wspace": 0.22})
    ax = axes[0]
    r = res.set_index("model")
    for m in ORDER:
        g = pr[(pr.model == m) & (pr.recall > 0)]
        ax.plot(g.recall, g.precision, color=MODEL_COLOR[m], lw=1.8,
                label=f"{_short(m)}: AP {r.ap[m]:.3f} [{r.ap_ci_low[m]:.3f}, {r.ap_ci_high[m]:.3f}]")
    ax.set_xlabel("Recall (share of drive-days within 30 days of a failure)")
    ax.set_ylabel("Precision, weighted to the full fleet (log scale)")
    ax.set_yscale("log")
    ax.set_ylim(1e-4, 1)
    ax.set_xlim(0, 1)
    ax.set_title("Drive-days of the test month: will the drive fail within 30 days?", loc="left")
    ax.legend(fontsize=8.3, loc="upper right")
    ax = axes[1]
    for m in ORDER:
        g = curves[(curves.model == m) & (curves.split == "test")].sort_values("flagged_per_10k")
        ax.plot(g.flagged_per_10k, 100 * g.recall, color=MODEL_COLOR[m], lw=1.8, label=_short(m))
        c = cost[(cost.model == m) & (cost.cost_ratio == ratio)]
        if len(c) and c.test_flagged_per_10k.iloc[0] > 0:
            ax.plot(c.test_flagged_per_10k, 100 * c.test_recall, "o", color=MODEL_COLOR[m], ms=8, mec="white")
    fr = fixed[fixed.cost_ratio == ratio].iloc[0]
    ax.plot(fr.test_flagged_per_10k, 100 * fr.test_recall, "X", color=INK, ms=9,
            label=f"Any critical counter > 0: {fr.test_flagged_per_10k:.0f} flagged, {100 * fr.test_recall:.0f}% caught")
    ax.set_xscale("log")
    ax.set_xlim(1, 3e3)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Drives flagged per 10,000 drives in the test month (log scale)")
    ax.set_ylabel("Failing drives flagged before failure (%)")
    ax.set_title(f"Unseen drives, one month. Dots: threshold chosen on validation\nfor a failure costing {ratio:g}x a "
                 "proactive swap", loc="left", fontsize=9.5)
    ax.legend(fontsize=8.3, loc="upper left")
    _suptitle(fig, title, gap_in=0.5)
    plotting.save(fig, path)
    return path


def fig_protocols(prot: pd.DataFrame, path: Path, title: str) -> Path:
    """Average precision and ROC AUC of the same LightGBM settings under four train/test split protocols."""
    p = prot.reset_index(drop=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 3.9), gridspec_kw={"wspace": 0.06})
    cols = [SERIES[7], SERIES[3], SERIES[6], SERIES[0]][:len(p)]
    y = np.arange(len(p))[::-1]
    for ax, metric, lab in ((axes[0], "ap", "Average precision"), (axes[1], "roc_auc", "ROC AUC")):
        ax.hlines(y, p[f"{metric}_ci_low"], p[f"{metric}_ci_high"], color=cols, lw=3, alpha=0.5)
        ax.scatter(p[metric], y, color=cols, s=60, zorder=3)
        for yi, v in zip(y, p[metric]):
            ax.text(v, yi + 0.22, f"{v:.3f}", ha="center", fontsize=8.5, color=INK_2)
        ax.set_xlabel(f"{lab} (95% CI, drives resampled)")
        ax.grid(axis="y", visible=False)
        ax.set_ylim(-0.6, len(p) - 0.4)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([f"{a}\n(test: {b:,} failing drives)" for a, b in zip(p.protocol, p.test_failing_drives)],
                            fontsize=8.5)
    axes[1].set_yticks(y)
    axes[1].set_yticklabels([])
    _suptitle(fig, title)
    plotting.save(fig, path)
    return path


def fig_cost(curves: pd.DataFrame, cost: pd.DataFrame, best: str, ratio: float, path: Path, title: str) -> Path:
    """Net saving against the number of drives flagged (validation and test) and saving by cost ratio."""
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8), gridspec_kw={"wspace": 0.25})
    ax = axes[0]
    for split, ls in (("val", "--"), ("test", "-")):
        for m in (best, RULE):
            g = curves[(curves.model == m) & (curves.split == split)].sort_values("flagged_per_10k")
            g = g[g.flagged_per_10k > 0]
            ax.plot(g.flagged_per_10k, g.net_saving_per_10k, color=MODEL_COLOR[m], ls=ls, lw=1.8,
                    label=f"{_short(m)}, {'validation drives (B), Nov' if split == 'val' else 'test drives (C), Jan'}")
    c = cost[(cost.model == best) & (cost.cost_ratio == ratio)].iloc[0]
    ax.plot(c.test_flagged_per_10k, c.test_net_saving_per_10k, "o", color=MODEL_COLOR[best], ms=8, mec="white")
    ax.annotate(f"validation threshold applied to test:\n{c.test_flagged_per_10k:.1f} flagged per 10k, net saving "
                f"{c.test_net_saving_per_10k:.1f}", (c.test_flagged_per_10k, c.test_net_saving_per_10k),
                xytext=(0.58, 0.9), textcoords="axes fraction", fontsize=8.5, color=INK_2, va="top",
                arrowprops={"arrowstyle": "-", "color": MUTED})
    ax.axhline(0, color=MUTED, lw=1)
    ax.set_xscale("log")
    ax.set_xlim(0.3, 3e3)
    top = float(curves[curves.model == best].net_saving_per_10k.max())
    ax.set_ylim(-max(10.0, top * 1.2), top * 1.5)
    ax.set_xlabel("Drives flagged per 10,000 drives per month (log scale)")
    ax.set_ylabel("Net saving per 10,000 drives (swap units)")
    ax.set_title(f"Failure = {ratio:g} swaps: net saving = {ratio - 1:g} x failures caught - false alarms", loc="left",
                 fontsize=9.5)
    ax.legend(fontsize=8.3, loc="lower left")
    ax = axes[1]
    ratios = sorted(cost.cost_ratio.unique())
    width = 0.8 / len(ORDER)
    x = np.arange(len(ratios))
    for i, m in enumerate(ORDER):
        g = cost[cost.model == m].set_index("cost_ratio").reindex(ratios)
        ax.bar(x + (i - 2) * width, 100 * g.test_saving_share, width, color=MODEL_COLOR[m], label=_short(m))
    ax.axhline(0, color=MUTED, lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r:g}x" for r in ratios])
    ax.set_xlabel("Cost of an unpredicted failure relative to a proactive swap")
    ax.set_ylabel("Test saving (% of the cost of all failures)")
    ax.set_title("Threshold chosen on validation for each cost ratio, applied to the test month", loc="left",
                 fontsize=9.5)
    ax.legend(fontsize=8, loc="upper left", ncol=2)
    _suptitle(fig, title)
    plotting.save(fig, path)
    return path


def fig_shap(npz: dict, imp: pd.DataFrame, path: Path, title: str, top: int = 14) -> Path:
    """SHAP beeswarm of the most important features and mean |SHAP| for failing vs healthy drive-days."""
    feats = list(npz["features"])
    sv, X, lab = npz["shap"], npz["X"], npz["label"]
    order = imp.feature.tolist()[:top]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.2), gridspec_kw={"width_ratios": [1.5, 1], "wspace": 0.75})
    ax = axes[0]
    rng = np.random.default_rng(0)
    cmap = plt.get_cmap("coolwarm")
    for i, f in enumerate(order[::-1]):
        j = feats.index(f)
        x = X[:, j].astype(float)
        col = np.full(len(x), 0.5)
        ok = np.isfinite(x)
        if f not in ("model_code", "mfg_code") and ok.sum() > 1:
            col[ok] = pd.Series(x[ok]).rank(pct=True).to_numpy()
        ax.scatter(sv[:, j], i + rng.uniform(-0.3, 0.3, len(x)), c=cmap(col), s=4, alpha=0.6, lw=0, rasterized=True)
    ax.axvline(0, color=MUTED, lw=1)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([feature_label(f) for f in order[::-1]], fontsize=8.5)
    ax.set_xlabel("SHAP value (contribution to the log-odds of failing within 30 days)")
    ax.grid(axis="y", visible=False)
    ax.set_title(f"{len(sv):,} test drive-days ({int(lab.sum()):,} within 30 days of a failure);\ncolour = feature "
                 "value rank (blue low, red high, grey for categories)", loc="left", fontsize=9)
    ax = axes[1]
    g = imp.set_index("feature").loc[order[::-1]]
    y = np.arange(len(g))
    ax.barh(y + 0.2, g.mean_abs_shap_failing, 0.4, color=SERIES[7], label="Drive-days before a failure")
    ax.barh(y - 0.2, g.mean_abs_shap_healthy, 0.4, color=SERIES[0], label="Healthy drive-days")
    ax.set_yticks(y)
    ax.set_yticklabels([feature_label(f) for f in g.index], fontsize=8.5)
    ax.set_xlabel("Mean |SHAP value|")
    ax.grid(axis="y", visible=False)
    ax.legend(fontsize=8.5, loc="lower right")
    ax.set_title("Importance by outcome", loc="left")
    _suptitle(fig, title, gap_in=0.6)
    plotting.save(fig, path)
    return path


def fig_lead(timeline: pd.DataFrame, lead: pd.DataFrame, best: str, path: Path, title: str) -> Path:
    """Share of failing drives flagged by days before failure, and lead time of the first alert."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4), gridspec_kw={"wspace": 0.25})
    ax = axes[0]
    t = timeline.sort_values("days_to_failure")
    ax.plot(t.days_to_failure, 100 * t.share_flagged, color=MODEL_COLOR[best], lw=2, marker="o", ms=3,
            label=f"{best} above the cost-based threshold")
    ax.plot(t.days_to_failure, 100 * t.share_crit_any, color=INK, lw=1.5, ls="--", label="Any critical counter > 0")
    ax.axhline(100 * t.healthy_share_flagged.iloc[0], color=MODEL_COLOR[best], lw=1, ls=":",
               label=f"Healthy drive-days flagged ({100 * t.healthy_share_flagged.iloc[0]:.2f}%)")
    ax.axhline(100 * t.healthy_share_crit_any.iloc[0], color=INK, lw=1, ls=":",
               label=f"Healthy drive-days with a critical counter > 0 ({100 * t.healthy_share_crit_any.iloc[0]:.1f}%)")
    ax.invert_xaxis()
    ax.set_xlabel("Days before the recorded failure")
    ax.set_ylabel("Share of failing drives' daily rows (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Signals before failure, test month (unseen drives)", loc="left")
    ax.legend(fontsize=8, loc="upper left")
    ax = axes[1]
    lt = lead[lead.model == best]
    caught = lt[lt.caught]
    hi = int(caught.lead_days.max()) + 2 if len(caught) else 31
    ax.hist(caught.lead_days, bins=np.arange(0, max(hi, 31), 2), color=MODEL_COLOR[best], alpha=0.85)
    ax.set_xlabel("Days from the first alert to the failure")
    ax.set_ylabel("Failing drives")
    med = caught.lead_days.median() if len(caught) else float("nan")
    ax.set_title(f"{len(caught)} of {len(lt)} failing test drives alerted; median lead {med:.0f} days", loc="left")
    _suptitle(fig, title)
    plotting.save(fig, path)
    return path


TITLES: dict[str, str] = {}  # filled from the results by make_all


def make_all(results: Path, figures: Path) -> list[Path]:
    """Draw every figure from the CSVs in ``results`` into ``figures``."""
    plotting.apply_style()
    r = {p.stem: pd.read_csv(p) for p in results.glob("*.csv")}
    meta_path = results / "run_metadata.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else None
    npz = dict(np.load(results / "shap_sample.npz", allow_pickle=False))
    best = r["shap_importance"].model.iloc[0]
    ratio = 10.0 if meta is None else float(meta["settings"]["main_cost_ratio"])
    out = []
    if meta is not None:
        out.append(fig_data_window(r["daily_fleet"], meta, figures / "00_data_window.png",
                                   "Time-based protocol: features need history, labels need the future"))
    chk = r["afr_crosscheck_q4_2025"]
    out.append(fig_afr_check(chk, figures / "01_afr_crosscheck.png",
                             f"Recomputed Q4 2025 failure rates match Backblaze's published table "
                             f"({int(chk.exact_match.sum())} of {len(chk)} models exactly)"))
    out.append(fig_survival(r["km_curves"], r["afr_by_age"], r["km_summary"], figures / "02_survival.png",
                            "Failure rates climb after five years of power-on time and differ several-fold between "
                            "models of the same age"))
    res = r["model_results"]
    rr = res.set_index("model")
    gb = max(["LightGBM", "XGBoost", "CatBoost"], key=lambda m: rr.ap[m])
    lr = rr.ap["Logistic regression"]
    verdict = "do not beat" if lr >= rr.ap[gb] else "edge out"
    out.append(fig_models(r["pr_curves"], res, r["cost_curves"], r["cost_by_ratio"], r["fixed_rule"], ratio,
                          figures / "03_model_comparison.png",
                          f"Learned models rank failing drive-days ~{min(lr, rr.ap[gb]) / rr.ap[RULE]:.0f}x better than "
                          f"the SMART rule; boosted trees {verdict} logistic regression (AP {rr.ap[gb]:.3f} vs {lr:.3f})"))
    p = r["split_protocols"]
    out.append(fig_protocols(p, figures / "04_split_protocols.png",
                             f"A random row split reports AP {p.ap.iloc[0]:.3f}; unseen drives in a later month "
                             f"give {p.ap.iloc[-1]:.3f}"))
    c = r["cost_by_ratio"]
    cb = c[(c.model == best) & (c.cost_ratio == ratio)].iloc[0]
    out.append(fig_cost(r["cost_curves"], c, best, ratio, figures / "05_cost_threshold.png",
                        f"At a {ratio:g}:1 cost ratio the validation threshold saves {100 * cb.test_saving_share:.0f}% "
                        f"of failure costs in the test month"))
    out.append(fig_shap(npz, r["shap_importance"], figures / "06_shap.png",
                        f"{best}: recent growth in error counters separates failing drives; usage and model set "
                        f"the baseline"))
    lt = r["lead_times"][r["lead_times"].model == best]
    out.append(fig_lead(r["signal_timeline"], r["lead_times"], best, figures / "07_lead_time.png",
                        f"Alerts build up in the last week, and {100 * (1 - lt.caught.mean()):.0f}% of failing drives "
                        f"are never flagged"))
    return out
