"""Run every stage end to end.

1. Download the selected days from Backblaze's quarterly zips (HTTP range requests) into per-day Parquet files.
2. PySpark ETL (local[2]): failure-rate aggregates, one record per drive, and rolling SMART features with
   "fails within 30 days" labels for a drive-level sample (cached under a key of every ETL setting).
3. Reliability: AFR by model with exact Poisson CIs, the check against Backblaze's published Q4 2025 table,
   Kaplan-Meier survival by power-on age with delayed entry, AFR by age.
4. Models (cached): SMART-rule and logistic baselines, LightGBM / XGBoost / CatBoost tuned with Optuna on a
   drive-disjoint validation group, the split-protocol (leakage) comparison and SHAP values.
5. Evaluation on unseen drives in a later month: row-level metrics with drive-clustered bootstrap CIs, drive-level
   alerting with a cost-based threshold chosen on validation, lead times. Tables -> reports/results, figures ->
   reports/figures, runs -> MLflow (./mlruns).

    python scripts/run_pipeline.py            # full run
    python scripts/run_pipeline.py --quick    # smoke test on 20 days with tiny budgets; outputs go to data/cache/smoke/
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "2")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import pickle  # noqa: E402
import platform  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from dataclasses import asdict, dataclass  # noqa: E402
from datetime import date, datetime, timezone  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Callable  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.compute as pc  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from drivefail import evaluation as ev  # noqa: E402
from drivefail import reliability as rel  # noqa: E402
from drivefail.download import BRONZE_VERSION, bronze_dir, date_range, download_days  # noqa: E402
from drivefail.features import FEATURES, manufacturer  # noqa: E402

CACHE = ROOT / "data" / "cache"
RAW = ROOT / "data" / "raw"
PUBLISHED = ROOT / "data" / "reference" / "backblaze_q4_2025_quarterly.csv"
VERSION = "v1"  # bump to invalidate every cached ETL and model output
RULE, LR = "SMART rule (critical-counter sum)", "Logistic regression"
GBDTS = ("LightGBM", "XGBoost", "CatBoost")


@dataclass(frozen=True)
class Settings:
    """Data window, label, splits, sampling, tuning and evaluation settings of one run."""

    start: date = date(2025, 10, 1)
    end: date = date(2026, 2, 28)
    horizon: int = 30
    train_start: date = date(2025, 10, 31)
    train_end: date = date(2025, 11, 30)
    test_start: date = date(2025, 12, 31)
    test_end: date = date(2026, 1, 30)
    drive_rate: float = 0.15
    group_cuts: tuple[int, int] = (55, 70)
    train_neg_rate: float = 0.12
    es_neg_rate: float = 0.1
    trials: tuple[tuple[str, int], ...] = (("LightGBM", 25), ("XGBoost", 15), ("CatBoost", 8))
    tune_timeout_s: float = 150.0
    seed: int = 0
    cost_ratios: tuple[float, ...] = (2, 3, 5, 10, 20, 50)
    main_cost_ratio: float = 10
    n_boot: int = 400
    n_boot_drive: int = 1000
    shap_rows: int = 3000
    afr_periods: tuple[tuple[str, date, date], ...] = (
        ("Q4 2025", date(2025, 10, 1), date(2025, 12, 31)),
        ("Jan-Feb 2026", date(2026, 1, 1), date(2026, 2, 28)),
        ("Oct 2025-Feb 2026", date(2025, 10, 1), date(2026, 2, 28)))
    km_models: int = 8
    km_min_at_risk: int = 1000
    workers: int = 4
    spark_memory: str = "4g"
    shuffle_partitions: int = 64


QUICK = Settings(end=date(2025, 10, 20), horizon=5, train_start=date(2025, 10, 4), train_end=date(2025, 10, 9),
                 test_start=date(2025, 10, 14), test_end=date(2025, 10, 16),
                 trials=(("LightGBM", 2), ("XGBoost", 2), ("CatBoost", 2)), tune_timeout_s=60, n_boot=30,
                 n_boot_drive=50, shap_rows=400, km_models=4, km_min_at_risk=100, shuffle_partitions=8,
                 afr_periods=(("Q4 2025", date(2025, 10, 1), date(2025, 10, 20)),))


def log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


class Timer:
    """Wall-clock seconds per stage."""

    def __init__(self) -> None:
        self.stages: dict[str, float] = {}

    def run(self, name: str, fn: Callable):
        t0 = time.perf_counter()
        out = fn()
        self.stages[name] = self.stages.get(name, 0.0) + time.perf_counter() - t0
        return out


def key_of(spec: dict) -> str:
    """Short hash of everything that changes an output."""
    return hashlib.sha1(json.dumps({**spec, "version": VERSION}, sort_keys=True, default=str).encode()).hexdigest()[:12]


# ----------------------------------------------------------------------------------------------------------- data
def bronze_quality(files: list[Path], manifest: dict) -> pd.DataFrame:
    """Per-day checks on the bronze files: rows, duplicate serial-days, missing serials, failures, CRC verified."""
    pa.set_cpu_count(2)
    out = []
    for f in files:
        t = pq.read_table(f, columns=["serial_number", "failure"])
        out.append({"date": f.stem, "rows": t.num_rows, "failures": int(pc.sum(t["failure"]).as_py() or 0),
                    "duplicate_serial_days": t.num_rows - len(pc.unique(t["serial_number"])),
                    "missing_serials": t["serial_number"].null_count,
                    "crc32_verified": f.stem in manifest,
                    "compressed_mb": round(manifest.get(f.stem, {}).get("compressed_size", 0) / 1e6, 2)})
    return pd.DataFrame(out)


def run_etl(S: Settings, files: list[Path], dedupe: bool, out: Path, models: tuple[str, ...]) -> dict[str, float]:
    """Spark stage; writes counts, drives, the drive sample and the feature table under ``out``."""
    from drivefail.spark_etl import (EtlConfig, daily_model_counts, drive_sampling, drive_table, feature_table,
                                     load_bronze, spark_session, write_parquet_via_arrow)

    cfg = EtlConfig(start=S.start, end=S.end, horizon=S.horizon, train_start=S.train_start, train_end=S.train_end,
                    test_start=S.test_start, test_end=S.test_end, models=models, drive_rate=S.drive_rate,
                    group_cuts=S.group_cuts)
    t: dict[str, float] = {}
    t0 = time.perf_counter()
    spark = spark_session(CACHE / "spark-tmp", memory=S.spark_memory, shuffle_partitions=S.shuffle_partitions)
    t["spark_start"] = time.perf_counter() - t0
    try:
        df = load_bronze(spark, files, dedupe=dedupe)
        t0 = time.perf_counter()
        counts = daily_model_counts(df)
        counts.to_parquet(out / "daily_model_counts.parquet")
        t["daily_model_counts"] = time.perf_counter() - t0
        log(f"spark: daily (date, model) counts {len(counts):,} rows in {t['daily_model_counts']:.0f}s")
        t0 = time.perf_counter()
        drives = drive_table(df, models)
        drives.to_parquet(out / "drives.parquet")
        t["drive_table"] = time.perf_counter() - t0
        log(f"spark: drive table {len(drives):,} drives in {t['drive_table']:.0f}s")
        sampled = drive_sampling(drives, cfg)
        sampled.to_parquet(out / "drive_sample.parquet")
        keep = sampled[sampled.keep][["serial_number", "group", "fail_day", "last_day", "weight"]].copy()
        keep["fail_day"] = keep.fail_day.astype("Int32")  # nullable: drives without a failure
        keep["last_day"] = keep.last_day.astype("int32")
        keep_sdf = spark.createDataFrame(pa.Table.from_pandas(keep, preserve_index=False))
        t0 = time.perf_counter()
        n = write_parquet_via_arrow(feature_table(df, keep_sdf, cfg), out / "features")
        t["feature_table"] = time.perf_counter() - t0
        log(f"spark: feature table {n:,} rows for {len(keep):,} sampled drives in {t['feature_table']:.0f}s")
    finally:
        spark.stop()
    (out / "_SUCCESS").write_text(json.dumps(t, indent=1))
    return t


# --------------------------------------------------------------------------------------------------- reliability
def reliability_tables(S: Settings, counts: pd.DataFrame, drives: pd.DataFrame, models: tuple[str, ...]
                       ) -> dict[str, pd.DataFrame]:
    """AFR tables, the published-table check, Kaplan-Meier curves and AFR by age."""
    afr = rel.afr_table(counts, drives, list(S.afr_periods))
    afr["manufacturer"] = afr.model.map(manufacturer)
    published = rel.load_published(PUBLISHED)
    q4 = afr[afr.period == S.afr_periods[0][0]]
    check = rel.crosscheck(q4, published)
    q4_all = q4[q4.model.isin(published.model)]
    totals = pd.DataFrame([{
        "quantity": k, "published": p, "recomputed": r} for k, p, r in [
        ("models", len(published), int(check.drive_days_recomputed.gt(0).sum())),
        ("drive_count", int(published.drive_count.sum()), int(q4_all.drive_count.sum())),
        ("drive_days", int(published.drive_days.sum()), int(q4_all.drive_days.sum())),
        ("failures", int(published.failures.sum()), int(q4_all.failures.sum())),
        ("afr_pct", round(100 * published.failures.sum() / (published.drive_days.sum() / 365), 2),
         round(100 * q4_all.failures.sum() / (q4_all.drive_days.sum() / 365), 2))]])

    # A model whose recomputed mean age is more than a year off the published one has unusable power-on hours
    # (16-bit wrap-around of SMART 9), so it is left out of the age-based analyses.
    age_gap = (check.avg_age_months_recomputed - check.avg_age_months_published).abs()
    bad_age = tuple(check.model[age_gap > 12])
    surv, invalid = rel.survival_frame(drives[~drives.model.isin(bad_age)])
    surv["manufacturer"] = surv.model.map(manufacturer)
    top = drives.model.value_counts().head(S.km_models).index.tolist()
    grid = np.round(np.arange(0, 12.01, 0.05), 2)
    curves, summary = [], []
    whole = afr[afr.period == S.afr_periods[-1][0]].set_index("model")
    for m in top:
        g = surv[surv.model == m]
        c = rel.kaplan_meier(g, grid, min_at_risk=S.km_min_at_risk)
        c.insert(0, "model", m)
        curves.append(c)
        med = float(g.entry.median())
        summary.append({"model": m, "manufacturer": manufacturer(m), "drives": len(g), "failures": int(g.event.sum()),
                        "entry_age_min": g.entry.min(), "entry_age_median": med, "entry_age_max": g.entry.max(),
                        "km_1yr_failure_pct_from_median_age": 100 * rel.km_one_year_failure(g, med),
                        "afr_pct_window": whole.afr_pct.get(m, np.nan),
                        "afr_ci_low": whole.afr_ci_low.get(m, np.nan), "afr_ci_high": whole.afr_ci_high.get(m, np.nan)})
    edges = np.arange(0, 11.0, 1.0)
    by_age = pd.concat([rel.afr_by_age(surv, edges), rel.afr_by_age(surv, edges, by="manufacturer")], ignore_index=True)
    validity = pd.DataFrame([{"drives": len(drives), "models_excluded_age_mismatch": ";".join(bad_age),
                              "drives_excluded_age_mismatch": int(drives.model.isin(bad_age).sum()),
                              "valid_power_on_hours": len(surv), "dropped_invalid_hours": len(invalid),
                              "dropped_failures": int(invalid.failed.sum()) if len(invalid) else 0}])
    return {"afr_by_model": afr, "afr_crosscheck_q4_2025": check, "afr_crosscheck_totals": totals,
            "km_curves": pd.concat(curves, ignore_index=True), "km_summary": pd.DataFrame(summary),
            "afr_by_age": by_age, "survival_data_check": validity}


# -------------------------------------------------------------------------------------------------------- models
def split_frames(S: Settings, feats: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Training sample (group A, train month, negatives subsampled), validation (B) and test (C, later month)."""
    f = feats[~feats.censored].sort_values(["serial_number", "date"]).reset_index(drop=True)
    rng = np.random.default_rng(S.seed)
    f["sub"] = (f.label == 1) | (rng.random(len(f)) < S.train_neg_rate)
    f["sub_weight"] = np.where(f.label == 1, f.weight, f.weight / S.train_neg_rate)
    return {"train": f[(f.group == "A") & (f.period == "train") & f["sub"]].reset_index(drop=True),
            "val": f[(f.group == "B") & (f.period == "train")].reset_index(drop=True),
            "test": f[(f.group == "C") & (f.period == "test")].reset_index(drop=True),
            "all": f}


def run_models(S: Settings, frames: dict[str, pd.DataFrame], n_models: int, out: Path) -> dict:
    """Fit every model, score validation and test rows, run the split-protocol comparison and SHAP; cached."""
    from drivefail import models as M

    if (out / "_SUCCESS").exists():
        return pickle.loads((out / "results.pkl").read_bytes())
    out.mkdir(parents=True, exist_ok=True)
    tr, va, te = frames["train"], frames["val"], frames["test"]
    Xtr, Xva, Xte = (M.matrix(d, n_models) for d in (tr, va, te))
    ytr, yva = tr.label.to_numpy(), va.label.to_numpy()
    wva = va.weight.to_numpy()
    log(f"models: train {len(tr):,} rows ({int(ytr.sum()):,} positive), val {len(va):,} ({int(yva.sum()):,}), "
        f"test {len(te):,} ({int(te.label.sum()):,})")
    r0 = np.random.default_rng(S.seed + 4)
    es_mask = (yva == 1) | (r0.random(len(yva)) < S.es_neg_rate)
    es = M.EarlyStopSet(Xva[es_mask], yva[es_mask], np.where(yva[es_mask] == 1, wva[es_mask], wva[es_mask] / S.es_neg_rate))
    tuned = {LR: M.fit_logistic(Xtr, ytr, Xva, yva, wva)}
    log(f"models: logistic regression val AP {tuned[LR].val_ap:.4f} ({tuned[LR].seconds:.0f}s)")
    tuners = {"LightGBM": M.tune_lightgbm, "XGBoost": M.tune_xgboost, "CatBoost": M.tune_catboost}
    for fam, n in S.trials:
        tuned[fam] = tuners[fam](Xtr, ytr, Xva, yva, wva, es, n_trials=n, seed=S.seed, timeout=S.tune_timeout_s)
        log(f"models: {fam} {len(tuned[fam].trials)} trials, val AP {tuned[fam].val_ap:.4f}, "
            f"{tuned[fam].best_iteration} rounds ({tuned[fam].seconds:.0f}s)")
    scores = {"val": {RULE: va.crit_sum.to_numpy()}, "test": {RULE: te.crit_sum.to_numpy()}}
    for name, t in tuned.items():
        scores["val"][name] = M.predict(t, Xva)
        scores["test"][name] = M.predict(t, Xte)

    # Split-protocol comparison with the best LightGBM settings and a fixed number of rounds.
    lgbm = tuned["LightGBM"]
    f = frames["all"]
    pool = f[(f.period == "train") & f["sub"] & f.group.isin(["A", "C"])].reset_index(drop=True)
    rng = np.random.default_rng(S.seed + 1)
    rand_test = rng.random(len(pool)) < (100 - S.group_cuts[1]) / (100 - S.group_cuts[1] + S.group_cuts[0])
    protocols = []

    def evaluate(name: str, train_df: pd.DataFrame, test_df: pd.DataFrame, weight_col: str) -> None:
        booster = M.fit_lightgbm_fixed(lgbm.params, max(lgbm.best_iteration, 1), M.matrix(train_df, n_models),
                                       train_df.label.to_numpy())
        s = booster.predict(M.matrix(test_df, n_models))
        protocols.append({"protocol": name, "train_rows": len(train_df), "train_drives": train_df.serial_number.nunique(),
                          "test_rows": len(test_df), "test_positive_rows": int(test_df.label.sum()),
                          "test_failing_drives": test_df.loc[test_df.label == 1, "serial_number"].nunique(),
                          "y": test_df.label.to_numpy(), "s": s, "w": test_df[weight_col].to_numpy(),
                          "drive": test_df.serial_number.to_numpy()})

    evaluate("Random row split (same month)", pool[~rand_test], pool[rand_test], "sub_weight")
    a_train = pool[pool.group == "A"]
    evaluate("Drive-grouped split (same month)", a_train, f[(f.group == "C") & (f.period == "train")], "weight")
    evaluate("Same drives, later month", a_train, f[(f.group == "A") & (f.period == "test")], "weight")
    evaluate("Drive-grouped + later month (deployment)", a_train, te, "weight")

    # SHAP values of the best gradient-boosted model on a sample of test rows.
    best = max(GBDTS, key=lambda k: tuned[k].val_ap)
    pos_idx = np.flatnonzero(te.label.to_numpy() == 1)
    neg_idx = np.flatnonzero(te.label.to_numpy() == 0)
    r2 = np.random.default_rng(S.seed + 2)
    idx = np.r_[r2.choice(pos_idx, min(len(pos_idx), S.shap_rows // 2), replace=False),
                r2.choice(neg_idx, min(len(neg_idx), S.shap_rows // 2), replace=False)]
    sv, base = M.shap_values(tuned[best], Xte.iloc[idx])
    shap_check = {}
    try:  # the shap library and LightGBM's own TreeSHAP should agree
        import shap
        lg_sv, _ = M.shap_values(lgbm, Xte.iloc[idx[:200]])
        lib = np.asarray(shap.TreeExplainer(lgbm.model).shap_values(Xte.iloc[idx[:200]]))
        lib = lib[1] if lib.ndim == 3 else lib
        shap_check = {"rows": 200, "max_abs_diff_vs_shap_library": float(np.abs(lib - lg_sv).max())}
    except Exception as e:  # pragma: no cover - diagnostic only
        shap_check = {"error": str(e)[:200]}

    res = {"tuned": {k: {"family": t.family, "params": t.params, "val_ap": t.val_ap, "best_iteration": t.best_iteration,
                         "seconds": t.seconds, "trials": t.trials} for k, t in tuned.items()},
           "scores": scores, "protocols": protocols, "best": best,
           "shap": {"values": sv, "base": base, "rows": idx, "features": FEATURES, "check": shap_check}}
    (out / "results.pkl").write_bytes(pickle.dumps(res))
    (out / "_SUCCESS").write_text("ok")
    return res


# ---------------------------------------------------------------------------------------------------- evaluation
def evaluation_tables(S: Settings, frames: dict[str, pd.DataFrame], res: dict) -> dict[str, pd.DataFrame]:
    """Row-level test metrics with CIs, PR curves, split protocols, drive-level cost analysis and lead times."""
    va, te = frames["val"], frames["test"]
    names = [RULE, LR, *GBDTS]
    y, w = te.label.to_numpy(), te.weight.to_numpy()
    codes = pd.factorize(te.serial_number)[0]
    test_scores = {k: res["scores"]["test"][k] for k in names}
    point = pd.DataFrame([{"model": k, **ev.row_metrics(y, s, w)} for k, s in test_scores.items()])
    point["val_ap"] = [ev.row_metrics(va.label.to_numpy(), res["scores"]["val"][k], va.weight.to_numpy())["ap"]
                       for k in names]
    vboot = ev.clustered_bootstrap(va.label.to_numpy(), va.weight.to_numpy(), pd.factorize(va.serial_number)[0],
                                   {k: res["scores"]["val"][k] for k in names}, max(S.n_boot // 2, 20), seed=S.seed + 5)
    vci = vboot.groupby("model").ap.quantile([0.025, 0.975]).unstack()
    point["val_ap_ci_low"] = point.model.map(vci[0.025])
    point["val_ap_ci_high"] = point.model.map(vci[0.975])
    boot = ev.clustered_bootstrap(y, w, codes, test_scores, S.n_boot, seed=S.seed)
    summ = ev.summarize_bootstrap(point, boot, baseline=LR)
    sig = summ.dropna(subset=["p_value"]).copy()
    sig["p_holm"] = ev.holm(sig.p_value.to_numpy())
    wide = summ.pivot_table(index="model", columns="metric", values=["value", "ci_low", "ci_high"], sort=False)
    wide.columns = [f"{m}" if s == "value" else f"{m}_{s}" for s, m in wide.columns]
    results = point.merge(wide[[c for c in wide.columns if c.endswith(("_ci_low", "_ci_high"))]].reset_index(), on="model")

    pr = pd.concat([ev.curve_points(y, s, w).assign(model=k) for k, s in test_scores.items()], ignore_index=True)

    prot = []
    pboots = []
    for i, p_ in enumerate(res["protocols"]):
        m = ev.row_metrics(p_["y"], p_["s"], p_["w"])
        b = ev.clustered_bootstrap(p_["y"], p_["w"], pd.factorize(p_["drive"])[0], {"x": p_["s"]}, max(S.n_boot // 2, 20),
                                   seed=S.seed + i)
        pboots.append(b.assign(protocol=p_["protocol"]))
        prot.append({k: v for k, v in p_.items() if k not in ("y", "s", "w", "drive")} | m |
                    {"ap_ci_low": b.ap.quantile(0.025), "ap_ci_high": b.ap.quantile(0.975),
                     "roc_auc_ci_low": b.roc_auc.quantile(0.025), "roc_auc_ci_high": b.roc_auc.quantile(0.975)})
    protocols = pd.DataFrame(prot)

    # Drive-level alerting: flag a drive when its daily score reaches the threshold at least once in the month.
    va_s, te_s = va.copy(), te.copy()
    for k in names:
        va_s[k] = res["scores"]["val"][k]
        te_s[k] = res["scores"]["test"][k]
    drive_rows, cost_rows, curve_rows, lead_rows = [], [], [], []
    rng = np.random.default_rng(S.seed + 3)
    for k in names:
        dv, dt = ev.drive_level(va_s, k), ev.drive_level(te_s, k)
        mult = rng.poisson(1.0, size=(S.n_boot_drive, len(dt)))
        for ratio in S.cost_ratios:
            tau = ev.best_threshold(dv, ratio)
            o_val = ev.alert_outcomes(dv, np.array([tau]), ratio).iloc[0]
            o_test = ev.alert_outcomes(dt, np.array([tau]), ratio).iloc[0]
            oracle = ev.alert_outcomes(dt, ev.threshold_grid(dt.max_score.to_numpy()), ratio)
            flag = (dt.max_score.to_numpy() >= tau)
            pos = dt.positive.to_numpy() == 1
            wb = mult * dt.weight.to_numpy()
            tp_b = (wb * (flag & pos)).sum(1)
            fp_b = (wb * (flag & ~pos)).sum(1)
            sav_b = 1e4 * ((ratio - 1) * tp_b - fp_b) / wb.sum(1)
            rec_b = tp_b / np.maximum((wb * pos).sum(1), 1e-12)
            cost_rows.append({"model": k, "cost_ratio": ratio, "threshold": tau,
                              "val_net_saving_per_10k": o_val.net_saving_per_10k,
                              "test_net_saving_per_10k": o_test.net_saving_per_10k,
                              "test_saving_ci_low": np.quantile(sav_b, 0.025), "test_saving_ci_high": np.quantile(sav_b, 0.975),
                              "test_saving_share": o_test.saving_share, "test_recall": o_test.recall,
                              "test_recall_ci_low": np.quantile(rec_b, 0.025), "test_recall_ci_high": np.quantile(rec_b, 0.975),
                              "test_precision": o_test.precision, "test_flagged_per_10k": o_test.flagged_per_10k,
                              "test_tp": o_test.tp, "test_fp": o_test.fp, "test_fn": o_test.fn,
                              "oracle_test_net_saving_per_10k": oracle.net_saving_per_10k.max()})
            if ratio == S.main_cost_ratio:
                lt = ev.lead_times(te_s, k, tau)
                lead_rows.append(lt.assign(model=k, threshold=tau))
        # fixed rule: any critical counter above zero on any day of the month
        if k == RULE:
            for ratio in S.cost_ratios:
                o = ev.alert_outcomes(dt, np.array([0.5]), ratio).iloc[0]
                drive_rows.append({"model": "SMART rule: any critical counter > 0", "cost_ratio": ratio,
                                   "test_net_saving_per_10k": o.net_saving_per_10k, "test_recall": o.recall,
                                   "test_precision": o.precision, "test_flagged_per_10k": o.flagged_per_10k})
        grid = ev.threshold_grid(np.r_[dv.max_score.to_numpy(), dt.max_score.to_numpy()], n=300)
        for split, d in (("val", dv), ("test", dt)):
            o = ev.alert_outcomes(d, grid, S.main_cost_ratio)
            curve_rows.append(o.assign(model=k, split=split, drives=len(d), failing=int(d.positive.sum())))
    cost = pd.DataFrame(cost_rows)
    fixed_rule = pd.DataFrame(drive_rows)
    lead = pd.concat(lead_rows, ignore_index=True)

    # How early do signals appear? Share of failing drives' rows that are flagged, by days before failure.
    best = res["best"]
    tau_best = cost[(cost.model == best) & (cost.cost_ratio == S.main_cost_ratio)].threshold.iloc[0]
    pos = te_s[te_s.label == 1]
    timeline = (pos.assign(model_flag=pos[best] >= tau_best, crit_any=pos.crit_sum > 0)
                .groupby("days_to_failure").agg(rows=("label", "size"), drives=("serial_number", "nunique"),
                                                 share_flagged=("model_flag", "mean"), share_crit_any=("crit_any", "mean"))
                .reset_index())
    neg = te_s[te_s.label == 0]
    timeline["healthy_share_flagged"] = np.average(neg[best] >= tau_best, weights=neg.weight)
    timeline["healthy_share_crit_any"] = np.average(neg.crit_sum > 0, weights=neg.weight)

    # Which failing test drives were caught, by manufacturer and by whether a critical counter was ever above zero.
    fail = te_s[te_s.serial_number.isin(te_s.loc[te_s.label == 1, "serial_number"])]
    flags = fail.assign(crit_pos=fail.crit_sum > 0)
    sig_rows = []
    for k in names:
        tau = cost[(cost.model == k) & (cost.cost_ratio == S.main_cost_ratio)].threshold.iloc[0]
        g = flags.groupby("serial_number").agg(model=("model", "first"), crit_any=("crit_pos", "max"),
                                               max_score=(k, "max"))
        g["caught"] = g.max_score >= tau
        g["manufacturer"] = g.model.map(manufacturer)
        for col in ("manufacturer", "crit_any"):
            t = g.groupby(col).agg(failing_drives=("caught", "size"), caught=("caught", "sum")).reset_index()
            sig_rows.append(t.rename(columns={col: "level"}).astype({"level": str}).assign(model=k, by=col))
    failure_signals = pd.concat(sig_rows, ignore_index=True)
    failure_signals["recall"] = failure_signals.caught / failure_signals.failing_drives

    drive_summary = []
    for split, d_ in (("val", va_s), ("test", te_s)):
        dd = ev.drive_level(d_, RULE)
        drive_summary.append({"split": split, "drives_sampled": len(dd), "failing_drives": int(dd.positive.sum()),
                              "fleet_estimate": float(dd.weight.sum())})
    return {"model_results": results, "model_significance": sig, "pr_curves": pr, "split_protocols": protocols,
            "cost_by_ratio": cost, "fixed_rule": fixed_rule, "cost_curves": pd.concat(curve_rows, ignore_index=True),
            "lead_times": lead, "signal_timeline": timeline, "drive_level_sizes": pd.DataFrame(drive_summary),
            "failure_signals": failure_signals}


def label_stats(feats: pd.DataFrame, sample: pd.DataFrame, counts: pd.DataFrame, S: Settings,
                models: tuple[str, ...]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rows, positives and drives per group and period, and a check of the weighted sample against exact counts."""
    g = feats.groupby(["period", "group"])
    stats = pd.DataFrame({
        "rows": g.size(), "weighted_rows": g.weight.sum(), "positive_rows": g.label.sum(),
        "censored_rows": g.censored.sum(), "drives": g.serial_number.nunique(),
        "failing_drives": feats[feats.label == 1].groupby(["period", "group"]).serial_number.nunique()}).fillna(0)
    stats["positive_rate_weighted"] = (feats.label * feats.weight).groupby([feats.period, feats.group]).sum() / stats.weighted_rows
    c = counts[counts.model.isin(models)].copy()
    c["date"] = pd.to_datetime(c.date).dt.date
    checks = []
    for period, (a, b) in (("train", (S.train_start, S.train_end)), ("test", (S.test_start, S.test_end))):
        exact = int(c[(c.date >= a) & (c.date <= b)].drives.sum())
        est = float(feats[feats.period == period].weight.sum())
        checks.append({"period": period, "exact_drive_days": exact, "weighted_sample_estimate": round(est),
                       "relative_error_pct": 100 * (est - exact) / exact})
    shares = sample.groupby("group").agg(drives=("serial_number", "size"), kept=("keep", "sum"),
                                         failing_in_window=("fails_in_window", "sum")).reset_index()
    shares["share_of_drives"] = shares.drives / shares.drives.sum()
    return stats.reset_index(), pd.concat([pd.DataFrame(checks), shares], ignore_index=True)


# ---------------------------------------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="smoke test on 20 days with tiny budgets")
    args = ap.parse_args()
    S = QUICK if args.quick else Settings()
    results_dir = CACHE / "smoke" / "results" if args.quick else ROOT / "reports" / "results"
    figures_dir = CACHE / "smoke" / "figures" if args.quick else ROOT / "reports" / "figures"
    work = CACHE / "smoke" if args.quick else CACHE
    for d in (results_dir, figures_dir, work):
        d.mkdir(parents=True, exist_ok=True)
    T = Timer()
    started = datetime.now(timezone.utc)

    days = date_range(S.start, S.end)
    manifest = T.run("download", lambda: download_days(days, RAW, workers=S.workers, log=log))
    files = [bronze_dir(RAW) / f"{d.isoformat()}.parquet" for d in days]
    quality = T.run("bronze_checks", lambda: bronze_quality(files, manifest))
    quality.to_csv(results_dir / "bronze_daily_checks.csv", index=False)
    dups = int(quality.duplicate_serial_days.sum())
    log(f"bronze: {len(files)} days, {quality.rows.sum():,} rows, {dups} duplicate serial-days, "
        f"{int(quality.missing_serials.sum())} missing serials, CRC verified {int(quality.crc32_verified.sum())}")

    published = rel.load_published(PUBLISHED)
    models = tuple(sorted(published.model))
    etl_spec = {k: v for k, v in asdict(S).items() if k in ("start", "end", "horizon", "train_start", "train_end",
                                                             "test_start", "test_end", "drive_rate", "group_cuts")}
    etl_key = key_of({**etl_spec, "bronze": BRONZE_VERSION, "features": FEATURES, "models": models, "dedupe": dups > 0})
    etl_dir = work / f"etl-{etl_key}"
    if (etl_dir / "_SUCCESS").exists():
        etl_times = json.loads((etl_dir / "_SUCCESS").read_text())
        log(f"etl: cached ({etl_dir.name})")
    else:
        etl_dir.mkdir(parents=True, exist_ok=True)
        etl_times = T.run("spark_etl", lambda: run_etl(S, files, dups > 0, etl_dir, models))
    counts = pd.read_parquet(etl_dir / "daily_model_counts.parquet")
    drives = pd.read_parquet(etl_dir / "drives.parquet")
    sample = pd.read_parquet(etl_dir / "drive_sample.parquet")
    feats = pd.read_parquet(etl_dir / "features")
    log(f"features: {len(feats):,} rows, {feats.serial_number.nunique():,} drives, {int(feats.label.sum()):,} positive")

    tables = T.run("reliability", lambda: reliability_tables(S, counts, drives, models))
    daily = counts[counts.model.isin(models)].groupby("date").agg(drives=("drives", "sum"), failures=("failures", "sum"))
    tables["daily_fleet"] = daily.reset_index()
    stats, sampling_check = label_stats(feats, sample, counts, S, models)
    tables["label_stats"], tables["sampling_check"] = stats, sampling_check
    tables["data_quality"] = pd.DataFrame([{
        "days": len(quality), "first_day": quality.date.min(), "last_day": quality.date.max(),
        "rows": int(quality.rows.sum()), "failures_all_models": int(quality.failures.sum()),
        "duplicate_serial_days": dups, "missing_serials": int(quality.missing_serials.sum()),
        "days_crc32_verified": int(quality.crc32_verified.sum()), "compressed_gb": round(quality.compressed_mb.sum() / 1e3, 3),
        "modelled_models": len(models), "modelled_drives": len(drives)}])

    frames = split_frames(S, feats)
    model_spec = {"etl": etl_key, "trials": S.trials, "train_neg_rate": S.train_neg_rate, "seed": S.seed,
                  "es_neg_rate": S.es_neg_rate,
                  "timeout": S.tune_timeout_s, "shap_rows": S.shap_rows}
    model_dir = work / f"models-{key_of(model_spec)}"
    cached_models = (model_dir / "_SUCCESS").exists()
    res = T.run("models" if not cached_models else "models_cached", lambda: run_models(S, frames, len(models), model_dir))
    if cached_models:
        log(f"models: cached ({model_dir.name})")
    tables.update(T.run("evaluation", lambda: evaluation_tables(S, frames, res)))

    trials = []
    for name, t in res["tuned"].items():
        tr = t["trials"].copy()
        tr.insert(0, "model", name)
        trials.append(tr)
    tables["tuning_trials"] = pd.concat(trials, ignore_index=True)
    tables["tuned_models"] = pd.DataFrame([{"model": k, "trials": len(t["trials"]), "val_ap": t["val_ap"],
                                            "best_iteration": t["best_iteration"], "tuning_seconds": t["seconds"],
                                            "params": json.dumps({p: v for p, v in t["params"].items()
                                                                  if p not in ("verbose", "verbosity")}, default=str)}
                                           for k, t in res["tuned"].items()])
    sv = res["shap"]
    te = frames["test"].iloc[sv["rows"]]
    lab = te.label.to_numpy()
    imp = pd.DataFrame({"feature": sv["features"], "mean_abs_shap": np.abs(sv["values"]).mean(0),
                        "mean_abs_shap_failing": np.abs(sv["values"][lab == 1]).mean(0),
                        "mean_abs_shap_healthy": np.abs(sv["values"][lab == 0]).mean(0)})
    tables["shap_importance"] = imp.sort_values("mean_abs_shap", ascending=False).assign(model=res["best"])
    np.savez_compressed(results_dir / "shap_sample.npz", shap=sv["values"].astype(np.float32),
                        X=te[sv["features"]].to_numpy(np.float32), label=lab.astype(np.int8),
                        features=np.array(sv["features"]), base=np.array([sv["base"]]))

    for name, df in tables.items():
        df.to_csv(results_dir / f"{name}.csv", index=False, float_format="%.10g")

    from drivefail.tracking import init_mlflow, log_result

    init_mlflow(ROOT)
    mr = tables["model_results"].set_index("model")
    for name, t in res["tuned"].items():
        log_result(f"{name}{' (quick)' if args.quick else ''}",
                   {"trials": len(t["trials"]), **{k: v for k, v in t["params"].items() if k not in ("verbose",)}},
                   {"val_ap": t["val_ap"], "test_ap": mr.loc[name, "ap"], "test_roc_auc": mr.loc[name, "roc_auc"],
                    "test_recall_at_fpr_0.1pct": mr.loc[name, "recall_at_fpr_0.1pct"], "tuning_seconds": t["seconds"]})

    from drivefail.figures import make_all

    T.run("figures", lambda: make_all(results_dir, figures_dir))
    meta = {"started_utc": started.isoformat(timespec="seconds"), "quick": args.quick,
            "settings": json.loads(json.dumps(asdict(S), default=str)), "etl_cache": etl_dir.name,
            "model_cache": model_dir.name, "etl_seconds": etl_times, "timings_s": {k: round(v, 1) for k, v in T.stages.items()},
            "best_model": res["best"], "shap_check": sv["check"],
            "versions": {"python": platform.python_version(), **{m: __import__(m).__version__ for m in (
                "pyspark", "lightgbm", "xgboost", "catboost", "optuna", "lifelines", "sklearn", "pandas", "numpy",
                "pyarrow")}},
            "platform": platform.platform(), "cpu_threads": 2, "spark_master": "local[2]"}
    (results_dir / "run_metadata.json").write_text(json.dumps(meta, indent=2, default=str))
    log(f"done: {', '.join(f'{k} {v:.0f}s' for k, v in T.stages.items())}")


if __name__ == "__main__":
    main()
