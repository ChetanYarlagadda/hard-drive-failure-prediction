"""Models: SMART-rule and logistic-regression baselines, and LightGBM / XGBoost / CatBoost tuned with Optuna."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from catboost import CatBoostClassifier, Pool
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from drivefail.evaluation import average_precision
from drivefail.features import CATEGORICAL, COUNTER_FEATURES, FEATURES, MANUFACTURERS, OTHER_FEATURES

THREADS = 2
NUMERIC = COUNTER_FEATURES + OTHER_FEATURES
optuna.logging.set_verbosity(optuna.logging.WARNING)


@dataclass
class EarlyStopSet:
    """Validation subsample (all positives, a share of negatives, re-weighted) scored every boosting round.

    Scoring average precision on the full validation set at every round would dominate the training time, so
    early stopping uses this subsample while trials are compared on the full validation set.
    """

    X: pd.DataFrame
    y: np.ndarray
    w: np.ndarray


@dataclass
class Tuned:
    """Result of one Optuna search: the best fitted model, its parameters and every trial."""

    family: str
    model: Any
    params: dict
    val_ap: float
    best_iteration: int
    trials: pd.DataFrame
    seconds: float
    extra: dict = field(default_factory=dict)


def matrix(df: pd.DataFrame, n_models: int) -> pd.DataFrame:
    """Feature matrix with the drive model and manufacturer as pandas categoricals (fixed category sets)."""
    X = df[FEATURES].copy()
    X["model_code"] = pd.Categorical(X.model_code.astype(int), categories=range(n_models))
    X["mfg_code"] = pd.Categorical(X.mfg_code.astype(int), categories=range(len(MANUFACTURERS)))
    return X


def _signed_log1p(x: np.ndarray) -> np.ndarray:
    return np.sign(x) * np.log1p(np.abs(x))


def fit_logistic(Xtr: pd.DataFrame, ytr: np.ndarray, Xva: pd.DataFrame, yva: np.ndarray, wva: np.ndarray,
                 cs: tuple[float, ...] = (0.01, 0.1, 1.0)) -> Tuned:
    """Logistic regression on signed-log counters with missing-value indicators; C chosen on validation AP."""
    t0 = time.perf_counter()
    rows = []
    best: tuple[float, Any, float] = (-1.0, None, 0.0)
    for c in cs:
        pre = ColumnTransformer([
            ("num", make_pipeline(FunctionTransformer(_signed_log1p), SimpleImputer(strategy="median", add_indicator=True),
                                  StandardScaler()), NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL)])
        pipe = make_pipeline(pre, LogisticRegression(C=c, max_iter=500))
        pipe.fit(Xtr, ytr)
        ap = average_precision(yva, pipe.predict_proba(Xva)[:, 1], wva)
        rows.append({"number": len(rows), "value": ap, "params_C": c})
        if ap > best[0]:
            best = (ap, pipe, c)
    return Tuned("Logistic regression", best[1], {"C": best[2]}, best[0], 0, pd.DataFrame(rows),
                 time.perf_counter() - t0)


def _study(objective: Callable[[optuna.Trial], float], n_trials: int, seed: int, timeout: float | None) -> optuna.Study:
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, timeout=timeout, gc_after_trial=False)
    return study


def tune_lightgbm(Xtr: pd.DataFrame, ytr: np.ndarray, Xva: pd.DataFrame, yva: np.ndarray, wva: np.ndarray,
                  es: EarlyStopSet, n_trials: int, seed: int = 0, max_rounds: int = 600,
                  timeout: float | None = None) -> Tuned:
    """Optuna search over LightGBM hyperparameters; early stopping and selection on weighted validation AP."""
    t0 = time.perf_counter()
    dtr = lgb.Dataset(Xtr, ytr, free_raw_data=False,
                      params={"verbose": -1, "num_threads": THREADS, "feature_pre_filter": False})
    des = lgb.Dataset(es.X, es.y, weight=es.w, reference=dtr, free_raw_data=False)
    best: dict[str, Any] = {"ap": -1.0}

    def objective(trial: optuna.Trial) -> float:
        params = {"objective": "binary", "metric": "average_precision", "verbose": -1, "num_threads": THREADS,
                  "seed": seed, "bagging_freq": 1,
                  "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
                  "num_leaves": trial.suggest_int("num_leaves", 15, 255, log=True),
                  "min_child_samples": trial.suggest_int("min_child_samples", 20, 1000, log=True),
                  "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
                  "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
                  "lambda_l2": trial.suggest_float("lambda_l2", 1e-3, 30.0, log=True),
                  "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 20.0, log=True)}
        booster = lgb.train(params, dtr, num_boost_round=max_rounds, valid_sets=[des],
                            callbacks=[lgb.early_stopping(50, verbose=False)])
        ap = average_precision(yva, booster.predict(Xva, num_iteration=booster.best_iteration), wva)
        trial.set_user_attr("best_iteration", booster.best_iteration)
        if ap > best["ap"]:
            best.update(ap=ap, model=booster, params=params, it=booster.best_iteration)
        return ap

    study = _study(objective, n_trials, seed, timeout)
    return Tuned("LightGBM", best["model"], best["params"], best["ap"], best["it"], study.trials_dataframe(),
                 time.perf_counter() - t0)


def tune_xgboost(Xtr: pd.DataFrame, ytr: np.ndarray, Xva: pd.DataFrame, yva: np.ndarray, wva: np.ndarray,
                  es: EarlyStopSet, n_trials: int, seed: int = 0, max_rounds: int = 600,
                  timeout: float | None = None) -> Tuned:
    """Optuna search over XGBoost (hist) hyperparameters with native categorical support."""
    t0 = time.perf_counter()
    dtr = xgb.DMatrix(Xtr, ytr, enable_categorical=True, nthread=THREADS)
    dva = xgb.DMatrix(Xva, yva, weight=wva, enable_categorical=True, nthread=THREADS)
    des = xgb.DMatrix(es.X, es.y, weight=es.w, enable_categorical=True, nthread=THREADS)
    best: dict[str, Any] = {"ap": -1.0}

    def objective(trial: optuna.Trial) -> float:
        params = {"objective": "binary:logistic", "eval_metric": "aucpr", "tree_method": "hist", "nthread": THREADS,
                  "seed": seed, "max_cat_to_onehot": 1,
                  "eta": trial.suggest_float("eta", 0.02, 0.2, log=True),
                  "max_depth": trial.suggest_int("max_depth", 3, 10),
                  "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 100.0, log=True),
                  "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                  "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
                  "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 30.0, log=True),
                  "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 20.0, log=True)}
        bst = xgb.train(params, dtr, num_boost_round=max_rounds, evals=[(des, "es")], early_stopping_rounds=50,
                        verbose_eval=False)
        ap = average_precision(yva, bst.predict(dva, iteration_range=(0, bst.best_iteration + 1)), wva)
        trial.set_user_attr("best_iteration", bst.best_iteration)
        if ap > best["ap"]:
            best.update(ap=ap, model=bst, params=params, it=bst.best_iteration)
        return ap

    study = _study(objective, n_trials, seed, timeout)
    return Tuned("XGBoost", best["model"], best["params"], best["ap"], best["it"], study.trials_dataframe(),
                 time.perf_counter() - t0)


def catboost_frame(X: pd.DataFrame) -> pd.DataFrame:
    """CatBoost wants categorical columns as integers."""
    Xc = X.copy()
    for c in CATEGORICAL:
        Xc[c] = Xc[c].astype(int)
    return Xc


def tune_catboost(Xtr: pd.DataFrame, ytr: np.ndarray, Xva: pd.DataFrame, yva: np.ndarray, wva: np.ndarray,
                  es: EarlyStopSet, n_trials: int, seed: int = 0, max_rounds: int = 500,
                  timeout: float | None = None) -> Tuned:
    """Optuna search over CatBoost hyperparameters (ordered boosting, native categorical handling)."""
    t0 = time.perf_counter()
    cat_idx = [list(Xtr.columns).index(c) for c in CATEGORICAL]
    ptr = Pool(catboost_frame(Xtr), ytr, cat_features=cat_idx, thread_count=THREADS)
    Xva_c = catboost_frame(Xva)
    pva = Pool(Xva_c, yva, weight=wva, cat_features=cat_idx, thread_count=THREADS)
    pes = Pool(catboost_frame(es.X), es.y, weight=es.w, cat_features=cat_idx, thread_count=THREADS)
    best: dict[str, Any] = {"ap": -1.0}

    def objective(trial: optuna.Trial) -> float:
        params = {"iterations": max_rounds, "eval_metric": "PRAUC", "od_type": "Iter", "od_wait": 50,
                  "thread_count": THREADS, "random_seed": seed, "verbose": False, "allow_writing_files": False,
                  "use_best_model": True,
                  "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.25, log=True),
                  "depth": trial.suggest_int("depth", 4, 8),
                  "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 30.0, log=True),
                  "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 20.0, log=True)}
        model = CatBoostClassifier(**params)
        model.fit(ptr, eval_set=pes)
        ap = average_precision(yva, model.predict_proba(pva)[:, 1], wva)
        trial.set_user_attr("best_iteration", model.get_best_iteration())
        if ap > best["ap"]:
            best.update(ap=ap, model=model, params=params, it=model.get_best_iteration())
        return ap

    study = _study(objective, n_trials, seed, timeout)
    return Tuned("CatBoost", best["model"], best["params"], best["ap"], best["it"], study.trials_dataframe(),
                 time.perf_counter() - t0)


def predict(tuned: Tuned, X: pd.DataFrame) -> np.ndarray:
    """Positive-class scores of a tuned model."""
    if tuned.family == "LightGBM":
        return tuned.model.predict(X, num_iteration=tuned.best_iteration)
    if tuned.family == "XGBoost":
        d = xgb.DMatrix(X, enable_categorical=True, nthread=THREADS)
        return tuned.model.predict(d, iteration_range=(0, tuned.best_iteration + 1))
    if tuned.family == "CatBoost":
        return tuned.model.predict_proba(catboost_frame(X))[:, 1]
    return tuned.model.predict_proba(X)[:, 1]


def fit_lightgbm_fixed(params: dict, rounds: int, Xtr: pd.DataFrame, ytr: np.ndarray) -> lgb.Booster:
    """LightGBM with given parameters and a fixed number of rounds (used by the split-protocol comparison)."""
    p = {k: v for k, v in params.items() if k != "metric"}
    return lgb.train(p, lgb.Dataset(Xtr, ytr, params={"verbose": -1, "num_threads": THREADS}), num_boost_round=rounds)


def shap_values(tuned: Tuned, X: pd.DataFrame) -> tuple[np.ndarray, float]:
    """Exact TreeSHAP contributions (log-odds scale) from the model's own implementation, and the expected value."""
    if tuned.family == "LightGBM":
        c = tuned.model.predict(X, num_iteration=tuned.best_iteration, pred_contrib=True)
    elif tuned.family == "XGBoost":
        d = xgb.DMatrix(X, enable_categorical=True, nthread=THREADS)
        c = tuned.model.predict(d, pred_contribs=True, iteration_range=(0, tuned.best_iteration + 1))
    elif tuned.family == "CatBoost":
        cat_idx = [list(X.columns).index(k) for k in CATEGORICAL]
        c = tuned.model.get_feature_importance(Pool(catboost_frame(X), cat_features=cat_idx), type="ShapValues")
    else:
        raise ValueError(f"no TreeSHAP for {tuned.family}")
    c = np.asarray(c)
    return c[:, :-1], float(c[0, -1])
