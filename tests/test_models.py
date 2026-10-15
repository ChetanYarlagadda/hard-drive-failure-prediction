"""Model fitting on a small synthetic problem: tuning returns a useful model and TreeSHAP values are additive."""

import numpy as np
import pandas as pd

from drivefail import models as M
from drivefail.features import CATEGORICAL, FEATURES


def _data(n: int, seed: int) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, len(FEATURES))).astype(np.float32), columns=FEATURES)
    X["model_code"] = rng.integers(0, 3, n)
    X["mfg_code"] = rng.integers(0, 4, n)
    logit = -3 + 2.5 * X.smart_5_d7 + 1.5 * (X.model_code == 2)
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return M.matrix(X, n_models=3), y.to_numpy()


def test_lightgbm_tuning_and_shap_additivity() -> None:
    Xtr, ytr = _data(3000, 0)
    Xva, yva = _data(2000, 1)
    w = np.ones(len(yva))
    es = M.EarlyStopSet(Xva, yva, w)
    t = M.tune_lightgbm(Xtr, ytr, Xva, yva, w, es, n_trials=2, max_rounds=100)
    base_rate = yva.mean()
    assert len(t.trials) == 2 and t.val_ap > 2 * base_rate  # learns the planted signal
    sv, base = M.shap_values(t, Xva.iloc[:200])
    raw = t.model.predict(Xva.iloc[:200], num_iteration=t.best_iteration, raw_score=True)
    assert np.allclose(sv.sum(1) + base, raw, atol=1e-6)
    top = FEATURES[int(np.argmax(np.abs(sv).mean(0)))]
    assert top == "smart_5_d7"


def test_logistic_baseline_and_categoricals() -> None:
    Xtr, ytr = _data(2000, 2)
    Xva, yva = _data(1000, 3)
    t = M.fit_logistic(Xtr, ytr, Xva, yva, np.ones(len(yva)), cs=(0.1, 1.0))
    assert t.val_ap > 2 * yva.mean() and t.params["C"] in (0.1, 1.0)
    assert all(str(Xtr[c].dtype) == "category" for c in CATEGORICAL)
    assert M.predict(t, Xva).shape == (len(yva),)
