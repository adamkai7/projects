"""Forecasting models. Fit only on the current train window; never look at test y."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from features import ALL_FEATURES, HAR_FEATURES

MODEL_NAMES = ("baseline", "har", "ridge", "rf")


def _floor(y_train: np.ndarray) -> float:
    med = float(np.nanmedian(y_train))
    return max(1e-8, 0.01 * med) if np.isfinite(med) else 1e-8


def _clip(pred: np.ndarray, floor: float) -> np.ndarray:
    return np.maximum(np.asarray(pred, dtype=float), floor)


class FittedModels:
    def __init__(self, floor: float, har, ridge, rf, ridge_coef: pd.Series, rf_imp: pd.Series, har_coef: pd.Series):
        self.floor = floor
        self.har = har
        self.ridge = ridge
        self.rf = rf
        self.ridge_coef = ridge_coef
        self.rf_imp = rf_imp
        self.har_coef = har_coef

    def predict(self, frame: pd.DataFrame) -> dict[str, np.ndarray]:
        floor = self.floor
        return {
            "baseline": _clip(frame["baseline"].to_numpy(dtype=float), floor),
            "har": _clip(self.har.predict(frame[HAR_FEATURES]), floor),
            "ridge": _clip(self.ridge.predict(frame[ALL_FEATURES]), floor),
            "rf": _clip(self.rf.predict(frame[ALL_FEATURES]), floor),
        }


def fit_models(train: pd.DataFrame, cfg: dict[str, Any]) -> FittedModels:
    y = train["rv_fwd_5d"].to_numpy(dtype=float)
    floor = _floor(y)
    model_cfg = cfg.get("models", {})
    seed = int(cfg.get("validation", {}).get("random_seed", 42))

    har = LinearRegression()
    har.fit(train[HAR_FEATURES], y)
    har_coef = pd.Series(har.coef_, index=HAR_FEATURES)
    har_coef["intercept"] = float(har.intercept_)

    alphas = [float(a) for a in model_cfg.get("ridge_alphas", [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0])]
    # TimeSeriesSplit is only valid if rows are in time order. GridSearchCV keeps
    # the scaler inside each fold so alpha is not chosen on scaled test-fold rows.
    train_sorted = train.sort_values(["date", "ticker"])
    ridge_gs = GridSearchCV(
        Pipeline([("scaler", StandardScaler()), ("ridge", Ridge())]),
        param_grid={"ridge__alpha": alphas},
        cv=TimeSeriesSplit(n_splits=5),
        scoring="neg_mean_squared_error",
        n_jobs=1,
    )
    ridge_gs.fit(train_sorted[ALL_FEATURES], train_sorted["rv_fwd_5d"].to_numpy(dtype=float))
    ridge = ridge_gs.best_estimator_
    ridge_coef = pd.Series(ridge.named_steps["ridge"].coef_, index=ALL_FEATURES)

    rf = RandomForestRegressor(
        n_estimators=int(model_cfg.get("rf_n_estimators", 100)),
        max_depth=int(model_cfg.get("rf_max_depth", 6)),
        min_samples_leaf=int(model_cfg.get("rf_min_samples_leaf", 1)),
        random_state=seed,
        n_jobs=-1,
    )
    rf.fit(train[ALL_FEATURES], y)
    rf_imp = pd.Series(rf.feature_importances_, index=ALL_FEATURES)

    return FittedModels(
        floor=floor,
        har=har,
        ridge=ridge,
        rf=rf,
        ridge_coef=ridge_coef,
        rf_imp=rf_imp,
        har_coef=har_coef,
    )
