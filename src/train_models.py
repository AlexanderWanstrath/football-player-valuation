"""Train baseline, Ridge and LightGBM on 2024/25 and evaluate on 2025/26.

Outputs
- reports/model_metrics.csv          CV metrics on the train season, test metrics on the test season
- data/processed/predictions.parquet out-of-fold predictions (train) and test predictions per model
- models/*.joblib                    fitted Ridge and LightGBM for the SHAP analysis (day 4)

Run from the project root:  python -m src.train_models
"""

import os
import sys
from pathlib import Path

# joblib counts CPU cores via `wmic`, which Windows 11 no longer ships (noisy traceback).
# Setting a limit below the logical core count skips that lookup.
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(max(1, (os.cpu_count() or 2) - 1)))

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, cross_val_predict
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler

# Make the project root importable when the file is run directly instead of via -m
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config

PREV_FEATURES = ["prev_minutes", "prev_goals", "prev_assists", "prev_big5"]
CV = KFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_data() -> pd.DataFrame:
    df = pd.read_parquet(config.DATASET_PATH)
    df["age_group"] = pd.cut(df["age"], bins=config.AGE_BINS, labels=config.AGE_LABELS, right=False)
    # Fixed category sets so train and test share the same encoding in LightGBM
    for col in config.CATEGORICAL_FEATURES:
        df[col] = pd.Categorical(df[col], categories=sorted(df[col].dropna().unique()))
    return df


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def evaluate(y_log: pd.Series, pred_log: np.ndarray) -> dict:
    """R2 and RMSE on the log scale, median absolute percentage error in EUR."""
    ape = np.abs(np.exp(pred_log) - np.exp(y_log)) / np.exp(y_log)
    return {
        "r2": r2_score(y_log, pred_log),
        "rmse_log": mean_squared_error(y_log, pred_log) ** 0.5,
        "mdape_eur": float(np.median(ape)),
        "n": len(y_log),
    }


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
class GroupMedianBaseline(BaseEstimator, RegressorMixin):
    """Median log value per position x league x age group, with coarser fallbacks."""

    def __init__(self, groups: list[str]):
        self.groups = groups

    def fit(self, X: pd.DataFrame, y: pd.Series):
        # Fallback chain: all groups, then drop the last one, ..., then the global median
        self.levels_ = [self.groups[:i] for i in range(len(self.groups), 0, -1)]
        data = X.assign(_y=np.asarray(y))
        self.tables_ = [data.groupby(lvl, observed=True)["_y"].median() for lvl in self.levels_]
        self.global_ = float(np.median(y))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        pred = pd.Series(np.nan, index=X.index)
        for lvl, table in zip(self.levels_, self.tables_):
            keys = pd.MultiIndex.from_frame(X[lvl]) if len(lvl) > 1 else X[lvl[0]]
            pred = pred.fillna(pd.Series(table.reindex(keys).values, index=X.index))
        return pred.fillna(self.global_).values


def make_ridge(numeric: list[str]) -> Pipeline:
    """Linear model: splines for the non-linear age curve, one-hot for categories."""
    prev = [c for c in numeric if c in PREV_FEATURES]
    other = [c for c in numeric if c not in PREV_FEATURES and c != "age"]
    pre = ColumnTransformer([
        ("age", make_pipeline(SplineTransformer(n_knots=5, degree=3), StandardScaler()), ["age"]),
        # No previous season in the data -> 0; `prev_season_covered` tells the model why
        ("prev", make_pipeline(SimpleImputer(strategy="constant", fill_value=0), StandardScaler()), prev),
        ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), other),
        ("cat", make_pipeline(SimpleImputer(strategy="most_frequent"),
                              OneHotEncoder(handle_unknown="ignore")), config.CATEGORICAL_FEATURES),
    ])
    return Pipeline([("pre", pre), ("model", RidgeCV(alphas=np.logspace(-2, 3, 30)))])


def tune_lgbm(X: pd.DataFrame, y: pd.Series) -> LGBMRegressor:
    """Small grid search with k-fold CV on the train season."""
    base = LGBMRegressor(
        learning_rate=0.03, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
        reg_lambda=1.0, random_state=config.RANDOM_STATE, verbose=-1,
    )
    grid = {
        "n_estimators": [300, 600],
        "num_leaves": [7, 15, 31],
        "min_child_samples": [10, 20, 40],
    }
    search = GridSearchCV(base, grid, cv=CV, scoring="neg_root_mean_squared_error", n_jobs=-1)
    search.fit(X, y)
    print(f"LightGBM best params: {search.best_params_} (CV RMSE {-search.best_score_:.3f})")
    return search.best_estimator_


def fit_quantiles(lgbm: LGBMRegressor, X_train, y_train, X_test) -> dict[str, np.ndarray]:
    """Lower / upper quantile LightGBM with the tuned settings: out-of-fold on train, refit for test.

    Raw quantile models under-cover (about 66% for a nominal 80%); value_gap.py calibrates them
    with conformalized quantile regression (CQR) on the out-of-fold errors.
    """
    alpha_lo = (1 - config.INTERVAL_COVERAGE) / 2
    out = {}
    for col, alpha in [("pred_log_q_lo", alpha_lo), ("pred_log_q_hi", 1 - alpha_lo)]:
        model = LGBMRegressor(**{**lgbm.get_params(), "objective": "quantile", "alpha": alpha})
        oof = cross_val_predict(model, X_train, y_train, cv=CV)
        out[col] = np.concatenate([oof, model.fit(X_train, y_train).predict(X_test)])
    return out


# ---------------------------------------------------------------------------
# Training run
# ---------------------------------------------------------------------------
def fit_and_score(name, model, X_train, y_train, X_test, y_test, rows, preds):
    """Out-of-fold predictions on train, refit on all of train, predict test."""
    oof = cross_val_predict(model, X_train, y_train, cv=CV)
    model.fit(X_train, y_train)
    test_pred = model.predict(X_test)
    rows.append({"model": name, "split": "train_cv", **evaluate(y_train, oof)})
    rows.append({"model": name, "split": "test", **evaluate(y_test, test_pred)})
    preds[name] = np.concatenate([oof, test_pred])
    return model


def main() -> None:
    df = load_data()
    train = df[df["season"] == config.TRAIN_SEASON].reset_index(drop=True)
    test = df[df["season"] == config.TEST_SEASON].reset_index(drop=True)
    y_train, y_test = train[config.TARGET], test[config.TARGET]
    features = config.NUMERIC_FEATURES + config.CATEGORICAL_FEATURES
    snapshot_features = features + config.SNAPSHOT_FEATURES
    print(f"Train {config.season_label(config.TRAIN_SEASON)}: {len(train):,} rows | "
          f"test {config.season_label(config.TEST_SEASON)}: {len(test):,} rows | {len(features)} features\n")

    rows, preds = [], {}
    fit_and_score("baseline", GroupMedianBaseline(config.BASELINE_GROUPS),
                  train, y_train, test, y_test, rows, preds)
    ridge = fit_and_score("ridge", make_ridge(config.NUMERIC_FEATURES),
                          train[features], y_train, test[features], y_test, rows, preds)
    lgbm = fit_and_score("lightgbm", tune_lgbm(train[features], y_train),
                         train[features], y_train, test[features], y_test, rows, preds)
    # Comparison only: same LightGBM settings plus the leaky snapshot columns
    lgbm_snap = LGBMRegressor(**lgbm.get_params())
    fit_and_score("lightgbm_with_snapshot", lgbm_snap,
                  train[snapshot_features], y_train, test[snapshot_features], y_test, rows, preds)

    # Metrics
    metrics = pd.DataFrame(rows)
    config.METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    metrics.round(4).to_csv(config.METRICS_PATH, index=False)
    print("\n" + metrics.pivot(index="model", columns="split", values=["r2", "rmse_log", "mdape_eur"])
          .loc[list(preds)].round(3).to_string())

    # Predictions: out-of-fold for the train season, true out-of-sample for the test season
    out = pd.concat([train, test], ignore_index=True)[
        ["player_id", "name", "season", "season_label", "league", "club_name", "position",
         "age", "minutes", "market_value", config.TARGET]
    ]
    out["split"] = np.where(out["season"] == config.TRAIN_SEASON, "train_oof", "test")
    for name, pred in preds.items():
        out[f"pred_log_{name}"] = pred
    # Raw quantile predictions for the prediction interval (calibrated in value_gap.py)
    for col, pred in fit_quantiles(lgbm, train[features], y_train, test[features]).items():
        out[col] = pred
    out.to_parquet(config.PREDICTIONS_PATH, index=False)

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(ridge, config.MODELS_DIR / "ridge.joblib")
    joblib.dump(lgbm, config.MODELS_DIR / "lightgbm.joblib")
    print(f"\nWrote {config.METRICS_PATH.name}, {config.PREDICTIONS_PATH.name} and models to {config.MODELS_DIR}")


if __name__ == "__main__":
    main()
