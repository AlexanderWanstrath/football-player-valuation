"""Compute the value gap per player-season and explain the test season with SHAP.

gap = log(actual market value) - log(predicted value). Predictions come from
data/processed/predictions.parquet (train season out-of-fold, test season out-of-sample).
Two ways to flag a player: top / bottom 10% of the season (`gap_category`) and outside the
80% prediction interval (`interval_flag`, conformalized quantile regression).

Outputs (for the Power BI dashboard)
- data/processed/value_gaps.csv   one row per player-season with gap, category and top SHAP drivers
- data/processed/shap_values.csv  long format: one row per player x feature (test season)

SHAP explains the model refit on the full train season. That model produced the test-season
predictions, so drivers are exported for the test season only (for the train season they would
not add up to the out-of-fold predictions used for the gap).

Run from the project root:  python -m src.value_gap
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap

# Make the project root importable when the file is run directly instead of via -m
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config
from src.train_models import load_data

FEATURES = config.NUMERIC_FEATURES + config.CATEGORICAL_FEATURES
CATEGORIES = ["Below model", "In line", "Above model"]


# ---------------------------------------------------------------------------
# Value gap
# ---------------------------------------------------------------------------
def categorise(gap: pd.Series, season: pd.Series) -> pd.Series:
    """Top / bottom GAP_QUANTILE of each season vs the rest."""
    pct = gap.groupby(season).rank(pct=True)
    q = config.GAP_QUANTILE
    return pd.Series(np.select([pct > 1 - q, pct <= q], ["Above model", "Below model"], "In line"),
                     index=gap.index)


def conformal_margin(y: pd.Series, q_lo: pd.Series, q_hi: pd.Series, coverage: float) -> float:
    """CQR: how far the raw quantile band must be widened to reach the target coverage.

    Conformity score = how far a player lies outside his raw band (negative if inside).
    Uses the finite-sample corrected quantile of the scores (Romano et al., 2019).
    """
    scores = np.maximum(q_lo - y, y - q_hi)
    n = len(scores)
    return float(np.quantile(scores, min(1.0, np.ceil((n + 1) * coverage) / n)))


def add_intervals(out: pd.DataFrame) -> pd.DataFrame:
    """Calibrate the quantile band on the train season (out-of-fold) and flag values outside it."""
    train = out["season"] == config.TRAIN_SEASON
    margin = conformal_margin(out.loc[train, "log_value"], out.loc[train, "pred_log_q_lo"],
                              out.loc[train, "pred_log_q_hi"], config.INTERVAL_COVERAGE)
    out["interval_log_lo"] = out["pred_log_q_lo"] - margin
    out["interval_log_hi"] = out["pred_log_q_hi"] + margin

    # Shift by the season's median gap (season-level shift), consistent with the centred gap
    shift = out.groupby("season")["gap_log"].transform("median")
    lo, hi = out["interval_log_lo"] + shift, out["interval_log_hi"] + shift
    out["value_lower"] = np.exp(lo)
    out["value_upper"] = np.exp(hi)
    out["interval_flag"] = np.select([out["log_value"] > hi, out["log_value"] < lo],
                                     ["Above interval", "Below interval"], "Within interval")
    return out


def compute_gaps(df: pd.DataFrame, preds: pd.DataFrame) -> pd.DataFrame:
    out = df.merge(
        preds[["player_id", "season", f"pred_log_{config.GAP_MODEL}", f"pred_log_{config.GAP_CHECK_MODEL}",
               "pred_log_q_lo", "pred_log_q_hi"]],
        on=["player_id", "season"], how="inner", validate="1:1",
    )
    pred = out[f"pred_log_{config.GAP_MODEL}"]
    out["predicted_value"] = np.exp(pred)
    out["gap_log"] = out["log_value"] - pred
    out["gap_pct"] = np.exp(out["gap_log"]) - 1

    # The raw gap is shifted in the test season (about half higher value level, half lower predictions
    # from a team-strength shift in the sample; see notebook 04). The centred gap compares a player
    # with the typical gap of his season.
    out["gap_log_centred"] = out["gap_log"] - out.groupby("season")["gap_log"].transform("median")
    out["gap_pct_centred"] = np.exp(out["gap_log_centred"]) - 1
    out["gap_percentile"] = out.groupby("season")["gap_log"].rank(pct=True)
    out["gap_category"] = categorise(out["gap_log"], out["season"])

    # Cross-check with a second model: does Ridge put the player in the same category?
    out["gap_log_ridge"] = out["log_value"] - out[f"pred_log_{config.GAP_CHECK_MODEL}"]
    out["models_agree"] = categorise(out["gap_log_ridge"], out["season"]) == out["gap_category"]
    out = add_intervals(out)

    # Persistence: the same player's gap in the previous season (if in the sample)
    prev = out[["player_id", "season", "gap_log"]].assign(season=lambda d: d["season"] + 1)
    out = out.merge(prev.rename(columns={"gap_log": "gap_log_prev_season"}),
                    on=["player_id", "season"], how="left")
    return out


# ---------------------------------------------------------------------------
# SHAP
# ---------------------------------------------------------------------------
def explain(model, X: pd.DataFrame) -> shap.Explanation:
    explanation = shap.TreeExplainer(model)(X)
    # Sanity check: base value + contributions must equal the model prediction
    recon = explanation.values.sum(axis=1) + np.ravel(explanation.base_values)
    assert np.allclose(recon, model.predict(X)), "SHAP values do not add up to the predictions"
    return explanation


def shap_long(explanation: shap.Explanation, rows: pd.DataFrame) -> pd.DataFrame:
    """One row per player x feature; effect_pct = multiplicative effect on the predicted value."""
    values = pd.DataFrame(explanation.values, columns=FEATURES)
    values[["player_id", "season"]] = rows[["player_id", "season"]].values
    long = values.melt(id_vars=["player_id", "season"], var_name="feature", value_name="shap_value")
    feature_values = rows[FEATURES].astype(str).assign(player_id=rows["player_id"].values,
                                                       season=rows["season"].values)
    long = long.merge(
        feature_values.melt(id_vars=["player_id", "season"], var_name="feature", value_name="feature_value"),
        on=["player_id", "season", "feature"],
    )
    long["feature_label"] = long["feature"].map(config.FEATURE_LABELS)
    long["effect_pct"] = np.exp(long["shap_value"]) - 1
    return long


def top_drivers(long: pd.DataFrame, n: int) -> pd.DataFrame:
    """Strongest n positive and n negative SHAP contributions per player as readable text."""
    long = long.assign(text=long["feature_label"] + " (" + long["effect_pct"].map("{:+.0%}".format) + ")")
    out = []
    for direction, ascending, mask in [("up", False, long["shap_value"] > 0), ("down", True, long["shap_value"] < 0)]:
        ranked = (long[mask].sort_values("shap_value", ascending=ascending)
                  .groupby(["player_id", "season"]).head(n))
        ranked["rank"] = ranked.groupby(["player_id", "season"]).cumcount() + 1
        wide = ranked.pivot(index=["player_id", "season"], columns="rank", values="text")
        wide.columns = [f"driver_{direction}_{r}" for r in wide.columns]
        out.append(wide)
    return pd.concat(out, axis=1).reset_index()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
EXPORT_COLUMNS = [
    "player_id", "name", "season", "season_label", "league", "league_name", "club_name",
    "position", "sub_position", "age", "minutes", "goals", "assists", "cl_minutes", "team_ppg",
    "market_value", "predicted_value", "gap_log", "gap_pct", "gap_log_centred", "gap_pct_centred",
    "gap_percentile", "gap_category", "gap_log_ridge", "models_agree", "gap_log_prev_season",
    "value_lower", "value_upper", "interval_flag",
]


def main() -> None:
    df = load_data()
    preds = pd.read_parquet(config.PREDICTIONS_PATH)
    gaps = compute_gaps(df, preds)
    gaps["league_name"] = gaps["league"].astype(str).map(config.TOP5_LEAGUES)

    # SHAP for the test season with the refit model
    model = joblib.load(config.MODELS_DIR / f"{config.GAP_MODEL}.joblib")
    test = gaps[gaps["season"] == config.TEST_SEASON].reset_index(drop=True)
    long = shap_long(explain(model, test[FEATURES]), test)
    drivers = top_drivers(long, config.N_TOP_DRIVERS)

    export = gaps[EXPORT_COLUMNS].merge(drivers, on=["player_id", "season"], how="left")
    export = export.sort_values(["season", "gap_log"], ascending=[True, False])
    # utf-8-sig so Excel and Power BI read accented names correctly
    export.to_csv(config.VALUE_GAPS_PATH, index=False, encoding="utf-8-sig")
    long.to_csv(config.SHAP_VALUES_PATH, index=False, encoding="utf-8-sig")

    print(f"Wrote {len(export):,} rows to {config.VALUE_GAPS_PATH.name} "
          f"and {len(long):,} rows to {config.SHAP_VALUES_PATH.name}\n")
    summary = export.groupby(["season_label", "gap_category"]).agg(
        players=("player_id", "size"), median_gap_pct=("gap_pct", "median"),
        ridge_agrees=("models_agree", "mean"))
    print(summary.round(3).to_string())
    print("\n" + pd.crosstab([export["season_label"], export["gap_category"]], export["interval_flag"]).to_string())


if __name__ == "__main__":
    main()
