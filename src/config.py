"""Central project configuration: paths, scope and modelling constants."""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
FIGURES = ROOT / "reports" / "figures"

# The DuckDB file can be relocated via the TM_DB_PATH environment variable.
DB_PATH = Path(os.environ.get("TM_DB_PATH", DATA_RAW / "transfermarkt-datasets.duckdb"))

# ---------------------------------------------------------------------------
# Scope
# ---------------------------------------------------------------------------
# Transfermarkt competition ids of the Big 5 European leagues
TOP5_LEAGUES = {
    "GB1": "Premier League",
    "ES1": "LaLiga",
    "IT1": "Serie A",
    "L1": "Bundesliga",
    "FR1": "Ligue 1",
}

# Seasons are identified by their start year (2024 = season 2024/25)
TRAIN_SEASON = 2024
TEST_SEASON = 2025
SEASONS = [TRAIN_SEASON, TEST_SEASON]

# Minimum league minutes for a player-season to enter the modelling sample
MIN_MINUTES = 900

# Goalkeepers follow a different valuation logic and are excluded in v1
EXCLUDED_POSITIONS = ["Goalkeeper", "Missing"]

# ---------------------------------------------------------------------------
# Target definition
# ---------------------------------------------------------------------------
# Target = latest market value on or before the cutoff date after a season.
# Note: the dataset snapshot ends in June 2026 (valuations until 2026-06-12),
# so the cutoff is set to mid-June for every season to keep them comparable.
CUTOFF_MONTH_DAY = "06-15"

# Valuations older than this (relative to the cutoff) are treated as stale
MAX_VALUATION_AGE_DAYS = 180

# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------
# European club competitions with appearance data (Conference League has none)
EUROPEAN_COMPETITIONS = {"CL": "Champions League", "EL": "Europa League"}

# A transfer inside this window (season start year / following year) counts as a
# mid-season move. It skips the summer window and end-of-loan returns in June.
MID_SEASON_WINDOW = ("09-15", "04-30")

# ---------------------------------------------------------------------------
# Model features (day 3)
# ---------------------------------------------------------------------------
TARGET = "log_value"

NUMERIC_FEATURES = [
    "age", "height_in_cm",
    "appearances", "minutes", "minutes_share", "goals", "assists",
    "goals_p90", "assists_p90", "goal_contrib_p90", "yellow_cards", "red_cards",
    "team_ppg", "team_goal_diff_pg", "cl_minutes", "el_minutes", "mid_season_transfer",
    "prev_season_covered", "prev_minutes", "prev_goals", "prev_assists", "prev_big5",
]
CATEGORICAL_FEATURES = ["league", "position", "sub_position", "foot"]

# Snapshot columns (mid 2026) that leak future information into 2024/25.
# Used only in the comparison model, see CLAUDE.md.
SNAPSHOT_FEATURES = ["contract_months_left", "international_caps"]

# Baseline: median log value per group (falls back to coarser groups if a cell is empty)
BASELINE_GROUPS = ["position", "league", "age_group"]
AGE_BINS = [0, 21, 24, 27, 30, 99]
AGE_LABELS = ["<21", "21-23", "24-26", "27-29", "30+"]

RANDOM_STATE = 42
CV_FOLDS = 5

# Readable feature names for charts and the dashboard
FEATURE_LABELS = {
    "age": "Age", "height_in_cm": "Height (cm)",
    "appearances": "League appearances", "minutes": "League minutes", "minutes_share": "Share of league minutes",
    "goals": "Goals", "assists": "Assists", "goals_p90": "Goals per 90", "assists_p90": "Assists per 90",
    "goal_contrib_p90": "Goals + assists per 90", "yellow_cards": "Yellow cards", "red_cards": "Red cards",
    "team_ppg": "Team points per game", "team_goal_diff_pg": "Team goal difference per game",
    "cl_minutes": "Champions League minutes", "el_minutes": "Europa League minutes",
    "mid_season_transfer": "Mid-season transfer", "prev_season_covered": "Previous season in data",
    "prev_minutes": "Previous season minutes", "prev_goals": "Previous season goals",
    "prev_assists": "Previous season assists", "prev_big5": "Played Big 5 previous season",
    "league": "League", "position": "Position", "sub_position": "Detailed position", "foot": "Preferred foot",
}

# ---------------------------------------------------------------------------
# Value gap (day 4)
# ---------------------------------------------------------------------------
# gap = log(actual value) - log(predicted value) from GAP_MODEL.
# Players in the top / bottom GAP_QUANTILE of their season are flagged.
GAP_MODEL = "lightgbm"
GAP_CHECK_MODEL = "ridge"
GAP_QUANTILE = 0.10
N_TOP_DRIVERS = 3

# Prediction interval: conformalized quantile regression with this target coverage.
# A player whose value lies outside his (season-centred) interval is flagged as well.
INTERVAL_COVERAGE = 0.80

# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------
DATASET_PATH = DATA_PROCESSED / "player_seasons.parquet"
PREDICTIONS_PATH = DATA_PROCESSED / "predictions.parquet"
MODELS_DIR = ROOT / "models"
METRICS_PATH = ROOT / "reports" / "model_metrics.csv"
VALUE_GAPS_PATH = DATA_PROCESSED / "value_gaps.csv"
SHAP_VALUES_PATH = DATA_PROCESSED / "shap_values.csv"


def season_cutoff(season: int) -> str:
    """Return the valuation cutoff date (ISO string) for a season start year."""
    return f"{season + 1}-{CUTOFF_MONTH_DAY}"


def season_label(season: int) -> str:
    """2024 -> '2024/25'."""
    return f"{season}/{str(season + 1)[-2:]}"
