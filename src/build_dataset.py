"""Build the player-season modelling table and write it to data/processed.

One row per outfield player and season (Big 5 league minutes summed across leagues),
with the market value at the season cutoff as target.

Run from the project root:  python -m src.build_dataset
(running the file directly, e.g. via the VS Code run button, also works)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Make the project root importable when the file is run directly instead of via -m
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config
from src.db import get_connection, query

LEAGUES = list(config.TOP5_LEAGUES)
SEASONS = config.SEASONS
PREV_SEASONS = [s - 1 for s in SEASONS]

# ---------------------------------------------------------------------------
# SQL building blocks
# ---------------------------------------------------------------------------
# League stats per player-season. Mid-season movers get one row: stats are summed,
# `league` and `club_id` are the ones with the most minutes.
LEAGUE_STATS_SQL = """
    WITH apps AS (
        SELECT a.player_id,
               CAST(g.season AS INTEGER) AS season,
               a.competition_id,
               a.player_club_id,
               a.minutes_played, a.goals, a.assists, a.yellow_cards, a.red_cards
        FROM appearances a
        JOIN games g USING (game_id)
        WHERE a.competition_id IN ? AND CAST(g.season AS INTEGER) IN ?
    ),
    by_club AS (
        SELECT player_id, season, competition_id, player_club_id, SUM(minutes_played) AS club_minutes
        FROM apps
        GROUP BY ALL
    ),
    main_club AS (
        SELECT player_id, season,
               arg_max(player_club_id, club_minutes) AS club_id,
               arg_max(competition_id, club_minutes) AS league
        FROM by_club
        GROUP BY ALL
    )
    SELECT s.player_id, s.season, m.league, m.club_id,
           s.appearances, s.minutes, s.goals, s.assists, s.yellow_cards, s.red_cards,
           s.n_leagues, s.n_clubs
    FROM (
        SELECT player_id, season,
               COUNT(*)                         AS appearances,
               SUM(minutes_played)              AS minutes,
               SUM(goals)                       AS goals,
               SUM(assists)                     AS assists,
               SUM(yellow_cards)                AS yellow_cards,
               SUM(red_cards)                   AS red_cards,
               COUNT(DISTINCT competition_id)   AS n_leagues,
               COUNT(DISTINCT player_club_id)   AS n_clubs
        FROM apps
        GROUP BY ALL
    ) s
    JOIN main_club m USING (player_id, season)
"""

# Team strength = league points per game of the main club. Uses results only, so the
# player's own market value cannot leak in (unlike squad value).
TEAM_STRENGTH_SQL = """
    SELECT cg.club_id,
           CAST(g.season AS INTEGER)                           AS season,
           COUNT(*)                                            AS team_games,
           AVG(CASE WHEN cg.own_goals > cg.opponent_goals THEN 3
                    WHEN cg.own_goals = cg.opponent_goals THEN 1
                    ELSE 0 END)                                AS team_ppg,
           AVG(cg.own_goals - cg.opponent_goals)               AS team_goal_diff_pg
    FROM club_games cg
    JOIN games g USING (game_id)
    WHERE g.competition_id IN ? AND CAST(g.season AS INTEGER) IN ?
    GROUP BY ALL
"""

EUROPE_SQL = """
    SELECT a.player_id,
           CAST(g.season AS INTEGER)                                            AS season,
           SUM(CASE WHEN a.competition_id = 'CL' THEN a.minutes_played ELSE 0 END) AS cl_minutes,
           SUM(CASE WHEN a.competition_id = 'EL' THEN a.minutes_played ELSE 0 END) AS el_minutes
    FROM appearances a
    JOIN games g USING (game_id)
    WHERE a.competition_id IN ? AND CAST(g.season AS INTEGER) IN ?
    GROUP BY ALL
"""

# Previous season in any first-tier domestic league covered by the dataset.
# Note: leagues with calendar-year seasons (e.g. MLS, Brazil) are matched by start year.
PREV_SEASON_SQL = """
    SELECT a.player_id,
           CAST(g.season AS INTEGER) + 1                        AS season,
           SUM(a.minutes_played)                                AS prev_minutes,
           SUM(a.goals)                                         AS prev_goals,
           SUM(a.assists)                                       AS prev_assists,
           MAX(CASE WHEN a.competition_id IN ? THEN 1 ELSE 0 END) AS prev_big5
    FROM appearances a
    JOIN games g USING (game_id)
    JOIN competitions c ON c.competition_id = a.competition_id
    WHERE c.type = 'domestic_league' AND CAST(g.season AS INTEGER) IN ?
    GROUP BY ALL
"""

MID_SEASON_TRANSFER_SQL = """
    SELECT DISTINCT t.player_id, s.season, 1 AS mid_season_transfer
    FROM transfers t
    JOIN (SELECT UNNEST(?::INTEGER[]) AS season) s
      ON t.transfer_date BETWEEN CAST(s.season || '-' || ? AS DATE)
                             AND CAST((s.season + 1) || '-' || ? AS DATE)
"""

PROFILE_SQL = """
    SELECT p.player_id, p.name, p.position, p.sub_position, p.foot, p.height_in_cm,
           p.date_of_birth, p.country_of_citizenship,
           -- snapshot columns (mid 2026): not for the main model, see CLAUDE.md
           p.contract_expiration_date, p.international_caps
    FROM players p
"""

CLUB_SQL = "SELECT CAST(club_id AS INTEGER) AS club_id, name AS club_name FROM clubs"

# Latest valuation on or before the cutoff date
TARGET_SQL = """
    SELECT k.player_id, k.season,
           v.date                AS valuation_date,
           v.market_value_in_eur AS market_value
    FROM keys_df k
    ASOF LEFT JOIN player_valuations v
      ON k.player_id = v.player_id AND k.cutoff >= v.date
"""


# ---------------------------------------------------------------------------
# Build steps
# ---------------------------------------------------------------------------
def load_features(con) -> pd.DataFrame:
    """Join league stats with team strength, Europe, previous season, transfers and profile."""
    stats = query(con, LEAGUE_STATS_SQL, [LEAGUES, SEASONS])
    team = query(con, TEAM_STRENGTH_SQL, [LEAGUES, SEASONS])
    europe = query(con, EUROPE_SQL, [list(config.EUROPEAN_COMPETITIONS), SEASONS])
    prev = query(con, PREV_SEASON_SQL, [LEAGUES, PREV_SEASONS])
    moves = query(con, MID_SEASON_TRANSFER_SQL, [SEASONS, *config.MID_SEASON_WINDOW])
    profile = query(con, PROFILE_SQL)
    clubs = query(con, CLUB_SQL)

    df = (
        stats.merge(team, on=["club_id", "season"], how="left")
        .merge(europe, on=["player_id", "season"], how="left")
        .merge(prev, on=["player_id", "season"], how="left")
        .merge(moves, on=["player_id", "season"], how="left")
        .merge(profile, on="player_id", how="left")
        .merge(clubs, on="club_id", how="left")
    )

    # Missing European / transfer rows mean "none"; missing previous season stays NaN
    df[["cl_minutes", "el_minutes"]] = df[["cl_minutes", "el_minutes"]].fillna(0)
    # A club change within the Big 5 counts as a move even without a transfers row
    df["mid_season_transfer"] = ((df["mid_season_transfer"] == 1) | (df["n_clubs"] > 1)).astype(int)
    df["prev_season_covered"] = df["prev_minutes"].notna().astype(int)
    return df


def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    """Per-90 rates, minutes share, age and contract length."""
    p90 = df["minutes"] / 90
    df["goals_p90"] = df["goals"] / p90
    df["assists_p90"] = df["assists"] / p90
    df["goal_contrib_p90"] = (df["goals"] + df["assists"]) / p90
    # Share of the main club's league minutes (34 vs 38 games per league)
    df["minutes_share"] = df["minutes"] / (df["team_games"] * 90)
    df["euro_minutes"] = df["cl_minutes"] + df["el_minutes"]

    # Age at the middle of the season (1 January)
    mid_season = pd.to_datetime((df["season"] + 1).astype(str) + "-01-01")
    df["age"] = (mid_season - pd.to_datetime(df["date_of_birth"])).dt.days / 365.25

    cutoff = pd.to_datetime(df["season"].map(config.season_cutoff))
    df["contract_months_left"] = (pd.to_datetime(df["contract_expiration_date"]) - cutoff).dt.days / 30.44
    df["season_label"] = df["season"].map(config.season_label)
    return df


def add_target(con, df: pd.DataFrame) -> pd.DataFrame:
    """Attach the latest valuation on or before the cutoff (ASOF join)."""
    keys = df[["player_id", "season"]].copy()
    keys["cutoff"] = pd.to_datetime(keys["season"].map(config.season_cutoff))
    con.register("keys_df", keys)
    target = query(con, TARGET_SQL)
    con.unregister("keys_df")

    df = df.merge(target, on=["player_id", "season"], how="left")
    # One valuation of 0 EUR exists in the data; treat it as missing (log undefined)
    mv = df["market_value"].astype("float64")
    df["market_value"] = mv.where(mv > 0)
    cutoff = pd.to_datetime(df["season"].map(config.season_cutoff))
    df["valuation_age_days"] = (cutoff - pd.to_datetime(df["valuation_date"])).dt.days
    df["log_value"] = np.log(df["market_value"])
    return df


def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Scope filters from CLAUDE.md; prints how many rows each step removes."""
    steps = [
        ("outfield only", ~df["position"].isin(config.EXCLUDED_POSITIONS)),
        (f">= {config.MIN_MINUTES} league minutes", df["minutes"] >= config.MIN_MINUTES),
        ("has valuation", df["market_value"].notna()),
        (f"valuation <= {config.MAX_VALUATION_AGE_DAYS} days old",
         df["valuation_age_days"] <= config.MAX_VALUATION_AGE_DAYS),
    ]
    keep = pd.Series(True, index=df.index)
    print(f"{'start':<32}{len(df):>7,}")
    for label, mask in steps:
        keep &= mask
        print(f"{label:<32}{keep.sum():>7,}")
    return df[keep].reset_index(drop=True)


COLUMNS = [
    # identifiers
    "player_id", "name", "season", "season_label", "league", "club_id", "club_name",
    # target
    "market_value", "log_value", "valuation_date", "valuation_age_days",
    # profile
    "position", "sub_position", "foot", "height_in_cm", "age", "country_of_citizenship",
    # league performance
    "appearances", "minutes", "minutes_share", "goals", "assists",
    "goals_p90", "assists_p90", "goal_contrib_p90", "yellow_cards", "red_cards",
    # context
    "team_ppg", "team_goal_diff_pg", "cl_minutes", "el_minutes", "euro_minutes",
    "mid_season_transfer", "n_leagues", "n_clubs",
    # previous season
    "prev_season_covered", "prev_minutes", "prev_goals", "prev_assists", "prev_big5",
    # snapshot columns (mid 2026): comparison model only
    "contract_months_left", "international_caps",
]


def build() -> pd.DataFrame:
    con = get_connection()
    try:
        df = load_features(con)
        df = add_derived(df)
        df = add_target(con, df)
    finally:
        con.close()

    df = apply_filters(df)[COLUMNS]
    assert not df.duplicated(["player_id", "season"]).any(), "duplicate player-seasons"
    return df


def main() -> None:
    df = build()
    config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    df.to_parquet(config.DATASET_PATH, index=False)
    print(f"\nWrote {len(df):,} rows x {df.shape[1]} columns to {config.DATASET_PATH}")
    print(df.groupby("season_label").size().rename("rows").to_string())


if __name__ == "__main__":
    main()
