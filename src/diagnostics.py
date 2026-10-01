"""Model diagnostics used in notebooks/04_deep_dive.ipynb.

- Driver robustness: grouped permutation importance
- Interactions: partial dependence and Friedman's H-statistic
- Drift between seasons: population stability index (PSI) and Kolmogorov-Smirnov test
"""

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import mean_squared_error

# Features grouped into business concepts (for grouped importance)
FEATURE_GROUPS = {
    "Age": ["age"],
    "Team strength": ["team_ppg", "team_goal_diff_pg"],
    "League": ["league"],
    "Playing time": ["appearances", "minutes", "minutes_share"],
    "Output (goals, assists)": ["goals", "assists", "goals_p90", "assists_p90", "goal_contrib_p90"],
    "European minutes": ["cl_minutes", "el_minutes"],
    "Previous season": ["prev_season_covered", "prev_minutes", "prev_goals", "prev_assists", "prev_big5"],
    "Position and profile": ["position", "sub_position", "foot", "height_in_cm"],
    "Discipline": ["yellow_cards", "red_cards"],
    "Mid-season transfer": ["mid_season_transfer"],
}


# ---------------------------------------------------------------------------
# Driver robustness
# ---------------------------------------------------------------------------
def rmse(y, pred) -> float:
    return mean_squared_error(y, pred) ** 0.5


def grouped_permutation_importance(model, X: pd.DataFrame, y: pd.Series, groups: dict[str, list[str]],
                                   n_repeats: int = 10, random_state: int = 42) -> pd.DataFrame:
    """Increase in RMSE when all features of a group are shuffled together.

    Shuffling correlated features jointly avoids the classic permutation-importance trap where
    two correlated features each look unimportant because the other one stands in for it.
    """
    rng = np.random.default_rng(random_state)
    base = rmse(y, model.predict(X))
    rows = []
    for name, cols in groups.items():
        cols = [c for c in cols if c in X.columns]
        scores = []
        for _ in range(n_repeats):
            Xp = X.copy()
            idx = rng.permutation(len(X))
            Xp[cols] = X[cols].iloc[idx].set_axis(X.index)   # keeps categorical dtypes
            scores.append(rmse(y, model.predict(Xp)) - base)
        rows.append({"group": name, "rmse_increase": np.mean(scores), "std": np.std(scores)})
    return pd.DataFrame(rows).sort_values("rmse_increase", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Interactions
# ---------------------------------------------------------------------------
def partial_dependence(model, X: pd.DataFrame, features: list[str], grid: pd.DataFrame) -> np.ndarray:
    """Average prediction when `features` are set to each row of `grid` for every player in X."""
    n = len(X)
    stacked = pd.concat([X] * len(grid), ignore_index=True)
    for col in features:
        values = np.repeat(grid[col].to_numpy(), n)
        if isinstance(X[col].dtype, pd.CategoricalDtype):
            stacked[col] = pd.Categorical(values, categories=X[col].cat.categories)
        else:
            stacked[col] = values
    return model.predict(stacked).reshape(len(grid), n).mean(axis=1)


def h_statistic(model, X: pd.DataFrame, pairs: list[tuple[str, str]], n_sample: int = 250,
                random_state: int = 42) -> pd.DataFrame:
    """Friedman's H-statistic per feature pair (Friedman & Popescu, 2008).

    H = share of the variance of the joint partial dependence that is not explained by the two
    individual partial dependences. 0 = purely additive effects, 1 = pure interaction.
    """
    S = X.sample(min(n_sample, len(X)), random_state=random_state).reset_index(drop=True)
    centred = lambda pd_: pd_ - pd_.mean()
    single = {f: centred(partial_dependence(model, S, [f], S[[f]])) for f in {f for p in pairs for f in p}}
    rows = []
    for a, b in pairs:
        joint = centred(partial_dependence(model, S, [a, b], S[[a, b]]))
        h2 = np.sum((joint - single[a] - single[b]) ** 2) / np.sum(joint ** 2)
        rows.append({"feature_a": a, "feature_b": b, "h_statistic": float(np.sqrt(h2))})
    return pd.DataFrame(rows).sort_values("h_statistic", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Drift
# ---------------------------------------------------------------------------
def psi(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    """Population stability index of `actual` against `expected`.

    Numeric: decile bins of the expected distribution. Categorical: one bin per category.
    Rule of thumb: < 0.10 stable, 0.10-0.25 moderate shift, > 0.25 major shift.
    """
    eps = 1e-4
    if isinstance(expected.dtype, pd.CategoricalDtype) or expected.dtype == object:
        e = expected.astype(str).value_counts(normalize=True)
        a = actual.astype(str).value_counts(normalize=True).reindex(e.index, fill_value=0)
    else:
        edges = np.unique(np.quantile(expected.dropna(), np.linspace(0, 1, bins + 1)))
        edges[0], edges[-1] = -np.inf, np.inf
        e = pd.cut(expected.dropna(), edges).value_counts(normalize=True, sort=False)
        a = pd.cut(actual.dropna(), edges).value_counts(normalize=True, sort=False)
    e, a = e.clip(lower=eps), a.clip(lower=eps)
    return float(np.sum((a - e) * np.log(a / e)))


def drift_table(reference: pd.DataFrame, current: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """PSI for every column plus the KS test and median shift for numeric columns."""
    rows = []
    for col in columns:
        row = {"feature": col, "psi": psi(reference[col], current[col])}
        if pd.api.types.is_numeric_dtype(reference[col]):
            ks = ks_2samp(reference[col].dropna(), current[col].dropna())
            row.update(ks_stat=ks.statistic, ks_pvalue=ks.pvalue,
                       median_ref=reference[col].median(), median_cur=current[col].median())
        rows.append(row)
    out = pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)
    out["psi_level"] = pd.cut(out["psi"], [-np.inf, 0.10, 0.25, np.inf], labels=["stable", "moderate", "major"])
    return out
