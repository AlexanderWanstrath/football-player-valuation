"""Thin helpers around the DuckDB database file."""

import duckdb
import pandas as pd

from src.config import DB_PATH

_DOWNLOAD_HINT = (
    "Download it from "
    "https://pub-e682421888d945d684bcae8890b0ec20.r2.dev/data/transfermarkt-datasets.duckdb "
    "and place it in data/raw/ (see README)."
)


def get_connection(read_only: bool = True) -> duckdb.DuckDBPyConnection:
    """Open a connection to the Transfermarkt DuckDB file."""
    if not DB_PATH.exists():
        raise FileNotFoundError(f"Database not found at {DB_PATH}. {_DOWNLOAD_HINT}")
    return duckdb.connect(str(DB_PATH), read_only=read_only)


def query(con: duckdb.DuckDBPyConnection, sql: str, params: list | None = None) -> pd.DataFrame:
    """Run a SQL query and return the result as a pandas DataFrame."""
    return con.execute(sql, params or []).df()
