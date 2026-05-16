"""DuckDB + Parquet storage for tick and bar data — replaces SQLite."""
from __future__ import annotations
import duckdb
import numpy as np
import pandas as pd
from pathlib import Path
from ..config import CFG, DUCKDB_PATH, PARQUET_DIR


SCHEMA = """
CREATE TABLE IF NOT EXISTS bars (
    symbol VARCHAR NOT NULL,
    tf VARCHAR NOT NULL,
    ts_ms BIGINT NOT NULL,
    open DOUBLE, high DOUBLE, low DOUBLE, close DOUBLE, volume DOUBLE,
    quote_volume DOUBLE, n_trades INTEGER,
    taker_buy_base DOUBLE, taker_buy_quote DOUBLE,
    PRIMARY KEY (symbol, tf, ts_ms)
);

CREATE TABLE IF NOT EXISTS funding (
    symbol VARCHAR, ts_ms BIGINT, rate DOUBLE,
    PRIMARY KEY (symbol, ts_ms)
);

CREATE TABLE IF NOT EXISTS open_interest (
    symbol VARCHAR, ts_ms BIGINT, oi DOUBLE,
    PRIMARY KEY (symbol, ts_ms)
);

CREATE TABLE IF NOT EXISTS macro_metrics (
    metric VARCHAR, ts_ms BIGINT, value DOUBLE,
    PRIMARY KEY (metric, ts_ms)
);

CREATE TABLE IF NOT EXISTS setups (
    id BIGINT PRIMARY KEY,
    created_ms BIGINT,
    symbol VARCHAR, direction VARCHAR,
    entry DOUBLE, sl DOUBLE, tp1 DOUBLE, tp2 DOUBLE, tp3 DOUBLE,
    rr DOUBLE, score INTEGER, grade VARCHAR, tf VARCHAR,
    confluence JSON, status VARCHAR
);

CREATE TABLE IF NOT EXISTS trades (
    id BIGINT PRIMARY KEY,
    setup_id BIGINT, symbol VARCHAR, direction VARCHAR,
    entry_ms BIGINT, exit_ms BIGINT,
    entry DOUBLE, exit DOUBLE, sl DOUBLE,
    size DOUBLE, pnl DOUBLE, pnl_pct DOUBLE,
    status VARCHAR
);
"""


def connect():
    con = duckdb.connect(str(DUCKDB_PATH))
    con.execute(SCHEMA)
    return con


def parquet_path(symbol: str, kind: str = "ticks") -> Path:
    p = PARQUET_DIR / kind / symbol
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_ticks(symbol: str, df: pd.DataFrame):
    """Append ticks to partitioned parquet (by date).

    df columns: ts_ms, price, qty, is_buyer_maker
    """
    if df.empty:
        return
    df = df.copy()
    df["date"] = (pd.to_datetime(df["ts_ms"], unit="ms")
                  .dt.strftime("%Y-%m-%d"))
    base = parquet_path(symbol, "ticks")
    for date, sub in df.groupby("date"):
        out = base / f"date={date}"
        out.mkdir(exist_ok=True)
        f = out / f"{int(sub['ts_ms'].iloc[0])}.parquet"
        sub.drop(columns=["date"]).to_parquet(f, compression="zstd")


def read_bars(symbol: str, tf: str,
              start_ms: int | None = None,
              end_ms: int | None = None) -> pd.DataFrame:
    con = connect()
    q = "SELECT * FROM bars WHERE symbol=? AND tf=?"
    params = [symbol, tf]
    if start_ms is not None:
        q += " AND ts_ms >= ?"; params.append(start_ms)
    if end_ms is not None:
        q += " AND ts_ms < ?"; params.append(end_ms)
    q += " ORDER BY ts_ms"
    return con.execute(q, params).df()


def upsert_bars(symbol: str, tf: str, rows: list[tuple]):
    con = connect()
    con.executemany(
        "INSERT OR REPLACE INTO bars VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        [(symbol, tf, *r) for r in rows],
    )
