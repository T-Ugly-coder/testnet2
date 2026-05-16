"""On-demand Binance aggTrade cache for tick-level validation.

This intentionally does not download years of ticks up front.  Callers ask for
specific time windows, usually the entry-to-exit windows of candidate trades,
and only those aggTrades are cached locally.
"""
from __future__ import annotations

import calendar
import io
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import httpx
import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data_store"
DATA_DIR.mkdir(exist_ok=True)
TICK_DB_PATH = DATA_DIR / "tick_cache.duckdb"
BINANCE_REST = "https://api.binance.com"
BINANCE_ARCHIVE = "https://data.binance.vision"
ARCHIVE_DIR = DATA_DIR / "binance_archive" / "spot" / "monthly" / "aggTrades"
ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)


SCHEMA = """
CREATE TABLE IF NOT EXISTS agg_trades (
    symbol VARCHAR NOT NULL,
    agg_id BIGINT NOT NULL,
    ts_ms BIGINT NOT NULL,
    price DOUBLE NOT NULL,
    qty DOUBLE NOT NULL,
    is_buyer_maker BOOLEAN NOT NULL,
    PRIMARY KEY (symbol, agg_id)
);

CREATE INDEX IF NOT EXISTS idx_agg_trades_symbol_ts
ON agg_trades(symbol, ts_ms);

CREATE TABLE IF NOT EXISTS cached_tick_ranges (
    symbol VARCHAR NOT NULL,
    start_ms BIGINT NOT NULL,
    end_ms BIGINT NOT NULL,
    created_ms BIGINT NOT NULL
);

CREATE TABLE IF NOT EXISTS imported_tick_archives (
    symbol VARCHAR NOT NULL,
    archive_name VARCHAR NOT NULL,
    start_ms BIGINT NOT NULL,
    end_ms BIGINT NOT NULL,
    rows_imported BIGINT NOT NULL,
    created_ms BIGINT NOT NULL,
    PRIMARY KEY (symbol, archive_name)
);
"""


def connect() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(str(TICK_DB_PATH))
    con.execute(SCHEMA)
    return con


def _covered_by_ranges(ranges: list[tuple[int, int]], start_ms: int, end_ms: int) -> bool:
    if start_ms >= end_ms:
        return True
    cursor = start_ms
    for start, end in sorted(ranges):
        if end <= cursor:
            continue
        if start > cursor:
            return False
        cursor = max(cursor, end)
        if cursor >= end_ms:
            return True
    return False


def is_cached(symbol: str, start_ms: int, end_ms: int) -> bool:
    con = connect()
    rows = con.execute(
        """
        SELECT start_ms, end_ms
        FROM cached_tick_ranges
        WHERE symbol = ? AND end_ms > ? AND start_ms < ?
        ORDER BY start_ms
        """,
        [symbol.upper(), start_ms, end_ms],
    ).fetchall()
    con.close()
    return _covered_by_ranges([(int(a), int(b)) for a, b in rows], start_ms, end_ms)


def read_ticks(symbol: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    con = connect()
    df = con.execute(
        """
        SELECT ts_ms, price, qty, is_buyer_maker
        FROM agg_trades
        WHERE symbol = ? AND ts_ms >= ? AND ts_ms < ?
        ORDER BY ts_ms, agg_id
        """,
        [symbol.upper(), start_ms, end_ms],
    ).df()
    con.close()
    return df


def read_first_tick(symbol: str, start_ms: int, end_ms: int) -> dict | None:
    con = connect()
    row = con.execute(
        """
        SELECT ts_ms, agg_id, price, qty, is_buyer_maker
        FROM agg_trades
        WHERE symbol = ? AND ts_ms >= ? AND ts_ms < ?
        ORDER BY ts_ms, agg_id
        LIMIT 1
        """,
        [symbol.upper(), start_ms, end_ms],
    ).fetchone()
    con.close()
    if row is None:
        return None
    return {
        "ts_ms": int(row[0]),
        "agg_id": int(row[1]),
        "price": float(row[2]),
        "qty": float(row[3]),
        "is_buyer_maker": bool(row[4]),
    }


def read_last_tick(symbol: str, start_ms: int, end_ms: int) -> dict | None:
    con = connect()
    row = con.execute(
        """
        SELECT ts_ms, agg_id, price, qty, is_buyer_maker
        FROM agg_trades
        WHERE symbol = ? AND ts_ms >= ? AND ts_ms < ?
        ORDER BY ts_ms DESC, agg_id DESC
        LIMIT 1
        """,
        [symbol.upper(), start_ms, end_ms],
    ).fetchone()
    con.close()
    if row is None:
        return None
    return {
        "ts_ms": int(row[0]),
        "agg_id": int(row[1]),
        "price": float(row[2]),
        "qty": float(row[3]),
        "is_buyer_maker": bool(row[4]),
    }


def read_first_price_cross(
    symbol: str,
    start_ms: int,
    end_ms: int,
    *,
    op: str,
    level: float,
) -> dict | None:
    if op not in {"<=", ">="}:
        raise ValueError("op must be '<=' or '>='")
    con = connect()
    row = con.execute(
        f"""
        SELECT ts_ms, agg_id, price, qty, is_buyer_maker
        FROM agg_trades
        WHERE symbol = ? AND ts_ms >= ? AND ts_ms < ? AND price {op} ?
        ORDER BY ts_ms, agg_id
        LIMIT 1
        """,
        [symbol.upper(), start_ms, end_ms, float(level)],
    ).fetchone()
    con.close()
    if row is None:
        return None
    return {
        "ts_ms": int(row[0]),
        "agg_id": int(row[1]),
        "price": float(row[2]),
        "qty": float(row[3]),
        "is_buyer_maker": bool(row[4]),
    }


def _store_chunk(symbol: str, rows: list[dict]) -> int:
    if not rows:
        return 0
    df = pd.DataFrame(rows)
    df.insert(0, "symbol", symbol.upper())
    con = connect()
    con.register("new_ticks", df)
    con.execute(
        """
        INSERT OR IGNORE INTO agg_trades
        SELECT symbol, agg_id, ts_ms, price, qty, is_buyer_maker
        FROM new_ticks
        """
    )
    con.unregister("new_ticks")
    con.close()
    return len(df)


def _mark_cached(symbol: str, start_ms: int, end_ms: int) -> None:
    con = connect()
    con.execute(
        "INSERT INTO cached_tick_ranges VALUES (?, ?, ?, ?)",
        [symbol.upper(), start_ms, end_ms, int(time.time() * 1000)],
    )
    con.close()


def _month_start_ms(year: int, month: int) -> int:
    return int(datetime(year, month, 1, tzinfo=timezone.utc).timestamp() * 1000)


def _next_month(year: int, month: int) -> tuple[int, int]:
    if month == 12:
        return year + 1, 1
    return year, month + 1


def _month_end_ms(year: int, month: int) -> int:
    next_year, next_month = _next_month(year, month)
    return _month_start_ms(next_year, next_month)


def _months_between(start_ms: int, end_ms: int) -> list[tuple[int, int]]:
    if start_ms >= end_ms:
        return []
    start = datetime.fromtimestamp(start_ms / 1000, tz=timezone.utc)
    end = datetime.fromtimestamp((end_ms - 1) / 1000, tz=timezone.utc)
    year, month = start.year, start.month
    months = []
    while (year, month) <= (end.year, end.month):
        months.append((year, month))
        year, month = _next_month(year, month)
    return months


def _archive_url(symbol: str, year: int, month: int) -> str:
    symbol = symbol.upper()
    filename = f"{symbol}-aggTrades-{year:04d}-{month:02d}.zip"
    return f"{BINANCE_ARCHIVE}/data/spot/monthly/aggTrades/{symbol}/{filename}"


def _archive_path(symbol: str, year: int, month: int) -> Path:
    symbol = symbol.upper()
    return ARCHIVE_DIR / symbol / f"{symbol}-aggTrades-{year:04d}-{month:02d}.zip"


def _is_archive_imported(symbol: str, archive_name: str) -> bool:
    con = connect()
    row = con.execute(
        """
        SELECT 1
        FROM imported_tick_archives
        WHERE symbol = ? AND archive_name = ?
        LIMIT 1
        """,
        [symbol.upper(), archive_name],
    ).fetchone()
    con.close()
    return row is not None


def _mark_archive_imported(
    symbol: str,
    archive_name: str,
    start_ms: int,
    end_ms: int,
    rows_imported: int,
) -> None:
    con = connect()
    con.execute(
        """
        INSERT OR REPLACE INTO imported_tick_archives
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        [
            symbol.upper(),
            archive_name,
            start_ms,
            end_ms,
            int(rows_imported),
            int(time.time() * 1000),
        ],
    )
    con.close()


def download_monthly_archive(
    symbol: str,
    year: int,
    month: int,
    *,
    client: httpx.Client | None = None,
) -> Path | None:
    """Download one official Binance monthly aggTrades zip, if it exists."""
    symbol = symbol.upper()
    path = _archive_path(symbol, year, month)
    if path.exists() and path.stat().st_size > 0:
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    own_client = client is None
    client = client or httpx.Client(timeout=120.0, follow_redirects=True)
    tmp = path.with_suffix(path.suffix + ".part")
    try:
        url = _archive_url(symbol, year, month)
        with client.stream("GET", url) as response:
            if response.status_code == 404:
                return None
            response.raise_for_status()
            with tmp.open("wb") as f:
                for chunk in response.iter_bytes():
                    if chunk:
                        f.write(chunk)
        tmp.replace(path)
        return path
    finally:
        if tmp.exists():
            tmp.unlink()
        if own_client:
            client.close()


def _store_archive_frame(symbol: str, df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    out = pd.DataFrame({
        "symbol": symbol.upper(),
        "agg_id": pd.to_numeric(df["agg_id"], errors="coerce").astype("Int64"),
        "ts_ms": pd.to_numeric(df["ts_ms"], errors="coerce").astype("Int64"),
        "price": pd.to_numeric(df["price"], errors="coerce"),
        "qty": pd.to_numeric(df["qty"], errors="coerce"),
        "is_buyer_maker": df["is_buyer_maker"].astype(str).str.lower().isin({"true", "1"}),
    }).dropna(subset=["agg_id", "ts_ms", "price", "qty"])
    if out.empty:
        return 0
    out["agg_id"] = out["agg_id"].astype("int64")
    out["ts_ms"] = out["ts_ms"].astype("int64")
    con = connect()
    con.register("archive_ticks", out)
    con.execute(
        """
        INSERT OR IGNORE INTO agg_trades
        SELECT symbol, agg_id, ts_ms, price, qty, is_buyer_maker
        FROM archive_ticks
        """
    )
    con.unregister("archive_ticks")
    con.close()
    return len(out)


def import_monthly_archive(symbol: str, path: Path, start_ms: int, end_ms: int) -> int:
    """Import one Binance monthly aggTrades zip into DuckDB."""
    symbol = symbol.upper()
    archive_name = path.name
    if _is_archive_imported(symbol, archive_name):
        _mark_cached(symbol, start_ms, end_ms)
        return 0

    names = [
        "agg_id",
        "price",
        "qty",
        "first_trade_id",
        "last_trade_id",
        "ts_ms",
        "is_buyer_maker",
        "is_best_match",
    ]
    rows = 0
    with zipfile.ZipFile(path) as zf:
        csv_names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not csv_names:
            raise ValueError(f"{path} does not contain a CSV file")
        with zf.open(csv_names[0]) as raw:
            prefix = raw.peek(64)[:64].decode("utf-8", errors="ignore").lower()
            has_header = "agg" in prefix and "price" in prefix
            text = io.TextIOWrapper(raw, encoding="utf-8", newline="")
            reader = pd.read_csv(
                text,
                header=0 if has_header else None,
                names=None if has_header else names,
                chunksize=500_000,
            )
            for chunk in reader:
                if has_header:
                    chunk = chunk.rename(columns={
                        "aggregate_trade_id": "agg_id",
                        "agg_trade_id": "agg_id",
                        "a": "agg_id",
                        "p": "price",
                        "price": "price",
                        "q": "qty",
                        "quantity": "qty",
                        "T": "ts_ms",
                        "transact_time": "ts_ms",
                        "timestamp": "ts_ms",
                        "m": "is_buyer_maker",
                        "is_buyer_maker": "is_buyer_maker",
                    })
                rows += _store_archive_frame(symbol, chunk)

    _mark_archive_imported(symbol, archive_name, start_ms, end_ms, rows)
    _mark_cached(symbol, start_ms, end_ms)
    return rows


def ensure_archive_ticks(
    symbol: str,
    start_ms: int,
    end_ms: int,
    *,
    client: httpx.Client | None = None,
) -> int:
    """Download/import official Binance monthly aggTrades archives for a range."""
    symbol = symbol.upper()
    own_client = client is None
    client = client or httpx.Client(timeout=120.0, follow_redirects=True)
    imported = 0
    try:
        for year, month in _months_between(start_ms, end_ms):
            month_start = _month_start_ms(year, month)
            month_end = _month_end_ms(year, month)
            if month_end <= start_ms or month_start >= end_ms:
                continue
            if is_cached(symbol, month_start, month_end):
                continue
            path = download_monthly_archive(symbol, year, month, client=client)
            if path is None:
                continue
            print(f"importing archive {path.name}")
            imported += import_monthly_archive(symbol, path, month_start, month_end)
    finally:
        if own_client:
            client.close()
    return imported


def ensure_ticks(
    symbol: str,
    start_ms: int,
    end_ms: int,
    *,
    client: httpx.Client | None = None,
    request_pause: float = 0.05,
    max_requests: int = 25_000,
    chunk_ms: int = 6 * 60 * 60 * 1000,
    use_archive: bool = True,
) -> int:
    """Ensure aggTrades for [start_ms, end_ms) exist in the local cache.

    Returns the number of rows received from Binance, including rows that may
    already exist locally.  This keeps repeated validations cheap while avoiding
    a giant all-history tick download.
    """
    symbol = symbol.upper()
    if start_ms >= end_ms or is_cached(symbol, start_ms, end_ms):
        return 0

    own_client = client is None
    client = client or httpx.Client(timeout=30.0)

    if use_archive:
        archived = ensure_archive_ticks(symbol, start_ms, end_ms, client=client)
        if is_cached(symbol, start_ms, end_ms):
            if own_client:
                client.close()
            return archived

    if chunk_ms > 0 and end_ms - start_ms > chunk_ms:
        fetched_total = 0
        cursor = start_ms
        try:
            while cursor < end_ms:
                chunk_end = min(cursor + chunk_ms, end_ms)
                fetched_total += ensure_ticks(
                    symbol,
                    cursor,
                    chunk_end,
                    client=client,
                    request_pause=request_pause,
                    max_requests=max_requests,
                    chunk_ms=0,
                    use_archive=False,
                )
                cursor = chunk_end
            return fetched_total
        finally:
            if own_client:
                client.close()

    fetched = 0
    requests = 0
    from_id: int | None = None

    try:
        while requests < max_requests:
            if from_id is None:
                params = {
                    "symbol": symbol,
                    "startTime": start_ms,
                    "endTime": end_ms - 1,
                    "limit": 1000,
                }
            else:
                params = {
                    "symbol": symbol,
                    "fromId": from_id,
                    "limit": 1000,
                }

            response = client.get(f"{BINANCE_REST}/api/v3/aggTrades", params=params)
            response.raise_for_status()
            payload = response.json()
            requests += 1
            if not payload:
                break

            rows = []
            stop = False
            for item in payload:
                ts_ms = int(item["T"])
                if ts_ms >= end_ms:
                    stop = True
                    break
                if ts_ms >= start_ms:
                    rows.append({
                        "agg_id": int(item["a"]),
                        "ts_ms": ts_ms,
                        "price": float(item["p"]),
                        "qty": float(item["q"]),
                        "is_buyer_maker": bool(item["m"]),
                    })

            fetched += _store_chunk(symbol, rows)
            from_id = int(payload[-1]["a"]) + 1

            if stop or len(payload) < 1000:
                break
            if request_pause > 0:
                time.sleep(request_pause)
        else:
            raise RuntimeError(
                f"hit max_requests={max_requests} while fetching {symbol} ticks"
            )

        _mark_cached(symbol, start_ms, end_ms)
        return fetched
    finally:
        if own_client:
            client.close()


def get_ticks(symbol: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    ensure_ticks(symbol, start_ms, end_ms)
    return read_ticks(symbol, start_ms, end_ms)
