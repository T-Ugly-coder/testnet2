"""Download Binance OHLC candles into the local DuckDB cache.

This script is intentionally kept outside ``algo_download`` so the downloaded
source remains untouched.  It uses the package's existing Binance fetcher and
DuckDB storage helpers.
"""
from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import pandas as pd

from algo.config import DUCKDB_PATH
from algo.data.binance_tick import fetch_klines
from algo.data.storage import read_bars, upsert_bars


INTERVAL_MS = {
    "1m": 60_000,
    "3m": 3 * 60_000,
    "5m": 5 * 60_000,
    "15m": 15 * 60_000,
    "30m": 30 * 60_000,
    "1h": 60 * 60_000,
    "2h": 2 * 60 * 60_000,
    "4h": 4 * 60 * 60_000,
    "6h": 6 * 60 * 60_000,
    "8h": 8 * 60 * 60_000,
    "12h": 12 * 60 * 60_000,
    "1d": 24 * 60 * 60_000,
    "1w": 7 * 24 * 60 * 60_000,
}


def parse_utc(value: str) -> datetime:
    """Parse YYYY-MM-DD or ISO datetime as UTC."""
    text = value.strip()
    if len(text) == 10:
        dt = datetime.strptime(text, "%Y-%m-%d")
        return dt.replace(tzinfo=timezone.utc)
    dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def to_ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


def rows_for_storage(df: pd.DataFrame) -> list[tuple]:
    cols = [
        "ts_ms",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_volume",
        "n_trades",
        "taker_buy_base",
        "taker_buy_quote",
    ]
    return [tuple(row) for row in df[cols].itertuples(index=False, name=None)]


def prepare_export(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values("ts_ms").copy()
    out.index = pd.to_datetime(out["ts_ms"], unit="ms", utc=True)
    out.index.name = "ts"
    return out


def missing_ranges(
    symbol: str,
    tf: str,
    start_ms: int,
    end_ms: int,
    step_ms: int,
    refresh: bool,
) -> list[tuple[int, int]]:
    if refresh:
        return [(start_ms, end_ms)]

    cached = read_bars(symbol, tf, start_ms, end_ms)
    if cached.empty:
        return [(start_ms, end_ms)]

    cached_start = int(cached["ts_ms"].min())
    cached_end = int(cached["ts_ms"].max())
    ranges: list[tuple[int, int]] = []

    if start_ms < cached_start:
        ranges.append((start_ms, cached_start))

    next_after_cache = cached_end + step_ms
    if next_after_cache < end_ms:
        ranges.append((next_after_cache, end_ms))

    return ranges


async def download_one(
    symbol: str,
    tf: str,
    start_ms: int,
    end_ms: int,
    refresh: bool,
) -> int:
    step_ms = INTERVAL_MS[tf]
    ranges = missing_ranges(symbol, tf, start_ms, end_ms, step_ms, refresh)
    if not ranges:
        print(f"{symbol} {tf}: already cached")
        return 0

    total = 0
    async with httpx.AsyncClient() as client:
        for range_start, range_end in ranges:
            print(f"{symbol} {tf}: downloading {range_start} -> {range_end}")
            # Binance endTime is inclusive; this script treats --end as
            # exclusive so repeated date-bounded runs are predictable.
            df = await fetch_klines(symbol, tf, range_start, range_end - 1, client)
            if df.empty:
                print(f"{symbol} {tf}: no rows returned")
                continue
            upsert_bars(symbol, tf, rows_for_storage(df))
            total += len(df)
            print(f"{symbol} {tf}: cached {len(df)} rows")
    return total


async def run(args: argparse.Namespace) -> int:
    end_dt = parse_utc(args.end) if args.end else datetime.now(timezone.utc)
    start_dt = (
        parse_utc(args.start)
        if args.start
        else end_dt - timedelta(days=365 * args.years)
    )
    start_ms = to_ms(start_dt)
    end_ms = to_ms(end_dt)

    if start_ms >= end_ms:
        raise ValueError("--start must be before --end")

    total = 0
    for symbol in args.symbol:
        for tf in args.tf:
            total += await download_one(
                symbol=symbol.upper(),
                tf=tf,
                start_ms=start_ms,
                end_ms=end_ms,
                refresh=args.refresh,
            )

    if args.out:
        if len(args.symbol) != 1 or len(args.tf) != 1:
            raise ValueError("--out can only be used with one --symbol and one --tf")
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cached = prepare_export(
            read_bars(args.symbol[0].upper(), args.tf[0], start_ms, end_ms)
        )
        if out_path.suffix.lower() in {".parquet", ".pq"}:
            cached.to_parquet(out_path)
        elif out_path.suffix.lower() == ".csv":
            cached.to_csv(out_path, index=False)
        else:
            raise ValueError("--out must end with .parquet, .pq, or .csv")
        print(f"Exported {len(cached)} cached rows to {out_path}")

    print(f"Done. Stored {total} new/updated rows in {DUCKDB_PATH}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download Binance candles into data_store/trading.duckdb"
    )
    parser.add_argument(
        "--symbol",
        nargs="+",
        default=["BTCUSDT", "ETHUSDT"],
        help="Binance symbol(s), for example BTCUSDT ETHUSDT",
    )
    parser.add_argument(
        "--tf",
        nargs="+",
        default=["1h"],
        choices=sorted(INTERVAL_MS),
        help="Binance interval(s), for example 1m 5m 1h 1d",
    )
    parser.add_argument("--start", help="UTC start date, e.g. 2024-01-01")
    parser.add_argument("--end", help="UTC end date, e.g. 2024-12-31")
    parser.add_argument(
        "--years",
        type=int,
        default=4,
        help="Years of history to fetch when --start is not provided",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Force redownload for the requested range",
    )
    parser.add_argument(
        "--out",
        help="Optional .parquet/.pq/.csv export for the requested cached range",
    )
    args = parser.parse_args()
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
