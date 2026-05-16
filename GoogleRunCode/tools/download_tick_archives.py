"""Download/import official Binance monthly aggTrades archives into tick cache."""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tick_cache import ensure_archive_ticks, is_cached


def _parse_date_ms(value: str) -> int:
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def _bars_range_ms(path: Path) -> tuple[int, int]:
    if path.suffix.lower() in {".parquet", ".pq"}:
        bars = pd.read_parquet(path, columns=["ts_ms"])
    elif path.suffix.lower() == ".csv":
        bars = pd.read_csv(path, usecols=["ts_ms"])
    else:
        raise ValueError(f"unsupported bars file type: {path.suffix}")
    start_ms = int(bars["ts_ms"].min())
    diffs = bars["ts_ms"].sort_values().diff().dropna()
    step_ms = int(diffs.median()) if not diffs.empty else 60 * 60 * 1000
    end_ms = int(bars["ts_ms"].max()) + step_ms
    return start_ms, end_ms


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Preload official Binance monthly aggTrades archives into data_store/tick_cache.duckdb"
    )
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--bars", help="Forward bars file; uses its ts_ms min/max")
    parser.add_argument("--start", help="UTC ISO date, e.g. 2024-01-01")
    parser.add_argument("--end", help="UTC ISO date, exclusive, e.g. 2025-05-01")
    args = parser.parse_args()

    if args.bars:
        start_ms, end_ms = _bars_range_ms(Path(args.bars))
    elif args.start and args.end:
        start_ms = _parse_date_ms(args.start)
        end_ms = _parse_date_ms(args.end)
    else:
        raise SystemExit("provide --bars or both --start and --end")

    start_dt = datetime.fromtimestamp(start_ms / 1000, tz=timezone.utc)
    end_dt = datetime.fromtimestamp(end_ms / 1000, tz=timezone.utc)
    print(f"preloading {args.symbol.upper()} aggTrades {start_dt.isoformat()} -> {end_dt.isoformat()}")
    if is_cached(args.symbol, start_ms, end_ms):
        print("range already cached")
        return 0
    rows = ensure_archive_ticks(args.symbol, start_ms, end_ms)
    print(f"imported rows: {rows}")
    if is_cached(args.symbol, start_ms, end_ms):
        print("range cached")
    else:
        print("archive preload incomplete; tick test will fall back to REST for remaining gaps")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
