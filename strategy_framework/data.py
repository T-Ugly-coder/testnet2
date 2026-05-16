"""Market-data download and parquet preparation utilities."""
from __future__ import annotations

import asyncio
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import pandas as pd


BINANCE_REST = "https://api.binance.com"
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


@dataclass(frozen=True)
class DownloadResult:
    symbol: str
    timeframe: str
    rows: int
    start_ms: int
    end_ms: int
    full_path: str
    backtest_path: str
    forward_path: str | None


def parse_utc(value: str) -> datetime:
    text = value.strip()
    if len(text) == 10:
        return datetime.strptime(text, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def to_ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


def prepare_bars(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values("ts_ms").drop_duplicates("ts_ms").copy()
    out.index = pd.to_datetime(out["ts_ms"], unit="ms", utc=True)
    out.index.name = "ts"
    return out


def infer_label(start: datetime, end: datetime, years: int | None) -> str:
    if years:
        return f"{years}y"
    days = max(1, int(round((end - start).total_seconds() / 86_400.0)))
    return f"{days}d"


async def fetch_bars(symbol: str, timeframe: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    if timeframe not in INTERVAL_MS:
        raise ValueError(f"unsupported timeframe: {timeframe}")
    rows = []
    cur = start_ms
    async with httpx.AsyncClient() as client:
        while cur < end_ms:
            response = await client.get(
                f"{BINANCE_REST}/api/v3/klines",
                params={
                    "symbol": symbol,
                    "interval": timeframe,
                    "startTime": cur,
                    "endTime": end_ms - 1,
                    "limit": 1000,
                },
                timeout=300.0,
            )
            response.raise_for_status()
            data = response.json()
            if not data:
                break
            rows.extend(data)
            cur = int(data[-1][0]) + 1
            if len(data) < 1000:
                break
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(
        rows,
        columns=[
            "ts_ms",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_ts",
            "quote_volume",
            "n_trades",
            "taker_buy_base",
            "taker_buy_quote",
            "_ignore",
        ],
    )
    numeric_cols = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_volume",
        "taker_buy_base",
        "taker_buy_quote",
    ]
    for col in numeric_cols:
        df[col] = df[col].astype(float)
    df["ts_ms"] = df["ts_ms"].astype("int64")
    df["n_trades"] = df["n_trades"].astype("int64")
    df = df.drop(columns=["close_ts", "_ignore"])
    return prepare_bars(df)


async def download_timeframe(
    *,
    symbol: str,
    timeframe: str,
    out_dir: Path,
    start: datetime,
    end: datetime,
    label: str,
    forward_frac: float,
    refresh: bool = False,
) -> DownloadResult:
    symbol = symbol.upper()
    out_dir.mkdir(parents=True, exist_ok=True)
    full_path = out_dir / f"{symbol}_{timeframe}_full_{label}.parquet"
    backtest_path = out_dir / f"{symbol}_{timeframe}_backtest_{label}.parquet"
    forward_path = (
        out_dir / f"{symbol}_{timeframe}_forward_{label}.parquet"
        if forward_frac > 0.0
        else None
    )
    start_ms = to_ms(start)
    end_ms = to_ms(end)
    if start_ms >= end_ms:
        raise ValueError("start must be before end")

    if not refresh and full_path.exists() and backtest_path.exists() and (
        forward_path is None or forward_path.exists()
    ):
        existing = pd.read_parquet(full_path)
        return DownloadResult(
            symbol=symbol,
            timeframe=timeframe,
            rows=int(len(existing)),
            start_ms=start_ms,
            end_ms=end_ms,
            full_path=str(full_path),
            backtest_path=str(backtest_path),
            forward_path=str(forward_path) if forward_path else None,
        )

    bars = await fetch_bars(symbol, timeframe, start_ms, end_ms)
    if bars.empty:
        raise RuntimeError(f"no bars returned for {symbol} {timeframe}")
    bars.to_parquet(full_path)

    split = len(bars)
    if forward_frac > 0.0:
        if not 0.0 < forward_frac < 1.0:
            raise ValueError("forward_frac must be in (0,1), or 0 to disable")
        split = max(1, min(len(bars) - 1, int(len(bars) * (1.0 - forward_frac))))
    bars.iloc[:split].to_parquet(backtest_path)
    if forward_path is not None:
        bars.iloc[split:].to_parquet(forward_path)

    return DownloadResult(
        symbol=symbol,
        timeframe=timeframe,
        rows=int(len(bars)),
        start_ms=start_ms,
        end_ms=end_ms,
        full_path=str(full_path),
        backtest_path=str(backtest_path),
        forward_path=str(forward_path) if forward_path else None,
    )


async def download_many(
    *,
    symbol: str,
    timeframes: list[str],
    out_dir: Path,
    years: int | None = None,
    start_value: str | None = None,
    end_value: str | None = None,
    label: str | None = None,
    forward_frac: float = 0.30,
    refresh: bool = False,
) -> list[DownloadResult]:
    end = parse_utc(end_value) if end_value else datetime.now(timezone.utc)
    start = parse_utc(start_value) if start_value else end - timedelta(days=365 * int(years or 4))
    run_label = label or infer_label(start, end, years)
    results = []
    for timeframe in timeframes:
        results.append(
            await download_timeframe(
                symbol=symbol,
                timeframe=timeframe,
                out_dir=out_dir,
                start=start,
                end=end,
                label=run_label,
                forward_frac=forward_frac,
                refresh=refresh,
            )
        )
    manifest = {
        "symbol": symbol.upper(),
        "timeframes": timeframes,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "label": run_label,
        "forward_frac": forward_frac,
        "results": [asdict(result) for result in results],
    }
    (out_dir / f"{symbol.upper()}_{run_label}_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    return results


def download_many_sync(**kwargs) -> list[DownloadResult]:
    return asyncio.run(download_many(**kwargs))
