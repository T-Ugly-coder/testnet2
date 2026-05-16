"""
Binance tick + historical data fetcher.

- Historical: REST /api/v3/aggTrades for ticks; /api/v3/klines for bars.
- Live: WebSocket @aggTrade for real-time tick stream.

For 4 years of ticks per symbol (~100GB BTCUSDT) use the official Binance
data dumps at https://data.binance.vision/?prefix=data/spot/monthly/aggTrades/
which is faster than scraping. We provide a downloader for those zips.
"""
from __future__ import annotations
import asyncio
import io
import zipfile
from pathlib import Path
from datetime import datetime, timezone
import httpx
import pandas as pd
import websockets
import orjson

from ..config import CFG, PARQUET_DIR


BINANCE_REST = "https://api.binance.com"
BINANCE_WS = "wss://stream.binance.com:9443/ws"
BINANCE_DUMP = "https://data.binance.vision/data/spot/monthly/aggTrades"


# ---------------------------------------------------------------------------
# Historical tick dumps from data.binance.vision
# ---------------------------------------------------------------------------
async def download_monthly_ticks(symbol: str, year: int, month: int,
                                 client: httpx.AsyncClient) -> pd.DataFrame:
    fname = f"{symbol}-aggTrades-{year:04d}-{month:02d}.zip"
    url = f"{BINANCE_DUMP}/{symbol}/{fname}"
    r = await client.get(url, timeout=300.0)
    r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        with z.open(z.namelist()[0]) as f:
            df = pd.read_csv(f, header=None, names=[
                "agg_id", "price", "qty", "first_id", "last_id",
                "ts_ms", "is_buyer_maker", "best_match",
            ])
    return df[["ts_ms", "price", "qty", "is_buyer_maker"]]


async def backfill_ticks(symbol: str, start_year: int, end_year: int,
                         dest: Path | None = None,
                         max_concurrency: int = 4):
    dest = dest or (PARQUET_DIR / "ticks" / symbol)
    dest.mkdir(parents=True, exist_ok=True)
    sem = asyncio.Semaphore(max_concurrency)
    async with httpx.AsyncClient() as client:
        async def one(y, m):
            async with sem:
                out = dest / f"{y:04d}-{m:02d}.parquet"
                if out.exists():
                    return
                try:
                    df = await download_monthly_ticks(symbol, y, m, client)
                    df.to_parquet(out, compression="zstd")
                except Exception as e:
                    print(f"skip {symbol} {y}-{m}: {e}")
        tasks = []
        for y in range(start_year, end_year + 1):
            for m in range(1, 13):
                tasks.append(one(y, m))
        await asyncio.gather(*tasks)


# ---------------------------------------------------------------------------
# REST klines (bars) — cheap, paged
# ---------------------------------------------------------------------------
async def fetch_klines(symbol: str, interval: str,
                       start_ms: int, end_ms: int,
                       client: httpx.AsyncClient,
                       limit: int = 1000) -> pd.DataFrame:
    rows = []
    cur = start_ms
    while cur < end_ms:
        r = await client.get(f"{BINANCE_REST}/api/v3/klines", params={
            "symbol": symbol, "interval": interval,
            "startTime": cur, "endTime": end_ms, "limit": limit,
        })
        r.raise_for_status()
        data = r.json()
        if not data:
            break
        rows.extend(data)
        cur = data[-1][0] + 1
        if len(data) < limit:
            break
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=[
        "ts_ms", "open", "high", "low", "close", "volume",
        "close_ts", "quote_volume", "n_trades",
        "taker_buy_base", "taker_buy_quote", "_ignore",
    ])
    for c in ("open", "high", "low", "close", "volume",
              "quote_volume", "taker_buy_base", "taker_buy_quote"):
        df[c] = df[c].astype(float)
    df["n_trades"] = df["n_trades"].astype(int)
    return df.drop(columns=["close_ts", "_ignore"])


# ---------------------------------------------------------------------------
# Live tick stream via WebSocket
# ---------------------------------------------------------------------------
async def stream_aggtrades(symbol: str, queue: asyncio.Queue):
    url = f"{BINANCE_WS}/{symbol.lower()}@aggTrade"
    while True:
        try:
            async with websockets.connect(url, ping_interval=20) as ws:
                async for msg in ws:
                    d = orjson.loads(msg)
                    await queue.put({
                        "ts_ms": d["T"],
                        "price": float(d["p"]),
                        "qty": float(d["q"]),
                        "is_buyer_maker": d["m"],
                    })
        except Exception as e:
            print(f"ws reconnect {symbol}: {e}")
            await asyncio.sleep(2)
