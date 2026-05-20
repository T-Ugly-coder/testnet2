"""Central config — no cron, all timing handled by event loop."""
from __future__ import annotations
import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data_store"
DATA_DIR.mkdir(exist_ok=True)

PARQUET_DIR = DATA_DIR / "parquet"
PARQUET_DIR.mkdir(exist_ok=True)

DUCKDB_PATH = DATA_DIR / "trading.duckdb"


@dataclass
class HardwareConfig:
    cpu_cores: int = 16
    cpu_threads: int = 32
    gpu_vram_gb: int = 8
    use_gpu: bool = True
    numba_threads: int = 32


@dataclass
class DataConfig:
    symbols: tuple[str, ...] = ("BTCUSDT", "ETHUSDT", "XAUUSD")
    crypto_symbols: tuple[str, ...] = ("BTCUSDT", "ETHUSDT")
    timeframes: tuple[str, ...] = ("1m", "5m", "15m", "1h", "4h", "1d", "1w")
    use_tick_data: bool = True
    tick_storage: str = "parquet"
    history_years: int = 4
    walkforward_years: int = 4


@dataclass
class BacktestConfig:
    initial_balance: float = 100_000.0
    risk_per_trade: float = 0.01
    fee_bps: float = 4.0
    slippage_bps: float = 1.5
    # Single-symbol strategy runs should not pyramid by default. Raising this
    # is an explicit portfolio/pyramiding choice, not the optimizer baseline.
    max_concurrent_trades: int = 1
    train_years: int = 4
    test_years: int = 4
    purge_days: int = 2
    embargo_days: int = 1
    n_splits: int = 6


@dataclass
class LLMConfig:
    backend: str = "llama_cpp"
    model_path: str = os.environ.get(
        "LLM_MODEL_PATH",
        str(DATA_DIR / "models" / "llama-3.1-8b-instruct-q4_k_m.gguf"),
    )
    n_gpu_layers: int = -1
    n_ctx: int = 8192
    n_threads: int = 16


@dataclass
class Config:
    hw: HardwareConfig = field(default_factory=HardwareConfig)
    data: DataConfig = field(default_factory=DataConfig)
    bt: BacktestConfig = field(default_factory=BacktestConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)


CFG = Config()
