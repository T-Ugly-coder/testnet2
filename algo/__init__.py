"""Compatibility package for code downloaded into ``algo_download``.

The source was originally authored as package ``algo``.  Keeping this shim lets
imports such as ``algo.backtest.engine`` work while the files remain in the
download destination directory.
"""
from __future__ import annotations

from pathlib import Path

_SOURCE = Path(__file__).resolve().parent.parent / "algo_download"

# Make submodule imports (algo.backtest, algo.metrics, etc.) resolve from the
# downloaded source directory.
__path__ = [str(_SOURCE)]

