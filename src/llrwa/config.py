"""Project paths and API-key access. Keys live only in the git-ignored config.toml.

Output paths depend on the selected study (LLRWA_STUDY, see study.py); raw data is shared.
"""
from __future__ import annotations

import tomllib
from pathlib import Path

from .study import current

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "config.toml"
DATA_RAW = ROOT / "data" / "raw"
HL_WS_RAW = DATA_RAW / "hyperliquid_ws.nosync" / "raw"
STUDY = current()
DATA_CLEAN = ROOT / "data" / "clean" / STUDY.subdir if STUDY.subdir else ROOT / "data" / "clean"
DATA_CURVES = ROOT / "data" / "curves" / STUDY.subdir if STUDY.subdir else ROOT / "data" / "curves"
RESULTS = ROOT / "results" / STUDY.subdir if STUDY.subdir else ROOT / "results"


def api_key(section: str, path: Path = CONFIG) -> str:
    """Return `[section] api_key` from config.toml. Raise a clear error if it is missing."""
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: copy config.example.toml to config.toml and add your keys")
    cfg = tomllib.loads(path.read_text())
    key = str(cfg.get(section, {}).get("api_key", "")).strip()
    if not key or key.startswith("YOUR_"):
        raise KeyError(f"no api_key set in [{section}] of {path}")
    return key
