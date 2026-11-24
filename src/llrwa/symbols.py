"""Instrument table: one curated perpetual per (asset, venue), plus public-API cross-checks.

The table lives in `instruments.csv` as data. This module refreshes its numeric fields
(tick size, contract size, listing date) from the venues' public APIs and validates it.
Blind alias matching is not trusted: the ticker "SPX" is the SPX6900 memecoin on all four
venues (see DECISIONS.md D1), gold also exists as crypto tokens (PAXG, XAUT), and "MUU" is a
leveraged Micron ETF (D27).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests

INSTRUMENTS_CSV = Path(__file__).with_name("instruments.csv")
ASSETS = ("WTI", "GOLD", "SNDK", "MU")
# display names: plain (figures) and LaTeX (paper tables)
ASSET_NAMES = {"WTI": "WTI", "GOLD": "Gold", "SNDK": "SanDisk", "MU": "Micron"}
ASSET_NAMES_TEX = {a: n.replace("&", "\\&") for a, n in ASSET_NAMES.items()}
VENUES = ("binance", "okx", "kraken", "hyperliquid")
# (venue, native symbol) pairs that must never be selected: memecoins sharing a TradFi alias,
# gold tokens (crypto assets, not gold perps) and the 2x leveraged Micron ETF
FORBIDDEN = {("binance", "SPXUSDT"), ("okx", "SPX-USDT-SWAP"), ("kraken", "PF_SPXUSD"),
             ("hyperliquid", "SPX"),
             ("binance", "PAXGUSDT"), ("binance", "XAUTUSDT"), ("kraken", "PF_PAXGUSD"),
             ("kraken", "PF_XAUTUSD"), ("hyperliquid", "PAXG"),
             ("binance", "MUUUSDT"), ("okx", "MUU-USDT-SWAP")}

_UA = {"User-Agent": "research"}


def load_instruments(path: Path = INSTRUMENTS_CSV) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    for c in ("contract_size", "tick_size"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def instrument(asset: str, venue: str, df: pd.DataFrame | None = None) -> pd.Series:
    df = load_instruments() if df is None else df
    row = df[(df["asset"] == asset) & (df["venue"] == venue)]
    if len(row) != 1:
        raise KeyError((asset, venue))
    return row.iloc[0]


def validate_offline(df: pd.DataFrame) -> list[str]:
    """Structural checks that need no network. Returns a list of problems."""
    problems = []
    cells = set(zip(df["asset"], df["venue"]))
    for a in ASSETS:
        for v in VENUES:
            if (a, v) not in cells:
                problems.append(f"missing cell {a}/{v}")
    if len(cells) != len(df):
        problems.append("duplicate (asset, venue) rows")
    for _, r in df.iterrows():
        if (r["venue"], r["native_symbol"]) in FORBIDDEN:
            problems.append(f"forbidden memecoin instrument {r['venue']}:{r['native_symbol']}")
        for c in ("native_symbol", "asset_type", "tardis_exchange", "tardis_symbol", "amber_exchange"):
            if not str(r[c]).strip():
                problems.append(f"{r['asset']}/{r['venue']}: empty {c}")
    return problems


# ---------------------------------------------------------------- public-API scan (network)

def scan_public(timeout: float = 30.0) -> pd.DataFrame:
    """Contract specs for our native symbols from the venues' public APIs."""
    s = requests.Session()
    s.headers.update(_UA)
    rows = []
    bi = s.get("https://fapi.binance.com/fapi/v1/exchangeInfo", timeout=timeout).json()["symbols"]
    for x in bi:
        tick = next((f["tickSize"] for f in x["filters"] if f["filterType"] == "PRICE_FILTER"), None)
        rows.append(dict(venue="binance", native_symbol=x["symbol"], status=x["status"],
                         contract_type=x.get("contractType"), underlying_type=x.get("underlyingType"),
                         tick_size=float(tick) if tick else None, contract_size=1.0,
                         listed=pd.to_datetime(x.get("onboardDate"), unit="ms", utc=True)))
    ok = s.get("https://www.okx.com/api/v5/public/instruments", params={"instType": "SWAP"},
               timeout=timeout).json()["data"]
    for x in ok:
        rows.append(dict(venue="okx", native_symbol=x["instId"], status=x["state"],
                         contract_type="SWAP", underlying_type=f"instCategory={x.get('instCategory')}",
                         tick_size=float(x["tickSz"]), contract_size=float(x["ctVal"]),
                         listed=pd.to_datetime(int(x["listTime"]), unit="ms", utc=True)))
    kr = s.get("https://futures.kraken.com/derivatives/api/v3/instruments", timeout=timeout).json()["instruments"]
    for x in kr:
        rows.append(dict(venue="kraken", native_symbol=x["symbol"],
                         status="tradeable" if x.get("tradeable") else "not-tradeable",
                         contract_type=x.get("type"), underlying_type=x.get("category"),
                         tick_size=x.get("tickSize"), contract_size=x.get("contractSize"),
                         listed=pd.to_datetime(x.get("openingDate"), utc=True)))
    dexes = s.post("https://api.hyperliquid.xyz/info", json={"type": "perpDexs"}, timeout=timeout).json()
    for d in dexes:
        body = {"type": "meta"} if not d else {"type": "meta", "dex": d["name"]}
        for u in s.post("https://api.hyperliquid.xyz/info", json=body, timeout=timeout).json().get("universe", []):
            rows.append(dict(venue="hyperliquid", native_symbol=u["name"],
                             status="delisted" if u.get("isDelisted") else "live",
                             contract_type="perp", underlying_type=f"dex={d['name'] if d else 'main'}",
                             tick_size=None, contract_size=1.0, listed=pd.NaT))
    return pd.DataFrame(rows)


def refresh_from_public(df: pd.DataFrame, scan: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Fill tick size, contract size and listing date from the scan; report mismatches."""
    out, problems = df.copy(), []
    for i, r in out.iterrows():
        m = scan[(scan["venue"] == r["venue"]) & (scan["native_symbol"] == r["native_symbol"])]
        if m.empty:
            problems.append(f"{r['venue']}:{r['native_symbol']} not listed on the venue API")
            continue
        m = m.iloc[0]
        if m["status"] not in ("TRADING", "live", "tradeable"):
            problems.append(f"{r['venue']}:{r['native_symbol']} status={m['status']}")
        if r["venue"] == "binance" and m["underlying_type"] == "COIN":
            problems.append(f"binance:{r['native_symbol']} is a crypto coin, not a TradFi perp")
        if pd.notna(m["tick_size"]):
            out.at[i, "tick_size"] = m["tick_size"]
        if pd.notna(m["contract_size"]):
            out.at[i, "contract_size"] = m["contract_size"]
        if pd.notna(m["listed"]):
            out.at[i, "listed"] = m["listed"].strftime("%Y-%m-%d")
    return out, problems


def tardis_availability(exchanges=("binance-futures", "okex-swap", "cryptofacilities", "hyperliquid"),
                        timeout: float = 60.0) -> pd.DataFrame:
    """Public Tardis metadata: symbol ids and the date from which each is available."""
    rows = []
    for ex in exchanges:
        meta = requests.get(f"https://api.tardis.dev/v1/exchanges/{ex}", headers=_UA, timeout=timeout).json()
        for s in meta.get("availableSymbols", []):
            rows.append(dict(tardis_exchange=ex, tardis_symbol=s["id"], type=s.get("type"),
                             since=str(s.get("availableSince"))[:10], to=str(s.get("availableTo"))[:10]))
    return pd.DataFrame(rows)
