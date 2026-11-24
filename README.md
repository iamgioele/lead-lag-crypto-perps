# Lead–lag of crypto perpetuals on traditional assets

Code for the ETH Zürich (D-MATH) semester paper *Lead Lag Estimation from Non-Synchronous Data for Crypto Perpetuals* (Gioele Pasteris, 2026).

## What the project does

Crypto exchanges list perpetual futures on traditional assets that trade 24/7, while the underlying markets have opening hours. We ask **which venue's price moves first, by how many milliseconds, and whether this changes when the underlying market is closed**.

- **Assets:** WTI crude oil, gold, SanDisk, Micron.
- **Venues:** Binance, OKX, Kraken (centralised) and Hyperliquid (decentralised).
- **Sample:** 26 October – 9 November 2026, top-of-book quotes at millisecond resolution.
- **Estimator:** the contrast of Hoffmann, Rosenbaum and Yoshida (2013), which works directly on non-synchronous data: shift one series by a candidate lag, compute the Hayashi–Yoshida covariation of overlapping increments, and take the lag that maximises it.
- **Algorithm:** each pair of overlapping increments contributes over one interval of lags, so the contrast is computed for all lags at once with a difference array, in 5-minute blocks that add up exactly (whole window, days, hours, trading states, bootstrap).

Main findings: Binance leads every other venue for all four assets (OKX by 73–80 ms, Kraken by 55–71 ms, Hyperliquid's quotes by 200–241 ms). Hyperliquid's trades trail its quotes by a further 84–403 ms. When CME is closed, Binance still leads, but the followers of the commodities change behaviour; the stocks barely change. Lags are between timestamps, so clock offsets and snapshot staleness cannot be ruled out.

## Layout

- `src/llrwa/` – Python package
  - `hry/` – all-lags contrast engine, peak selection and flags, block bootstrap, venue offsets
  - `data/` – vendor client, Kraken order-book rebuild, cleaning, reader for the Hyperliquid recording
  - `calendars/` – trading states (New York time)
  - `simulate/` – synthetic lead–lag series used for validation
  - `study.py`, `symbols.py`, `instruments.csv` – sample windows and instruments
- `collector/` – websocket recorder for Hyperliquid's `bbo` and trade channels (three redundant connections, systemd service)
- `scripts/` – data pulls and the analysis pipeline (`asset_pipeline.sh` runs one asset end to end)
- `tests/` – pytest suite, including property-based checks of the engine against a brute-force evaluation

## Running it

```bash
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt && .venv/bin/pip install -e .
cp config.example.toml config.toml   # add your own vendor keys
.venv/bin/pytest
LLRWA_STUDY=nov09 bash scripts/asset_pipeline.sh WTI
```

`scripts/install_leadlag.sh` installs the open-source `lead-lag` package, used only for the parity test.

## Data

No market data is included. Binance, OKX and Kraken data come from Amberdata under its terms of use. Hyperliquid quotes come from our own recording of the public websocket feed (`collector/`). With your own access the pipeline rebuilds all tables and figures.
