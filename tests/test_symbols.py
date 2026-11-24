from llrwa.symbols import ASSETS, FORBIDDEN, VENUES, load_instruments, validate_offline


def test_instrument_table_is_complete_and_clean():
    df = load_instruments()
    assert validate_offline(df) == []
    assert len(df) == len(ASSETS) * len(VENUES)


def test_alias_traps_never_selected():
    df = load_instruments()
    chosen = set(zip(df["venue"], df["native_symbol"]))
    assert not chosen & FORBIDDEN
    gold = df[df["asset"] == "GOLD"]
    assert set(gold["native_symbol"]) == {"XAUUSDT", "XAU-USDT-SWAP", "PF_XAUUSD", "xyz:GOLD"}


def test_listing_dates_precede_window():
    # the one exception is documented (DECISIONS D27): Kraken's Micron perp listed on 23 Sep
    df = load_instruments()
    late = df[df["listed"] >= "2026-09-18"]
    assert set(zip(late["asset"], late["venue"])) == {("MU", "kraken")}
