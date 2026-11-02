"""
Institutional Forex Currency Specifications and Constants.
Zero synthetic generation: pure deterministic market specifications.
"""

from __future__ import annotations

PAIR_SPECS: dict[str, dict[str, float | int | str]] = {
    "EURUSD": {"dec": 5, "pip_unit": 0.0001, "pip_mult": 10000.0, "base_price": 1.0850, "base_spread_pips": 1.1, "point_div": 100000.0, "dukascopy_symbol": "EUR/USD", "quote_curr": "USD"},
    "GBPUSD": {"dec": 5, "pip_unit": 0.0001, "pip_mult": 10000.0, "base_price": 1.2720, "base_spread_pips": 1.4, "point_div": 100000.0, "dukascopy_symbol": "GBP/USD", "quote_curr": "USD"},
    "USDJPY": {"dec": 3, "pip_unit": 0.01,   "pip_mult": 100.0,   "base_price": 154.50, "base_spread_pips": 1.2, "point_div": 1000.0,   "dukascopy_symbol": "USD/JPY", "quote_curr": "JPY"},
    "AUDUSD": {"dec": 5, "pip_unit": 0.0001, "pip_mult": 10000.0, "base_price": 0.6550, "base_spread_pips": 1.5, "point_div": 100000.0, "dukascopy_symbol": "AUD/USD", "quote_curr": "USD"},
    "USDCHF": {"dec": 5, "pip_unit": 0.0001, "pip_mult": 10000.0, "base_price": 0.8850, "base_spread_pips": 1.6, "point_div": 100000.0, "dukascopy_symbol": "USD/CHF", "quote_curr": "CHF"},
    "EURGBP": {"dec": 5, "pip_unit": 0.0001, "pip_mult": 10000.0, "base_price": 0.8530, "base_spread_pips": 1.5, "point_div": 100000.0, "dukascopy_symbol": "EUR/GBP", "quote_curr": "GBP"},
    "EURJPY": {"dec": 3, "pip_unit": 0.01,   "pip_mult": 100.0,   "base_price": 167.60, "base_spread_pips": 1.8, "point_div": 1000.0,   "dukascopy_symbol": "EUR/JPY", "quote_curr": "JPY"},
}

ALL_PAIRS: list[str] = list(PAIR_SPECS.keys())
ALL_TIMEFRAMES: list[str] = ["M1", "M5", "M15", "H1", "H4", "D1", "W1", "MN"]
PRIMARY_PAIR: str = "EURUSD"
