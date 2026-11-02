"""
Institutional FX Costs and Precision Accounting Module.
Handles pair-specific pip multipliers (JPY dec=3, pip_mult=100.0 vs others dec=5, pip_mult=10000.0),
causal quote-to-USD conversion rates, rollover spread protection (17:00 NY),
intra-candle pessimistic SL/TP conflict resolution, and scenario multipliers (1.0x, 1.5x, 2.0x).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

PAIR_SPECS: dict[str, dict[str, float | int | str]] = {
    "EURUSD": {"dec": 5, "pip_unit": 0.0001, "pip_mult": 10000.0, "quote_curr": "USD"},
    "GBPUSD": {"dec": 5, "pip_unit": 0.0001, "pip_mult": 10000.0, "quote_curr": "USD"},
    "USDJPY": {"dec": 3, "pip_unit": 0.01,   "pip_mult": 100.0,   "quote_curr": "JPY"},
    "AUDUSD": {"dec": 5, "pip_unit": 0.0001, "pip_mult": 10000.0, "quote_curr": "USD"},
    "USDCHF": {"dec": 5, "pip_unit": 0.0001, "pip_mult": 10000.0, "quote_curr": "CHF"},
    "EURGBP": {"dec": 5, "pip_unit": 0.0001, "pip_mult": 10000.0, "quote_curr": "GBP"},
    "EURJPY": {"dec": 3, "pip_unit": 0.01,   "pip_mult": 100.0,   "quote_curr": "JPY"},
}


def get_pair_specs(pair: str) -> dict[str, float | int | str]:
    if pair not in PAIR_SPECS:
        # Default assumption if non-standard pair
        if "JPY" in pair:
            return {"dec": 3, "pip_unit": 0.01, "pip_mult": 100.0, "quote_curr": "JPY"}
        return {"dec": 5, "pip_unit": 0.0001, "pip_mult": 10000.0, "quote_curr": "USD"}
    return PAIR_SPECS[pair]


def price_to_pips(pair: str, price_diff: float | np.ndarray) -> float | np.ndarray:
    spec = get_pair_specs(pair)
    pip_mult = float(spec["pip_mult"])
    return price_diff * pip_mult


def pips_to_price(pair: str, pips: float | np.ndarray) -> float | np.ndarray:
    spec = get_pair_specs(pair)
    pip_unit = float(spec["pip_unit"])
    return pips * pip_unit


def get_quote_to_usd_rate(
    pair: str,
    time_idx: int,
    market_closes: dict[str, np.ndarray],
) -> float:
    """
    Computes causal conversion rate from quote currency to USD at time_idx.
    - USD-quoted (EURUSD, GBPUSD, AUDUSD): 1.0
    - JPY-quoted (USDJPY, EURJPY): 1.0 / close_USDJPY[time_idx]
    - CHF-quoted (USDCHF): 1.0 / close_USDCHF[time_idx]
    - GBP-quoted (EURGBP): close_GBPUSD[time_idx]
    """
    spec = get_pair_specs(pair)
    quote_curr = str(spec["quote_curr"])

    if quote_curr == "USD":
        return 1.0
    elif quote_curr == "JPY":
        if "USDJPY" in market_closes and len(market_closes["USDJPY"]) > time_idx:
            usdjpy = market_closes["USDJPY"][time_idx]
            return 1.0 / usdjpy if usdjpy > 0 else 1.0 / 150.0
        return 1.0 / 150.0
    elif quote_curr == "CHF":
        if "USDCHF" in market_closes and len(market_closes["USDCHF"]) > time_idx:
            usdchf = market_closes["USDCHF"][time_idx]
            return 1.0 / usdchf if usdchf > 0 else 1.0 / 0.90
        return 1.0 / 0.90
    elif quote_curr == "GBP":
        if "GBPUSD" in market_closes and len(market_closes["GBPUSD"]) > time_idx:
            gbpusd = market_closes["GBPUSD"][time_idx]
            return float(gbpusd)
        return 1.27
    return 1.0


def compute_quote_to_usd_vector(
    pair: str,
    n_bars: int,
    market_closes: dict[str, np.ndarray],
) -> np.ndarray:
    """
    Returns array of length n_bars with quote-to-USD conversion rates.
    """
    spec = get_pair_specs(pair)
    quote_curr = str(spec["quote_curr"])

    if quote_curr == "USD":
        return np.ones(n_bars, dtype=np.float64)
    elif quote_curr == "JPY":
        if "USDJPY" in market_closes and len(market_closes["USDJPY"]) >= n_bars:
            usdjpy = market_closes["USDJPY"][:n_bars]
            return np.where(usdjpy > 0, 1.0 / usdjpy, 1.0 / 150.0).astype(np.float64)
        return np.full(n_bars, 1.0 / 150.0, dtype=np.float64)
    elif quote_curr == "CHF":
        if "USDCHF" in market_closes and len(market_closes["USDCHF"]) >= n_bars:
            usdchf = market_closes["USDCHF"][:n_bars]
            return np.where(usdchf > 0, 1.0 / usdchf, 1.0 / 0.90).astype(np.float64)
        return np.full(n_bars, 1.0 / 0.90, dtype=np.float64)
    elif quote_curr == "GBP":
        if "GBPUSD" in market_closes and len(market_closes["GBPUSD"]) >= n_bars:
            return market_closes["GBPUSD"][:n_bars].astype(np.float64)
        return np.full(n_bars, 1.27, dtype=np.float64)
    return np.ones(n_bars, dtype=np.float64)


def is_rollover_hour(utc_timestamp: pd.Timestamp) -> bool:
    """
    Checks if bar is in the rollover hour (17:00-18:00 America/New_York).
    In UTC, this corresponds to 21:00 or 22:00 depending on DST.
    """
    try:
        ny_time = utc_timestamp.tz_convert("America/New_York")
        return ny_time.hour == 17
    except (AttributeError, ValueError, TypeError):
        # Fallback to UTC hour 21 or 22
        return utc_timestamp.hour in (21, 22)


def get_diurnal_spread_multiplier(utc_timestamp: pd.Timestamp) -> float:
    """
    Returns the institutional diurnal spread multiplier.
    During the 17:00 NY rollover hour (17:00-18:00 America/New_York), spreads expand 3.5x.
    """
    return 3.5 if is_rollover_hour(utc_timestamp) else 1.0


def is_market_entry_allowed(
    spread_price: float,
    atr_price: float,
    max_spread_atr_frac: float = 0.25,
) -> bool:
    """
    Rejects market entry if current spread exceeds max_spread_atr_frac * ATR20.
    """
    if atr_price <= 0:
        return True
    return spread_price <= (max_spread_atr_frac * atr_price)


def resolve_intracandle_conflict(
    is_long: bool,
    open_price: float,
    high_price: float,
    low_price: float,
    sl_price: float,
    tp_price: float,
) -> tuple[bool, bool, str]:
    """
    Pessimistic resolution rule:
    If SL and TP levels are both touched within the same candle [Low, High],
    ALWAYS resolve pessimistically as SL.
    Returns: (sl_hit, tp_hit, exit_reason)
    """
    if is_long:
        sl_hit = low_price <= sl_price
        tp_hit = high_price >= tp_price
    else:
        sl_hit = high_price >= sl_price
        tp_hit = low_price <= tp_price

    if sl_hit and tp_hit:
        # Pessimistic conflict resolution: SL wins
        return True, False, "SL"
    elif sl_hit:
        return True, False, "SL"
    elif tp_hit:
        return False, True, "TP"
    return False, False, "NONE"
