"""
Institutional Forex Currency Specifications and Constants.
All synthetic generation routines have been removed permanently.
Use data.download for Dukascopy Bank real market data.
"""

from __future__ import annotations

from data.specs import ALL_PAIRS, ALL_TIMEFRAMES, PAIR_SPECS, PRIMARY_PAIR

__all__ = [
    "ALL_PAIRS",
    "ALL_TIMEFRAMES",
    "PAIR_SPECS",
    "PRIMARY_PAIR",
]
