"""
Test utilities and authentic test data loader.
Zero synthetic generation: strictly loads verified Dukascopy real market data.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from data.specs import ALL_PAIRS


def load_test_basket(n_bars: int = 300) -> dict[str, pd.DataFrame]:
    """
    Loads a slice of verified real historical market data for all 7 pairs.
    """
    timeframes_dir = Path("output/timeframes")
    eur_h1 = timeframes_dir / "EURUSD_H1.csv"
    eur_parq = timeframes_dir / "EURUSD_H1.parquet"
    has_data = False
    if eur_parq.exists():
        has_data = True
    elif eur_h1.exists():
        try:
            with open(eur_h1, "r", encoding="utf-8") as f:
                has_data = sum(1 for _ in f) > 50
        except Exception:
            has_data = False

    if not has_data:
        from data.download import prepare_all_timeframes
        prepare_all_timeframes(n_days=60)

    basket = {}
    for p in ALL_PAIRS:
        f_csv = timeframes_dir / f"{p}_H1.csv"
        f_parq = timeframes_dir / f"{p}_H1.parquet"
        if f_parq.exists():
            df = pd.read_parquet(f_parq)
        elif f_csv.exists():
            df = pd.read_csv(f_csv)
        else:
            raise RuntimeError(f"Missing real data for {p} in {timeframes_dir}")

        df["time"] = pd.to_datetime(df["time"], utc=True)
        basket[p] = df.tail(min(len(df), n_bars)).copy().reset_index(drop=True)

    return basket
