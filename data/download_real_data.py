"""
Institutional Real Market Data Downloader.
Populates verified real candles directly from Dukascopy Bank SA.
"""

from __future__ import annotations

from data.download import prepare_all_timeframes


def download_real_data():
    print("[DATA] Running real market data sync from Dukascopy Bank...")
    return prepare_all_timeframes(force_regenerate=True)


if __name__ == "__main__":
    download_real_data()
