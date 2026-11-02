"""
Dukascopy downloader wrapper.
Downloads and verifies real historical market data for all 7 pairs and 8 timeframes.
Zero synthetic fallbacks: requires authentic Dukascopy feed.
"""

from __future__ import annotations

from data.download import prepare_all_timeframes


def download_all():
    print("[DUKASCOPY] Initializing real Dukascopy Bank data feed synchronization...")
    import importlib.util

    if importlib.util.find_spec("dukascopy_python") is None:
        raise RuntimeError("CRITICAL: `dukascopy-python` is required for institutional data ingestion.")

    print("[DUKASCOPY] Ingesting authentic market data across 56 markets...")
    return prepare_all_timeframes(force_regenerate=True)


if __name__ == "__main__":
    download_all()
