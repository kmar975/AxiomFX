"""
Fetches and aggregates W1 and MN bars for all 7 pairs.
"""

from data.download import prepare_all_timeframes


def fetch_w1_mn():
    print("[DATA] Updating W1 and MN data...")
    prepare_all_timeframes()


if __name__ == "__main__":
    fetch_w1_mn()
