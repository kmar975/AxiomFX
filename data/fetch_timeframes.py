"""
Synchronizes and verifies all timeframes for all 7 pairs.
"""

from data.download import prepare_all_timeframes


def fetch_all():
    return prepare_all_timeframes()


if __name__ == "__main__":
    fetch_all()
