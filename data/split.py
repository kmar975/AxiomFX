"""
Chronological split module: DEV (80%) vs HOLDOUT (20%).
Enforces a common UTC boundary across all pairs and protects HOLDOUT
from accidental snooping via a strict context manager / gatekeeper.
"""

from __future__ import annotations

from contextlib import contextmanager

import pandas as pd


class HoldoutAccessViolation(RuntimeError):
    """Raised when holdout data is accessed without explicit unlock context."""


class SplitManager:
    _holdout_unlocked: bool = False

    @classmethod
    @contextmanager
    def unlock_holdout_for_finalists(cls):
        """Context manager allowing temporary read-only access to HOLDOUT for frozen finalists."""
        prev = cls._holdout_unlocked
        cls._holdout_unlocked = True
        try:
            yield
        finally:
            cls._holdout_unlocked = prev

    @classmethod
    def compute_cutoff_date(
        cls,
        data: dict[str, pd.DataFrame],
        dev_ratio: float = 0.80,
    ) -> pd.Timestamp:
        """
        Determines the synchronized UTC cutoff timestamp across all pairs
        using the primary pair (EURUSD) or the minimum timestamp at the dev_ratio percentile.
        """
        if "EURUSD" in data:
            df = data["EURUSD"]
            idx = int(len(df) * dev_ratio)
            cutoff = pd.to_datetime(df["time"].iloc[idx], utc=True)
        else:
            first_df = next(iter(data.values()))
            idx = int(len(first_df) * dev_ratio)
            cutoff = pd.to_datetime(first_df["time"].iloc[idx], utc=True)
        return cutoff

    @classmethod
    def split_pair(
        cls,
        df: pd.DataFrame,
        cutoff_date: pd.Timestamp,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Splits dataframe into DEV and HOLDOUT by the cutoff date.
        """
        df_time = pd.to_datetime(df["time"], utc=True)
        dev_df = df[df_time < cutoff_date].copy().reset_index(drop=True)
        holdout_df = df[df_time >= cutoff_date].copy().reset_index(drop=True)
        return dev_df, holdout_df

    @classmethod
    def get_dev_data(
        cls,
        data: dict[str, pd.DataFrame],
        cutoff_date: pd.Timestamp,
    ) -> dict[str, pd.DataFrame]:
        """
        Returns DEV portions for all pairs.
        """
        dev_data = {}
        for pair, df in data.items():
            dev_df, _ = cls.split_pair(df, cutoff_date)
            dev_data[pair] = dev_df
        return dev_data

    @classmethod
    def get_holdout_data(
        cls,
        data: dict[str, pd.DataFrame],
        cutoff_date: pd.Timestamp,
    ) -> dict[str, pd.DataFrame]:
        """
        Returns HOLDOUT portions for all pairs only if explicitly unlocked.
        """
        if not cls._holdout_unlocked:
            raise HoldoutAccessViolation(
                "CRITICAL: Attempted to access HOLDOUT data during discovery! "
                "Holdout is strictly reserved for frozen finalists under SplitManager.unlock_holdout_for_finalists()."
            )
        holdout_data = {}
        for pair, df in data.items():
            _, holdout_df = cls.split_pair(df, cutoff_date)
            holdout_data[pair] = holdout_df
        return holdout_data
