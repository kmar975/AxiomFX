"""
V2 Synchronous Null World Engine (§ 9).
Permutes full FX 24-hour trading days (17:00 NY to 17:00 NY) synchronously across all pairs,
applies random directional reflection sign epsilon_d in {-1, +1},
and reconstructs volatility paths, perfectly preserving session diurnal dynamics,
cross-pair correlations, and HTF consistency while destroying inter-day predictive edge.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from data.process_and_clean_timeframes import clean_and_enforce_continuity
from engine.costs import get_pair_specs


def build_synchronous_null_world(
    basket_data: dict[str, pd.DataFrame],
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    """
    Constructs a single synchronous null world across all pairs in basket_data.
    1. Identifies common FX trading days (17:00 NY to 17:00 NY).
    2. Synchronously permutes day indices across all pairs with random sign epsilon_d.
    3. Rebuilds price path enforcing continuity Open[i] = Close[i-1].
    """
    rng = np.random.RandomState(seed)
    pairs = list(basket_data.keys())
    primary_pair = "EURUSD" if "EURUSD" in basket_data else pairs[0]
    primary_df = basket_data[primary_pair]

    times = pd.to_datetime(primary_df["time"], utc=True)
    ny_times = times.dt.tz_convert("America/New_York")
    # Date adjusted for 17:00 cutoff: if hour >= 17, belongs to next trading date
    adj_dates = (ny_times + pd.Timedelta(hours=7)).dt.date
    unique_days = sorted(set(adj_dates))
    n_days = len(unique_days)

    if n_days < 5:
        # Fallback if too short: return copy
        return {p: df.copy() for p, df in basket_data.items()}

    # Permutation of day order
    permuted_day_indices = rng.permutation(n_days)
    day_signs = rng.choice([-1.0, 1.0], size=n_days)

    # Map original day -> (new_day_idx, sign)
    day_map = {
        unique_days[i]: (unique_days[permuted_day_indices[i]], day_signs[i])
        for i in range(n_days)
    }

    null_basket = {}

    for pair, df in basket_data.items():
        spec = get_pair_specs(pair)
        pip_unit = float(spec["pip_unit"])
        dec = int(spec["dec"])

        p_times = pd.to_datetime(df["time"], utc=True)
        p_ny = p_times.dt.tz_convert("America/New_York")
        p_adj_dates = (p_ny + pd.Timedelta(hours=7)).dt.date

        # Group rows by day
        day_groups = {}
        for idx, d_val in enumerate(p_adj_dates):
            day_groups.setdefault(d_val, []).append(idx)

        # Assemble new row sequence
        new_opens = []
        new_highs = []
        new_lows = []
        new_closes = []
        new_spreads = []
        new_volumes = []
        new_times = []

        df_open = df["open"].values
        df_high = df["high"].values
        df_low = df["low"].values
        df_close = df["close"].values
        df_spread = df["spread"].values if "spread" in df.columns else np.full(len(df), 1.5)
        df_vol = df["tick_volume"].values if "tick_volume" in df.columns else np.full(len(df), 500)
        df_time = df["time"].values

        cur_price = df_close[0]

        for orig_d in unique_days:
            if orig_d not in day_map or orig_d not in day_groups:
                continue

            orig_indices = day_groups[orig_d]
            target_d, eps = day_map[orig_d]
            if target_d not in day_groups:
                target_d = orig_d
                eps = 1.0

            target_indices = day_groups[target_d]
            target_hm = [(p_ny.iloc[ti].hour, p_ny.iloc[ti].minute) for ti in target_indices]
            target_hm_map = dict(zip(target_hm, target_indices))

            for o_idx in orig_indices:
                o_hm = (p_ny.iloc[o_idx].hour, p_ny.iloc[o_idx].minute)
                if o_hm in target_hm_map:
                    t_idx = target_hm_map[o_hm]
                else:
                    o_mins = o_hm[0] * 60 + o_hm[1]
                    t_idx = min(target_indices, key=lambda ti: abs((p_ny.iloc[ti].hour * 60 + p_ny.iloc[ti].minute) - o_mins))

                t_open = df_open[t_idx]
                t_high = df_high[t_idx]
                t_low = df_low[t_idx]
                t_close = df_close[t_idx]
                sp = df_spread[t_idx]
                tv = df_vol[t_idx]
                tm = df_time[o_idx]

                # Increments
                d_body = (t_close - t_open) * eps
                u_wick = t_high - max(t_open, t_close)
                l_wick = min(t_open, t_close) - t_low

                # If reflected (eps == -1), swap upper and lower wicks
                if eps == -1.0:
                    u_wick, l_wick = l_wick, u_wick

                o = cur_price
                c = round(o + d_body, dec)
                h = round(max(o, c) + u_wick, dec)
                l = round(min(o, c) - l_wick, dec)

                if (h - l) < pip_unit:
                    h = round(l + pip_unit * 2, dec)

                new_opens.append(o)
                new_highs.append(h)
                new_lows.append(l)
                new_closes.append(c)
                new_spreads.append(sp)
                new_volumes.append(tv)
                new_times.append(tm)

                cur_price = c

        null_df = pd.DataFrame({
            "time": new_times,
            "open": new_opens,
            "high": new_highs,
            "low": new_lows,
            "close": new_closes,
            "spread": new_spreads,
            "tick_volume": new_volumes,
        })
        null_df = clean_and_enforce_continuity(null_df, pair)
        null_basket[pair] = null_df

    return null_basket


def compute_v2_p_value(
    real_fitness: float,
    null_world_fitnesses: list[float],
) -> float:
    """
    Computes rigorous empirical p-value for hypothesis against null worlds:
    p = (1 + #{nulls with fitness >= real}) / (1 + N_nulls)
    """
    n_nulls = len(null_world_fitnesses)
    if n_nulls == 0:
        return 1.0

    beat_count = sum(1 for null_fit in null_world_fitnesses if null_fit >= real_fitness)
    p_val = (1.0 + beat_count) / (1.0 + n_nulls)
    return float(p_val)
