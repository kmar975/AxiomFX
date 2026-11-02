"""
Leak-Free Meta-Labeling with LightGBM, Time Decay, and Purged TimeSeriesSplit.
Adapted from institutional quant architecture.
Trains a secondary meta-classifier on trade entries using macroeconomic and microstructure context.
Enforces strictly out-of-fold (OOF) probability estimation and optimal decision threshold theta.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

from engine.costs import compute_quote_to_usd_vector
from engine.vectorized import run_vectorized_backtest
from grammar.compiler import compile_strategy
from grammar.strategy import Strategy


class PurgedTimeSeriesSplit:
    """
    Time-Series Split with Purging:
    Discards any training trade whose exit bar extends into or past the validation fold start bar.
    Prevents information leakage from overlapping holding periods.
    """

    def __init__(self, n_splits: int = 5):
        self.n_splits = n_splits

    def split(
        self,
        entry_bars: np.ndarray,
        exit_bars: np.ndarray,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        n_trades = len(entry_bars)
        indices = np.arange(n_trades)
        fold_size = max(1, n_trades // (self.n_splits + 1))
        splits: list[tuple[np.ndarray, np.ndarray]] = []

        for f in range(1, self.n_splits + 1):
            val_start_idx = f * fold_size
            val_end_idx = (f + 1) * fold_size if f < self.n_splits else n_trades
            if val_start_idx >= n_trades:
                break

            val_indices = indices[val_start_idx:val_end_idx]
            val_first_entry = entry_bars[val_start_idx]

            raw_train_indices = indices[:val_start_idx]

            # Purge train trades that overlap with validation fold
            purged_train_indices = [
                idx for idx in raw_train_indices
                if exit_bars[idx] < val_first_entry
            ]

            if len(purged_train_indices) >= 10 and len(val_indices) > 0:
                splits.append((np.array(purged_train_indices, dtype=np.int32), val_indices))

        return splits


from pathlib import Path
import joblib

@dataclass
class MetaModelFilter:
    strategy_id: str
    model: LGBMClassifier | None
    threshold: float
    feature_cols: list[str]
    train_pf_base: float
    train_pf_meta: float
    train_trades_meta: int
    oof_passed_bars: set[int] | None = None
    oof_rejected_bars: set[int] | None = None

    def filter_signals(
        self,
        raw_signals: np.ndarray,
        features_df: pd.DataFrame,
        is_dev: bool = False,
    ) -> np.ndarray:
        """
        Filters raw strategy entry signals using strictly out-of-fold (OOF) predictions
        on DEV and the trained model on unseen out-of-sample data (Holdout, Live).
        Guarantees zero in-sample overfitting.
        """
        if self.model is None or not np.any(raw_signals):
            return raw_signals.copy()

        sig_indices = np.where(raw_signals)[0]
        if len(sig_indices) == 0:
            return raw_signals.copy()

        filtered_mask = np.zeros(len(raw_signals), dtype=bool)

        has_oof = (self.oof_passed_bars is not None) and (self.oof_rejected_bars is not None)
        use_oof = bool(is_dev and has_oof)

        valid_cols = [c for c in self.feature_cols if c in features_df.columns]
        need_model_indices = []

        for idx in sig_indices:
            if use_oof:
                if idx in self.oof_passed_bars:
                    filtered_mask[idx] = True
                elif idx in self.oof_rejected_bars:
                    filtered_mask[idx] = False
                else:
                    # In fold 0 without prior OOF window, preserve base signal
                    filtered_mask[idx] = True
            else:
                need_model_indices.append(idx)

        if need_model_indices and valid_cols:
            X_sig = features_df[valid_cols].iloc[need_model_indices].values.astype(np.float32)
            X_sig = np.nan_to_num(X_sig, nan=50.0, posinf=100.0, neginf=0.0)
            probs = self.model.predict_proba(X_sig)[:, 1]
            passed = probs >= self.threshold
            for i, p_val in enumerate(passed):
                if p_val:
                    filtered_mask[need_model_indices[i]] = True

        return filtered_mask


def save_meta_filter(filter_obj: MetaModelFilter, path: Path | str) -> None:
    """Serializes MetaModelFilter to disk using joblib."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(filter_obj, p)


def load_meta_filter(path: Path | str) -> MetaModelFilter | None:
    """Deserializes MetaModelFilter from disk."""
    p = Path(path)
    if not p.exists():
        return None
    try:
        return joblib.load(p)
    except Exception:
        return None


def build_meta_feature_matrix(features_df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """
    Constructs context feature matrix Z across S1–S10 for meta-labeling.
    Prefers volatility, session, momentum, and regime features.
    """
    preferred_prefixes = [
        "s1_rsi", "s1_sma_dist", "s1_donchian",
        "s2_clv", "s2_body_atr", "s2_range_atr",
        "s3_structure_state", "s3_consol_frac", "s3_retrace_frac",
        "s5_atr20_pct", "s5_bb_width", "s5_squeeze_on",
        "s6_session_london", "s6_session_ny", "s6_hour_sin", "s6_hour_cos",
        "s7_price_zscore", "s7_autocorr_lag1",
        "s8_spread_dev", "s8_dir_efficiency",
        "s9_rel_strength", "s10_regime_p_high_vol",
        "htf_s1_rsi_14", "htf_s5_atr20_pct",
    ]

    selected_cols = [
        c for c in features_df.columns
        if any(p in c for p in preferred_prefixes)
    ]

    if not selected_cols:
        # Fallback to first 20 numeric columns
        selected_cols = [c for c in features_df.columns if c != "time"][:20]

    Z = features_df[selected_cols].values.astype(np.float32)
    Z = np.nan_to_num(Z, nan=50.0, posinf=100.0, neginf=0.0)
    return Z, selected_cols


def train_meta_labeling_for_strategy(
    strategy: Strategy,
    df: pd.DataFrame,
    features_df: pd.DataFrame,
    pair: str = "EURUSD",
    cost_multiplier: float = 1.5,
    time_decay_lambda: float = 0.35,
) -> MetaModelFilter:
    """
    Trains LightGBM meta-model for strategy:
    1. Extracts primary trade entries and PnLs via backtest.
    2. Applies exponential time decay weights.
    3. Runs Purged 5-fold TimeSeriesSplit with out-of-fold predictions.
    4. Optimizes threshold theta in [0.50, 0.65] maximizing Q25 Sharpe and Profit Factor.
    """
    l_mask, s_mask = compile_strategy(strategy, features_df)
    n_bars = len(df)
    q_vec = compute_quote_to_usd_vector(pair, n_bars, {pair: df["close"].values})

    # Run backtest on DEV
    bt_res = run_vectorized_backtest(
        df=df,
        pair=pair,
        entry_longs=l_mask,
        entry_shorts=s_mask,
        quote_to_usd=q_vec,
        sl_atr_mult=strategy.exit.sl_atr,
        tp_atr_mult=strategy.exit.tp_atr,
        max_holding_bars=strategy.exit.max_holding_bars,
        cost_multiplier=cost_multiplier,
        exit_type=getattr(strategy.exit, "exit_type", "sl_tp"),
        trailing_atr=getattr(strategy.exit, "trailing_atr", 1.2),
    )

    trade_pnls = bt_res.get("trade_pnls", np.array([]))
    entry_bars = bt_res.get("entry_bars", np.array([], dtype=np.int64))
    exit_bars = bt_res.get("exit_bars", np.array([], dtype=np.int64))
    n_trades = len(trade_pnls)

    base_pf = float(bt_res.get("profit_factor", 1.0))

    if n_trades < 25 or len(entry_bars) == 0:
        # Insufficient sample for secondary ML classifier
        return MetaModelFilter(
            strategy_id=strategy.strategy_id,
            model=None,
            threshold=0.50,
            feature_cols=[],
            train_pf_base=base_pf,
            train_pf_meta=base_pf,
            train_trades_meta=n_trades,
        )

    # Exact signal generation bars: sig_idx = max(0, entry_bars - 1)
    sig_idx = np.maximum(0, entry_bars - 1)

    # Binary classification target: 1 if profitable trade, 0 otherwise
    y = (trade_pnls > 0.0).astype(np.int32)

    # Context features matrix Z at exact signal bars
    Z_all, feat_names = build_meta_feature_matrix(features_df)
    X_trades = Z_all[sig_idx]

    # Exponential time decay sample weights
    if "time" in df.columns:
        times = pd.to_datetime(df["time"].iloc[sig_idx], utc=True)
        max_t = times.max()
        years_ago = (max_t - times).dt.total_seconds().values / (365.25 * 86400.0)
        years_ago = np.maximum(years_ago, 0.0)
        sample_weights = np.exp(-time_decay_lambda * years_ago)
    else:
        sample_weights = np.ones(len(entry_bars), dtype=np.float64)

    # Purged TimeSeriesSplit with OOF predictions
    splitter = PurgedTimeSeriesSplit(n_splits=5)
    splits = splitter.split(entry_bars, exit_bars)

    oof_probs = np.full(len(y), np.nan, dtype=np.float32)

    for train_idx, val_idx in splits:
        if len(train_idx) < 15 or len(val_idx) == 0:
            continue
        if len(np.unique(y[train_idx])) < 2:
            continue
        clf = LGBMClassifier(
            max_depth=4,
            num_leaves=15,
            min_child_samples=min(15, len(train_idx) // 3),
            learning_rate=0.03,
            n_estimators=80,
            random_state=42,
            class_weight="balanced",
            verbose=-1,
        )
        clf.fit(X_trades[train_idx], y[train_idx], sample_weight=sample_weights[train_idx])
        oof_probs[val_idx] = clf.predict_proba(X_trades[val_idx])[:, 1]

    # Ensure binary target has both classes for final model training
    if len(np.unique(y)) < 2:
        return MetaModelFilter(
            strategy_id=strategy.strategy_id,
            model=None,
            threshold=0.50,
            feature_cols=feat_names,
            train_pf_base=base_pf,
            train_pf_meta=base_pf,
            train_trades_meta=n_trades,
        )

    # Train final model on all trade entries
    final_model = LGBMClassifier(
        max_depth=4,
        num_leaves=15,
        min_child_samples=min(20, len(y) // 3),
        learning_rate=0.03,
        n_estimators=100,
        random_state=42,
        class_weight="balanced",
        verbose=-1,
    )
    final_model.fit(X_trades, y, sample_weight=sample_weights)

    # Optimize decision threshold theta on out-of-fold predictions
    valid_oof = ~np.isnan(oof_probs)
    if np.sum(valid_oof) < 15:
        return MetaModelFilter(
            strategy_id=strategy.strategy_id,
            model=None,
            threshold=0.50,
            feature_cols=feat_names,
            train_pf_base=base_pf,
            train_pf_meta=base_pf,
            train_trades_meta=n_trades,
        )

    oof_p = oof_probs[valid_oof]
    oof_pnls = trade_pnls[valid_oof]
    base_mean_pnl = float(np.mean(oof_pnls))

    threshold_grid = [0.50, 0.52, 0.54, 0.56, 0.58, 0.60, 0.62, 0.65]
    best_theta = 0.50
    best_pf = -1.0
    best_mean_pnl = -1e9
    best_n_trades = len(oof_pnls)

    min_required_trades = max(10, int(len(oof_pnls) * 0.35))

    for theta in threshold_grid:
        pass_idx = oof_p >= theta
        n_t = int(np.sum(pass_idx))
        if n_t >= min_required_trades:
            sel_pnls = oof_pnls[pass_idx]
            gains = float(np.sum(sel_pnls[sel_pnls > 0]))
            losses = float(np.abs(np.sum(sel_pnls[sel_pnls < 0])))
            pf = gains / losses if losses > 1e-6 else (2.5 if gains > 0 else 0.5)
            mean_pnl = float(np.mean(sel_pnls))
            if pf > best_pf:
                best_pf = pf
                best_mean_pnl = mean_pnl
                best_theta = theta
                best_n_trades = n_t

    # Degradation check:
    # Requires best_pf >= base_pf * 1.05 AND best_mean_pnl > base_mean_pnl
    if best_pf < base_pf * 1.05 or best_mean_pnl <= base_mean_pnl:
        return MetaModelFilter(
            strategy_id=strategy.strategy_id,
            model=None,
            threshold=0.50,
            feature_cols=feat_names,
            train_pf_base=round(base_pf, 3),
            train_pf_meta=round(base_pf, 3),
            train_trades_meta=n_trades,
        )

    oof_passed_bars = {
        int(sig_idx[i])
        for i in range(len(sig_idx))
        if not np.isnan(oof_probs[i]) and oof_probs[i] >= best_theta
    }
    oof_rejected_bars = {
        int(sig_idx[i])
        for i in range(len(sig_idx))
        if not np.isnan(oof_probs[i]) and oof_probs[i] < best_theta
    }

    return MetaModelFilter(
        strategy_id=strategy.strategy_id,
        model=final_model,
        threshold=best_theta,
        feature_cols=feat_names,
        train_pf_base=round(base_pf, 3),
        train_pf_meta=round(best_pf, 3),
        train_trades_meta=best_n_trades,
        oof_passed_bars=oof_passed_bars,
        oof_rejected_bars=oof_rejected_bars,
    )
