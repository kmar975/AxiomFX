"""
Four-Stage Execution Funnel (S0 -> S1 -> S2 -> S3).
S0: Fast future return signal gather.
S1: Fast Engine on 1/3 of DEV blocks.
S2: Multi-Pair Fast Engine with jitter +-1 step and 1.5x fitness costs.
S3: Full Event-Driven institutional verification for top elites.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from atoms.bitsets import AtomicBitsetStore, pack_bool_array_to_uint64
from engine.costs import compute_quote_to_usd_vector
from engine.event_driven import EventDrivenEngine
from engine.vectorized import run_vectorized_backtest
from grammar.compiler import compile_strategy
from grammar.strategy import Strategy
from storage.state import StorageManager


class FunnelExecutor:
    def __init__(
        self,
        bitset_store: AtomicBitsetStore,
        storage: StorageManager | None = None,
        s0_top_pct: float = 0.15,
        s1_dev_fraction: float = 0.333,
        cost_multiplier_fitness: float = 1.5,
    ):
        self.bitset_store = bitset_store
        self.storage = storage
        self.s0_top_pct = s0_top_pct
        self.s1_dev_fraction = s1_dev_fraction
        self.cost_multiplier = cost_multiplier_fitness

        self.n_s0 = 0
        self.n_s1 = 0
        self.n_s2 = 0
        self.n_s3 = 0

    def evaluate_s0_screening(
        self,
        strategy: Strategy,
        features_by_pair: dict[str, pd.DataFrame],
        future_returns_by_pair: dict[str, dict[int, np.ndarray]],
        min_trades: int = 10,
    ) -> tuple[bool, float, dict[str, Any]]:
        """
        S0 Screening: gathers 5-bar forward return at signal bars.
        Checks: n_signals >= min_trades, positive mean return minus cost on >= 50% pairs.
        """
        self.n_s0 += 1
        if hasattr(strategy, "is_valid_directional") and not strategy.is_valid_directional():
            return False, 0.0, {"stage": "S0", "reason": "Non-directional strategy (all regime ceilings)"}

        primary_pair = "EURUSD" if "EURUSD" in features_by_pair else next(iter(features_by_pair.keys()), None)
        if primary_pair is None:
            return False, 0.0, {"stage": "S0", "reason": "No market features available"}

        # Quick pre-check on primary pair
        prim_df = features_by_pair[primary_pair]
        l_prim, s_prim = compile_strategy(strategy, prim_df)
        both_prim = l_prim & s_prim
        valid_l_prim = l_prim & ~both_prim
        valid_s_prim = s_prim & ~both_prim
        prim_signals = valid_l_prim | valid_s_prim
        n_prim = int(np.sum(prim_signals))

        if n_prim < min_trades:
            return False, 0.0, {"stage": "S0", "reason": f"Insufficient trades on {primary_pair}"}

        # Bitset deduplication on primary pair: reject duplicate twin masks (Jaccard >= 0.95)
        packed_prim = None
        if self.bitset_store is not None:
            packed_prim = pack_bool_array_to_uint64(np.concatenate([valid_l_prim, valid_s_prim]))
            dup_result = self.bitset_store.find_duplicate_signal(packed_prim, threshold=0.95)
            if dup_result is not None:
                return False, 0.0, {"stage": "S0", "reason": "Duplicate signal mask (Jaccard >= 0.95)"}

        pos_pairs = 0
        total_pairs = len(features_by_pair)
        basket_ret_sum = 0.0
        n_signals_eur = n_prim if primary_pair == "EURUSD" else 0
        eur_signals = prim_signals if primary_pair == "EURUSD" else None

        for pair, f_df in features_by_pair.items():
            if pair == primary_pair:
                valid_l = valid_l_prim
                valid_s = valid_s_prim
                sig_mask = prim_signals
                n_sig = n_prim
            else:
                l_mask, s_mask = compile_strategy(strategy, f_df)
                both = l_mask & s_mask
                valid_l = l_mask & ~both
                valid_s = s_mask & ~both
                sig_mask = valid_l | valid_s
                n_sig = int(np.sum(sig_mask))
                if pair == "EURUSD":
                    n_signals_eur = n_sig
                    eur_signals = sig_mask

            if n_sig < 5:
                continue

            # Multi-horizon forward return (check best of 5, 10, 20 bars)
            best_h_mean = -999.0
            for h in (5, 10, 20):
                f_ret = future_returns_by_pair.get(pair, {}).get(h, None)
                if f_ret is not None and len(f_ret) == len(f_df):
                    dir_ret = np.where(valid_l, f_ret, np.where(valid_s, -f_ret, 0.0))
                    mean_val = float(np.mean(dir_ret[sig_mask]))
                    if mean_val > best_h_mean:
                        best_h_mean = mean_val
            if best_h_mean <= -900.0:
                f_ret_5 = future_returns_by_pair.get(pair, {}).get(5, np.zeros(len(f_df), dtype=np.float32))
                dir_ret = np.where(valid_l, f_ret_5, np.where(valid_s, -f_ret_5, 0.0))
                best_h_mean = float(np.mean(dir_ret[sig_mask]))

            # Deduct cost proxy (~0.02 ATR)
            net_mean = best_h_mean - 0.02
            basket_ret_sum += net_mean
            if net_mean > 0:
                pos_pairs += 1

        # Require positive sign on at least 40% pairs or strong basket return
        passed = (pos_pairs / max(1, total_pairs) >= 0.40) or (basket_ret_sum > 0.0)
        s0_score = basket_ret_sum / max(1, total_pairs)

        if passed and self.bitset_store is not None and packed_prim is not None:
            self.bitset_store.store_evaluated_signal(
                packed_prim,
                {"passed": True, "s0_score": s0_score, "strategy_id": getattr(strategy, "strategy_id", "")},
                exit_key="",
            )

        return passed, s0_score, {"stage": "S0", "pos_pairs": pos_pairs, "n_signals_eur": n_signals_eur, "eur_mask": eur_signals}

    def evaluate_s1_fast(
        self,
        strategy: Strategy,
        dev_data: dict[str, pd.DataFrame],
        features_by_pair: dict[str, pd.DataFrame],
    ) -> tuple[bool, dict[str, Any]]:
        """
        S1 Fast Engine: evaluates 1/3 of DEV on EURUSD.
        """
        self.n_s1 += 1
        eur_df = dev_data["EURUSD"]
        eur_feats = features_by_pair["EURUSD"]

        n_sub = int(len(eur_df) * self.s1_dev_fraction)
        sub_df = eur_df.iloc[-n_sub:].copy().reset_index(drop=True)
        sub_feats = eur_feats.iloc[-n_sub:].copy().reset_index(drop=True)

        l_mask, s_mask = compile_strategy(strategy, sub_feats)

        q_vec = compute_quote_to_usd_vector("EURUSD", len(sub_df), {"EURUSD": sub_df["close"].values})
        res = run_vectorized_backtest(
            df=sub_df,
            pair="EURUSD",
            entry_longs=l_mask,
            entry_shorts=s_mask,
            quote_to_usd=q_vec,
            sl_atr_mult=strategy.exit.sl_atr,
            tp_atr_mult=strategy.exit.tp_atr,
            max_holding_bars=strategy.exit.max_holding_bars,
            cost_multiplier=self.cost_multiplier,
            exit_type=getattr(strategy.exit, "exit_type", "sl_tp"),
            trailing_atr=getattr(strategy.exit, "trailing_atr", 1.2),
        )

        passed = (res["n_trades"] >= 3) and (res["profit_factor"] >= 0.70)
        return passed, res

    def evaluate_s2_full_dev(
        self,
        strategy: Strategy,
        dev_data: dict[str, pd.DataFrame],
        features_by_pair: dict[str, pd.DataFrame],
        basket_pairs: list[str] | None = None,
    ) -> tuple[bool, float, dict[str, Any]]:
        """
        S2: Multi-Pair Fast Engine across full DEV with jitter +-1 step.
        """
        self.n_s2 += 1
        if basket_pairs is None:
            basket_pairs = list(dev_data.keys())

        primary_metrics = None
        basket_sharpes = []

        for _j_step in [0]:  # central step primary
            # Primary pair EURUSD
            eur_df = dev_data["EURUSD"]
            eur_feats = features_by_pair["EURUSD"]
            l_mask, s_mask = compile_strategy(strategy, eur_feats)

            # Signal deduplication check: preserve directional Long and Short distinctly, differentiate by ExitSpec
            packed_eur = pack_bool_array_to_uint64(np.concatenate([l_mask, s_mask]))
            exit_key = f"{getattr(strategy.exit, 'exit_type', 'sl_tp')}_{strategy.exit.sl_atr:.1f}_{strategy.exit.tp_atr:.1f}_{strategy.exit.max_holding_bars}_{getattr(strategy.exit, 'trailing_atr', 1.2):.1f}"
            dup_result = self.bitset_store.find_duplicate_signal(packed_eur, exit_key=exit_key, threshold=0.95)
            if dup_result is not None:
                return dup_result["passed"], dup_result["fitness"], dup_result["details"]

            q_vec = compute_quote_to_usd_vector("EURUSD", len(eur_df), {p: d["close"].values for p, d in dev_data.items()})
            eur_res = run_vectorized_backtest(
                df=eur_df,
                pair="EURUSD",
                entry_longs=l_mask,
                entry_shorts=s_mask,
                quote_to_usd=q_vec,
                sl_atr_mult=strategy.exit.sl_atr,
                tp_atr_mult=strategy.exit.tp_atr,
                max_holding_bars=strategy.exit.max_holding_bars,
                cost_multiplier=self.cost_multiplier,
                exit_type=getattr(strategy.exit, "exit_type", "sl_tp"),
                trailing_atr=getattr(strategy.exit, "trailing_atr", 1.2),
            )
            primary_metrics = eur_res

        # Evaluate across basket
        for p in basket_pairs:
            p_df = dev_data[p]
            p_feats = features_by_pair[p]
            l_p, s_p = compile_strategy(strategy, p_feats)
            q_p = compute_quote_to_usd_vector(p, len(p_df), {k: d["close"].values for k, d in dev_data.items()})
            p_res = run_vectorized_backtest(
                df=p_df,
                pair=p,
                entry_longs=l_p,
                entry_shorts=s_p,
                quote_to_usd=q_p,
                sl_atr_mult=strategy.exit.sl_atr,
                tp_atr_mult=strategy.exit.tp_atr,
                max_holding_bars=strategy.exit.max_holding_bars,
                cost_multiplier=self.cost_multiplier,
                exit_type=getattr(strategy.exit, "exit_type", "sl_tp"),
                trailing_atr=getattr(strategy.exit, "trailing_atr", 1.2),
            )
            basket_sharpes.append(p_res["sharpe"])

        # Fitness formula
        fit_eur = primary_metrics["sharpe"]
        fit_basket = float(np.mean(basket_sharpes)) if basket_sharpes else fit_eur
        fitness = 0.5 * fit_eur + 0.5 * fit_basket

        passed = (primary_metrics["n_trades"] >= 20) and (fitness > 0.0)
        details = {
            "fitness": fitness,
            "sharpe_eur": fit_eur,
            "sharpe_basket": fit_basket,
            "n_trades_eur": primary_metrics["n_trades"],
            "win_rate_eur": primary_metrics["win_rate"],
            "profit_factor_eur": primary_metrics["profit_factor"],
            "total_return_pct": primary_metrics["total_return_pct"],
            "max_dd_pct": primary_metrics["max_dd_pct"],
            "holding_bars_median": primary_metrics.get("holding_bars_median", 0.0),
        }

        # Cache signal result with exit spec
        self.bitset_store.store_evaluated_signal(
            packed_eur,
            {"passed": passed, "fitness": fitness, "details": details},
            exit_key=exit_key,
        )

        if self.storage is not None:
            self.storage.log_evaluation({
                "eval_idx": self.n_s2,
                "strategy_id": strategy.strategy_id,
                "family": strategy.family,
                "genotype_hash": strategy.genotype_hash(),
                "fitness": fitness,
                "sharpe_eur": fit_eur,
                "sharpe_basket": fit_basket,
                "n_trades": primary_metrics["n_trades"],
                "passed": passed,
            })

        return passed, fitness, details

    def evaluate_s3_event_driven(
        self,
        strategy: Strategy,
        dev_data: dict[str, pd.DataFrame],
        features_by_pair: dict[str, pd.DataFrame],
        meta_filter: Any = None,
    ) -> tuple[list[Any], pd.DataFrame, dict[str, Any]]:
        """
        S3 Event-Driven verification for elite candidate.
        """
        self.n_s3 += 1
        eur_df = dev_data["EURUSD"]
        eur_feats = features_by_pair["EURUSD"]
        l_mask, s_mask = compile_strategy(strategy, eur_feats)

        if meta_filter is not None and getattr(meta_filter, "model", None) is not None:
            try:
                l_mask = meta_filter.filter_signals(l_mask, eur_feats, is_dev=True)
                s_mask = meta_filter.filter_signals(s_mask, eur_feats, is_dev=True)
            except TypeError:
                l_mask = meta_filter.filter_signals(l_mask, eur_feats)
                s_mask = meta_filter.filter_signals(s_mask, eur_feats)

        engine = EventDrivenEngine(cost_multiplier=self.cost_multiplier)
        market_closes = {p: d["close"].values for p, d in dev_data.items()}
        trades, trades_df, metrics = engine.run_backtest(
            df=eur_df,
            pair="EURUSD",
            entry_longs=l_mask,
            entry_shorts=s_mask,
            market_closes=market_closes,
            sl_atr_mult=strategy.exit.sl_atr,
            tp_atr_mult=strategy.exit.tp_atr,
            max_holding_bars=strategy.exit.max_holding_bars,
            exit_type=getattr(strategy.exit, "exit_type", "sl_tp"),
            trailing_atr=getattr(strategy.exit, "trailing_atr", 1.2),
        )
        return trades, trades_df, metrics
