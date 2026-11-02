"""
Multi-Timeframe Matrix and Live Signals Compiler (§ 10.2).
Processes all 7 pairs × 8 timeframes (56 markets), re-evaluates top strategies,
remaps historical chart markers to 0..N_bars-1, computes transaction paths,
sparklines, and produces output/live_signals_multi_tf.json.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from data.download import prepare_all_timeframes
from data.specs import ALL_PAIRS, ALL_TIMEFRAMES
from engine.costs import compute_quote_to_usd_vector, get_pair_specs

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_RUNS_DIR = BASE_DIR / "output" / "runs" if (BASE_DIR / "output" / "runs").exists() else Path("output/runs")
DEFAULT_TIMEFRAMES_DIR = BASE_DIR / "output" / "timeframes" if (BASE_DIR / "output" / "timeframes").exists() else Path("output/timeframes")
DEFAULT_OUTPUT_JSON = BASE_DIR / "output" / "live_signals_multi_tf.json"
from engine.live_scanner import scan_strategy_on_live_candle
from engine.vectorized import run_vectorized_backtest
from features.store import FeatureStore
from grammar.compiler import compile_strategy
from grammar.strategy import Condition, ExitSpec, Strategy


def _dict_to_strategy(g: dict[str, Any], sid: str = "STRAT") -> Strategy:
    return Strategy.from_dict(g, sid=sid)


def parse_strategy_from_card(card_path: Path) -> Strategy:
    """
    Parses Strategy object from an experiment card.md file or strategy.yaml.
    """
    sid = card_path.parent.name
    strat_yaml = card_path.parent / "strategy.yaml"
    if strat_yaml.exists():
        try:
            with open(strat_yaml, "r", encoding="utf-8") as f:
                y = yaml.safe_load(f)
            if isinstance(y, dict):
                return _dict_to_strategy(y, sid)
        except Exception:
            pass

    txt = card_path.read_text(encoding="utf-8")
    fam_match = re.search(r"\*\*Family:\*\*\s*`([^`]+)`", txt)
    family = fam_match.group(1) if fam_match else "F1_Trend"

    def _parse_cond(pat: str) -> tuple[Condition | None, int | None]:
        m = re.search(pat, txt)
        if not m:
            return None, None
        feat = m.group(1)
        op = m.group(2)
        th = float(m.group(3))
        win = int(m.group(4)) if len(m.groups()) >= 4 and m.group(4) else None
        source = feat.split("_")[0] if feat.startswith("s") else "s1"
        return Condition(feature=feat, source=source, operator=op, threshold=th), win

    ctx, _ = _parse_cond(r"-\s*\*\*Context 1:\*\*\s*`([^`]+)`\s*(\w+)\s*([\d\.-]+)")
    ctx2, _ = _parse_cond(r"-\s*\*\*Context 2:\*\*\s*`([^`]+)`\s*(\w+)\s*([\d\.-]+)")
    setup, win1 = _parse_cond(r"-\s*\*\*Setup 1:\*\*\s*`([^`]+)`\s*(\w+)\s*([\d\.-]+)(?:\s*\(window M=(\d+)\))?")
    setup2, win2 = _parse_cond(r"-\s*\*\*Setup 2:\*\*\s*`([^`]+)`\s*(\w+)\s*([\d\.-]+)(?:\s*\(window M=(\d+)\))?")
    trig, _ = _parse_cond(r"-\s*\*\*Trigger:\*\*\s*`([^`]+)`\s*(\w+)\s*([\d\.-]+)")
    if trig is None:
        trig = Condition("s1_rsi_14", "s1", "gt", 70.0)

    exit_type_m = re.search(r"Type\s*`([^`]+)`", txt)
    exit_type = exit_type_m.group(1) if exit_type_m else "sl_tp"

    sl_m = re.search(r"SL\s*=\s*`([\d\.]+)\s*ATR`", txt)
    tp_m = re.search(r"TP\s*=\s*`([\d\.]+)\s*ATR`", txt)
    trail_m = re.search(r"Trailing\s*=\s*`([\d\.]+)\s*ATR`", txt)
    mh_m = re.search(r"Max Holding\s*=\s*`(\d+)\s*bars`", txt)

    sl_atr = float(sl_m.group(1)) if sl_m else 1.5
    tp_atr = float(tp_m.group(1)) if tp_m else 2.5
    trailing_atr = float(trail_m.group(1)) if trail_m else 1.2
    mh = int(mh_m.group(1)) if mh_m else 24

    sym_m = re.search(r"Symmetric:\s*`(\w+)`", txt)
    symmetric = (sym_m.group(1).lower() != "false") if sym_m else True
    tf_m = re.search(r"Timeframe:\s*`(\w+)`", txt)
    timeframe = tf_m.group(1) if tf_m else "H1"

    return Strategy(
        strategy_id=sid,
        family=family,
        trigger=trig,
        context=ctx,
        context_2=ctx2,
        setup=setup,
        setup_expiry=win1 or 5,
        setup_2=setup2,
        setup_2_expiry=win2 or 5,
        exit=ExitSpec(exit_type=exit_type, sl_atr=sl_atr, tp_atr=tp_atr, max_holding_bars=mh, trailing_atr=trailing_atr),
        symmetric=symmetric,
        timeframe=timeframe,
    )


def _load_from_yaml(yaml_path: Path) -> list[Strategy]:
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    cand_list = data.get("portfolio_candidates", [])
    strategies = []
    for item in cand_list:
        g = item.get("genotype", {})
        sid = item.get("strategy_id", "STRAT_1")
        fam = item.get("family", "F1_Trend")
        if "family" not in g:
            g["family"] = fam
        strat = _dict_to_strategy(g, sid=sid)
        strat.status = item.get("status", "C")
        strat.rank = item.get("rank", 1)
        strategies.append(strat)
    return strategies


def _save_candidates_yaml(strategies: list[Strategy], out_path: Path) -> None:
    cand_list = []
    for rank, s in enumerate(strategies, start=1):
        cand_list.append({
            "rank": rank,
            "strategy_id": s.strategy_id,
            "family": s.family,
            "genotype": s.genotype_dict(),
        })
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.dump({"portfolio_candidates": cand_list}, f, sort_keys=False)


def load_candidate_strategies(
    candidates_path: Path | None = None,
    run_id: str | None = None,
    runs_dir: Path = DEFAULT_RUNS_DIR,
) -> tuple[list[Strategy], str]:
    """
    Loads candidate strategies from candidates.yaml, from a specific run_id,
    or generates top 5 fallbacks. Returns (strategies, resolved_run_id).
    """
    resolved_run_id = run_id or "latest"

    # 1. If run_id is given, inspect output/runs/{run_id}/
    if run_id:
        r_dir = runs_dir / run_id
        if r_dir.exists():
            run_cand_yaml = r_dir / "candidates.yaml"
            if run_cand_yaml.exists():
                strats = _load_from_yaml(run_cand_yaml)
                if strats:
                    return strats, run_id

            # Parse from experiments/*/card.md
            cards = sorted(r_dir.glob("experiments/*/card.md"))
            if cards:
                lb_file = r_dir / "leaderboard" / "current.json"
                ordered_ids = []
                if lb_file.exists():
                    try:
                        lb_data = json.loads(lb_file.read_text(encoding="utf-8"))
                        ordered_ids = [item.get("strategy_id") for item in lb_data]
                    except (json.JSONDecodeError, OSError, KeyError, TypeError, ValueError):
                        ordered_ids = []

                card_map = {c.parent.name: c for c in cards}
                strats = []
                for s_id in ordered_ids:
                    if s_id in card_map:
                        strats.append(parse_strategy_from_card(card_map[s_id]))
                for s_id, card_p in card_map.items():
                    if s_id not in ordered_ids:
                        strats.append(parse_strategy_from_card(card_p))

                if strats:
                    _save_candidates_yaml(strats, run_cand_yaml)
                    return strats, run_id

    # 2. Try candidates_path or output/candidates.yaml
    target_yaml = candidates_path or Path("output/candidates.yaml")
    if target_yaml.exists():
        strats = _load_from_yaml(target_yaml)
        if strats:
            return strats, resolved_run_id

    # 3. Standard top 5 fallbacks if not yet generated
    s1 = Strategy("ALPHA_TREND_1", "F1_Trend", Condition("s1_rsi_14", "s1", "gt", 70.0), exit=ExitSpec(1.5, 2.5))
    s2 = Strategy("ALPHA_FADE_2", "F3_MeanReversion", Condition("s1_rsi_14", "s1", "lt", 30.0, direction_mode="fade"), exit=ExitSpec(1.0, 1.5))
    s3 = Strategy("ALPHA_SQUEEZE_3", "F2_Breakout", Condition("s1_donchian_pos", "s1", "gt", 80.0), exit=ExitSpec(2.0, 3.0))
    s4 = Strategy("ALPHA_PIN_4", "F5_Candles", Condition("s2_seed_pin_bull", "s2", "event", 1.0), exit=ExitSpec(1.2, 2.4))
    s5 = Strategy("ALPHA_CURR_5", "F12_CurrencyStrength", Condition("s9_rel_strength_diff_pct", "s9", "gt", 70.0), exit=ExitSpec(1.5, 2.5))
    return [s1, s2, s3, s4, s5], resolved_run_id


def compile_56_markets_payload(
    timeframes_dir: Path = DEFAULT_TIMEFRAMES_DIR,
    output_json_path: Path = DEFAULT_OUTPUT_JSON,
    n_bars_window: int = 600,
    run_id: str | None = None,
) -> dict[str, Any]:
    """
    Compiles complete multi-timeframe payload for Pro Terminal GUI.
    """
    timeframes_dir.mkdir(parents=True, exist_ok=True)
    output_json_path.parent.mkdir(parents=True, exist_ok=True)

    strategies, resolved_run_id = load_candidate_strategies(run_id=run_id)
    has_tier3 = any(
        (c is not None and getattr(c, "feature", "").startswith("tier3_"))
        for s in strategies
        for c in [s.trigger, s.context, getattr(s, "context_2", None), s.setup, getattr(s, "setup_2", None)]
    )
    feature_store = FeatureStore(enable_tier3=has_tier3)

    strategy_meta = {}
    for idx, s in enumerate(strategies, start=1):
        strategy_meta[s.strategy_id] = {
            "rank": getattr(s, "rank", idx),
            "strategy_id": s.strategy_id,
            "family": s.family,
            "tier": getattr(s, "status", "C"),
            "timeframe": s.timeframe,
            "sl_atr": s.exit.sl_atr,
            "tp_atr": s.exit.tp_atr,
            "max_holding_bars": getattr(s.exit, "max_holding_bars", 24),
        }

    payload: dict[str, Any] = {
        "generated_at": pd.Timestamp.now("UTC").isoformat(),
        "run_id": resolved_run_id,
        "strategies": [s.genotype_dict() for s in strategies],
        "strategy_ids": [s.strategy_id for s in strategies],
        "strategy_meta": strategy_meta,
        "pairs": ALL_PAIRS,
        "timeframes": ALL_TIMEFRAMES,
        "pair_signals": {p: "NEUTRAL" for p in ALL_PAIRS},
        "markets": {},
        "radar_matrix": {},
    }

    print("[COMPILER] Processing 56 markets (7 pairs × 8 timeframes)...")

    meta_filters = {}
    from search.meta_labeling import load_meta_filter
    meta_dir = BASE_DIR / "output" / "meta_filters"
    for strat in strategies:
        mf_path = meta_dir / f"{strat.strategy_id}.joblib"
        if not mf_path.exists() and resolved_run_id and resolved_run_id != "latest":
            mf_path = DEFAULT_RUNS_DIR / resolved_run_id / "experiments" / strat.strategy_id / "meta_filter.joblib"
        if mf_path.exists():
            mf = load_meta_filter(mf_path)
            if mf:
                meta_filters[strat.strategy_id] = mf

    for tf in ALL_TIMEFRAMES:
        # Pre-load dataframes with causal warmup buffer (n_bars_window + 600)
        warmup_len = 600
        tf_raw_basket: dict[str, pd.DataFrame] = {}
        tf_warmup_offsets: dict[str, int] = {}
        tf_sliced_basket: dict[str, pd.DataFrame] = {}
        tf_basket_closes: dict[str, np.ndarray] = {}

        for pair in ALL_PAIRS:
            market_key = f"{pair}_{tf}"
            csv_path = timeframes_dir / f"{market_key}.csv"
            if not csv_path.exists():
                prepare_all_timeframes()

            df = pd.read_csv(csv_path)
            df["time"] = pd.to_datetime(df["time"], utc=True)
            n_total = len(df)
            n_slice = min(n_total, n_bars_window)
            n_load = min(n_total, n_slice + warmup_len)
            load_offset = n_total - n_load
            w_df = df.iloc[load_offset:].copy().reset_index(drop=True)
            tf_raw_basket[pair] = w_df
            warmup_offset = n_load - n_slice
            tf_warmup_offsets[pair] = warmup_offset

            s_df = w_df.iloc[warmup_offset:].copy().reset_index(drop=True)
            tf_sliced_basket[pair] = s_df
            tf_basket_closes[pair] = s_df["close"].values

        for pair in ALL_PAIRS:
            market_key = f"{pair}_{tf}"
            if pair not in payload["radar_matrix"]:
                payload["radar_matrix"][pair] = {}

            spec = get_pair_specs(pair)
            dec = int(spec["dec"])
            pip_unit = float(spec["pip_unit"])
            pip_mult = float(spec["pip_mult"])

            # Compute features with causal warmup buffer and slice strictly to display window
            w_df = tf_raw_basket[pair]
            raw_feats = feature_store.compute_all_features_for_pair(w_df, pair, basket_data=tf_raw_basket)
            warmup_offset = tf_warmup_offsets[pair]
            feats = raw_feats.iloc[warmup_offset:].copy().reset_index(drop=True)
            sliced_df = tf_sliced_basket[pair]

            # Format candles for HTML5 Canvas
            candles = []
            for i in range(len(sliced_df)):
                candles.append({
                    "time": str(sliced_df["time"].iloc[i]),
                    "open": float(sliced_df["open"].iloc[i]),
                    "high": float(sliced_df["high"].iloc[i]),
                    "low": float(sliced_df["low"].iloc[i]),
                    "close": float(sliced_df["close"].iloc[i]),
                    "spread": float(sliced_df["spread"].iloc[i]) if "spread" in sliced_df.columns else 1.5,
                    "volume": int(sliced_df["tick_volume"].iloc[i]) if "tick_volume" in sliced_df.columns else 100,
                })

            market_entry: dict[str, Any] = {
                "pair": pair,
                "tf": tf,
                "dec": dec,
                "pip_unit": pip_unit,
                "pip_mult": pip_mult,
                "candles": candles,
                "strategies": {},
            }

            # Evaluate each of the 5 strategies on this market
            dominant_state = "NEUTRAL"

            for s_idx, strat in enumerate(strategies):
                mf = meta_filters.get(strat.strategy_id)
                scan_res = scan_strategy_on_live_candle(strat, feats, sliced_df, pair, tf, meta_filter=mf)

                # Update dominant state for radar matrix
                if scan_res["state"] in ("BUY", "SELL"):
                    dominant_state = scan_res["state"]
                elif scan_res["state"] == "ARMED" and dominant_state == "NEUTRAL":
                    dominant_state = "ARMED"

                # Historical signals on this sliced chart
                l_mask, s_mask = compile_strategy(strat, feats)
                if mf is not None and getattr(mf, "model", None) is not None:
                    l_mask = mf.filter_signals(l_mask, feats)
                    s_mask = mf.filter_signals(s_mask, feats)

                # Real backtest with accurate cross-pair exchange rates and strategy exit spec
                q_vec = compute_quote_to_usd_vector(pair, len(sliced_df), tf_basket_closes)
                bt_res = run_vectorized_backtest(
                    df=sliced_df,
                    pair=pair,
                    entry_longs=l_mask,
                    entry_shorts=s_mask,
                    quote_to_usd=q_vec,
                    sl_atr_mult=strat.exit.sl_atr,
                    tp_atr_mult=strat.exit.tp_atr,
                    max_holding_bars=strat.exit.max_holding_bars,
                    exit_type=getattr(strat.exit, "exit_type", "sl_tp"),
                    trailing_atr=getattr(strat.exit, "trailing_atr", 1.2),
                )

                # Extract REAL non-overlapping transaction paths & entry markers
                entry_bars = bt_res.get("entry_bars", np.array([], dtype=np.int64))
                exit_bars = bt_res.get("exit_bars", np.array([], dtype=np.int64))
                trade_pnls = bt_res.get("trade_pnls", np.array([], dtype=np.float64))
                trade_dirs = bt_res.get("trade_directions", np.array([], dtype=np.int32))
                entry_prices = bt_res.get("entry_prices", np.array([], dtype=np.float64))
                exit_prices = bt_res.get("exit_prices", np.array([], dtype=np.float64))
                trade_exit_reasons = bt_res.get("exit_reasons", np.array([], dtype=np.int8))
                reason_map = {0: "SL", 1: "TP", 2: "TIME", 3: "ROLLOVER"}

                transaction_paths = []
                chart_markers = []
                n_real_trades = len(entry_bars)

                for t_i in range(n_real_trades):
                    e_b = int(entry_bars[t_i])
                    x_b = int(exit_bars[t_i])
                    d_code = int(trade_dirs[t_i])
                    is_long = (d_code == 1)
                    pnl_usd = float(trade_pnls[t_i])

                    if len(entry_prices) > t_i:
                        e_price = float(entry_prices[t_i])
                    else:
                        e_price = float(sliced_df["open"].iloc[e_b])

                    if len(exit_prices) > t_i:
                        x_price = float(exit_prices[t_i])
                    else:
                        x_price = float(sliced_df["close"].iloc[x_b]) if x_b < len(sliced_df) else float(sliced_df["close"].iloc[-1])

                    pnl_pips = round((x_price - e_price if is_long else e_price - x_price) * pip_mult, 1)

                    if len(trade_exit_reasons) > t_i:
                        r_code = int(trade_exit_reasons[t_i])
                        exit_reason = reason_map.get(r_code, "TP" if pnl_pips > 0 else "SL")
                    else:
                        exit_reason = "TP" if pnl_pips > 0 else "SL"

                    transaction_paths.append({
                        "entry_idx": e_b,
                        "exit_idx": x_b,
                        "type": "BUY" if is_long else "SELL",
                        "entry_price": round(e_price, dec),
                        "exit_price": round(x_price, dec),
                        "exit_reason": exit_reason,
                        "pnl_pips": pnl_pips,
                        "pnl_usd": round(pnl_usd, 2),
                        "is_win": (pnl_usd > 0),
                    })

                    chart_markers.append({
                        "index": e_b,
                        "type": "BUY" if is_long else "SELL",
                        "price": round(e_price, dec),
                        "date": str(sliced_df["time"].iloc[e_b]),
                    })

                # Live signal marker for current candle if active
                if scan_res["state"] in ("BUY", "SELL"):
                    chart_markers.append({
                        "index": len(sliced_df) - 1,
                        "type": scan_res["state"],
                        "price": round(float(sliced_df["close"].iloc[-1]), dec),
                        "date": str(sliced_df["time"].iloc[-1]),
                        "is_live": True,
                    })

                # Synchronized bar-by-bar equity (% return) and drawdown series
                eq_curve = bt_res.get("equity_curve", np.full(len(sliced_df), 100000.0))
                equity_pct_series = [round(float((v - 100000.0) / 1000.0), 3) for v in eq_curve]
                peaks = np.maximum.accumulate(eq_curve)
                drawdown_series = [round(float((p - v) / p * 100.0), 3) if p > 0 else 0.0 for p, v in zip(peaks, eq_curve)]

                # 20-point sparkline for cards (true % return)
                step = max(1, len(equity_pct_series) // 20)
                sparkline = [equity_pct_series[i] for i in range(0, len(equity_pct_series), step)][:20]

                market_entry["strategies"][strat.strategy_id] = {
                    "live_state": scan_res["state"],
                    "max_holding_bars": getattr(strat.exit, "max_holding_bars", 24),
                    "entry_price": scan_res["entry_price"],
                    "sl_price": scan_res["sl_price"],
                    "tp_price": scan_res["tp_price"],
                    "sl_pips": scan_res["sl_pips"],
                    "tp_pips": scan_res["tp_pips"],
                    "alert_desc": scan_res["alert_desc"],
                    "chart_markers": chart_markers,
                    "transaction_paths": transaction_paths,
                    "equity_series": equity_pct_series,
                    "drawdown_series": drawdown_series,
                    "metrics": {
                        "n_trades": bt_res["n_trades"],
                        "win_rate": bt_res["win_rate"],
                        "profit_factor": bt_res["profit_factor"],
                        "sharpe": bt_res["sharpe"],
                        "total_return_pct": bt_res["total_return_pct"],
                        "max_dd_pct": bt_res["max_dd_pct"],
                    },
                    "sparkline": sparkline,
                }

            payload["markets"][market_key] = market_entry
            payload["radar_matrix"][pair][tf] = dominant_state
            if dominant_state in ("BUY", "SELL"):
                payload["pair_signals"][pair] = dominant_state
            elif dominant_state == "ARMED" and payload["pair_signals"].get(pair) == "NEUTRAL":
                payload["pair_signals"][pair] = "ARMED"

    # Save JSON payload
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"[COMPILER] Successfully wrote live payload for 56 markets -> {output_json_path}")
    return payload


def get_or_compile_run_payload(
    run_id: str,
    runs_dir: Path = DEFAULT_RUNS_DIR,
    timeframes_dir: Path = DEFAULT_TIMEFRAMES_DIR,
    min_bars: int = 500,
) -> dict[str, Any]:
    """
    Returns cached live payload for run_id, or compiles and caches it if not yet existing
    or if the cached version is outdated (< min_bars candles or missing strategy_meta).
    """
    run_dir = runs_dir / run_id
    if not run_dir.exists():
        raise FileNotFoundError(f"Run folder not found: {run_id}")

    cached_payload_path = run_dir / "live_signals_multi_tf.json"
    if cached_payload_path.exists():
        try:
            with open(cached_payload_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            sample_candles = len(data.get("markets", {}).get("EURUSD_H1", {}).get("candles", []))
            has_meta = "strategy_meta" in data and "equity_series" in data.get("markets", {}).get("EURUSD_H1", {}).get("strategies", {}).get(data.get("strategy_ids", [""])[0], {})
            if sample_candles >= min_bars and has_meta:
                return data
            print(f"[COMPILER] Cache for {run_id} is outdated ({sample_candles} candles < {min_bars} or missing meta). Recompiling 600 bars...")
        except Exception as err:
            print(f"[COMPILER] Cache read error for {run_id}: {err}. Recompiling...")

    # Compile now with full 600-bar window and store into run cache
    print(f"[COMPILER] Compiling 56 markets (600 bars) for {run_id}...")
    payload = compile_56_markets_payload(
        timeframes_dir=timeframes_dir,
        output_json_path=cached_payload_path,
        n_bars_window=600,
        run_id=run_id,
    )
    return payload


if __name__ == "__main__":
    compile_56_markets_payload()
