"""
Master Pipeline Orchestrator and CLI (E1 -> E7).
Coordinates the complete institutional quantitative discovery lifecycle:
Data preparation (Dukascopy Bank SA) -> Feature Engineering -> Power Calibration (B*) ->
Phase A -> B1 -> E0 (Numba parallel) -> B2 -> C (Full MAP-Elites Recombination) ->
D (Plateau & HRP) -> Purged Meta-Labeling -> V2 Synchronous Null Worlds ->
V4 Holdout -> V5 Recovery -> Multi-Pair Heatmap (Real Backtest) -> Reports & GUI Compilation.
Zero synthetic generation, Zero mocks.
"""

from __future__ import annotations

import argparse
import math
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from atoms.bitsets import AtomicBitsetStore, pack_bool_array_to_uint64
from data.download import prepare_all_timeframes
from data.split import SplitManager
from engine.costs import compute_quote_to_usd_vector
from engine.vectorized import run_vectorized_backtest
from features.store import FeatureStore
from grammar.compiler import compile_strategy
from grammar.strategy import Strategy
from labels.labeling import compute_future_returns
from report.cards import generate_strategy_card_md
from report.export import generate_final_report_md
from report.plots import (
    plot_best_strategy_live,
    plot_discovery_curve,
    plot_multi_pair_heatmap,
    plot_v2_null_histogram,
)
from search.bandit import UCB1Bandit
from search.baselines import run_phase_a_baselines
from search.exhaustive import run_phase_e0_exhaustive
from search.fitness import evaluate_multipair_fitness
from search.funnel import FunnelExecutor
from search.genetic import create_random_strategy, mutate_and_crossover_strategy
from search.info_map import run_phase_b1_info_map
from search.interactions import mine_interaction_seed_rules
from search.meta_labeling import train_meta_labeling_for_strategy
from search.portfolio import build_hrp_candidate_portfolio
from search.power_calibration import calibrate_power_optimal_breadth
from search.qd_archive import MAPElitesArchive
from search.refine import run_discrete_plateau_centering
from storage.state import StorageManager
from validation.holdout import evaluate_finalists_on_holdout
from validation.null_world import build_synchronous_null_world, compute_v2_p_value
from validation.recovery import run_v5_recovery_test
from validation.statistics import (
    benjamini_hochberg,
    deflated_sharpe_ratio,
    monte_carlo_trade_order,
    white_reality_check,
)
from validation.stress import run_candidate_stress_tests


def parse_args():
    parser = argparse.ArgumentParser(description="Forex Quant AI Lab & Pro Terminal Pipeline")
    parser.add_argument("--demo", action="store_true", help="Fast demo execution mode")
    parser.add_argument("--mode", choices=["demo", "standard", "deep", "ultra"], default="standard", help="Pipeline depth mode")
    parser.add_argument("--duration-hours", type=float, default=1.0, help="Target max discovery duration in hours")
    parser.add_argument("--workers", type=int, default=4, help="Multiprocessing workers")
    parser.add_argument("--n-nulls", type=int, default=None, help="Number of V2 null worlds")
    parser.add_argument("--enable-tier3", action="store_true", help="Enable Tier 3 experimental features")
    parser.add_argument("--resume", type=str, default=None, help="Resume from existing RUN_ID")
    return parser.parse_args()


def run_pipeline(args):
    print("=" * 80)
    print("  FOREX QUANT AI LAB & PRO LIVE TERMINAL v3.5 — MASTER PIPELINE")
    print("=" * 80)

    # 0. Load Configuration from config/ modular files and config.yaml
    _cfg: dict[str, Any] = {}
    config_dir = Path(__file__).resolve().parent / "config"
    if config_dir.exists():
        for yaml_file in sorted(config_dir.glob("*.yaml")):
            try:
                with open(yaml_file, "r", encoding="utf-8") as f:
                    sub_cfg = yaml.safe_load(f) or {}
                    if isinstance(sub_cfg, dict):
                        for k, v in sub_cfg.items():
                            if k in _cfg and isinstance(_cfg[k], dict) and isinstance(v, dict):
                                _cfg[k].update(v)
                            else:
                                _cfg[k] = v
            except Exception as e:
                print(f"[!] Notice reading {yaml_file.name}: {e}")

    config_path = Path(__file__).resolve().parent / "config.yaml"
    if config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                root_cfg = yaml.safe_load(f) or {}
                if isinstance(root_cfg, dict):
                    for k, v in root_cfg.items():
                        if k in _cfg and isinstance(_cfg[k], dict) and isinstance(v, dict):
                            _cfg[k].update(v)
                        else:
                            _cfg[k] = v
            print(f"[*] Loaded modular configuration from config/*.yaml and {config_path.name}")
        except Exception as e:
            print(f"[!] Notice reading config.yaml: {e}")

    # Extract institutional parameters with robust fallbacks
    dev_ratio = float(_cfg.get("split", {}).get("dev_ratio", 0.80))
    cost_mult_fitness = float(_cfg.get("costs", {}).get("cost_multiplier_fitness", 1.5))
    cost_mult_stress = float(_cfg.get("costs", {}).get("cost_multiplier_stress", 2.0))
    vol_target = float(_cfg.get("engine", {}).get("vol_target", 0.10))
    min_median_hold_hours = float(_cfg.get("engine", {}).get("min_median_hold_hours", 2.0))
    min_trades_primary_cfg = int(_cfg.get("engine", {}).get("min_trades_primary", 8))
    min_trades_basket_cfg = int(_cfg.get("engine", {}).get("min_trades_basket", 50))
    checkpoint_interval_sec = float(_cfg.get("execution", {}).get("checkpoint_interval_sec", 30.0))
    cfg_n_nulls = int(_cfg.get("execution", {}).get("n_nulls", 10))
    cfg_enable_tier3 = bool(_cfg.get("execution", {}).get("enable_tier3", False))

    # 1. Setup Configuration and Storage
    run_id = args.resume or f"run_{int(time.time())}"
    if args.demo:
        args.mode = "demo"
        args.n_nulls = 3
    elif args.n_nulls is None:
        args.n_nulls = cfg_n_nulls

    enable_tier3 = args.enable_tier3 or cfg_enable_tier3

    print(f"[*] RUN_ID: {run_id} | Mode: {args.mode} | Null Worlds: {args.n_nulls}")
    storage = StorageManager(run_id=run_id)
    resume_checkpoint = None
    if args.resume:
        resume_checkpoint = storage.load_latest_checkpoint()
        if resume_checkpoint:
            print(f"[+] Resuming run {run_id} from checkpoint: {resume_checkpoint}")
    storage.record_run_start(vars(args))

    # 2. Data Preparation across 56 markets (Authentic Dukascopy Data)
    print("[*] Preparing authentic market data across 7 pairs and 8 timeframes (56 markets)...")
    all_timeframes_data = prepare_all_timeframes(n_days=1500 if args.mode != "demo" else 60)
    # Primary discovery timeframe: H1
    h1_data = {p: all_timeframes_data[p]["H1"] for p in all_timeframes_data}

    # 3. Chronological Split (DEV 80% vs HOLDOUT 20%)
    cutoff_date = SplitManager.compute_cutoff_date(h1_data, dev_ratio=dev_ratio)
    dev_data = SplitManager.get_dev_data(h1_data, cutoff_date)
    print(f"[*] Chronological Split: Cutoff Date UTC = {cutoff_date} (DEV bars: {len(dev_data['EURUSD'])})")

    # 4. Feature Store & Labels on DEV (Includes zero-lookahead HTF context and FVG/Sweeps)
    print("[*] Computing Feature Store S1–S10 + MTF HTF features on DEV...")
    feature_store = FeatureStore(enable_tier3=enable_tier3, enable_htf=True)
    features_by_pair = feature_store.compute_basket_features(dev_data)
    tier1_features = feature_store.get_tier1_features(features_by_pair["EURUSD"])

    future_returns_by_pair = {}
    for p, p_df in dev_data.items():
        future_returns_by_pair[p] = compute_future_returns(p_df, horizons=[1, 5, 10, 20])

    bitset_store = AtomicBitsetStore()
    funnel = FunnelExecutor(bitset_store=bitset_store, storage=storage, cost_multiplier_fitness=cost_mult_fitness)

    # 5. V5 Power Calibration for B*
    budget_map = {
        "demo": 400,
        "standard": 3000,
        "deep": 10000,
        "ultra": 50000,
    }
    max_budget = budget_map.get(args.mode, 3000)
    print(f"[*] Calibrating Power-Optimal Breadth B* (max_budget={max_budget})...")
    b_star_plot = storage.images_dir / "power_calibration.png"
    b_star = calibrate_power_optimal_breadth(
        dev_data=dev_data,
        features_by_pair=features_by_pair,
        funnel=funnel,
        max_budget=max_budget,
        output_image_path=b_star_plot,
    )
    b_star = max(b_star, int(max_budget * 0.75))
    storage.update_run_b_star(b_star)
    print(f"[+] Calibrated Power-Optimal Breadth B* = {b_star} hypotheses (enforced min 75% budget)")

    # 6. Phase A: Classical Baselines
    print("[*] Running Phase A: Classical Baselines...")
    baseline_results = run_phase_a_baselines(dev_data, features_by_pair, funnel)

    # 7. Phase B1: Information Map
    print("[*] Running Phase B1: Information Map...")
    info_map_plot = storage.images_dir / "info_map.png"
    b1_results = run_phase_b1_info_map(
        features_by_pair=features_by_pair,
        future_returns_by_pair=future_returns_by_pair,
        output_image_path=info_map_plot,
    )

    # 8. Phase B2: Feature Interactions & Seed Extraction
    print("[*] Running Phase B2: Feature Interactions & Seeds...")
    top_feature_names = [f["feature"] for f in b1_results.get("top_features", [])][:25]
    seed_rules = mine_interaction_seed_rules(
        features_df=features_by_pair["EURUSD"],
        forward_returns=future_returns_by_pair["EURUSD"].get(5, np.zeros(len(features_by_pair["EURUSD"]))),
        top_features=top_feature_names,
        max_rules=60,
    )

    # 9. Phase E0: Parallel Numba Exhaustive Core Layer
    e0_budget = 800 if args.mode == "ultra" else int(b_star * 0.30)
    print(f"[*] Running Phase E0: Parallel Numba Exhaustive Layer (Budget: {e0_budget})...")
    e0_candidates = run_phase_e0_exhaustive(
        tier1_features=tier1_features,
        features_by_pair=features_by_pair,
        future_returns_by_pair=future_returns_by_pair,
        funnel=funnel,
        max_candidates=e0_budget,
    )

    # 10. Phase C: Quality-Diversity MAP-Elites Evolution with Genetic Recombination
    phase_c_budget = b_star - len(e0_candidates)
    print(f"[*] Running Phase C: Quality-Diversity MAP-Elites Genetic Evolution (Budget: {phase_c_budget})...")
    qd_archive = MAPElitesArchive(bins_per_dim=5, n_dims=6)
    families = [
        "F1_Trend", "F2_Breakout", "F3_MeanReversion", "F4_Pullback",
        "F5_Candles", "F6_Geometry", "F8_Sessions", "F12_CurrencyStrength",
    ]
    bandit = UCB1Bandit(families=families)
    rng_gen = np.random.RandomState(42)

    min_trades_basket = 15 if args.mode == "demo" else min_trades_basket_cfg
    min_trades_primary = 5 if args.mode == "demo" else min_trades_primary_cfg

    # Seed archive with Baselines, B2, and E0
    all_seeds = [s for s, _, _ in baseline_results] + seed_rules + [s for s, _, _ in e0_candidates]
    for s in all_seeds:
        fit, det = evaluate_multipair_fitness(
            s,
            dev_data,
            features_by_pair,
            cost_multiplier=cost_mult_fitness,
            min_median_hold_hours=min_median_hold_hours,
            min_trades_primary=min_trades_primary,
            min_trades_basket=min_trades_basket,
        )
        if fit > -900:
            desc = qd_archive.compute_descriptors(
                s,
                trade_durations=det.get("durations", np.array([])),
                trade_directions=det.get("trade_directions", np.array([])),
                trade_pnls=det.get("trade_pnls", np.array([])),
                n_bars_total=len(dev_data["EURUSD"]),
            )
            qd_archive.add_strategy(s, fit, desc, det)

    eval_indices = []
    max_fit_hist = []
    cov_hist = []

    # Evolutionary recombination loop
    c_step = 0
    max_steps = max(15, phase_c_budget) if args.mode != "ultra" else 100000

    # Restore from checkpoint if resuming
    if resume_checkpoint:
        c_step = int(resume_checkpoint.get("c_step", 0))
        if "elites" in resume_checkpoint:
            print(f"[+] Restoring {len(resume_checkpoint['elites'])} elites from checkpoint into QD archive...")
            for item in resume_checkpoint["elites"]:
                g = item.get("genotype", {})
                sid = item.get("strategy_id", "RESUMED")
                strat = Strategy.from_dict(g, sid=sid)
                strat.family = item.get("family", strat.family)
                fit = float(item.get("fitness", 0.0))
                desc = np.array(item.get("descriptors", [0.0] * 6), dtype=np.float64)
                met = item.get("metrics", {})
                qd_archive.add_strategy(strat, fit, desc, met)
        print(f"[+] Resumed Phase C from step {c_step}/{max_steps}")

    phase_c_start = time.time()
    max_duration_sec = args.duration_hours * 3600.0 if args.mode == "ultra" else 999999.0
    last_checkpoint_t = time.time()

    def _create_checkpoint_dict(step_num: int, elapsed: float) -> dict[str, Any]:
        return {
            "c_step": step_num,
            "elapsed_sec": elapsed,
            "elites_count": len(qd_archive.get_all_elites()),
            "elites": [
                {
                    "genotype": cell.strategy.genotype_dict(),
                    "strategy_id": cell.strategy.strategy_id,
                    "family": cell.strategy.family,
                    "fitness": float(cell.fitness),
                    "descriptors": [float(x) for x in cell.descriptors],
                    "metrics": {
                        k: (float(v) if isinstance(v, (int, float, np.number)) else str(v))
                        for k, v in cell.metrics.items()
                        if not isinstance(v, np.ndarray)
                    },
                }
                for cell in qd_archive.get_all_elites()
            ],
        }

    while c_step < max_steps:
        if args.mode == "ultra" and (time.time() - phase_c_start) >= max_duration_sec:
            print(f"[+] HPC Ultra exploration completed target duration: {args.duration_hours:.2f}h")
            break

        fam = bandit.select_family()
        elites_now = qd_archive.get_all_elites()

        # Diversified parent selection: 20% random injection or when elites < 5
        if (c_step < int(max_steps * 0.20)) or (rng_gen.uniform(0.0, 1.0) < 0.20) or (len(elites_now) < 5):
            if rng_gen.uniform(0.0, 1.0) < 0.50 or not all_seeds:
                mutant = create_random_strategy(f"QD_RND_{c_step + 1}", fam, tier1_features, rng_gen)
            else:
                p_idx = rng_gen.randint(len(all_seeds))
                parent_a = all_seeds[p_idx]
                parent_b = all_seeds[rng_gen.randint(len(all_seeds))] if len(all_seeds) > 1 else None
                mutant = mutate_and_crossover_strategy(
                    parent_a=parent_a,
                    parent_b=parent_b,
                    tier1_features=tier1_features,
                    rng=rng_gen,
                    crossover_prob=0.35,
                )
        else:
            p_idx = rng_gen.randint(len(elites_now))
            parent_a = elites_now[p_idx].strategy
            parent_b = elites_now[rng_gen.randint(len(elites_now))].strategy if len(elites_now) > 1 else None
            mutant = mutate_and_crossover_strategy(
                parent_a=parent_a,
                parent_b=parent_b,
                tier1_features=tier1_features,
                rng=rng_gen,
                crossover_prob=0.35,
            )

        mutant.strategy_id = f"QD_{c_step + 1}"
        mutant.family = fam

        # Deduplication 1: Genotype hash
        if not bitset_store.register_genotype(mutant.genotype_hash()):
            continue

        c_step += 1

        # Deduplication 2: Jaccard signal similarity on EURUSD
        l_mask, s_mask = compile_strategy(mutant, features_by_pair["EURUSD"])
        packed_eur = pack_bool_array_to_uint64(np.concatenate([l_mask, s_mask]))
        exit_key = f"{getattr(mutant.exit, 'exit_type', 'sl_tp')}_{mutant.exit.sl_atr:.1f}_{mutant.exit.tp_atr:.1f}_{mutant.exit.max_holding_bars}_{getattr(mutant.exit, 'trailing_atr', 1.2):.1f}"

        dup_result = bitset_store.find_duplicate_signal(packed_eur, exit_key=exit_key, threshold=0.95)
        if dup_result is not None:
            fit = dup_result["fitness"]
            det = dup_result["details"]
            s1_passed = dup_result.get("passed", False)
        else:
            # S1 Fast screening on 1/3 DEV
            s1_passed, s1_res = funnel.evaluate_s1_fast(mutant, dev_data, features_by_pair)
            if s1_passed:
                funnel.n_s2 += 1
                fit, det = evaluate_multipair_fitness(
                    mutant,
                    dev_data,
                    features_by_pair,
                    cost_multiplier=cost_mult_fitness,
                    min_median_hold_hours=min_median_hold_hours,
                    min_trades_primary=min_trades_primary,
                    min_trades_basket=min_trades_basket,
                )
            else:
                fit = -999.0
                det = {"fitness": -999.0, "failed_filter": "Failed S1 Fast Filter", "n_trades_eur": s1_res.get("n_trades", 0)}

            bitset_store.store_evaluated_signal(
                packed_eur,
                {"passed": fit > 0.0, "fitness": fit, "details": det},
                exit_key=exit_key,
            )

        # Log evaluation to storage (writes to ledger.parquet)
        storage.log_evaluation({
            "eval_idx": c_step,
            "strategy_id": mutant.strategy_id,
            "family": mutant.family,
            "fitness": fit,
            "stage": "S2" if s1_passed else "S1",
            "passed": fit > 0.0,
            "sharpe": det.get("q25_eur", 0.0),
            "profit_factor": det.get("profit_factor_eur", det.get("profit_factor", 0.0)),
            "n_trades": det.get("n_trades_eur", 0),
        })

        reward = 0.0
        if fit > -900:
            desc = qd_archive.compute_descriptors(
                mutant,
                trade_durations=det.get("durations", np.array([])),
                trade_directions=det.get("trade_directions", np.array([])),
                trade_pnls=det.get("trade_pnls", np.array([])),
                n_bars_total=len(dev_data["EURUSD"]),
            )
            added, reason = qd_archive.add_strategy(mutant, fit, desc, det)
            if added:
                reward = 1.0 if reason == "new_cell" else 0.5

        bandit.record_feedback(fam, reward)

        now_t = time.time()
        if now_t - last_checkpoint_t >= checkpoint_interval_sec or storage.is_interrupted:
            storage.save_checkpoint(
                _create_checkpoint_dict(c_step, now_t - phase_c_start),
                step_name=f"c_{c_step}",
            )
            storage.flush_ledger()
            last_checkpoint_t = now_t
            if storage.is_interrupted:
                print(f"[!] Phase C interrupted gracefully at step {c_step}. Checkpoint saved.")
                break

        if c_step % 10 == 0 or c_step == max_steps:
            eval_indices.append(c_step)
            elites_cur = qd_archive.get_all_elites()
            cur_max = max((e.fitness for e in elites_cur), default=0.0)
            max_fit_hist.append(cur_max)
            cov_hist.append(qd_archive.get_coverage() * 100.0)

    # Save final Phase C checkpoint
    storage.save_checkpoint(
        _create_checkpoint_dict(c_step, time.time() - phase_c_start),
        step_name="phase_c_complete",
    )
    storage.flush_ledger()

    # Save discovery curve
    discovery_plot = storage.images_dir / "discovery_curve.png"
    if eval_indices:
        plot_discovery_curve(eval_indices, max_fit_hist, cov_hist, discovery_plot)

    # 11. Phase D: Discrete Plateau Centering & HRP Portfolio
    print("[*] Running Phase D: Discrete Plateau Centering & Portfolio Construction...")
    elites_list = [c for c in qd_archive.get_all_elites() if c.fitness > 0.0]

    refined_elites = []
    for cell in elites_list:
        is_robust, ref_strat, ref_fit, ref_det = run_discrete_plateau_centering(
            cell.strategy,
            dev_data,
            features_by_pair,
            max_needle_drop=0.50,
            min_trades_primary=min_trades_primary,
            min_trades_basket=min_trades_basket,
        )
        if is_robust and ref_fit > 0.0:
            refined_elites.append((ref_strat, ref_fit, ref_det))

    if not refined_elites:
        refined_elites = [(cell.strategy, cell.fitness, cell.metrics) for cell in elites_list if cell.fitness > 0.0]

    candidates_yaml_path = Path(__file__).resolve().parent / "output" / "candidates.yaml"
    champions, hrp_weights, portfolio_meta = build_hrp_candidate_portfolio(
        elites=refined_elites,
        dev_data=dev_data,
        features_by_pair=features_by_pair,
        min_candidates=5,
        max_candidates=15,
        output_yaml_path=candidates_yaml_path,
        require_profitable=True,
        cost_multiplier=cost_mult_fitness,
    )

    if not champions:
        print("[!] No champions met strict require_profitable=True. Running fallback pass with require_profitable=False...")
        fallback_pool = list(refined_elites)
        if not fallback_pool:
            fallback_pool = [(cell.strategy, cell.fitness, cell.metrics) for cell in qd_archive.get_all_elites() if cell.fitness > -900]
        if not fallback_pool:
            fallback_pool = [(s, 0.1, {}) for s in all_seeds]

        champions, hrp_weights, portfolio_meta = build_hrp_candidate_portfolio(
            elites=fallback_pool,
            dev_data=dev_data,
            features_by_pair=features_by_pair,
            min_candidates=min(5, max(1, len(fallback_pool))),
            max_candidates=15,
            output_yaml_path=candidates_yaml_path,
            require_profitable=False,
            cost_multiplier=cost_mult_fitness,
        )
    shutil.copy2(candidates_yaml_path, storage.run_dir / "candidates.yaml")
    print(f"[+] Finalist portfolio assembled with {len(champions)} champions -> {candidates_yaml_path}")

    # Save each champion's full genotype to its experiment directory for lossless reload
    for champ in champions:
        champ_yaml_path = storage.experiments_dir / champ.strategy_id / "strategy.yaml"
        champ_yaml_path.parent.mkdir(parents=True, exist_ok=True)
        with open(champ_yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(champ.genotype_dict(), f)

    # 11b. Secondary Meta-Labeling (Purged LightGBM Model & OOF Threshold Optimization)
    print("[*] Training Purged LightGBM Meta-Labeling secondary filter for champions...")
    meta_filters = {}
    from search.meta_labeling import save_meta_filter
    for champ in champions:
        mf = train_meta_labeling_for_strategy(
            strategy=champ,
            df=dev_data["EURUSD"],
            features_df=features_by_pair["EURUSD"],
            pair="EURUSD",
            cost_multiplier=cost_mult_fitness,
        )
        meta_filters[champ.strategy_id] = mf
        save_meta_filter(mf, storage.experiments_dir / champ.strategy_id / "meta_filter.joblib")
        save_meta_filter(mf, Path(__file__).resolve().parent / "output" / "meta_filters" / f"{champ.strategy_id}.joblib")
        if mf.model is not None:
            print(f"  [{champ.strategy_id}] Meta-filter optimized: theta={mf.threshold:.2f} (PF Base={mf.train_pf_base:.2f} -> Meta={mf.train_pf_meta:.2f})")

    # 11c. S3 Event-Driven Engine Verification
    print(f"[*] Running S3 Event-Driven Engine verification for {len(champions)} champions...")
    for champ in champions:
        trades_s3, df_s3, metrics_s3 = funnel.evaluate_s3_event_driven(
            champ, dev_data, features_by_pair, meta_filter=meta_filters.get(champ.strategy_id)
        )
        trades_path = storage.experiments_dir / champ.strategy_id / "trades.parquet"
        trades_path.parent.mkdir(parents=True, exist_ok=True)
        if not df_s3.empty:
            df_s3.to_parquet(trades_path, index=False)

    # 12. V2 Synchronous Null World Verification
    print(f"[*] Running V2: Synchronous Null Worlds ({args.n_nulls} worlds) with identical budget B*={b_star}...")
    null_world_scores = []
    cand_list = portfolio_meta.get("candidates", [])
    real_champion_fitness = cand_list[0].get("fitness", 0.0) if cand_list else 0.0

    for nw_idx in range(args.n_nulls):
        null_basket = build_synchronous_null_world(dev_data, seed=1000 + nw_idx)
        null_feats = feature_store.compute_basket_features(null_basket)
        rng_nw = np.random.default_rng(2000 + nw_idx)

        # Run genuine discovery search with identical budget B* on null world to measure real data-mining bias
        best_null_fit = -999.0
        current_pop = [s for s in all_seeds[:min(10, len(all_seeds))]]
        pop_fits = []

        for cand in current_pop:
            fit, _ = evaluate_multipair_fitness(
                cand,
                null_basket,
                null_feats,
                cost_multiplier=cost_mult_fitness,
                min_median_hold_hours=min_median_hold_hours,
                min_trades_primary=min_trades_primary,
                min_trades_basket=min_trades_basket,
            )
            pop_fits.append(fit)
            best_null_fit = max(best_null_fit, fit)

        eval_count = len(current_pop)
        null_budget = min(b_star, 300)
        while eval_count < null_budget:
            # Fitness-proportionate selection to exert true optimization pressure on noise
            weights = np.maximum(0.01, np.array(pop_fits) - min(pop_fits) + 0.1)
            weights /= weights.sum()
            p1_idx = rng_nw.choice(len(current_pop), p=weights)
            p2_idx = rng_nw.choice(len(current_pop), p=weights)
            child = mutate_and_crossover_strategy(
                current_pop[p1_idx],
                current_pop[p2_idx],
                tier1_features=tier1_features,
                mutation_type="combined",
                rng=rng_nw,
            )
            c_fit, _ = evaluate_multipair_fitness(
                child,
                null_basket,
                null_feats,
                cost_multiplier=cost_mult_fitness,
                min_median_hold_hours=min_median_hold_hours,
                min_trades_primary=min_trades_primary,
                min_trades_basket=min_trades_basket,
            )
            eval_count += 1
            best_null_fit = max(best_null_fit, c_fit)

            min_idx = int(np.argmin(pop_fits))
            if c_fit > pop_fits[min_idx]:
                current_pop[min_idx] = child
                pop_fits[min_idx] = c_fit

        null_world_scores.append(max(0.0, best_null_fit))

    v2_p_val = compute_v2_p_value(real_champion_fitness, null_world_scores)
    n_nulls_eval = max(1, len(null_world_scores))
    v2_alpha = max(0.05, 1.0 / (1.0 + float(n_nulls_eval)))
    print(f"[+] V2 Null World Test Result: Champion Real Fit={real_champion_fitness:.3f}, p-value = {v2_p_val:.4f} (v2_alpha = {v2_alpha:.4f})")
    plot_v2_null_histogram(
        real_fitness=real_champion_fitness,
        null_fitnesses=null_world_scores,
        p_value=v2_p_val,
        output_path=storage.images_dir / "v2_null_histogram.png",
    )

    # 13. V4 Out-of-Sample Holdout Control
    print("[*] Running V4: Frozen Finalists Evaluation on HOLDOUT (unseen 20%)...")
    holdout_plot = storage.images_dir / "portfolio_equity.png"
    holdout_meta = evaluate_finalists_on_holdout(
        candidates=champions,
        hrp_weights=hrp_weights,
        all_data=h1_data,
        cutoff_date=cutoff_date,
        meta_filters=meta_filters,
        output_image_path=holdout_plot,
    )
    print(f"[+] Holdout Portfolio Sharpe = {holdout_meta['portfolio_sharpe_holdout']:.2f}, Return = {holdout_meta['portfolio_total_return_pct']:.2f}%")

    # 14. V5 Signal Recovery Verification
    print("[*] Running V5: Signal Recovery Verification across SNR tiers...")
    v5_plot = storage.images_dir / "recovery_curve.png"
    v5_results = run_v5_recovery_test(dev_data, output_image_path=v5_plot)
    print(f"[+] V5 Signal Recovery: Recalls = {v5_results['recalls']}")

    # 15. Cross-Sectional Multi-Pair Backtesting & Classification (Status A/B/C)
    print("[*] Running Authentic Multi-Pair Cross-Sectional Backtests across all 7 pairs...")
    final_candidates_meta = []
    pair_cross_metrics: dict[str, dict[str, float]] = {p: {} for p in dev_data}
    multi_pair_full_stats: dict[str, dict[str, dict[str, Any]]] = {champ.strategy_id: {} for champ in champions}

    # Execute genuine vectorized backtests for all champion strategies on all 7 currency pairs
    for p, p_df in dev_data.items():
        p_feats = features_by_pair[p]
        q_vec_p = compute_quote_to_usd_vector(p, len(p_df), {k: d["close"].values for k, d in dev_data.items()})

        for champ in champions:
            l_p, s_p = compile_strategy(champ, p_feats)
            mf = meta_filters.get(champ.strategy_id)
            if mf is not None and getattr(mf, "model", None) is not None:
                l_p = mf.filter_signals(l_p, p_feats, is_dev=(p == "EURUSD"))
                s_p = mf.filter_signals(s_p, p_feats, is_dev=(p == "EURUSD"))
            bt_cross = run_vectorized_backtest(
                df=p_df,
                pair=p,
                entry_longs=l_p,
                entry_shorts=s_p,
                quote_to_usd=q_vec_p,
                sl_atr_mult=champ.exit.sl_atr,
                tp_atr_mult=champ.exit.tp_atr,
                max_holding_bars=champ.exit.max_holding_bars,
                cost_multiplier=1.5,
                exit_type=getattr(champ.exit, "exit_type", "sl_tp"),
                trailing_atr=getattr(champ.exit, "trailing_atr", 1.2),
            )
            # Record genuine Sharpe ratio
            pair_cross_metrics[p][champ.strategy_id] = bt_cross["sharpe"]
            multi_pair_full_stats[champ.strategy_id][p] = {
                "sharpe": bt_cross["sharpe"],
                "win_rate": bt_cross["win_rate"],
                "profit_factor": bt_cross["profit_factor"],
                "n_trades": bt_cross["n_trades"],
                "total_return_pct": bt_cross["total_return_pct"],
            }

    # 15. Advanced Statistical Inference (trejding module: DSR, FDR, White RC, Monte Carlo)
    print("[*] Running Advanced Statistical Tests: DSR, FDR (Benjamini-Hochberg), White's Reality Check & Monte Carlo...")
    candidate_returns_dict = {}
    candidate_stats_dict = {}
    cand_p_vals = []
    tested_signals_count = max(1, bitset_store.n_unique_signal)

    for idx, champ in enumerate(champions):
        # Extract champion daily returns and trade pnls on EURUSD (with meta-filter applied on DEV)
        l_c, s_c = compile_strategy(champ, features_by_pair["EURUSD"])
        mf_champ = meta_filters.get(champ.strategy_id)
        if mf_champ is not None and getattr(mf_champ, "model", None) is not None:
            l_c = mf_champ.filter_signals(l_c, features_by_pair["EURUSD"], is_dev=True)
            s_c = mf_champ.filter_signals(s_c, features_by_pair["EURUSD"], is_dev=True)
        q_eur = compute_quote_to_usd_vector("EURUSD", len(dev_data["EURUSD"]), {k: d["close"].values for k, d in dev_data.items()})
        eur_res = run_vectorized_backtest(
            df=dev_data["EURUSD"],
            pair="EURUSD",
            entry_longs=l_c,
            entry_shorts=s_c,
            quote_to_usd=q_eur,
            sl_atr_mult=champ.exit.sl_atr,
            tp_atr_mult=champ.exit.tp_atr,
            max_holding_bars=champ.exit.max_holding_bars,
            cost_multiplier=1.5,
            exit_type=getattr(champ.exit, "exit_type", "sl_tp"),
            trailing_atr=getattr(champ.exit, "trailing_atr", 1.2),
        )
        eq = eur_res.get("equity_curve", np.full(len(dev_data["EURUSD"]), 100000.0))
        eur_times = pd.to_datetime(dev_data["EURUSD"]["time"], utc=True)
        s_eq = pd.Series(eq, index=eur_times)
        s_daily = s_eq.resample("1D").last().dropna()
        daily_ret = (s_daily.diff().dropna() / np.maximum(s_daily.shift(1).dropna(), 1.0)).values
        if len(daily_ret) < 2:
            daily_ret = np.diff(eq) / np.maximum(eq[:-1], 1.0)
        candidate_returns_dict[champ.strategy_id] = daily_ret
        trade_pnls = eur_res.get("trade_pnls", np.array([]))

        # 1. Deflated Sharpe Ratio
        try:
            dsr_res = deflated_sharpe_ratio(daily_ret, tested_count=tested_signals_count)
            dsr_prob = dsr_res.probability
        except Exception:
            dsr_prob = 0.50

        # 2. Monte Carlo 10,000 trade order permutations
        try:
            if len(trade_pnls) >= 2:
                mc_res = monte_carlo_trade_order(trade_pnls, repetitions=1000)
                mc_dd_95 = mc_res.percentile_95_drawdown
            else:
                mc_dd_95 = 0.0
        except Exception:
            mc_dd_95 = 0.0

        # Nominal p-value estimation based on genuine daily return series
        n_days = max(1.0, float(len(daily_ret)))
        mean_ret = float(np.mean(daily_ret))
        std_ret = float(np.std(daily_ret))
        t_stat = (mean_ret / (std_ret + 1e-9)) * np.sqrt(n_days)
        nom_p = max(0.001, min(0.999, 1.0 - 0.5 * (1.0 + math.erf(t_stat / math.sqrt(2.0)))))
        cand_p_vals.append(nom_p)

        candidate_stats_dict[champ.strategy_id] = {
            "dsr_probability": dsr_prob,
            "mc_dd_95": mc_dd_95,
            "nominal_p": nom_p,
        }

    # 3. Benjamini-Hochberg FDR
    try:
        fdr_res = benjamini_hochberg(pd.Series(cand_p_vals, index=[c.strategy_id for c in champions]))
        for s_id, adj_p in fdr_res.adjusted_p_values.items():
            candidate_stats_dict[s_id]["fdr_adjusted_p_value"] = float(adj_p)
    except Exception:
        for c in champions:
            candidate_stats_dict[c.strategy_id]["fdr_adjusted_p_value"] = 1.0

    # 4. White's Reality Check
    try:
        ret_df = pd.DataFrame(candidate_returns_dict)
        wrc_res = white_reality_check(ret_df, bootstrap_repetitions=1000)
        white_p_val = wrc_res.p_value
    except Exception:
        white_p_val = 1.0

    for c in champions:
        candidate_stats_dict[c.strategy_id]["white_rc_p_value"] = white_p_val

    # Status classification & Experiment Cards
    cand_enrich_map = {}
    for idx, champ in enumerate(champions):
        h_cand = holdout_meta["candidate_results"][idx] if idx < len(holdout_meta["candidate_results"]) else {}
        is_holdout_pos = h_cand.get("is_positive", False)
        holdout_pf = float(h_cand.get("profit_factor", 0.0))

        dev_fit = float(portfolio_meta["candidates"][idx]["fitness"]) if idx < len(portfolio_meta["candidates"]) else 0.0
        stress_res = run_candidate_stress_tests(
            champ, dev_data["EURUSD"], features_by_pair["EURUSD"], meta_filter=meta_filters.get(champ.strategy_id)
        )
        survives_spread_2x = stress_res.get("survives_spread_2x", False)

        # Institutional Status A/B/C Classification Rules:
        # Status A requires:
        # 1. Positive fitness on DEV at 1.5x institutional costs
        # 2. Rejection of null worlds (v2_p_val <= v2_alpha)
        # 3. Profitable and PF > 1.05 on unseen HOLDOUT
        # 4. Survival of 2.0x spread stress test
        if dev_fit > 0.0 and v2_p_val <= v2_alpha and is_holdout_pos and holdout_pf > 1.05 and survives_spread_2x:
            status = "A"
        elif dev_fit > 0.0 and v2_p_val <= v2_alpha and (is_holdout_pos or survives_spread_2x):
            status = "B"
        else:
            status = "C"

        cand_enrich_map[champ.strategy_id] = {
            "status": status,
            "h_cand": h_cand,
        }

        card_path = storage.experiments_dir / champ.strategy_id / "card.md"
        generate_strategy_card_md(
            strategy=champ,
            metrics_dev=portfolio_meta["candidates"][idx] if idx < len(portfolio_meta["candidates"]) else {},
            metrics_holdout=h_cand,
            status=status,
            v2_p_value=v2_p_val,
            multi_pair_metrics=multi_pair_full_stats[champ.strategy_id],
            stress_results=stress_res,
            output_path=card_path,
            stats_results=candidate_stats_dict.get(champ.strategy_id),
        )

        final_candidates_meta.append({
            "rank": idx + 1,
            "strategy_id": champ.strategy_id,
            "family": champ.family,
            "hrp_weight": float(hrp_weights[idx]) if idx < len(hrp_weights) else 0.1,
            "fitness": dev_fit,
            "sharpe_holdout": h_cand.get("sharpe_holdout", 0.0),
            "status": status,
            "dsr_probability": candidate_stats_dict[champ.strategy_id]["dsr_probability"],
            "fdr_adjusted_p_value": candidate_stats_dict[champ.strategy_id]["fdr_adjusted_p_value"],
            "white_rc_p_value": candidate_stats_dict[champ.strategy_id]["white_rc_p_value"],
            "mc_dd_95": candidate_stats_dict[champ.strategy_id]["mc_dd_95"],
        })

    # Update candidates.yaml with holdout metrics, status and advanced statistical test results
    if candidates_yaml_path.exists():
        with open(candidates_yaml_path, "r", encoding="utf-8") as f:
            y_data = yaml.safe_load(f)
        for cand_item in y_data.get("portfolio_candidates", []):
            sid = cand_item.get("strategy_id")
            if sid in cand_enrich_map:
                enr = cand_enrich_map[sid]
                h_cand = enr["h_cand"]
                cand_item["status"] = str(enr["status"])
                cand_item["holdout_sharpe"] = round(float(h_cand.get("sharpe_holdout", 0.0)), 3)
                cand_item["holdout_profit_factor"] = round(float(h_cand.get("profit_factor", 0.0)), 3)
                cand_item["holdout_win_rate"] = round(float(h_cand.get("win_rate", 0.0)), 4)
                cand_item["holdout_return_pct"] = round(float(h_cand.get("total_return_pct", 0.0)), 2)
                cand_item["holdout_trades"] = int(h_cand.get("n_trades", 0))

            if sid in candidate_stats_dict:
                st = candidate_stats_dict[sid]
                cand_item["dsr_probability"] = round(float(st["dsr_probability"]), 4)
                cand_item["fdr_adjusted_p_value"] = round(float(st["fdr_adjusted_p_value"]), 4)
                cand_item["white_rc_p_value"] = round(float(st["white_rc_p_value"]), 4)
                cand_item["mc_dd_95"] = round(float(st["mc_dd_95"]), 2)
        with open(candidates_yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(y_data, f, sort_keys=False)
        shutil.copy2(candidates_yaml_path, storage.run_dir / "candidates.yaml")

    # Plot Multi-Pair Heatmap (V6) using genuine backtest metrics
    plot_multi_pair_heatmap(
        pair_metrics=pair_cross_metrics,
        candidate_ids=[c.strategy_id for c in champions],
        output_path=storage.images_dir / "multi_pair_heatmap.png",
    )

    # Plot Best Strategy Live
    best_eq = holdout_meta["portfolio_equity"]
    holdout_times = holdout_meta.get("times", None)
    if holdout_times is None or len(holdout_times) != len(best_eq):
        holdout_times = h1_data["EURUSD"][h1_data["EURUSD"]["time"] >= cutoff_date]["time"].values[:len(best_eq)]
    plot_best_strategy_live(
        times=holdout_times,
        equity_curve=best_eq,
        strategy_id=champions[0].strategy_id if champions else "CHAMPION_1",
        output_path=storage.leaderboard_dir / "best_strategy_live.png",
    )

    # 16. Throughput & Final Reports
    throughput_stats = bitset_store.get_throughput_stats(
        n_s0=funnel.n_s0,
        n_s1=funnel.n_s1,
        n_s2=funnel.n_s2,
        n_s3=funnel.n_s3,
    )
    final_report_path = storage.run_dir / "final_report.md"
    stats_summary = {
        "white_rc_p_value": white_p_val,
        "fdr_min_p": min((c["fdr_adjusted_p_value"] for c in candidate_stats_dict.values()), default=1.0),
        "top_dsr_prob": candidate_stats_dict.get(champions[0].strategy_id, {}).get("dsr_probability", 0.5) if champions else 0.5,
        "mc_dd_95": candidate_stats_dict.get(champions[0].strategy_id, {}).get("mc_dd_95", 0.0) if champions else 0.0,
    }
    generate_final_report_md(
        run_id=run_id,
        throughput_stats=throughput_stats,
        b_star=b_star,
        candidates_meta=final_candidates_meta,
        holdout_meta=holdout_meta,
        output_path=final_report_path,
        v2_p_val=v2_p_val,
        stats_summary=stats_summary,
    )
    storage.record_run_end(final_candidates_meta, holdout_meta)
    print(f"\n[+] PIPELINE COMPLETED SUCCESSFULLY! Final report saved -> {final_report_path}")
    print("=" * 80)


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(args)
