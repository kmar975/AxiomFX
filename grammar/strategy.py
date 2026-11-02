"""
DSL Strategy Grammar Definitions.
Encodes the structural formula:
STRATEGY = [CONTEXT] -> [SETUP (window M)] -> TRIGGER -> ENTRY -> EXIT -> SIZING.
Enforces orthogonality (<3 conditions from different sources), symmetry, and strict parameter bounds.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any

REGIME_CEILING_FEATURES = {
    # Volatility and squeeze
    "s5_atr20_pct",
    "s5_bb_width_pct",
    "s5_squeeze_ratio_pct",
    "s5_squeeze_on",
    "s5_gk_vol_pct",
    "s5_ratio_5_20_pct",
    "s5_ratio_10_40_pct",
    "s5_ratio_20_80_pct",
    "s5_atr_ratio_5_20_pct",
    "s5_atr_ratio_10_40_pct",
    "s5_atr_ratio_20_80_pct",
    "s5_adx_14",
    # Microstructure & Spread
    "s8_spread_dev_pct",
    "s8_choppiness_10",
    "s8_choppiness_20",
    "s8_dir_efficiency",
    # Price Action & Levels: Non-directional distances and consolidation
    "s4_dist_round100_pct",
    "s4_dist_round50_pct",
    "s3_consol_frac_pct",
    "s3_retrace_frac_pct",
    "s1_adx_14",
    "s1_er_10_pct",
    # Cyclical & Time
    "s6_hour_sin",
    "s6_hour_cos",
    "s6_dow_sin",
    "s6_dow_cos",
    # Statistical series properties
    "s7_return_entropy",
    "s7_hurst_rs",
    "s7_variance_ratio_5_pct",
    "s7_autocorr_lag1_pct",
    "s7_skew_20_pct",
    "s7_kurt_20_pct",
    # Regimes & Institutional liquidity
    "s10_regime_p_high_vol",
    "s11_adr14_exhaustion_pct",
    "s11_prior_day_comp",
    "s11_prior_day_compression",
    "tier3_smc_premium_discount",
}

NON_INVERTIBLE_BINARY_FEATURES = {
    "s2_inside_bar",
    "s2_outside_bar",
    "s2_double_inside_bar",
    "s2_nr4",
    "s2_nr7",
    "s2_wr7",
    "s2_seed_doji",
    "s5_squeeze_on",
    "s6_session_asia",
    "s6_session_london",
    "s6_session_ny",
    "s6_session_london_fix",
    "s8_volume_spike_1p5x",
    "s8_volume_spike_2p0x",
    "s8_volume_spike_2p5x",
    "s8_volume_spike_3p0x",
    "s8_low_volume_pullback",
    "s11_prior_day_comp",
    "s11_prior_day_compression",
    "tier3_cup_and_handle",
}

SIGNED_DIRECTIONAL_FEATURES = {
    "s2_seed_three_bar_rev",
    "s2_seed_fakey",
    "s3_structure_state",
    "s3_bos",
    "s3_choch",
    "s3_liquidity_sweep",
    "s3_sweep_20",
    "s3_sweep_60",
    "s6_asian_breakout",
    "s6_judas_swing",
    "s11_pwh_pwl_sweep",
    "s12_wick_absorption",
    "s1_supertrend_10_3_dir",
    "s1_supertrend_14_2_dir",
    "s1_ema_stack_20_50_200",
    "mtf_unanimous",
    "mtf_hierarchical",
    "tier3_smc_fair_value_gaps",
    "tier3_smc_order_blocks",
    "tier3_harmonic_gartley",
    "tier3_harmonic_bat",
    "tier3_harmonic_butterfly",
    "tier3_harmonic_crab",
    "tier3_harmonic_abcd",
}


def is_signed_directional_feature(feat: str) -> bool:
    clean = feat.removeprefix("htf_")
    return clean in SIGNED_DIRECTIONAL_FEATURES


def is_binary_feature(feat: str) -> bool:
    if is_signed_directional_feature(feat):
        return False
    clean = feat.removeprefix("htf_")
    if clean in NON_INVERTIBLE_BINARY_FEATURES:
        return True
    bin_keywords = [
        "bull", "bear", "inside_bar", "outside_bar", "engulf", "fvg", "pin",
        "doji", "fakey", "prior", "nr4", "nr7", "wr7", "cross_up", "cross_dn",
        "_mr_lower", "_mr_upper", "_bo_upper", "_bo_lower", "spike_", "ema_stack"
    ]
    return any(k in feat for k in bin_keywords)


@dataclass
class Condition:
    feature: str
    source: str  # s1..s10
    operator: str  # gt, lt, between, cross_above, cross_below, event
    threshold: float  # e.g. 10.0, 20.0, 30.0, 70.0, 80.0, 90.0
    direction_mode: str = "continuation"  # continuation or fade
    threshold_high: float | None = None  # for 'between'
    is_regime_ceiling: bool = False

    def __post_init__(self):
        # Auto-detect non-invertible binary regime / session / distance conditions
        clean_feat = self.feature.removeprefix("htf_")
        if (
            self.feature in NON_INVERTIBLE_BINARY_FEATURES
            or clean_feat in NON_INVERTIBLE_BINARY_FEATURES
            or self.feature in REGIME_CEILING_FEATURES
            or clean_feat in REGIME_CEILING_FEATURES
        ):
            self.is_regime_ceiling = True

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["feature"] = str(self.feature)
        d["source"] = str(self.source)
        d["operator"] = str(self.operator)
        d["threshold"] = float(self.threshold)
        if self.threshold_high is not None:
            d["threshold_high"] = float(self.threshold_high)
        d["direction_mode"] = str(self.direction_mode)
        d["is_regime_ceiling"] = bool(self.is_regime_ceiling)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Condition:
        return cls(**d)


@dataclass
class ExitSpec:
    exit_type: str = "sl_tp"  # sl_tp, breakeven_trail, dynamic_mean_reversion, pre_rollover_flat
    sl_atr: float = 1.5
    tp_atr: float = 2.5
    max_holding_bars: int = 24
    trailing_atr: float = 1.2

    def to_dict(self) -> dict[str, Any]:
        return {
            "exit_type": str(self.exit_type),
            "sl_atr": float(self.sl_atr),
            "tp_atr": float(self.tp_atr),
            "max_holding_bars": int(self.max_holding_bars),
            "trailing_atr": float(self.trailing_atr),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ExitSpec:
        return cls(
            exit_type=str(d.get("exit_type", "sl_tp")),
            sl_atr=float(d.get("sl_atr", 1.5)),
            tp_atr=float(d.get("tp_atr", 2.5)),
            max_holding_bars=int(d.get("max_holding_bars", 24)),
            trailing_atr=float(d.get("trailing_atr", 1.2)),
        )


@dataclass
class SizingSpec:
    vol_target: float = 0.10
    risk_per_trade_pct: float = 0.01

    def to_dict(self) -> dict[str, Any]:
        return {
            "vol_target": float(self.vol_target),
            "risk_per_trade_pct": float(self.risk_per_trade_pct),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SizingSpec:
        return cls(
            vol_target=float(d.get("vol_target", 0.10)),
            risk_per_trade_pct=float(d.get("risk_per_trade_pct", 0.01)),
        )


@dataclass
class Strategy:
    strategy_id: str
    family: str  # F1..F15
    trigger: Condition
    context: Condition | None = None
    context_2: Condition | None = None
    setup: Condition | None = None
    setup_expiry: int = 5
    setup_2: Condition | None = None
    setup_2_expiry: int = 5
    exit: ExitSpec = field(default_factory=ExitSpec)
    sizing: SizingSpec = field(default_factory=SizingSpec)
    symmetric: bool = True
    timeframe: str = "H1"

    @classmethod
    def from_dict(cls, d: dict[str, Any], sid: str = "STRAT") -> Strategy:
        trig = Condition.from_dict(d["trigger"])
        ctx = Condition.from_dict(d["context"]) if d.get("context") else None
        ctx2 = Condition.from_dict(d["context_2"]) if d.get("context_2") else None
        setup = Condition.from_dict(d["setup"]) if d.get("setup") else None
        setup_exp = int(d.get("setup_expiry", 5)) if d.get("setup_expiry") is not None else 5
        setup2 = Condition.from_dict(d["setup_2"]) if d.get("setup_2") else None
        setup2_exp = int(d.get("setup_2_expiry", 5)) if d.get("setup_2_expiry") is not None else 5

        exit_spec = ExitSpec.from_dict(d.get("exit", {}))
        sizing_spec = SizingSpec.from_dict(d.get("sizing", {}))

        return cls(
            strategy_id=d.get("strategy_id", sid),
            family=d.get("family", "F1_Trend"),
            trigger=trig,
            context=ctx,
            context_2=ctx2,
            setup=setup,
            setup_expiry=setup_exp,
            setup_2=setup2,
            setup_2_expiry=setup2_exp,
            exit=exit_spec,
            sizing=sizing_spec,
            symmetric=bool(d.get("symmetric", True)),
            timeframe=str(d.get("timeframe", "H1")),
        )

    def genotype_dict(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "context": self.context.to_dict() if self.context else None,
            "context_2": self.context_2.to_dict() if self.context_2 else None,
            "setup": self.setup.to_dict() if self.setup else None,
            "setup_expiry": self.setup_expiry if self.setup else None,
            "setup_2": self.setup_2.to_dict() if self.setup_2 else None,
            "setup_2_expiry": self.setup_2_expiry if self.setup_2 else None,
            "trigger": self.trigger.to_dict(),
            "exit": self.exit.to_dict(),
            "sizing": self.sizing.to_dict(),
            "symmetric": self.symmetric,
            "timeframe": self.timeframe,
        }

    def genotype_hash(self) -> str:
        s = json.dumps(self.genotype_dict(), sort_keys=True)
        return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]

    def is_valid_directional(self) -> bool:
        """
        Ensures the strategy has a genuinely directional TRIGGER condition.
        Regime ceiling, session, and volatility envelope conditions (is_regime_ceiling=True)
        are valid filters for context and setup, but the trigger itself must be directional.
        """
        return not self.trigger.is_regime_ceiling

    def validate_rules(self) -> bool:
        """
        Anti-noise limits:
        - Max 5 conditions (k=1..5 depth)
        - All from distinct information sources S1..S12 / HTF
        - Trigger must be directional (not a regime ceiling)
        """
        conditions = [self.trigger]
        for c in [self.context, self.context_2, self.setup, self.setup_2]:
            if c is not None:
                conditions.append(c)

        if len(conditions) > 5:
            return False

        if not self.is_valid_directional():
            return False

        sources = [c.source for c in conditions]
        return len(sources) == len(set(sources))
