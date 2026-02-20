"""
APEX SIGNAL™ — Strategy Engine Groups
Groups related strategies into engines that produce confluence scores.
"""
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone
import pandas as pd

from apex_signal.strategies.base import StrategySignal
from apex_signal.strategies.registry import get_strategy_registry
from apex_signal.utils.logger import get_logger
from apex_signal.utils.helpers import clamp

logger = get_logger("strategy_engine")


class EngineResult:
    """Result from a strategy engine group."""

    def __init__(
        self,
        engine_name: str,
        signal: str,
        confluence_score: float,
        weighted_reason: str,
        strategy_signals: List[StrategySignal],
    ):
        self.engine_name = engine_name
        self.signal = signal
        self.confluence_score = clamp(confluence_score, 0, 100)
        self.weighted_reason = weighted_reason
        self.strategy_signals = strategy_signals
        self.timestamp = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "engine_name": self.engine_name,
            "signal": self.signal,
            "confluence_score": round(self.confluence_score, 2),
            "weighted_reason": self.weighted_reason,
            "strategy_count": len(self.strategy_signals),
            "strategies": [s.to_dict() for s in self.strategy_signals],
            "timestamp": self.timestamp,
        }


class StrategyEngineGroup:
    """A group of related strategies that work together as an engine."""

    def __init__(self, name: str, strategy_names: List[str], weights: Optional[Dict[str, float]] = None):
        self.name = name
        self.strategy_names = strategy_names
        self.weights = weights or {n: 1.0 for n in strategy_names}

    def evaluate(self, df: pd.DataFrame, symbol: str = "") -> EngineResult:
        """Evaluate all strategies in this engine and produce a confluence result."""
        registry = get_strategy_registry()
        signals = registry.evaluate_group(df, self.strategy_names, symbol)

        if not signals:
            return EngineResult(self.name, "HOLD", 0, "No strategies produced signals", [])

        # Calculate weighted confluence
        buy_score = 0.0
        sell_score = 0.0
        total_weight = 0.0
        reasons = []

        for sig in signals:
            weight = self.weights.get(sig.strategy_name, 1.0)
            total_weight += weight

            if sig.signal == "BUY":
                buy_score += sig.confidence * weight
                reasons.append(f"[BUY] {sig.strategy_name}: {sig.reason}")
            elif sig.signal == "SELL":
                sell_score += sig.confidence * weight
                reasons.append(f"[SELL] {sig.strategy_name}: {sig.reason}")

        if total_weight == 0:
            return EngineResult(self.name, "HOLD", 0, "Zero total weight", signals)

        buy_avg = buy_score / total_weight
        sell_avg = sell_score / total_weight

        # Determine engine signal
        if buy_avg > sell_avg and buy_avg > 15:
            signal = "BUY"
            confluence = buy_avg
        elif sell_avg > buy_avg and sell_avg > 15:
            signal = "SELL"
            confluence = sell_avg
        else:
            signal = "HOLD"
            confluence = max(buy_avg, sell_avg)

        # Build weighted reason
        active_reasons = [r for r in reasons if signal.upper() in r.upper()] or reasons
        weighted_reason = f"{self.name} ({signal}, confluence={confluence:.0f}): " + "; ".join(active_reasons[:3])

        return EngineResult(self.name, signal, confluence, weighted_reason, signals)


# Pre-defined engine groups
ENGINE_DEFINITIONS = {
    "trend_engine": {
        "strategies": ["ema_crossover", "sma_crossover", "multi_tf_alignment",
                       "vwap_trend_bias", "supertrend_direction", "donchian_breakout"],
        "weights": {
            "ema_crossover": 1.2, "sma_crossover": 1.0, "multi_tf_alignment": 1.5,
            "vwap_trend_bias": 1.0, "supertrend_direction": 1.0, "donchian_breakout": 1.1,
        },
    },
    "momentum_engine": {
        "strategies": ["rsi_ob_os", "macd_trend_momentum", "stochastic_scalping",
                       "obv_momentum", "chaikin_breakout"],
        "weights": {
            "rsi_ob_os": 1.2, "macd_trend_momentum": 1.3, "stochastic_scalping": 1.0,
            "obv_momentum": 1.1, "chaikin_breakout": 1.0,
        },
    },
    "smart_structure_engine": {
        "strategies": ["liquidity_sweep", "order_block_retest", "fair_value_gap",
                       "break_of_structure", "volume_imbalance_burst", "accumulation_distribution"],
        "weights": {
            "liquidity_sweep": 1.3, "order_block_retest": 1.2, "fair_value_gap": 1.1,
            "break_of_structure": 1.4, "volume_imbalance_burst": 1.2,
            "accumulation_distribution": 1.0,
        },
    },
    "volatility_breakout_engine": {
        "strategies": ["atr_volatility_breakout", "range_expansion", "breakout_retest",
                       "keltner_adaptive_breakout", "channel_width_spike"],
        "weights": {
            "atr_volatility_breakout": 1.2, "range_expansion": 1.0, "breakout_retest": 1.3,
            "keltner_adaptive_breakout": 1.1, "channel_width_spike": 1.0,
        },
    },
    "mean_reversion_engine": {
        "strategies": ["bollinger_mean_reversion", "zscore_mean_reversion",
                       "donchian_mean_reversion", "vwap_pullback_reversion"],
        "weights": {
            "bollinger_mean_reversion": 1.2, "zscore_mean_reversion": 1.3,
            "donchian_mean_reversion": 1.0, "vwap_pullback_reversion": 1.1,
        },
    },
    "session_engine": {
        "strategies": ["us_open_momentum", "london_open_momentum", "tokyo_open_range",
                       "ny_close_rebalance", "session_overlap_surge"],
        "weights": {
            "us_open_momentum": 1.2, "london_open_momentum": 1.1, "tokyo_open_range": 1.0,
            "ny_close_rebalance": 1.0, "session_overlap_surge": 1.3,
        },
    },
    "hybrid_engine": {
        "strategies": ["hybrid_confluence", "smart_money_enhanced_trend",
                       "momentum_structural_filter", "volatility_surprise_trend"],
        "weights": {
            "hybrid_confluence": 1.5, "smart_money_enhanced_trend": 1.4,
            "momentum_structural_filter": 1.2, "volatility_surprise_trend": 1.1,
        },
    },
}


def build_engine(name: str) -> StrategyEngineGroup:
    """Build a strategy engine from predefined definitions."""
    defn = ENGINE_DEFINITIONS.get(name)
    if not defn:
        raise ValueError(f"Unknown engine: {name}")
    return StrategyEngineGroup(name, defn["strategies"], defn["weights"])


def build_all_engines() -> Dict[str, StrategyEngineGroup]:
    """Build all predefined strategy engines."""
    engines = {}
    for name in ENGINE_DEFINITIONS:
        engines[name] = build_engine(name)
    logger.info("engines_built", count=len(engines), names=list(engines.keys()))
    return engines