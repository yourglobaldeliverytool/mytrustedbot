"""
APEX SIGNAL™ — Strategy Registry
Central registry managing all 35+ trading strategies.
"""
from typing import Dict, List, Optional
import pandas as pd

from apex_signal.strategies.base import BaseStrategy, StrategySignal
from apex_signal.strategies.trend_following import (
    EMACrossoverStrategy, SMACrossoverStrategy, MultiTimeframeAlignmentStrategy,
    VWAPTrendBiasStrategy, SupertrendDirectionStrategy, DonchianBreakoutStrategy,
)
from apex_signal.strategies.momentum import (
    RSIOverboughtOversoldStrategy, MACDTrendMomentumStrategy,
    StochasticScalpingStrategy, OBVMomentumStrategy, ChaikinOscillatorBreakoutStrategy,
)
from apex_signal.strategies.volatility_breakout import (
    ATRVolatilityBreakoutStrategy, RangeExpansionStrategy, BreakoutRetestStrategy,
    KeltnerAdaptiveBreakoutStrategy, ChannelWidthSpikeStrategy,
)
from apex_signal.strategies.mean_reversion import (
    BollingerMeanReversionStrategy, ZScoreMeanReversionStrategy,
    DonchianMeanReversionStrategy, VWAPPullbackReversionStrategy,
)
from apex_signal.strategies.smart_money import (
    LiquiditySweepStrategy, OrderBlockRetestStrategy, FairValueGapStrategy,
    BreakOfStructureStrategy, VolumeImbalanceBurstStrategy,
    AccumulationDistributionStrategy,
)
from apex_signal.strategies.session_aware import (
    USOpenMomentumStrategy, LondonOpenMomentumStrategy, TokyoOpenRangeStrategy,
    NYCloseRebalanceStrategy, SessionOverlapSurgeStrategy,
)
from apex_signal.strategies.hybrid import (
    HybridConfluenceStrategy, SmartMoneyEnhancedTrendStrategy,
    MomentumStructuralFilterStrategy, VolatilitySurpriseTrendFilterStrategy,
)
from apex_signal.utils.logger import get_logger

logger = get_logger("strategy_registry")


class StrategyRegistry:
    """Central registry for all trading strategies."""

    def __init__(self):
        self._strategies: Dict[str, BaseStrategy] = {}
        self._register_all()

    def _register_all(self) -> None:
        """Register all 35 strategies."""
        all_strategies = [
            # Trend Following (6)
            EMACrossoverStrategy(),
            SMACrossoverStrategy(),
            MultiTimeframeAlignmentStrategy(),
            VWAPTrendBiasStrategy(),
            SupertrendDirectionStrategy(),
            DonchianBreakoutStrategy(),
            # Momentum (5)
            RSIOverboughtOversoldStrategy(),
            MACDTrendMomentumStrategy(),
            StochasticScalpingStrategy(),
            OBVMomentumStrategy(),
            ChaikinOscillatorBreakoutStrategy(),
            # Volatility / Breakout (5)
            ATRVolatilityBreakoutStrategy(),
            RangeExpansionStrategy(),
            BreakoutRetestStrategy(),
            KeltnerAdaptiveBreakoutStrategy(),
            ChannelWidthSpikeStrategy(),
            # Mean Reversion (4)
            BollingerMeanReversionStrategy(),
            ZScoreMeanReversionStrategy(),
            DonchianMeanReversionStrategy(),
            VWAPPullbackReversionStrategy(),
            # Smart-Money / Structural (6)
            LiquiditySweepStrategy(),
            OrderBlockRetestStrategy(),
            FairValueGapStrategy(),
            BreakOfStructureStrategy(),
            VolumeImbalanceBurstStrategy(),
            AccumulationDistributionStrategy(),
            # Session-Aware (5)
            USOpenMomentumStrategy(),
            LondonOpenMomentumStrategy(),
            TokyoOpenRangeStrategy(),
            NYCloseRebalanceStrategy(),
            SessionOverlapSurgeStrategy(),
            # Hybrid & Confluence (4)
            HybridConfluenceStrategy(),
            SmartMoneyEnhancedTrendStrategy(),
            MomentumStructuralFilterStrategy(),
            VolatilitySurpriseTrendFilterStrategy(),
        ]

        for strategy in all_strategies:
            self._strategies[strategy.name] = strategy

        logger.info("strategies_registered", count=len(self._strategies),
                     names=list(self._strategies.keys()))

    def register(self, strategy: BaseStrategy) -> None:
        """Register a custom strategy."""
        self._strategies[strategy.name] = strategy

    def get(self, name: str) -> Optional[BaseStrategy]:
        """Get a strategy by name."""
        return self._strategies.get(name)

    @property
    def strategy_names(self) -> List[str]:
        return list(self._strategies.keys())

    @property
    def count(self) -> int:
        return len(self._strategies)

    def evaluate_all(self, df: pd.DataFrame, symbol: str = "") -> List[StrategySignal]:
        """Evaluate all strategies on the given indicator-enriched DataFrame."""
        signals = []
        for name, strategy in self._strategies.items():
            try:
                signal = strategy.evaluate(df, symbol)
                signals.append(signal)
            except Exception as e:
                logger.error("strategy_eval_error", strategy=name, error=str(e))
                signals.append(StrategySignal(
                    strategy_name=name, signal="HOLD", confidence=0,
                    reason=f"Error: {str(e)}"
                ))
        return signals

    def evaluate_group(self, df: pd.DataFrame, names: List[str], symbol: str = "") -> List[StrategySignal]:
        """Evaluate a specific group of strategies."""
        signals = []
        for name in names:
            strategy = self._strategies.get(name)
            if strategy:
                try:
                    signal = strategy.evaluate(df, symbol)
                    signals.append(signal)
                except Exception as e:
                    logger.error("strategy_eval_error", strategy=name, error=str(e))
        return signals

    def get_strategies_by_family(self) -> Dict[str, List[str]]:
        """Return strategies grouped by family."""
        return {
            "trend_following": [
                "ema_crossover", "sma_crossover", "multi_tf_alignment",
                "vwap_trend_bias", "supertrend_direction", "donchian_breakout",
            ],
            "momentum": [
                "rsi_ob_os", "macd_trend_momentum", "stochastic_scalping",
                "obv_momentum", "chaikin_breakout",
            ],
            "volatility_breakout": [
                "atr_volatility_breakout", "range_expansion", "breakout_retest",
                "keltner_adaptive_breakout", "channel_width_spike",
            ],
            "mean_reversion": [
                "bollinger_mean_reversion", "zscore_mean_reversion",
                "donchian_mean_reversion", "vwap_pullback_reversion",
            ],
            "smart_money": [
                "liquidity_sweep", "order_block_retest", "fair_value_gap",
                "break_of_structure", "volume_imbalance_burst",
                "accumulation_distribution",
            ],
            "session_aware": [
                "us_open_momentum", "london_open_momentum", "tokyo_open_range",
                "ny_close_rebalance", "session_overlap_surge",
            ],
            "hybrid": [
                "hybrid_confluence", "smart_money_enhanced_trend",
                "momentum_structural_filter", "volatility_surprise_trend",
            ],
        }


# Singleton
_registry: Optional[StrategyRegistry] = None


def get_strategy_registry() -> StrategyRegistry:
    global _registry
    if _registry is None:
        _registry = StrategyRegistry()
    return _registry