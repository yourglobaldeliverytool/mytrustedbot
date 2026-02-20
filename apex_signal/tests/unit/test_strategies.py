"""
APEX SIGNAL™ — Unit Tests for Strategies
Tests all 35 strategies for correct output format and signal logic.
"""
import pytest
import pandas as pd
import numpy as np

from apex_signal.strategies.base import StrategySignal
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
from apex_signal.strategies.registry import StrategyRegistry


ALL_STRATEGIES = [
    EMACrossoverStrategy, SMACrossoverStrategy, MultiTimeframeAlignmentStrategy,
    VWAPTrendBiasStrategy, SupertrendDirectionStrategy, DonchianBreakoutStrategy,
    RSIOverboughtOversoldStrategy, MACDTrendMomentumStrategy,
    StochasticScalpingStrategy, OBVMomentumStrategy, ChaikinOscillatorBreakoutStrategy,
    ATRVolatilityBreakoutStrategy, RangeExpansionStrategy, BreakoutRetestStrategy,
    KeltnerAdaptiveBreakoutStrategy, ChannelWidthSpikeStrategy,
    BollingerMeanReversionStrategy, ZScoreMeanReversionStrategy,
    DonchianMeanReversionStrategy, VWAPPullbackReversionStrategy,
    LiquiditySweepStrategy, OrderBlockRetestStrategy, FairValueGapStrategy,
    BreakOfStructureStrategy, VolumeImbalanceBurstStrategy,
    AccumulationDistributionStrategy,
    USOpenMomentumStrategy, LondonOpenMomentumStrategy, TokyoOpenRangeStrategy,
    NYCloseRebalanceStrategy, SessionOverlapSurgeStrategy,
    HybridConfluenceStrategy, SmartMoneyEnhancedTrendStrategy,
    MomentumStructuralFilterStrategy, VolatilitySurpriseTrendFilterStrategy,
]


class TestStrategySignalFormat:
    """Test that all strategies return properly formatted signals."""

    @pytest.mark.parametrize("StrategyClass", ALL_STRATEGIES)
    def test_returns_strategy_signal(self, StrategyClass, enriched_df):
        strategy = StrategyClass()
        signal = strategy.evaluate(enriched_df, "TEST/USD")
        assert isinstance(signal, StrategySignal)

    @pytest.mark.parametrize("StrategyClass", ALL_STRATEGIES)
    def test_signal_has_required_fields(self, StrategyClass, enriched_df):
        strategy = StrategyClass()
        signal = strategy.evaluate(enriched_df, "TEST/USD")
        d = signal.to_dict()
        assert "strategy_name" in d
        assert "signal" in d
        assert "confidence" in d
        assert "reason" in d
        assert "timestamp" in d

    @pytest.mark.parametrize("StrategyClass", ALL_STRATEGIES)
    def test_signal_valid_values(self, StrategyClass, enriched_df):
        strategy = StrategyClass()
        signal = strategy.evaluate(enriched_df, "TEST/USD")
        assert signal.signal in ("BUY", "SELL", "HOLD")
        assert 0 <= signal.confidence <= 100
        assert len(signal.reason) > 0
        assert len(signal.strategy_name) > 0

    @pytest.mark.parametrize("StrategyClass", ALL_STRATEGIES)
    def test_handles_empty_df(self, StrategyClass, empty_df):
        strategy = StrategyClass()
        signal = strategy.evaluate(empty_df, "TEST/USD")
        assert isinstance(signal, StrategySignal)
        assert signal.signal == "HOLD"

    @pytest.mark.parametrize("StrategyClass", ALL_STRATEGIES)
    def test_handles_small_df(self, StrategyClass, small_ohlcv_df):
        strategy = StrategyClass()
        signal = strategy.evaluate(small_ohlcv_df, "TEST/USD")
        assert isinstance(signal, StrategySignal)


class TestTrendStrategies:
    def test_ema_crossover_uptrend(self, enriched_df):
        strategy = EMACrossoverStrategy()
        signal = strategy.evaluate(enriched_df, "TEST")
        assert signal.signal in ("BUY", "SELL", "HOLD")

    def test_multi_tf_alignment(self, enriched_df):
        strategy = MultiTimeframeAlignmentStrategy()
        signal = strategy.evaluate(enriched_df, "TEST")
        assert signal.confidence >= 0


class TestMomentumStrategies:
    def test_rsi_strategy(self, enriched_df):
        strategy = RSIOverboughtOversoldStrategy()
        signal = strategy.evaluate(enriched_df, "TEST")
        assert "rsi" in signal.reason.lower() or "hold" in signal.reason.lower() or "insufficient" in signal.reason.lower()

    def test_macd_strategy(self, enriched_df):
        strategy = MACDTrendMomentumStrategy()
        signal = strategy.evaluate(enriched_df, "TEST")
        assert isinstance(signal, StrategySignal)


class TestSmartMoneyStrategies:
    def test_liquidity_sweep(self, enriched_df):
        strategy = LiquiditySweepStrategy()
        signal = strategy.evaluate(enriched_df, "TEST")
        assert signal.signal in ("BUY", "SELL", "HOLD")

    def test_order_block(self, enriched_df):
        strategy = OrderBlockRetestStrategy()
        signal = strategy.evaluate(enriched_df, "TEST")
        assert isinstance(signal, StrategySignal)


class TestStrategyRegistry:
    def test_registry_count(self):
        registry = StrategyRegistry()
        assert registry.count >= 35

    def test_evaluate_all(self, enriched_df):
        registry = StrategyRegistry()
        signals = registry.evaluate_all(enriched_df, "TEST/USD")
        assert len(signals) >= 35
        for sig in signals:
            assert isinstance(sig, StrategySignal)
            assert sig.signal in ("BUY", "SELL", "HOLD")

    def test_evaluate_group(self, enriched_df):
        registry = StrategyRegistry()
        signals = registry.evaluate_group(
            enriched_df, ["ema_crossover", "rsi_ob_os"], "TEST"
        )
        assert len(signals) == 2

    def test_get_families(self):
        registry = StrategyRegistry()
        families = registry.get_strategies_by_family()
        assert "trend_following" in families
        assert "momentum" in families
        assert "smart_money" in families
        assert len(families) >= 7

    def test_all_strategies_unique_names(self):
        registry = StrategyRegistry()
        names = registry.strategy_names
        assert len(names) == len(set(names))