"""
APEX SIGNAL™ — Tests for Enhancement Modules
Risk Manager, Adaptive SL/TP, Signal Quality, Divergence, Enhanced ML, Enhanced Backtester
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timezone

from apex_signal.engines.risk_manager import RiskManager
from apex_signal.engines.adaptive_sltp import AdaptiveSLTP, SLTPLevels
from apex_signal.engines.signal_quality import (
    SignalQualityScorer, RegimeAdaptiveSelector, StrategyPerformanceTracker,
)
from apex_signal.indicators.divergence import DivergenceDetector
from apex_signal.strategies.base import StrategySignal


# ─── Risk Manager Tests ─────────────────────────────────────────

class TestRiskManager:
    def test_initial_state(self):
        rm = RiskManager(initial_equity=100000)
        assert rm.state.equity == 100000
        assert rm.state.is_killed is False
        assert rm.state.consecutive_losses == 0

    def test_can_trade_initially(self):
        rm = RiskManager()
        result = rm.check_can_trade("BTC/USD", 80, "Elite")
        assert result["allowed"] is True

    def test_drawdown_kill_switch(self):
        rm = RiskManager(initial_equity=100000, max_drawdown_pct=10.0)
        # Simulate large loss
        rm.state.equity = 89000
        rm.state.peak_equity = 100000
        result = rm.check_can_trade("BTC/USD", 80, "Elite")
        assert result["allowed"] is False
        assert rm.state.is_killed is True

    def test_daily_loss_limit(self):
        rm = RiskManager(initial_equity=100000, daily_loss_limit_pct=3.0)
        rm.state.daily_pnl = -3100
        rm.state.last_reset_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        result = rm.check_can_trade("BTC/USD", 80, "Elite")
        assert result["allowed"] is False

    def test_max_daily_trades(self):
        rm = RiskManager(max_daily_trades=5)
        rm.state.daily_trades = 5
        rm.state.last_reset_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        result = rm.check_can_trade("BTC/USD", 80, "Elite")
        assert result["allowed"] is False

    def test_consecutive_loss_breaker(self):
        rm = RiskManager(max_consecutive_losses=3)
        rm.state.consecutive_losses = 3
        # Non-Elite should be blocked
        result = rm.check_can_trade("BTC/USD", 70, "Strong")
        assert result["allowed"] is False
        # Elite should pass
        result = rm.check_can_trade("BTC/USD", 85, "Elite")
        assert result["allowed"] is True

    def test_record_trade(self):
        rm = RiskManager(initial_equity=100000)
        rm.record_trade(500)
        assert rm.state.equity == 100500
        assert rm.state.winning_trades == 1
        assert rm.state.consecutive_losses == 0

        rm.record_trade(-200)
        assert rm.state.equity == 100300
        assert rm.state.consecutive_losses == 1

    def test_position_sizing(self):
        rm = RiskManager()
        size = rm.calculate_position_size(80, "Elite")
        assert rm.min_position_pct <= size <= rm.max_position_pct

    def test_drawdown_adjusted_confidence(self):
        rm = RiskManager(initial_equity=100000, max_drawdown_pct=20.0)
        rm.state.equity = 90000
        rm.state.peak_equity = 100000
        rm._update_drawdown()
        adjusted = rm.adjust_confidence_for_drawdown(80)
        assert adjusted < 80  # Should be reduced

    def test_reset_kill_switch(self):
        rm = RiskManager()
        rm.state.is_killed = True
        rm.state.kill_reason = "test"
        rm.reset_kill_switch()
        assert rm.state.is_killed is False

    def test_risk_report(self):
        rm = RiskManager()
        report = rm.risk_report
        assert "equity" in report
        assert "current_drawdown_pct" in report
        assert "win_rate" in report
        assert "is_killed" in report

    def test_max_open_positions(self):
        rm = RiskManager(max_open_positions=2)
        rm.state.open_positions = 2
        result = rm.check_can_trade("BTC/USD", 80, "Elite")
        assert result["allowed"] is False


# ─── Adaptive SL/TP Tests ───────────────────────────────────────

class TestAdaptiveSLTP:
    def test_compute_buy(self, enriched_df):
        sltp = AdaptiveSLTP()
        levels = sltp.compute(enriched_df, "BUY", 80, "Elite")
        assert isinstance(levels, SLTPLevels)
        assert levels.stop_loss < levels.entry_price
        assert levels.take_profit_1 > levels.entry_price
        assert levels.take_profit_2 > levels.take_profit_1
        assert levels.risk_reward_ratio >= 2.0

    def test_compute_sell(self, enriched_df):
        sltp = AdaptiveSLTP()
        levels = sltp.compute(enriched_df, "SELL", 70, "Strong")
        assert levels.stop_loss > levels.entry_price
        assert levels.take_profit_1 < levels.entry_price
        assert levels.take_profit_2 < levels.take_profit_1

    def test_min_rr_enforced(self, enriched_df):
        sltp = AdaptiveSLTP(min_rr_ratio=2.0)
        levels = sltp.compute(enriched_df, "BUY", 60, "Moderate")
        assert levels.risk_reward_ratio >= 2.0

    def test_to_dict(self, enriched_df):
        sltp = AdaptiveSLTP()
        levels = sltp.compute(enriched_df, "BUY", 80, "Elite")
        d = levels.to_dict()
        assert "entry_price" in d
        assert "stop_loss" in d
        assert "take_profit_1" in d
        assert "risk_reward_ratio" in d
        assert "method" in d

    def test_tier_affects_levels(self, enriched_df):
        sltp = AdaptiveSLTP()
        elite = sltp.compute(enriched_df, "BUY", 90, "Elite")
        weak = sltp.compute(enriched_df, "BUY", 30, "Weak")
        # Elite should have tighter SL
        assert elite.sl_pct <= weak.sl_pct

    def test_empty_df_default(self):
        sltp = AdaptiveSLTP()
        levels = sltp.compute(pd.DataFrame(), "BUY", 50, "Moderate", entry_price=100.0)
        assert levels.entry_price == 100.0
        assert levels.stop_loss < 100.0


# ─── Signal Quality Scorer Tests ────────────────────────────────

class TestSignalQualityScorer:
    def test_score_buy_signal(self, enriched_df):
        scorer = SignalQualityScorer()
        signal = StrategySignal("test", "BUY", 70, "Test reason")
        result = scorer.score_signal(enriched_df, signal)
        assert "quality_score" in result
        assert "passed" in result
        assert "factors" in result
        assert 0 <= result["quality_score"] <= 1

    def test_score_sell_signal(self, enriched_df):
        scorer = SignalQualityScorer()
        signal = StrategySignal("test", "SELL", 70, "Test reason")
        result = scorer.score_signal(enriched_df, signal)
        assert 0 <= result["quality_score"] <= 1

    def test_hold_signal_fails(self, enriched_df):
        scorer = SignalQualityScorer()
        signal = StrategySignal("test", "HOLD", 0, "No signal")
        result = scorer.score_signal(enriched_df, signal)
        assert result["passed"] is False

    def test_empty_df(self):
        scorer = SignalQualityScorer()
        signal = StrategySignal("test", "BUY", 70, "Test")
        result = scorer.score_signal(pd.DataFrame(), signal)
        assert result["passed"] is False

    def test_quality_factors_present(self, enriched_df):
        scorer = SignalQualityScorer()
        signal = StrategySignal("test", "BUY", 70, "Test")
        result = scorer.score_signal(enriched_df, signal)
        factors = result["factors"]
        assert "volume" in factors
        assert "trend_alignment" in factors
        assert "momentum" in factors
        assert "confirmation_candle" in factors


# ─── Regime Adaptive Selector Tests ─────────────────────────────

class TestRegimeAdaptiveSelector:
    def test_low_vol_blocks_breakout(self):
        selector = RegimeAdaptiveSelector()
        blocked = selector.get_blocked_families(0)
        assert "volatility_breakout" in blocked

    def test_high_vol_blocks_mean_reversion(self):
        selector = RegimeAdaptiveSelector()
        blocked = selector.get_blocked_families(2)
        assert "mean_reversion" in blocked

    def test_normal_vol_allows_all(self):
        selector = RegimeAdaptiveSelector()
        blocked = selector.get_blocked_families(1)
        assert len(blocked) == 0

    def test_confidence_boost(self):
        selector = RegimeAdaptiveSelector()
        boost = selector.get_confidence_boost(2, "trend_following")
        assert boost > 0

    def test_filter_signals(self):
        selector = RegimeAdaptiveSelector()
        signals = [
            StrategySignal("bollinger_mean_reversion", "BUY", 70, "test"),
            StrategySignal("ema_crossover", "BUY", 60, "test"),
        ]
        families = {"mean_reversion": ["bollinger_mean_reversion"], "trend_following": ["ema_crossover"]}
        filtered = selector.filter_signals(signals, 2, families)  # High vol
        names = [s.strategy_name for s in filtered]
        assert "bollinger_mean_reversion" not in names
        assert "ema_crossover" in names


# ─── Strategy Performance Tracker Tests ──────────────────────────

class TestStrategyPerformanceTracker:
    def test_record_and_stats(self):
        tracker = StrategyPerformanceTracker(min_trades=3, min_win_rate=0.4)
        tracker.record_outcome("test_strat", 100)
        tracker.record_outcome("test_strat", -50)
        tracker.record_outcome("test_strat", 80)
        stats = tracker.get_strategy_stats("test_strat")
        assert stats["trades"] == 3
        assert stats["win_rate"] > 0

    def test_auto_disable(self):
        tracker = StrategyPerformanceTracker(min_trades=5, min_win_rate=0.5)
        # Record 5 losses
        for _ in range(5):
            tracker.record_outcome("bad_strat", -100)
        assert not tracker.is_enabled("bad_strat")

    def test_enabled_by_default(self):
        tracker = StrategyPerformanceTracker()
        assert tracker.is_enabled("unknown_strat") is True


# ─── Divergence Detector Tests ──────────────────────────────────

class TestDivergenceDetector:
    def test_divergence_columns(self, sample_ohlcv_df):
        # First compute RSI and MACD
        from apex_signal.indicators.momentum import RSIIndicator
        from apex_signal.indicators.oscillators import MACDIndicator
        df = RSIIndicator().calculate(sample_ohlcv_df)
        df = MACDIndicator().calculate(df)

        detector = DivergenceDetector()
        result = detector.calculate(df)
        assert "div_rsi_bull" in result.columns
        assert "div_rsi_bear" in result.columns
        assert "div_macd_bull" in result.columns
        assert "div_macd_bear" in result.columns
        assert "div_score" in result.columns

    def test_divergence_values_valid(self, sample_ohlcv_df):
        from apex_signal.indicators.momentum import RSIIndicator
        from apex_signal.indicators.oscillators import MACDIndicator
        df = RSIIndicator().calculate(sample_ohlcv_df)
        df = MACDIndicator().calculate(df)

        detector = DivergenceDetector()
        result = detector.calculate(df)
        assert result["div_rsi_bull"].isin([0, 1]).all()
        assert result["div_rsi_bear"].isin([0, 1]).all()

    def test_reset(self, sample_ohlcv_df):
        detector = DivergenceDetector()
        from apex_signal.indicators.momentum import RSIIndicator
        df = RSIIndicator().calculate(sample_ohlcv_df)
        detector.calculate(df)
        detector.reset()
        assert detector.last_result is None


# ─── Enhanced ML Tests ──────────────────────────────────────────

class TestEnhancedML:
    def test_train_and_predict(self, enriched_df):
        from apex_signal.ml.enhanced_trainer import EnhancedMLPipeline
        from apex_signal.ml.features.feature_engineering import prepare_training_data

        pipeline = EnhancedMLPipeline()
        features, labels = prepare_training_data(enriched_df)
        if len(features) >= 80:
            result = pipeline.train(features, labels)
            assert "models_trained" in result
            assert len(result["models_trained"]) > 0

            pred = pipeline.predict_with_disagreement(enriched_df)
            assert "probability" in pred
            assert "disagreement" in pred
            assert "adjusted" in pred
            assert 0 <= pred["adjusted"] <= 100

    def test_untrained_returns_neutral(self, enriched_df):
        from apex_signal.ml.enhanced_trainer import EnhancedMLPipeline
        pipeline = EnhancedMLPipeline()
        pred = pipeline.predict_with_disagreement(enriched_df)
        assert pred["probability"] == 50.0

    def test_model_info(self):
        from apex_signal.ml.enhanced_trainer import EnhancedMLPipeline
        pipeline = EnhancedMLPipeline()
        info = pipeline.model_info
        assert "is_trained" in info
        assert "features_selected" in info


# ─── Enhanced Backtester Tests ──────────────────────────────────

class TestEnhancedBacktester:
    def test_monte_carlo(self, sample_ohlcv_df):
        from apex_signal.backtest.enhanced_backtester import EnhancedBacktester
        from apex_signal.engines.signal_engine import SignalEngine

        bt = EnhancedBacktester()
        engine = SignalEngine()
        result = bt.run_enhanced(sample_ohlcv_df, "TEST", engine, n_monte_carlo=50)

        assert result.base_result is not None
        if result.monte_carlo:
            mc = result.monte_carlo
            assert mc.n_simulations == 50
            assert mc.probability_of_profit >= 0
            assert mc.p5_return_pct <= mc.p95_return_pct

    def test_advanced_metrics(self, sample_ohlcv_df):
        from apex_signal.backtest.enhanced_backtester import EnhancedBacktester
        from apex_signal.engines.signal_engine import SignalEngine

        bt = EnhancedBacktester()
        engine = SignalEngine()
        result = bt.run_enhanced(sample_ohlcv_df, "TEST", engine)

        if result.advanced_metrics:
            assert "mar_ratio" in result.advanced_metrics
            assert "omega_ratio" in result.advanced_metrics
            assert "tail_ratio" in result.advanced_metrics

    def test_to_dict(self, sample_ohlcv_df):
        from apex_signal.backtest.enhanced_backtester import EnhancedBacktester
        from apex_signal.engines.signal_engine import SignalEngine

        bt = EnhancedBacktester()
        engine = SignalEngine()
        result = bt.run_enhanced(sample_ohlcv_df, "TEST", engine, n_monte_carlo=20)
        d = result.to_dict()
        assert "total_return_pct" in d
        assert "total_trades" in d


# ─── Enhanced Signal Engine Integration ─────────────────────────

class TestEnhancedSignalEngine:
    def test_signal_includes_quality(self, enriched_df):
        from apex_signal.engines.signal_engine import SignalEngine
        engine = SignalEngine()
        signal = engine.generate_signal(enriched_df, "BTC/USD")
        assert signal.quality is not None
        assert "quality_score" in signal.quality

    def test_signal_includes_sltp(self, enriched_df):
        from apex_signal.engines.signal_engine import SignalEngine
        engine = SignalEngine()
        signal = engine.generate_signal(enriched_df, "BTC/USD")
        if signal.side in ("BUY", "SELL"):
            assert signal.sltp is not None
            assert "stop_loss" in signal.sltp
            assert "take_profit_1" in signal.sltp

    def test_signal_includes_risk_check(self, enriched_df):
        from apex_signal.engines.signal_engine import SignalEngine
        engine = SignalEngine()
        signal = engine.generate_signal(enriched_df, "BTC/USD")
        assert signal.risk_check is not None

    def test_signal_components_include_quality(self, enriched_df):
        from apex_signal.engines.signal_engine import SignalEngine
        engine = SignalEngine()
        signal = engine.generate_signal(enriched_df, "BTC/USD")
        assert "quality_score" in signal.components

    def test_indicator_count_increased(self):
        from apex_signal.indicators.registry import IndicatorRegistry
        registry = IndicatorRegistry()
        assert registry.count >= 23  # 22 original + divergence