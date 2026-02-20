"""
APEX SIGNAL™ — Integration Tests for Engines, ML, RL, Backtest, Telegram, and API
"""
import pytest
import pandas as pd
import numpy as np
from unittest.mock import AsyncMock, patch, MagicMock

from apex_signal.engines.strategy_engine import StrategyEngineGroup, build_all_engines, EngineResult
from apex_signal.engines.signal_engine import SignalEngine, MasterSignal
from apex_signal.smart_money.detector import SmartMoneyDetector, SmartMoneyResult
from apex_signal.ml.features.feature_engineering import build_features, build_labels, prepare_training_data
from apex_signal.ml.trainer import MLModelSuite
from apex_signal.rl.environment import TradingEnvironment, SimpleRLAgent
from apex_signal.backtest.backtester import Backtester, BacktestResult
from apex_signal.telegram.notifier import TelegramNotifier
from apex_signal.data.models import PriceTick, VerifiedPrice, DataSource
from apex_signal.data.cache.price_cache import PriceCache
from apex_signal.utils.helpers import classify_tier, clamp, normalize_symbol, is_crypto


# ─── Strategy Engine Tests ──────────────────────────────────────

class TestStrategyEngines:
    def test_build_all_engines(self):
        engines = build_all_engines()
        assert len(engines) >= 7
        assert "trend_engine" in engines
        assert "momentum_engine" in engines
        assert "smart_structure_engine" in engines

    def test_engine_evaluate(self, enriched_df):
        engines = build_all_engines()
        for name, engine in engines.items():
            result = engine.evaluate(enriched_df, "TEST/USD")
            assert isinstance(result, EngineResult)
            assert result.signal in ("BUY", "SELL", "HOLD")
            assert 0 <= result.confluence_score <= 100
            assert len(result.engine_name) > 0

    def test_engine_result_to_dict(self, enriched_df):
        engines = build_all_engines()
        result = engines["trend_engine"].evaluate(enriched_df, "TEST")
        d = result.to_dict()
        assert "engine_name" in d
        assert "signal" in d
        assert "confluence_score" in d
        assert "strategies" in d


# ─── Master Signal Engine Tests ─────────────────────────────────

class TestSignalEngine:
    def test_generate_signal(self, enriched_df):
        engine = SignalEngine()
        signal = engine.generate_signal(enriched_df, "BTC/USD")
        assert isinstance(signal, MasterSignal)
        assert signal.side in ("BUY", "SELL", "HOLD")
        assert 0 <= signal.confidence <= 100
        assert signal.tier in ("Elite", "Strong", "Moderate", "Weak")
        assert len(signal.reason) > 0

    def test_signal_to_dict(self, enriched_df):
        engine = SignalEngine()
        signal = engine.generate_signal(enriched_df, "ETH/USD")
        d = signal.to_dict()
        assert "symbol" in d
        assert "side" in d
        assert "confidence" in d
        assert "tier" in d
        assert "components" in d
        assert "engines" in d

    def test_signal_components(self, enriched_df):
        engine = SignalEngine()
        signal = engine.generate_signal(enriched_df, "SPY")
        assert "ml_probability" in signal.components
        assert "strategy_confluence" in signal.components
        assert "smart_money_score" in signal.components

    def test_empty_df_returns_hold(self, empty_df):
        engine = SignalEngine()
        signal = engine.generate_signal(empty_df, "TEST")
        assert signal.side == "HOLD"
        assert signal.confidence == 0


# ─── Smart Money Detector Tests ─────────────────────────────────

class TestSmartMoneyDetector:
    def test_detect(self, enriched_df):
        detector = SmartMoneyDetector()
        result = detector.detect(enriched_df)
        assert isinstance(result, SmartMoneyResult)
        assert 0 <= result.smart_money_score <= 1
        assert result.bias in ("bullish", "bearish", "neutral")

    def test_detect_empty(self, empty_df):
        detector = SmartMoneyDetector()
        result = detector.detect(empty_df)
        assert result.smart_money_score == 0.0

    def test_result_to_dict(self, enriched_df):
        detector = SmartMoneyDetector()
        result = detector.detect(enriched_df)
        d = result.to_dict()
        assert "smart_money_score" in d
        assert "events" in d
        assert "bias" in d


# ─── ML Feature Engineering Tests ───────────────────────────────

class TestFeatureEngineering:
    def test_build_features(self, enriched_df):
        features = build_features(enriched_df)
        assert not features.empty
        assert len(features.columns) > 20
        assert not features.isna().all().any()

    def test_build_labels(self, enriched_df):
        labels = build_labels(enriched_df)
        assert len(labels) == len(enriched_df)
        assert labels.isin([0, 1]).all()

    def test_prepare_training_data(self, enriched_df):
        features, labels = prepare_training_data(enriched_df)
        assert len(features) == len(labels)
        assert len(features) > 0


# ─── ML Model Suite Tests ───────────────────────────────────────

class TestMLModelSuite:
    def test_train_and_predict(self, enriched_df):
        suite = MLModelSuite()
        features, labels = prepare_training_data(enriched_df)
        if len(features) >= 50:
            metrics = suite.train(features, labels)
            assert len(metrics) > 0
            prob = suite.predict_probability(enriched_df)
            assert 0 <= prob <= 100

    def test_untrained_returns_neutral(self, enriched_df):
        suite = MLModelSuite()
        prob = suite.predict_probability(enriched_df)
        assert prob == 50.0

    def test_model_info(self):
        suite = MLModelSuite()
        info = suite.model_info
        assert "is_trained" in info
        assert "version" in info


# ─── RL Environment Tests ───────────────────────────────────────

class TestRLEnvironment:
    def test_env_creation(self, enriched_df):
        env = TradingEnvironment(enriched_df)
        assert env.observation_space.shape == (10,)
        assert env.action_space.n == 3

    def test_env_reset(self, enriched_df):
        env = TradingEnvironment(enriched_df)
        state, info = env.reset()
        assert state.shape == (10,)
        assert not np.isnan(state).any()

    def test_env_step(self, enriched_df):
        env = TradingEnvironment(enriched_df)
        env.reset()
        state, reward, done, truncated, info = env.step(1)  # BUY
        assert state.shape == (10,)
        assert isinstance(reward, float)
        assert "capital" in info

    def test_env_full_episode(self, enriched_df):
        env = TradingEnvironment(enriched_df)
        state, _ = env.reset()
        done = False
        steps = 0
        while not done and steps < 1000:
            action = env.action_space.sample()
            state, reward, done, truncated, info = env.step(action)
            steps += 1
        assert steps > 0


class TestRLAgent:
    def test_agent_train(self, enriched_df):
        env = TradingEnvironment(enriched_df)
        agent = SimpleRLAgent()
        metrics = agent.train(env, episodes=10)
        assert "avg_reward" in metrics
        assert agent.is_trained

    def test_agent_scaling_factor(self, enriched_df):
        agent = SimpleRLAgent()
        env = TradingEnvironment(enriched_df)
        agent.train(env, episodes=10)
        factor = agent.get_scaling_factor(enriched_df)
        assert 0 <= factor <= 100

    def test_untrained_agent_neutral(self, enriched_df):
        agent = SimpleRLAgent()
        factor = agent.get_scaling_factor(enriched_df)
        assert factor == 50.0


# ─── Backtester Tests ───────────────────────────────────────────

class TestBacktester:
    def test_backtest_run(self, sample_ohlcv_df):
        backtester = Backtester()
        engine = SignalEngine()
        result = backtester.run(sample_ohlcv_df, "TEST/USD", engine)
        assert isinstance(result, BacktestResult)
        assert result.initial_capital > 0
        assert result.final_capital > 0

    def test_backtest_metrics(self, sample_ohlcv_df):
        backtester = Backtester()
        engine = SignalEngine()
        result = backtester.run(sample_ohlcv_df, "TEST", engine)
        d = result.to_dict()
        assert "total_return_pct" in d
        assert "win_rate" in d
        assert "sharpe_ratio" in d
        assert "max_drawdown_pct" in d

    def test_backtest_empty_df(self, empty_df):
        backtester = Backtester()
        engine = SignalEngine()
        result = backtester.run(empty_df, "TEST", engine)
        assert result.total_trades == 0


# ─── Telegram Notifier Tests ────────────────────────────────────

class TestTelegramNotifier:
    def test_format_signal_message(self):
        notifier = TelegramNotifier()
        signal_data = {
            "symbol": "BTC/USD",
            "side": "BUY",
            "confidence": 85,
            "tier": "Elite",
            "reason": "EMA crossover with smart money confirmation",
            "components": {
                "ml_probability": 80,
                "strategy_confluence": 75,
                "smart_money_score": 90,
                "volatility_regime": 60,
                "rl_scaling_factor": 55,
            },
            "engines": [
                {"engine_name": "trend_engine", "signal": "BUY", "confluence_score": 75},
            ],
            "timestamp": "2024-01-01T12:00:00+00:00",
        }
        message = notifier.format_signal_message(signal_data)
        assert "APEX SIGNAL" in message
        assert "BTC/USD" in message
        assert "BUY" in message
        assert "Elite" in message

    def test_quiet_hours_check(self):
        notifier = TelegramNotifier()
        result = notifier._is_quiet_hours()
        assert isinstance(result, bool)

    def test_mute_unmute(self):
        notifier = TelegramNotifier()
        notifier.mute_symbol("BTC/USD")
        assert "BTC/USD" in notifier._muted_chats
        notifier.unmute_symbol("BTC/USD")
        assert "BTC/USD" not in notifier._muted_chats

    def test_stats(self):
        notifier = TelegramNotifier()
        stats = notifier.stats
        assert "initialized" in stats
        assert "messages_sent" in stats


# ─── Data Verification Tests ────────────────────────────────────

class TestCrossSourceDeviation:
    def test_verified_price_within_threshold(self):
        """Simulate cross-source verification with <0.5% deviation."""
        prices = {"alpaca": 100.0, "polygon": 100.3, "coingecko": 100.1}
        avg = sum(prices.values()) / len(prices)
        max_dev = max(abs(p - avg) / avg for p in prices.values())
        assert max_dev < 0.005  # <0.5%

    def test_deviation_exceeds_threshold(self):
        """Simulate cross-source verification with >0.5% deviation."""
        prices = {"alpaca": 100.0, "polygon": 101.0, "coingecko": 100.0}
        avg = sum(prices.values()) / len(prices)
        max_dev = max(abs(p - avg) / avg for p in prices.values())
        assert max_dev > 0.005  # >0.5%


# ─── Price Cache Tests ──────────────────────────────────────────

class TestPriceCache:
    def test_put_get_tick(self):
        from datetime import datetime, timezone
        cache = PriceCache()
        tick = PriceTick(
            symbol="BTC/USD", source=DataSource.ALPACA,
            price=50000.0, timestamp=datetime.now(timezone.utc)
        )
        cache.put_tick(tick)
        result = cache.get_tick("BTC/USD", "alpaca")
        assert result is not None
        assert result.price == 50000.0

    def test_cache_stats(self):
        cache = PriceCache()
        stats = cache.stats
        assert "price_entries" in stats
        assert "candle_entries" in stats


# ─── Utility Tests ──────────────────────────────────────────────

class TestHelpers:
    def test_classify_tier(self):
        assert classify_tier(90) == "Elite"
        assert classify_tier(70) == "Strong"
        assert classify_tier(50) == "Moderate"
        assert classify_tier(20) == "Weak"

    def test_clamp(self):
        assert clamp(150, 0, 100) == 100
        assert clamp(-10, 0, 100) == 0
        assert clamp(50, 0, 100) == 50

    def test_normalize_symbol(self):
        assert normalize_symbol("BTC/USD") == "BTCUSD"
        assert normalize_symbol("SPY") == "SPY"

    def test_is_crypto(self):
        assert is_crypto("BTC/USD") is True
        assert is_crypto("ETH/USD") is True
        assert is_crypto("SPY") is False
        assert is_crypto("AAPL") is False


# ─── API Health Endpoint Tests ──────────────────────────────────

class TestAPIEndpoints:
    @pytest.mark.asyncio
    async def test_healthz(self):
        from fastapi.testclient import TestClient
        from apex_signal.api.app import app
        client = TestClient(app)
        response = client.get("/healthz")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"

    @pytest.mark.asyncio
    async def test_metrics(self):
        from fastapi.testclient import TestClient
        from apex_signal.api.app import app
        client = TestClient(app)
        response = client.get("/metrics")
        assert response.status_code == 200
        data = response.json()
        assert "app" in data
        assert "signals" in data

    @pytest.mark.asyncio
    async def test_list_strategies(self):
        from fastapi.testclient import TestClient
        from apex_signal.api.app import app
        client = TestClient(app)
        response = client.get("/api/v1/strategies")
        assert response.status_code == 200
        data = response.json()
        assert data["total"] >= 35

    @pytest.mark.asyncio
    async def test_list_indicators(self):
        from fastapi.testclient import TestClient
        from apex_signal.api.app import app
        client = TestClient(app)
        response = client.get("/api/v1/indicators")
        assert response.status_code == 200
        data = response.json()
        assert data["total"] >= 20