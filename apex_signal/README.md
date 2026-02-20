# 🚀 APEX SIGNAL™ v2.0 — Production-Grade Quantitative Trading Signal Platform

[![Tests](https://img.shields.io/badge/tests-307%20passed-brightgreen)]()
[![Python](https://img.shields.io/badge/python-3.11-blue)]()
[![License](https://img.shields.io/badge/license-MIT-green)]()

## 🌟 Overview

**APEX SIGNAL™** is a fully-functional, production-grade quantitative trading signal platform that ingests verified real-time market data, processes 23+ technical indicators through 35+ trading strategies organized into 7 strategy engines, computes AI confidence scoring with ML and RL layers, detects smart-money institutional patterns, enforces comprehensive risk management, and delivers branded Telegram notifications.

### v2.0 Profit-Certainty Enhancements

- 🛡️ **Risk Manager** — Kill switch, daily loss limits, consecutive loss breaker, Kelly position sizing
- 🎯 **Adaptive SL/TP** — ATR-based dynamic stops with structure-aware levels, minimum 2:1 R:R enforced
- 📊 **Signal Quality Scorer** — 7-factor quality gate rejects bad setups before output
- 🔄 **Regime-Adaptive Selection** — Only runs strategies suited to current volatility regime
- 📈 **Win-Rate Tracker** — Auto-disables underperforming strategies in real-time
- 🧠 **Enhanced ML** — Feature importance pruning, Platt calibration, disagreement penalty
- 📉 **Divergence Detector** — RSI/MACD regular + hidden divergence detection
- 🎲 **Monte Carlo Backtesting** — Robustness testing with confidence intervals
- 💰 **Drawdown-Adjusted Confidence** — Automatically reduces risk during drawdowns

---

## 📁 Project Structure

```
apex_signal/
├── config/                  # Configuration & settings
│   └── settings.py          # Pydantic settings with env vars
├── data/                    # Data ingestion & verification
│   ├── adapters/
│   │   ├── base.py          # Abstract adapter interface
│   │   ├── alpaca_adapter.py    # Alpaca Markets (primary)
│   │   ├── polygon_adapter.py   # Polygon.io (secondary)
│   │   └── crypto_adapter.py    # CoinGecko + CoinCap (tertiary)
│   ├── cache/
│   │   └── price_cache.py   # TTL-based in-memory cache
│   ├── models.py            # Canonical data models
│   └── verification.py      # Cross-source price verification
├── indicators/              # 23+ Technical Indicators
│   ├── base.py              # Abstract indicator interface
│   ├── trend.py             # EMA, SMA
│   ├── volume.py            # OBV, CMF, Relative Volume
│   ├── momentum.py          # RSI, Stochastic, CCI
│   ├── volatility.py        # ATR, Bollinger Bands, Keltner
│   ├── directional.py       # ADX, DMI
│   ├── breakout.py          # Donchian Channel
│   ├── composite.py         # Ichimoku Cloud, VWAP
│   ├── oscillators.py       # Williams %R, MACD
│   ├── structural.py        # Volatility Regime, Market Structure
│   ├── quant.py             # Z-Score, Open Range Breakout, Pre-Market Skew
│   ├── divergence.py        # RSI/MACD Divergence Detector [NEW v2]
│   └── registry.py          # Central indicator registry
├── strategies/              # 35+ Trading Strategies
│   ├── base.py              # Abstract strategy interface
│   ├── trend_following.py   # 6 trend strategies
│   ├── momentum.py          # 5 momentum strategies
│   ├── volatility_breakout.py # 5 volatility strategies
│   ├── mean_reversion.py    # 4 mean reversion strategies
│   ├── smart_money.py       # 6 smart-money strategies
│   ├── session_aware.py     # 5 session-aware strategies
│   ├── hybrid.py            # 4 hybrid confluence strategies
│   └── registry.py          # Central strategy registry
├── engines/                 # Strategy, Signal & Risk Engines
│   ├── strategy_engine.py   # 7 strategy engine groups
│   ├── signal_engine.py     # Master signal engine v2 with quality + risk
│   ├── risk_manager.py      # Risk management (kill switch, sizing) [NEW v2]
│   ├── adaptive_sltp.py     # Adaptive SL/TP engine [NEW v2]
│   └── signal_quality.py    # Quality scorer, regime selector, perf tracker [NEW v2]
├── smart_money/             # Institutional Pattern Detection
│   └── detector.py          # 6 smart-money detectors
├── ml/                      # Machine Learning
│   ├── features/
│   │   └── feature_engineering.py  # Feature pipeline & labels
│   ├── models/              # Persisted model files
│   ├── trainer.py           # LightGBM, RF, LogReg ensemble
│   └── enhanced_trainer.py  # Calibrated ensemble + disagreement [NEW v2]
├── rl/                      # Reinforcement Learning
│   └── environment.py       # Gym env + Q-learning agent
├── backtest/                # Backtesting Engine
│   ├── backtester.py        # Walk-forward backtester
│   └── enhanced_backtester.py # Monte Carlo + regime breakdown [NEW v2]
├── telegram/                # Telegram Notifications
│   └── notifier.py          # Branded async notifier
├── api/                     # FastAPI Application
│   └── app.py               # REST API with all endpoints
├── db/                      # Database
│   └── schema.py            # SQLAlchemy models & schema
├── utils/                   # Utilities
│   ├── logger.py            # Structured logging
│   └── helpers.py           # Common helper functions
├── tests/                   # Test Suite (307 tests)
│   ├── conftest.py          # Shared fixtures
│   ├── unit/
│   │   ├── test_indicators.py   # Indicator tests
│   │   ├── test_strategies.py   # Strategy tests
│   │   └── test_enhancements.py # Enhancement module tests [NEW v2]
│   └── integration/
│       └── test_engines.py      # Engine, ML, RL, API tests
├── main.py                  # Entry point
├── requirements.txt         # Dependencies
├── Dockerfile               # Production Docker image
├── .env.example             # Environment template
└── pytest.ini               # Test configuration
```

---

## 🛡️ Profit-Certainty Architecture (v2.0)

### Signal Flow with Safety Layers

```
Raw Data → Cross-Source Verification (reject >0.5% deviation)
    ↓
23 Indicators → Divergence Detection
    ↓
Regime-Adaptive Strategy Selection (block unsuited strategies)
    ↓
35 Strategies → Per-Strategy Win-Rate Filter (auto-disable losers)
    ↓
7 Strategy Engines → Weighted Confluence
    ↓
Signal Quality Scorer (7-factor gate: volume, trend, momentum,
                       confirmation candle, divergence, ADX, volatility)
    ↓
Master Signal Engine (ML + Confluence + Smart Money + Vol + RL + Quality)
    ↓
Risk Manager Gate (drawdown kill switch, daily limits, position limits)
    ↓
Drawdown-Adjusted Confidence → Tier Classification
    ↓
Adaptive SL/TP (ATR + structure + regime + tier, min 2:1 R:R)
    ↓
Telegram Notification (branded, rate-limited, quiet hours)
```

### Risk Management Features

| Feature | Description |
|---------|-------------|
| **Kill Switch** | Auto-halts all trading at max drawdown (default 15%) |
| **Daily Loss Limit** | Stops trading after 3% daily loss |
| **Consecutive Loss Breaker** | After 5 losses, only Elite signals allowed |
| **Max Position Limit** | Maximum 5 concurrent positions |
| **Kelly Position Sizing** | Half-Kelly with tier/volatility/drawdown adjustments |
| **Correlation Filter** | Blocks correlated positions (>0.7 correlation) |
| **Drawdown Confidence Scaling** | Reduces confidence up to 50% during drawdowns |

### Signal Quality Factors (7-Point Check)

| Factor | Weight | Description |
|--------|--------|-------------|
| Volume Confirmation | 15% | Relative volume must be above average |
| Trend Alignment | 25% | Signal must align with EMA 50/200 trend |
| Momentum Confirmation | 15% | RSI + MACD must support direction |
| Confirmation Candle | 15% | Previous candle must confirm direction |
| No Adverse Divergence | 10% | No RSI/MACD divergence against signal |
| Trend Strength (ADX) | 10% | ADX must show adequate trend strength |
| Volatility Suitability | 10% | Signal type must match volatility regime |

---

## 🧠 Enhanced Confidence Formula (v2)

```
confidence = 0.35 × ML_probability (calibrated, disagreement-penalized)
           + 0.20 × strategy_confluence (regime-filtered)
           + 0.15 × smart_money_score
           + 0.10 × volatility_regime
           + 0.10 × RL_scaling_factor
           + 0.10 × quality_score (7-factor)
```

Then: `final_confidence = drawdown_adjustment(confidence)`

### Confidence Tiers

| Tier | Range | Risk Behavior |
|------|-------|---------------|
| 🔥 Elite | 80-100 | Full position, always delivered, passes loss breaker |
| 💪 Strong | 60-79 | Standard position, active hours delivery |
| 📊 Moderate | 40-59 | Reduced position, active hours delivery |
| 📉 Weak | 0-39 | Suppressed, no trade taken |

---

## 🎯 Adaptive SL/TP System

| Component | Method |
|-----------|--------|
| **Stop Loss** | ATR × 1.5 × regime_mult × tier_mult, snapped to structure support/resistance |
| **Take Profit 1** | ATR × 2.5 × regime_mult × tier_mult (partial exit) |
| **Take Profit 2** | ATR × 4.0 × regime_mult × tier_mult (full exit) |
| **Trailing Stop** | ATR × 2.0 × regime_mult |
| **Min R:R** | 2.0:1 enforced (TP1 ≥ 2× SL distance) |
| **SL Limits** | 0.3% minimum, 5.0% maximum |

---

## 📊 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/healthz` | Health check |
| GET | `/metrics` | System metrics |
| POST | `/api/v1/signal` | Generate signal for symbol |
| POST | `/api/v1/scan` | Scan all symbols |
| GET | `/api/v1/smart-money/{symbol}` | Smart-money analysis |
| GET | `/api/v1/verify-price/{symbol}` | Cross-source verification |
| GET | `/api/v1/deviation-report` | Deviation report |
| GET | `/api/v1/strategies` | List strategies |
| GET | `/api/v1/indicators` | List indicators |
| GET | `/api/v1/risk/report` | Risk management state |
| POST | `/api/v1/risk/reset-kill-switch` | Reset kill switch |
| GET | `/api/v1/strategy-performance` | Per-strategy win rates |
| POST | `/api/v1/telegram/mute/{symbol}` | Mute symbol |
| POST | `/api/v1/telegram/unmute/{symbol}` | Unmute symbol |
| GET | `/api/v1/telegram/stats` | Telegram stats |

---

## 🧪 Test Coverage (307 Tests)

| Category | Tests | Status |
|----------|-------|--------|
| Indicator Unit Tests | 35 | ✅ All Pass |
| Strategy Unit Tests | 180 | ✅ All Pass |
| Engine Integration Tests | 10 | ✅ All Pass |
| Signal Engine Tests (v2) | 5 | ✅ All Pass |
| Smart Money Tests | 3 | ✅ All Pass |
| ML/Feature Tests | 6 | ✅ All Pass |
| Enhanced ML Tests | 3 | ✅ All Pass |
| RL Environment Tests | 7 | ✅ All Pass |
| Backtester Tests | 3 | ✅ All Pass |
| Enhanced Backtester Tests | 3 | ✅ All Pass |
| Risk Manager Tests | 12 | ✅ All Pass |
| Adaptive SL/TP Tests | 6 | ✅ All Pass |
| Signal Quality Tests | 5 | ✅ All Pass |
| Regime Selector Tests | 5 | ✅ All Pass |
| Performance Tracker Tests | 3 | ✅ All Pass |
| Divergence Detector Tests | 3 | ✅ All Pass |
| Telegram Tests | 4 | ✅ All Pass |
| Data Verification Tests | 4 | ✅ All Pass |
| API Endpoint Tests | 4 | ✅ All Pass |
| Utility Tests | 6 | ✅ All Pass |
| **Total** | **307** | **✅ All Pass** |

---

## 🔧 Quick Start

```bash
# Install
pip install -r requirements.txt

# Configure
cp .env.example .env
# Edit .env with your API keys

# Test
python -m pytest tests/ -v

# Run
python main.py
```

## 🐳 Docker

```bash
docker build -t apex-signal .
docker run -p 8000:8000 --env-file .env apex-signal
```

---

*Built with ❤️ by APEX SIGNAL™ Team — v2.0 Enhanced Edition*