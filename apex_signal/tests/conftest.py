"""
APEX SIGNAL™ — Test Configuration & Fixtures
Shared fixtures for all test modules.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta


@pytest.fixture
def sample_ohlcv_df():
    """Generate a realistic OHLCV DataFrame for testing."""
    np.random.seed(42)
    n = 200
    dates = pd.date_range(start="2024-01-01", periods=n, freq="1min", tz=timezone.utc)

    # Generate realistic price data with trend and noise
    base_price = 100.0
    returns = np.random.normal(0.0001, 0.002, n)
    prices = base_price * np.exp(np.cumsum(returns))

    # Add some trend
    trend = np.linspace(0, 5, n)
    prices = prices + trend

    high_noise = np.abs(np.random.normal(0, 0.5, n))
    low_noise = np.abs(np.random.normal(0, 0.5, n))

    df = pd.DataFrame({
        "open": prices + np.random.normal(0, 0.1, n),
        "high": prices + high_noise,
        "low": prices - low_noise,
        "close": prices,
        "volume": np.random.randint(1000, 50000, n).astype(float),
    }, index=dates)

    # Ensure high >= close >= low and high >= open >= low
    df["high"] = df[["open", "high", "close"]].max(axis=1) + 0.01
    df["low"] = df[["open", "low", "close"]].min(axis=1) - 0.01

    return df


@pytest.fixture
def small_ohlcv_df():
    """Small DataFrame for edge case testing."""
    dates = pd.date_range(start="2024-01-01", periods=10, freq="1min", tz=timezone.utc)
    return pd.DataFrame({
        "open": [100, 101, 102, 101, 103, 104, 103, 105, 106, 107],
        "high": [101, 102, 103, 102, 104, 105, 104, 106, 107, 108],
        "low": [99, 100, 101, 100, 102, 103, 102, 104, 105, 106],
        "close": [100.5, 101.5, 102.5, 101.5, 103.5, 104.5, 103.5, 105.5, 106.5, 107.5],
        "volume": [10000, 12000, 15000, 8000, 20000, 25000, 11000, 30000, 18000, 22000],
    }, index=dates, dtype=float)


@pytest.fixture
def enriched_df(sample_ohlcv_df):
    """DataFrame with all indicators computed."""
    from apex_signal.indicators.registry import IndicatorRegistry
    registry = IndicatorRegistry()
    return registry.compute_all(sample_ohlcv_df)


@pytest.fixture
def empty_df():
    """Empty DataFrame for edge case testing."""
    return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])


@pytest.fixture
def trending_up_df():
    """DataFrame with clear uptrend for strategy testing."""
    np.random.seed(123)
    n = 200
    dates = pd.date_range(start="2024-01-01", periods=n, freq="1min", tz=timezone.utc)

    # Strong uptrend
    base = 100.0
    prices = base + np.linspace(0, 30, n) + np.random.normal(0, 0.3, n)

    df = pd.DataFrame({
        "open": prices - 0.1,
        "high": prices + np.abs(np.random.normal(0.3, 0.2, n)),
        "low": prices - np.abs(np.random.normal(0.3, 0.2, n)),
        "close": prices,
        "volume": np.random.randint(5000, 30000, n).astype(float),
    }, index=dates)

    df["high"] = df[["open", "high", "close"]].max(axis=1) + 0.01
    df["low"] = df[["open", "low", "close"]].min(axis=1) - 0.01
    return df


@pytest.fixture
def trending_down_df():
    """DataFrame with clear downtrend for strategy testing."""
    np.random.seed(456)
    n = 200
    dates = pd.date_range(start="2024-01-01", periods=n, freq="1min", tz=timezone.utc)

    base = 130.0
    prices = base - np.linspace(0, 30, n) + np.random.normal(0, 0.3, n)

    df = pd.DataFrame({
        "open": prices + 0.1,
        "high": prices + np.abs(np.random.normal(0.3, 0.2, n)),
        "low": prices - np.abs(np.random.normal(0.3, 0.2, n)),
        "close": prices,
        "volume": np.random.randint(5000, 30000, n).astype(float),
    }, index=dates)

    df["high"] = df[["open", "high", "close"]].max(axis=1) + 0.01
    df["low"] = df[["open", "low", "close"]].min(axis=1) - 0.01
    return df