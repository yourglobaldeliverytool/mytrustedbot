"""
APEX SIGNAL™ — Feature Engineering Pipeline
Builds ML features from indicators and smart-money signals.
"""
import pandas as pd
import numpy as np
from typing import List, Tuple, Optional

from apex_signal.utils.logger import get_logger

logger = get_logger("feature_engineering")

# Feature columns used by ML models
FEATURE_COLUMNS = [
    # Trend
    "ema_8", "ema_20", "ema_50", "ema_200",
    "sma_20", "sma_50",
    # Momentum
    "rsi", "stoch_k", "stoch_d", "cci",
    "macd_line", "macd_signal", "macd_histogram",
    "williams_r",
    # Volume
    "obv", "cmf", "rvol",
    # Volatility
    "atr", "atr_pct", "bb_bandwidth", "bb_pct_b",
    "kc_pct",
    # Directional
    "adx", "plus_di", "minus_di",
    # Structure
    "vol_regime", "vol_regime_percentile",
    "ms_position",
    "dc_position", "dc_width",
    # VWAP
    "vwap_distance_pct", "vwap_bias",
    # Derived
    "zscore", "zscore_volume",
]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build ML feature matrix from indicator-enriched DataFrame.
    Adds derived features and normalizes.
    """
    features = pd.DataFrame(index=df.index)

    # Copy available indicator columns
    for col in FEATURE_COLUMNS:
        if col in df.columns:
            features[col] = df[col].astype(float)
        else:
            features[col] = 0.0

    # Derived features
    if "close" in df.columns:
        close = df["close"]

        # Price momentum (returns)
        features["return_1"] = close.pct_change(1).fillna(0)
        features["return_3"] = close.pct_change(3).fillna(0)
        features["return_5"] = close.pct_change(5).fillna(0)
        features["return_10"] = close.pct_change(10).fillna(0)

        # Price relative to EMAs
        if "ema_20" in df.columns:
            features["price_vs_ema20"] = ((close - df["ema_20"]) / df["ema_20"].replace(0, np.nan)).fillna(0)
        if "ema_50" in df.columns:
            features["price_vs_ema50"] = ((close - df["ema_50"]) / df["ema_50"].replace(0, np.nan)).fillna(0)

        # Volatility features
        features["log_return_std_10"] = np.log(close / close.shift(1)).rolling(10).std().fillna(0)
        features["log_return_std_20"] = np.log(close / close.shift(1)).rolling(20).std().fillna(0)

        # Candle features
        if all(c in df.columns for c in ["open", "high", "low"]):
            body = (close - df["open"]).abs()
            full_range = (df["high"] - df["low"]).replace(0, np.nan)
            features["body_ratio"] = (body / full_range).fillna(0.5)
            features["upper_wick_ratio"] = ((df["high"] - df[["close", "open"]].max(axis=1)) / full_range).fillna(0)
            features["lower_wick_ratio"] = ((df[["close", "open"]].min(axis=1) - df["low"]) / full_range).fillna(0)
            features["bullish_candle"] = (close > df["open"]).astype(float)

    # Fill remaining NaN
    features = features.fillna(0)

    # Replace infinities
    features = features.replace([np.inf, -np.inf], 0)

    return features


def build_labels(df: pd.DataFrame, tp_pips: float = 20.0, sl_pips: float = 10.0,
                 forward_bars: int = 10) -> pd.Series:
    """
    Build binary labels: 1 if take-profit hit before stop-loss, 0 otherwise.
    Uses forward-looking price action.
    """
    if "close" not in df.columns or len(df) < forward_bars + 1:
        return pd.Series(0, index=df.index)

    close = df["close"].values
    labels = np.zeros(len(df))

    # Normalize pips to price units (approximate)
    avg_price = np.mean(close[close > 0]) if np.any(close > 0) else 1.0
    tp_price = tp_pips * avg_price / 10000.0
    sl_price = sl_pips * avg_price / 10000.0

    for i in range(len(df) - forward_bars):
        entry = close[i]
        future = close[i + 1:i + forward_bars + 1]

        max_gain = np.max(future) - entry
        max_loss = entry - np.min(future)

        # TP hit before SL
        if max_gain >= tp_price and max_loss < sl_price:
            labels[i] = 1
        # Check which hit first
        elif max_gain >= tp_price and max_loss >= sl_price:
            tp_idx = np.argmax(future >= entry + tp_price)
            sl_idx = np.argmax(future <= entry - sl_price)
            if tp_idx <= sl_idx:
                labels[i] = 1

    return pd.Series(labels, index=df.index)


def prepare_training_data(df: pd.DataFrame, tp_pips: float = 20.0,
                          sl_pips: float = 10.0) -> Tuple[pd.DataFrame, pd.Series]:
    """Prepare features and labels for ML training."""
    features = build_features(df)
    labels = build_labels(df, tp_pips, sl_pips)

    # Remove rows with NaN
    valid_mask = features.notna().all(axis=1) & labels.notna()
    features = features[valid_mask]
    labels = labels[valid_mask]

    # Remove last N rows (no valid labels)
    cutoff = max(1, len(features) - 10)
    features = features.iloc[:cutoff]
    labels = labels.iloc[:cutoff]

    logger.info("training_data_prepared", features_shape=features.shape,
                label_distribution=dict(labels.value_counts()))

    return features, labels