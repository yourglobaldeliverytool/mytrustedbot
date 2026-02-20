"""
APEX SIGNAL™ — Structural Indicators
Volatility Regime Detector, Market Structure Bands
"""
import pandas as pd
import numpy as np
from apex_signal.indicators.base import BaseIndicator


class VolatilityRegimeDetector(BaseIndicator):
    """
    Volatility Regime Detector — classifies market into low/normal/high volatility regimes
    using ATR percentile ranking and rolling standard deviation analysis.
    """

    def __init__(self, atr_period: int = 14, lookback: int = 100):
        self.atr_period = atr_period
        self.lookback = lookback
        super().__init__(name="vol_regime", params={
            "atr_period": atr_period, "lookback": lookback
        })

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()

        # Calculate ATR
        high_low = df["high"] - df["low"]
        high_close = (df["high"] - df["close"].shift(1)).abs()
        low_close = (df["low"] - df["close"].shift(1)).abs()
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.ewm(alpha=1.0 / self.atr_period, min_periods=self.atr_period, adjust=False).mean()

        # ATR as percentage of close
        atr_pct = (atr / df["close"].replace(0, np.nan)) * 100.0

        # Rolling percentile rank of ATR
        df["vol_regime_atr_pct"] = atr_pct.fillna(0)
        df["vol_regime_percentile"] = atr_pct.rolling(window=self.lookback).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else 0.5,
            raw=False
        ).fillna(0.5)

        # Returns volatility (rolling std of log returns)
        log_returns = np.log(df["close"] / df["close"].shift(1))
        df["vol_regime_ret_std"] = log_returns.rolling(window=self.lookback).std().fillna(0)

        # Regime classification
        # Low: percentile < 0.25, Normal: 0.25-0.75, High: > 0.75
        df["vol_regime"] = np.where(
            df["vol_regime_percentile"] < 0.25, 0,  # Low volatility
            np.where(df["vol_regime_percentile"] > 0.75, 2, 1)  # High=2, Normal=1
        )

        # Regime labels for readability
        regime_map = {0: "low", 1: "normal", 2: "high"}
        df["vol_regime_label"] = df["vol_regime"].map(regime_map)

        # Regime change detection
        df["vol_regime_change"] = (df["vol_regime"] != df["vol_regime"].shift(1)).astype(int)

        self._last_result = df
        return df

    def reset(self) -> None:
        super().reset()


class MarketStructureBands(BaseIndicator):
    """
    Market Structure Bands — adaptive support/resistance bands based on
    swing highs/lows and pivot point analysis.
    """

    def __init__(self, swing_period: int = 10, band_lookback: int = 50):
        self.swing_period = swing_period
        self.band_lookback = band_lookback
        super().__init__(name="market_structure", params={
            "swing_period": swing_period, "band_lookback": band_lookback
        })

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        n = self.swing_period

        # Detect swing highs and swing lows
        df["swing_high"] = 0.0
        df["swing_low"] = 0.0

        highs = df["high"].values
        lows = df["low"].values

        swing_highs = np.full(len(df), np.nan)
        swing_lows = np.full(len(df), np.nan)

        for i in range(n, len(df) - n):
            # Swing high: highest in window
            if highs[i] == max(highs[i - n:i + n + 1]):
                swing_highs[i] = highs[i]
            # Swing low: lowest in window
            if lows[i] == min(lows[i - n:i + n + 1]):
                swing_lows[i] = lows[i]

        df["swing_high"] = swing_highs
        df["swing_low"] = swing_lows

        # Forward-fill swing levels to create structure bands
        df["ms_resistance"] = df["swing_high"].ffill()
        df["ms_support"] = df["swing_low"].ffill()

        # Fill initial NaN values
        df["ms_resistance"] = df["ms_resistance"].fillna(df["high"].rolling(self.band_lookback).max())
        df["ms_support"] = df["ms_support"].fillna(df["low"].rolling(self.band_lookback).min())

        # Structure band midline
        df["ms_midline"] = (df["ms_resistance"] + df["ms_support"]) / 2.0

        # Position within structure
        ms_range = df["ms_resistance"] - df["ms_support"]
        df["ms_position"] = (df["close"] - df["ms_support"]) / ms_range.replace(0, np.nan)
        df["ms_position"] = df["ms_position"].fillna(0.5).clip(0, 1)

        # Break of structure signals
        df["ms_break_above"] = (
            (df["close"] > df["ms_resistance"]) &
            (df["close"].shift(1) <= df["ms_resistance"].shift(1))
        ).astype(int)

        df["ms_break_below"] = (
            (df["close"] < df["ms_support"]) &
            (df["close"].shift(1) >= df["ms_support"].shift(1))
        ).astype(int)

        # Higher highs / lower lows detection
        df["ms_higher_high"] = (df["ms_resistance"] > df["ms_resistance"].shift(n)).astype(int)
        df["ms_lower_low"] = (df["ms_support"] < df["ms_support"].shift(n)).astype(int)

        self._last_result = df
        return df

    def reset(self) -> None:
        super().reset()