"""
APEX SIGNAL™ — Directional Indicators
ADX (14), DMI (Directional Movement Index)
"""
import pandas as pd
import numpy as np
from apex_signal.indicators.base import BaseIndicator


class ADXIndicator(BaseIndicator):
    """Average Directional Index with +DI/-DI — measures trend strength."""

    def __init__(self, period: int = 14):
        self.period = period
        super().__init__(name="adx", params={"period": period})

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()

        # True Range
        high_low = df["high"] - df["low"]
        high_close = (df["high"] - df["close"].shift(1)).abs()
        low_close = (df["low"] - df["close"].shift(1)).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

        # Directional Movement
        up_move = df["high"] - df["high"].shift(1)
        down_move = df["low"].shift(1) - df["low"]

        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

        plus_dm = pd.Series(plus_dm, index=df.index)
        minus_dm = pd.Series(minus_dm, index=df.index)

        # Smoothed values using Wilder's smoothing (EMA with alpha=1/period)
        alpha = 1.0 / self.period
        atr_smooth = tr.ewm(alpha=alpha, min_periods=self.period, adjust=False).mean()
        plus_dm_smooth = plus_dm.ewm(alpha=alpha, min_periods=self.period, adjust=False).mean()
        minus_dm_smooth = minus_dm.ewm(alpha=alpha, min_periods=self.period, adjust=False).mean()

        # Directional Indicators
        df["plus_di"] = 100.0 * plus_dm_smooth / atr_smooth.replace(0, np.nan)
        df["minus_di"] = 100.0 * minus_dm_smooth / atr_smooth.replace(0, np.nan)
        df["plus_di"] = df["plus_di"].fillna(0)
        df["minus_di"] = df["minus_di"].fillna(0)

        # DX and ADX
        di_sum = df["plus_di"] + df["minus_di"]
        di_diff = (df["plus_di"] - df["minus_di"]).abs()
        dx = 100.0 * di_diff / di_sum.replace(0, np.nan)
        dx = dx.fillna(0)

        df["adx"] = dx.ewm(alpha=alpha, min_periods=self.period, adjust=False).mean()

        # Trend strength classification
        df["adx_strong_trend"] = (df["adx"] > 25).astype(int)
        df["adx_weak_trend"] = (df["adx"] < 20).astype(int)

        # Directional bias
        df["dmi_bullish"] = (df["plus_di"] > df["minus_di"]).astype(int)
        df["dmi_bearish"] = (df["minus_di"] > df["plus_di"]).astype(int)

        self._last_result = df
        return df

    def reset(self) -> None:
        super().reset()