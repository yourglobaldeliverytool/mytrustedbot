"""
APEX SIGNAL™ — Volatility Indicators
ATR (14), Bollinger Bands (20,2), Keltner Channels
"""
import pandas as pd
import numpy as np
from apex_signal.indicators.base import BaseIndicator


class ATRIndicator(BaseIndicator):
    """Average True Range — measures market volatility."""

    def __init__(self, period: int = 14):
        self.period = period
        super().__init__(name="atr", params={"period": period})

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        high_low = df["high"] - df["low"]
        high_close = (df["high"] - df["close"].shift(1)).abs()
        low_close = (df["low"] - df["close"].shift(1)).abs()

        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df["tr"] = true_range
        df["atr"] = true_range.ewm(alpha=1.0 / self.period, min_periods=self.period, adjust=False).mean()

        # ATR percentage (normalized)
        df["atr_pct"] = (df["atr"] / df["close"]) * 100.0
        self._last_result = df
        return df

    def reset(self) -> None:
        super().reset()


class BollingerBandsIndicator(BaseIndicator):
    """Bollinger Bands — volatility bands around a moving average."""

    def __init__(self, period: int = 20, std_dev: float = 2.0):
        self.period = period
        self.std_dev = std_dev
        super().__init__(name="bollinger", params={"period": period, "std_dev": std_dev})

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        sma = df["close"].rolling(window=self.period).mean()
        std = df["close"].rolling(window=self.period).std()

        df["bb_middle"] = sma
        df["bb_upper"] = sma + (self.std_dev * std)
        df["bb_lower"] = sma - (self.std_dev * std)

        # Bandwidth and %B
        bb_range = df["bb_upper"] - df["bb_lower"]
        df["bb_bandwidth"] = bb_range / df["bb_middle"].replace(0, np.nan)
        df["bb_bandwidth"] = df["bb_bandwidth"].fillna(0)

        df["bb_pct_b"] = (df["close"] - df["bb_lower"]) / bb_range.replace(0, np.nan)
        df["bb_pct_b"] = df["bb_pct_b"].fillna(0.5)

        # Squeeze detection (bandwidth below 20-period low)
        df["bb_squeeze"] = (df["bb_bandwidth"] <= df["bb_bandwidth"].rolling(20).min() * 1.05).astype(int)
        self._last_result = df
        return df

    def reset(self) -> None:
        super().reset()


class KeltnerChannelIndicator(BaseIndicator):
    """Keltner Channels — volatility-based envelope around EMA."""

    def __init__(self, period: int = 20, atr_mult: float = 1.5):
        self.period = period
        self.atr_mult = atr_mult
        super().__init__(name="keltner", params={"period": period, "atr_mult": atr_mult})

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        # EMA center line
        df["kc_middle"] = df["close"].ewm(span=self.period, adjust=False).mean()

        # ATR for channel width
        high_low = df["high"] - df["low"]
        high_close = (df["high"] - df["close"].shift(1)).abs()
        low_close = (df["low"] - df["close"].shift(1)).abs()
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.ewm(alpha=1.0 / self.period, min_periods=self.period, adjust=False).mean()

        df["kc_upper"] = df["kc_middle"] + (self.atr_mult * atr)
        df["kc_lower"] = df["kc_middle"] - (self.atr_mult * atr)

        # Position within channel
        kc_range = df["kc_upper"] - df["kc_lower"]
        df["kc_pct"] = (df["close"] - df["kc_lower"]) / kc_range.replace(0, np.nan)
        df["kc_pct"] = df["kc_pct"].fillna(0.5)

        # Breakout flags
        df["kc_above"] = (df["close"] > df["kc_upper"]).astype(int)
        df["kc_below"] = (df["close"] < df["kc_lower"]).astype(int)
        self._last_result = df
        return df

    def reset(self) -> None:
        super().reset()