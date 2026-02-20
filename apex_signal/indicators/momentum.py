"""
APEX SIGNAL™ — Momentum Indicators
RSI (14), Stochastic Oscillator, CCI
"""
import pandas as pd
import numpy as np
from apex_signal.indicators.base import BaseIndicator


class RSIIndicator(BaseIndicator):
    """Relative Strength Index — momentum oscillator measuring speed of price changes."""

    def __init__(self, period: int = 14):
        self.period = period
        super().__init__(name="rsi", params={"period": period})

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)

        avg_gain = gain.ewm(alpha=1.0 / self.period, min_periods=self.period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1.0 / self.period, min_periods=self.period, adjust=False).mean()

        rs = avg_gain / avg_loss.replace(0, np.nan)
        df["rsi"] = 100.0 - (100.0 / (1.0 + rs))
        df["rsi"] = df["rsi"].fillna(50.0)

        # Overbought/Oversold zones
        df["rsi_overbought"] = (df["rsi"] > 70).astype(int)
        df["rsi_oversold"] = (df["rsi"] < 30).astype(int)
        self._last_result = df
        return df

    def reset(self) -> None:
        super().reset()


class StochasticOscillator(BaseIndicator):
    """Stochastic Oscillator — compares closing price to price range over a period."""

    def __init__(self, k_period: int = 14, d_period: int = 3):
        self.k_period = k_period
        self.d_period = d_period
        super().__init__(name="stochastic", params={"k_period": k_period, "d_period": d_period})

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        low_min = df["low"].rolling(window=self.k_period).min()
        high_max = df["high"].rolling(window=self.k_period).max()

        hl_range = high_max - low_min
        hl_range = hl_range.replace(0, np.nan)

        df["stoch_k"] = 100.0 * (df["close"] - low_min) / hl_range
        df["stoch_k"] = df["stoch_k"].fillna(50.0)
        df["stoch_d"] = df["stoch_k"].rolling(window=self.d_period).mean()
        df["stoch_d"] = df["stoch_d"].fillna(50.0)

        # Crossover signals
        df["stoch_overbought"] = (df["stoch_k"] > 80).astype(int)
        df["stoch_oversold"] = (df["stoch_k"] < 20).astype(int)
        self._last_result = df
        return df

    def reset(self) -> None:
        super().reset()


class CCIIndicator(BaseIndicator):
    """Commodity Channel Index — measures deviation from statistical mean."""

    def __init__(self, period: int = 20):
        self.period = period
        super().__init__(name="cci", params={"period": period})

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        typical_price = (df["high"] + df["low"] + df["close"]) / 3.0
        sma_tp = typical_price.rolling(window=self.period).mean()
        mean_dev = typical_price.rolling(window=self.period).apply(
            lambda x: np.mean(np.abs(x - x.mean())), raw=True
        )
        mean_dev = mean_dev.replace(0, np.nan)
        df["cci"] = (typical_price - sma_tp) / (0.015 * mean_dev)
        df["cci"] = df["cci"].fillna(0.0)

        df["cci_overbought"] = (df["cci"] > 100).astype(int)
        df["cci_oversold"] = (df["cci"] < -100).astype(int)
        self._last_result = df
        return df

    def reset(self) -> None:
        super().reset()