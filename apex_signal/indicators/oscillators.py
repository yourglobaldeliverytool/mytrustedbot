"""
APEX SIGNAL™ — Oscillator Indicators
Williams %R, MACD (12, 26, 9)
"""
import pandas as pd
import numpy as np
from apex_signal.indicators.base import BaseIndicator


class WilliamsRIndicator(BaseIndicator):
    """Williams %R — momentum oscillator measuring overbought/oversold levels."""

    def __init__(self, period: int = 14):
        self.period = period
        super().__init__(name="williams_r", params={"period": period})

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()

        highest_high = df["high"].rolling(window=self.period).max()
        lowest_low = df["low"].rolling(window=self.period).min()

        hl_range = highest_high - lowest_low
        df["williams_r"] = -100.0 * (highest_high - df["close"]) / hl_range.replace(0, np.nan)
        df["williams_r"] = df["williams_r"].fillna(-50.0)

        # Overbought/Oversold zones
        df["wr_overbought"] = (df["williams_r"] > -20).astype(int)
        df["wr_oversold"] = (df["williams_r"] < -80).astype(int)

        # Signal line (smoothed)
        df["williams_r_signal"] = df["williams_r"].rolling(window=5).mean()

        self._last_result = df
        return df

    def reset(self) -> None:
        super().reset()


class MACDIndicator(BaseIndicator):
    """MACD — Moving Average Convergence Divergence trend-following momentum indicator."""

    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9):
        self.fast = fast
        self.slow = slow
        self.signal_period = signal
        super().__init__(name="macd", params={
            "fast": fast, "slow": slow, "signal": signal
        })

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()

        ema_fast = df["close"].ewm(span=self.fast, adjust=False).mean()
        ema_slow = df["close"].ewm(span=self.slow, adjust=False).mean()

        df["macd_line"] = ema_fast - ema_slow
        df["macd_signal"] = df["macd_line"].ewm(span=self.signal_period, adjust=False).mean()
        df["macd_histogram"] = df["macd_line"] - df["macd_signal"]

        # Crossover signals
        df["macd_cross_bull"] = (
            (df["macd_line"] > df["macd_signal"]) &
            (df["macd_line"].shift(1) <= df["macd_signal"].shift(1))
        ).astype(int)

        df["macd_cross_bear"] = (
            (df["macd_line"] < df["macd_signal"]) &
            (df["macd_line"].shift(1) >= df["macd_signal"].shift(1))
        ).astype(int)

        # Histogram momentum (increasing or decreasing)
        df["macd_hist_rising"] = (df["macd_histogram"] > df["macd_histogram"].shift(1)).astype(int)

        # Zero line cross
        df["macd_above_zero"] = (df["macd_line"] > 0).astype(int)

        self._last_result = df
        return df

    def reset(self) -> None:
        super().reset()