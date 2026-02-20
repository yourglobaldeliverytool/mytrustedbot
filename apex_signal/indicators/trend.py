"""
APEX SIGNAL™ — Trend Indicators
EMA (8, 20, 50, 200), SMA (20, 50, 100)
"""
import pandas as pd
import numpy as np
from typing import List
from apex_signal.indicators.base import BaseIndicator


class EMAIndicator(BaseIndicator):
    """Exponential Moving Average indicator for multiple periods."""

    def __init__(self, periods: List[int] = None):
        self.periods = periods or [8, 20, 50, 200]
        super().__init__(name="ema", params={"periods": self.periods})

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        for period in self.periods:
            col_name = f"ema_{period}"
            df[col_name] = df["close"].ewm(span=period, adjust=False).mean()
        self._last_result = df
        return df

    def reset(self) -> None:
        super().reset()


class SMAIndicator(BaseIndicator):
    """Simple Moving Average indicator for multiple periods."""

    def __init__(self, periods: List[int] = None):
        self.periods = periods or [20, 50, 100]
        super().__init__(name="sma", params={"periods": self.periods})

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        for period in self.periods:
            col_name = f"sma_{period}"
            df[col_name] = df["close"].rolling(window=period).mean()
        self._last_result = df
        return df

    def reset(self) -> None:
        super().reset()