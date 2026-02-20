"""
APEX SIGNAL™ — Breakout Indicators
Donchian Channel, Donchian Width
"""
import pandas as pd
import numpy as np
from apex_signal.indicators.base import BaseIndicator


class DonchianChannelIndicator(BaseIndicator):
    """Donchian Channel — highest high and lowest low over a period."""

    def __init__(self, period: int = 20):
        self.period = period
        super().__init__(name="donchian", params={"period": period})

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()

        df["dc_upper"] = df["high"].rolling(window=self.period).max()
        df["dc_lower"] = df["low"].rolling(window=self.period).min()
        df["dc_middle"] = (df["dc_upper"] + df["dc_lower"]) / 2.0

        # Donchian Width (normalized by middle)
        dc_range = df["dc_upper"] - df["dc_lower"]
        df["dc_width"] = dc_range / df["dc_middle"].replace(0, np.nan)
        df["dc_width"] = df["dc_width"].fillna(0)

        # Breakout flags
        df["dc_upper_break"] = (df["close"] >= df["dc_upper"]).astype(int)
        df["dc_lower_break"] = (df["close"] <= df["dc_lower"]).astype(int)

        # Position within channel (0=bottom, 1=top)
        df["dc_position"] = (df["close"] - df["dc_lower"]) / dc_range.replace(0, np.nan)
        df["dc_position"] = df["dc_position"].fillna(0.5)

        # Width expansion detection
        df["dc_width_expanding"] = (df["dc_width"] > df["dc_width"].shift(1)).astype(int)

        self._last_result = df
        return df

    def reset(self) -> None:
        super().reset()