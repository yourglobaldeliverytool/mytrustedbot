"""
APEX SIGNAL™ — Volume Indicators
OBV, Chaikin Money Flow, Relative Volume
"""
import pandas as pd
import numpy as np
from apex_signal.indicators.base import BaseIndicator


class OBVIndicator(BaseIndicator):
    """On-Balance Volume — cumulative volume flow indicator."""

    def __init__(self):
        super().__init__(name="obv")

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        direction = np.where(df["close"] > df["close"].shift(1), 1,
                    np.where(df["close"] < df["close"].shift(1), -1, 0))
        df["obv"] = (direction * df["volume"]).cumsum()
        # OBV EMA for signal line
        df["obv_ema"] = df["obv"].ewm(span=20, adjust=False).mean()
        self._last_result = df
        return df

    def reset(self) -> None:
        super().reset()


class ChaikinMoneyFlowIndicator(BaseIndicator):
    """Chaikin Money Flow — measures buying/selling pressure over a period."""

    def __init__(self, period: int = 20):
        self.period = period
        super().__init__(name="cmf", params={"period": period})

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        hl_range = df["high"] - df["low"]
        # Avoid division by zero
        hl_range = hl_range.replace(0, np.nan)
        mf_multiplier = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / hl_range
        mf_multiplier = mf_multiplier.fillna(0)
        mf_volume = mf_multiplier * df["volume"]
        df["cmf"] = mf_volume.rolling(window=self.period).sum() / df["volume"].rolling(window=self.period).sum()
        df["cmf"] = df["cmf"].fillna(0)
        self._last_result = df
        return df

    def reset(self) -> None:
        super().reset()


class RelativeVolumeIndicator(BaseIndicator):
    """Relative Volume — current volume vs average volume ratio."""

    def __init__(self, period: int = 20):
        self.period = period
        super().__init__(name="rvol", params={"period": period})

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        avg_vol = df["volume"].rolling(window=self.period).mean()
        df["rvol"] = df["volume"] / avg_vol.replace(0, np.nan)
        df["rvol"] = df["rvol"].fillna(1.0)
        # Flag high relative volume (>1.5x average)
        df["rvol_high"] = (df["rvol"] > 1.5).astype(int)
        self._last_result = df
        return df

    def reset(self) -> None:
        super().reset()