"""
APEX SIGNAL™ — Quant & Session Indicators
Z-Score Mean Reversion, Pre-Market Skew, Open-Range Breakouts
"""
import pandas as pd
import numpy as np
from apex_signal.indicators.base import BaseIndicator


class ZScoreIndicator(BaseIndicator):
    """Z-Score — measures how many standard deviations price is from the mean."""

    def __init__(self, period: int = 20):
        self.period = period
        super().__init__(name="zscore", params={"period": period})

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()

        rolling_mean = df["close"].rolling(window=self.period).mean()
        rolling_std = df["close"].rolling(window=self.period).std()

        df["zscore"] = (df["close"] - rolling_mean) / rolling_std.replace(0, np.nan)
        df["zscore"] = df["zscore"].fillna(0)

        # Extreme zones for mean reversion
        df["zscore_extreme_high"] = (df["zscore"] > 2.0).astype(int)
        df["zscore_extreme_low"] = (df["zscore"] < -2.0).astype(int)
        df["zscore_reverting"] = (
            ((df["zscore"].shift(1).abs() > 1.5) & (df["zscore"].abs() < df["zscore"].shift(1).abs()))
        ).astype(int)

        # Z-Score of volume
        vol_mean = df["volume"].rolling(window=self.period).mean()
        vol_std = df["volume"].rolling(window=self.period).std()
        df["zscore_volume"] = (df["volume"] - vol_mean) / vol_std.replace(0, np.nan)
        df["zscore_volume"] = df["zscore_volume"].fillna(0)

        self._last_result = df
        return df

    def reset(self) -> None:
        super().reset()


class OpenRangeBreakoutIndicator(BaseIndicator):
    """
    Open-Range Breakout — detects breakouts from the opening range.
    Uses the first N candles to define the opening range, then flags breakouts.
    """

    def __init__(self, range_candles: int = 5):
        self.range_candles = range_candles
        super().__init__(name="orb", params={"range_candles": range_candles})

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        n = self.range_candles

        if len(df) < n + 1:
            df["orb_high"] = np.nan
            df["orb_low"] = np.nan
            df["orb_break_up"] = 0
            df["orb_break_down"] = 0
            df["orb_range_pct"] = 0.0
            self._last_result = df
            return df

        # Opening range from first N candles
        orb_high = df["high"].iloc[:n].max()
        orb_low = df["low"].iloc[:n].min()
        orb_mid = (orb_high + orb_low) / 2.0

        df["orb_high"] = orb_high
        df["orb_low"] = orb_low
        df["orb_mid"] = orb_mid

        # Range as percentage
        df["orb_range_pct"] = ((orb_high - orb_low) / orb_mid) * 100.0 if orb_mid > 0 else 0.0

        # Breakout detection (only after opening range period)
        df["orb_break_up"] = 0
        df["orb_break_down"] = 0

        if len(df) > n:
            df.iloc[n:, df.columns.get_loc("orb_break_up")] = (
                df["close"].iloc[n:] > orb_high
            ).astype(int).values

            df.iloc[n:, df.columns.get_loc("orb_break_down")] = (
                df["close"].iloc[n:] < orb_low
            ).astype(int).values

        # Distance from ORB levels
        df["orb_dist_high_pct"] = ((df["close"] - orb_high) / orb_high) * 100.0 if orb_high > 0 else 0.0
        df["orb_dist_low_pct"] = ((df["close"] - orb_low) / orb_low) * 100.0 if orb_low > 0 else 0.0

        self._last_result = df
        return df

    def reset(self) -> None:
        super().reset()


class PreMarketSkewIndicator(BaseIndicator):
    """
    Pre-Market Skew — measures the directional bias from pre-market activity.
    Compares the first candle's open to the previous close to detect gap direction.
    """

    def __init__(self):
        super().__init__(name="premarket_skew")

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()

        # Gap analysis: open vs previous close
        df["gap"] = df["open"] - df["close"].shift(1)
        df["gap_pct"] = (df["gap"] / df["close"].shift(1).replace(0, np.nan)) * 100.0
        df["gap_pct"] = df["gap_pct"].fillna(0)

        # Gap classification
        df["gap_up"] = (df["gap_pct"] > 0.1).astype(int)
        df["gap_down"] = (df["gap_pct"] < -0.1).astype(int)
        df["gap_significant"] = (df["gap_pct"].abs() > 0.5).astype(int)

        # Skew direction: positive = bullish pre-market, negative = bearish
        df["premarket_skew"] = np.where(
            df["gap_pct"] > 0.1, 1,
            np.where(df["gap_pct"] < -0.1, -1, 0)
        )

        # Gap fill detection (price returns to previous close level)
        prev_close = df["close"].shift(1)
        df["gap_filled"] = (
            ((df["gap_up"] == 1) & (df["low"] <= prev_close)) |
            ((df["gap_down"] == 1) & (df["high"] >= prev_close))
        ).astype(int)

        self._last_result = df
        return df

    def reset(self) -> None:
        super().reset()