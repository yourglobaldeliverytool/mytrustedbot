"""
APEX SIGNAL™ — Composite Indicators
Ichimoku Cloud, VWAP (anchored)
"""
import pandas as pd
import numpy as np
from apex_signal.indicators.base import BaseIndicator


class IchimokuIndicator(BaseIndicator):
    """Ichimoku Cloud — comprehensive trend, support/resistance, and momentum system."""

    def __init__(self, tenkan: int = 9, kijun: int = 26, senkou_b: int = 52):
        self.tenkan = tenkan
        self.kijun = kijun
        self.senkou_b = senkou_b
        super().__init__(name="ichimoku", params={
            "tenkan": tenkan, "kijun": kijun, "senkou_b": senkou_b
        })

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()

        # Tenkan-sen (Conversion Line)
        tenkan_high = df["high"].rolling(window=self.tenkan).max()
        tenkan_low = df["low"].rolling(window=self.tenkan).min()
        df["ichi_tenkan"] = (tenkan_high + tenkan_low) / 2.0

        # Kijun-sen (Base Line)
        kijun_high = df["high"].rolling(window=self.kijun).max()
        kijun_low = df["low"].rolling(window=self.kijun).min()
        df["ichi_kijun"] = (kijun_high + kijun_low) / 2.0

        # Senkou Span A (Leading Span A) — shifted forward by kijun periods
        df["ichi_senkou_a"] = ((df["ichi_tenkan"] + df["ichi_kijun"]) / 2.0).shift(self.kijun)

        # Senkou Span B (Leading Span B) — shifted forward by kijun periods
        senkou_b_high = df["high"].rolling(window=self.senkou_b).max()
        senkou_b_low = df["low"].rolling(window=self.senkou_b).min()
        df["ichi_senkou_b"] = ((senkou_b_high + senkou_b_low) / 2.0).shift(self.kijun)

        # Chikou Span (Lagging Span) — close shifted back by kijun periods
        df["ichi_chikou"] = df["close"].shift(-self.kijun)

        # Cloud color: green if senkou_a > senkou_b, red otherwise
        df["ichi_cloud_green"] = (df["ichi_senkou_a"] > df["ichi_senkou_b"]).astype(int)

        # Price position relative to cloud
        cloud_top = df[["ichi_senkou_a", "ichi_senkou_b"]].max(axis=1)
        cloud_bottom = df[["ichi_senkou_a", "ichi_senkou_b"]].min(axis=1)
        df["ichi_above_cloud"] = (df["close"] > cloud_top).astype(int)
        df["ichi_below_cloud"] = (df["close"] < cloud_bottom).astype(int)
        df["ichi_in_cloud"] = ((df["close"] >= cloud_bottom) & (df["close"] <= cloud_top)).astype(int)

        # TK Cross signal
        df["ichi_tk_cross_bull"] = (
            (df["ichi_tenkan"] > df["ichi_kijun"]) &
            (df["ichi_tenkan"].shift(1) <= df["ichi_kijun"].shift(1))
        ).astype(int)
        df["ichi_tk_cross_bear"] = (
            (df["ichi_tenkan"] < df["ichi_kijun"]) &
            (df["ichi_tenkan"].shift(1) >= df["ichi_kijun"].shift(1))
        ).astype(int)

        self._last_result = df
        return df

    def reset(self) -> None:
        super().reset()


class VWAPIndicator(BaseIndicator):
    """Volume Weighted Average Price — anchored to session start."""

    def __init__(self):
        super().__init__(name="vwap")

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()

        typical_price = (df["high"] + df["low"] + df["close"]) / 3.0
        tp_volume = typical_price * df["volume"]

        # Cumulative VWAP (anchored to start of data)
        cum_tp_vol = tp_volume.cumsum()
        cum_vol = df["volume"].cumsum()

        df["vwap"] = cum_tp_vol / cum_vol.replace(0, np.nan)
        df["vwap"] = df["vwap"].fillna(df["close"])

        # VWAP standard deviation bands
        vwap_var = ((typical_price - df["vwap"]) ** 2 * df["volume"]).cumsum() / cum_vol.replace(0, np.nan)
        vwap_std = np.sqrt(vwap_var.fillna(0))

        df["vwap_upper_1"] = df["vwap"] + vwap_std
        df["vwap_lower_1"] = df["vwap"] - vwap_std
        df["vwap_upper_2"] = df["vwap"] + 2 * vwap_std
        df["vwap_lower_2"] = df["vwap"] - 2 * vwap_std

        # Price relative to VWAP
        df["vwap_bias"] = np.where(df["close"] > df["vwap"], 1,
                          np.where(df["close"] < df["vwap"], -1, 0))

        # Distance from VWAP as percentage
        df["vwap_distance_pct"] = ((df["close"] - df["vwap"]) / df["vwap"].replace(0, np.nan)) * 100.0
        df["vwap_distance_pct"] = df["vwap_distance_pct"].fillna(0)

        self._last_result = df
        return df

    def reset(self) -> None:
        super().reset()