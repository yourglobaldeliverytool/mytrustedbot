"""
APEX SIGNAL™ — Trend Following Strategies (6)
EMA Crossover, SMA Crossover, Multi-TF Alignment, VWAP Trend Bias,
Supertrend Direction, Donchian Breakout
"""
import pandas as pd
from apex_signal.strategies.base import BaseStrategy, StrategySignal


class EMACrossoverStrategy(BaseStrategy):
    """EMA 8/20 crossover with 50 EMA trend filter."""

    def __init__(self):
        super().__init__(name="ema_crossover")

    def evaluate(self, df: pd.DataFrame, symbol: str = "") -> StrategySignal:
        if len(df) < 5 or "ema_8" not in df.columns:
            return self._hold("Insufficient data for EMA crossover")

        ema8 = self._safe_last(df["ema_8"])
        ema20 = self._safe_last(df["ema_20"])
        ema50 = self._safe_last(df["ema_50"])
        prev_ema8 = self._safe_prev(df["ema_8"])
        prev_ema20 = self._safe_prev(df["ema_20"])

        bullish_cross = ema8 > ema20 and prev_ema8 <= prev_ema20
        bearish_cross = ema8 < ema20 and prev_ema8 >= prev_ema20
        trend_up = ema20 > ema50
        trend_down = ema20 < ema50

        if bullish_cross and trend_up:
            return StrategySignal(self.name, "BUY", 75,
                "EMA 8 crossed above EMA 20 with uptrend confirmed by EMA 50",
                {"ema8": ema8, "ema20": ema20, "ema50": ema50})
        elif bullish_cross:
            return StrategySignal(self.name, "BUY", 55,
                "EMA 8 crossed above EMA 20 but trend filter neutral",
                {"ema8": ema8, "ema20": ema20, "ema50": ema50})
        elif bearish_cross and trend_down:
            return StrategySignal(self.name, "SELL", 75,
                "EMA 8 crossed below EMA 20 with downtrend confirmed by EMA 50",
                {"ema8": ema8, "ema20": ema20, "ema50": ema50})
        elif bearish_cross:
            return StrategySignal(self.name, "SELL", 55,
                "EMA 8 crossed below EMA 20 but trend filter neutral",
                {"ema8": ema8, "ema20": ema20, "ema50": ema50})

        # Ongoing trend strength
        if ema8 > ema20 > ema50:
            return StrategySignal(self.name, "BUY", 40,
                "EMAs aligned bullish (8 > 20 > 50)", {"alignment": "bullish"})
        elif ema8 < ema20 < ema50:
            return StrategySignal(self.name, "SELL", 40,
                "EMAs aligned bearish (8 < 20 < 50)", {"alignment": "bearish"})

        return self._hold("No EMA crossover signal")


class SMACrossoverStrategy(BaseStrategy):
    """SMA 20/50 crossover — golden/death cross detection."""

    def __init__(self):
        super().__init__(name="sma_crossover")

    def evaluate(self, df: pd.DataFrame, symbol: str = "") -> StrategySignal:
        if len(df) < 5 or "sma_20" not in df.columns:
            return self._hold("Insufficient data for SMA crossover")

        sma20 = self._safe_last(df["sma_20"])
        sma50 = self._safe_last(df["sma_50"])
        prev_sma20 = self._safe_prev(df["sma_20"])
        prev_sma50 = self._safe_prev(df["sma_50"])
        sma100 = self._safe_last(df["sma_100"]) if "sma_100" in df.columns else 0

        golden_cross = sma20 > sma50 and prev_sma20 <= prev_sma50
        death_cross = sma20 < sma50 and prev_sma20 >= prev_sma50

        if golden_cross:
            conf = 80 if sma100 and sma50 > sma100 else 65
            return StrategySignal(self.name, "BUY", conf,
                "Golden cross: SMA 20 crossed above SMA 50",
                {"sma20": sma20, "sma50": sma50})
        elif death_cross:
            conf = 80 if sma100 and sma50 < sma100 else 65
            return StrategySignal(self.name, "SELL", conf,
                "Death cross: SMA 20 crossed below SMA 50",
                {"sma20": sma20, "sma50": sma50})

        if sma20 > sma50:
            return StrategySignal(self.name, "BUY", 35,
                "SMA 20 above SMA 50 — bullish bias", {"spread": sma20 - sma50})
        elif sma20 < sma50:
            return StrategySignal(self.name, "SELL", 35,
                "SMA 20 below SMA 50 — bearish bias", {"spread": sma20 - sma50})

        return self._hold("SMAs converging, no clear direction")


class MultiTimeframeAlignmentStrategy(BaseStrategy):
    """Multi-timeframe trend alignment using EMA stack."""

    def __init__(self):
        super().__init__(name="multi_tf_alignment")

    def evaluate(self, df: pd.DataFrame, symbol: str = "") -> StrategySignal:
        required = ["ema_8", "ema_20", "ema_50", "ema_200"]
        if len(df) < 5 or not all(c in df.columns for c in required):
            return self._hold("Insufficient data for multi-TF alignment")

        ema8 = self._safe_last(df["ema_8"])
        ema20 = self._safe_last(df["ema_20"])
        ema50 = self._safe_last(df["ema_50"])
        ema200 = self._safe_last(df["ema_200"])
        close = self._safe_last(df["close"])

        # Perfect bullish alignment
        if close > ema8 > ema20 > ema50 > ema200:
            return StrategySignal(self.name, "BUY", 85,
                "Perfect bullish alignment: Price > EMA8 > EMA20 > EMA50 > EMA200",
                {"alignment": "perfect_bull"})
        # Perfect bearish alignment
        elif close < ema8 < ema20 < ema50 < ema200:
            return StrategySignal(self.name, "SELL", 85,
                "Perfect bearish alignment: Price < EMA8 < EMA20 < EMA50 < EMA200",
                {"alignment": "perfect_bear"})
        # Partial bullish
        elif ema8 > ema20 > ema50 and close > ema20:
            return StrategySignal(self.name, "BUY", 60,
                "Partial bullish alignment: EMA8 > EMA20 > EMA50, price above EMA20",
                {"alignment": "partial_bull"})
        # Partial bearish
        elif ema8 < ema20 < ema50 and close < ema20:
            return StrategySignal(self.name, "SELL", 60,
                "Partial bearish alignment: EMA8 < EMA20 < EMA50, price below EMA20",
                {"alignment": "partial_bear"})

        return self._hold("No clear multi-timeframe alignment")


class VWAPTrendBiasStrategy(BaseStrategy):
    """VWAP-based trend bias with distance analysis."""

    def __init__(self):
        super().__init__(name="vwap_trend_bias")

    def evaluate(self, df: pd.DataFrame, symbol: str = "") -> StrategySignal:
        if len(df) < 5 or "vwap" not in df.columns:
            return self._hold("Insufficient data for VWAP trend bias")

        close = self._safe_last(df["close"])
        vwap = self._safe_last(df["vwap"])
        vwap_dist = self._safe_last(df["vwap_distance_pct"])
        vwap_bias = self._safe_last(df["vwap_bias"])

        if close > vwap and vwap_dist > 0.3:
            conf = min(80, 50 + abs(vwap_dist) * 10)
            return StrategySignal(self.name, "BUY", conf,
                f"Price {vwap_dist:.2f}% above VWAP — bullish institutional bias",
                {"vwap": vwap, "distance_pct": vwap_dist})
        elif close < vwap and vwap_dist < -0.3:
            conf = min(80, 50 + abs(vwap_dist) * 10)
            return StrategySignal(self.name, "SELL", conf,
                f"Price {abs(vwap_dist):.2f}% below VWAP — bearish institutional bias",
                {"vwap": vwap, "distance_pct": vwap_dist})

        return self._hold("Price near VWAP, no clear bias")


class SupertrendDirectionStrategy(BaseStrategy):
    """Supertrend-like signal using ATR bands and daily range direction."""

    def __init__(self):
        super().__init__(name="supertrend_direction")

    def evaluate(self, df: pd.DataFrame, symbol: str = "") -> StrategySignal:
        if len(df) < 20 or "atr" not in df.columns:
            return self._hold("Insufficient data for supertrend direction")

        close = self._safe_last(df["close"])
        atr = self._safe_last(df["atr"])
        ema20 = self._safe_last(df["ema_20"]) if "ema_20" in df.columns else close

        # Supertrend bands
        upper_band = ema20 + (2.0 * atr)
        lower_band = ema20 - (2.0 * atr)

        # Daily range direction
        daily_high = df["high"].iloc[-20:].max()
        daily_low = df["low"].iloc[-20:].min()
        daily_mid = (daily_high + daily_low) / 2

        if close > ema20 and close > daily_mid:
            conf = 70 if close < upper_band else 55
            return StrategySignal(self.name, "BUY", conf,
                "Supertrend bullish: price above EMA20 and daily midpoint",
                {"upper_band": upper_band, "lower_band": lower_band})
        elif close < ema20 and close < daily_mid:
            conf = 70 if close > lower_band else 55
            return StrategySignal(self.name, "SELL", conf,
                "Supertrend bearish: price below EMA20 and daily midpoint",
                {"upper_band": upper_band, "lower_band": lower_band})

        return self._hold("Supertrend neutral")


class DonchianBreakoutStrategy(BaseStrategy):
    """Donchian Channel breakout with width confirmation."""

    def __init__(self):
        super().__init__(name="donchian_breakout")

    def evaluate(self, df: pd.DataFrame, symbol: str = "") -> StrategySignal:
        if len(df) < 5 or "dc_upper" not in df.columns:
            return self._hold("Insufficient data for Donchian breakout")

        dc_upper_break = self._safe_last(df["dc_upper_break"])
        dc_lower_break = self._safe_last(df["dc_lower_break"])
        dc_width_expanding = self._safe_last(df["dc_width_expanding"])
        dc_width = self._safe_last(df["dc_width"])

        if dc_upper_break:
            conf = 75 if dc_width_expanding else 55
            return StrategySignal(self.name, "BUY", conf,
                "Donchian upper channel breakout" +
                (" with expanding width" if dc_width_expanding else ""),
                {"dc_width": dc_width})
        elif dc_lower_break:
            conf = 75 if dc_width_expanding else 55
            return StrategySignal(self.name, "SELL", conf,
                "Donchian lower channel breakout" +
                (" with expanding width" if dc_width_expanding else ""),
                {"dc_width": dc_width})

        return self._hold("No Donchian breakout detected")