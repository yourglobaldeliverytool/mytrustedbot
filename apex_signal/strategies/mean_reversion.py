"""
APEX SIGNAL™ — Mean Reversion Strategies (4)
Bollinger Mean Reversion, Z-Score Mean Reversion,
Donchian Mean Reversion, VWAP Pullback Reversion
"""
import pandas as pd
from apex_signal.strategies.base import BaseStrategy, StrategySignal


class BollingerMeanReversionStrategy(BaseStrategy):
    """Bollinger Bands mean reversion — fade moves to band extremes."""

    def __init__(self):
        super().__init__(name="bollinger_mean_reversion")

    def evaluate(self, df: pd.DataFrame, symbol: str = "") -> StrategySignal:
        if len(df) < 5 or "bb_pct_b" not in df.columns:
            return self._hold("Insufficient data for Bollinger mean reversion")

        pct_b = self._safe_last(df["bb_pct_b"])
        prev_pct_b = self._safe_prev(df["bb_pct_b"])
        close = self._safe_last(df["close"])
        bb_lower = self._safe_last(df["bb_lower"])
        bb_upper = self._safe_last(df["bb_upper"])
        bb_middle = self._safe_last(df["bb_middle"])
        rsi = self._safe_last(df["rsi"]) if "rsi" in df.columns else 50

        # Price touched lower band and bouncing (oversold reversion)
        if pct_b < 0.05 and pct_b > prev_pct_b:
            conf = 80 if rsi < 30 else 65
            return StrategySignal(self.name, "BUY", conf,
                f"Bollinger lower band touch with bounce (%B={pct_b:.2f})" +
                (f", RSI oversold at {rsi:.0f}" if rsi < 30 else ""),
                {"pct_b": pct_b, "bb_lower": bb_lower, "target": bb_middle})

        # Price touched upper band and reversing (overbought reversion)
        elif pct_b > 0.95 and pct_b < prev_pct_b:
            conf = 80 if rsi > 70 else 65
            return StrategySignal(self.name, "SELL", conf,
                f"Bollinger upper band touch with reversal (%B={pct_b:.2f})" +
                (f", RSI overbought at {rsi:.0f}" if rsi > 70 else ""),
                {"pct_b": pct_b, "bb_upper": bb_upper, "target": bb_middle})

        # Near lower band
        elif pct_b < 0.15:
            return StrategySignal(self.name, "BUY", 45,
                f"Price near Bollinger lower band (%B={pct_b:.2f})",
                {"pct_b": pct_b})
        # Near upper band
        elif pct_b > 0.85:
            return StrategySignal(self.name, "SELL", 45,
                f"Price near Bollinger upper band (%B={pct_b:.2f})",
                {"pct_b": pct_b})

        return self._hold(f"Bollinger %B at {pct_b:.2f} — within normal range")


class ZScoreMeanReversionStrategy(BaseStrategy):
    """Z-Score mean reversion — trade extreme statistical deviations back to mean."""

    def __init__(self):
        super().__init__(name="zscore_mean_reversion")

    def evaluate(self, df: pd.DataFrame, symbol: str = "") -> StrategySignal:
        if len(df) < 5 or "zscore" not in df.columns:
            return self._hold("Insufficient data for Z-Score mean reversion")

        zscore = self._safe_last(df["zscore"])
        prev_zscore = self._safe_prev(df["zscore"])
        reverting = self._safe_last(df["zscore_reverting"])
        vol_zscore = self._safe_last(df["zscore_volume"])

        # Extreme low Z-Score reverting
        if zscore < -2.0 and abs(zscore) < abs(prev_zscore):
            conf = 80 if vol_zscore > 1.0 else 65
            return StrategySignal(self.name, "BUY", conf,
                f"Z-Score extreme low ({zscore:.2f}) and reverting to mean" +
                (f" with volume confirmation" if vol_zscore > 1.0 else ""),
                {"zscore": zscore, "vol_zscore": vol_zscore})

        # Extreme high Z-Score reverting
        elif zscore > 2.0 and abs(zscore) < abs(prev_zscore):
            conf = 80 if vol_zscore > 1.0 else 65
            return StrategySignal(self.name, "SELL", conf,
                f"Z-Score extreme high ({zscore:.2f}) and reverting to mean" +
                (f" with volume confirmation" if vol_zscore > 1.0 else ""),
                {"zscore": zscore, "vol_zscore": vol_zscore})

        # Approaching extremes
        elif zscore < -1.5:
            return StrategySignal(self.name, "BUY", 50,
                f"Z-Score approaching extreme low ({zscore:.2f})",
                {"zscore": zscore})
        elif zscore > 1.5:
            return StrategySignal(self.name, "SELL", 50,
                f"Z-Score approaching extreme high ({zscore:.2f})",
                {"zscore": zscore})

        return self._hold(f"Z-Score at {zscore:.2f} — within normal range")


class DonchianMeanReversionStrategy(BaseStrategy):
    """Donchian channel mean reversion — fade moves to channel extremes."""

    def __init__(self):
        super().__init__(name="donchian_mean_reversion")

    def evaluate(self, df: pd.DataFrame, symbol: str = "") -> StrategySignal:
        if len(df) < 5 or "dc_position" not in df.columns:
            return self._hold("Insufficient data for Donchian mean reversion")

        dc_pos = self._safe_last(df["dc_position"])
        prev_dc_pos = self._safe_prev(df["dc_position"])
        dc_middle = self._safe_last(df["dc_middle"])
        close = self._safe_last(df["close"])

        # At channel bottom and bouncing
        if dc_pos < 0.1 and dc_pos > prev_dc_pos:
            return StrategySignal(self.name, "BUY", 70,
                f"Donchian channel bottom bounce (position={dc_pos:.2f}), target midline",
                {"dc_position": dc_pos, "target": dc_middle})

        # At channel top and reversing
        elif dc_pos > 0.9 and dc_pos < prev_dc_pos:
            return StrategySignal(self.name, "SELL", 70,
                f"Donchian channel top reversal (position={dc_pos:.2f}), target midline",
                {"dc_position": dc_pos, "target": dc_middle})

        # Near extremes
        elif dc_pos < 0.15:
            return StrategySignal(self.name, "BUY", 45,
                f"Near Donchian channel bottom (position={dc_pos:.2f})",
                {"dc_position": dc_pos})
        elif dc_pos > 0.85:
            return StrategySignal(self.name, "SELL", 45,
                f"Near Donchian channel top (position={dc_pos:.2f})",
                {"dc_position": dc_pos})

        return self._hold(f"Donchian position at {dc_pos:.2f} — mid-channel")


class VWAPPullbackReversionStrategy(BaseStrategy):
    """VWAP pullback reversion — trade pullbacks to VWAP in trending markets."""

    def __init__(self):
        super().__init__(name="vwap_pullback_reversion")

    def evaluate(self, df: pd.DataFrame, symbol: str = "") -> StrategySignal:
        if len(df) < 10 or "vwap" not in df.columns:
            return self._hold("Insufficient data for VWAP pullback")

        close = self._safe_last(df["close"])
        vwap = self._safe_last(df["vwap"])
        vwap_dist = self._safe_last(df["vwap_distance_pct"])
        ema20 = self._safe_last(df["ema_20"]) if "ema_20" in df.columns else close

        # Determine trend from EMA
        uptrend = close > ema20 and ema20 > self._safe_prev(df["ema_20"], offset=5) if "ema_20" in df.columns else False
        downtrend = close < ema20 and ema20 < self._safe_prev(df["ema_20"], offset=5) if "ema_20" in df.columns else False

        # Bullish: uptrend + pullback to VWAP
        if uptrend and abs(vwap_dist) < 0.2 and close >= vwap:
            return StrategySignal(self.name, "BUY", 75,
                "VWAP pullback in uptrend — price retesting VWAP as support",
                {"vwap": vwap, "distance_pct": vwap_dist})

        # Bearish: downtrend + rally to VWAP
        elif downtrend and abs(vwap_dist) < 0.2 and close <= vwap:
            return StrategySignal(self.name, "SELL", 75,
                "VWAP pullback in downtrend — price retesting VWAP as resistance",
                {"vwap": vwap, "distance_pct": vwap_dist})

        # Overextended from VWAP in trend
        elif uptrend and vwap_dist > 1.0:
            return StrategySignal(self.name, "HOLD", 30,
                f"Price overextended {vwap_dist:.1f}% above VWAP — wait for pullback",
                {"vwap_dist": vwap_dist})
        elif downtrend and vwap_dist < -1.0:
            return StrategySignal(self.name, "HOLD", 30,
                f"Price overextended {abs(vwap_dist):.1f}% below VWAP — wait for pullback",
                {"vwap_dist": vwap_dist})

        return self._hold("No VWAP pullback setup detected")