"""
APEX SIGNAL™ — Session-Aware Strategies (5)
US Open Momentum, London Open Momentum, Tokyo Open Range,
New York Close Rebalance, Session Overlap Surge
"""
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from apex_signal.strategies.base import BaseStrategy, StrategySignal


def _get_utc_hour() -> int:
    """Get current UTC hour."""
    return datetime.now(timezone.utc).hour


class USOpenMomentumStrategy(BaseStrategy):
    """US market open momentum — captures directional moves at 9:30 ET (13:30-15:00 UTC)."""

    def __init__(self):
        super().__init__(name="us_open_momentum")

    def evaluate(self, df: pd.DataFrame, symbol: str = "") -> StrategySignal:
        if len(df) < 10:
            return self._hold("Insufficient data for US open momentum")

        hour = _get_utc_hour()
        # US open window: 13:30-15:00 UTC (9:30-11:00 ET)
        in_us_open = 13 <= hour <= 15

        if not in_us_open:
            return self._hold("Outside US open window (13:30-15:00 UTC)")

        # Analyze first candles of the session
        recent = df.iloc[-5:]
        direction = recent["close"].iloc[-1] - recent["open"].iloc[0]
        range_size = recent["high"].max() - recent["low"].min()
        avg_range = (df["high"] - df["low"]).iloc[-20:].mean()

        rvol = self._safe_last(df["rvol"]) if "rvol" in df.columns else 1.0

        if range_size > avg_range * 1.3 and rvol > 1.2:
            if direction > 0:
                conf = min(80, 55 + rvol * 10)
                return StrategySignal(self.name, "BUY", conf,
                    f"US open bullish momentum: expanded range with {rvol:.1f}x volume",
                    {"session": "US_open", "rvol": rvol, "range_expansion": range_size / avg_range})
            else:
                conf = min(80, 55 + rvol * 10)
                return StrategySignal(self.name, "SELL", conf,
                    f"US open bearish momentum: expanded range with {rvol:.1f}x volume",
                    {"session": "US_open", "rvol": rvol, "range_expansion": range_size / avg_range})

        return self._hold("US open — no significant momentum detected")


class LondonOpenMomentumStrategy(BaseStrategy):
    """London market open momentum — captures moves at 8:00 GMT (08:00-10:00 UTC)."""

    def __init__(self):
        super().__init__(name="london_open_momentum")

    def evaluate(self, df: pd.DataFrame, symbol: str = "") -> StrategySignal:
        if len(df) < 10:
            return self._hold("Insufficient data for London open momentum")

        hour = _get_utc_hour()
        in_london_open = 7 <= hour <= 10

        if not in_london_open:
            return self._hold("Outside London open window (07:00-10:00 UTC)")

        recent = df.iloc[-5:]
        direction = recent["close"].iloc[-1] - recent["open"].iloc[0]
        range_size = recent["high"].max() - recent["low"].min()
        avg_range = (df["high"] - df["low"]).iloc[-20:].mean()

        gap_pct = self._safe_last(df["gap_pct"]) if "gap_pct" in df.columns else 0

        if range_size > avg_range * 1.2:
            if direction > 0:
                conf = 65 if abs(gap_pct) > 0.3 else 55
                return StrategySignal(self.name, "BUY", conf,
                    f"London open bullish momentum with range expansion" +
                    (f" and gap up ({gap_pct:.2f}%)" if gap_pct > 0.3 else ""),
                    {"session": "London_open", "gap_pct": gap_pct})
            else:
                conf = 65 if abs(gap_pct) > 0.3 else 55
                return StrategySignal(self.name, "SELL", conf,
                    f"London open bearish momentum with range expansion" +
                    (f" and gap down ({gap_pct:.2f}%)" if gap_pct < -0.3 else ""),
                    {"session": "London_open", "gap_pct": gap_pct})

        return self._hold("London open — no significant momentum detected")


class TokyoOpenRangeStrategy(BaseStrategy):
    """Tokyo session open range — captures Asian session range breakouts (00:00-03:00 UTC)."""

    def __init__(self):
        super().__init__(name="tokyo_open_range")

    def evaluate(self, df: pd.DataFrame, symbol: str = "") -> StrategySignal:
        if len(df) < 10:
            return self._hold("Insufficient data for Tokyo open range")

        hour = _get_utc_hour()
        # Tokyo session: 00:00-06:00 UTC, breakout window: 03:00-06:00
        in_tokyo_breakout = 3 <= hour <= 6

        if not in_tokyo_breakout:
            return self._hold("Outside Tokyo breakout window (03:00-06:00 UTC)")

        # Use ORB indicators if available
        orb_break_up = self._safe_last(df["orb_break_up"]) if "orb_break_up" in df.columns else 0
        orb_break_down = self._safe_last(df["orb_break_down"]) if "orb_break_down" in df.columns else 0
        orb_range_pct = self._safe_last(df["orb_range_pct"]) if "orb_range_pct" in df.columns else 0

        if orb_break_up:
            return StrategySignal(self.name, "BUY", 65,
                f"Tokyo session ORB breakout to upside (range: {orb_range_pct:.2f}%)",
                {"session": "Tokyo", "orb_range_pct": orb_range_pct})
        elif orb_break_down:
            return StrategySignal(self.name, "SELL", 65,
                f"Tokyo session ORB breakout to downside (range: {orb_range_pct:.2f}%)",
                {"session": "Tokyo", "orb_range_pct": orb_range_pct})

        return self._hold("Tokyo session — no ORB breakout detected")


class NYCloseRebalanceStrategy(BaseStrategy):
    """New York close rebalance — captures end-of-day positioning shifts (20:00-22:00 UTC)."""

    def __init__(self):
        super().__init__(name="ny_close_rebalance")

    def evaluate(self, df: pd.DataFrame, symbol: str = "") -> StrategySignal:
        if len(df) < 15:
            return self._hold("Insufficient data for NY close rebalance")

        hour = _get_utc_hour()
        in_ny_close = 20 <= hour <= 22

        if not in_ny_close:
            return self._hold("Outside NY close window (20:00-22:00 UTC)")

        close = self._safe_last(df["close"])
        vwap = self._safe_last(df["vwap"]) if "vwap" in df.columns else close
        ema20 = self._safe_last(df["ema_20"]) if "ema_20" in df.columns else close

        # End-of-day rebalance: price reverting toward VWAP
        vwap_dist = ((close - vwap) / vwap) * 100 if vwap > 0 else 0

        # Overextended above VWAP at close — expect rebalance down
        if vwap_dist > 0.5:
            return StrategySignal(self.name, "SELL", 60,
                f"NY close rebalance: price {vwap_dist:.2f}% above VWAP — expect mean reversion",
                {"session": "NY_close", "vwap_dist": vwap_dist})
        # Overextended below VWAP at close — expect rebalance up
        elif vwap_dist < -0.5:
            return StrategySignal(self.name, "BUY", 60,
                f"NY close rebalance: price {abs(vwap_dist):.2f}% below VWAP — expect mean reversion",
                {"session": "NY_close", "vwap_dist": vwap_dist})

        return self._hold("NY close — price near VWAP, no rebalance signal")


class SessionOverlapSurgeStrategy(BaseStrategy):
    """Session overlap surge — captures volatility during London/NY overlap (13:00-17:00 UTC)."""

    def __init__(self):
        super().__init__(name="session_overlap_surge")

    def evaluate(self, df: pd.DataFrame, symbol: str = "") -> StrategySignal:
        if len(df) < 10:
            return self._hold("Insufficient data for session overlap")

        hour = _get_utc_hour()
        # London/NY overlap: 13:00-17:00 UTC
        in_overlap = 13 <= hour <= 17

        if not in_overlap:
            return self._hold("Outside London/NY overlap window (13:00-17:00 UTC)")

        rvol = self._safe_last(df["rvol"]) if "rvol" in df.columns else 1.0
        atr_pct = self._safe_last(df["atr_pct"]) if "atr_pct" in df.columns else 0
        adx = self._safe_last(df["adx"]) if "adx" in df.columns else 0

        close = self._safe_last(df["close"])
        ema8 = self._safe_last(df["ema_8"]) if "ema_8" in df.columns else close
        ema20 = self._safe_last(df["ema_20"]) if "ema_20" in df.columns else close

        # High activity overlap with trend
        if rvol > 1.5 and adx > 25:
            if close > ema8 > ema20:
                conf = min(80, 55 + rvol * 8)
                return StrategySignal(self.name, "BUY", conf,
                    f"Session overlap surge: {rvol:.1f}x volume, ADX={adx:.0f}, bullish trend",
                    {"session": "overlap", "rvol": rvol, "adx": adx})
            elif close < ema8 < ema20:
                conf = min(80, 55 + rvol * 8)
                return StrategySignal(self.name, "SELL", conf,
                    f"Session overlap surge: {rvol:.1f}x volume, ADX={adx:.0f}, bearish trend",
                    {"session": "overlap", "rvol": rvol, "adx": adx})

        # Moderate overlap activity
        if rvol > 1.2:
            signal = "BUY" if close > ema20 else "SELL"
            return StrategySignal(self.name, signal, 45,
                f"Session overlap with elevated volume ({rvol:.1f}x)",
                {"session": "overlap", "rvol": rvol})

        return self._hold("Session overlap — no significant surge detected")