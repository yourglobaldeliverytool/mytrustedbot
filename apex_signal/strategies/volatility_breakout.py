"""
APEX SIGNAL™ — Volatility / Breakout Strategies (5)
ATR Volatility Breakout, Range Expansion, Breakout + Retest,
Keltner Adaptive Breakout, Channel Width Spike
"""
import pandas as pd
import numpy as np
from apex_signal.strategies.base import BaseStrategy, StrategySignal


class ATRVolatilityBreakoutStrategy(BaseStrategy):
    """ATR-based volatility breakout — detects explosive moves beyond normal range."""

    def __init__(self):
        super().__init__(name="atr_volatility_breakout")

    def evaluate(self, df: pd.DataFrame, symbol: str = "") -> StrategySignal:
        if len(df) < 20 or "atr" not in df.columns:
            return self._hold("Insufficient data for ATR breakout")

        close = self._safe_last(df["close"])
        prev_close = self._safe_prev(df["close"])
        atr = self._safe_last(df["atr"])
        atr_pct = self._safe_last(df["atr_pct"])

        move = close - prev_close
        move_in_atr = abs(move) / atr if atr > 0 else 0

        # Breakout: move exceeds 1.5x ATR
        if move_in_atr > 1.5:
            if move > 0:
                conf = min(85, 60 + move_in_atr * 10)
                return StrategySignal(self.name, "BUY", conf,
                    f"Bullish ATR breakout: move of {move_in_atr:.1f}x ATR",
                    {"move_atr": move_in_atr, "atr": atr, "atr_pct": atr_pct})
            else:
                conf = min(85, 60 + move_in_atr * 10)
                return StrategySignal(self.name, "SELL", conf,
                    f"Bearish ATR breakout: move of {move_in_atr:.1f}x ATR",
                    {"move_atr": move_in_atr, "atr": atr, "atr_pct": atr_pct})

        # Moderate expansion
        if move_in_atr > 1.0:
            signal = "BUY" if move > 0 else "SELL"
            return StrategySignal(self.name, signal, 50,
                f"Moderate ATR expansion: {move_in_atr:.1f}x ATR move",
                {"move_atr": move_in_atr})

        return self._hold(f"Move within normal ATR range ({move_in_atr:.1f}x)")


class RangeExpansionStrategy(BaseStrategy):
    """Range expansion — detects when current candle range significantly exceeds recent average."""

    def __init__(self):
        super().__init__(name="range_expansion")

    def evaluate(self, df: pd.DataFrame, symbol: str = "") -> StrategySignal:
        if len(df) < 20:
            return self._hold("Insufficient data for range expansion")

        current_range = df["high"].iloc[-1] - df["low"].iloc[-1]
        avg_range = (df["high"] - df["low"]).iloc[-20:-1].mean()

        if avg_range == 0:
            return self._hold("Zero average range")

        expansion_ratio = current_range / avg_range
        close = self._safe_last(df["close"])
        open_price = df["open"].iloc[-1]
        bullish_candle = close > open_price

        if expansion_ratio > 2.0:
            signal = "BUY" if bullish_candle else "SELL"
            conf = min(85, 55 + expansion_ratio * 10)
            return StrategySignal(self.name, signal, conf,
                f"Strong range expansion ({expansion_ratio:.1f}x average) — "
                f"{'bullish' if bullish_candle else 'bearish'} candle",
                {"expansion_ratio": expansion_ratio})
        elif expansion_ratio > 1.5:
            signal = "BUY" if bullish_candle else "SELL"
            return StrategySignal(self.name, signal, 50,
                f"Moderate range expansion ({expansion_ratio:.1f}x average)",
                {"expansion_ratio": expansion_ratio})

        return self._hold(f"Range expansion ratio: {expansion_ratio:.1f}x — normal")


class BreakoutRetestStrategy(BaseStrategy):
    """Breakout + Retest — detects breakout from recent high/low followed by retest."""

    def __init__(self):
        super().__init__(name="breakout_retest")

    def evaluate(self, df: pd.DataFrame, symbol: str = "") -> StrategySignal:
        if len(df) < 30:
            return self._hold("Insufficient data for breakout retest")

        close = self._safe_last(df["close"])
        lookback = df.iloc[-30:-5]

        if lookback.empty:
            return self._hold("Insufficient lookback data")

        resistance = lookback["high"].max()
        support = lookback["low"].min()

        recent = df.iloc[-5:]
        broke_above = (recent["close"] > resistance).any()
        broke_below = (recent["close"] < support).any()

        # Check for retest (price came back near the level)
        near_resistance = abs(close - resistance) / resistance < 0.005 if resistance > 0 else False
        near_support = abs(close - support) / support < 0.005 if support > 0 else False

        if broke_above and near_resistance and close >= resistance:
            return StrategySignal(self.name, "BUY", 75,
                f"Bullish breakout + retest: broke above {resistance:.2f} and retesting as support",
                {"resistance": resistance, "support": support})
        elif broke_below and near_support and close <= support:
            return StrategySignal(self.name, "SELL", 75,
                f"Bearish breakout + retest: broke below {support:.2f} and retesting as resistance",
                {"resistance": resistance, "support": support})
        elif close > resistance:
            return StrategySignal(self.name, "BUY", 55,
                f"Price above recent resistance {resistance:.2f}",
                {"resistance": resistance})
        elif close < support:
            return StrategySignal(self.name, "SELL", 55,
                f"Price below recent support {support:.2f}",
                {"support": support})

        return self._hold("No breakout-retest pattern detected")


class KeltnerAdaptiveBreakoutStrategy(BaseStrategy):
    """Keltner Channel adaptive breakout with squeeze detection."""

    def __init__(self):
        super().__init__(name="keltner_adaptive_breakout")

    def evaluate(self, df: pd.DataFrame, symbol: str = "") -> StrategySignal:
        if len(df) < 5 or "kc_upper" not in df.columns:
            return self._hold("Insufficient data for Keltner breakout")

        kc_above = self._safe_last(df["kc_above"])
        kc_below = self._safe_last(df["kc_below"])
        kc_pct = self._safe_last(df["kc_pct"])
        bb_squeeze = self._safe_last(df["bb_squeeze"]) if "bb_squeeze" in df.columns else 0

        # Breakout from squeeze is highest conviction
        if kc_above:
            conf = 80 if bb_squeeze else 65
            return StrategySignal(self.name, "BUY", conf,
                "Keltner upper channel breakout" +
                (" from Bollinger squeeze — high conviction" if bb_squeeze else ""),
                {"kc_pct": kc_pct, "squeeze": bool(bb_squeeze)})
        elif kc_below:
            conf = 80 if bb_squeeze else 65
            return StrategySignal(self.name, "SELL", conf,
                "Keltner lower channel breakout" +
                (" from Bollinger squeeze — high conviction" if bb_squeeze else ""),
                {"kc_pct": kc_pct, "squeeze": bool(bb_squeeze)})

        # Near channel edges
        if kc_pct > 0.9:
            return StrategySignal(self.name, "BUY", 45,
                "Price near Keltner upper channel — potential breakout",
                {"kc_pct": kc_pct})
        elif kc_pct < 0.1:
            return StrategySignal(self.name, "SELL", 45,
                "Price near Keltner lower channel — potential breakdown",
                {"kc_pct": kc_pct})

        return self._hold("Price within Keltner channels")


class ChannelWidthSpikeStrategy(BaseStrategy):
    """Channel width spike — detects sudden volatility expansion across multiple channels."""

    def __init__(self):
        super().__init__(name="channel_width_spike")

    def evaluate(self, df: pd.DataFrame, symbol: str = "") -> StrategySignal:
        if len(df) < 20:
            return self._hold("Insufficient data for channel width spike")

        signals = []
        total_conf = 0

        # Bollinger bandwidth spike
        if "bb_bandwidth" in df.columns:
            bw = self._safe_last(df["bb_bandwidth"])
            bw_avg = df["bb_bandwidth"].iloc[-20:].mean()
            if bw_avg > 0 and bw > bw_avg * 1.5:
                signals.append("Bollinger bandwidth spike")
                total_conf += 25

        # Donchian width expansion
        if "dc_width" in df.columns:
            dw = self._safe_last(df["dc_width"])
            dw_avg = df["dc_width"].iloc[-20:].mean()
            if dw_avg > 0 and dw > dw_avg * 1.5:
                signals.append("Donchian width expansion")
                total_conf += 25

        # ATR spike
        if "atr_pct" in df.columns:
            atr_pct = self._safe_last(df["atr_pct"])
            atr_avg = df["atr_pct"].iloc[-20:].mean()
            if atr_avg > 0 and atr_pct > atr_avg * 1.5:
                signals.append("ATR spike")
                total_conf += 25

        if not signals:
            return self._hold("No channel width spike detected")

        # Determine direction from recent price action
        close = self._safe_last(df["close"])
        prev_close = self._safe_prev(df["close"], offset=3)
        direction = "BUY" if close > prev_close else "SELL"

        return StrategySignal(self.name, direction, min(85, total_conf + 15),
            f"Channel width spike detected: {', '.join(signals)} — volatility expanding {direction.lower()}",
            {"components": signals})