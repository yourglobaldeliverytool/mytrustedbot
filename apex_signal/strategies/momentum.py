"""
APEX SIGNAL™ — Momentum Strategies (5)
RSI Overbought/Oversold, MACD Trend Momentum, Stochastic Scalping,
OBV Momentum Confirmation, Chaikin Oscillator Breakout
"""
import pandas as pd
from apex_signal.strategies.base import BaseStrategy, StrategySignal


class RSIOverboughtOversoldStrategy(BaseStrategy):
    """RSI-based overbought/oversold reversal strategy with trend context."""

    def __init__(self):
        super().__init__(name="rsi_ob_os")

    def evaluate(self, df: pd.DataFrame, symbol: str = "") -> StrategySignal:
        if len(df) < 5 or "rsi" not in df.columns:
            return self._hold("Insufficient data for RSI strategy")

        rsi = self._safe_last(df["rsi"])
        prev_rsi = self._safe_prev(df["rsi"])
        close = self._safe_last(df["close"])
        ema50 = self._safe_last(df["ema_50"]) if "ema_50" in df.columns else close

        # Oversold bounce
        if rsi < 30 and prev_rsi < rsi:
            conf = 75 if close > ema50 else 60
            return StrategySignal(self.name, "BUY", conf,
                f"RSI oversold at {rsi:.1f} and turning up — potential reversal",
                {"rsi": rsi, "prev_rsi": prev_rsi})
        # Deep oversold
        elif rsi < 20:
            return StrategySignal(self.name, "BUY", 70,
                f"RSI deeply oversold at {rsi:.1f} — extreme reversal zone",
                {"rsi": rsi})
        # Overbought reversal
        elif rsi > 70 and prev_rsi > rsi:
            conf = 75 if close < ema50 else 60
            return StrategySignal(self.name, "SELL", conf,
                f"RSI overbought at {rsi:.1f} and turning down — potential reversal",
                {"rsi": rsi, "prev_rsi": prev_rsi})
        # Deep overbought
        elif rsi > 80:
            return StrategySignal(self.name, "SELL", 70,
                f"RSI deeply overbought at {rsi:.1f} — extreme reversal zone",
                {"rsi": rsi})

        return self._hold(f"RSI at {rsi:.1f} — no extreme condition")


class MACDTrendMomentumStrategy(BaseStrategy):
    """MACD crossover with histogram momentum confirmation."""

    def __init__(self):
        super().__init__(name="macd_trend_momentum")

    def evaluate(self, df: pd.DataFrame, symbol: str = "") -> StrategySignal:
        if len(df) < 5 or "macd_line" not in df.columns:
            return self._hold("Insufficient data for MACD strategy")

        macd = self._safe_last(df["macd_line"])
        signal = self._safe_last(df["macd_signal"])
        histogram = self._safe_last(df["macd_histogram"])
        bull_cross = self._safe_last(df["macd_cross_bull"])
        bear_cross = self._safe_last(df["macd_cross_bear"])
        hist_rising = self._safe_last(df["macd_hist_rising"])
        above_zero = self._safe_last(df["macd_above_zero"])

        if bull_cross:
            conf = 80 if above_zero else 65
            return StrategySignal(self.name, "BUY", conf,
                "MACD bullish crossover" + (" above zero line" if above_zero else " below zero line"),
                {"macd": macd, "signal": signal, "histogram": histogram})
        elif bear_cross:
            conf = 80 if not above_zero else 65
            return StrategySignal(self.name, "SELL", conf,
                "MACD bearish crossover" + (" below zero line" if not above_zero else " above zero line"),
                {"macd": macd, "signal": signal, "histogram": histogram})

        # Histogram momentum
        if macd > signal and hist_rising and above_zero:
            return StrategySignal(self.name, "BUY", 55,
                "MACD bullish with rising histogram above zero",
                {"histogram": histogram})
        elif macd < signal and not hist_rising and not above_zero:
            return StrategySignal(self.name, "SELL", 55,
                "MACD bearish with falling histogram below zero",
                {"histogram": histogram})

        return self._hold("No clear MACD momentum signal")


class StochasticScalpingStrategy(BaseStrategy):
    """Stochastic oscillator scalping with K/D crossovers in extreme zones."""

    def __init__(self):
        super().__init__(name="stochastic_scalping")

    def evaluate(self, df: pd.DataFrame, symbol: str = "") -> StrategySignal:
        if len(df) < 5 or "stoch_k" not in df.columns:
            return self._hold("Insufficient data for Stochastic strategy")

        k = self._safe_last(df["stoch_k"])
        d = self._safe_last(df["stoch_d"])
        prev_k = self._safe_prev(df["stoch_k"])
        prev_d = self._safe_prev(df["stoch_d"])

        # Bullish crossover in oversold zone
        k_cross_above_d = k > d and prev_k <= prev_d
        k_cross_below_d = k < d and prev_k >= prev_d

        if k_cross_above_d and k < 30:
            return StrategySignal(self.name, "BUY", 80,
                f"Stochastic bullish crossover in oversold zone (K={k:.1f}, D={d:.1f})",
                {"stoch_k": k, "stoch_d": d})
        elif k_cross_above_d and k < 50:
            return StrategySignal(self.name, "BUY", 60,
                f"Stochastic bullish crossover (K={k:.1f}, D={d:.1f})",
                {"stoch_k": k, "stoch_d": d})
        elif k_cross_below_d and k > 70:
            return StrategySignal(self.name, "SELL", 80,
                f"Stochastic bearish crossover in overbought zone (K={k:.1f}, D={d:.1f})",
                {"stoch_k": k, "stoch_d": d})
        elif k_cross_below_d and k > 50:
            return StrategySignal(self.name, "SELL", 60,
                f"Stochastic bearish crossover (K={k:.1f}, D={d:.1f})",
                {"stoch_k": k, "stoch_d": d})

        return self._hold(f"Stochastic K={k:.1f}, D={d:.1f} — no crossover in extreme zone")


class OBVMomentumStrategy(BaseStrategy):
    """OBV momentum confirmation — volume validates price direction."""

    def __init__(self):
        super().__init__(name="obv_momentum")

    def evaluate(self, df: pd.DataFrame, symbol: str = "") -> StrategySignal:
        if len(df) < 10 or "obv" not in df.columns:
            return self._hold("Insufficient data for OBV strategy")

        obv = self._safe_last(df["obv"])
        obv_ema = self._safe_last(df["obv_ema"])
        prev_obv = self._safe_prev(df["obv"], offset=3)
        close = self._safe_last(df["close"])
        prev_close = self._safe_prev(df["close"], offset=3)

        obv_rising = obv > prev_obv
        obv_above_ema = obv > obv_ema
        price_rising = close > prev_close

        # Bullish confirmation: price up + OBV up + OBV above EMA
        if price_rising and obv_rising and obv_above_ema:
            return StrategySignal(self.name, "BUY", 70,
                "OBV confirms bullish momentum: volume supports price advance",
                {"obv": obv, "obv_ema": obv_ema})
        # Bearish confirmation
        elif not price_rising and not obv_rising and not obv_above_ema:
            return StrategySignal(self.name, "SELL", 70,
                "OBV confirms bearish momentum: volume supports price decline",
                {"obv": obv, "obv_ema": obv_ema})
        # Bullish divergence: price down but OBV up
        elif not price_rising and obv_rising:
            return StrategySignal(self.name, "BUY", 55,
                "Bullish OBV divergence: volume accumulating despite price decline",
                {"divergence": "bullish"})
        # Bearish divergence: price up but OBV down
        elif price_rising and not obv_rising:
            return StrategySignal(self.name, "SELL", 55,
                "Bearish OBV divergence: volume declining despite price advance",
                {"divergence": "bearish"})

        return self._hold("OBV neutral")


class ChaikinOscillatorBreakoutStrategy(BaseStrategy):
    """Chaikin Money Flow breakout — institutional buying/selling pressure."""

    def __init__(self):
        super().__init__(name="chaikin_breakout")

    def evaluate(self, df: pd.DataFrame, symbol: str = "") -> StrategySignal:
        if len(df) < 5 or "cmf" not in df.columns:
            return self._hold("Insufficient data for Chaikin strategy")

        cmf = self._safe_last(df["cmf"])
        prev_cmf = self._safe_prev(df["cmf"])
        rvol = self._safe_last(df["rvol"]) if "rvol" in df.columns else 1.0

        # Strong buying pressure
        if cmf > 0.15 and cmf > prev_cmf:
            conf = 75 if rvol > 1.5 else 60
            return StrategySignal(self.name, "BUY", conf,
                f"Strong Chaikin buying pressure (CMF={cmf:.3f})" +
                (f" with high relative volume ({rvol:.1f}x)" if rvol > 1.5 else ""),
                {"cmf": cmf, "rvol": rvol})
        # Strong selling pressure
        elif cmf < -0.15 and cmf < prev_cmf:
            conf = 75 if rvol > 1.5 else 60
            return StrategySignal(self.name, "SELL", conf,
                f"Strong Chaikin selling pressure (CMF={cmf:.3f})" +
                (f" with high relative volume ({rvol:.1f}x)" if rvol > 1.5 else ""),
                {"cmf": cmf, "rvol": rvol})
        # Zero-line crossover
        elif cmf > 0 and prev_cmf <= 0:
            return StrategySignal(self.name, "BUY", 50,
                "CMF crossed above zero — buying pressure emerging",
                {"cmf": cmf})
        elif cmf < 0 and prev_cmf >= 0:
            return StrategySignal(self.name, "SELL", 50,
                "CMF crossed below zero — selling pressure emerging",
                {"cmf": cmf})

        return self._hold(f"CMF at {cmf:.3f} — no breakout")