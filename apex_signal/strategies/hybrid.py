"""
APEX SIGNAL™ — Hybrid & Confluence Strategies (4)
Hybrid Confluence Score, Smart-Money Enhanced Trend,
Momentum + Structural Filter, Volatility Surprise + Trend Filter
"""
import pandas as pd
import numpy as np
from apex_signal.strategies.base import BaseStrategy, StrategySignal


class HybridConfluenceStrategy(BaseStrategy):
    """
    Hybrid confluence score — combines multiple indicator families
    into a single weighted confluence signal.
    """

    def __init__(self):
        super().__init__(name="hybrid_confluence")

    def evaluate(self, df: pd.DataFrame, symbol: str = "") -> StrategySignal:
        if len(df) < 20:
            return self._hold("Insufficient data for hybrid confluence")

        scores = []
        reasons = []

        # 1. Trend score (EMA alignment)
        if "ema_8" in df.columns and "ema_20" in df.columns and "ema_50" in df.columns:
            ema8 = self._safe_last(df["ema_8"])
            ema20 = self._safe_last(df["ema_20"])
            ema50 = self._safe_last(df["ema_50"])
            close = self._safe_last(df["close"])
            if close > ema8 > ema20 > ema50:
                scores.append(1.0)
                reasons.append("Trend: bullish EMA alignment")
            elif close < ema8 < ema20 < ema50:
                scores.append(-1.0)
                reasons.append("Trend: bearish EMA alignment")
            elif ema8 > ema20:
                scores.append(0.3)
                reasons.append("Trend: mild bullish")
            elif ema8 < ema20:
                scores.append(-0.3)
                reasons.append("Trend: mild bearish")
            else:
                scores.append(0.0)

        # 2. Momentum score (RSI + MACD)
        if "rsi" in df.columns:
            rsi = self._safe_last(df["rsi"])
            if rsi > 60:
                scores.append(0.5)
                reasons.append(f"Momentum: RSI bullish ({rsi:.0f})")
            elif rsi < 40:
                scores.append(-0.5)
                reasons.append(f"Momentum: RSI bearish ({rsi:.0f})")
            else:
                scores.append(0.0)

        if "macd_histogram" in df.columns:
            hist = self._safe_last(df["macd_histogram"])
            hist_rising = self._safe_last(df["macd_hist_rising"])
            if hist > 0 and hist_rising:
                scores.append(0.5)
                reasons.append("Momentum: MACD bullish")
            elif hist < 0 and not hist_rising:
                scores.append(-0.5)
                reasons.append("Momentum: MACD bearish")
            else:
                scores.append(0.0)

        # 3. Volume score
        if "rvol" in df.columns and "cmf" in df.columns:
            rvol = self._safe_last(df["rvol"])
            cmf = self._safe_last(df["cmf"])
            if rvol > 1.3 and cmf > 0.05:
                scores.append(0.5)
                reasons.append("Volume: bullish flow")
            elif rvol > 1.3 and cmf < -0.05:
                scores.append(-0.5)
                reasons.append("Volume: bearish flow")
            else:
                scores.append(0.0)

        # 4. Volatility regime
        if "vol_regime" in df.columns:
            regime = self._safe_last(df["vol_regime"])
            if regime == 2:  # High vol
                scores.append(0.3 if sum(scores) > 0 else -0.3)
                reasons.append("Volatility: high regime amplifying signal")

        if not scores:
            return self._hold("No confluence data available")

        avg_score = sum(scores) / len(scores)
        confidence = min(90, abs(avg_score) * 80 + 20)

        if avg_score > 0.2:
            return StrategySignal(self.name, "BUY", confidence,
                f"Hybrid confluence bullish ({avg_score:.2f}): {'; '.join(reasons)}",
                {"confluence_score": avg_score, "components": len(scores)})
        elif avg_score < -0.2:
            return StrategySignal(self.name, "SELL", confidence,
                f"Hybrid confluence bearish ({avg_score:.2f}): {'; '.join(reasons)}",
                {"confluence_score": avg_score, "components": len(scores)})

        return self._hold(f"Hybrid confluence neutral ({avg_score:.2f})")


class SmartMoneyEnhancedTrendStrategy(BaseStrategy):
    """Smart-money enhanced trend — combines trend signals with institutional flow detection."""

    def __init__(self):
        super().__init__(name="smart_money_enhanced_trend")

    def evaluate(self, df: pd.DataFrame, symbol: str = "") -> StrategySignal:
        if len(df) < 30:
            return self._hold("Insufficient data for SM enhanced trend")

        close = self._safe_last(df["close"])

        # Trend component
        ema20 = self._safe_last(df["ema_20"]) if "ema_20" in df.columns else close
        ema50 = self._safe_last(df["ema_50"]) if "ema_50" in df.columns else close
        trend_bull = close > ema20 > ema50
        trend_bear = close < ema20 < ema50

        # Smart money component: volume + structure
        cmf = self._safe_last(df["cmf"]) if "cmf" in df.columns else 0
        rvol = self._safe_last(df["rvol"]) if "rvol" in df.columns else 1.0
        ms_break_above = self._safe_last(df["ms_break_above"]) if "ms_break_above" in df.columns else 0
        ms_break_below = self._safe_last(df["ms_break_below"]) if "ms_break_below" in df.columns else 0

        sm_bullish = cmf > 0.05 and (rvol > 1.3 or ms_break_above)
        sm_bearish = cmf < -0.05 and (rvol > 1.3 or ms_break_below)

        if trend_bull and sm_bullish:
            conf = min(90, 70 + rvol * 5)
            return StrategySignal(self.name, "BUY", conf,
                "Smart-money enhanced bullish trend: institutional buying confirms uptrend",
                {"cmf": cmf, "rvol": rvol, "trend": "bullish", "sm": "bullish"})
        elif trend_bear and sm_bearish:
            conf = min(90, 70 + rvol * 5)
            return StrategySignal(self.name, "SELL", conf,
                "Smart-money enhanced bearish trend: institutional selling confirms downtrend",
                {"cmf": cmf, "rvol": rvol, "trend": "bearish", "sm": "bearish"})
        # Divergence: SM against trend (early reversal warning)
        elif trend_bull and sm_bearish:
            return StrategySignal(self.name, "SELL", 50,
                "Warning: smart-money selling into bullish trend — potential reversal",
                {"divergence": "sm_bearish_in_uptrend"})
        elif trend_bear and sm_bullish:
            return StrategySignal(self.name, "BUY", 50,
                "Warning: smart-money buying into bearish trend — potential reversal",
                {"divergence": "sm_bullish_in_downtrend"})

        return self._hold("No smart-money enhanced trend signal")


class MomentumStructuralFilterStrategy(BaseStrategy):
    """Momentum + structural filter — momentum signals validated by market structure."""

    def __init__(self):
        super().__init__(name="momentum_structural_filter")

    def evaluate(self, df: pd.DataFrame, symbol: str = "") -> StrategySignal:
        if len(df) < 20:
            return self._hold("Insufficient data for momentum + structural filter")

        # Momentum signals
        rsi = self._safe_last(df["rsi"]) if "rsi" in df.columns else 50
        macd_bull = self._safe_last(df["macd_cross_bull"]) if "macd_cross_bull" in df.columns else 0
        macd_bear = self._safe_last(df["macd_cross_bear"]) if "macd_cross_bear" in df.columns else 0
        stoch_k = self._safe_last(df["stoch_k"]) if "stoch_k" in df.columns else 50

        # Structural filter
        ms_pos = self._safe_last(df["ms_position"]) if "ms_position" in df.columns else 0.5
        adx = self._safe_last(df["adx"]) if "adx" in df.columns else 0
        dmi_bull = self._safe_last(df["dmi_bullish"]) if "dmi_bullish" in df.columns else 0

        # Bullish: momentum + structure aligned
        momentum_bull = (rsi > 50 and stoch_k > 50) or macd_bull
        structure_bull = ms_pos > 0.5 and dmi_bull and adx > 20

        momentum_bear = (rsi < 50 and stoch_k < 50) or macd_bear
        structure_bear = ms_pos < 0.5 and not dmi_bull and adx > 20

        if momentum_bull and structure_bull:
            conf = min(85, 60 + adx)
            return StrategySignal(self.name, "BUY", conf,
                f"Momentum + structure aligned bullish: RSI={rsi:.0f}, ADX={adx:.0f}, "
                f"structure position={ms_pos:.2f}",
                {"rsi": rsi, "adx": adx, "ms_position": ms_pos})
        elif momentum_bear and structure_bear:
            conf = min(85, 60 + adx)
            return StrategySignal(self.name, "SELL", conf,
                f"Momentum + structure aligned bearish: RSI={rsi:.0f}, ADX={adx:.0f}, "
                f"structure position={ms_pos:.2f}",
                {"rsi": rsi, "adx": adx, "ms_position": ms_pos})

        return self._hold("Momentum and structure not aligned")


class VolatilitySurpriseTrendFilterStrategy(BaseStrategy):
    """Volatility surprise + trend filter — trades unexpected volatility in trending markets."""

    def __init__(self):
        super().__init__(name="volatility_surprise_trend")

    def evaluate(self, df: pd.DataFrame, symbol: str = "") -> StrategySignal:
        if len(df) < 20:
            return self._hold("Insufficient data for volatility surprise")

        # Volatility surprise
        vol_regime = self._safe_last(df["vol_regime"]) if "vol_regime" in df.columns else 1
        vol_change = self._safe_last(df["vol_regime_change"]) if "vol_regime_change" in df.columns else 0
        atr_pct = self._safe_last(df["atr_pct"]) if "atr_pct" in df.columns else 0

        # Trend filter
        ema20 = self._safe_last(df["ema_20"]) if "ema_20" in df.columns else 0
        ema50 = self._safe_last(df["ema_50"]) if "ema_50" in df.columns else 0
        close = self._safe_last(df["close"])
        adx = self._safe_last(df["adx"]) if "adx" in df.columns else 0

        trend_bull = close > ema20 > ema50 and adx > 20
        trend_bear = close < ema20 < ema50 and adx > 20

        # Volatility regime change (surprise) in trending market
        if vol_change and vol_regime == 2:  # Shift to high volatility
            if trend_bull:
                return StrategySignal(self.name, "BUY", 75,
                    f"Volatility surprise in bullish trend: regime shifted to high, "
                    f"ATR={atr_pct:.2f}%, ADX={adx:.0f}",
                    {"vol_regime": "high", "atr_pct": atr_pct, "adx": adx})
            elif trend_bear:
                return StrategySignal(self.name, "SELL", 75,
                    f"Volatility surprise in bearish trend: regime shifted to high, "
                    f"ATR={atr_pct:.2f}%, ADX={adx:.0f}",
                    {"vol_regime": "high", "atr_pct": atr_pct, "adx": adx})

        # Sustained high volatility in trend
        if vol_regime == 2 and (trend_bull or trend_bear):
            signal = "BUY" if trend_bull else "SELL"
            return StrategySignal(self.name, signal, 55,
                f"High volatility regime in {'bullish' if trend_bull else 'bearish'} trend",
                {"vol_regime": "high", "atr_pct": atr_pct})

        return self._hold("No volatility surprise in trending market")