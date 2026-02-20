"""
APEX SIGNAL™ — Smart-Money / Structural Strategies (6)
Liquidity Sweep, Order Block Retest, Fair Value Gap Capture,
Break of Structure Momentum, Volume Imbalance Burst, Accumulation/Distribution
"""
import pandas as pd
import numpy as np
from apex_signal.strategies.base import BaseStrategy, StrategySignal


class LiquiditySweepStrategy(BaseStrategy):
    """Liquidity sweep detector — identifies stop hunts beyond key levels."""

    def __init__(self):
        super().__init__(name="liquidity_sweep")

    def evaluate(self, df: pd.DataFrame, symbol: str = "") -> StrategySignal:
        if len(df) < 30:
            return self._hold("Insufficient data for liquidity sweep")

        close = self._safe_last(df["close"])
        high = df["high"].iloc[-1]
        low = df["low"].iloc[-1]

        # Recent swing levels
        lookback = df.iloc[-30:-2]
        recent_high = lookback["high"].max()
        recent_low = lookback["low"].min()

        # Sweep above: wick above recent high but close below it
        swept_above = high > recent_high and close < recent_high
        # Sweep below: wick below recent low but close above it
        swept_below = low < recent_low and close > recent_low

        if swept_below:
            sweep_depth = ((recent_low - low) / recent_low) * 100 if recent_low > 0 else 0
            conf = min(85, 65 + sweep_depth * 20)
            return StrategySignal(self.name, "BUY", conf,
                f"Liquidity sweep below {recent_low:.2f} — stop hunt reversal detected "
                f"(sweep depth: {sweep_depth:.2f}%)",
                {"sweep_level": recent_low, "sweep_depth_pct": sweep_depth})

        elif swept_above:
            sweep_depth = ((high - recent_high) / recent_high) * 100 if recent_high > 0 else 0
            conf = min(85, 65 + sweep_depth * 20)
            return StrategySignal(self.name, "SELL", conf,
                f"Liquidity sweep above {recent_high:.2f} — stop hunt reversal detected "
                f"(sweep depth: {sweep_depth:.2f}%)",
                {"sweep_level": recent_high, "sweep_depth_pct": sweep_depth})

        return self._hold("No liquidity sweep detected")


class OrderBlockRetestStrategy(BaseStrategy):
    """Order block retest — identifies institutional order blocks and retests."""

    def __init__(self):
        super().__init__(name="order_block_retest")

    def evaluate(self, df: pd.DataFrame, symbol: str = "") -> StrategySignal:
        if len(df) < 30:
            return self._hold("Insufficient data for order block detection")

        close = self._safe_last(df["close"])

        # Find bullish order blocks: last bearish candle before a strong bullish move
        bullish_ob = None
        bearish_ob = None

        for i in range(len(df) - 20, len(df) - 3):
            if i < 1:
                continue
            # Bullish OB: bearish candle followed by strong bullish move
            if (df["close"].iloc[i] < df["open"].iloc[i] and
                df["close"].iloc[i + 1] > df["open"].iloc[i + 1] and
                (df["close"].iloc[i + 1] - df["open"].iloc[i + 1]) >
                    1.5 * abs(df["close"].iloc[i] - df["open"].iloc[i])):
                bullish_ob = {
                    "high": df["high"].iloc[i],
                    "low": df["low"].iloc[i],
                    "idx": i
                }

            # Bearish OB: bullish candle followed by strong bearish move
            if (df["close"].iloc[i] > df["open"].iloc[i] and
                df["close"].iloc[i + 1] < df["open"].iloc[i + 1] and
                abs(df["close"].iloc[i + 1] - df["open"].iloc[i + 1]) >
                    1.5 * (df["close"].iloc[i] - df["open"].iloc[i])):
                bearish_ob = {
                    "high": df["high"].iloc[i],
                    "low": df["low"].iloc[i],
                    "idx": i
                }

        # Check for retest of bullish OB
        if bullish_ob and bullish_ob["low"] <= close <= bullish_ob["high"]:
            return StrategySignal(self.name, "BUY", 75,
                f"Price retesting bullish order block zone "
                f"({bullish_ob['low']:.2f} - {bullish_ob['high']:.2f})",
                {"ob_zone": bullish_ob})

        # Check for retest of bearish OB
        if bearish_ob and bearish_ob["low"] <= close <= bearish_ob["high"]:
            return StrategySignal(self.name, "SELL", 75,
                f"Price retesting bearish order block zone "
                f"({bearish_ob['low']:.2f} - {bearish_ob['high']:.2f})",
                {"ob_zone": bearish_ob})

        return self._hold("No order block retest detected")


class FairValueGapStrategy(BaseStrategy):
    """Fair Value Gap capture — identifies and trades FVG imbalances."""

    def __init__(self):
        super().__init__(name="fair_value_gap")

    def evaluate(self, df: pd.DataFrame, symbol: str = "") -> StrategySignal:
        if len(df) < 10:
            return self._hold("Insufficient data for FVG detection")

        close = self._safe_last(df["close"])

        # Scan for recent FVGs (gaps between candle 1 high and candle 3 low)
        bullish_fvg = None
        bearish_fvg = None

        for i in range(len(df) - 15, len(df) - 3):
            if i < 2:
                continue
            # Bullish FVG: candle[i] high < candle[i+2] low (gap up)
            if df["high"].iloc[i] < df["low"].iloc[i + 2]:
                bullish_fvg = {
                    "top": df["low"].iloc[i + 2],
                    "bottom": df["high"].iloc[i],
                    "mid": (df["low"].iloc[i + 2] + df["high"].iloc[i]) / 2,
                }

            # Bearish FVG: candle[i] low > candle[i+2] high (gap down)
            if df["low"].iloc[i] > df["high"].iloc[i + 2]:
                bearish_fvg = {
                    "top": df["low"].iloc[i],
                    "bottom": df["high"].iloc[i + 2],
                    "mid": (df["low"].iloc[i] + df["high"].iloc[i + 2]) / 2,
                }

        # Price entering bullish FVG zone (buy the dip into gap)
        if bullish_fvg and bullish_fvg["bottom"] <= close <= bullish_fvg["top"]:
            return StrategySignal(self.name, "BUY", 70,
                f"Price filling bullish FVG zone "
                f"({bullish_fvg['bottom']:.2f} - {bullish_fvg['top']:.2f})",
                {"fvg": bullish_fvg})

        # Price entering bearish FVG zone (sell the rally into gap)
        if bearish_fvg and bearish_fvg["bottom"] <= close <= bearish_fvg["top"]:
            return StrategySignal(self.name, "SELL", 70,
                f"Price filling bearish FVG zone "
                f"({bearish_fvg['bottom']:.2f} - {bearish_fvg['top']:.2f})",
                {"fvg": bearish_fvg})

        return self._hold("No fair value gap interaction detected")


class BreakOfStructureStrategy(BaseStrategy):
    """Break of structure momentum — detects BOS transitions in market structure."""

    def __init__(self):
        super().__init__(name="break_of_structure")

    def evaluate(self, df: pd.DataFrame, symbol: str = "") -> StrategySignal:
        if len(df) < 10 or "ms_break_above" not in df.columns:
            return self._hold("Insufficient data for BOS detection")

        bos_above = self._safe_last(df["ms_break_above"])
        bos_below = self._safe_last(df["ms_break_below"])
        higher_high = self._safe_last(df["ms_higher_high"])
        lower_low = self._safe_last(df["ms_lower_low"])
        ms_pos = self._safe_last(df["ms_position"])

        if bos_above and higher_high:
            return StrategySignal(self.name, "BUY", 80,
                "Bullish break of structure with higher high — trend continuation confirmed",
                {"ms_position": ms_pos, "pattern": "BOS_bull_HH"})
        elif bos_above:
            return StrategySignal(self.name, "BUY", 65,
                "Bullish break of structure — resistance broken",
                {"ms_position": ms_pos, "pattern": "BOS_bull"})
        elif bos_below and lower_low:
            return StrategySignal(self.name, "SELL", 80,
                "Bearish break of structure with lower low — trend continuation confirmed",
                {"ms_position": ms_pos, "pattern": "BOS_bear_LL"})
        elif bos_below:
            return StrategySignal(self.name, "SELL", 65,
                "Bearish break of structure — support broken",
                {"ms_position": ms_pos, "pattern": "BOS_bear"})

        return self._hold("No break of structure detected")


class VolumeImbalanceBurstStrategy(BaseStrategy):
    """Volume imbalance burst — detects sudden volume spikes indicating institutional activity."""

    def __init__(self):
        super().__init__(name="volume_imbalance_burst")

    def evaluate(self, df: pd.DataFrame, symbol: str = "") -> StrategySignal:
        if len(df) < 20 or "rvol" not in df.columns:
            return self._hold("Insufficient data for volume imbalance")

        rvol = self._safe_last(df["rvol"])
        rvol_high = self._safe_last(df["rvol_high"])
        close = self._safe_last(df["close"])
        open_price = df["open"].iloc[-1]
        cmf = self._safe_last(df["cmf"]) if "cmf" in df.columns else 0

        bullish_candle = close > open_price
        candle_body = abs(close - open_price)
        candle_range = df["high"].iloc[-1] - df["low"].iloc[-1]
        body_ratio = candle_body / candle_range if candle_range > 0 else 0

        # High volume + strong directional candle = institutional activity
        if rvol > 2.0 and body_ratio > 0.6:
            if bullish_candle:
                conf = min(85, 60 + rvol * 5)
                return StrategySignal(self.name, "BUY", conf,
                    f"Volume imbalance burst: {rvol:.1f}x relative volume with strong bullish candle"
                    + (f", CMF confirms buying ({cmf:.3f})" if cmf > 0.1 else ""),
                    {"rvol": rvol, "body_ratio": body_ratio, "cmf": cmf})
            else:
                conf = min(85, 60 + rvol * 5)
                return StrategySignal(self.name, "SELL", conf,
                    f"Volume imbalance burst: {rvol:.1f}x relative volume with strong bearish candle"
                    + (f", CMF confirms selling ({cmf:.3f})" if cmf < -0.1 else ""),
                    {"rvol": rvol, "body_ratio": body_ratio, "cmf": cmf})

        elif rvol > 1.5 and rvol_high:
            signal = "BUY" if bullish_candle else "SELL"
            return StrategySignal(self.name, signal, 50,
                f"Elevated volume ({rvol:.1f}x) with {'bullish' if bullish_candle else 'bearish'} bias",
                {"rvol": rvol})

        return self._hold(f"Relative volume at {rvol:.1f}x — no imbalance")


class AccumulationDistributionStrategy(BaseStrategy):
    """Accumulation/Distribution signal — detects institutional accumulation or distribution phases."""

    def __init__(self):
        super().__init__(name="accumulation_distribution")

    def evaluate(self, df: pd.DataFrame, symbol: str = "") -> StrategySignal:
        if len(df) < 20 or "cmf" not in df.columns:
            return self._hold("Insufficient data for A/D detection")

        cmf = self._safe_last(df["cmf"])
        obv = self._safe_last(df["obv"]) if "obv" in df.columns else 0
        obv_ema = self._safe_last(df["obv_ema"]) if "obv_ema" in df.columns else 0
        close = self._safe_last(df["close"])
        sma20 = self._safe_last(df["sma_20"]) if "sma_20" in df.columns else close

        # CMF trend over last 5 bars
        cmf_values = df["cmf"].iloc[-5:].values
        cmf_trending_up = all(cmf_values[i] <= cmf_values[i + 1] for i in range(len(cmf_values) - 1)) if len(cmf_values) >= 2 else False
        cmf_trending_down = all(cmf_values[i] >= cmf_values[i + 1] for i in range(len(cmf_values) - 1)) if len(cmf_values) >= 2 else False

        obv_above_ema = obv > obv_ema

        # Accumulation: CMF positive + trending up + OBV above EMA
        if cmf > 0.05 and cmf_trending_up and obv_above_ema:
            conf = 75 if close > sma20 else 60
            return StrategySignal(self.name, "BUY", conf,
                f"Accumulation phase detected: CMF={cmf:.3f} trending up, OBV above EMA",
                {"cmf": cmf, "phase": "accumulation"})

        # Distribution: CMF negative + trending down + OBV below EMA
        elif cmf < -0.05 and cmf_trending_down and not obv_above_ema:
            conf = 75 if close < sma20 else 60
            return StrategySignal(self.name, "SELL", conf,
                f"Distribution phase detected: CMF={cmf:.3f} trending down, OBV below EMA",
                {"cmf": cmf, "phase": "distribution"})

        # Early accumulation
        elif cmf > 0 and obv_above_ema and close < sma20:
            return StrategySignal(self.name, "BUY", 50,
                "Early accumulation: positive CMF with OBV strength despite price below SMA20",
                {"cmf": cmf, "phase": "early_accumulation"})

        return self._hold(f"CMF at {cmf:.3f} — no clear A/D phase")