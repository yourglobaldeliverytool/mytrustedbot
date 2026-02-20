"""
APEX SIGNAL™ — Smart-Money Detection Engine
Detects institutional patterns: liquidity sweeps, order blocks, FVGs,
break of structure, volume imbalance, and accumulation/distribution.
"""
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime, timezone
import pandas as pd
import numpy as np

from apex_signal.utils.logger import get_logger
from apex_signal.utils.helpers import clamp

logger = get_logger("smart_money")


@dataclass
class SmartMoneyEvent:
    """A single smart-money event detection."""
    event_type: str
    description: str
    score: float  # 0-1
    level: Optional[float] = None
    timestamp: Optional[str] = None


@dataclass
class SmartMoneyResult:
    """Aggregated smart-money detection result."""
    smart_money_score: float  # 0-1
    events: List[SmartMoneyEvent] = field(default_factory=list)
    bias: str = "neutral"  # bullish / bearish / neutral

    def to_dict(self) -> Dict[str, Any]:
        return {
            "smart_money_score": round(self.smart_money_score, 4),
            "events": [
                {
                    "type": e.event_type,
                    "description": e.description,
                    "score": round(e.score, 4),
                    "level": e.level,
                }
                for e in self.events
            ],
            "bias": self.bias,
        }


class SmartMoneyDetector:
    """
    Comprehensive smart-money detection engine.
    Analyzes price action for institutional footprints.
    """

    def __init__(self, swing_lookback: int = 20, volume_lookback: int = 20):
        self.swing_lookback = swing_lookback
        self.volume_lookback = volume_lookback

    def detect(self, df: pd.DataFrame) -> SmartMoneyResult:
        """Run all smart-money detections and return aggregated result."""
        if df.empty or len(df) < self.swing_lookback + 5:
            return SmartMoneyResult(smart_money_score=0.0, bias="neutral")

        events: List[SmartMoneyEvent] = []

        # Run all detectors
        events.extend(self._detect_liquidity_sweeps(df))
        events.extend(self._detect_order_blocks(df))
        events.extend(self._detect_fair_value_gaps(df))
        events.extend(self._detect_break_of_structure(df))
        events.extend(self._detect_volume_imbalance(df))
        events.extend(self._detect_accumulation_distribution(df))

        if not events:
            return SmartMoneyResult(smart_money_score=0.0, bias="neutral")

        # Aggregate score
        total_score = sum(e.score for e in events) / len(events)
        total_score = clamp(total_score, 0.0, 1.0)

        # Determine bias
        bullish_events = [e for e in events if "bullish" in e.description.lower() or "buy" in e.event_type.lower()]
        bearish_events = [e for e in events if "bearish" in e.description.lower() or "sell" in e.event_type.lower()]

        bull_score = sum(e.score for e in bullish_events) if bullish_events else 0
        bear_score = sum(e.score for e in bearish_events) if bearish_events else 0

        if bull_score > bear_score * 1.2:
            bias = "bullish"
        elif bear_score > bull_score * 1.2:
            bias = "bearish"
        else:
            bias = "neutral"

        return SmartMoneyResult(
            smart_money_score=total_score,
            events=events,
            bias=bias,
        )

    def _detect_liquidity_sweeps(self, df: pd.DataFrame) -> List[SmartMoneyEvent]:
        """Detect liquidity sweeps — wicks beyond key levels that reverse."""
        events = []
        lookback = df.iloc[-self.swing_lookback - 5:-3]
        recent = df.iloc[-3:]

        if lookback.empty or recent.empty:
            return events

        swing_high = lookback["high"].max()
        swing_low = lookback["low"].min()

        for i in range(len(recent)):
            row = recent.iloc[i]
            # Sweep above: wick above swing high, close below
            if row["high"] > swing_high and row["close"] < swing_high:
                depth = (row["high"] - swing_high) / swing_high if swing_high > 0 else 0
                events.append(SmartMoneyEvent(
                    event_type="liquidity_sweep_sell",
                    description=f"Bearish liquidity sweep above {swing_high:.2f}",
                    score=min(0.9, 0.5 + depth * 50),
                    level=swing_high,
                ))
            # Sweep below: wick below swing low, close above
            if row["low"] < swing_low and row["close"] > swing_low:
                depth = (swing_low - row["low"]) / swing_low if swing_low > 0 else 0
                events.append(SmartMoneyEvent(
                    event_type="liquidity_sweep_buy",
                    description=f"Bullish liquidity sweep below {swing_low:.2f}",
                    score=min(0.9, 0.5 + depth * 50),
                    level=swing_low,
                ))

        return events

    def _detect_order_blocks(self, df: pd.DataFrame) -> List[SmartMoneyEvent]:
        """Detect order blocks — last opposing candle before an impulsive move."""
        events = []
        close = df["close"].iloc[-1]

        for i in range(len(df) - 15, len(df) - 3):
            if i < 1:
                continue

            c_curr = df["close"].iloc[i]
            o_curr = df["open"].iloc[i]
            c_next = df["close"].iloc[i + 1]
            o_next = df["open"].iloc[i + 1]

            body_curr = abs(c_curr - o_curr)
            body_next = abs(c_next - o_next)

            # Bullish OB: bearish candle before strong bullish
            if c_curr < o_curr and c_next > o_next and body_next > 1.5 * body_curr:
                ob_high = df["high"].iloc[i]
                ob_low = df["low"].iloc[i]
                if ob_low <= close <= ob_high:
                    events.append(SmartMoneyEvent(
                        event_type="order_block_buy",
                        description=f"Bullish order block retest at {ob_low:.2f}-{ob_high:.2f}",
                        score=0.7,
                        level=(ob_high + ob_low) / 2,
                    ))

            # Bearish OB: bullish candle before strong bearish
            if c_curr > o_curr and c_next < o_next and body_next > 1.5 * body_curr:
                ob_high = df["high"].iloc[i]
                ob_low = df["low"].iloc[i]
                if ob_low <= close <= ob_high:
                    events.append(SmartMoneyEvent(
                        event_type="order_block_sell",
                        description=f"Bearish order block retest at {ob_low:.2f}-{ob_high:.2f}",
                        score=0.7,
                        level=(ob_high + ob_low) / 2,
                    ))

        return events[-2:]  # Limit to most recent

    def _detect_fair_value_gaps(self, df: pd.DataFrame) -> List[SmartMoneyEvent]:
        """Detect Fair Value Gaps — price imbalances between candles."""
        events = []
        close = df["close"].iloc[-1]

        for i in range(len(df) - 12, len(df) - 3):
            if i < 2:
                continue

            # Bullish FVG: candle[i] high < candle[i+2] low
            if df["high"].iloc[i] < df["low"].iloc[i + 2]:
                fvg_top = df["low"].iloc[i + 2]
                fvg_bottom = df["high"].iloc[i]
                if fvg_bottom <= close <= fvg_top:
                    gap_size = (fvg_top - fvg_bottom) / fvg_bottom if fvg_bottom > 0 else 0
                    events.append(SmartMoneyEvent(
                        event_type="fvg_buy",
                        description=f"Bullish FVG fill at {fvg_bottom:.2f}-{fvg_top:.2f}",
                        score=min(0.8, 0.4 + gap_size * 20),
                        level=(fvg_top + fvg_bottom) / 2,
                    ))

            # Bearish FVG: candle[i] low > candle[i+2] high
            if df["low"].iloc[i] > df["high"].iloc[i + 2]:
                fvg_top = df["low"].iloc[i]
                fvg_bottom = df["high"].iloc[i + 2]
                if fvg_bottom <= close <= fvg_top:
                    gap_size = (fvg_top - fvg_bottom) / fvg_bottom if fvg_bottom > 0 else 0
                    events.append(SmartMoneyEvent(
                        event_type="fvg_sell",
                        description=f"Bearish FVG fill at {fvg_bottom:.2f}-{fvg_top:.2f}",
                        score=min(0.8, 0.4 + gap_size * 20),
                        level=(fvg_top + fvg_bottom) / 2,
                    ))

        return events[-2:]

    def _detect_break_of_structure(self, df: pd.DataFrame) -> List[SmartMoneyEvent]:
        """Detect Break of Structure — when price breaks a swing high/low."""
        events = []
        n = min(10, len(df) // 4)
        if n < 3:
            return events

        highs = df["high"].values
        lows = df["low"].values
        close = df["close"].iloc[-1]

        # Find recent swing points
        swing_highs = []
        swing_lows = []

        for i in range(n, len(df) - n):
            if highs[i] == max(highs[max(0, i - n):i + n + 1]):
                swing_highs.append((i, highs[i]))
            if lows[i] == min(lows[max(0, i - n):i + n + 1]):
                swing_lows.append((i, lows[i]))

        # Check for BOS
        if len(swing_highs) >= 2:
            last_sh = swing_highs[-1][1]
            prev_sh = swing_highs[-2][1]
            if close > last_sh and last_sh > prev_sh:
                events.append(SmartMoneyEvent(
                    event_type="bos_buy",
                    description=f"Bullish BOS: higher high break above {last_sh:.2f}",
                    score=0.75,
                    level=last_sh,
                ))

        if len(swing_lows) >= 2:
            last_sl = swing_lows[-1][1]
            prev_sl = swing_lows[-2][1]
            if close < last_sl and last_sl < prev_sl:
                events.append(SmartMoneyEvent(
                    event_type="bos_sell",
                    description=f"Bearish BOS: lower low break below {last_sl:.2f}",
                    score=0.75,
                    level=last_sl,
                ))

        return events

    def _detect_volume_imbalance(self, df: pd.DataFrame) -> List[SmartMoneyEvent]:
        """Detect volume imbalance — sudden volume spikes with directional bias."""
        events = []
        if "volume" not in df.columns or len(df) < self.volume_lookback:
            return events

        avg_vol = df["volume"].iloc[-self.volume_lookback:-1].mean()
        if avg_vol == 0:
            return events

        recent_vol = df["volume"].iloc[-1]
        ratio = recent_vol / avg_vol

        if ratio > 2.0:
            close = df["close"].iloc[-1]
            open_p = df["open"].iloc[-1]
            bullish = close > open_p

            events.append(SmartMoneyEvent(
                event_type="volume_imbalance_buy" if bullish else "volume_imbalance_sell",
                description=f"{'Bullish' if bullish else 'Bearish'} volume imbalance ({ratio:.1f}x average)",
                score=min(0.85, 0.4 + (ratio - 2.0) * 0.15),
            ))

        return events

    def _detect_accumulation_distribution(self, df: pd.DataFrame) -> List[SmartMoneyEvent]:
        """Detect accumulation/distribution phases from volume and price patterns."""
        events = []
        if len(df) < 10:
            return events

        recent = df.iloc[-10:]
        closes = recent["close"].values
        volumes = recent["volume"].values

        if len(closes) < 5:
            return events

        # Price trend
        price_trend = closes[-1] - closes[0]
        # Volume trend
        vol_first_half = volumes[:5].mean()
        vol_second_half = volumes[5:].mean()

        if vol_first_half == 0:
            return events

        vol_trend = vol_second_half / vol_first_half

        # Accumulation: price flat/down but volume increasing
        if price_trend <= 0 and vol_trend > 1.3:
            events.append(SmartMoneyEvent(
                event_type="accumulation_buy",
                description=f"Accumulation detected: volume rising ({vol_trend:.1f}x) while price consolidates",
                score=0.6,
            ))
        # Distribution: price flat/up but volume pattern suggests selling
        elif price_trend >= 0 and vol_trend > 1.3:
            # Check if candles show selling pressure (upper wicks)
            upper_wicks = (recent["high"] - recent[["close", "open"]].max(axis=1)).mean()
            body_size = (recent["close"] - recent["open"]).abs().mean()
            if body_size > 0 and upper_wicks / body_size > 0.5:
                events.append(SmartMoneyEvent(
                    event_type="distribution_sell",
                    description="Distribution detected: rising volume with selling pressure (upper wicks)",
                    score=0.6,
                ))

        return events


# Singleton
_detector: Optional[SmartMoneyDetector] = None


def get_smart_money_detector() -> SmartMoneyDetector:
    global _detector
    if _detector is None:
        _detector = SmartMoneyDetector()
    return _detector