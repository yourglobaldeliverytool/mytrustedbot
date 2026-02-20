"""
APEX SIGNAL™ — Divergence Detector
Detects bullish/bearish divergences between price and RSI/MACD.
Divergence is one of the strongest reversal confirmation signals.
"""
import pandas as pd
import numpy as np
from apex_signal.indicators.base import BaseIndicator
from apex_signal.utils.logger import get_logger

logger = get_logger("divergence")


class DivergenceDetector(BaseIndicator):
    """
    Detects regular and hidden divergences between price and oscillators.
    
    Regular Bullish: Price makes lower low, RSI/MACD makes higher low → reversal up
    Regular Bearish: Price makes higher high, RSI/MACD makes lower high → reversal down
    Hidden Bullish: Price makes higher low, RSI makes lower low → trend continuation up
    Hidden Bearish: Price makes lower high, RSI makes higher high → trend continuation down
    """

    def __init__(self, lookback: int = 20, min_swing_pct: float = 0.3):
        self.lookback = lookback
        self.min_swing_pct = min_swing_pct
        super().__init__(name="divergence", params={
            "lookback": lookback, "min_swing_pct": min_swing_pct
        })

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()

        # Initialize divergence columns
        df["div_rsi_bull"] = 0
        df["div_rsi_bear"] = 0
        df["div_rsi_hidden_bull"] = 0
        df["div_rsi_hidden_bear"] = 0
        df["div_macd_bull"] = 0
        df["div_macd_bear"] = 0
        df["div_score"] = 0.0  # Composite divergence score (-1 to 1)

        if len(df) < self.lookback + 5:
            self._last_result = df
            return df

        # Find swing points
        swing_lows, swing_highs = self._find_swings(df)

        # RSI divergence
        if "rsi" in df.columns:
            self._detect_divergence(
                df, swing_lows, swing_highs, "rsi",
                "div_rsi_bull", "div_rsi_bear",
                "div_rsi_hidden_bull", "div_rsi_hidden_bear"
            )

        # MACD divergence
        if "macd_histogram" in df.columns:
            self._detect_divergence(
                df, swing_lows, swing_highs, "macd_histogram",
                "div_macd_bull", "div_macd_bear",
                None, None  # Skip hidden for MACD
            )

        # Composite divergence score
        bull_signals = (
            df["div_rsi_bull"] * 0.4 +
            df["div_rsi_hidden_bull"] * 0.2 +
            df["div_macd_bull"] * 0.4
        )
        bear_signals = (
            df["div_rsi_bear"] * 0.4 +
            df["div_rsi_hidden_bear"] * 0.2 +
            df["div_macd_bear"] * 0.4
        )
        df["div_score"] = bull_signals - bear_signals

        self._last_result = df
        return df

    def _find_swings(self, df: pd.DataFrame):
        """Find swing high and swing low indices."""
        n = min(5, self.lookback // 4)
        highs = df["high"].values
        lows = df["low"].values
        closes = df["close"].values

        swing_lows = []  # (index, price, close)
        swing_highs = []

        for i in range(n, len(df) - n):
            window_low = lows[max(0, i - n):i + n + 1]
            window_high = highs[max(0, i - n):i + n + 1]

            if lows[i] == np.min(window_low):
                swing_lows.append((i, lows[i], closes[i]))
            if highs[i] == np.max(window_high):
                swing_highs.append((i, highs[i], closes[i]))

        return swing_lows, swing_highs

    def _detect_divergence(
        self, df: pd.DataFrame,
        swing_lows, swing_highs,
        osc_col: str,
        bull_col: str, bear_col: str,
        hidden_bull_col, hidden_bear_col
    ):
        """Detect divergences between price swings and oscillator."""
        osc = df[osc_col].values

        # Regular Bullish: price lower low + oscillator higher low
        for i in range(1, len(swing_lows)):
            idx_prev, price_prev, _ = swing_lows[i - 1]
            idx_curr, price_curr, _ = swing_lows[i]

            if idx_curr >= len(df) or idx_prev >= len(df):
                continue

            price_lower = price_curr < price_prev
            osc_higher = osc[idx_curr] > osc[idx_prev]

            # Price makes lower low but oscillator makes higher low
            if price_lower and osc_higher:
                pct_diff = abs(price_curr - price_prev) / price_prev * 100
                if pct_diff >= self.min_swing_pct:
                    df.iloc[idx_curr, df.columns.get_loc(bull_col)] = 1

            # Hidden Bullish: price higher low + oscillator lower low
            if hidden_bull_col and not price_lower and not osc_higher:
                pct_diff = abs(price_curr - price_prev) / price_prev * 100
                if pct_diff >= self.min_swing_pct:
                    df.iloc[idx_curr, df.columns.get_loc(hidden_bull_col)] = 1

        # Regular Bearish: price higher high + oscillator lower high
        for i in range(1, len(swing_highs)):
            idx_prev, price_prev, _ = swing_highs[i - 1]
            idx_curr, price_curr, _ = swing_highs[i]

            if idx_curr >= len(df) or idx_prev >= len(df):
                continue

            price_higher = price_curr > price_prev
            osc_lower = osc[idx_curr] < osc[idx_prev]

            if price_higher and osc_lower:
                pct_diff = abs(price_curr - price_prev) / price_prev * 100
                if pct_diff >= self.min_swing_pct:
                    df.iloc[idx_curr, df.columns.get_loc(bear_col)] = 1

            # Hidden Bearish: price lower high + oscillator higher high
            if hidden_bear_col and not price_higher and not osc_lower:
                pct_diff = abs(price_curr - price_prev) / price_prev * 100
                if pct_diff >= self.min_swing_pct:
                    df.iloc[idx_curr, df.columns.get_loc(hidden_bear_col)] = 1

    def reset(self) -> None:
        super().reset()