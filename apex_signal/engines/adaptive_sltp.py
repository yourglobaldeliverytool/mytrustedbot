"""
APEX SIGNAL™ — Adaptive Stop-Loss / Take-Profit Engine
Dynamic SL/TP based on ATR, volatility regime, market structure, and confidence tier.
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from dataclasses import dataclass

from apex_signal.utils.logger import get_logger
from apex_signal.utils.helpers import clamp

logger = get_logger("adaptive_sltp")


@dataclass
class SLTPLevels:
    """Computed stop-loss and take-profit levels."""
    entry_price: float
    stop_loss: float
    take_profit_1: float  # Primary TP (partial exit)
    take_profit_2: float  # Secondary TP (full exit)
    trailing_stop_distance: float
    risk_reward_ratio: float
    sl_pct: float
    tp1_pct: float
    tp2_pct: float
    method: str  # Description of how levels were computed

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entry_price": round(self.entry_price, 6),
            "stop_loss": round(self.stop_loss, 6),
            "take_profit_1": round(self.take_profit_1, 6),
            "take_profit_2": round(self.take_profit_2, 6),
            "trailing_stop_distance": round(self.trailing_stop_distance, 6),
            "risk_reward_ratio": round(self.risk_reward_ratio, 2),
            "sl_pct": round(self.sl_pct, 3),
            "tp1_pct": round(self.tp1_pct, 3),
            "tp2_pct": round(self.tp2_pct, 3),
            "method": self.method,
        }


class AdaptiveSLTP:
    """
    Adaptive Stop-Loss / Take-Profit engine.
    
    Computes dynamic levels based on:
    1. ATR (volatility-scaled distance)
    2. Market structure (support/resistance levels)
    3. Volatility regime (wider in high vol, tighter in low vol)
    4. Confidence tier (tighter SL for high confidence)
    5. Minimum risk-reward ratio enforcement (>= 2:1)
    """

    def __init__(
        self,
        min_rr_ratio: float = 2.0,
        atr_sl_multiplier: float = 1.5,
        atr_tp1_multiplier: float = 2.5,
        atr_tp2_multiplier: float = 4.0,
        trailing_atr_multiplier: float = 2.0,
        max_sl_pct: float = 5.0,
        min_sl_pct: float = 0.3,
    ):
        self.min_rr_ratio = min_rr_ratio
        self.atr_sl_mult = atr_sl_multiplier
        self.atr_tp1_mult = atr_tp1_multiplier
        self.atr_tp2_mult = atr_tp2_multiplier
        self.trailing_atr_mult = trailing_atr_multiplier
        self.max_sl_pct = max_sl_pct
        self.min_sl_pct = min_sl_pct

    def compute(
        self,
        df: pd.DataFrame,
        side: str,
        confidence: float,
        tier: str,
        entry_price: Optional[float] = None,
    ) -> SLTPLevels:
        """
        Compute adaptive SL/TP levels for a signal.
        
        Args:
            df: Indicator-enriched DataFrame
            side: BUY or SELL
            confidence: Signal confidence (0-100)
            tier: Signal tier (Elite/Strong/Moderate/Weak)
            entry_price: Override entry price (default: last close)
        """
        if df.empty:
            return self._default_levels(entry_price or 0, side)

        close = float(df["close"].iloc[-1])
        price = entry_price or close

        # Get ATR
        atr = float(df["atr"].iloc[-1]) if "atr" in df.columns else close * 0.01
        atr = max(atr, close * 0.001)  # Floor at 0.1% of price

        # Volatility regime adjustment
        vol_regime = int(df["vol_regime"].iloc[-1]) if "vol_regime" in df.columns else 1
        regime_mult = {0: 0.7, 1: 1.0, 2: 1.4}.get(vol_regime, 1.0)

        # Tier adjustment (higher confidence → tighter SL, wider TP)
        tier_sl_mult = {"Elite": 0.8, "Strong": 0.9, "Moderate": 1.0, "Weak": 1.2}.get(tier, 1.0)
        tier_tp_mult = {"Elite": 1.3, "Strong": 1.1, "Moderate": 1.0, "Weak": 0.8}.get(tier, 1.0)

        # Compute raw distances
        sl_distance = atr * self.atr_sl_mult * regime_mult * tier_sl_mult
        tp1_distance = atr * self.atr_tp1_mult * regime_mult * tier_tp_mult
        tp2_distance = atr * self.atr_tp2_mult * regime_mult * tier_tp_mult
        trailing_distance = atr * self.trailing_atr_mult * regime_mult

        # Market structure adjustment: snap SL to nearest support/resistance
        sl_distance, tp1_distance = self._adjust_for_structure(
            df, price, side, sl_distance, tp1_distance
        )

        # Enforce SL percentage limits
        sl_pct = (sl_distance / price) * 100
        sl_pct = clamp(sl_pct, self.min_sl_pct, self.max_sl_pct)
        sl_distance = price * sl_pct / 100

        # Enforce minimum risk-reward ratio
        if tp1_distance < sl_distance * self.min_rr_ratio:
            tp1_distance = sl_distance * self.min_rr_ratio
        if tp2_distance < sl_distance * (self.min_rr_ratio + 1):
            tp2_distance = sl_distance * (self.min_rr_ratio + 1)

        # Compute actual levels
        if side == "BUY":
            stop_loss = price - sl_distance
            take_profit_1 = price + tp1_distance
            take_profit_2 = price + tp2_distance
        else:  # SELL
            stop_loss = price + sl_distance
            take_profit_1 = price - tp1_distance
            take_profit_2 = price - tp2_distance

        rr_ratio = tp1_distance / sl_distance if sl_distance > 0 else 0

        method_parts = [
            f"ATR-based ({atr:.4f})",
            f"regime={'low' if vol_regime == 0 else 'normal' if vol_regime == 1 else 'high'}",
            f"tier={tier}",
            f"R:R={rr_ratio:.1f}:1",
        ]

        return SLTPLevels(
            entry_price=price,
            stop_loss=stop_loss,
            take_profit_1=take_profit_1,
            take_profit_2=take_profit_2,
            trailing_stop_distance=trailing_distance,
            risk_reward_ratio=rr_ratio,
            sl_pct=sl_pct,
            tp1_pct=(tp1_distance / price) * 100,
            tp2_pct=(tp2_distance / price) * 100,
            method="; ".join(method_parts),
        )

    def _adjust_for_structure(
        self, df: pd.DataFrame, price: float, side: str,
        sl_dist: float, tp_dist: float
    ) -> tuple:
        """Adjust SL/TP to align with market structure levels."""
        if "ms_support" not in df.columns or "ms_resistance" not in df.columns:
            return sl_dist, tp_dist

        support = float(df["ms_support"].iloc[-1])
        resistance = float(df["ms_resistance"].iloc[-1])

        if side == "BUY":
            # Place SL just below support if it's closer than ATR-based SL
            structure_sl_dist = price - support + (price * 0.001)  # Small buffer
            if 0 < structure_sl_dist < sl_dist * 1.3:
                sl_dist = structure_sl_dist

            # Extend TP to resistance if it's reasonable
            structure_tp_dist = resistance - price
            if structure_tp_dist > tp_dist * 0.8:
                tp_dist = max(tp_dist, structure_tp_dist * 0.9)

        elif side == "SELL":
            structure_sl_dist = resistance - price + (price * 0.001)
            if 0 < structure_sl_dist < sl_dist * 1.3:
                sl_dist = structure_sl_dist

            structure_tp_dist = price - support
            if structure_tp_dist > tp_dist * 0.8:
                tp_dist = max(tp_dist, structure_tp_dist * 0.9)

        return sl_dist, tp_dist

    def _default_levels(self, price: float, side: str) -> SLTPLevels:
        """Default levels when no data is available."""
        sl_pct = 1.5
        tp1_pct = 3.0
        tp2_pct = 5.0
        sl_dist = price * sl_pct / 100
        tp1_dist = price * tp1_pct / 100
        tp2_dist = price * tp2_pct / 100

        if side == "BUY":
            return SLTPLevels(price, price - sl_dist, price + tp1_dist, price + tp2_dist,
                            sl_dist, tp1_pct / sl_pct, sl_pct, tp1_pct, tp2_pct, "default")
        else:
            return SLTPLevels(price, price + sl_dist, price - tp1_dist, price - tp2_dist,
                            sl_dist, tp1_pct / sl_pct, sl_pct, tp1_pct, tp2_pct, "default")


# Singleton
_sltp: Optional[AdaptiveSLTP] = None

def get_adaptive_sltp() -> AdaptiveSLTP:
    global _sltp
    if _sltp is None:
        _sltp = AdaptiveSLTP()
    return _sltp