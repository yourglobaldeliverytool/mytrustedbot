"""
APEX SIGNAL™ — Base Strategy Interface
All strategies must return a standardized signal dict.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from datetime import datetime, timezone
import pandas as pd


class StrategySignal:
    """Standardized strategy signal output."""

    def __init__(
        self,
        strategy_name: str,
        signal: str,  # BUY, SELL, HOLD
        confidence: float,  # 0-100
        reason: str,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.strategy_name = strategy_name
        self.signal = signal
        self.confidence = max(0.0, min(100.0, confidence))
        self.reason = reason
        self.metadata = metadata or {}
        self.timestamp = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy_name": self.strategy_name,
            "signal": self.signal,
            "confidence": round(self.confidence, 2),
            "reason": self.reason,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
        }

    def __repr__(self) -> str:
        return f"StrategySignal({self.strategy_name}: {self.signal} @ {self.confidence:.0f}%)"


class BaseStrategy(ABC):
    """Abstract base class for all trading strategies."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def evaluate(self, df: pd.DataFrame, symbol: str = "") -> StrategySignal:
        """
        Evaluate the strategy on indicator-enriched DataFrame.
        Must return a StrategySignal.
        """
        pass

    def _hold(self, reason: str = "No clear signal") -> StrategySignal:
        """Convenience method to return a HOLD signal."""
        return StrategySignal(
            strategy_name=self.name,
            signal="HOLD",
            confidence=0.0,
            reason=reason,
        )

    def _safe_last(self, series: pd.Series, default: float = 0.0) -> float:
        """Safely get the last value of a series."""
        if series is None or series.empty:
            return default
        val = series.iloc[-1]
        if pd.isna(val):
            return default
        return float(val)

    def _safe_prev(self, series: pd.Series, offset: int = 1, default: float = 0.0) -> float:
        """Safely get a previous value from a series."""
        if series is None or len(series) < offset + 1:
            return default
        val = series.iloc[-(offset + 1)]
        if pd.isna(val):
            return default
        return float(val)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name})"