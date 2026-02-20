"""
APEX SIGNAL™ — Base Indicator Interface
All indicators must implement calculate() and reset().
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import pandas as pd


class BaseIndicator(ABC):
    """Abstract base class for all technical indicators."""

    def __init__(self, name: str, params: Optional[Dict[str, Any]] = None):
        self.name = name
        self.params = params or {}
        self._last_result: Optional[pd.DataFrame] = None

    @abstractmethod
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate indicator values and add columns to the DataFrame.
        Must return the DataFrame with new columns added.
        The input DataFrame has columns: open, high, low, close, volume
        """
        pass

    def reset(self) -> None:
        """Reset any internal state."""
        self._last_result = None

    @property
    def last_result(self) -> Optional[pd.DataFrame]:
        return self._last_result

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name}, params={self.params})"