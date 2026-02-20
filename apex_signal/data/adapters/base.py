"""
APEX SIGNAL™ — Base Data Adapter Interface
All data source adapters must implement this interface.
"""
from abc import ABC, abstractmethod
from typing import Optional, List
from apex_signal.data.models import PriceTick, OHLCV, DataSource
import pandas as pd


class BaseDataAdapter(ABC):
    """Abstract base class for all market data adapters."""

    def __init__(self, source: DataSource):
        self.source = source
        self._session = None

    @abstractmethod
    async def connect(self) -> None:
        """Initialize connection / session."""
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Clean up connection / session."""
        pass

    @abstractmethod
    async def get_latest_price(self, symbol: str) -> Optional[PriceTick]:
        """Fetch the latest price tick for a symbol."""
        pass

    @abstractmethod
    async def get_candles(
        self, symbol: str, timeframe: str = "1m", limit: int = 200
    ) -> List[OHLCV]:
        """Fetch historical OHLCV candles."""
        pass

    def candles_to_dataframe(self, candles: List[OHLCV]) -> pd.DataFrame:
        """Convert list of OHLCV candles to a pandas DataFrame."""
        if not candles:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume", "timestamp"])
        data = [
            {
                "open": c.open,
                "high": c.high,
                "low": c.low,
                "close": c.close,
                "volume": c.volume,
                "timestamp": c.timestamp,
            }
            for c in candles
        ]
        df = pd.DataFrame(data)
        df.set_index("timestamp", inplace=True)
        df.sort_index(inplace=True)
        return df