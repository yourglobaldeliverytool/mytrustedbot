"""
APEX SIGNAL™ — Price Cache Layer
In-memory cache with TTL to reduce latency and prevent flickering signals.
"""
from typing import Optional, Dict, Any, List
from datetime import datetime, timezone
from cachetools import TTLCache
import pandas as pd

from apex_signal.data.models import PriceTick, OHLCV, VerifiedPrice
from apex_signal.config.settings import get_settings
from apex_signal.utils.logger import get_logger

logger = get_logger("price_cache")


class PriceCache:
    """Thread-safe in-memory cache for market data with TTL expiration."""

    def __init__(self):
        settings = get_settings().data
        ttl = settings.cache_ttl_seconds
        # Cache for verified prices
        self._price_cache: TTLCache = TTLCache(maxsize=500, ttl=ttl)
        # Cache for candle DataFrames
        self._candle_cache: TTLCache = TTLCache(maxsize=100, ttl=ttl * 4)
        # Cache for raw ticks per source
        self._tick_cache: TTLCache = TTLCache(maxsize=2000, ttl=ttl)
        # Historical verified prices for analysis
        self._history: Dict[str, List[VerifiedPrice]] = {}
        self._max_history = 1000

    def put_tick(self, tick: PriceTick) -> None:
        """Store a raw price tick from a specific source."""
        key = f"{tick.symbol}:{tick.source.value}"
        self._tick_cache[key] = tick

    def get_tick(self, symbol: str, source: str) -> Optional[PriceTick]:
        """Retrieve the latest tick for a symbol from a specific source."""
        key = f"{symbol}:{source}"
        return self._tick_cache.get(key)

    def get_all_ticks(self, symbol: str) -> Dict[str, PriceTick]:
        """Get all cached ticks for a symbol across all sources."""
        result = {}
        for key, tick in self._tick_cache.items():
            if key.startswith(f"{symbol}:"):
                source = key.split(":")[1]
                result[source] = tick
        return result

    def put_verified_price(self, vp: VerifiedPrice) -> None:
        """Store a cross-source verified price."""
        self._price_cache[vp.symbol] = vp
        # Also append to history
        if vp.symbol not in self._history:
            self._history[vp.symbol] = []
        self._history[vp.symbol].append(vp)
        if len(self._history[vp.symbol]) > self._max_history:
            self._history[vp.symbol] = self._history[vp.symbol][-self._max_history:]

    def get_verified_price(self, symbol: str) -> Optional[VerifiedPrice]:
        """Retrieve the latest verified price for a symbol."""
        return self._price_cache.get(symbol)

    def put_candles(self, symbol: str, timeframe: str, df: pd.DataFrame) -> None:
        """Cache a candle DataFrame."""
        key = f"{symbol}:{timeframe}"
        self._candle_cache[key] = df

    def get_candles(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """Retrieve cached candle DataFrame."""
        key = f"{symbol}:{timeframe}"
        return self._candle_cache.get(key)

    def get_price_history(self, symbol: str, limit: int = 100) -> List[VerifiedPrice]:
        """Get recent verified price history for a symbol."""
        history = self._history.get(symbol, [])
        return history[-limit:]

    def clear(self) -> None:
        """Clear all caches."""
        self._price_cache.clear()
        self._candle_cache.clear()
        self._tick_cache.clear()
        self._history.clear()
        logger.info("cache_cleared")

    @property
    def stats(self) -> Dict[str, Any]:
        """Return cache statistics."""
        return {
            "price_entries": len(self._price_cache),
            "candle_entries": len(self._candle_cache),
            "tick_entries": len(self._tick_cache),
            "history_symbols": len(self._history),
            "total_history_points": sum(len(v) for v in self._history.values()),
        }


# Singleton instance
_cache: Optional[PriceCache] = None


def get_cache() -> PriceCache:
    global _cache
    if _cache is None:
        _cache = PriceCache()
    return _cache