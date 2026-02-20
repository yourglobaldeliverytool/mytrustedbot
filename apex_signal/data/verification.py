"""
APEX SIGNAL™ — Cross-Source Price Verification Engine
Validates prices across multiple sources, rejects signals if deviation > 0.5%.
"""
import asyncio
from typing import List, Optional, Dict
from datetime import datetime, timezone

from apex_signal.data.models import PriceTick, VerifiedPrice, DataSource
from apex_signal.data.adapters.base import BaseDataAdapter
from apex_signal.data.adapters.alpaca_adapter import AlpacaAdapter
from apex_signal.data.adapters.polygon_adapter import PolygonAdapter
from apex_signal.data.adapters.crypto_adapter import CoinGeckoAdapter, CoinCapAdapter
from apex_signal.data.cache.price_cache import get_cache
from apex_signal.config.settings import get_settings
from apex_signal.utils.logger import get_logger
from apex_signal.utils.helpers import is_crypto

logger = get_logger("verification")


class PriceVerificationEngine:
    """
    Cross-source price verification engine.
    Polls all sources asynchronously, validates deviation, and produces
    a VerifiedPrice or rejects the data point.
    """

    def __init__(self):
        self.settings = get_settings().data
        self.cache = get_cache()
        self._adapters: List[BaseDataAdapter] = []
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize all data adapters."""
        if self._initialized:
            return

        # Always include Alpaca and Polygon
        self._adapters = [
            AlpacaAdapter(),
            PolygonAdapter(),
            CoinGeckoAdapter(),
            CoinCapAdapter(),
        ]

        for adapter in self._adapters:
            try:
                await adapter.connect()
            except Exception as e:
                logger.warning("adapter_connect_failed", adapter=adapter.source.value, error=str(e))

        self._initialized = True
        logger.info("verification_engine_initialized", adapters=len(self._adapters))

    async def shutdown(self) -> None:
        """Disconnect all adapters."""
        for adapter in self._adapters:
            try:
                await adapter.disconnect()
            except Exception:
                pass
        self._initialized = False

    def _get_adapters_for_symbol(self, symbol: str) -> List[BaseDataAdapter]:
        """Return appropriate adapters based on symbol type."""
        if is_crypto(symbol):
            # Use all adapters for crypto
            return self._adapters
        else:
            # Use only Alpaca and Polygon for stocks
            return [a for a in self._adapters if a.source in (DataSource.ALPACA, DataSource.POLYGON)]

    async def verify_price(self, symbol: str) -> Optional[VerifiedPrice]:
        """
        Fetch price from all relevant sources, cross-validate, and return
        a VerifiedPrice if deviation is within threshold.
        """
        if not self._initialized:
            await self.initialize()

        adapters = self._get_adapters_for_symbol(symbol)

        # Poll all sources concurrently with timeout
        tasks = [adapter.get_latest_price(symbol) for adapter in adapters]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Collect valid ticks
        valid_ticks: List[PriceTick] = []
        for result in results:
            if isinstance(result, PriceTick) and result.price > 0:
                valid_ticks.append(result)
                self.cache.put_tick(result)

        if not valid_ticks:
            logger.warning("no_valid_prices", symbol=symbol)
            return None

        # Calculate consensus price and deviation
        prices = [t.price for t in valid_ticks]
        avg_price = sum(prices) / len(prices)
        source_prices = {t.source.value: t.price for t in valid_ticks}
        sources_used = [t.source for t in valid_ticks]

        # Calculate max deviation from average
        if len(prices) > 1:
            max_dev = max(abs(p - avg_price) / avg_price for p in prices)
        else:
            max_dev = 0.0

        is_valid = max_dev <= self.settings.deviation_threshold

        if not is_valid:
            logger.warning(
                "price_deviation_exceeded",
                symbol=symbol,
                max_deviation=f"{max_dev:.4%}",
                threshold=f"{self.settings.deviation_threshold:.4%}",
                source_prices=source_prices,
            )

        verified = VerifiedPrice(
            symbol=symbol,
            price=avg_price,
            sources_used=sources_used,
            source_prices=source_prices,
            max_deviation_pct=max_dev * 100,
            is_valid=is_valid,
            timestamp=datetime.now(timezone.utc),
        )

        self.cache.put_verified_price(verified)
        return verified

    async def get_candles_df(self, symbol: str, timeframe: str = "1Min", limit: int = 200):
        """
        Fetch candles from the primary adapter and return as DataFrame.
        Falls back to secondary sources if primary fails.
        Uses cache when available.
        """
        import pandas as pd

        # Check cache first
        cached = self.cache.get_candles(symbol, timeframe)
        if cached is not None and len(cached) > 0:
            return cached

        if not self._initialized:
            await self.initialize()

        adapters = self._get_adapters_for_symbol(symbol)

        for adapter in adapters:
            try:
                candles = await adapter.get_candles(symbol, timeframe, limit)
                if candles:
                    df = adapter.candles_to_dataframe(candles)
                    if len(df) > 0:
                        self.cache.put_candles(symbol, timeframe, df)
                        logger.info(
                            "candles_fetched",
                            symbol=symbol,
                            source=adapter.source.value,
                            count=len(df),
                        )
                        return df
            except Exception as e:
                logger.warning(
                    "candles_fetch_failed",
                    symbol=symbol,
                    source=adapter.source.value,
                    error=str(e),
                )
                continue

        logger.warning("no_candles_available", symbol=symbol)
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    async def get_deviation_report(self, symbols: List[str]) -> Dict[str, Dict]:
        """Generate a deviation report across all symbols for monitoring."""
        report = {}
        for symbol in symbols:
            vp = await self.verify_price(symbol)
            if vp:
                report[symbol] = {
                    "price": vp.price,
                    "sources": len(vp.sources_used),
                    "max_deviation_pct": round(vp.max_deviation_pct, 4),
                    "is_valid": vp.is_valid,
                    "source_prices": vp.source_prices,
                }
            else:
                report[symbol] = {"error": "no_data"}
        return report


# Singleton
_engine: Optional[PriceVerificationEngine] = None


def get_verification_engine() -> PriceVerificationEngine:
    global _engine
    if _engine is None:
        _engine = PriceVerificationEngine()
    return _engine