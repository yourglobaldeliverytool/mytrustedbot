"""
APEX SIGNAL™ — Polygon.io Data Adapter
Secondary cross-check source for verified OHLCV and spread tracking.
"""
import aiohttp
from typing import Optional, List
from datetime import datetime, timezone, timedelta

from apex_signal.data.adapters.base import BaseDataAdapter
from apex_signal.data.models import PriceTick, OHLCV, DataSource
from apex_signal.config.settings import get_settings
from apex_signal.utils.logger import get_logger
from apex_signal.utils.helpers import normalize_symbol, is_crypto

logger = get_logger("polygon_adapter")


class PolygonAdapter(BaseDataAdapter):
    """Polygon.io data adapter — secondary verification source."""

    def __init__(self):
        super().__init__(source=DataSource.POLYGON)
        self.settings = get_settings().data
        self.api_key = self.settings.polygon_api_key
        self.base_url = self.settings.polygon_base_url

    async def connect(self) -> None:
        timeout = aiohttp.ClientTimeout(total=self.settings.poll_timeout_seconds)
        self._session = aiohttp.ClientSession(timeout=timeout)
        logger.info("polygon_adapter_connected")

    async def disconnect(self) -> None:
        if self._session:
            await self._session.close()
            self._session = None
        logger.info("polygon_adapter_disconnected")

    def _format_ticker(self, symbol: str) -> str:
        """Format symbol for Polygon API."""
        if is_crypto(symbol):
            normalized = normalize_symbol(symbol)
            # Polygon crypto format: X:BTCUSD
            return f"X:{normalized}"
        return normalize_symbol(symbol)

    async def get_latest_price(self, symbol: str) -> Optional[PriceTick]:
        """Fetch latest price snapshot from Polygon."""
        try:
            if not self._session:
                await self.connect()

            ticker = self._format_ticker(symbol)
            url = f"{self.base_url}/v2/last/trade/{ticker}"
            params = {"apiKey": self.api_key}

            async with self._session.get(url, params=params) as resp:
                if resp.status != 200:
                    logger.warning("polygon_price_error", status=resp.status, symbol=symbol)
                    return None
                data = await resp.json()

            results = data.get("results", data.get("result", {}))
            if not results:
                # Try snapshot endpoint as fallback
                return await self._get_snapshot_price(symbol)

            price = float(results.get("p", results.get("price", 0)))
            if price <= 0:
                return await self._get_snapshot_price(symbol)

            return PriceTick(
                symbol=symbol,
                source=DataSource.POLYGON,
                price=price,
                volume=float(results.get("s", results.get("size", 0))),
                timestamp=datetime.now(timezone.utc),
                raw=results,
            )
        except Exception as e:
            logger.error("polygon_price_exception", symbol=symbol, error=str(e))
            return None

    async def _get_snapshot_price(self, symbol: str) -> Optional[PriceTick]:
        """Fallback: get price from snapshot endpoint."""
        try:
            ticker = self._format_ticker(symbol)
            if is_crypto(symbol):
                url = f"{self.base_url}/v2/snapshot/locale/global/markets/crypto/tickers/{ticker}"
            else:
                url = f"{self.base_url}/v2/snapshot/locale/us/markets/stocks/tickers/{ticker}"
            params = {"apiKey": self.api_key}

            async with self._session.get(url, params=params) as resp:
                if resp.status != 200:
                    return None
                data = await resp.json()

            ticker_data = data.get("ticker", {})
            last_trade = ticker_data.get("lastTrade", {})
            price = float(last_trade.get("p", 0))
            if price <= 0:
                day = ticker_data.get("day", {})
                price = float(day.get("c", 0))

            if price <= 0:
                return None

            return PriceTick(
                symbol=symbol,
                source=DataSource.POLYGON,
                price=price,
                timestamp=datetime.now(timezone.utc),
                raw=ticker_data,
            )
        except Exception as e:
            logger.error("polygon_snapshot_exception", symbol=symbol, error=str(e))
            return None

    async def get_candles(
        self, symbol: str, timeframe: str = "1", limit: int = 200
    ) -> List[OHLCV]:
        """Fetch aggregate bars from Polygon."""
        try:
            if not self._session:
                await self.connect()

            ticker = self._format_ticker(symbol)
            end = datetime.now(timezone.utc)
            start = end - timedelta(days=5)

            # Map timeframe to Polygon multiplier/timespan
            tf_map = {
                "1": ("1", "minute"),
                "1m": ("1", "minute"),
                "1Min": ("1", "minute"),
                "5": ("5", "minute"),
                "5m": ("5", "minute"),
                "5Min": ("5", "minute"),
                "15": ("15", "minute"),
                "15m": ("15", "minute"),
                "1h": ("1", "hour"),
                "1H": ("1", "hour"),
                "1Hour": ("1", "hour"),
                "1d": ("1", "day"),
                "1D": ("1", "day"),
                "1Day": ("1", "day"),
            }
            multiplier, timespan = tf_map.get(timeframe, ("1", "minute"))

            url = (
                f"{self.base_url}/v2/aggs/ticker/{ticker}"
                f"/range/{multiplier}/{timespan}"
                f"/{start.strftime('%Y-%m-%d')}/{end.strftime('%Y-%m-%d')}"
            )
            params = {
                "apiKey": self.api_key,
                "adjusted": "true",
                "sort": "asc",
                "limit": limit,
            }

            async with self._session.get(url, params=params) as resp:
                if resp.status != 200:
                    logger.warning("polygon_candles_error", status=resp.status, symbol=symbol)
                    return []
                data = await resp.json()

            results = data.get("results", [])
            candles = []
            for bar in results[-limit:]:
                ts = bar.get("t", 0)
                if isinstance(ts, (int, float)):
                    ts_dt = datetime.fromtimestamp(ts / 1000, tz=timezone.utc)
                else:
                    ts_dt = datetime.now(timezone.utc)

                candles.append(
                    OHLCV(
                        symbol=symbol,
                        source=DataSource.POLYGON,
                        open=float(bar["o"]),
                        high=float(bar["h"]),
                        low=float(bar["l"]),
                        close=float(bar["c"]),
                        volume=float(bar.get("v", 0)),
                        timestamp=ts_dt,
                    )
                )
            return candles
        except Exception as e:
            logger.error("polygon_candles_exception", symbol=symbol, error=str(e))
            return []