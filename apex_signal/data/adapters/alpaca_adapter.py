"""
APEX SIGNAL™ — Alpaca Data Adapter
Primary live feed for stocks and crypto via Alpaca Markets API.
"""
import aiohttp
from typing import Optional, List
from datetime import datetime, timezone, timedelta

from apex_signal.data.adapters.base import BaseDataAdapter
from apex_signal.data.models import PriceTick, OHLCV, DataSource
from apex_signal.config.settings import get_settings
from apex_signal.utils.logger import get_logger
from apex_signal.utils.helpers import is_crypto, normalize_symbol

logger = get_logger("alpaca_adapter")


class AlpacaAdapter(BaseDataAdapter):
    """Alpaca Markets data adapter — primary live feed."""

    def __init__(self):
        super().__init__(source=DataSource.ALPACA)
        self.settings = get_settings().data
        self._headers = {
            "APCA-API-KEY-ID": self.settings.alpaca_api_key,
            "APCA-API-SECRET-KEY": self.settings.alpaca_secret_key,
        }

    async def connect(self) -> None:
        timeout = aiohttp.ClientTimeout(total=self.settings.poll_timeout_seconds)
        self._session = aiohttp.ClientSession(headers=self._headers, timeout=timeout)
        logger.info("alpaca_adapter_connected")

    async def disconnect(self) -> None:
        if self._session:
            await self._session.close()
            self._session = None
        logger.info("alpaca_adapter_disconnected")

    async def get_latest_price(self, symbol: str) -> Optional[PriceTick]:
        """Fetch latest quote from Alpaca."""
        try:
            if not self._session:
                await self.connect()

            normalized = normalize_symbol(symbol)
            if is_crypto(symbol):
                # Crypto endpoint
                url = f"{self.settings.alpaca_data_url}/v1beta3/crypto/us/latest/quotes"
                params = {"symbols": normalized}
            else:
                # Stock endpoint
                url = f"{self.settings.alpaca_data_url}/v2/stocks/{normalized}/quotes/latest"
                params = {}

            async with self._session.get(url, params=params) as resp:
                if resp.status != 200:
                    logger.warning("alpaca_price_error", status=resp.status, symbol=symbol)
                    return None
                data = await resp.json()

            if is_crypto(symbol):
                quote = data.get("quotes", {}).get(normalized, {})
            else:
                quote = data.get("quote", data)

            if not quote:
                return None

            bid = float(quote.get("bp", 0) or quote.get("bid_price", 0) or 0)
            ask = float(quote.get("ap", 0) or quote.get("ask_price", 0) or 0)
            mid_price = (bid + ask) / 2 if bid and ask else bid or ask

            return PriceTick(
                symbol=symbol,
                source=DataSource.ALPACA,
                price=mid_price,
                bid=bid,
                ask=ask,
                timestamp=datetime.now(timezone.utc),
            )
        except Exception as e:
            logger.error("alpaca_price_exception", symbol=symbol, error=str(e))
            return None

    async def get_candles(
        self, symbol: str, timeframe: str = "1Min", limit: int = 200
    ) -> List[OHLCV]:
        """Fetch historical bars from Alpaca."""
        try:
            if not self._session:
                await self.connect()

            normalized = normalize_symbol(symbol)
            end = datetime.now(timezone.utc)
            start = end - timedelta(days=5)

            if is_crypto(symbol):
                url = f"{self.settings.alpaca_data_url}/v1beta3/crypto/us/bars"
                params = {
                    "symbols": normalized,
                    "timeframe": timeframe,
                    "start": start.isoformat(),
                    "end": end.isoformat(),
                    "limit": limit,
                }
            else:
                url = f"{self.settings.alpaca_data_url}/v2/stocks/{normalized}/bars"
                params = {
                    "timeframe": timeframe,
                    "start": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "end": end.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "limit": limit,
                    "adjustment": "raw",
                }

            async with self._session.get(url, params=params) as resp:
                if resp.status != 200:
                    logger.warning("alpaca_candles_error", status=resp.status, symbol=symbol)
                    return []
                data = await resp.json()

            if is_crypto(symbol):
                bars = data.get("bars", {}).get(normalized, [])
            else:
                bars = data.get("bars", [])

            candles = []
            for bar in bars[-limit:]:
                candles.append(
                    OHLCV(
                        symbol=symbol,
                        source=DataSource.ALPACA,
                        open=float(bar["o"]),
                        high=float(bar["h"]),
                        low=float(bar["l"]),
                        close=float(bar["c"]),
                        volume=float(bar["v"]),
                        timestamp=datetime.fromisoformat(
                            bar["t"].replace("Z", "+00:00")
                        ) if isinstance(bar["t"], str) else bar["t"],
                    )
                )
            return candles
        except Exception as e:
            logger.error("alpaca_candles_exception", symbol=symbol, error=str(e))
            return []