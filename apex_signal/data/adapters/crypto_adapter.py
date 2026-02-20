"""
APEX SIGNAL™ — CoinGecko & CoinCap Data Adapter
Tertiary cross-validation source for cryptocurrency prices.
"""
import aiohttp
from typing import Optional, List, Dict
from datetime import datetime, timezone, timedelta

from apex_signal.data.adapters.base import BaseDataAdapter
from apex_signal.data.models import PriceTick, OHLCV, DataSource
from apex_signal.config.settings import get_settings
from apex_signal.utils.logger import get_logger

logger = get_logger("crypto_adapter")

# Mapping from common symbols to CoinGecko IDs
COINGECKO_ID_MAP: Dict[str, str] = {
    "BTC": "bitcoin",
    "ETH": "ethereum",
    "SOL": "solana",
    "ADA": "cardano",
    "DOT": "polkadot",
    "AVAX": "avalanche-2",
    "MATIC": "matic-network",
    "LINK": "chainlink",
    "XRP": "ripple",
    "DOGE": "dogecoin",
    "BNB": "binancecoin",
    "LTC": "litecoin",
    "ATOM": "cosmos",
    "UNI": "uniswap",
    "AAVE": "aave",
}

# Mapping from common symbols to CoinCap IDs
COINCAP_ID_MAP: Dict[str, str] = {
    "BTC": "bitcoin",
    "ETH": "ethereum",
    "SOL": "solana",
    "ADA": "cardano",
    "DOT": "polkadot",
    "AVAX": "avalanche",
    "MATIC": "polygon",
    "LINK": "chainlink",
    "XRP": "xrp",
    "DOGE": "dogecoin",
    "BNB": "binance-coin",
    "LTC": "litecoin",
    "ATOM": "cosmos",
    "UNI": "uniswap",
    "AAVE": "aave",
}


def _extract_base(symbol: str) -> str:
    """Extract base currency from pair like BTC/USD -> BTC."""
    return symbol.upper().replace("-", "/").split("/")[0]


class CoinGeckoAdapter(BaseDataAdapter):
    """CoinGecko data adapter — tertiary crypto verification."""

    def __init__(self):
        super().__init__(source=DataSource.COINGECKO)
        self.settings = get_settings().data
        self.base_url = self.settings.coingecko_base_url

    async def connect(self) -> None:
        timeout = aiohttp.ClientTimeout(total=self.settings.poll_timeout_seconds)
        self._session = aiohttp.ClientSession(timeout=timeout)
        logger.info("coingecko_adapter_connected")

    async def disconnect(self) -> None:
        if self._session:
            await self._session.close()
            self._session = None
        logger.info("coingecko_adapter_disconnected")

    async def get_latest_price(self, symbol: str) -> Optional[PriceTick]:
        """Fetch latest price from CoinGecko simple/price endpoint."""
        try:
            if not self._session:
                await self.connect()

            base = _extract_base(symbol)
            coin_id = COINGECKO_ID_MAP.get(base)
            if not coin_id:
                logger.warning("coingecko_unknown_symbol", symbol=symbol, base=base)
                return None

            url = f"{self.base_url}/simple/price"
            params = {
                "ids": coin_id,
                "vs_currencies": "usd",
                "include_24hr_vol": "true",
            }

            async with self._session.get(url, params=params) as resp:
                if resp.status != 200:
                    logger.warning("coingecko_price_error", status=resp.status, symbol=symbol)
                    return None
                data = await resp.json()

            coin_data = data.get(coin_id, {})
            price = coin_data.get("usd")
            if price is None:
                return None

            return PriceTick(
                symbol=symbol,
                source=DataSource.COINGECKO,
                price=float(price),
                volume=float(coin_data.get("usd_24h_vol", 0)),
                timestamp=datetime.now(timezone.utc),
                raw=coin_data,
            )
        except Exception as e:
            logger.error("coingecko_price_exception", symbol=symbol, error=str(e))
            return None

    async def get_candles(
        self, symbol: str, timeframe: str = "1m", limit: int = 200
    ) -> List[OHLCV]:
        """Fetch OHLC data from CoinGecko (limited granularity)."""
        try:
            if not self._session:
                await self.connect()

            base = _extract_base(symbol)
            coin_id = COINGECKO_ID_MAP.get(base)
            if not coin_id:
                return []

            # CoinGecko OHLC supports 1/7/14/30/90/180/365 days
            days = 1  # For minute-level data
            url = f"{self.base_url}/coins/{coin_id}/ohlc"
            params = {"vs_currency": "usd", "days": days}

            async with self._session.get(url, params=params) as resp:
                if resp.status != 200:
                    logger.warning("coingecko_candles_error", status=resp.status)
                    return []
                data = await resp.json()

            candles = []
            for item in data[-limit:]:
                ts_ms, o, h, l, c = item[0], item[1], item[2], item[3], item[4]
                candles.append(
                    OHLCV(
                        symbol=symbol,
                        source=DataSource.COINGECKO,
                        open=float(o),
                        high=float(h),
                        low=float(l),
                        close=float(c),
                        volume=0.0,  # CoinGecko OHLC doesn't include volume
                        timestamp=datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc),
                    )
                )
            return candles
        except Exception as e:
            logger.error("coingecko_candles_exception", symbol=symbol, error=str(e))
            return []


class CoinCapAdapter(BaseDataAdapter):
    """CoinCap data adapter — additional crypto cross-validation."""

    def __init__(self):
        super().__init__(source=DataSource.COINCAP)
        self.settings = get_settings().data
        self.base_url = self.settings.coincap_base_url

    async def connect(self) -> None:
        timeout = aiohttp.ClientTimeout(total=self.settings.poll_timeout_seconds)
        self._session = aiohttp.ClientSession(timeout=timeout)
        logger.info("coincap_adapter_connected")

    async def disconnect(self) -> None:
        if self._session:
            await self._session.close()
            self._session = None
        logger.info("coincap_adapter_disconnected")

    async def get_latest_price(self, symbol: str) -> Optional[PriceTick]:
        """Fetch latest price from CoinCap assets endpoint."""
        try:
            if not self._session:
                await self.connect()

            base = _extract_base(symbol)
            asset_id = COINCAP_ID_MAP.get(base)
            if not asset_id:
                logger.warning("coincap_unknown_symbol", symbol=symbol, base=base)
                return None

            url = f"{self.base_url}/assets/{asset_id}"

            async with self._session.get(url) as resp:
                if resp.status != 200:
                    logger.warning("coincap_price_error", status=resp.status, symbol=symbol)
                    return None
                data = await resp.json()

            asset_data = data.get("data", {})
            price_str = asset_data.get("priceUsd")
            if price_str is None:
                return None

            return PriceTick(
                symbol=symbol,
                source=DataSource.COINCAP,
                price=float(price_str),
                volume=float(asset_data.get("volumeUsd24Hr", 0)),
                timestamp=datetime.now(timezone.utc),
                raw=asset_data,
            )
        except Exception as e:
            logger.error("coincap_price_exception", symbol=symbol, error=str(e))
            return None

    async def get_candles(
        self, symbol: str, timeframe: str = "m1", limit: int = 200
    ) -> List[OHLCV]:
        """Fetch candle history from CoinCap."""
        try:
            if not self._session:
                await self.connect()

            base = _extract_base(symbol)
            asset_id = COINCAP_ID_MAP.get(base)
            if not asset_id:
                return []

            # CoinCap intervals: m1, m5, m15, m30, h1, h2, h6, h12, d1
            interval_map = {
                "1m": "m1", "1Min": "m1", "5m": "m5", "5Min": "m5",
                "15m": "m15", "15Min": "m15", "1h": "h1", "1H": "h1",
                "1Hour": "h1", "1d": "d1", "1D": "d1", "1Day": "d1",
            }
            interval = interval_map.get(timeframe, "m1")

            end = datetime.now(timezone.utc)
            start = end - timedelta(days=1)

            url = f"{self.base_url}/candles"
            params = {
                "exchange": "binance",
                "interval": interval,
                "baseId": asset_id,
                "quoteId": "united-states-dollar",
                "start": int(start.timestamp() * 1000),
                "end": int(end.timestamp() * 1000),
            }

            async with self._session.get(url, params=params) as resp:
                if resp.status != 200:
                    logger.warning("coincap_candles_error", status=resp.status)
                    return []
                data = await resp.json()

            candles = []
            for item in data.get("data", [])[-limit:]:
                candles.append(
                    OHLCV(
                        symbol=symbol,
                        source=DataSource.COINCAP,
                        open=float(item["open"]),
                        high=float(item["high"]),
                        low=float(item["low"]),
                        close=float(item["close"]),
                        volume=float(item.get("volume", 0)),
                        timestamp=datetime.fromtimestamp(
                            int(item["period"]) / 1000, tz=timezone.utc
                        ),
                    )
                )
            return candles
        except Exception as e:
            logger.error("coincap_candles_exception", symbol=symbol, error=str(e))
            return []