"""
APEX SIGNAL™ — Data Models for Market Data
Canonical data structures used across the entire platform.
"""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


class DataSource(str, Enum):
    ALPACA = "alpaca"
    POLYGON = "polygon"
    COINGECKO = "coingecko"
    COINCAP = "coincap"


class PriceTick(BaseModel):
    """Single price observation from a data source."""
    symbol: str
    source: DataSource
    price: float
    bid: Optional[float] = None
    ask: Optional[float] = None
    volume: Optional[float] = None
    timestamp: datetime
    raw: Optional[Dict[str, Any]] = None


class OHLCV(BaseModel):
    """Single OHLCV candle."""
    symbol: str
    source: DataSource
    open: float
    high: float
    low: float
    close: float
    volume: float
    timestamp: datetime


class VerifiedPrice(BaseModel):
    """Cross-source verified price with deviation metrics."""
    symbol: str
    price: float
    sources_used: List[DataSource]
    source_prices: Dict[str, float]
    max_deviation_pct: float
    is_valid: bool
    timestamp: datetime


class MarketSnapshot(BaseModel):
    """Complete market snapshot for a symbol."""
    symbol: str
    verified_price: Optional[VerifiedPrice] = None
    candles: Optional[List[OHLCV]] = None
    latest_tick: Optional[PriceTick] = None
    timestamp: datetime