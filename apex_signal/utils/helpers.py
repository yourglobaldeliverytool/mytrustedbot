"""
APEX SIGNAL™ — Common Utility Functions
"""
from datetime import datetime, timezone
from typing import Any, Dict, Optional
import hashlib
import json


def utc_now() -> datetime:
    """Return current UTC datetime."""
    return datetime.now(timezone.utc)


def utc_timestamp() -> str:
    """Return current UTC timestamp as ISO string."""
    return utc_now().isoformat()


def classify_tier(confidence: float) -> str:
    """Classify confidence score into risk tier."""
    if confidence >= 80:
        return "Elite"
    elif confidence >= 60:
        return "Strong"
    elif confidence >= 40:
        return "Moderate"
    else:
        return "Weak"


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safe division avoiding ZeroDivisionError."""
    if denominator == 0:
        return default
    return numerator / denominator


def clamp(value: float, min_val: float = 0.0, max_val: float = 100.0) -> float:
    """Clamp a value between min and max."""
    return max(min_val, min(max_val, value))


def hash_signal(signal_dict: Dict[str, Any]) -> str:
    """Generate a unique hash for a signal to prevent duplicates."""
    payload = json.dumps(signal_dict, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def normalize_symbol(symbol: str) -> str:
    """Normalize symbol format: BTC/USD -> BTCUSD, SPY -> SPY."""
    return symbol.replace("/", "").replace("-", "").upper()


def is_crypto(symbol: str) -> bool:
    """Check if a symbol is a cryptocurrency pair."""
    crypto_bases = ["BTC", "ETH", "SOL", "ADA", "DOT", "AVAX", "MATIC", "LINK", "XRP", "DOGE"]
    upper = symbol.upper().replace("/", "").replace("-", "")
    return any(upper.startswith(c) for c in crypto_bases)


def pct_change(old_val: float, new_val: float) -> float:
    """Calculate percentage change between two values."""
    if old_val == 0:
        return 0.0
    return ((new_val - old_val) / abs(old_val)) * 100.0


def format_confidence(confidence: float) -> str:
    """Format confidence with tier label."""
    tier = classify_tier(confidence)
    return f"{confidence:.0f} ({tier})"