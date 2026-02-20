"""
APEX SIGNAL™ — FastAPI Application
Production API with /healthz, /metrics, signal dashboard, and control endpoints.
"""
import asyncio
import uuid
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from apex_signal.config.settings import get_settings
from apex_signal.utils.logger import get_logger, setup_logging
from apex_signal.utils.helpers import utc_timestamp
from apex_signal.data.verification import get_verification_engine
from apex_signal.data.cache.price_cache import get_cache
from apex_signal.indicators.registry import get_indicator_registry
from apex_signal.strategies.registry import get_strategy_registry
from apex_signal.engines.signal_engine import get_signal_engine
from apex_signal.smart_money.detector import get_smart_money_detector
from apex_signal.telegram.notifier import get_notifier
from apex_signal.engines.risk_manager import get_risk_manager
from apex_signal.engines.signal_quality import get_performance_tracker

logger = get_logger("api")

# Application state
app_state: Dict[str, Any] = {
    "instance_id": str(uuid.uuid4())[:8],
    "started_at": None,
    "signals_generated": 0,
    "last_scan_time": None,
    "is_scanning": False,
    "errors": 0,
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan — startup and shutdown."""
    setup_logging()
    settings = get_settings()
    app_state["started_at"] = utc_timestamp()

    logger.info("apex_signal_starting",
                version=settings.version,
                instance=app_state["instance_id"],
                symbols=settings.symbols)

    # Initialize components
    verification = get_verification_engine()
    await verification.initialize()

    notifier = get_notifier()
    await notifier.initialize()

    logger.info("apex_signal_ready")

    yield

    # Shutdown
    logger.info("apex_signal_shutting_down")
    await verification.shutdown()
    await notifier.shutdown()


app = FastAPI(
    title="APEX SIGNAL™",
    description="Production-grade quantitative trading signal platform",
    version="1.0.0",
    lifespan=lifespan,
)


# ─── Health & Metrics ───────────────────────────────────────────

@app.get("/healthz", tags=["System"])
async def health_check():
    """Fast health check endpoint for load balancers and monitoring."""
    return JSONResponse(
        status_code=200,
        content={
            "status": "healthy",
            "instance": app_state["instance_id"],
            "uptime_since": app_state["started_at"],
            "timestamp": utc_timestamp(),
        },
    )


@app.get("/metrics", tags=["System"])
async def metrics():
    """Prometheus-compatible metrics endpoint."""
    settings = get_settings()
    cache = get_cache()
    indicator_reg = get_indicator_registry()
    strategy_reg = get_strategy_registry()
    notifier = get_notifier()

    return {
        "app": {
            "name": settings.app_name,
            "version": settings.version,
            "instance_id": app_state["instance_id"],
            "started_at": app_state["started_at"],
        },
        "signals": {
            "total_generated": app_state["signals_generated"],
            "last_scan_time": app_state["last_scan_time"],
            "is_scanning": app_state["is_scanning"],
            "errors": app_state["errors"],
        },
        "components": {
            "indicators_registered": indicator_reg.count,
            "strategies_registered": strategy_reg.count,
            "cache_stats": cache.stats,
            "telegram_stats": notifier.stats,
        },
        "symbols": settings.symbols,
        "timestamp": utc_timestamp(),
    }


# ─── Signal Endpoints ───────────────────────────────────────────

class SignalRequest(BaseModel):
    symbol: str
    timeframe: str = "1Min"


class ScanRequest(BaseModel):
    symbols: Optional[List[str]] = None
    min_confidence: float = 40.0
    notify: bool = True


@app.post("/api/v1/signal", tags=["Signals"])
async def generate_signal(request: SignalRequest):
    """Generate a signal for a specific symbol."""
    try:
        verification = get_verification_engine()
        df = await verification.get_candles_df(request.symbol, request.timeframe)

        if df.empty:
            raise HTTPException(status_code=404, detail=f"No data available for {request.symbol}")

        # Compute indicators
        indicator_reg = get_indicator_registry()
        enriched = indicator_reg.compute_all(df)

        # Generate signal
        signal_engine = get_signal_engine()
        signal = signal_engine.generate_signal(enriched, request.symbol)

        app_state["signals_generated"] += 1
        app_state["last_scan_time"] = utc_timestamp()

        return signal.to_dict()

    except HTTPException:
        raise
    except Exception as e:
        app_state["errors"] += 1
        logger.error("signal_generation_error", symbol=request.symbol, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/scan", tags=["Signals"])
async def scan_all_symbols(request: ScanRequest):
    """Scan all configured symbols and return signals."""
    settings = get_settings()
    symbols = request.symbols or settings.symbols
    results = []

    verification = get_verification_engine()
    indicator_reg = get_indicator_registry()
    signal_engine = get_signal_engine()
    notifier = get_notifier()

    app_state["is_scanning"] = True

    for symbol in symbols:
        try:
            df = await verification.get_candles_df(symbol)
            if df.empty:
                continue

            enriched = indicator_reg.compute_all(df)
            signal = signal_engine.generate_signal(enriched, symbol)
            signal_dict = signal.to_dict()
            results.append(signal_dict)

            app_state["signals_generated"] += 1

            # Send Telegram notification if enabled and meets threshold
            if request.notify and signal.confidence >= request.min_confidence:
                await notifier.send_signal(signal_dict)

        except Exception as e:
            app_state["errors"] += 1
            logger.error("scan_error", symbol=symbol, error=str(e))
            results.append({"symbol": symbol, "error": str(e)})

    app_state["is_scanning"] = False
    app_state["last_scan_time"] = utc_timestamp()

    return {
        "signals": results,
        "scanned": len(symbols),
        "timestamp": utc_timestamp(),
    }


# ─── Smart Money ────────────────────────────────────────────────

@app.get("/api/v1/smart-money/{symbol}", tags=["Smart Money"])
async def smart_money_analysis(symbol: str, timeframe: str = "1Min"):
    """Run smart-money detection for a symbol."""
    try:
        verification = get_verification_engine()
        df = await verification.get_candles_df(symbol, timeframe)

        if df.empty:
            raise HTTPException(status_code=404, detail=f"No data for {symbol}")

        indicator_reg = get_indicator_registry()
        enriched = indicator_reg.compute_all(df)

        detector = get_smart_money_detector()
        result = detector.detect(enriched)

        return result.to_dict()

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── Data Verification ──────────────────────────────────────────

@app.get("/api/v1/verify-price/{symbol}", tags=["Data"])
async def verify_price(symbol: str):
    """Cross-source price verification for a symbol."""
    verification = get_verification_engine()
    result = await verification.verify_price(symbol)

    if not result:
        raise HTTPException(status_code=404, detail=f"No price data for {symbol}")

    return {
        "symbol": result.symbol,
        "price": result.price,
        "sources": [s.value for s in result.sources_used],
        "source_prices": result.source_prices,
        "max_deviation_pct": round(result.max_deviation_pct, 4),
        "is_valid": result.is_valid,
        "timestamp": result.timestamp.isoformat(),
    }


@app.get("/api/v1/deviation-report", tags=["Data"])
async def deviation_report():
    """Cross-source deviation report for all symbols."""
    settings = get_settings()
    verification = get_verification_engine()
    report = await verification.get_deviation_report(settings.symbols)
    return {"report": report, "timestamp": utc_timestamp()}


# ─── Strategy Info ───────────────────────────────────────────────

@app.get("/api/v1/strategies", tags=["Strategies"])
async def list_strategies():
    """List all registered strategies grouped by family."""
    registry = get_strategy_registry()
    return {
        "total": registry.count,
        "families": registry.get_strategies_by_family(),
        "all_names": registry.strategy_names,
    }


@app.get("/api/v1/indicators", tags=["Indicators"])
async def list_indicators():
    """List all registered indicators."""
    registry = get_indicator_registry()
    return {
        "total": registry.count,
        "indicators": registry.indicator_names,
    }


# ─── Telegram Control ───────────────────────────────────────────

@app.post("/api/v1/telegram/mute/{symbol}", tags=["Telegram"])
async def mute_symbol(symbol: str):
    """Mute Telegram notifications for a symbol."""
    notifier = get_notifier()
    notifier.mute_symbol(symbol)
    return {"status": "muted", "symbol": symbol}


@app.post("/api/v1/telegram/unmute/{symbol}", tags=["Telegram"])
async def unmute_symbol(symbol: str):
    """Unmute Telegram notifications for a symbol."""
    notifier = get_notifier()
    notifier.unmute_symbol(symbol)
    return {"status": "unmuted", "symbol": symbol}


@app.get("/api/v1/telegram/stats", tags=["Telegram"])
async def telegram_stats():
    """Get Telegram notifier statistics."""
    notifier = get_notifier()
    return notifier.stats


# ─── Risk Management ────────────────────────────────────────────

@app.get("/api/v1/risk/report", tags=["Risk"])
async def risk_report():
    """Get current risk management state and report."""
    rm = get_risk_manager()
    return rm.risk_report


@app.post("/api/v1/risk/reset-kill-switch", tags=["Risk"])
async def reset_kill_switch():
    """Manually reset the kill switch after review."""
    rm = get_risk_manager()
    rm.reset_kill_switch()
    return {"status": "kill_switch_reset", "report": rm.risk_report}


# ─── Strategy Performance ───────────────────────────────────────

@app.get("/api/v1/strategy-performance", tags=["Strategies"])
async def strategy_performance():
    """Get per-strategy win rate and performance tracking."""
    tracker = get_performance_tracker()
    return {
        "strategy_stats": tracker.all_stats,
        "timestamp": utc_timestamp(),
    }