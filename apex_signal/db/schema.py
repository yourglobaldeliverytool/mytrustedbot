"""
APEX SIGNAL™ — Database Schema Design
SQLAlchemy models for signals, trades, metrics, and system state.
"""
from datetime import datetime, timezone
from sqlalchemy import (
    Column, Integer, String, Float, Boolean, DateTime, Text, JSON,
    Index, create_engine
)
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker as async_sessionmaker

Base = declarative_base()


class SignalRecord(Base):
    """Persisted signal record."""
    __tablename__ = "signals"

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False, index=True)
    side = Column(String(10), nullable=False)  # BUY, SELL, HOLD
    confidence = Column(Float, nullable=False)
    tier = Column(String(20), nullable=False)
    reason = Column(Text)
    components = Column(JSON)  # ML, confluence, SM, vol, RL scores
    engines_data = Column(JSON)  # Engine results summary
    signal_hash = Column(String(32), unique=True, index=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    __table_args__ = (
        Index("idx_signals_symbol_time", "symbol", "created_at"),
        Index("idx_signals_tier", "tier"),
    )


class TradeRecord(Base):
    """Backtest or live trade record."""
    __tablename__ = "trades"

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False, index=True)
    side = Column(String(10), nullable=False)
    entry_price = Column(Float, nullable=False)
    exit_price = Column(Float)
    entry_time = Column(DateTime, nullable=False)
    exit_time = Column(DateTime)
    pnl = Column(Float, default=0.0)
    pnl_pct = Column(Float, default=0.0)
    confidence = Column(Float)
    tier = Column(String(20))
    reason = Column(Text)
    status = Column(String(20), default="open")  # open, closed, cancelled
    source = Column(String(20), default="backtest")  # backtest, live
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    __table_args__ = (
        Index("idx_trades_symbol_time", "symbol", "entry_time"),
    )


class PerformanceMetric(Base):
    """Periodic performance metrics snapshot."""
    __tablename__ = "performance_metrics"

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False, index=True)
    period = Column(String(20), nullable=False)  # daily, weekly, monthly
    total_signals = Column(Integer, default=0)
    buy_signals = Column(Integer, default=0)
    sell_signals = Column(Integer, default=0)
    avg_confidence = Column(Float, default=0.0)
    elite_count = Column(Integer, default=0)
    strong_count = Column(Integer, default=0)
    win_rate = Column(Float, default=0.0)
    total_return_pct = Column(Float, default=0.0)
    sharpe_ratio = Column(Float, default=0.0)
    max_drawdown_pct = Column(Float, default=0.0)
    computed_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    __table_args__ = (
        Index("idx_perf_symbol_period", "symbol", "period"),
    )


class SystemState(Base):
    """System state for leader election and singleton enforcement."""
    __tablename__ = "system_state"

    id = Column(Integer, primary_key=True, autoincrement=True)
    instance_id = Column(String(64), nullable=False, unique=True)
    is_leader = Column(Boolean, default=False)
    last_heartbeat = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    started_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    metadata_ = Column("metadata", JSON)


class DataDeviationLog(Base):
    """Log of cross-source price deviations."""
    __tablename__ = "deviation_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False, index=True)
    source_prices = Column(JSON)
    max_deviation_pct = Column(Float)
    is_valid = Column(Boolean)
    logged_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class MLModelVersion(Base):
    """ML model version tracking."""
    __tablename__ = "ml_model_versions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    version = Column(String(32), nullable=False, unique=True)
    model_type = Column(String(32))  # lightgbm, random_forest, etc.
    metrics = Column(JSON)
    feature_count = Column(Integer)
    training_samples = Column(Integer)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


async def init_db(db_url: str) -> AsyncSession:
    """Initialize database and create all tables."""
    engine = create_async_engine(db_url, echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    return session_factory


def init_db_sync(db_url: str = "sqlite:///apex_signal.db"):
    """Synchronous DB init for testing."""
    engine = create_engine(db_url, echo=False)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)