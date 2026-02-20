"""
APEX SIGNAL™ — Central Configuration
All settings are loaded from environment variables with sensible defaults.
"""
from pydantic_settings import BaseSettings
from pydantic import Field
from typing import List, Optional
import os


class DataSourceSettings(BaseSettings):
    """Data source API keys and endpoints."""
    alpaca_api_key: str = Field(default="", env="ALPACA_API_KEY")
    alpaca_secret_key: str = Field(default="", env="ALPACA_SECRET_KEY")
    alpaca_base_url: str = Field(default="https://paper-api.alpaca.markets", env="ALPACA_BASE_URL")
    alpaca_data_url: str = Field(default="https://data.alpaca.markets", env="ALPACA_DATA_URL")

    polygon_api_key: str = Field(default="", env="POLYGON_API_KEY")
    polygon_base_url: str = Field(default="https://api.polygon.io", env="POLYGON_BASE_URL")

    coingecko_base_url: str = Field(default="https://api.coingecko.com/api/v3", env="COINGECKO_BASE_URL")
    coincap_base_url: str = Field(default="https://api.coincap.io/v2", env="COINCAP_BASE_URL")

    poll_timeout_seconds: float = Field(default=5.0, env="POLL_TIMEOUT_SECONDS")
    deviation_threshold: float = Field(default=0.005, env="DEVIATION_THRESHOLD")  # 0.5%
    cache_ttl_seconds: int = Field(default=30, env="CACHE_TTL_SECONDS")

    class Config:
        env_file = ".env"
        extra = "ignore"


class IndicatorSettings(BaseSettings):
    """Indicator computation parameters."""
    ema_periods: List[int] = [8, 20, 50, 200]
    sma_periods: List[int] = [20, 50, 100]
    rsi_period: int = 14
    atr_period: int = 14
    bb_period: int = 20
    bb_std: float = 2.0
    adx_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    keltner_period: int = 20
    keltner_atr_mult: float = 1.5
    donchian_period: int = 20
    ichimoku_tenkan: int = 9
    ichimoku_kijun: int = 26
    ichimoku_senkou_b: int = 52
    williams_period: int = 14
    stoch_k_period: int = 14
    stoch_d_period: int = 3
    cci_period: int = 20
    zscore_period: int = 20

    class Config:
        env_file = ".env"
        extra = "ignore"


class SignalSettings(BaseSettings):
    """Signal engine weights and thresholds."""
    ml_weight: float = 0.40
    confluence_weight: float = 0.20
    smart_money_weight: float = 0.20
    volatility_weight: float = 0.10
    rl_weight: float = 0.10

    tier_elite_min: int = 80
    tier_strong_min: int = 60
    tier_moderate_min: int = 40
    # Below moderate = Weak

    class Config:
        env_file = ".env"
        extra = "ignore"


class TelegramSettings(BaseSettings):
    """Telegram bot configuration."""
    bot_token: str = Field(default="", env="TELEGRAM_BOT_TOKEN")
    chat_id: str = Field(default="", env="TELEGRAM_CHAT_ID")
    rate_limit_per_second: float = Field(default=1.0, env="TG_RATE_LIMIT")
    quiet_hours_start: int = Field(default=2, env="TG_QUIET_START")  # UTC hour
    quiet_hours_end: int = Field(default=6, env="TG_QUIET_END")  # UTC hour
    max_retries: int = Field(default=3, env="TG_MAX_RETRIES")
    retry_delay: float = Field(default=2.0, env="TG_RETRY_DELAY")

    class Config:
        env_file = ".env"
        extra = "ignore"


class DatabaseSettings(BaseSettings):
    """Database configuration."""
    db_url: str = Field(default="sqlite+aiosqlite:///apex_signal.db", env="DATABASE_URL")
    echo_sql: bool = Field(default=False, env="DB_ECHO_SQL")

    class Config:
        env_file = ".env"
        extra = "ignore"


class MLSettings(BaseSettings):
    """Machine Learning configuration."""
    model_dir: str = Field(default="ml/models", env="ML_MODEL_DIR")
    train_lookback_days: int = Field(default=90, env="ML_TRAIN_LOOKBACK")
    retrain_interval_hours: int = Field(default=24, env="ML_RETRAIN_INTERVAL")
    tp_pips: float = Field(default=20.0, env="ML_TP_PIPS")
    sl_pips: float = Field(default=10.0, env="ML_SL_PIPS")

    class Config:
        env_file = ".env"
        extra = "ignore"


class RLSettings(BaseSettings):
    """Reinforcement Learning configuration."""
    learning_rate: float = Field(default=0.001, env="RL_LEARNING_RATE")
    gamma: float = Field(default=0.99, env="RL_GAMMA")
    episodes: int = Field(default=500, env="RL_EPISODES")
    risk_penalty: float = Field(default=0.1, env="RL_RISK_PENALTY")
    drawdown_penalty: float = Field(default=0.2, env="RL_DRAWDOWN_PENALTY")

    class Config:
        env_file = ".env"
        extra = "ignore"


class BacktestSettings(BaseSettings):
    """Backtesting configuration."""
    initial_capital: float = Field(default=100000.0, env="BT_INITIAL_CAPITAL")
    commission_pct: float = Field(default=0.001, env="BT_COMMISSION_PCT")
    slippage_pct: float = Field(default=0.0005, env="BT_SLIPPAGE_PCT")
    position_size_pct: float = Field(default=0.02, env="BT_POSITION_SIZE_PCT")

    class Config:
        env_file = ".env"
        extra = "ignore"


class AppSettings(BaseSettings):
    """Top-level application settings."""
    app_name: str = "APEX SIGNAL™"
    version: str = "1.0.0"
    debug: bool = Field(default=False, env="DEBUG")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=8000, env="PORT")

    symbols: List[str] = Field(
        default=["BTC/USD", "ETH/USD", "SPY", "AAPL", "TSLA"],
        env="SYMBOLS"
    )
    scan_interval_seconds: float = Field(default=10.0, env="SCAN_INTERVAL")

    data: DataSourceSettings = DataSourceSettings()
    indicators: IndicatorSettings = IndicatorSettings()
    signals: SignalSettings = SignalSettings()
    telegram: TelegramSettings = TelegramSettings()
    database: DatabaseSettings = DatabaseSettings()
    ml: MLSettings = MLSettings()
    rl: RLSettings = RLSettings()
    backtest: BacktestSettings = BacktestSettings()

    class Config:
        env_file = ".env"
        extra = "ignore"


# Singleton
_settings: Optional[AppSettings] = None


def get_settings() -> AppSettings:
    global _settings
    if _settings is None:
        _settings = AppSettings()
    return _settings