"""
APEX SIGNAL™ — Indicator Registry
Central registry that manages all indicators and computes them in a single pass.
"""
import pandas as pd
from typing import Dict, List, Optional
from apex_signal.indicators.base import BaseIndicator
from apex_signal.indicators.trend import EMAIndicator, SMAIndicator
from apex_signal.indicators.volume import OBVIndicator, ChaikinMoneyFlowIndicator, RelativeVolumeIndicator
from apex_signal.indicators.momentum import RSIIndicator, StochasticOscillator, CCIIndicator
from apex_signal.indicators.volatility import ATRIndicator, BollingerBandsIndicator, KeltnerChannelIndicator
from apex_signal.indicators.directional import ADXIndicator
from apex_signal.indicators.breakout import DonchianChannelIndicator
from apex_signal.indicators.composite import IchimokuIndicator, VWAPIndicator
from apex_signal.indicators.oscillators import WilliamsRIndicator, MACDIndicator
from apex_signal.indicators.structural import VolatilityRegimeDetector, MarketStructureBands
from apex_signal.indicators.quant import ZScoreIndicator, OpenRangeBreakoutIndicator, PreMarketSkewIndicator
from apex_signal.indicators.divergence import DivergenceDetector
from apex_signal.config.settings import get_settings
from apex_signal.utils.logger import get_logger

logger = get_logger("indicator_registry")


class IndicatorRegistry:
    """
    Central registry for all technical indicators.
    Computes all indicators on a DataFrame in a single optimized pass.
    """

    def __init__(self):
        self.settings = get_settings().indicators
        self._indicators: Dict[str, BaseIndicator] = {}
        self._register_all()

    def _register_all(self) -> None:
        """Register all standard indicators with configured parameters."""
        indicators = [
            # Trend (2 indicators, multiple periods each)
            EMAIndicator(periods=self.settings.ema_periods),
            SMAIndicator(periods=self.settings.sma_periods),
            # Volume (3 indicators)
            OBVIndicator(),
            ChaikinMoneyFlowIndicator(),
            RelativeVolumeIndicator(),
            # Momentum (3 indicators)
            RSIIndicator(period=self.settings.rsi_period),
            StochasticOscillator(
                k_period=self.settings.stoch_k_period,
                d_period=self.settings.stoch_d_period
            ),
            CCIIndicator(period=self.settings.cci_period),
            # Volatility (3 indicators)
            ATRIndicator(period=self.settings.atr_period),
            BollingerBandsIndicator(
                period=self.settings.bb_period,
                std_dev=self.settings.bb_std
            ),
            KeltnerChannelIndicator(
                period=self.settings.keltner_period,
                atr_mult=self.settings.keltner_atr_mult
            ),
            # Directional (1 indicator with +DI/-DI/ADX)
            ADXIndicator(period=self.settings.adx_period),
            # Breakout (1 indicator)
            DonchianChannelIndicator(period=self.settings.donchian_period),
            # Composite (2 indicators)
            IchimokuIndicator(
                tenkan=self.settings.ichimoku_tenkan,
                kijun=self.settings.ichimoku_kijun,
                senkou_b=self.settings.ichimoku_senkou_b
            ),
            VWAPIndicator(),
            # Oscillators (2 indicators)
            WilliamsRIndicator(period=self.settings.williams_period),
            MACDIndicator(
                fast=self.settings.macd_fast,
                slow=self.settings.macd_slow,
                signal=self.settings.macd_signal
            ),
            # Structural (2 indicators)
            VolatilityRegimeDetector(),
            MarketStructureBands(),
            # Quant & Session (3 indicators)
            ZScoreIndicator(period=self.settings.zscore_period),
            OpenRangeBreakoutIndicator(),
            PreMarketSkewIndicator(),
            # Divergence (1 indicator)
            DivergenceDetector(),
        ]

        for ind in indicators:
            self._indicators[ind.name] = ind

        logger.info("indicators_registered", count=len(self._indicators),
                     names=list(self._indicators.keys()))

    def register(self, indicator: BaseIndicator) -> None:
        """Register a custom indicator."""
        self._indicators[indicator.name] = indicator
        logger.info("indicator_added", name=indicator.name)

    def unregister(self, name: str) -> None:
        """Remove an indicator from the registry."""
        if name in self._indicators:
            del self._indicators[name]

    def get(self, name: str) -> Optional[BaseIndicator]:
        """Get a specific indicator by name."""
        return self._indicators.get(name)

    @property
    def indicator_names(self) -> List[str]:
        """List all registered indicator names."""
        return list(self._indicators.keys())

    @property
    def count(self) -> int:
        """Number of registered indicators."""
        return len(self._indicators)

    def compute_all(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Compute all registered indicators on the given OHLCV DataFrame.
        Returns a single DataFrame with all indicator columns added.
        """
        if data.empty:
            logger.warning("compute_all_empty_data")
            return data

        df = data.copy()
        computed = 0

        for name, indicator in self._indicators.items():
            try:
                df = indicator.calculate(df)
                computed += 1
            except Exception as e:
                logger.error("indicator_compute_error", indicator=name, error=str(e))

        logger.info("indicators_computed", total=computed, columns=len(df.columns))
        return df

    def reset_all(self) -> None:
        """Reset all indicators."""
        for indicator in self._indicators.values():
            indicator.reset()


# Singleton
_registry: Optional[IndicatorRegistry] = None


def get_indicator_registry() -> IndicatorRegistry:
    global _registry
    if _registry is None:
        _registry = IndicatorRegistry()
    return _registry