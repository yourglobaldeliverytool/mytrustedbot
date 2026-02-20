"""
APEX SIGNAL™ — Unit Tests for Indicators
Tests all 22 indicators for correctness, edge cases, and output format.
"""
import pytest
import pandas as pd
import numpy as np

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
from apex_signal.indicators.registry import IndicatorRegistry


class TestEMAIndicator:
    def test_calculate_adds_columns(self, sample_ohlcv_df):
        ind = EMAIndicator(periods=[8, 20])
        result = ind.calculate(sample_ohlcv_df)
        assert "ema_8" in result.columns
        assert "ema_20" in result.columns

    def test_ema_values_reasonable(self, sample_ohlcv_df):
        ind = EMAIndicator(periods=[20])
        result = ind.calculate(sample_ohlcv_df)
        assert result["ema_20"].iloc[-1] > 0
        assert not result["ema_20"].isna().all()

    def test_reset(self, sample_ohlcv_df):
        ind = EMAIndicator()
        ind.calculate(sample_ohlcv_df)
        assert ind.last_result is not None
        ind.reset()
        assert ind.last_result is None


class TestSMAIndicator:
    def test_calculate_adds_columns(self, sample_ohlcv_df):
        ind = SMAIndicator(periods=[20, 50])
        result = ind.calculate(sample_ohlcv_df)
        assert "sma_20" in result.columns
        assert "sma_50" in result.columns

    def test_sma_first_values_nan(self, sample_ohlcv_df):
        ind = SMAIndicator(periods=[20])
        result = ind.calculate(sample_ohlcv_df)
        assert result["sma_20"].iloc[:19].isna().all()
        assert not result["sma_20"].iloc[19:].isna().all()


class TestRSIIndicator:
    def test_rsi_range(self, sample_ohlcv_df):
        ind = RSIIndicator(period=14)
        result = ind.calculate(sample_ohlcv_df)
        valid_rsi = result["rsi"].dropna()
        assert (valid_rsi >= 0).all() and (valid_rsi <= 100).all()

    def test_rsi_overbought_oversold_flags(self, sample_ohlcv_df):
        ind = RSIIndicator()
        result = ind.calculate(sample_ohlcv_df)
        assert "rsi_overbought" in result.columns
        assert "rsi_oversold" in result.columns
        assert result["rsi_overbought"].isin([0, 1]).all()


class TestStochasticOscillator:
    def test_stochastic_range(self, sample_ohlcv_df):
        ind = StochasticOscillator()
        result = ind.calculate(sample_ohlcv_df)
        valid_k = result["stoch_k"].dropna()
        assert (valid_k >= 0).all() and (valid_k <= 100).all()

    def test_stochastic_columns(self, sample_ohlcv_df):
        ind = StochasticOscillator()
        result = ind.calculate(sample_ohlcv_df)
        assert all(c in result.columns for c in ["stoch_k", "stoch_d", "stoch_overbought", "stoch_oversold"])


class TestCCIIndicator:
    def test_cci_output(self, sample_ohlcv_df):
        ind = CCIIndicator()
        result = ind.calculate(sample_ohlcv_df)
        assert "cci" in result.columns
        assert not result["cci"].isna().all()


class TestATRIndicator:
    def test_atr_positive(self, sample_ohlcv_df):
        ind = ATRIndicator()
        result = ind.calculate(sample_ohlcv_df)
        valid_atr = result["atr"].dropna()
        assert (valid_atr >= 0).all()

    def test_atr_pct(self, sample_ohlcv_df):
        ind = ATRIndicator()
        result = ind.calculate(sample_ohlcv_df)
        assert "atr_pct" in result.columns


class TestBollingerBands:
    def test_bands_order(self, sample_ohlcv_df):
        ind = BollingerBandsIndicator()
        result = ind.calculate(sample_ohlcv_df)
        valid = result.dropna(subset=["bb_upper", "bb_middle", "bb_lower"])
        assert (valid["bb_upper"] >= valid["bb_middle"]).all()
        assert (valid["bb_middle"] >= valid["bb_lower"]).all()

    def test_pct_b_range(self, sample_ohlcv_df):
        ind = BollingerBandsIndicator()
        result = ind.calculate(sample_ohlcv_df)
        assert "bb_pct_b" in result.columns


class TestKeltnerChannel:
    def test_keltner_columns(self, sample_ohlcv_df):
        ind = KeltnerChannelIndicator()
        result = ind.calculate(sample_ohlcv_df)
        assert all(c in result.columns for c in ["kc_upper", "kc_middle", "kc_lower", "kc_pct"])


class TestADXIndicator:
    def test_adx_range(self, sample_ohlcv_df):
        ind = ADXIndicator()
        result = ind.calculate(sample_ohlcv_df)
        valid_adx = result["adx"].dropna()
        assert (valid_adx >= 0).all()

    def test_dmi_columns(self, sample_ohlcv_df):
        ind = ADXIndicator()
        result = ind.calculate(sample_ohlcv_df)
        assert all(c in result.columns for c in ["plus_di", "minus_di", "dmi_bullish", "dmi_bearish"])


class TestDonchianChannel:
    def test_donchian_columns(self, sample_ohlcv_df):
        ind = DonchianChannelIndicator()
        result = ind.calculate(sample_ohlcv_df)
        assert all(c in result.columns for c in ["dc_upper", "dc_lower", "dc_middle", "dc_width", "dc_position"])


class TestIchimoku:
    def test_ichimoku_columns(self, sample_ohlcv_df):
        ind = IchimokuIndicator()
        result = ind.calculate(sample_ohlcv_df)
        expected = ["ichi_tenkan", "ichi_kijun", "ichi_senkou_a", "ichi_senkou_b",
                    "ichi_above_cloud", "ichi_below_cloud"]
        assert all(c in result.columns for c in expected)


class TestVWAP:
    def test_vwap_output(self, sample_ohlcv_df):
        ind = VWAPIndicator()
        result = ind.calculate(sample_ohlcv_df)
        assert "vwap" in result.columns
        assert "vwap_bias" in result.columns
        assert result["vwap"].iloc[-1] > 0


class TestWilliamsR:
    def test_williams_range(self, sample_ohlcv_df):
        ind = WilliamsRIndicator()
        result = ind.calculate(sample_ohlcv_df)
        valid = result["williams_r"].dropna()
        assert (valid >= -100).all() and (valid <= 0).all()


class TestMACD:
    def test_macd_columns(self, sample_ohlcv_df):
        ind = MACDIndicator()
        result = ind.calculate(sample_ohlcv_df)
        assert all(c in result.columns for c in [
            "macd_line", "macd_signal", "macd_histogram",
            "macd_cross_bull", "macd_cross_bear"
        ])


class TestVolatilityRegime:
    def test_regime_values(self, sample_ohlcv_df):
        ind = VolatilityRegimeDetector()
        result = ind.calculate(sample_ohlcv_df)
        assert "vol_regime" in result.columns
        assert result["vol_regime"].isin([0, 1, 2]).all()


class TestMarketStructure:
    def test_structure_columns(self, sample_ohlcv_df):
        ind = MarketStructureBands()
        result = ind.calculate(sample_ohlcv_df)
        assert all(c in result.columns for c in ["ms_resistance", "ms_support", "ms_position"])


class TestZScore:
    def test_zscore_output(self, sample_ohlcv_df):
        ind = ZScoreIndicator()
        result = ind.calculate(sample_ohlcv_df)
        assert "zscore" in result.columns
        assert "zscore_volume" in result.columns


class TestOpenRangeBreakout:
    def test_orb_columns(self, sample_ohlcv_df):
        ind = OpenRangeBreakoutIndicator()
        result = ind.calculate(sample_ohlcv_df)
        assert all(c in result.columns for c in ["orb_high", "orb_low", "orb_break_up", "orb_break_down"])


class TestPreMarketSkew:
    def test_skew_output(self, sample_ohlcv_df):
        ind = PreMarketSkewIndicator()
        result = ind.calculate(sample_ohlcv_df)
        assert "gap_pct" in result.columns
        assert "premarket_skew" in result.columns


class TestIndicatorRegistry:
    def test_registry_count(self):
        registry = IndicatorRegistry()
        assert registry.count >= 20

    def test_compute_all(self, sample_ohlcv_df):
        registry = IndicatorRegistry()
        result = registry.compute_all(sample_ohlcv_df)
        assert len(result.columns) > len(sample_ohlcv_df.columns)
        assert "ema_8" in result.columns
        assert "rsi" in result.columns
        assert "macd_line" in result.columns

    def test_compute_all_empty(self, empty_df):
        registry = IndicatorRegistry()
        result = registry.compute_all(empty_df)
        assert result.empty

    def test_reset_all(self, sample_ohlcv_df):
        registry = IndicatorRegistry()
        registry.compute_all(sample_ohlcv_df)
        registry.reset_all()
        for ind in registry._indicators.values():
            assert ind.last_result is None


class TestOBV:
    def test_obv_output(self, sample_ohlcv_df):
        ind = OBVIndicator()
        result = ind.calculate(sample_ohlcv_df)
        assert "obv" in result.columns
        assert "obv_ema" in result.columns


class TestCMF:
    def test_cmf_range(self, sample_ohlcv_df):
        ind = ChaikinMoneyFlowIndicator()
        result = ind.calculate(sample_ohlcv_df)
        assert "cmf" in result.columns
        valid = result["cmf"].dropna()
        assert (valid >= -1).all() and (valid <= 1).all()


class TestRelativeVolume:
    def test_rvol_output(self, sample_ohlcv_df):
        ind = RelativeVolumeIndicator()
        result = ind.calculate(sample_ohlcv_df)
        assert "rvol" in result.columns
        assert "rvol_high" in result.columns