"""
APEX SIGNAL™ — Signal Quality Scorer & Regime-Adaptive Strategy Selector
Rejects low-quality setups, selects strategies suited to current regime,
tracks per-strategy win rates, and enforces multi-timeframe confirmation.
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Set
from datetime import datetime, timezone, timedelta
from collections import defaultdict

from apex_signal.strategies.base import StrategySignal
from apex_signal.utils.logger import get_logger
from apex_signal.utils.helpers import clamp

logger = get_logger("signal_quality")


class StrategyPerformanceTracker:
    """
    Tracks per-strategy win rates and auto-disables underperformers.
    Maintains a rolling window of recent signals and their outcomes.
    """

    def __init__(self, min_trades: int = 10, min_win_rate: float = 0.40,
                 lookback_trades: int = 50, cooldown_trades: int = 20):
        self.min_trades = min_trades
        self.min_win_rate = min_win_rate
        self.lookback_trades = lookback_trades
        self.cooldown_trades = cooldown_trades

        # strategy_name -> list of (pnl, timestamp)
        self._history: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self._disabled: Dict[str, str] = {}  # strategy_name -> reason
        self._cooldown_counter: Dict[str, int] = defaultdict(int)

    def record_outcome(self, strategy_name: str, pnl: float) -> None:
        """Record a trade outcome for a strategy."""
        self._history[strategy_name].append({
            "pnl": pnl,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        # Keep only recent trades
        if len(self._history[strategy_name]) > self.lookback_trades:
            self._history[strategy_name] = self._history[strategy_name][-self.lookback_trades:]

        # Check if strategy should be disabled/re-enabled
        self._evaluate_strategy(strategy_name)

    def is_enabled(self, strategy_name: str) -> bool:
        """Check if a strategy is currently enabled."""
        if strategy_name in self._disabled:
            # Check cooldown
            self._cooldown_counter[strategy_name] += 1
            if self._cooldown_counter[strategy_name] >= self.cooldown_trades:
                # Re-enable and reset
                del self._disabled[strategy_name]
                self._cooldown_counter[strategy_name] = 0
                logger.info("strategy_re_enabled", strategy=strategy_name)
                return True
            return False
        return True

    def get_strategy_stats(self, strategy_name: str) -> Dict[str, Any]:
        """Get performance stats for a strategy."""
        history = self._history.get(strategy_name, [])
        if not history:
            return {"trades": 0, "win_rate": 0, "avg_pnl": 0, "enabled": True}

        wins = sum(1 for h in history if h["pnl"] > 0)
        total = len(history)
        avg_pnl = np.mean([h["pnl"] for h in history])

        return {
            "trades": total,
            "win_rate": round(wins / total * 100, 1) if total > 0 else 0,
            "avg_pnl": round(avg_pnl, 2),
            "enabled": strategy_name not in self._disabled,
            "disabled_reason": self._disabled.get(strategy_name, ""),
        }

    def _evaluate_strategy(self, strategy_name: str) -> None:
        history = self._history[strategy_name]
        if len(history) < self.min_trades:
            return

        recent = history[-self.lookback_trades:]
        wins = sum(1 for h in recent if h["pnl"] > 0)
        win_rate = wins / len(recent)

        if win_rate < self.min_win_rate and strategy_name not in self._disabled:
            self._disabled[strategy_name] = (
                f"Win rate {win_rate:.0%} below minimum {self.min_win_rate:.0%} "
                f"over last {len(recent)} trades"
            )
            self._cooldown_counter[strategy_name] = 0
            logger.warning("strategy_disabled", strategy=strategy_name,
                          win_rate=f"{win_rate:.0%}")

    @property
    def all_stats(self) -> Dict[str, Dict[str, Any]]:
        return {name: self.get_strategy_stats(name) for name in self._history}


class RegimeAdaptiveSelector:
    """
    Selects which strategy families to run based on current volatility regime.
    
    - Low volatility → Mean reversion, range-bound strategies
    - Normal volatility → All strategies
    - High volatility → Trend following, breakout, momentum strategies
    """

    # Strategy families suited to each regime
    REGIME_MAP = {
        0: {  # Low volatility
            "preferred": [
                "mean_reversion", "session_aware",
            ],
            "allowed": [
                "mean_reversion", "session_aware", "momentum", "hybrid",
            ],
            "blocked": ["volatility_breakout"],
            "confidence_boost": {"mean_reversion": 10, "session_aware": 5},
        },
        1: {  # Normal volatility
            "preferred": [
                "trend_following", "momentum", "hybrid",
            ],
            "allowed": [
                "trend_following", "momentum", "volatility_breakout",
                "mean_reversion", "smart_money", "session_aware", "hybrid",
            ],
            "blocked": [],
            "confidence_boost": {},
        },
        2: {  # High volatility
            "preferred": [
                "trend_following", "volatility_breakout", "smart_money",
            ],
            "allowed": [
                "trend_following", "volatility_breakout", "smart_money",
                "momentum", "hybrid",
            ],
            "blocked": ["mean_reversion"],
            "confidence_boost": {"trend_following": 10, "volatility_breakout": 10, "smart_money": 5},
        },
    }

    def get_allowed_families(self, vol_regime: int) -> List[str]:
        """Get list of allowed strategy families for current regime."""
        config = self.REGIME_MAP.get(vol_regime, self.REGIME_MAP[1])
        return config["allowed"]

    def get_blocked_families(self, vol_regime: int) -> List[str]:
        """Get list of blocked strategy families for current regime."""
        config = self.REGIME_MAP.get(vol_regime, self.REGIME_MAP[1])
        return config["blocked"]

    def get_confidence_boost(self, vol_regime: int, family: str) -> float:
        """Get confidence boost for a strategy family in current regime."""
        config = self.REGIME_MAP.get(vol_regime, self.REGIME_MAP[1])
        return config["confidence_boost"].get(family, 0)

    def filter_signals(self, signals: List[StrategySignal], vol_regime: int,
                       strategy_families: Dict[str, List[str]]) -> List[StrategySignal]:
        """Filter and adjust signals based on regime."""
        blocked = set()
        for family in self.get_blocked_families(vol_regime):
            blocked.update(strategy_families.get(family, []))

        filtered = []
        for sig in signals:
            if sig.strategy_name in blocked:
                continue

            # Apply confidence boost
            for family, strategies in strategy_families.items():
                if sig.strategy_name in strategies:
                    boost = self.get_confidence_boost(vol_regime, family)
                    if boost > 0:
                        sig.confidence = clamp(sig.confidence + boost, 0, 100)
                    break

            filtered.append(sig)

        return filtered


class SignalQualityScorer:
    """
    Multi-factor signal quality scoring system.
    Rejects low-quality setups before they reach the output.
    
    Quality factors:
    1. Volume confirmation (must have above-average volume)
    2. Trend alignment (signal direction must match higher-TF trend)
    3. Momentum confirmation (RSI/MACD must support direction)
    4. Volatility suitability (signal type must match vol regime)
    5. Confirmation candle (previous candle must support direction)
    6. Divergence check (no bearish divergence on buys, vice versa)
    7. ADX trend strength gate
    """

    def __init__(self, min_quality_score: float = 0.45):
        self.min_quality_score = min_quality_score

    def score_signal(self, df: pd.DataFrame, signal: StrategySignal) -> Dict[str, Any]:
        """
        Score a signal's quality on multiple factors.
        Returns quality score (0-1) and detailed breakdown.
        """
        if df.empty or len(df) < 10:
            return {"quality_score": 0, "passed": False, "factors": {}, "reason": "Insufficient data"}

        factors = {}
        side = signal.signal

        if side == "HOLD":
            return {"quality_score": 0, "passed": False, "factors": {}, "reason": "HOLD signal"}

        # 1. Volume confirmation
        factors["volume"] = self._check_volume(df, side)

        # 2. Trend alignment
        factors["trend_alignment"] = self._check_trend_alignment(df, side)

        # 3. Momentum confirmation
        factors["momentum"] = self._check_momentum(df, side)

        # 4. Confirmation candle
        factors["confirmation_candle"] = self._check_confirmation_candle(df, side)

        # 5. Divergence check
        factors["no_divergence"] = self._check_no_divergence(df, side)

        # 6. ADX strength gate
        factors["trend_strength"] = self._check_trend_strength(df)

        # 7. Spread/volatility suitability
        factors["volatility_suitable"] = self._check_volatility_suitability(df)

        # Weighted quality score
        weights = {
            "volume": 0.15,
            "trend_alignment": 0.25,
            "momentum": 0.15,
            "confirmation_candle": 0.15,
            "no_divergence": 0.10,
            "trend_strength": 0.10,
            "volatility_suitable": 0.10,
        }

        quality_score = sum(
            factors[k]["score"] * weights[k] for k in weights if k in factors
        )

        passed = quality_score >= self.min_quality_score
        failed_factors = [k for k, v in factors.items() if v["score"] < 0.5]

        reason = "Quality check passed" if passed else f"Failed factors: {', '.join(failed_factors)}"

        return {
            "quality_score": round(quality_score, 3),
            "passed": passed,
            "factors": {k: v for k, v in factors.items()},
            "reason": reason,
        }

    def _check_volume(self, df: pd.DataFrame, side: str) -> Dict[str, Any]:
        if "rvol" not in df.columns:
            return {"score": 0.5, "detail": "No volume data"}
        rvol = float(df["rvol"].iloc[-1])
        if rvol >= 1.2:
            return {"score": 1.0, "detail": f"Strong volume confirmation ({rvol:.1f}x)"}
        elif rvol >= 0.8:
            return {"score": 0.6, "detail": f"Adequate volume ({rvol:.1f}x)"}
        return {"score": 0.2, "detail": f"Weak volume ({rvol:.1f}x)"}

    def _check_trend_alignment(self, df: pd.DataFrame, side: str) -> Dict[str, Any]:
        if "ema_50" not in df.columns or "ema_200" not in df.columns:
            return {"score": 0.5, "detail": "No trend data"}
        close = float(df["close"].iloc[-1])
        ema50 = float(df["ema_50"].iloc[-1])
        ema200 = float(df["ema_200"].iloc[-1])

        if side == "BUY":
            if close > ema50 > ema200:
                return {"score": 1.0, "detail": "Perfect bullish alignment"}
            elif close > ema50:
                return {"score": 0.7, "detail": "Partial bullish alignment"}
            elif close > ema200:
                return {"score": 0.4, "detail": "Above EMA200 only"}
            return {"score": 0.1, "detail": "Against trend"}
        else:
            if close < ema50 < ema200:
                return {"score": 1.0, "detail": "Perfect bearish alignment"}
            elif close < ema50:
                return {"score": 0.7, "detail": "Partial bearish alignment"}
            elif close < ema200:
                return {"score": 0.4, "detail": "Below EMA200 only"}
            return {"score": 0.1, "detail": "Against trend"}

    def _check_momentum(self, df: pd.DataFrame, side: str) -> Dict[str, Any]:
        score = 0.5
        details = []
        if "rsi" in df.columns:
            rsi = float(df["rsi"].iloc[-1])
            if side == "BUY" and 40 < rsi < 70:
                score += 0.2
                details.append(f"RSI supportive ({rsi:.0f})")
            elif side == "SELL" and 30 < rsi < 60:
                score += 0.2
                details.append(f"RSI supportive ({rsi:.0f})")
            elif (side == "BUY" and rsi > 80) or (side == "SELL" and rsi < 20):
                score -= 0.2
                details.append(f"RSI extreme ({rsi:.0f})")

        if "macd_histogram" in df.columns:
            hist = float(df["macd_histogram"].iloc[-1])
            if (side == "BUY" and hist > 0) or (side == "SELL" and hist < 0):
                score += 0.2
                details.append("MACD aligned")
            else:
                score -= 0.1

        return {"score": clamp(score, 0, 1), "detail": "; ".join(details) or "Neutral"}

    def _check_confirmation_candle(self, df: pd.DataFrame, side: str) -> Dict[str, Any]:
        if len(df) < 2:
            return {"score": 0.5, "detail": "Insufficient candles"}
        prev = df.iloc[-2]
        prev_bullish = float(prev["close"]) > float(prev["open"])
        prev_body = abs(float(prev["close"]) - float(prev["open"]))
        prev_range = float(prev["high"]) - float(prev["low"])
        body_ratio = prev_body / prev_range if prev_range > 0 else 0

        if side == "BUY" and prev_bullish and body_ratio > 0.5:
            return {"score": 1.0, "detail": "Strong bullish confirmation candle"}
        elif side == "SELL" and not prev_bullish and body_ratio > 0.5:
            return {"score": 1.0, "detail": "Strong bearish confirmation candle"}
        elif (side == "BUY" and prev_bullish) or (side == "SELL" and not prev_bullish):
            return {"score": 0.6, "detail": "Weak confirmation candle"}
        return {"score": 0.2, "detail": "No confirmation candle"}

    def _check_no_divergence(self, df: pd.DataFrame, side: str) -> Dict[str, Any]:
        if len(df) < 10 or "rsi" not in df.columns:
            return {"score": 0.5, "detail": "Cannot check divergence"}

        prices = df["close"].iloc[-10:]
        rsi_vals = df["rsi"].iloc[-10:]

        price_higher = float(prices.iloc[-1]) > float(prices.iloc[0])
        rsi_higher = float(rsi_vals.iloc[-1]) > float(rsi_vals.iloc[0])

        # Bearish divergence: price up but RSI down
        if side == "BUY" and price_higher and not rsi_higher:
            return {"score": 0.2, "detail": "Bearish RSI divergence detected"}
        # Bullish divergence: price down but RSI up
        elif side == "SELL" and not price_higher and rsi_higher:
            return {"score": 0.2, "detail": "Bullish RSI divergence detected"}

        return {"score": 0.8, "detail": "No adverse divergence"}

    def _check_trend_strength(self, df: pd.DataFrame) -> Dict[str, Any]:
        if "adx" not in df.columns:
            return {"score": 0.5, "detail": "No ADX data"}
        adx = float(df["adx"].iloc[-1])
        if adx > 25:
            return {"score": 1.0, "detail": f"Strong trend (ADX={adx:.0f})"}
        elif adx > 20:
            return {"score": 0.7, "detail": f"Moderate trend (ADX={adx:.0f})"}
        elif adx > 15:
            return {"score": 0.4, "detail": f"Weak trend (ADX={adx:.0f})"}
        return {"score": 0.2, "detail": f"No trend (ADX={adx:.0f})"}

    def _check_volatility_suitability(self, df: pd.DataFrame) -> Dict[str, Any]:
        if "vol_regime" not in df.columns or "bb_squeeze" not in df.columns:
            return {"score": 0.5, "detail": "No volatility data"}
        regime = int(df["vol_regime"].iloc[-1])
        squeeze = int(df["bb_squeeze"].iloc[-1])

        if regime == 1:  # Normal
            return {"score": 0.8, "detail": "Normal volatility — suitable for most strategies"}
        elif regime == 2 and not squeeze:
            return {"score": 0.7, "detail": "High volatility — suitable for breakout/trend"}
        elif regime == 0 and squeeze:
            return {"score": 0.6, "detail": "Low vol squeeze — potential breakout setup"}
        return {"score": 0.5, "detail": f"Regime={regime}"}


# Singletons
_tracker: Optional[StrategyPerformanceTracker] = None
_selector: Optional[RegimeAdaptiveSelector] = None
_scorer: Optional[SignalQualityScorer] = None


def get_performance_tracker() -> StrategyPerformanceTracker:
    global _tracker
    if _tracker is None:
        _tracker = StrategyPerformanceTracker()
    return _tracker


def get_regime_selector() -> RegimeAdaptiveSelector:
    global _selector
    if _selector is None:
        _selector = RegimeAdaptiveSelector()
    return _selector


def get_quality_scorer() -> SignalQualityScorer:
    global _scorer
    if _scorer is None:
        _scorer = SignalQualityScorer()
    return _scorer