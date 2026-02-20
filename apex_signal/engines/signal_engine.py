"""
APEX SIGNAL™ — Master Signal Engine (Enhanced v2)
Combines all strategy engines per symbol into the final signal with
AI confidence scoring, risk management, quality filtering,
regime-adaptive selection, and adaptive SL/TP.
"""
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone
import pandas as pd

from apex_signal.engines.strategy_engine import (
    StrategyEngineGroup, EngineResult, build_all_engines,
)
from apex_signal.engines.risk_manager import RiskManager, get_risk_manager
from apex_signal.engines.adaptive_sltp import AdaptiveSLTP, SLTPLevels, get_adaptive_sltp
from apex_signal.engines.signal_quality import (
    SignalQualityScorer, RegimeAdaptiveSelector, StrategyPerformanceTracker,
    get_quality_scorer, get_regime_selector, get_performance_tracker,
)
from apex_signal.config.settings import get_settings
from apex_signal.utils.logger import get_logger
from apex_signal.utils.helpers import clamp, classify_tier, utc_timestamp

logger = get_logger("signal_engine")


class MasterSignal:
    """The final, master-level signal for a symbol."""

    def __init__(
        self,
        symbol: str,
        side: str,
        confidence: float,
        tier: str,
        reason: str,
        engine_results: List[EngineResult],
        components: Optional[Dict[str, float]] = None,
        sltp: Optional[Dict[str, Any]] = None,
        quality: Optional[Dict[str, Any]] = None,
        risk_check: Optional[Dict[str, Any]] = None,
    ):
        self.symbol = symbol
        self.side = side
        self.confidence = clamp(confidence, 0, 100)
        self.tier = tier
        self.reason = reason
        self.engine_results = engine_results
        self.components = components or {}
        self.sltp = sltp
        self.quality = quality
        self.risk_check = risk_check
        self.timestamp = utc_timestamp()

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "symbol": self.symbol,
            "side": self.side,
            "confidence": round(self.confidence, 2),
            "tier": self.tier,
            "reason": self.reason,
            "components": self.components,
            "engines": [e.to_dict() for e in self.engine_results],
            "timestamp": self.timestamp,
        }
        if self.sltp:
            result["sltp"] = self.sltp
        if self.quality:
            result["quality"] = self.quality
        if self.risk_check:
            result["risk_check"] = self.risk_check
        return result

    def __repr__(self) -> str:
        return (
            f"MasterSignal({self.symbol}: {self.side} | "
            f"Confidence={self.confidence:.0f} [{self.tier}])"
        )


class SignalEngine:
    """
    Master Signal Engine v2 — Enhanced Decision Maker.
    
    Confidence Formula:
        confidence = 0.35 * ML_model_prob
                   + 0.20 * strategy_confluence
                   + 0.15 * smart_money_score
                   + 0.10 * volatility_regime
                   + 0.10 * RL_scaling_factor
                   + 0.10 * quality_score
    
    Additional Layers:
        - Signal quality filtering (reject bad setups)
        - Regime-adaptive strategy selection
        - Risk management gate (drawdown, daily limits)
        - Adaptive SL/TP computation
        - Drawdown-adjusted confidence
        - Model disagreement penalty
        - Per-strategy win-rate tracking
    """

    def __init__(self):
        self.settings = get_settings().signals
        self.engines: Dict[str, StrategyEngineGroup] = build_all_engines()
        self._ml_model = None
        self._enhanced_ml = None
        self._rl_agent = None
        self._risk_manager: Optional[RiskManager] = None
        self._sltp: Optional[AdaptiveSLTP] = None
        self._quality_scorer: Optional[SignalQualityScorer] = None
        self._regime_selector: Optional[RegimeAdaptiveSelector] = None
        self._perf_tracker: Optional[StrategyPerformanceTracker] = None

    def set_ml_model(self, model) -> None:
        self._ml_model = model

    def set_enhanced_ml(self, model) -> None:
        self._enhanced_ml = model

    def set_rl_agent(self, agent) -> None:
        self._rl_agent = agent

    def _get_risk_manager(self) -> RiskManager:
        if self._risk_manager is None:
            self._risk_manager = get_risk_manager()
        return self._risk_manager

    def _get_sltp(self) -> AdaptiveSLTP:
        if self._sltp is None:
            self._sltp = get_adaptive_sltp()
        return self._sltp

    def _get_quality_scorer(self) -> SignalQualityScorer:
        if self._quality_scorer is None:
            self._quality_scorer = get_quality_scorer()
        return self._quality_scorer

    def _get_regime_selector(self) -> RegimeAdaptiveSelector:
        if self._regime_selector is None:
            self._regime_selector = get_regime_selector()
        return self._regime_selector

    def _get_perf_tracker(self) -> StrategyPerformanceTracker:
        if self._perf_tracker is None:
            self._perf_tracker = get_performance_tracker()
        return self._perf_tracker

    def _compute_strategy_confluence(self, engine_results: List[EngineResult]) -> tuple:
        """Compute overall strategy confluence score and direction."""
        buy_total = 0.0
        sell_total = 0.0
        weight_total = 0.0

        engine_weights = {
            "trend_engine": 1.5,
            "momentum_engine": 1.2,
            "smart_structure_engine": 1.4,
            "volatility_breakout_engine": 1.0,
            "mean_reversion_engine": 0.8,
            "session_engine": 0.7,
            "hybrid_engine": 1.3,
        }

        for result in engine_results:
            w = engine_weights.get(result.engine_name, 1.0)
            weight_total += w
            if result.signal == "BUY":
                buy_total += result.confluence_score * w
            elif result.signal == "SELL":
                sell_total += result.confluence_score * w

        if weight_total == 0:
            return 0.0, "HOLD"

        buy_score = buy_total / weight_total
        sell_score = sell_total / weight_total

        if buy_score > sell_score and buy_score > 10:
            return buy_score, "BUY"
        elif sell_score > buy_score and sell_score > 10:
            return sell_score, "SELL"
        return max(buy_score, sell_score), "HOLD"

    def _compute_smart_money_score(self, engine_results: List[EngineResult]) -> float:
        for result in engine_results:
            if result.engine_name == "smart_structure_engine":
                return result.confluence_score
        return 0.0

    def _compute_volatility_regime_score(self, df: pd.DataFrame) -> float:
        if "vol_regime" not in df.columns or df.empty:
            return 50.0
        regime = df["vol_regime"].iloc[-1]
        percentile = df["vol_regime_percentile"].iloc[-1] if "vol_regime_percentile" in df.columns else 0.5
        if regime == 2:
            return 70.0 + percentile * 20
        elif regime == 0:
            return 40.0 + (1 - percentile) * 20
        return 50.0 + percentile * 10

    def _get_ml_probability(self, df: pd.DataFrame, symbol: str) -> float:
        # Try enhanced ML first
        if self._enhanced_ml is not None:
            try:
                result = self._enhanced_ml.predict_with_disagreement(df)
                return result.get("adjusted", 50.0)
            except Exception:
                pass
        # Fallback to basic ML
        if self._ml_model is not None:
            try:
                return self._ml_model.predict_probability(df, symbol)
            except Exception:
                pass
        return 50.0

    def _get_rl_scaling(self, df: pd.DataFrame, symbol: str) -> float:
        if self._rl_agent is None:
            return 50.0
        try:
            return self._rl_agent.get_scaling_factor(df, symbol)
        except Exception:
            return 50.0

    def generate_signal(self, df: pd.DataFrame, symbol: str) -> MasterSignal:
        """
        Generate the master signal for a symbol.
        Enhanced with quality filtering, risk management, and adaptive SL/TP.
        """
        if df.empty:
            return MasterSignal(
                symbol=symbol, side="HOLD", confidence=0, tier="Weak",
                reason="No data available", engine_results=[]
            )

        # 0. Get current volatility regime
        vol_regime = int(df["vol_regime"].iloc[-1]) if "vol_regime" in df.columns else 1

        # 1. Run all strategy engines
        engine_results: List[EngineResult] = []
        regime_selector = self._get_regime_selector()
        perf_tracker = self._get_perf_tracker()

        from apex_signal.strategies.registry import get_strategy_registry
        strategy_families = get_strategy_registry().get_strategies_by_family()

        for name, engine in self.engines.items():
            try:
                result = engine.evaluate(df, symbol)

                # Filter signals by regime
                result.strategy_signals = regime_selector.filter_signals(
                    result.strategy_signals, vol_regime, strategy_families
                )

                # Filter by per-strategy performance
                result.strategy_signals = [
                    s for s in result.strategy_signals
                    if perf_tracker.is_enabled(s.strategy_name)
                ]

                engine_results.append(result)
            except Exception as e:
                logger.error("engine_eval_error", engine=name, error=str(e))

        # 2. Compute components
        strategy_confluence, confluence_direction = self._compute_strategy_confluence(engine_results)
        smart_money_score = self._compute_smart_money_score(engine_results)
        volatility_score = self._compute_volatility_regime_score(df)
        ml_prob = self._get_ml_probability(df, symbol)
        rl_factor = self._get_rl_scaling(df, symbol)

        # 3. Signal quality scoring
        quality_scorer = self._get_quality_scorer()
        # Create a temporary signal for quality check
        from apex_signal.strategies.base import StrategySignal
        temp_signal = StrategySignal("master", confluence_direction, strategy_confluence, "")
        quality_result = quality_scorer.score_signal(df, temp_signal)
        quality_score = quality_result["quality_score"] * 100  # Scale to 0-100

        # 4. Apply enhanced confidence formula
        confidence = (
            0.35 * ml_prob +
            0.20 * strategy_confluence +
            0.15 * smart_money_score +
            0.10 * volatility_score +
            0.10 * rl_factor +
            0.10 * quality_score
        )
        confidence = clamp(confidence, 0, 100)

        # 5. Determine final side
        side = confluence_direction

        # 6. Quality gate — reject if quality too low
        if not quality_result["passed"] and side != "HOLD":
            confidence *= 0.6  # Significant penalty for low quality
            if confidence < 30:
                side = "HOLD"

        # 7. Risk management gate
        tier = classify_tier(confidence)
        risk_mgr = self._get_risk_manager()
        risk_check = risk_mgr.check_can_trade(symbol, confidence, tier)

        if not risk_check["allowed"] and side != "HOLD":
            logger.info("signal_blocked_by_risk", symbol=symbol, reason=risk_check["reason"])
            # Don't change to HOLD — still report the signal but mark as blocked
            risk_check["signal_blocked"] = True
        else:
            risk_check["signal_blocked"] = False

        # 8. Apply drawdown-adjusted confidence
        if risk_check.get("adjusted_confidence"):
            confidence = risk_check["adjusted_confidence"]
            tier = classify_tier(confidence)

        # 9. Compute adaptive SL/TP
        sltp_data = None
        if side in ("BUY", "SELL"):
            sltp_engine = self._get_sltp()
            sltp_levels = sltp_engine.compute(df, side, confidence, tier)
            sltp_data = sltp_levels.to_dict()

        # 10. Build narrative reason
        active_engines = [e for e in engine_results if e.signal == side and e.confluence_score > 20]
        engine_names = [e.engine_name.replace("_engine", "").replace("_", " ").title()
                       for e in active_engines[:3]]

        reason_parts = []
        if engine_names:
            reason_parts.append(f"Engines: {' + '.join(engine_names)}")
        for e in active_engines[:2]:
            top_strats = [s for s in e.strategy_signals if s.signal == side and s.confidence > 30]
            if top_strats:
                top = max(top_strats, key=lambda s: s.confidence)
                reason_parts.append(top.reason)

        # Add quality and risk context
        if quality_result["quality_score"] >= 0.7:
            reason_parts.append("✅ High quality setup")
        if sltp_data and sltp_data.get("risk_reward_ratio", 0) >= 2.5:
            reason_parts.append(f"R:R={sltp_data['risk_reward_ratio']:.1f}:1")

        reason = "; ".join(reason_parts) if reason_parts else "No strong confluence detected"

        components = {
            "ml_probability": round(ml_prob, 2),
            "strategy_confluence": round(strategy_confluence, 2),
            "smart_money_score": round(smart_money_score, 2),
            "volatility_regime": round(volatility_score, 2),
            "rl_scaling_factor": round(rl_factor, 2),
            "quality_score": round(quality_score, 2),
        }

        signal = MasterSignal(
            symbol=symbol,
            side=side,
            confidence=confidence,
            tier=tier,
            reason=reason,
            engine_results=engine_results,
            components=components,
            sltp=sltp_data,
            quality=quality_result,
            risk_check={k: v for k, v in risk_check.items() if k != "adjusted_confidence"},
        )

        logger.info(
            "master_signal_generated",
            symbol=symbol,
            side=side,
            confidence=f"{confidence:.0f}",
            tier=tier,
            quality=f"{quality_result['quality_score']:.2f}",
        )

        return signal


# Singleton
_signal_engine: Optional[SignalEngine] = None


def get_signal_engine() -> SignalEngine:
    global _signal_engine
    if _signal_engine is None:
        _signal_engine = SignalEngine()
    return _signal_engine