"""
APEX SIGNAL™ — Enhanced Backtester
Monte Carlo simulation, regime-specific breakdown, strategy attribution,
and advanced risk-adjusted metrics (MAR, Omega, Tail Ratio).
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

from apex_signal.backtest.backtester import Backtester, BacktestResult, Trade
from apex_signal.engines.signal_engine import SignalEngine
from apex_signal.utils.logger import get_logger
from apex_signal.utils.helpers import safe_divide

logger = get_logger("enhanced_backtester")


@dataclass
class MonteCarloResult:
    """Monte Carlo simulation results."""
    n_simulations: int
    median_return_pct: float
    mean_return_pct: float
    p5_return_pct: float   # 5th percentile (worst case)
    p25_return_pct: float
    p75_return_pct: float
    p95_return_pct: float  # 95th percentile (best case)
    probability_of_profit: float
    median_max_drawdown: float
    worst_max_drawdown: float
    confidence_interval_95: tuple

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_simulations": self.n_simulations,
            "median_return_pct": round(self.median_return_pct, 2),
            "mean_return_pct": round(self.mean_return_pct, 2),
            "p5_return_pct": round(self.p5_return_pct, 2),
            "p95_return_pct": round(self.p95_return_pct, 2),
            "probability_of_profit": round(self.probability_of_profit, 2),
            "median_max_drawdown": round(self.median_max_drawdown, 2),
            "worst_max_drawdown": round(self.worst_max_drawdown, 2),
            "confidence_interval_95": (
                round(self.confidence_interval_95[0], 2),
                round(self.confidence_interval_95[1], 2),
            ),
        }


@dataclass
class EnhancedBacktestResult:
    """Enhanced backtest result with all additional analytics."""
    base_result: BacktestResult
    monte_carlo: Optional[MonteCarloResult] = None
    regime_breakdown: Optional[Dict[str, Dict[str, Any]]] = None
    strategy_attribution: Optional[Dict[str, Dict[str, Any]]] = None
    advanced_metrics: Optional[Dict[str, float]] = None

    def to_dict(self) -> Dict[str, Any]:
        result = self.base_result.to_dict()
        if self.monte_carlo:
            result["monte_carlo"] = self.monte_carlo.to_dict()
        if self.regime_breakdown:
            result["regime_breakdown"] = self.regime_breakdown
        if self.strategy_attribution:
            result["strategy_attribution"] = self.strategy_attribution
        if self.advanced_metrics:
            result["advanced_metrics"] = self.advanced_metrics
        return result


class EnhancedBacktester(Backtester):
    """
    Enhanced backtester with Monte Carlo simulation,
    regime-specific analysis, and strategy attribution.
    """

    def run_enhanced(
        self,
        df: pd.DataFrame,
        symbol: str,
        signal_engine: SignalEngine,
        min_confidence: float = 40.0,
        n_monte_carlo: int = 500,
    ) -> EnhancedBacktestResult:
        """Run enhanced backtest with all additional analytics."""
        # Run base backtest
        base_result = self.run(df, symbol, signal_engine, min_confidence)

        # Monte Carlo simulation
        mc_result = self._monte_carlo(base_result.trades, n_monte_carlo)

        # Regime breakdown
        regime_breakdown = self._regime_breakdown(df, base_result.trades)

        # Strategy attribution
        strategy_attr = self._strategy_attribution(base_result.trades)

        # Advanced metrics
        advanced = self._compute_advanced_metrics(base_result)

        return EnhancedBacktestResult(
            base_result=base_result,
            monte_carlo=mc_result,
            regime_breakdown=regime_breakdown,
            strategy_attribution=strategy_attr,
            advanced_metrics=advanced,
        )

    def _monte_carlo(self, trades: List[Trade], n_sims: int = 500) -> Optional[MonteCarloResult]:
        """
        Monte Carlo simulation by randomly resampling trade returns.
        Tests robustness of the strategy across different trade orderings.
        """
        if len(trades) < 5:
            return None

        trade_returns = [t.pnl_pct for t in trades]
        initial = self.settings.initial_capital

        sim_returns = []
        sim_drawdowns = []

        for _ in range(n_sims):
            # Resample trades with replacement
            sampled = np.random.choice(trade_returns, size=len(trade_returns), replace=True)

            # Compute equity curve
            equity = [initial]
            for ret in sampled:
                new_eq = equity[-1] * (1 + ret / 100)
                equity.append(new_eq)

            equity = np.array(equity)
            total_return = (equity[-1] - initial) / initial * 100
            sim_returns.append(total_return)

            # Max drawdown
            peak = np.maximum.accumulate(equity)
            dd = (peak - equity) / peak * 100
            sim_drawdowns.append(np.max(dd))

        sim_returns = np.array(sim_returns)
        sim_drawdowns = np.array(sim_drawdowns)

        return MonteCarloResult(
            n_simulations=n_sims,
            median_return_pct=float(np.median(sim_returns)),
            mean_return_pct=float(np.mean(sim_returns)),
            p5_return_pct=float(np.percentile(sim_returns, 5)),
            p25_return_pct=float(np.percentile(sim_returns, 25)),
            p75_return_pct=float(np.percentile(sim_returns, 75)),
            p95_return_pct=float(np.percentile(sim_returns, 95)),
            probability_of_profit=float(np.mean(sim_returns > 0) * 100),
            median_max_drawdown=float(np.median(sim_drawdowns)),
            worst_max_drawdown=float(np.max(sim_drawdowns)),
            confidence_interval_95=(
                float(np.percentile(sim_returns, 2.5)),
                float(np.percentile(sim_returns, 97.5)),
            ),
        )

    def _regime_breakdown(self, df: pd.DataFrame, trades: List[Trade]) -> Dict[str, Dict[str, Any]]:
        """Break down performance by volatility regime."""
        if not trades or "vol_regime_label" not in df.columns:
            return {}

        # Map trades to regimes based on entry time
        regime_trades: Dict[str, List[Trade]] = {"low": [], "normal": [], "high": []}

        for trade in trades:
            # Find the regime at entry time
            try:
                if hasattr(trade.entry_time, 'isoformat'):
                    mask = df.index <= trade.entry_time
                    if mask.any():
                        regime = str(df.loc[mask, "vol_regime_label"].iloc[-1])
                        if regime in regime_trades:
                            regime_trades[regime].append(trade)
                        continue
            except Exception:
                pass
            regime_trades["normal"].append(trade)

        breakdown = {}
        for regime, rtrades in regime_trades.items():
            if not rtrades:
                breakdown[regime] = {"trades": 0, "win_rate": 0, "avg_pnl_pct": 0, "total_pnl": 0}
                continue

            wins = sum(1 for t in rtrades if t.pnl > 0)
            breakdown[regime] = {
                "trades": len(rtrades),
                "win_rate": round(wins / len(rtrades) * 100, 1),
                "avg_pnl_pct": round(np.mean([t.pnl_pct for t in rtrades]), 2),
                "total_pnl": round(sum(t.pnl for t in rtrades), 2),
                "avg_confidence": round(np.mean([t.confidence for t in rtrades]), 1),
            }

        return breakdown

    def _strategy_attribution(self, trades: List[Trade]) -> Dict[str, Dict[str, Any]]:
        """Attribute performance to individual strategies/reasons."""
        if not trades:
            return {}

        # Group by tier
        tier_groups: Dict[str, List[Trade]] = {}
        for trade in trades:
            tier = trade.tier or "Unknown"
            if tier not in tier_groups:
                tier_groups[tier] = []
            tier_groups[tier].append(trade)

        attribution = {}
        for tier, ttrades in tier_groups.items():
            wins = sum(1 for t in ttrades if t.pnl > 0)
            attribution[tier] = {
                "trades": len(ttrades),
                "win_rate": round(wins / len(ttrades) * 100, 1) if ttrades else 0,
                "total_pnl": round(sum(t.pnl for t in ttrades), 2),
                "avg_pnl_pct": round(np.mean([t.pnl_pct for t in ttrades]), 2),
                "avg_confidence": round(np.mean([t.confidence for t in ttrades]), 1),
                "best_trade_pct": round(max(t.pnl_pct for t in ttrades), 2),
                "worst_trade_pct": round(min(t.pnl_pct for t in ttrades), 2),
            }

        return attribution

    def _compute_advanced_metrics(self, result: BacktestResult) -> Dict[str, float]:
        """Compute advanced risk-adjusted return metrics."""
        metrics = {}

        if result.equity_curve is None or len(result.equity_curve) < 2:
            return metrics

        returns = result.equity_curve.pct_change().dropna()

        if len(returns) < 2:
            return metrics

        # MAR Ratio (return / max drawdown)
        metrics["mar_ratio"] = round(
            safe_divide(result.total_return_pct, result.max_drawdown_pct), 2
        )

        # Omega Ratio (probability weighted ratio of gains vs losses)
        threshold = 0
        gains = returns[returns > threshold].sum()
        losses = abs(returns[returns <= threshold].sum())
        metrics["omega_ratio"] = round(safe_divide(gains, losses, 1.0), 2)

        # Tail Ratio (95th percentile gain / 5th percentile loss)
        p95 = np.percentile(returns, 95)
        p5 = abs(np.percentile(returns, 5))
        metrics["tail_ratio"] = round(safe_divide(p95, p5, 1.0), 2)

        # Gain-to-Pain Ratio
        total_gain = returns[returns > 0].sum()
        total_pain = abs(returns[returns < 0].sum())
        metrics["gain_to_pain"] = round(safe_divide(total_gain, total_pain, 0), 2)

        # Expectancy per trade
        if result.total_trades > 0:
            win_rate = result.win_rate / 100
            metrics["expectancy_pct"] = round(
                win_rate * result.avg_win_pct + (1 - win_rate) * result.avg_loss_pct, 3
            )
        else:
            metrics["expectancy_pct"] = 0

        # Recovery Factor (total return / max drawdown)
        metrics["recovery_factor"] = round(
            safe_divide(result.total_return_pct, result.max_drawdown_pct), 2
        )

        # Profit Factor (already in base, but ensure it's here)
        metrics["profit_factor"] = result.profit_factor

        return metrics