"""
APEX SIGNAL™ — Risk Manager Module
Position sizing, max drawdown kill switch, daily loss limits,
correlation filtering, and drawdown-adjusted confidence.
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field

from apex_signal.config.settings import get_settings
from apex_signal.utils.logger import get_logger
from apex_signal.utils.helpers import clamp, safe_divide

logger = get_logger("risk_manager")


@dataclass
class RiskState:
    """Current risk state tracking."""
    equity: float = 100000.0
    peak_equity: float = 100000.0
    daily_pnl: float = 0.0
    daily_trades: int = 0
    open_positions: int = 0
    consecutive_losses: int = 0
    total_trades: int = 0
    winning_trades: int = 0
    current_drawdown_pct: float = 0.0
    is_killed: bool = False
    kill_reason: str = ""
    last_reset_date: str = ""
    trade_history: List[float] = field(default_factory=list)


class RiskManager:
    """
    Comprehensive risk management engine.
    
    Features:
    - Kelly Criterion position sizing
    - Max drawdown kill switch (auto-halt trading)
    - Daily loss limit enforcement
    - Consecutive loss circuit breaker
    - Drawdown-adjusted confidence scaling
    - Correlation-based position limiting
    - Volatility-adjusted position sizing
    """

    def __init__(
        self,
        initial_equity: float = 100000.0,
        max_drawdown_pct: float = 15.0,
        daily_loss_limit_pct: float = 3.0,
        max_daily_trades: int = 20,
        max_consecutive_losses: int = 5,
        max_open_positions: int = 5,
        max_position_pct: float = 5.0,
        min_position_pct: float = 0.5,
        max_correlation: float = 0.7,
    ):
        self.initial_equity = initial_equity
        self.max_drawdown_pct = max_drawdown_pct
        self.daily_loss_limit_pct = daily_loss_limit_pct
        self.max_daily_trades = max_daily_trades
        self.max_consecutive_losses = max_consecutive_losses
        self.max_open_positions = max_open_positions
        self.max_position_pct = max_position_pct
        self.min_position_pct = min_position_pct
        self.max_correlation = max_correlation

        self.state = RiskState(equity=initial_equity, peak_equity=initial_equity)
        self._correlation_cache: Dict[str, pd.Series] = {}

    def check_can_trade(self, symbol: str, confidence: float, tier: str) -> Dict[str, Any]:
        """
        Master gate: check if a new trade is allowed.
        Returns dict with 'allowed', 'reason', and 'adjusted_confidence'.
        """
        self._check_daily_reset()

        # Kill switch check
        if self.state.is_killed:
            return {
                "allowed": False,
                "reason": f"Kill switch active: {self.state.kill_reason}",
                "adjusted_confidence": 0,
            }

        # Max drawdown kill switch
        self._update_drawdown()
        if self.state.current_drawdown_pct >= self.max_drawdown_pct:
            self.state.is_killed = True
            self.state.kill_reason = f"Max drawdown {self.state.current_drawdown_pct:.1f}% >= {self.max_drawdown_pct}%"
            logger.critical("kill_switch_activated", reason=self.state.kill_reason)
            return {
                "allowed": False,
                "reason": self.state.kill_reason,
                "adjusted_confidence": 0,
            }

        # Daily loss limit
        daily_loss_pct = abs(self.state.daily_pnl) / self.state.equity * 100 if self.state.daily_pnl < 0 else 0
        if daily_loss_pct >= self.daily_loss_limit_pct:
            return {
                "allowed": False,
                "reason": f"Daily loss limit reached: {daily_loss_pct:.1f}% >= {self.daily_loss_limit_pct}%",
                "adjusted_confidence": 0,
            }

        # Max daily trades
        if self.state.daily_trades >= self.max_daily_trades:
            return {
                "allowed": False,
                "reason": f"Max daily trades reached: {self.state.daily_trades}",
                "adjusted_confidence": 0,
            }

        # Max open positions
        if self.state.open_positions >= self.max_open_positions:
            return {
                "allowed": False,
                "reason": f"Max open positions reached: {self.state.open_positions}",
                "adjusted_confidence": 0,
            }

        # Consecutive loss circuit breaker
        if self.state.consecutive_losses >= self.max_consecutive_losses:
            # Allow only Elite tier signals during losing streak
            if tier != "Elite":
                return {
                    "allowed": False,
                    "reason": f"Consecutive loss breaker: {self.state.consecutive_losses} losses, only Elite signals allowed",
                    "adjusted_confidence": 0,
                }

        # Apply drawdown-adjusted confidence
        adjusted = self.adjust_confidence_for_drawdown(confidence)

        return {
            "allowed": True,
            "reason": "Trade approved by risk manager",
            "adjusted_confidence": adjusted,
            "position_size_pct": self.calculate_position_size(confidence, tier),
        }

    def adjust_confidence_for_drawdown(self, confidence: float) -> float:
        """
        Reduce confidence proportionally during drawdown periods.
        At 0% DD → no reduction. At max DD → 50% reduction.
        """
        if self.state.current_drawdown_pct <= 0:
            return confidence

        dd_ratio = self.state.current_drawdown_pct / self.max_drawdown_pct
        reduction_factor = 1.0 - (dd_ratio * 0.5)  # Max 50% reduction
        adjusted = confidence * clamp(reduction_factor, 0.5, 1.0)

        if adjusted < confidence:
            logger.info("confidence_reduced_by_drawdown",
                       original=f"{confidence:.0f}",
                       adjusted=f"{adjusted:.0f}",
                       drawdown=f"{self.state.current_drawdown_pct:.1f}%")

        return adjusted

    def calculate_position_size(self, confidence: float, tier: str, atr_pct: float = 1.0) -> float:
        """
        Calculate position size using modified Kelly Criterion.
        
        Factors:
        - Confidence level (higher = larger position)
        - Tier (Elite gets more, Weak gets less)
        - Current drawdown (reduce during DD)
        - Volatility (reduce in high vol)
        - Win rate history
        """
        # Base Kelly fraction
        win_rate = self._get_win_rate()
        avg_win = self._get_avg_win()
        avg_loss = self._get_avg_loss()

        if avg_loss > 0 and win_rate > 0:
            # Kelly: f = (bp - q) / b where b = avg_win/avg_loss, p = win_rate, q = 1-p
            b = avg_win / avg_loss
            kelly = (b * win_rate - (1 - win_rate)) / b
            kelly = max(0, kelly)
        else:
            kelly = 0.02  # Default 2%

        # Half-Kelly for safety
        base_size = kelly * 0.5 * 100  # Convert to percentage

        # Tier multiplier
        tier_mult = {"Elite": 1.2, "Strong": 1.0, "Moderate": 0.7, "Weak": 0.3}.get(tier, 0.5)

        # Confidence scaling (0-100 → 0.5-1.5)
        conf_mult = 0.5 + (confidence / 100.0)

        # Drawdown reduction
        dd_mult = 1.0 - (self.state.current_drawdown_pct / self.max_drawdown_pct * 0.5)
        dd_mult = clamp(dd_mult, 0.3, 1.0)

        # Volatility adjustment (high vol → smaller position)
        vol_mult = 1.0 / max(0.5, atr_pct) if atr_pct > 0 else 1.0
        vol_mult = clamp(vol_mult, 0.5, 1.5)

        # Final position size
        size = base_size * tier_mult * conf_mult * dd_mult * vol_mult
        size = clamp(size, self.min_position_pct, self.max_position_pct)

        return round(size, 2)

    def record_trade(self, pnl: float) -> None:
        """Record a completed trade for tracking."""
        self.state.daily_pnl += pnl
        self.state.daily_trades += 1
        self.state.total_trades += 1
        self.state.trade_history.append(pnl)

        if pnl > 0:
            self.state.winning_trades += 1
            self.state.consecutive_losses = 0
        else:
            self.state.consecutive_losses += 1

        self.state.equity += pnl
        self.state.peak_equity = max(self.state.peak_equity, self.state.equity)
        self._update_drawdown()

        # Keep last 200 trades
        if len(self.state.trade_history) > 200:
            self.state.trade_history = self.state.trade_history[-200:]

    def open_position(self) -> None:
        self.state.open_positions += 1

    def close_position(self) -> None:
        self.state.open_positions = max(0, self.state.open_positions - 1)

    def check_correlation(self, symbol: str, existing_symbols: List[str],
                          price_data: Dict[str, pd.Series]) -> bool:
        """
        Check if a new symbol is too correlated with existing positions.
        Returns True if correlation is acceptable (can trade).
        """
        if not existing_symbols or symbol not in price_data:
            return True

        new_returns = price_data[symbol].pct_change().dropna()

        for existing in existing_symbols:
            if existing in price_data:
                existing_returns = price_data[existing].pct_change().dropna()
                # Align series
                common_idx = new_returns.index.intersection(existing_returns.index)
                if len(common_idx) < 20:
                    continue
                corr = new_returns.loc[common_idx].corr(existing_returns.loc[common_idx])
                if abs(corr) > self.max_correlation:
                    logger.info("correlation_filter_blocked",
                               symbol=symbol, existing=existing, correlation=f"{corr:.2f}")
                    return False
        return True

    def reset_kill_switch(self) -> None:
        """Manually reset the kill switch (requires human intervention)."""
        self.state.is_killed = False
        self.state.kill_reason = ""
        logger.info("kill_switch_reset")

    def _update_drawdown(self) -> None:
        if self.state.peak_equity > 0:
            self.state.current_drawdown_pct = (
                (self.state.peak_equity - self.state.equity) / self.state.peak_equity * 100
            )

    def _check_daily_reset(self) -> None:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if self.state.last_reset_date != today:
            self.state.daily_pnl = 0.0
            self.state.daily_trades = 0
            self.state.last_reset_date = today

    def _get_win_rate(self) -> float:
        if not self.state.trade_history:
            return 0.5  # Default assumption
        wins = sum(1 for t in self.state.trade_history if t > 0)
        return wins / len(self.state.trade_history)

    def _get_avg_win(self) -> float:
        wins = [t for t in self.state.trade_history if t > 0]
        return np.mean(wins) if wins else 1.0

    def _get_avg_loss(self) -> float:
        losses = [abs(t) for t in self.state.trade_history if t < 0]
        return np.mean(losses) if losses else 1.0

    @property
    def risk_report(self) -> Dict[str, Any]:
        return {
            "equity": round(self.state.equity, 2),
            "peak_equity": round(self.state.peak_equity, 2),
            "current_drawdown_pct": round(self.state.current_drawdown_pct, 2),
            "daily_pnl": round(self.state.daily_pnl, 2),
            "daily_trades": self.state.daily_trades,
            "open_positions": self.state.open_positions,
            "consecutive_losses": self.state.consecutive_losses,
            "total_trades": self.state.total_trades,
            "win_rate": round(self._get_win_rate() * 100, 1),
            "is_killed": self.state.is_killed,
            "kill_reason": self.state.kill_reason,
        }


# Singleton
_risk_manager: Optional[RiskManager] = None


def get_risk_manager() -> RiskManager:
    global _risk_manager
    if _risk_manager is None:
        settings = get_settings().backtest
        _risk_manager = RiskManager(initial_equity=settings.initial_capital)
    return _risk_manager