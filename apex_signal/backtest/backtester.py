"""
APEX SIGNAL™ — Walk-Forward Backtesting Engine
Supports multiple time-ranges, volatility regimes, equity curves, and risk metrics.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime, timezone
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

from apex_signal.engines.signal_engine import SignalEngine, MasterSignal
from apex_signal.indicators.registry import get_indicator_registry
from apex_signal.config.settings import get_settings
from apex_signal.utils.logger import get_logger
from apex_signal.utils.helpers import safe_divide

logger = get_logger("backtester")


@dataclass
class Trade:
    """Record of a single backtest trade."""
    entry_time: Any
    exit_time: Any
    side: str
    entry_price: float
    exit_price: float
    pnl: float
    pnl_pct: float
    confidence: float
    tier: str
    reason: str


@dataclass
class BacktestResult:
    """Complete backtest result with metrics and trade log."""
    symbol: str
    start_date: str
    end_date: str
    initial_capital: float
    final_capital: float
    total_return_pct: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win_pct: float
    avg_loss_pct: float
    profit_factor: float
    max_drawdown_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    avg_trade_duration: float
    trades: List[Trade] = field(default_factory=list)
    equity_curve: Optional[pd.Series] = None
    drawdown_curve: Optional[pd.Series] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "period": f"{self.start_date} to {self.end_date}",
            "initial_capital": self.initial_capital,
            "final_capital": round(self.final_capital, 2),
            "total_return_pct": round(self.total_return_pct, 2),
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": round(self.win_rate, 2),
            "avg_win_pct": round(self.avg_win_pct, 2),
            "avg_loss_pct": round(self.avg_loss_pct, 2),
            "profit_factor": round(self.profit_factor, 2),
            "max_drawdown_pct": round(self.max_drawdown_pct, 2),
            "sharpe_ratio": round(self.sharpe_ratio, 2),
            "sortino_ratio": round(self.sortino_ratio, 2),
            "calmar_ratio": round(self.calmar_ratio, 2),
        }


class Backtester:
    """
    Walk-forward backtesting engine.
    Simulates signal generation and trade execution on historical data.
    """

    def __init__(self):
        self.settings = get_settings().backtest
        self.indicator_registry = get_indicator_registry()

    def run(
        self,
        df: pd.DataFrame,
        symbol: str,
        signal_engine: SignalEngine,
        min_confidence: float = 40.0,
        min_tier: str = "Moderate",
    ) -> BacktestResult:
        """
        Run a full backtest on historical data.
        
        Args:
            df: OHLCV DataFrame
            symbol: Trading symbol
            signal_engine: Signal engine to generate signals
            min_confidence: Minimum confidence to take a trade
            min_tier: Minimum tier to take a trade
        """
        if df.empty or len(df) < 50:
            return self._empty_result(symbol)

        # Compute indicators
        enriched_df = self.indicator_registry.compute_all(df)

        capital = self.settings.initial_capital
        position = 0  # 0=flat, 1=long, -1=short
        entry_price = 0.0
        entry_time = None
        entry_confidence = 0.0
        entry_tier = ""
        entry_reason = ""

        trades: List[Trade] = []
        equity = [capital]
        equity_times = [enriched_df.index[0] if hasattr(enriched_df.index[0], 'isoformat') else 0]

        tier_order = {"Elite": 4, "Strong": 3, "Moderate": 2, "Weak": 1}
        min_tier_val = tier_order.get(min_tier, 2)

        # Walk through each bar
        window_size = 50
        for i in range(window_size, len(enriched_df)):
            window = enriched_df.iloc[i - window_size:i + 1]
            current_bar = enriched_df.iloc[i]
            close = float(current_bar["close"])
            timestamp = enriched_df.index[i]

            # Generate signal
            try:
                signal = signal_engine.generate_signal(window, symbol)
            except Exception:
                continue

            signal_tier_val = tier_order.get(signal.tier, 0)

            # Position management
            if position == 0:
                # Entry logic
                if (signal.side in ("BUY", "SELL") and
                    signal.confidence >= min_confidence and
                    signal_tier_val >= min_tier_val):

                    position = 1 if signal.side == "BUY" else -1
                    entry_price = close * (1 + self.settings.slippage_pct * position)
                    entry_time = timestamp
                    entry_confidence = signal.confidence
                    entry_tier = signal.tier
                    entry_reason = signal.reason

            elif position != 0:
                # Exit logic
                should_exit = False
                exit_reason = ""

                # Signal reversal
                if position == 1 and signal.side == "SELL" and signal.confidence >= min_confidence:
                    should_exit = True
                    exit_reason = "Signal reversal to SELL"
                elif position == -1 and signal.side == "BUY" and signal.confidence >= min_confidence:
                    should_exit = True
                    exit_reason = "Signal reversal to BUY"

                # Stop loss (2x ATR)
                if "atr" in current_bar.index:
                    atr = float(current_bar.get("atr", 0))
                    if atr > 0:
                        if position == 1 and close < entry_price - 2 * atr:
                            should_exit = True
                            exit_reason = "Stop loss hit (2x ATR)"
                        elif position == -1 and close > entry_price + 2 * atr:
                            should_exit = True
                            exit_reason = "Stop loss hit (2x ATR)"

                # Take profit (3x ATR)
                if "atr" in current_bar.index:
                    atr = float(current_bar.get("atr", 0))
                    if atr > 0:
                        if position == 1 and close > entry_price + 3 * atr:
                            should_exit = True
                            exit_reason = "Take profit hit (3x ATR)"
                        elif position == -1 and close < entry_price - 3 * atr:
                            should_exit = True
                            exit_reason = "Take profit hit (3x ATR)"

                if should_exit:
                    exit_price = close * (1 - self.settings.slippage_pct * position)
                    commission = capital * self.settings.position_size_pct * self.settings.commission_pct * 2

                    if position == 1:
                        pnl = (exit_price - entry_price) / entry_price * capital * self.settings.position_size_pct
                    else:
                        pnl = (entry_price - exit_price) / entry_price * capital * self.settings.position_size_pct

                    pnl -= commission
                    pnl_pct = (pnl / (capital * self.settings.position_size_pct)) * 100

                    trades.append(Trade(
                        entry_time=entry_time,
                        exit_time=timestamp,
                        side="BUY" if position == 1 else "SELL",
                        entry_price=entry_price,
                        exit_price=exit_price,
                        pnl=pnl,
                        pnl_pct=pnl_pct,
                        confidence=entry_confidence,
                        tier=entry_tier,
                        reason=entry_reason,
                    ))

                    capital += pnl
                    position = 0
                    entry_price = 0.0

            equity.append(capital)
            equity_times.append(timestamp)

        # Close any open position
        if position != 0 and len(enriched_df) > 0:
            final_close = float(enriched_df.iloc[-1]["close"])
            if position == 1:
                pnl = (final_close - entry_price) / entry_price * capital * self.settings.position_size_pct
            else:
                pnl = (entry_price - final_close) / entry_price * capital * self.settings.position_size_pct
            capital += pnl

        # Compute metrics
        equity_series = pd.Series(equity, index=range(len(equity)))
        return self._compute_metrics(symbol, enriched_df, capital, trades, equity_series)

    def _compute_metrics(
        self, symbol: str, df: pd.DataFrame, final_capital: float,
        trades: List[Trade], equity: pd.Series
    ) -> BacktestResult:
        """Compute comprehensive backtest metrics."""
        initial = self.settings.initial_capital

        # Basic metrics
        total_return = ((final_capital - initial) / initial) * 100
        winning = [t for t in trades if t.pnl > 0]
        losing = [t for t in trades if t.pnl <= 0]
        win_rate = safe_divide(len(winning), len(trades)) * 100

        avg_win = np.mean([t.pnl_pct for t in winning]) if winning else 0
        avg_loss = np.mean([t.pnl_pct for t in losing]) if losing else 0

        gross_profit = sum(t.pnl for t in winning) if winning else 0
        gross_loss = abs(sum(t.pnl for t in losing)) if losing else 1
        profit_factor = safe_divide(gross_profit, gross_loss, 0)

        # Drawdown
        peak = equity.expanding().max()
        drawdown = (equity - peak) / peak * 100
        max_dd = abs(drawdown.min()) if len(drawdown) > 0 else 0

        # Returns for Sharpe/Sortino
        returns = equity.pct_change().dropna()
        if len(returns) > 1:
            sharpe = safe_divide(returns.mean(), returns.std()) * np.sqrt(252)
            downside = returns[returns < 0].std()
            sortino = safe_divide(returns.mean(), downside) * np.sqrt(252) if downside > 0 else 0
        else:
            sharpe = 0
            sortino = 0

        calmar = safe_divide(total_return, max_dd) if max_dd > 0 else 0

        start_date = str(df.index[0]) if len(df) > 0 else ""
        end_date = str(df.index[-1]) if len(df) > 0 else ""

        return BacktestResult(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            initial_capital=initial,
            final_capital=final_capital,
            total_return_pct=total_return,
            total_trades=len(trades),
            winning_trades=len(winning),
            losing_trades=len(losing),
            win_rate=win_rate,
            avg_win_pct=avg_win,
            avg_loss_pct=avg_loss,
            profit_factor=profit_factor,
            max_drawdown_pct=max_dd,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            avg_trade_duration=0,
            trades=trades,
            equity_curve=equity,
            drawdown_curve=drawdown,
        )

    def _empty_result(self, symbol: str) -> BacktestResult:
        return BacktestResult(
            symbol=symbol, start_date="", end_date="",
            initial_capital=self.settings.initial_capital,
            final_capital=self.settings.initial_capital,
            total_return_pct=0, total_trades=0, winning_trades=0,
            losing_trades=0, win_rate=0, avg_win_pct=0, avg_loss_pct=0,
            profit_factor=0, max_drawdown_pct=0, sharpe_ratio=0,
            sortino_ratio=0, calmar_ratio=0, avg_trade_duration=0,
        )

    @staticmethod
    def plot_equity_curve(result: BacktestResult, save_path: str = "backtest_equity.png") -> str:
        """Plot and save equity curve with drawdown."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={"height_ratios": [3, 1]})
        fig.suptitle(f"APEX SIGNAL™ Backtest — {result.symbol}", fontsize=14, fontweight="bold")

        if result.equity_curve is not None and len(result.equity_curve) > 0:
            ax1.plot(result.equity_curve.values, color="#2196F3", linewidth=1.5, label="Equity")
            ax1.axhline(y=result.initial_capital, color="gray", linestyle="--", alpha=0.5, label="Initial Capital")
            ax1.fill_between(range(len(result.equity_curve)),
                           result.initial_capital, result.equity_curve.values,
                           where=result.equity_curve.values >= result.initial_capital,
                           alpha=0.1, color="green")
            ax1.fill_between(range(len(result.equity_curve)),
                           result.initial_capital, result.equity_curve.values,
                           where=result.equity_curve.values < result.initial_capital,
                           alpha=0.1, color="red")

        ax1.set_ylabel("Capital ($)")
        ax1.legend(loc="upper left")
        ax1.grid(True, alpha=0.3)

        # Drawdown
        if result.drawdown_curve is not None and len(result.drawdown_curve) > 0:
            ax2.fill_between(range(len(result.drawdown_curve)),
                           0, result.drawdown_curve.values, color="red", alpha=0.3)
            ax2.plot(result.drawdown_curve.values, color="red", linewidth=0.8)

        ax2.set_ylabel("Drawdown (%)")
        ax2.set_xlabel("Bar")
        ax2.grid(True, alpha=0.3)

        # Stats box
        stats_text = (
            f"Return: {result.total_return_pct:.1f}% | "
            f"Trades: {result.total_trades} | "
            f"Win Rate: {result.win_rate:.0f}% | "
            f"Sharpe: {result.sharpe_ratio:.2f} | "
            f"Max DD: {result.max_drawdown_pct:.1f}%"
        )
        fig.text(0.5, 0.01, stats_text, ha="center", fontsize=10, style="italic")

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

        logger.info("equity_curve_saved", path=save_path)
        return save_path