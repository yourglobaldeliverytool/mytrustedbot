"""
APEX SIGNAL™ — Branded Telegram Notifier
Async, non-blocking delivery with inline buttons, rate limiting,
quiet hours, and retry logic.
"""
import asyncio
import json
from typing import Optional, Dict, Any, List
from datetime import datetime, timezone

from apex_signal.config.settings import get_settings
from apex_signal.utils.logger import get_logger
from apex_signal.utils.helpers import classify_tier, format_confidence

logger = get_logger("telegram_notifier")


# Tier emoji mapping
TIER_EMOJI = {
    "Elite": "🔥",
    "Strong": "💪",
    "Moderate": "📊",
    "Weak": "📉",
}

SIDE_EMOJI = {
    "BUY": "🟢",
    "SELL": "🔴",
    "HOLD": "⚪",
}


class TelegramNotifier:
    """
    Production-grade Telegram notification system.
    Features: branded messages, inline buttons, rate limiting,
    quiet hours filtering, and retry logic.
    """

    def __init__(self):
        self.settings = get_settings().telegram
        self._last_send_time = 0.0
        self._message_count = 0
        self._muted_chats: set = set()
        self._initialized = False
        self._bot = None

    async def initialize(self) -> None:
        """Initialize the Telegram bot."""
        if not self.settings.bot_token:
            logger.warning("telegram_no_token", msg="Bot token not configured")
            return

        try:
            from telegram import Bot
            self._bot = Bot(token=self.settings.bot_token)
            self._initialized = True
            logger.info("telegram_initialized")
        except ImportError:
            logger.warning("telegram_import_error", msg="python-telegram-bot not installed")
        except Exception as e:
            logger.error("telegram_init_error", error=str(e))

    async def shutdown(self) -> None:
        """Shutdown the bot."""
        if self._bot:
            try:
                await self._bot.shutdown()
            except Exception:
                pass
        self._initialized = False

    def _is_quiet_hours(self) -> bool:
        """Check if current time is within quiet hours."""
        hour = datetime.now(timezone.utc).hour
        start = self.settings.quiet_hours_start
        end = self.settings.quiet_hours_end
        if start < end:
            return start <= hour < end
        else:
            return hour >= start or hour < end

    async def _rate_limit(self) -> None:
        """Enforce rate limiting (max 1 msg/sec)."""
        import time
        now = time.time()
        elapsed = now - self._last_send_time
        min_interval = 1.0 / self.settings.rate_limit_per_second
        if elapsed < min_interval:
            await asyncio.sleep(min_interval - elapsed)
        self._last_send_time = time.time()

    def format_signal_message(self, signal_data: Dict[str, Any]) -> str:
        """Format a branded signal message."""
        symbol = signal_data.get("symbol", "UNKNOWN")
        side = signal_data.get("side", "HOLD")
        confidence = signal_data.get("confidence", 0)
        tier = signal_data.get("tier", classify_tier(confidence))
        reason = signal_data.get("reason", "No reason provided")
        timestamp = signal_data.get("timestamp", datetime.now(timezone.utc).isoformat())
        components = signal_data.get("components", {})

        side_emoji = SIDE_EMOJI.get(side, "⚪")
        tier_emoji = TIER_EMOJI.get(tier, "📊")

        # Engine summary
        engines = signal_data.get("engines", [])
        active_engines = [e["engine_name"].replace("_engine", "").replace("_", " ").title()
                         for e in engines if e.get("signal") == side and e.get("confluence_score", 0) > 20]
        engine_str = " + ".join(active_engines[:3]) if active_engines else "Multi-strategy"

        message = (
            f"🚀 *APEX SIGNAL™*\n"
            f"{'━' * 28}\n"
            f"\n"
            f"📌 *Symbol:* `{symbol}`\n"
            f"{side_emoji} *Signal:* *{side}*\n"
            f"{tier_emoji} *Confidence:* `{confidence:.0f}` ({tier})\n"
            f"🔧 *Engines:* {engine_str}\n"
            f"\n"
            f"💡 *Reason:*\n"
            f"_{reason[:200]}_\n"
            f"\n"
        )

        # Component breakdown
        if components:
            message += f"📊 *Confidence Breakdown:*\n"
            comp_labels = {
                "ml_probability": "🤖 ML Model",
                "strategy_confluence": "📈 Confluence",
                "smart_money_score": "🏦 Smart Money",
                "volatility_regime": "📉 Volatility",
                "rl_scaling_factor": "🧠 RL Agent",
            }
            for key, label in comp_labels.items():
                val = components.get(key, 0)
                bar = "█" * int(val / 10) + "░" * (10 - int(val / 10))
                message += f"  {label}: `{bar}` {val:.0f}\n"
            message += "\n"

        message += (
            f"🕐 *Time:* `{timestamp[:19]} UTC`\n"
            f"{'━' * 28}\n"
            f"_Powered by APEX SIGNAL™ v1.0_"
        )

        return message

    def _build_inline_keyboard(self, signal_data: Dict[str, Any]) -> Optional[Any]:
        """Build inline keyboard with action buttons."""
        try:
            from telegram import InlineKeyboardButton, InlineKeyboardMarkup

            symbol = signal_data.get("symbol", "")
            side = signal_data.get("side", "")
            confidence = signal_data.get("confidence", 0)

            # Create callback data
            payload = json.dumps({
                "s": symbol, "d": side, "c": round(confidence),
            }, separators=(",", ":"))

            keyboard = [
                [
                    InlineKeyboardButton("📋 View Strategy", callback_data=f"view_{symbol}"),
                    InlineKeyboardButton("📑 Copy Trade", callback_data=f"copy_{payload}"),
                ],
                [
                    InlineKeyboardButton("🔇 Mute Symbol", callback_data=f"mute_{symbol}"),
                    InlineKeyboardButton("📊 Performance", callback_data=f"perf_{symbol}"),
                ],
            ]
            return InlineKeyboardMarkup(keyboard)
        except ImportError:
            return None

    async def send_signal(self, signal_data: Dict[str, Any], chat_id: Optional[str] = None) -> bool:
        """
        Send a branded signal notification to Telegram.
        Respects rate limits, quiet hours, and mute settings.
        """
        target_chat = chat_id or self.settings.chat_id
        if not target_chat:
            logger.warning("telegram_no_chat_id")
            return False

        # Check mute
        symbol = signal_data.get("symbol", "")
        if symbol in self._muted_chats:
            logger.debug("telegram_muted", symbol=symbol)
            return False

        # Check quiet hours (only Elite signals pass through)
        tier = signal_data.get("tier", "Weak")
        if self._is_quiet_hours() and tier != "Elite":
            logger.debug("telegram_quiet_hours", tier=tier)
            return False

        if not self._initialized:
            await self.initialize()

        if not self._bot:
            logger.warning("telegram_bot_not_available")
            return False

        # Format message
        message = self.format_signal_message(signal_data)
        keyboard = self._build_inline_keyboard(signal_data)

        # Send with retry logic
        for attempt in range(self.settings.max_retries):
            try:
                await self._rate_limit()
                await self._bot.send_message(
                    chat_id=target_chat,
                    text=message,
                    parse_mode="Markdown",
                    reply_markup=keyboard,
                )
                self._message_count += 1
                logger.info("telegram_sent", symbol=symbol, tier=tier,
                           attempt=attempt + 1, total_sent=self._message_count)
                return True

            except Exception as e:
                logger.warning("telegram_send_error", attempt=attempt + 1,
                             error=str(e))
                if attempt < self.settings.max_retries - 1:
                    await asyncio.sleep(self.settings.retry_delay * (attempt + 1))

        logger.error("telegram_send_failed", symbol=symbol, max_retries=self.settings.max_retries)
        return False

    async def send_backtest_report(self, result_data: Dict[str, Any],
                                    chat_id: Optional[str] = None) -> bool:
        """Send a backtest performance report."""
        target_chat = chat_id or self.settings.chat_id
        if not target_chat or not self._bot:
            return False

        message = (
            f"📊 *APEX SIGNAL™ — Backtest Report*\n"
            f"{'━' * 28}\n\n"
            f"📌 *Symbol:* `{result_data.get('symbol', 'N/A')}`\n"
            f"📅 *Period:* {result_data.get('period', 'N/A')}\n\n"
            f"💰 *Return:* `{result_data.get('total_return_pct', 0):.1f}%`\n"
            f"📈 *Trades:* `{result_data.get('total_trades', 0)}`\n"
            f"✅ *Win Rate:* `{result_data.get('win_rate', 0):.0f}%`\n"
            f"📊 *Profit Factor:* `{result_data.get('profit_factor', 0):.2f}`\n"
            f"📉 *Max Drawdown:* `{result_data.get('max_drawdown_pct', 0):.1f}%`\n"
            f"📐 *Sharpe Ratio:* `{result_data.get('sharpe_ratio', 0):.2f}`\n\n"
            f"{'━' * 28}\n"
            f"_APEX SIGNAL™ Backtesting Engine_"
        )

        try:
            await self._rate_limit()
            await self._bot.send_message(
                chat_id=target_chat, text=message, parse_mode="Markdown"
            )
            return True
        except Exception as e:
            logger.error("telegram_backtest_report_error", error=str(e))
            return False

    def mute_symbol(self, symbol: str) -> None:
        """Mute notifications for a symbol."""
        self._muted_chats.add(symbol)
        logger.info("symbol_muted", symbol=symbol)

    def unmute_symbol(self, symbol: str) -> None:
        """Unmute notifications for a symbol."""
        self._muted_chats.discard(symbol)
        logger.info("symbol_unmuted", symbol=symbol)

    @property
    def stats(self) -> Dict[str, Any]:
        return {
            "initialized": self._initialized,
            "messages_sent": self._message_count,
            "muted_symbols": list(self._muted_chats),
            "is_quiet_hours": self._is_quiet_hours(),
        }


# Singleton
_notifier: Optional[TelegramNotifier] = None


def get_notifier() -> TelegramNotifier:
    global _notifier
    if _notifier is None:
        _notifier = TelegramNotifier()
    return _notifier