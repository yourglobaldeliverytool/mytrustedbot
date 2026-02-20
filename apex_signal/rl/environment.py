"""
APEX SIGNAL™ — Reinforcement Learning Environment & Agent
Custom Gym environment simulating signal → action rewards.
PPO/DQN-style agent that outputs a scaling factor for confidence adjustment.
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass

import gymnasium as gym
from gymnasium import spaces

from apex_signal.config.settings import get_settings
from apex_signal.utils.logger import get_logger
from apex_signal.utils.helpers import clamp

logger = get_logger("rl_environment")


class TradingEnvironment(gym.Env):
    """
    Custom Gym environment for signal-based trading.
    
    State: [price_features, indicator_summary, position_state]
    Action: 0=HOLD, 1=BUY, 2=SELL
    Reward: net_pnl - risk_penalty - drawdown_penalty
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self, df: pd.DataFrame, initial_capital: float = 100000.0):
        super().__init__()
        self.settings = get_settings().rl
        self.df = df.reset_index(drop=True)
        self.initial_capital = initial_capital

        # State space: 10 features
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32
        )

        # Action space: 0=HOLD, 1=BUY, 2=SELL
        self.action_space = spaces.Discrete(3)

        # Internal state
        self.current_step = 0
        self.capital = initial_capital
        self.position = 0  # 0=flat, 1=long, -1=short
        self.entry_price = 0.0
        self.peak_capital = initial_capital
        self.total_pnl = 0.0
        self.trades = 0

    def _get_state(self) -> np.ndarray:
        """Build observation state vector."""
        if self.current_step >= len(self.df):
            return np.zeros(10, dtype=np.float32)

        row = self.df.iloc[self.current_step]
        close = float(row.get("close", 0))

        state = np.array([
            float(row.get("rsi", 50)) / 100.0,
            float(row.get("macd_histogram", 0)),
            float(row.get("bb_pct_b", 0.5)),
            float(row.get("adx", 0)) / 100.0,
            float(row.get("atr_pct", 0)) / 10.0,
            float(row.get("rvol", 1.0)),
            float(row.get("vwap_distance_pct", 0)) / 5.0,
            float(row.get("zscore", 0)) / 3.0,
            float(self.position),
            float(self.capital / self.initial_capital - 1.0),
        ], dtype=np.float32)

        return np.nan_to_num(state, nan=0.0, posinf=1.0, neginf=-1.0)

    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        """Reset environment to initial state."""
        super().reset(seed=seed)
        self.current_step = 0
        self.capital = self.initial_capital
        self.position = 0
        self.entry_price = 0.0
        self.peak_capital = self.initial_capital
        self.total_pnl = 0.0
        self.trades = 0
        return self._get_state(), {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step in the environment."""
        if self.current_step >= len(self.df) - 1:
            return self._get_state(), 0.0, True, False, {}

        close = float(self.df.iloc[self.current_step].get("close", 0))
        next_close = float(self.df.iloc[self.current_step + 1].get("close", 0))

        reward = 0.0
        pnl = 0.0

        # Execute action
        if action == 1:  # BUY
            if self.position == 0:
                self.position = 1
                self.entry_price = close
                self.trades += 1
            elif self.position == -1:
                # Close short
                pnl = (self.entry_price - close) * (self.capital * 0.02 / self.entry_price) if self.entry_price > 0 else 0
                self.capital += pnl
                self.position = 1
                self.entry_price = close
                self.trades += 1

        elif action == 2:  # SELL
            if self.position == 0:
                self.position = -1
                self.entry_price = close
                self.trades += 1
            elif self.position == 1:
                # Close long
                pnl = (close - self.entry_price) * (self.capital * 0.02 / self.entry_price) if self.entry_price > 0 else 0
                self.capital += pnl
                self.position = -1
                self.entry_price = close
                self.trades += 1

        # Calculate unrealized PnL
        if self.position == 1 and self.entry_price > 0:
            unrealized = (next_close - self.entry_price) / self.entry_price
        elif self.position == -1 and self.entry_price > 0:
            unrealized = (self.entry_price - next_close) / self.entry_price
        else:
            unrealized = 0.0

        # Reward calculation
        reward = pnl + unrealized * self.capital * 0.02

        # Risk penalty
        risk_penalty = self.settings.risk_penalty * abs(unrealized) if self.position != 0 else 0
        reward -= risk_penalty

        # Drawdown penalty
        self.peak_capital = max(self.peak_capital, self.capital)
        drawdown = (self.peak_capital - self.capital) / self.peak_capital if self.peak_capital > 0 else 0
        reward -= self.settings.drawdown_penalty * drawdown

        self.total_pnl += pnl
        self.current_step += 1

        done = self.current_step >= len(self.df) - 1
        truncated = False

        info = {
            "capital": self.capital,
            "position": self.position,
            "pnl": pnl,
            "total_pnl": self.total_pnl,
            "trades": self.trades,
            "drawdown": drawdown,
        }

        return self._get_state(), float(reward), done, truncated, info


class SimpleRLAgent:
    """
    Simple tabular Q-learning agent for confidence scaling.
    Outputs a scaling factor (0-100) based on market state.
    """

    def __init__(self, state_bins: int = 10, n_actions: int = 3):
        self.settings = get_settings().rl
        self.state_bins = state_bins
        self.n_actions = n_actions
        self.q_table: Dict[tuple, np.ndarray] = {}
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        self._is_trained = False
        self._scaling_history = []

    def _discretize_state(self, state: np.ndarray) -> tuple:
        """Discretize continuous state into bins."""
        binned = np.clip(
            ((state + 1) / 2 * self.state_bins).astype(int),
            0, self.state_bins - 1
        )
        return tuple(binned)

    def _get_q_values(self, state_key: tuple) -> np.ndarray:
        """Get Q-values for a state, initializing if needed."""
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.n_actions)
        return self.q_table[state_key]

    def select_action(self, state: np.ndarray) -> int:
        """Select action using epsilon-greedy policy."""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        state_key = self._discretize_state(state)
        q_values = self._get_q_values(state_key)
        return int(np.argmax(q_values))

    def update(self, state: np.ndarray, action: int, reward: float,
               next_state: np.ndarray, done: bool) -> None:
        """Update Q-table using Q-learning update rule."""
        state_key = self._discretize_state(state)
        next_state_key = self._discretize_state(next_state)

        q_values = self._get_q_values(state_key)
        next_q_values = self._get_q_values(next_state_key)

        target = reward + (0 if done else self.settings.gamma * np.max(next_q_values))
        q_values[action] += self.settings.learning_rate * (target - q_values[action])

        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def train(self, env: TradingEnvironment, episodes: Optional[int] = None) -> Dict[str, Any]:
        """Train the agent on the environment."""
        n_episodes = episodes or self.settings.episodes
        episode_rewards = []
        episode_trades = []

        for ep in range(n_episodes):
            state, _ = env.reset()
            total_reward = 0
            done = False

            while not done:
                action = self.select_action(state)
                next_state, reward, done, truncated, info = env.step(action)
                self.update(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward

            episode_rewards.append(total_reward)
            episode_trades.append(info.get("trades", 0))

        self._is_trained = True

        metrics = {
            "episodes": n_episodes,
            "avg_reward": float(np.mean(episode_rewards[-50:])),
            "max_reward": float(np.max(episode_rewards)),
            "avg_trades": float(np.mean(episode_trades[-50:])),
            "q_table_size": len(self.q_table),
            "final_epsilon": self.epsilon,
        }

        logger.info("rl_training_complete", **metrics)
        return metrics

    def get_scaling_factor(self, df: pd.DataFrame, symbol: str = "") -> float:
        """
        Get RL scaling factor (0-100) for confidence adjustment.
        Higher = more confident in current market conditions.
        """
        if not self._is_trained:
            return 50.0  # Neutral

        try:
            # Build state from latest data
            if df.empty:
                return 50.0

            row = df.iloc[-1]
            state = np.array([
                float(row.get("rsi", 50)) / 100.0,
                float(row.get("macd_histogram", 0)),
                float(row.get("bb_pct_b", 0.5)),
                float(row.get("adx", 0)) / 100.0,
                float(row.get("atr_pct", 0)) / 10.0,
                float(row.get("rvol", 1.0)),
                float(row.get("vwap_distance_pct", 0)) / 5.0,
                float(row.get("zscore", 0)) / 3.0,
                0.0,  # No position context
                0.0,  # No capital context
            ], dtype=np.float32)

            state = np.nan_to_num(state, nan=0.0)
            state_key = self._discretize_state(state)
            q_values = self._get_q_values(state_key)

            # Convert Q-value confidence to 0-100 scale
            max_q = np.max(q_values)
            min_q = np.min(q_values)
            q_range = max_q - min_q

            if q_range > 0:
                # Normalize best action confidence
                confidence = ((max_q - min_q) / (abs(max_q) + abs(min_q) + 1e-8)) * 100
                return float(clamp(confidence, 0, 100))

            return 50.0

        except Exception as e:
            logger.error("rl_scaling_error", error=str(e))
            return 50.0

    @property
    def is_trained(self) -> bool:
        return self._is_trained


# Singletons
_rl_agent: Optional[SimpleRLAgent] = None


def get_rl_agent() -> SimpleRLAgent:
    global _rl_agent
    if _rl_agent is None:
        _rl_agent = SimpleRLAgent()
    return _rl_agent