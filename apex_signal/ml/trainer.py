"""
APEX SIGNAL™ — ML Model Training & Prediction
Trains LightGBM, RandomForest, and Logistic Regression models.
Persists models with versioning and metrics.
"""
import os
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple
from datetime import datetime, timezone
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

from apex_signal.ml.features.feature_engineering import build_features, FEATURE_COLUMNS
from apex_signal.config.settings import get_settings
from apex_signal.utils.logger import get_logger

logger = get_logger("ml_trainer")


class MLModelSuite:
    """
    Machine Learning model suite for signal confidence scoring.
    Trains and manages multiple models with ensemble prediction.
    """

    def __init__(self):
        self.settings = get_settings().ml
        self.model_dir = Path(self.settings.model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.scaler: Optional[StandardScaler] = None
        self.models: Dict[str, Any] = {}
        self.metrics: Dict[str, Dict[str, float]] = {}
        self.version: str = ""
        self._is_trained = False

    def train(self, features: pd.DataFrame, labels: pd.Series) -> Dict[str, Dict[str, float]]:
        """Train all models using time-series cross-validation."""
        if len(features) < 50:
            logger.warning("insufficient_training_data", count=len(features))
            return {}

        logger.info("training_started", samples=len(features), features=features.shape[1])

        # Scale features
        self.scaler = StandardScaler()
        X = self.scaler.fit_transform(features.values)
        y = labels.values

        # Time-series split
        n_splits = min(5, max(2, len(features) // 50))
        tscv = TimeSeriesSplit(n_splits=n_splits)

        # Define models
        model_configs = {
            "random_forest": RandomForestClassifier(
                n_estimators=100, max_depth=10, min_samples_leaf=5,
                random_state=42, n_jobs=-1
            ),
            "logistic_regression": LogisticRegression(
                max_iter=1000, random_state=42, C=1.0
            ),
        }

        if HAS_LIGHTGBM:
            model_configs["lightgbm"] = lgb.LGBMClassifier(
                n_estimators=200, max_depth=8, learning_rate=0.05,
                min_child_samples=10, random_state=42, verbose=-1,
                num_leaves=31,
            )

        all_metrics = {}

        for name, model in model_configs.items():
            try:
                fold_metrics = []

                for train_idx, val_idx in tscv.split(X):
                    X_train, X_val = X[train_idx], X[val_idx]
                    y_train, y_val = y[train_idx], y[val_idx]

                    # Skip if only one class in training
                    if len(np.unique(y_train)) < 2:
                        continue

                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_val)
                    y_prob = model.predict_proba(X_val)[:, 1] if hasattr(model, 'predict_proba') else y_pred

                    fold_metrics.append({
                        "accuracy": accuracy_score(y_val, y_pred),
                        "precision": precision_score(y_val, y_pred, zero_division=0),
                        "recall": recall_score(y_val, y_pred, zero_division=0),
                        "f1": f1_score(y_val, y_pred, zero_division=0),
                        "auc": roc_auc_score(y_val, y_prob) if len(np.unique(y_val)) > 1 else 0.5,
                    })

                if fold_metrics:
                    # Average metrics across folds
                    avg_metrics = {
                        k: np.mean([m[k] for m in fold_metrics])
                        for k in fold_metrics[0]
                    }

                    # Final fit on all data
                    if len(np.unique(y)) >= 2:
                        model.fit(X, y)

                    self.models[name] = model
                    all_metrics[name] = avg_metrics

                    logger.info("model_trained", model=name,
                               accuracy=f"{avg_metrics['accuracy']:.3f}",
                               auc=f"{avg_metrics['auc']:.3f}")

            except Exception as e:
                logger.error("model_training_error", model=name, error=str(e))

        self.metrics = all_metrics
        self._is_trained = bool(self.models)

        # Save models
        if self._is_trained:
            self.version = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            self._save_models()

        return all_metrics

    def predict_probability(self, df: pd.DataFrame, symbol: str = "") -> float:
        """
        Predict probability of successful trade (TP before SL).
        Returns 0-100 scaled probability.
        """
        if not self._is_trained or not self.models:
            return 50.0  # Neutral

        try:
            features = build_features(df)
            if features.empty:
                return 50.0

            # Use last row
            X = features.iloc[[-1]].values
            X = self.scaler.transform(X)

            probabilities = []
            weights = {
                "lightgbm": 0.5,
                "random_forest": 0.3,
                "logistic_regression": 0.2,
            }

            for name, model in self.models.items():
                try:
                    if hasattr(model, 'predict_proba'):
                        prob = model.predict_proba(X)[0][1]
                    else:
                        prob = float(model.predict(X)[0])
                    w = weights.get(name, 0.33)
                    probabilities.append((prob, w))
                except Exception:
                    continue

            if not probabilities:
                return 50.0

            # Weighted average
            total_weight = sum(w for _, w in probabilities)
            if total_weight == 0:
                return 50.0

            weighted_prob = sum(p * w for p, w in probabilities) / total_weight

            # Scale to 0-100
            return float(np.clip(weighted_prob * 100, 0, 100))

        except Exception as e:
            logger.error("ml_predict_error", error=str(e))
            return 50.0

    def _save_models(self) -> None:
        """Save all models, scaler, and metrics to disk."""
        version_dir = self.model_dir / self.version
        version_dir.mkdir(parents=True, exist_ok=True)

        # Save scaler
        if self.scaler:
            joblib.dump(self.scaler, version_dir / "scaler.joblib")

        # Save models
        for name, model in self.models.items():
            joblib.dump(model, version_dir / f"{name}.joblib")

        # Save metrics
        with open(version_dir / "metrics.json", "w") as f:
            json.dump({
                "version": self.version,
                "metrics": self.metrics,
                "feature_columns": FEATURE_COLUMNS,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }, f, indent=2)

        # Update latest symlink
        latest_path = self.model_dir / "latest"
        if latest_path.exists():
            latest_path.unlink()
        try:
            latest_path.symlink_to(self.version)
        except OSError:
            # Fallback: write version to file
            with open(self.model_dir / "latest_version.txt", "w") as f:
                f.write(self.version)

        logger.info("models_saved", version=self.version, path=str(version_dir))

    def load_latest(self) -> bool:
        """Load the latest saved models."""
        try:
            # Find latest version
            latest_path = self.model_dir / "latest"
            if latest_path.exists() and latest_path.is_symlink():
                version_dir = self.model_dir / os.readlink(str(latest_path))
            else:
                version_file = self.model_dir / "latest_version.txt"
                if version_file.exists():
                    version = version_file.read_text().strip()
                    version_dir = self.model_dir / version
                else:
                    # Find most recent directory
                    dirs = [d for d in self.model_dir.iterdir() if d.is_dir()]
                    if not dirs:
                        return False
                    version_dir = max(dirs, key=lambda d: d.name)

            if not version_dir.exists():
                return False

            # Load scaler
            scaler_path = version_dir / "scaler.joblib"
            if scaler_path.exists():
                self.scaler = joblib.load(scaler_path)

            # Load models
            self.models = {}
            for model_file in version_dir.glob("*.joblib"):
                if model_file.name == "scaler.joblib":
                    continue
                name = model_file.stem
                self.models[name] = joblib.load(model_file)

            # Load metrics
            metrics_path = version_dir / "metrics.json"
            if metrics_path.exists():
                with open(metrics_path) as f:
                    data = json.load(f)
                    self.metrics = data.get("metrics", {})
                    self.version = data.get("version", "")

            self._is_trained = bool(self.models)
            logger.info("models_loaded", version=self.version, models=list(self.models.keys()))
            return self._is_trained

        except Exception as e:
            logger.error("model_load_error", error=str(e))
            return False

    @property
    def is_trained(self) -> bool:
        return self._is_trained

    @property
    def model_info(self) -> Dict[str, Any]:
        return {
            "is_trained": self._is_trained,
            "version": self.version,
            "models": list(self.models.keys()),
            "metrics": self.metrics,
        }


# Singleton
_ml_suite: Optional[MLModelSuite] = None


def get_ml_suite() -> MLModelSuite:
    global _ml_suite
    if _ml_suite is None:
        _ml_suite = MLModelSuite()
    return _ml_suite