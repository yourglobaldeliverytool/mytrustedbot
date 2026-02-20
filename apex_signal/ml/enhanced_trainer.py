"""
APEX SIGNAL™ — Enhanced ML Pipeline
Feature importance analysis, walk-forward optimization,
Platt scaling calibration, ensemble voting with disagreement detection.
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timezone

from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss
from sklearn.preprocessing import StandardScaler

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

from apex_signal.ml.features.feature_engineering import build_features, FEATURE_COLUMNS
from apex_signal.utils.logger import get_logger
from apex_signal.utils.helpers import clamp

logger = get_logger("enhanced_ml")


class EnhancedMLPipeline:
    """
    Enhanced ML pipeline with:
    1. Feature importance analysis and pruning
    2. Walk-forward optimization with OOS validation
    3. Platt scaling for probability calibration
    4. Ensemble voting with disagreement detection
    5. Confidence penalty when models disagree
    """

    def __init__(self, top_features_pct: float = 0.75, disagreement_penalty: float = 15.0):
        self.top_features_pct = top_features_pct
        self.disagreement_penalty = disagreement_penalty
        self.scaler: Optional[StandardScaler] = None
        self.calibrated_models: Dict[str, Any] = {}
        self.feature_importance: Optional[pd.Series] = None
        self.selected_features: Optional[List[str]] = None
        self.oos_metrics: Dict[str, Dict[str, float]] = {}
        self._is_trained = False

    def train(self, features: pd.DataFrame, labels: pd.Series) -> Dict[str, Any]:
        """Full enhanced training pipeline."""
        if len(features) < 80:
            logger.warning("enhanced_ml_insufficient_data", count=len(features))
            return {}

        logger.info("enhanced_training_started", samples=len(features))

        # Step 1: Feature importance analysis
        self.selected_features = self._analyze_feature_importance(features, labels)
        X_selected = features[self.selected_features]

        # Step 2: Scale
        self.scaler = StandardScaler()
        X = self.scaler.fit_transform(X_selected.values)
        y = labels.values

        # Step 3: Walk-forward with calibration
        n_splits = min(5, max(2, len(features) // 60))
        tscv = TimeSeriesSplit(n_splits=n_splits)

        base_models = self._build_base_models()
        all_oos_metrics = {}

        for name, model in base_models.items():
            fold_metrics = []
            last_model = None

            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]

                if len(np.unique(y_train)) < 2:
                    continue

                model.fit(X_train, y_train)
                last_model = model

                y_pred = model.predict(X_val)
                y_prob = model.predict_proba(X_val)[:, 1] if hasattr(model, 'predict_proba') else y_pred.astype(float)

                fold_metrics.append({
                    "accuracy": accuracy_score(y_val, y_pred),
                    "precision": precision_score(y_val, y_pred, zero_division=0),
                    "recall": recall_score(y_val, y_pred, zero_division=0),
                    "f1": f1_score(y_val, y_pred, zero_division=0),
                })

            if fold_metrics and last_model is not None:
                avg = {k: np.mean([m[k] for m in fold_metrics]) for k in fold_metrics[0]}
                all_oos_metrics[name] = avg

                # Step 4: Calibrate with Platt scaling
                try:
                    if len(np.unique(y)) >= 2:
                        calibrated = CalibratedClassifierCV(last_model, method="sigmoid", cv=3)
                        calibrated.fit(X, y)
                        self.calibrated_models[name] = calibrated
                    else:
                        last_model.fit(X, y)
                        self.calibrated_models[name] = last_model
                except Exception as e:
                    logger.warning("calibration_failed", model=name, error=str(e))
                    last_model.fit(X, y)
                    self.calibrated_models[name] = last_model

                logger.info("enhanced_model_trained", model=name,
                           f1=f"{avg['f1']:.3f}", precision=f"{avg['precision']:.3f}")

        self.oos_metrics = all_oos_metrics
        self._is_trained = bool(self.calibrated_models)

        return {
            "models_trained": list(self.calibrated_models.keys()),
            "oos_metrics": all_oos_metrics,
            "features_selected": len(self.selected_features),
            "features_total": len(features.columns),
        }

    def predict_with_disagreement(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Predict with ensemble and detect model disagreement.
        Returns calibrated probability with disagreement penalty.
        """
        if not self._is_trained or not self.calibrated_models:
            return {"probability": 50.0, "disagreement": 0, "adjusted": 50.0, "model_votes": {}}

        try:
            features = build_features(df)
            if features.empty or self.selected_features is None:
                return {"probability": 50.0, "disagreement": 0, "adjusted": 50.0, "model_votes": {}}

            # Select features
            available = [f for f in self.selected_features if f in features.columns]
            if not available:
                return {"probability": 50.0, "disagreement": 0, "adjusted": 50.0, "model_votes": {}}

            X = features[available].iloc[[-1]].values
            X = self.scaler.transform(X)

            # Get predictions from all models
            predictions = {}
            probabilities = []

            model_weights = {"lightgbm": 0.45, "random_forest": 0.35, "logistic_regression": 0.20}

            for name, model in self.calibrated_models.items():
                try:
                    if hasattr(model, 'predict_proba'):
                        prob = float(model.predict_proba(X)[0][1])
                    else:
                        prob = float(model.predict(X)[0])
                    predictions[name] = prob
                    w = model_weights.get(name, 0.33)
                    probabilities.append((prob, w))
                except Exception:
                    continue

            if not probabilities:
                return {"probability": 50.0, "disagreement": 0, "adjusted": 50.0, "model_votes": {}}

            # Weighted average probability
            total_w = sum(w for _, w in probabilities)
            avg_prob = sum(p * w for p, w in probabilities) / total_w if total_w > 0 else 0.5

            # Disagreement detection
            probs_only = [p for p, _ in probabilities]
            disagreement = np.std(probs_only) * 100  # 0-50 scale roughly

            # Apply disagreement penalty
            penalty = min(self.disagreement_penalty, disagreement * 0.5)
            adjusted_prob = avg_prob * 100
            if disagreement > 10:
                adjusted_prob = max(0, adjusted_prob - penalty)

            return {
                "probability": round(avg_prob * 100, 2),
                "disagreement": round(disagreement, 2),
                "penalty_applied": round(penalty, 2),
                "adjusted": round(adjusted_prob, 2),
                "model_votes": {k: round(v * 100, 1) for k, v in predictions.items()},
            }

        except Exception as e:
            logger.error("enhanced_predict_error", error=str(e))
            return {"probability": 50.0, "disagreement": 0, "adjusted": 50.0, "model_votes": {}}

    def _analyze_feature_importance(self, features: pd.DataFrame, labels: pd.Series) -> List[str]:
        """Analyze feature importance and select top features."""
        try:
            # Use RandomForest for feature importance
            rf = RandomForestClassifier(n_estimators=50, max_depth=8, random_state=42, n_jobs=-1)
            X = features.fillna(0).values
            y = labels.values

            if len(np.unique(y)) < 2:
                return list(features.columns)

            rf.fit(X, y)
            importance = pd.Series(rf.feature_importances_, index=features.columns)
            importance = importance.sort_values(ascending=False)

            self.feature_importance = importance

            # Select top N% features
            n_select = max(10, int(len(importance) * self.top_features_pct))
            selected = importance.head(n_select).index.tolist()

            logger.info("features_selected",
                       total=len(features.columns),
                       selected=len(selected),
                       top_5=list(importance.head(5).index))

            return selected

        except Exception as e:
            logger.warning("feature_importance_error", error=str(e))
            return list(features.columns)

    def _build_base_models(self) -> Dict[str, Any]:
        models = {
            "random_forest": RandomForestClassifier(
                n_estimators=150, max_depth=10, min_samples_leaf=5,
                min_samples_split=10, random_state=42, n_jobs=-1
            ),
            "logistic_regression": LogisticRegression(
                max_iter=1000, random_state=42, C=0.5, penalty="l2"
            ),
        }
        if HAS_LIGHTGBM:
            models["lightgbm"] = lgb.LGBMClassifier(
                n_estimators=250, max_depth=8, learning_rate=0.03,
                min_child_samples=15, random_state=42, verbose=-1,
                num_leaves=31, reg_alpha=0.1, reg_lambda=0.1,
                subsample=0.8, colsample_bytree=0.8,
            )
        return models

    @property
    def is_trained(self) -> bool:
        return self._is_trained

    @property
    def model_info(self) -> Dict[str, Any]:
        return {
            "is_trained": self._is_trained,
            "models": list(self.calibrated_models.keys()),
            "oos_metrics": self.oos_metrics,
            "features_selected": len(self.selected_features) if self.selected_features else 0,
            "top_features": list(self.feature_importance.head(10).index) if self.feature_importance is not None else [],
        }


# Singleton
_enhanced_ml: Optional[EnhancedMLPipeline] = None

def get_enhanced_ml() -> EnhancedMLPipeline:
    global _enhanced_ml
    if _enhanced_ml is None:
        _enhanced_ml = EnhancedMLPipeline()
    return _enhanced_ml