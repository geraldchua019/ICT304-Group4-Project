#!/usr/bin/env python3
"""
AI DECISION ENGINE
Subsystem 3 — Intelligent Auto Courier Email System
ICT304 — Warehouse Intelligence System

Single AI model: Random Forest for Shipping Method classification.
Fragility is rule-based. Courier is always FedEx.
"""

import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass, asdict
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


# ============================================================================
# CONFIGURATION
# ============================================================================

SHIPPING_METHODS = ["Same-Day", "Express", "Standard"]

# Single delivery partner
COURIER = "FedEx"
COURIER_EMAIL = "pickup@fedex.com"


# ============================================================================
# DECISION RESULT
# ============================================================================

@dataclass
class ShippingDecision:
    """Container for the complete shipping decision."""
    # ML prediction
    shipping_method: str
    shipping_confidence: float
    shipping_probabilities: Dict[str, float]

    # Rule-based
    is_fragile: bool

    # Meta
    model_name: str = "Random Forest"
    accuracy: float = 0.0

    def to_dict(self) -> Dict:
        return asdict(self)

    def summary(self) -> str:
        frag = "⚠️ FRAGILE" if self.is_fragile else "✅ Non-Fragile"
        return (
            f"📦 {self.shipping_method} ({self.shipping_confidence:.0%}) | "
            f"{frag}"
        )


# ============================================================================
# TRAINING DATA GENERATOR
# ============================================================================

class TrainingDataGenerator:
    """
    Generates synthetic labelled data for the shipping method classifier.

    Rules:
    - Same-Day: local zone, urgent, lightweight
    - Express:  high priority, long distance, high value
    - Standard: low priority, heavy, short distance
    """

    def __init__(self, n_samples: int = 600):
        self.n = n_samples
        np.random.seed(42)

    def generate(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns (X, y) where:
            X: [weight_kg, volume_cm3, distance_km, priority]
            y: 0=Same-Day, 1=Express, 2=Standard
        """
        X, y = [], []

        for _ in range(self.n):
            weight = np.random.uniform(0.1, 50)
            volume = np.random.uniform(500, 500_000)
            distance = np.random.uniform(1, 15_000)
            priority = np.random.choice([1, 2, 3], p=[0.2, 0.3, 0.5])

            # ── Decision rules with noise ──
            if priority == 1 and distance < 20 and weight < 10:
                label = 0   # Same-Day
            elif priority <= 2 and distance > 500:
                label = 1   # Express
            elif priority == 1 and distance > 100:
                label = 1   # Express
            elif priority == 3 and weight > 15:
                label = 2   # Standard
            elif distance < 50 and weight < 5:
                label = 0   # Same-Day
            elif distance > 3000:
                label = 1   # Express
            else:
                # Weighted random for edge cases
                label = np.random.choice([0, 1, 2], p=[0.15, 0.35, 0.50])

            X.append([weight, volume, distance, priority])
            y.append(label)

        return np.array(X), np.array(y)


# ============================================================================
# AI DECISION ENGINE
# ============================================================================

class AIDecisionEngine:
    """
    Single Random Forest classifier for shipping method.
    All other decisions are rule-based.
    """

    def __init__(self, n_training_samples: int = 600):
        self.model = RandomForestClassifier(
            n_estimators=100, max_depth=8, random_state=42
        )
        self.accuracy = 0.0
        self.feature_importances: Dict[str, float] = {}
        self.classification_report_text: str = ""
        self.is_trained = False
        self.training_samples = n_training_samples

        self._train()

    def _train(self):
        """Train the Random Forest on synthetic data."""
        gen = TrainingDataGenerator(n_samples=self.training_samples)
        X, y = gen.generate()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)

        self.accuracy = accuracy_score(y_test, y_pred)
        self.classification_report_text = classification_report(
            y_test, y_pred,
            target_names=SHIPPING_METHODS,
            zero_division=0
        )

        names = ["weight_kg", "volume_cm3", "distance_km", "priority"]
        self.feature_importances = dict(
            zip(names, self.model.feature_importances_.tolist())
        )
        self.is_trained = True

    def predict(self, feature_vector, order) -> ShippingDecision:
        """
        Make a complete shipping decision.

        - Shipping method: ML (Random Forest)
        - Fragility: Rule-based (order.is_fragile)
        - Courier: Always FedEx (single partner)

        Args:
            feature_vector: FeatureVector from feature_engineering
            order: Order from order_database

        Returns:
            ShippingDecision with all fields populated
        """
        X = feature_vector.to_array().reshape(1, -1)

        # ── ML Prediction ──
        pred = self.model.predict(X)[0]
        proba = self.model.predict_proba(X)[0]
        method = SHIPPING_METHODS[pred]
        confidence = float(proba[pred])
        probs = {SHIPPING_METHODS[i]: round(float(p), 4)
                 for i, p in enumerate(proba)}

        # ── Rule-based: Fragility ──
        is_fragile = order.is_fragile

        return ShippingDecision(
            shipping_method=method,
            shipping_confidence=round(confidence, 4),
            shipping_probabilities=probs,
            is_fragile=is_fragile,
            model_name="Random Forest",
            accuracy=round(self.accuracy, 4)
        )

    def get_analytics(self) -> Dict:
        """Return model training metrics."""
        return {
            "model": "Random Forest",
            "accuracy": round(self.accuracy, 4),
            "training_samples": self.training_samples,
            "feature_importances": self.feature_importances,
            "classification_report": self.classification_report_text,
            "shipping_methods": SHIPPING_METHODS,
            "courier": COURIER,
            "n_estimators": 100,
            "max_depth": 8
        }
