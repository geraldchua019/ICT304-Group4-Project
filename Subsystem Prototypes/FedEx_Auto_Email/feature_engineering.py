#!/usr/bin/env python3
"""
FEATURE ENGINEERING MODULE
Subsystem 3 — Intelligent Auto Courier Email System
ICT304 — Warehouse Intelligence System

Converts raw order data into ML-ready features for the
Shipping Method Random Forest classifier.
"""

import numpy as np
from typing import Dict, List
from dataclasses import dataclass, asdict


# ============================================================================
# FEATURE VECTOR
# ============================================================================

PRIORITY_MAP = {"Urgent": 1, "High": 2, "Standard": 3}

@dataclass
class FeatureVector:
    """ML-ready feature vector for the Shipping Method classifier."""
    weight_kg: float
    volume_cm3: float
    distance_km: float
    priority_encoded: int        # 1=Urgent, 2=High, 3=Standard

    def to_array(self) -> np.ndarray:
        """Convert to numpy array for the ML model."""
        return np.array([
            self.weight_kg,
            self.volume_cm3,
            self.distance_km,
            self.priority_encoded
        ])

    def to_dict(self) -> Dict:
        return asdict(self)

    @staticmethod
    def feature_names() -> List[str]:
        return ["weight_kg", "volume_cm3", "distance_km", "priority"]


# ============================================================================
# FEATURE ENGINEER
# ============================================================================

class FeatureEngineer:
    """Transforms raw order data into ML-ready features."""

    def extract_features(self, order) -> FeatureVector:
        """
        Convert an Order object into a FeatureVector.

        Args:
            order: Order dataclass from order_database

        Returns:
            FeatureVector ready for the Random Forest model
        """
        volume = order.length_cm * order.width_cm * order.height_cm
        priority_enc = PRIORITY_MAP.get(order.order_priority, 3)

        return FeatureVector(
            weight_kg=order.weight_kg,
            volume_cm3=round(volume, 2),
            distance_km=order.distance_km,
            priority_encoded=priority_enc
        )

    def extract_batch(self, orders: list) -> np.ndarray:
        """Extract features from a list of orders → feature matrix."""
        vectors = [self.extract_features(o) for o in orders]
        return np.array([v.to_array() for v in vectors])
