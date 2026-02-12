#!/usr/bin/env python3
"""
COURIER SYSTEM — MAIN PIPELINE CONTROLLER
Subsystem 3 — Intelligent Auto Courier Email System
ICT304 — Warehouse Intelligence System

Orchestrates the full pipeline:
  Barcode Scan → DB Retrieval → Feature Extraction →
  ML Prediction → Rule-Based Decisions → Send Email
"""

import json
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict, field

from order_database import OrderDatabase, Order, WAREHOUSE_INFO
from feature_engineering import FeatureEngineer, FeatureVector
from ai_decision_engine import AIDecisionEngine, ShippingDecision
from email_generator import EmailGenerator, EmailSender, EMAIL_CONFIG


# ============================================================================
# SHIPMENT RECORD
# ============================================================================

@dataclass
class ShipmentRecord:
    """Complete record of a processed shipment."""
    request_id: str
    order_id: str
    status: str = "pending"
    order_data: Optional[Dict] = None
    features: Optional[Dict] = None
    decision: Optional[Dict] = None
    email_subject: str = ""
    email_body: str = ""
    email_to: str = ""
    email_status: str = ""
    created_at: str = ""
    completed_at: str = ""
    error: str = ""

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()


# ============================================================================
# COURIER SYSTEM
# ============================================================================

class CourierSystem:
    """
    Intelligent Auto Courier Email System — Main Controller.

    Pipeline (fully autonomous):
    1. Barcode Scan → Order ID
    2. Database Retrieval → Order
    3. Feature Extraction → [weight, volume, distance, priority]
    4. ML Prediction → Shipping Method (Random Forest)
    5. Rule-Based → Fragility
    6. Fixed Template → Email to FedEx
    7. Send Email → Confirmation
    """

    def __init__(self, num_orders: int = 10, training_samples: int = 600):
        self.database = OrderDatabase(num_orders=num_orders)
        self.feature_engineer = FeatureEngineer()
        self.ai_engine = AIDecisionEngine(
            n_training_samples=training_samples
        )
        self.email_generator = EmailGenerator()
        self.email_sender = EmailSender()

        self.shipments: Dict[str, ShipmentRecord] = {}
        self._counter = 0

    def preview_shipment(self, order_id: str) -> Dict:
        """
        Step 1: Run AI pipeline and generate email draft (DO NOT SEND).
        """
        self._counter += 1
        request_id = f"REQ-{self._counter:05d}"
        record = ShipmentRecord(request_id=request_id, order_id=order_id)

        # 1. DB Retrieval
        order = self.database.scan_barcode(order_id)
        if not order:
            record.status = "not_found"
            record.error = f"Order {order_id} not found"
            self.shipments[request_id] = record
            return self._to_result(record)

        record.order_data = order.to_dict()

        # 2. Features
        features = self.feature_engineer.extract_features(order)
        record.features = features.to_dict()

        # 3. AI Decision
        decision = self.ai_engine.predict(features, order)
        record.decision = decision.to_dict()

        # 4. Generate Draft
        email_data = self.email_generator.generate(order, decision)
        record.email_subject = email_data["subject"]
        record.email_body = email_data["body"]
        record.email_to = email_data["to"]
        record.status = "preview"  # New status

        self.shipments[request_id] = record
        return self._to_result(record)

    def confirm_shipment(self, request_id: str) -> Dict:
        """
        Step 2: Send the email for a previewed shipment.
        """
        record = self.shipments.get(request_id)
        if not record:
            return {"error": "Request not found"}

        if record.status != "preview":
            return self._to_result(record)

        # Re-construct email data from record for sending
        email_data = {
            "to": record.email_to,
            "subject": record.email_subject,
            "body": record.email_body,
            "from": f"{EMAIL_CONFIG['sender_name']} <{EMAIL_CONFIG['sender_email']}>"
        }

        # 5. Send Email
        tx = self.email_sender.send(email_data, record.order_id)
        record.email_status = tx["status"]

        # 6. Finalize
        record.status = "confirmed" if tx["success"] else "failed"
        record.completed_at = datetime.now().isoformat()
        if not tx["success"]:
            record.error = tx["message"]

        return self._to_result(record)

    def process_shipment(self, order_id: str) -> Dict:
        """Legacy: Auto-process (Preview + Confirm immediately)."""
        result = self.preview_shipment(order_id)
        if result["status"] == "preview":
            return self.confirm_shipment(result["request_id"])
        return result

    def _to_result(self, record: ShipmentRecord) -> Dict:
        return {
            "request_id": record.request_id,
            "order_id": record.order_id,
            "status": record.status,
            "decision": record.decision,
            "email_subject": record.email_subject,
            "email_body": record.email_body,
            "email_to": record.email_to,
            "email_status": record.email_status,
            "created_at": record.created_at,
            "completed_at": record.completed_at,
            "error": record.error
        }

    # ── Dashboard Helpers ───────────────────────────────────────

    def get_dashboard_data(self) -> Dict:
        total = len(self.shipments)
        confirmed = sum(1 for s in self.shipments.values()
                        if s.status == "confirmed")
        failed = sum(1 for s in self.shipments.values()
                     if s.status == "failed")
        rate = (confirmed / total * 100) if total > 0 else 0

        recent = []
        for rid, rec in sorted(
            self.shipments.items(),
            key=lambda x: x[1].created_at, reverse=True
        )[:20]:
            d = rec.decision or {}
            recent.append({
                "request_id": rid,
                "order_id": rec.order_id,
                "status": rec.status,
                "shipping": d.get("shipping_method", "—"),
                "fragile": d.get("is_fragile", False),
                "confidence": d.get("shipping_confidence", 0),
                "created_at": rec.created_at[:19]
            })

        return {
            "summary": {
                "total": total,
                "confirmed": confirmed,
                "failed": failed,
                "not_found": total - confirmed - failed,
                "success_rate": round(rate, 1)
            },
            "recent": recent,
            "analytics": self.ai_engine.get_analytics(),
            "email_logs": self.email_sender.get_logs()
        }


# ============================================================================
# CLI DEMO
# ============================================================================

def run_demo():
    """Run full demonstration."""
    print("=" * 60)
    print("  INTELLIGENT AUTO COURIER EMAIL SYSTEM — DEMO")
    print("  ICT304 — Warehouse Intelligence System")
    print("=" * 60)

    system = CourierSystem(num_orders=100, training_samples=600)

    # Model info
    analytics = system.ai_engine.get_analytics()
    print(f"\n🤖 Model: {analytics['model']}")
    print(f"   Accuracy: {analytics['accuracy']:.1%}")
    print(f"   Features: {list(analytics['feature_importances'].keys())}")
    print(f"   Importance: {analytics['feature_importances']}")

    # Process 5 orders
    demo_ids = ["ORD-00001", "ORD-00010", "ORD-00025",
                "ORD-00050", "ORD-00075"]

    print(f"\n{'─' * 60}")
    for oid in demo_ids:
        result = system.process_shipment(oid)
        d = result.get("decision", {})
        icon = "✅" if result["status"] == "confirmed" else "❌"
        print(f"  {icon} {oid} → {d.get('shipping_method','?')} | "
              f"{'FRAGILE' if d.get('is_fragile') else 'OK'} | "
              f"FedEx | {result['status']}")

    # Bad order
    result = system.process_shipment("ORD-99999")
    print(f"  ❌ ORD-99999 → {result['status']} ({result['error']})")

    # Summary
    dash = system.get_dashboard_data()
    s = dash["summary"]
    print(f"\n{'─' * 60}")
    print(f"  📊 Total: {s['total']} | Confirmed: {s['confirmed']} | "
          f"Failed: {s['failed']} | Rate: {s['success_rate']}%")
    print("=" * 60)

    return system


if __name__ == "__main__":
    run_demo()
