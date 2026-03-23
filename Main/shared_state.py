"""
shared_state.py  (v2 — database-backed)
────────────────────────────────────────────────────────────────────────
Centralised session-state + SQLite bridge for the Warehouse Intelligence
System.

Every public method that mutates state:
  1. Updates the in-memory SharedStore (for fast UI reads within the session)
  2. Calls the matching WISDatabase method (for persistence across sessions)

Subsystem ownership
  SS1 (Object Scanner)  → push_scan, push_delivery_order, push_do_comparison,
                           push_sticker
  SS2 (Auto-Picker)     → sync_inventory, push_picker_order
  SS3 (Email / Courier) → push_shipment
────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from database import get_db, WISDatabase


# ──────────────────────────────────────────────────────────────────────
# SHARED DATA TYPES  (unchanged — app.py imports these)
# ──────────────────────────────────────────────────────────────────────

@dataclass
class ScanDetection:
    label: str
    confidence: float
    brands: List[Dict]
    position: int
    bbox: tuple
    detection_id: Optional[str] = None   # set after DB insert


@dataclass
class ScanResult:
    file_name: str
    count: int
    detections: List[ScanDetection]
    annotated_image: Any          # np.ndarray — never written to DB
    scanned_at: str = field(default_factory=lambda: datetime.now().isoformat())
    session_id: Optional[str] = None   # set after DB insert


@dataclass
class PickerOrder:
    order_id: str
    customer_id: str
    customer_type: str
    status: str
    items: List[Dict]             # [{sku, name, quantity, picked, value}]
    route: List[str]
    route_metrics: Dict
    fulfillment: float = 0.0
    order_value: float = 0.0
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    completed_at: Optional[str] = None


@dataclass
class ShipmentRecord:
    shipment_id: str
    order_id: str
    status: str
    shipping_method: str
    courier: str
    is_fragile: bool
    email_to: str
    email_subject: str
    email_body: str
    ml_confidence: float = 0.0
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


# ──────────────────────────────────────────────────────────────────────
# CENTRAL STORE
# ──────────────────────────────────────────────────────────────────────

class SharedStore:
    """
    Single instance kept in st.session_state under key 'wis_store'.
    Access via get_store().

    In-memory dicts/lists serve the UI within a session.
    Every mutation also calls get_db() so data survives restarts.
    """

    def __init__(self):
        db = get_db()               # ensure DB is initialised

        # ── SS1 ────────────────────────────────────────────────────
        self.scan_results: List[ScanResult] = []
        self.actual_counts: Dict[str, int] = {}
        self.do_data: Optional[Dict] = None
        self.do_comparison: Optional[Dict] = None
        self.current_session_id: Optional[str] = None
        self.scan_ready: bool = False

        # ── SS2 ────────────────────────────────────────────────────
        self.picker_orders: Dict[str, PickerOrder] = {}
        self.inventory: Dict[str, Any] = {}   # ElectronicsItem objects
        self.picker_ready: bool = False

        # ── SS3 ────────────────────────────────────────────────────
        self.shipments: Dict[str, ShipmentRecord] = {}
        self.last_email: Optional[Dict] = None
        self.courier_ready: bool = False

        # ── Pipeline ───────────────────────────────────────────────
        self.pipeline_step: int = 1

        # ── Restore persisted data into memory ────────────────────
        self._restore_from_db(db)

    # ──────────────────────────────────────────────────────────────
    # RESTORE
    # ──────────────────────────────────────────────────────────────

    def _restore_from_db(self, db: WISDatabase):
        """
        Populate in-memory collections from the SQLite database so the
        UI reflects data from previous sessions without re-scanning.
        Annotated images are NOT restored (they are transient).
        """
        # Picker orders — also restore their line items from DB
        for row in db.get_picker_orders():
            full = db.get_picker_order(row["order_id"])
            db_items = full.get("items", []) if full else []
            restored_items = [
                {
                    "sku":      it.get("sku", ""),
                    "name":     it.get("name") or it.get("sku", ""),
                    "quantity": it.get("quantity", 0),
                    "picked":   it.get("picked", 0),
                    "value":    (it.get("value") or 0.0),
                    "status":   it.get("item_status", "pending"),
                }
                for it in db_items
            ]
            po = PickerOrder(
                order_id=row["order_id"],
                customer_id=row["customer_id"],
                customer_type=row["customer_type"],
                status=row["status"],
                items=restored_items,
                route=full.get("route", []) if full else [],
                route_metrics={},
                fulfillment=row.get("fulfillment", 0.0),
                order_value=row.get("order_value", 0.0),
                created_at=row.get("created_at", ""),
                completed_at=row.get("completed_at"),
            )
            self.picker_orders[po.order_id] = po
        if self.picker_orders:
            self.picker_ready = True

        # Shipments
        for row in db.get_shipments():
            sr = ShipmentRecord(
                shipment_id=row["shipment_id"],
                order_id=row["order_id"],
                status=row["status"],
                shipping_method=row["shipping_method"],
                courier=row["courier"],
                is_fragile=bool(row["is_fragile"]),
                email_to=row["email_to"],
                email_subject=row["email_subject"],
                email_body=row["email_body"],
                ml_confidence=row.get("ml_confidence") or 0.0,
                created_at=row.get("created_at", ""),
            )
            self.shipments[sr.shipment_id] = sr
        if self.shipments:
            self.courier_ready = True

        # Actual counts from latest scan session
        # Verify the session actually exists (guards against stale IDs after DB reset)
        sessions = db.get_scan_sessions(limit=1)
        if sessions:
            latest = sessions[0]
            self.current_session_id = latest["session_id"]
            self.actual_counts = db.get_detection_counts(latest["session_id"])
            self.scan_ready = bool(self.actual_counts)
        else:
            # No sessions in DB — ensure current_session_id is cleared
            self.current_session_id = None

    # ──────────────────────────────────────────────────────────────
    # SS1 — SCAN
    # ──────────────────────────────────────────────────────────────

    def start_scan_session(self, total_images: int, notes: str = None) -> str:
        """Create a DB scan session and return its ID."""
        sid = get_db().create_scan_session(total_images=total_images, notes=notes)
        self.current_session_id = sid
        return sid

    def push_scan(self, scan: ScanResult):
        """
        Store scan in memory and persist each detection to DB.
        scan.session_id should be set before calling (uses current_session_id as fallback).
        """
        self.scan_results.append(scan)
        db = get_db()
        sid = scan.session_id or self.current_session_id

        for det in scan.detections:
            top_brand = det.brands[0] if det.brands else None
            did = db.insert_detection(
                session_id=sid,
                file_name=scan.file_name,
                item_label=det.label,
                confidence=det.confidence,
                brand=top_brand["text"] if top_brand else None,
                brand_conf=top_brand["confidence"] if top_brand else None,
                detect_method=top_brand.get("detection_method") if top_brand else None,
                bbox=det.bbox,
                position=det.position,
            )
            det.detection_id = did
            self.actual_counts[det.label] = self.actual_counts.get(det.label, 0) + 1

        db.update_scan_session(sid, total_items=sum(self.actual_counts.values()))
        self.scan_ready = True

    def push_delivery_order(self, do_data: Dict, session_id: str) -> str:
        """Persist PDF DO to DB and return do_id."""
        self.do_data = do_data
        do_id = get_db().insert_delivery_order(
            do_number=do_data["metadata"].get("do_number", "N/A"),
            supplier=do_data["metadata"].get("supplier", "N/A"),
            session_id=session_id,
            delivery_date=do_data["metadata"].get("delivery_date"),
            items=do_data.get("items", {}),
        )
        get_db().update_scan_session(session_id, total_items=0, do_id=do_id)
        return do_id

    def push_do_comparison(self, do_id: str, session_id: str, comparison: Dict):
        self.do_comparison = comparison
        get_db().insert_do_comparison(do_id, session_id, comparison)

    def push_sticker(self, asset_id: str, detection_id: str,
                     session_id: str, item_label: str,
                     brand: str = None, confidence: float = None,
                     file_name: str = None):
        get_db().insert_asset_sticker(
            asset_id=asset_id, detection_id=detection_id,
            session_id=session_id, item_label=item_label,
            brand=brand, confidence=confidence, file_name=file_name,
        )

    def clear_scans(self):
        """Clear in-memory scan state (DB history preserved)."""
        self.scan_results.clear()
        self.actual_counts.clear()
        self.do_data = None
        self.do_comparison = None
        self.scan_ready = False

    # ──────────────────────────────────────────────────────────────
    # SS2 — PICKER ORDERS
    # ──────────────────────────────────────────────────────────────

    def sync_inventory(self, items: List[Dict]):
        """
        Called once by StreamlitAutoPicker.__init__ to seed the DB.
        items: [{"sku","name","category",
                 "location":{"aisle":int,"rack":int},
                 "quantity","weight","value","dimensions","fragile"}]
        """
        get_db().seed_inventory(items)

    def push_picker_order(self, order: PickerOrder):
        """Persist a new/updated order to DB and add to memory."""
        self.picker_orders[order.order_id] = order
        self.picker_ready = True

        db = get_db()
        existing = db.get_picker_order(order.order_id)
        if not existing:
            db.insert_picker_order(
                order_id=order.order_id,
                customer_id=order.customer_id,
                customer_type=order.customer_type,
                priority={"wholesale": 1, "repair_shop": 2, "retail": 3}.get(
                    order.customer_type, 3),
                items=[{"sku": it["sku"], "quantity": it["quantity"]}
                       for it in order.items],
                session_id=self.current_session_id,
            )
        else:
            db.update_picker_order_status(
                order_id=order.order_id,
                status=order.status,
                fulfillment=order.fulfillment,
                order_value=order.order_value,
            )

    def update_picker_order_in_db(self, order: PickerOrder):
        """Sync an already-existing order's status + picks to DB."""
        db = get_db()
        db.update_picker_order_status(
            order_id=order.order_id,
            status=order.status,
            fulfillment=order.fulfillment,
            order_value=order.order_value,
            route=order.route or None,
            route_distance=order.route_metrics.get("total_distance_m") if order.route_metrics else None,
            route_time=order.route_metrics.get("total_time_min") if order.route_metrics else None,
        )
        for it in order.items:
            db.update_order_item_pick(
                order_id=order.order_id,
                sku=it["sku"],
                picked=it.get("picked", 0),
                item_status=it.get("status", "pending"),
            )
        # Sync live inventory quantities
        for it in order.items:
            inv_item = self.inventory.get(it["sku"])
            if inv_item:
                db.update_inventory_quantity(it["sku"], inv_item.quantity)

    # ──────────────────────────────────────────────────────────────
    # SS3 — SHIPMENTS
    # ──────────────────────────────────────────────────────────────

    def push_shipment(self, shipment: ShipmentRecord):
        self.shipments[shipment.shipment_id] = shipment
        get_db().insert_shipment(
            shipment_id=shipment.shipment_id,
            order_id=shipment.order_id,
            status=shipment.status,
            shipping_method=shipment.shipping_method,
            courier=shipment.courier,
            is_fragile=shipment.is_fragile,
            ml_confidence=shipment.ml_confidence,
            email_to=shipment.email_to,
            email_subject=shipment.email_subject,
            email_body=shipment.email_body,
        )

    # ──────────────────────────────────────────────────────────────
    # QUERY HELPERS
    # ──────────────────────────────────────────────────────────────

    def get_completed_orders(self) -> List[PickerOrder]:
        return [o for o in self.picker_orders.values()
                if o.status == "complete"]

    def get_pending_picker_orders(self) -> List[PickerOrder]:
        return [o for o in self.picker_orders.values()
                if o.status == "pending"]

    def get_unshipped_orders(self) -> List[PickerOrder]:
        shipped_ids = {s.order_id for s in self.shipments.values()
                       if s.status == "confirmed"}
        shipped_ids |= get_db().get_shipped_order_ids()
        return [o for o in self.picker_orders.values()
                if o.status == "complete" and o.order_id not in shipped_ids]

    def pipeline_summary(self) -> Dict:
        db_summary = get_db().get_pipeline_summary()
        return {
            "scans":               max(len(self.scan_results),
                                       db_summary.get("scans", 0)),
            "items_detected":      max(sum(self.actual_counts.values()),
                                       db_summary.get("items_detected", 0)),
            "picker_orders":       max(len(self.picker_orders),
                                       db_summary.get("picker_orders", 0)),
            "completed_picks":     max(len(self.get_completed_orders()),
                                       db_summary.get("completed_picks", 0)),
            "shipments":           max(len(self.shipments),
                                       db_summary.get("shipments", 0)),
            "confirmed_shipments": max(
                sum(1 for s in self.shipments.values() if s.status == "confirmed"),
                db_summary.get("confirmed_shipments", 0),
            ),
        }


# ──────────────────────────────────────────────────────────────────────
# ACCESSOR
# ──────────────────────────────────────────────────────────────────────

def get_store() -> SharedStore:
    import streamlit as st
    if "wis_store" not in st.session_state:
        st.session_state.wis_store = SharedStore()
    return st.session_state.wis_store
