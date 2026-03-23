"""
database.py
═══════════════════════════════════════════════════════════════════════
Warehouse Intelligence System — Centralised SQLite Database Layer

Tables
──────
  SS1  scan_sessions        one row per scanning run
       detections           every object detected in every image
       delivery_orders      PDF DO header data
       delivery_order_items expected quantities from the PDF DO
       do_comparisons       per-item match/discrepancy results
       asset_stickers       generated sticker records

  SS2  inventory            every warehouse SKU
       picker_orders        order header
       order_items          per-SKU lines within an order
       route_steps          individual stops in a picking route

  SS3  shipments            courier booking + email content

  XREF pipeline_events      audit log linking SS1→SS2→SS3

All public methods accept plain Python types (str, int, float, dict,
list) so callers (shared_state.py, app.py) never import sqlite3
directly.
═══════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

# ── Database file location ────────────────────────────────────────────
DB_PATH = Path("wis_warehouse.db")


# ═════════════════════════════════════════════════════════════════════
# SCHEMA DDL
# ═════════════════════════════════════════════════════════════════════

_DDL = """
-- ── SS1: Object Scanner ──────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS scan_sessions (
    session_id      TEXT PRIMARY KEY,
    started_at      TEXT NOT NULL DEFAULT (datetime('now')),
    total_images    INTEGER NOT NULL DEFAULT 0,
    total_items     INTEGER NOT NULL DEFAULT 0,
    do_id           TEXT,
    notes           TEXT,
    FOREIGN KEY (do_id) REFERENCES delivery_orders(do_id)
);

CREATE TABLE IF NOT EXISTS detections (
    detection_id    TEXT PRIMARY KEY,
    session_id      TEXT NOT NULL,
    file_name       TEXT NOT NULL,
    item_label      TEXT NOT NULL,
    confidence      REAL NOT NULL,
    brand           TEXT,
    brand_conf      REAL,
    detect_method   TEXT,
    bbox_x1         INTEGER,
    bbox_y1         INTEGER,
    bbox_x2         INTEGER,
    bbox_y2         INTEGER,
    position        INTEGER,
    detected_at     TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (session_id) REFERENCES scan_sessions(session_id)
);

CREATE TABLE IF NOT EXISTS delivery_orders (
    do_id           TEXT PRIMARY KEY,
    do_number       TEXT NOT NULL,
    supplier        TEXT,
    delivery_date   TEXT,
    session_id      TEXT,
    created_at      TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (session_id) REFERENCES scan_sessions(session_id)
);

CREATE TABLE IF NOT EXISTS delivery_order_items (
    doi_id          TEXT PRIMARY KEY,
    do_id           TEXT NOT NULL,
    item_label      TEXT NOT NULL,
    expected_qty    INTEGER NOT NULL,
    FOREIGN KEY (do_id) REFERENCES delivery_orders(do_id)
);

CREATE TABLE IF NOT EXISTS do_comparisons (
    comp_id         TEXT PRIMARY KEY,
    do_id           TEXT NOT NULL,
    session_id      TEXT NOT NULL,
    item_label      TEXT NOT NULL,
    expected_qty    INTEGER NOT NULL,
    actual_qty      INTEGER NOT NULL,
    difference      INTEGER NOT NULL,
    status          TEXT NOT NULL,
    compared_at     TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (do_id)      REFERENCES delivery_orders(do_id),
    FOREIGN KEY (session_id) REFERENCES scan_sessions(session_id)
);

CREATE TABLE IF NOT EXISTS asset_stickers (
    sticker_id      TEXT PRIMARY KEY,
    asset_id        TEXT NOT NULL UNIQUE,
    detection_id    TEXT,
    session_id      TEXT,
    item_label      TEXT NOT NULL,
    brand           TEXT,
    confidence      REAL,
    file_name       TEXT,
    generated_at    TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (detection_id) REFERENCES detections(detection_id),
    FOREIGN KEY (session_id)   REFERENCES scan_sessions(session_id)
);

-- ── SS2: Auto-Picker ─────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS inventory (
    sku             TEXT PRIMARY KEY,
    name            TEXT NOT NULL,
    category        TEXT NOT NULL,
    aisle           INTEGER NOT NULL,
    rack            INTEGER NOT NULL,
    quantity        INTEGER NOT NULL DEFAULT 0,
    weight_kg       REAL NOT NULL DEFAULT 0.0,
    value           REAL NOT NULL DEFAULT 0.0,
    dim_x           INTEGER,
    dim_y           INTEGER,
    dim_z           INTEGER,
    fragile         INTEGER NOT NULL DEFAULT 0,
    updated_at      TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS picker_orders (
    order_id        TEXT PRIMARY KEY,
    customer_id     TEXT NOT NULL,
    customer_type   TEXT NOT NULL,
    priority        INTEGER NOT NULL DEFAULT 3,
    status          TEXT NOT NULL DEFAULT 'pending',
    fulfillment     REAL NOT NULL DEFAULT 0.0,
    order_value     REAL NOT NULL DEFAULT 0.0,
    route_distance  REAL,
    route_time      REAL,
    route_json      TEXT,
    session_id      TEXT,
    created_at      TEXT NOT NULL DEFAULT (datetime('now')),
    completed_at    TEXT,
    FOREIGN KEY (session_id) REFERENCES scan_sessions(session_id)
);

CREATE TABLE IF NOT EXISTS order_items (
    item_id         TEXT PRIMARY KEY,
    order_id        TEXT NOT NULL,
    sku             TEXT NOT NULL,
    quantity        INTEGER NOT NULL,
    picked          INTEGER NOT NULL DEFAULT 0,
    item_status     TEXT NOT NULL DEFAULT 'pending',
    FOREIGN KEY (order_id) REFERENCES picker_orders(order_id)
    -- sku FK removed: inventory is managed in-memory; DB is persistence-only
);

CREATE TABLE IF NOT EXISTS route_steps (
    step_id         TEXT PRIMARY KEY,
    order_id        TEXT NOT NULL,
    step_number     INTEGER NOT NULL,
    location        TEXT NOT NULL,
    sku             TEXT,
    FOREIGN KEY (order_id) REFERENCES picker_orders(order_id)
);

-- ── SS3: Courier Email ───────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS shipments (
    shipment_id     TEXT PRIMARY KEY,
    order_id        TEXT NOT NULL,
    status          TEXT NOT NULL DEFAULT 'pending',
    shipping_method TEXT NOT NULL,
    courier         TEXT NOT NULL DEFAULT 'FedEx',
    is_fragile      INTEGER NOT NULL DEFAULT 0,
    ml_confidence   REAL,
    email_to        TEXT NOT NULL,
    email_subject   TEXT NOT NULL,
    email_body      TEXT NOT NULL,
    created_at      TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (order_id) REFERENCES picker_orders(order_id)
);

-- ── Cross-subsystem audit log ────────────────────────────────────────

CREATE TABLE IF NOT EXISTS pipeline_events (
    event_id        TEXT PRIMARY KEY,
    event_type      TEXT NOT NULL,
    subsystem       TEXT NOT NULL,
    reference_id    TEXT NOT NULL,
    description     TEXT,
    occurred_at     TEXT NOT NULL DEFAULT (datetime('now'))
);

-- ── Indexes ──────────────────────────────────────────────────────────

CREATE INDEX IF NOT EXISTS idx_detections_session  ON detections(session_id);
CREATE INDEX IF NOT EXISTS idx_detections_label    ON detections(item_label);
CREATE INDEX IF NOT EXISTS idx_order_items_order   ON order_items(order_id);
CREATE INDEX IF NOT EXISTS idx_order_items_sku     ON order_items(sku);
CREATE INDEX IF NOT EXISTS idx_route_steps_order   ON route_steps(order_id);
CREATE INDEX IF NOT EXISTS idx_shipments_order     ON shipments(order_id);
CREATE INDEX IF NOT EXISTS idx_pipeline_ref        ON pipeline_events(reference_id);
CREATE INDEX IF NOT EXISTS idx_do_items_do         ON delivery_order_items(do_id);
CREATE INDEX IF NOT EXISTS idx_do_comp_do          ON do_comparisons(do_id);
CREATE INDEX IF NOT EXISTS idx_stickers_session    ON asset_stickers(session_id);
"""


# ═════════════════════════════════════════════════════════════════════
# CONNECTION HELPER
# ═════════════════════════════════════════════════════════════════════

@contextmanager
def _conn(db_path: Path = DB_PATH):
    """Thread-safe SQLite connection with row_factory."""
    con = sqlite3.connect(str(db_path), check_same_thread=False)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA foreign_keys=ON")
    try:
        yield con
        con.commit()
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()


def _uid() -> str:
    return str(uuid.uuid4())


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


# ═════════════════════════════════════════════════════════════════════
# WISDatabase CLASS
# ═════════════════════════════════════════════════════════════════════

class WISDatabase:
    """
    All database operations for the Warehouse Intelligence System.
    Instantiate once via get_db(); the instance is cached in module scope.
    """

    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self._initialise()

    # ── Init ─────────────────────────────────────────────────────────

    def _initialise(self):
        with _conn(self.db_path) as con:
            con.executescript(_DDL)
            # Migration: rebuild order_items without the sku FK if the old
            # schema (with FK constraint) exists — SQLite cannot DROP CONSTRAINT
            # so we recreate the table only when needed.
            try:
                fk_info = con.execute("PRAGMA foreign_key_list(order_items)").fetchall()
                sku_fk = [r for r in fk_info if r["table"] == "inventory"]
                if sku_fk:
                    con.executescript("""
                        PRAGMA foreign_keys=OFF;
                        CREATE TABLE order_items_new (
                            item_id     TEXT PRIMARY KEY,
                            order_id    TEXT NOT NULL,
                            sku         TEXT NOT NULL,
                            quantity    INTEGER NOT NULL,
                            picked      INTEGER NOT NULL DEFAULT 0,
                            item_status TEXT NOT NULL DEFAULT 'pending',
                            FOREIGN KEY (order_id) REFERENCES picker_orders(order_id)
                        );
                        INSERT INTO order_items_new
                            SELECT item_id, order_id, sku, quantity, picked, item_status
                            FROM order_items;
                        DROP TABLE order_items;
                        ALTER TABLE order_items_new RENAME TO order_items;
                        CREATE INDEX IF NOT EXISTS idx_order_items_order
                            ON order_items(order_id);
                        CREATE INDEX IF NOT EXISTS idx_order_items_sku
                            ON order_items(sku);
                        PRAGMA foreign_keys=ON;
                    """)
            except Exception:
                pass  # table may not exist yet; DDL above handles it

    # ══════════════════════════════════════════════════════════════════
    # SS1 — SCAN SESSIONS
    # ══════════════════════════════════════════════════════════════════

    def create_scan_session(self, total_images: int = 0,
                            notes: str = None) -> str:
        sid = _uid()
        with _conn(self.db_path) as con:
            con.execute(
                "INSERT INTO scan_sessions (session_id, total_images, notes) "
                "VALUES (?, ?, ?)",
                (sid, total_images, notes),
            )
        self._log_event("scan_session_started", "SS1", sid,
                        f"New scan session: {total_images} image(s)")
        return sid

    def update_scan_session(self, session_id: str,
                            total_items: int = None,
                            do_id: str = None):
        with _conn(self.db_path) as con:
            if total_items is not None and do_id is not None:
                con.execute(
                    "UPDATE scan_sessions SET total_items=?, do_id=? "
                    "WHERE session_id=?",
                    (total_items, do_id, session_id),
                )
            elif total_items is not None:
                con.execute(
                    "UPDATE scan_sessions SET total_items=? WHERE session_id=?",
                    (total_items, session_id),
                )
            elif do_id is not None:
                con.execute(
                    "UPDATE scan_sessions SET do_id=? WHERE session_id=?",
                    (do_id, session_id),
                )

    def get_scan_sessions(self, limit: int = 20) -> List[Dict]:
        with _conn(self.db_path) as con:
            rows = con.execute(
                "SELECT * FROM scan_sessions ORDER BY started_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_detection_counts(self, session_id: str) -> Dict[str, int]:
        with _conn(self.db_path) as con:
            rows = con.execute(
                "SELECT item_label, COUNT(*) AS cnt FROM detections "
                "WHERE session_id=? GROUP BY item_label",
                (session_id,),
            ).fetchall()
        return {r["item_label"]: r["cnt"] for r in rows}

    # ── Detections ────────────────────────────────────────────────────

    def insert_detection(self, session_id: str, file_name: str,
                         item_label: str, confidence: float,
                         brand: str = None, brand_conf: float = None,
                         detect_method: str = None,
                         bbox: tuple = None,
                         position: int = None) -> str:
        did = _uid()
        x1 = y1 = x2 = y2 = None
        if bbox and len(bbox) == 4:
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        with _conn(self.db_path) as con:
            con.execute(
                """INSERT INTO detections
                   (detection_id, session_id, file_name, item_label,
                    confidence, brand, brand_conf, detect_method,
                    bbox_x1, bbox_y1, bbox_x2, bbox_y2, position)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (did, session_id, file_name, item_label,
                 confidence, brand, brand_conf, detect_method,
                 x1, y1, x2, y2, position),
            )
        return did

    def get_detections(self, session_id: str) -> List[Dict]:
        with _conn(self.db_path) as con:
            rows = con.execute(
                "SELECT * FROM detections WHERE session_id=? ORDER BY position",
                (session_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_all_detections(self, limit: int = 500) -> List[Dict]:
        with _conn(self.db_path) as con:
            rows = con.execute(
                "SELECT d.*, s.started_at AS session_started "
                "FROM detections d JOIN scan_sessions s "
                "  ON d.session_id = s.session_id "
                "ORDER BY d.detected_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    # ── Delivery Orders ───────────────────────────────────────────────

    def insert_delivery_order(self, do_number: str, supplier: str = None,
                              session_id: str = None,
                              delivery_date: str = None,
                              items: Dict[str, int] = None) -> str:
        do_id = _uid()
        with _conn(self.db_path) as con:
            con.execute(
                "INSERT INTO delivery_orders "
                "(do_id, do_number, supplier, delivery_date, session_id) "
                "VALUES (?,?,?,?,?)",
                (do_id, do_number, supplier, delivery_date, session_id),
            )
            if items:
                for label, qty in items.items():
                    con.execute(
                        "INSERT INTO delivery_order_items "
                        "(doi_id, do_id, item_label, expected_qty) "
                        "VALUES (?,?,?,?)",
                        (_uid(), do_id, label, qty),
                    )
        return do_id

    def get_delivery_order(self, do_id: str) -> Optional[Dict]:
        with _conn(self.db_path) as con:
            row = con.execute(
                "SELECT * FROM delivery_orders WHERE do_id=?", (do_id,)
            ).fetchone()
            if not row:
                return None
            result = dict(row)
            items = con.execute(
                "SELECT item_label, expected_qty FROM delivery_order_items "
                "WHERE do_id=?", (do_id,)
            ).fetchall()
            result["items"] = {r["item_label"]: r["expected_qty"] for r in items}
        return result

    def get_delivery_orders(self, limit: int = 50) -> List[Dict]:
        with _conn(self.db_path) as con:
            rows = con.execute(
                "SELECT * FROM delivery_orders ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    # ── DO Comparisons ────────────────────────────────────────────────

    def insert_do_comparison(self, do_id: str, session_id: str,
                             comparison: Dict):
        with _conn(self.db_path) as con:
            all_items = (comparison.get("matches", []) +
                         comparison.get("discrepancies", []))
            for item in all_items:
                raw_status = item.get("status", "")
                if "MATCH" in raw_status:
                    db_status = "MATCH"
                elif "UNEXPECTED" in raw_status:
                    db_status = "UNEXPECTED"
                else:
                    db_status = "DISCREPANCY"
                con.execute(
                    """INSERT INTO do_comparisons
                       (comp_id, do_id, session_id, item_label,
                        expected_qty, actual_qty, difference, status)
                       VALUES (?,?,?,?,?,?,?,?)""",
                    (_uid(), do_id, session_id,
                     item["item_type"], item["expected"],
                     item["actual"], item["difference"], db_status),
                )

    def get_do_comparisons(self, do_id: str) -> List[Dict]:
        with _conn(self.db_path) as con:
            rows = con.execute(
                "SELECT * FROM do_comparisons WHERE do_id=? "
                "ORDER BY compared_at DESC", (do_id,)
            ).fetchall()
        return [dict(r) for r in rows]

    # ── Asset Stickers ────────────────────────────────────────────────

    def insert_asset_sticker(self, asset_id: str,
                             detection_id: str = None,
                             session_id: str = None,
                             item_label: str = "",
                             brand: str = None,
                             confidence: float = None,
                             file_name: str = None) -> str:
        sid = _uid()
        with _conn(self.db_path) as con:
            con.execute(
                """INSERT OR IGNORE INTO asset_stickers
                   (sticker_id, asset_id, detection_id, session_id,
                    item_label, brand, confidence, file_name)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (sid, asset_id, detection_id, session_id,
                 item_label, brand, confidence, file_name),
            )
        return sid

    def get_asset_stickers(self, session_id: str = None,
                           limit: int = 200) -> List[Dict]:
        with _conn(self.db_path) as con:
            if session_id:
                rows = con.execute(
                    "SELECT * FROM asset_stickers WHERE session_id=? "
                    "ORDER BY generated_at DESC LIMIT ?",
                    (session_id, limit),
                ).fetchall()
            else:
                rows = con.execute(
                    "SELECT * FROM asset_stickers "
                    "ORDER BY generated_at DESC LIMIT ?",
                    (limit,),
                ).fetchall()
        return [dict(r) for r in rows]

    # ══════════════════════════════════════════════════════════════════
    # SS2 — INVENTORY
    # ══════════════════════════════════════════════════════════════════

    def seed_inventory(self, items: List[Dict]):
        """
        Bulk-insert inventory on first run. Skips rows that already exist.
        Each dict: sku, name, category, location{aisle,rack},
                   quantity, weight, value, dimensions, fragile
        """
        with _conn(self.db_path) as con:
            for item in items:
                loc  = item.get("location", {})
                dims = item.get("dimensions", (0, 0, 0))
                con.execute(
                    """INSERT OR IGNORE INTO inventory
                       (sku, name, category, aisle, rack,
                        quantity, weight_kg, value,
                        dim_x, dim_y, dim_z, fragile)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (item["sku"], item["name"], item["category"],
                     loc.get("aisle", 0), loc.get("rack", 0),
                     item["quantity"], item.get("weight", 0.5),
                     item["value"],
                     dims[0] if len(dims) > 0 else 0,
                     dims[1] if len(dims) > 1 else 0,
                     dims[2] if len(dims) > 2 else 0,
                     1 if item.get("fragile") else 0),
                )

    def get_inventory(self, category: str = None,
                      low_stock_threshold: int = None) -> List[Dict]:
        with _conn(self.db_path) as con:
            if category:
                rows = con.execute(
                    "SELECT * FROM inventory WHERE category=? ORDER BY sku",
                    (category,),
                ).fetchall()
            elif low_stock_threshold is not None:
                rows = con.execute(
                    "SELECT * FROM inventory WHERE quantity < ? ORDER BY quantity",
                    (low_stock_threshold,),
                ).fetchall()
            else:
                rows = con.execute(
                    "SELECT * FROM inventory ORDER BY sku"
                ).fetchall()
        return [dict(r) for r in rows]

    def get_inventory_item(self, sku: str) -> Optional[Dict]:
        with _conn(self.db_path) as con:
            row = con.execute(
                "SELECT * FROM inventory WHERE sku=?", (sku,)
            ).fetchone()
        return dict(row) if row else None

    def update_inventory_quantity(self, sku: str, new_quantity: int):
        with _conn(self.db_path) as con:
            con.execute(
                "UPDATE inventory SET quantity=?, updated_at=? WHERE sku=?",
                (max(0, new_quantity), _now(), sku),
            )

    def update_inventory_name(self, sku: str, name: str):
        """Persist a renamed inventory item (e.g. 'Laptop 1' → 'Dell Laptop')."""
        with _conn(self.db_path) as con:
            con.execute(
                "UPDATE inventory SET name=?, updated_at=? WHERE sku=?",
                (name, _now(), sku),
            )

    def inventory_stats(self) -> Dict:
        with _conn(self.db_path) as con:
            row = con.execute("""
                SELECT
                    COUNT(*)                                               AS total_skus,
                    COALESCE(SUM(quantity), 0)                             AS total_items,
                    COALESCE(SUM(quantity * value), 0.0)                   AS total_value,
                    COALESCE(SUM(CASE WHEN quantity < 10 THEN 1 ELSE 0 END), 0) AS low_stock_count
                FROM inventory
            """).fetchone()
        return dict(row) if row else {"total_skus":0,"total_items":0,"total_value":0.0,"low_stock_count":0}

    # ══════════════════════════════════════════════════════════════════
    # SS2 — PICKER ORDERS
    # ══════════════════════════════════════════════════════════════════

    def insert_picker_order(self, order_id: str, customer_id: str,
                            customer_type: str, priority: int,
                            items: List[Dict],
                            session_id: str = None) -> str:
        with _conn(self.db_path) as con:
            # Validate session_id — use NULL if it doesn't exist in scan_sessions
            # (prevents FK violation when picker is used without scanning first)
            safe_sid = None
            if session_id:
                row = con.execute(
                    "SELECT 1 FROM scan_sessions WHERE session_id=?", (session_id,)
                ).fetchone()
                safe_sid = session_id if row else None

            con.execute(
                """INSERT OR IGNORE INTO picker_orders
                   (order_id, customer_id, customer_type, priority, session_id)
                   VALUES (?,?,?,?,?)""",
                (order_id, customer_id, customer_type, priority, safe_sid),
            )
            for it in items:
                con.execute(
                    """INSERT OR IGNORE INTO order_items
                       (item_id, order_id, sku, quantity)
                       VALUES (?,?,?,?)""",
                    (_uid(), order_id, it["sku"], it["quantity"]),
                )
        self._log_event("order_created", "SS2", order_id,
                        f"Customer {customer_id} ({customer_type})")
        return order_id

    def update_picker_order_status(self, order_id: str, status: str,
                                   fulfillment: float = None,
                                   order_value: float = None,
                                   route: List[str] = None,
                                   route_distance: float = None,
                                   route_time: float = None):
        with _conn(self.db_path) as con:
            route_json = json.dumps(route) if route else None
            completed  = _now() if status in (
                "complete", "partially_fulfilled", "failed") else None
            con.execute(
                """UPDATE picker_orders SET
                   status=?,
                   fulfillment   = COALESCE(?, fulfillment),
                   order_value   = COALESCE(?, order_value),
                   route_json    = COALESCE(?, route_json),
                   route_distance= COALESCE(?, route_distance),
                   route_time    = COALESCE(?, route_time),
                   completed_at  = COALESCE(?, completed_at)
                   WHERE order_id=?""",
                (status, fulfillment, order_value, route_json,
                 route_distance, route_time, completed, order_id),
            )
        if status == "complete":
            self._log_event("order_picked", "SS2", order_id,
                            f"Fulfillment {fulfillment:.1f}%" if fulfillment else "")

    def update_order_item_pick(self, order_id: str, sku: str,
                               picked: int, item_status: str):
        with _conn(self.db_path) as con:
            con.execute(
                """UPDATE order_items SET picked=?, item_status=?
                   WHERE order_id=? AND sku=?""",
                (picked, item_status, order_id, sku),
            )

    def get_picker_order(self, order_id: str) -> Optional[Dict]:
        with _conn(self.db_path) as con:
            row = con.execute(
                "SELECT * FROM picker_orders WHERE order_id=?", (order_id,)
            ).fetchone()
            if not row:
                return None
            result = dict(row)
            items = con.execute(
                "SELECT oi.*, i.name, i.value FROM order_items oi "
                "LEFT JOIN inventory i ON oi.sku = i.sku "
                "WHERE oi.order_id=?", (order_id,)
            ).fetchall()
            result["items"] = [dict(r) for r in items]
            if result.get("route_json"):
                result["route"] = json.loads(result["route_json"])
        return result

    def get_picker_orders(self, status: str = None,
                          limit: int = 200) -> List[Dict]:
        with _conn(self.db_path) as con:
            if status:
                rows = con.execute(
                    "SELECT * FROM picker_orders WHERE status=? "
                    "ORDER BY created_at DESC LIMIT ?",
                    (status, limit),
                ).fetchall()
            else:
                rows = con.execute(
                    "SELECT * FROM picker_orders "
                    "ORDER BY created_at DESC LIMIT ?",
                    (limit,),
                ).fetchall()
        return [dict(r) for r in rows]

    def get_shipped_order_ids(self) -> Set[str]:
        with _conn(self.db_path) as con:
            rows = con.execute(
                "SELECT DISTINCT order_id FROM shipments WHERE status='confirmed'"
            ).fetchall()
        return {r["order_id"] for r in rows}

    # ── Route Steps ───────────────────────────────────────────────────

    def insert_route_steps(self, order_id: str, route: List[str]):
        with _conn(self.db_path) as con:
            con.execute("DELETE FROM route_steps WHERE order_id=?", (order_id,))
            for i, location in enumerate(route, 1):
                con.execute(
                    """INSERT INTO route_steps
                       (step_id, order_id, step_number, location)
                       VALUES (?,?,?,?)""",
                    (_uid(), order_id, i, location),
                )

    def get_route_steps(self, order_id: str) -> List[Dict]:
        with _conn(self.db_path) as con:
            rows = con.execute(
                "SELECT * FROM route_steps WHERE order_id=? "
                "ORDER BY step_number", (order_id,)
            ).fetchall()
        return [dict(r) for r in rows]

    # ══════════════════════════════════════════════════════════════════
    # SS3 — SHIPMENTS
    # ══════════════════════════════════════════════════════════════════

    def insert_shipment(self, shipment_id: str, order_id: str,
                        status: str, shipping_method: str,
                        courier: str, is_fragile: bool,
                        ml_confidence: float,
                        email_to: str, email_subject: str,
                        email_body: str) -> str:
        with _conn(self.db_path) as con:
            con.execute(
                """INSERT OR IGNORE INTO shipments
                   (shipment_id, order_id, status, shipping_method,
                    courier, is_fragile, ml_confidence,
                    email_to, email_subject, email_body)
                   VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (shipment_id, order_id, status, shipping_method,
                 courier, 1 if is_fragile else 0, ml_confidence,
                 email_to, email_subject, email_body),
            )
        self._log_event("email_sent", "SS3", shipment_id,
                        f"Order {order_id} → {courier} ({shipping_method})")
        return shipment_id

    def update_shipment_status(self, shipment_id: str, status: str):
        with _conn(self.db_path) as con:
            con.execute(
                "UPDATE shipments SET status=? WHERE shipment_id=?",
                (status, shipment_id),
            )

    def get_shipment(self, shipment_id: str) -> Optional[Dict]:
        with _conn(self.db_path) as con:
            row = con.execute(
                "SELECT * FROM shipments WHERE shipment_id=?",
                (shipment_id,),
            ).fetchone()
        return dict(row) if row else None

    def get_shipments(self, status: str = None,
                      limit: int = 200) -> List[Dict]:
        with _conn(self.db_path) as con:
            if status:
                rows = con.execute(
                    "SELECT * FROM shipments WHERE status=? "
                    "ORDER BY created_at DESC LIMIT ?",
                    (status, limit),
                ).fetchall()
            else:
                rows = con.execute(
                    "SELECT * FROM shipments ORDER BY created_at DESC LIMIT ?",
                    (limit,),
                ).fetchall()
        return [dict(r) for r in rows]

    def shipment_stats(self) -> Dict:
        with _conn(self.db_path) as con:
            row = con.execute("""
                SELECT
                    COUNT(*) AS total,
                    COALESCE(SUM(CASE WHEN status='confirmed' THEN 1 ELSE 0 END), 0) AS confirmed,
                    COALESCE(SUM(CASE WHEN status='failed'    THEN 1 ELSE 0 END), 0) AS failed
                FROM shipments
            """).fetchone()
        return dict(row) if row else {"total":0,"confirmed":0,"failed":0}

    # ══════════════════════════════════════════════════════════════════
    # CROSS-SUBSYSTEM QUERIES
    # ══════════════════════════════════════════════════════════════════

    def get_pipeline_summary(self) -> Dict:
        with _conn(self.db_path) as con:
            scans     = con.execute("SELECT COUNT(*) AS n FROM scan_sessions").fetchone()["n"]
            items     = con.execute("SELECT COALESCE(SUM(total_items),0) AS n FROM scan_sessions").fetchone()["n"]
            orders    = con.execute("SELECT COUNT(*) AS n FROM picker_orders").fetchone()["n"]
            completed = con.execute("SELECT COUNT(*) AS n FROM picker_orders WHERE status='complete'").fetchone()["n"]
            shipments = con.execute("SELECT COUNT(*) AS n FROM shipments").fetchone()["n"]
            confirmed = con.execute("SELECT COUNT(*) AS n FROM shipments WHERE status='confirmed'").fetchone()["n"]
        return {
            "scans":               scans,
            "items_detected":      items,
            "picker_orders":       orders,
            "completed_picks":     completed,
            "shipments":           shipments,
            "confirmed_shipments": confirmed,
        }

    def get_full_order_trail(self, order_id: str) -> Dict:
        """Complete lifecycle: scan → order lines → route → shipment."""
        with _conn(self.db_path) as con:
            order = con.execute(
                "SELECT * FROM picker_orders WHERE order_id=?", (order_id,)
            ).fetchone()
            if not order:
                return {}
            items = con.execute(
                "SELECT oi.*, i.name, i.category, i.value "
                "FROM order_items oi LEFT JOIN inventory i ON oi.sku=i.sku "
                "WHERE oi.order_id=?", (order_id,)
            ).fetchall()
            route = con.execute(
                "SELECT * FROM route_steps WHERE order_id=? ORDER BY step_number",
                (order_id,),
            ).fetchall()
            shipment = con.execute(
                "SELECT * FROM shipments WHERE order_id=? "
                "ORDER BY created_at DESC LIMIT 1", (order_id,)
            ).fetchone()
            events = con.execute(
                "SELECT * FROM pipeline_events WHERE reference_id=? "
                "ORDER BY occurred_at", (order_id,)
            ).fetchall()
        return {
            "order":    dict(order),
            "items":    [dict(r) for r in items],
            "route":    [dict(r) for r in route],
            "shipment": dict(shipment) if shipment else None,
            "events":   [dict(r) for r in events],
        }

    def get_analytics(self) -> Dict:
        with _conn(self.db_path) as con:
            det_by_label = con.execute("""
                SELECT item_label, COUNT(*) AS cnt,
                       AVG(confidence) AS avg_conf,
                       COUNT(brand) AS with_brand
                FROM detections
                GROUP BY item_label ORDER BY cnt DESC
            """).fetchall()

            brands = con.execute("""
                SELECT brand, COUNT(*) AS cnt FROM detections
                WHERE brand IS NOT NULL
                GROUP BY brand ORDER BY cnt DESC LIMIT 10
            """).fetchall()

            order_by_type = con.execute("""
                SELECT customer_type,
                       COUNT(*) AS orders,
                       AVG(fulfillment) AS avg_fulfillment,
                       SUM(order_value) AS total_value
                FROM picker_orders GROUP BY customer_type
            """).fetchall()

            shipping = con.execute("""
                SELECT shipping_method, COUNT(*) AS cnt,
                       AVG(ml_confidence) AS avg_confidence
                FROM shipments GROUP BY shipping_method
            """).fetchall()

            daily = con.execute("""
                SELECT DATE(occurred_at) AS day, subsystem, COUNT(*) AS events
                FROM pipeline_events
                WHERE occurred_at >= DATE('now', '-14 days')
                GROUP BY day, subsystem ORDER BY day, subsystem
            """).fetchall()

            do_acc = con.execute("""
                SELECT COUNT(*) AS total_items,
                       SUM(CASE WHEN status='MATCH' THEN 1 ELSE 0 END) AS matched
                FROM do_comparisons
            """).fetchone()

        return {
            "detections_by_label": [dict(r) for r in det_by_label],
            "brand_distribution":  [dict(r) for r in brands],
            "orders_by_type":      [dict(r) for r in order_by_type],
            "shipping_breakdown":  [dict(r) for r in shipping],
            "daily_activity":      [dict(r) for r in daily],
            "do_accuracy":         dict(do_acc) if do_acc else {},
        }

    def get_pipeline_events(self, limit: int = 100) -> List[Dict]:
        with _conn(self.db_path) as con:
            rows = con.execute(
                "SELECT * FROM pipeline_events ORDER BY occurred_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    # ══════════════════════════════════════════════════════════════════
    # ADMIN / MAINTENANCE
    # ══════════════════════════════════════════════════════════════════

    def _log_event(self, event_type: str, subsystem: str,
                   reference_id: str, description: str = None):
        try:
            with _conn(self.db_path) as con:
                con.execute(
                    """INSERT INTO pipeline_events
                       (event_id, event_type, subsystem, reference_id, description)
                       VALUES (?,?,?,?,?)""",
                    (_uid(), event_type, subsystem, reference_id, description),
                )
        except Exception:
            pass  # Never let logging crash the main flow

    def log_event(self, event_type: str, subsystem: str,
                  reference_id: str, description: str = None):
        self._log_event(event_type, subsystem, reference_id, description)

    def purge_old_sessions(self, keep_days: int = 30):
        with _conn(self.db_path) as con:
            con.execute(
                "DELETE FROM scan_sessions WHERE started_at < DATE('now', ?)",
                (f"-{keep_days} days",),
            )

    def export_full_db(self) -> Dict:
        """Dump entire database as JSON-serialisable dict."""
        tables = [
            "scan_sessions", "detections",
            "delivery_orders", "delivery_order_items", "do_comparisons",
            "asset_stickers",
            "inventory", "picker_orders", "order_items", "route_steps",
            "shipments", "pipeline_events",
        ]
        dump = {}
        with _conn(self.db_path) as con:
            for table in tables:
                rows = con.execute(f"SELECT * FROM {table}").fetchall()
                dump[table] = [dict(r) for r in rows]
        dump["exported_at"] = _now()
        return dump

    def get_db_stats(self) -> Dict:
        """Row counts for every table."""
        tables = [
            "scan_sessions", "detections",
            "delivery_orders", "delivery_order_items", "do_comparisons",
            "asset_stickers",
            "inventory", "picker_orders", "order_items", "route_steps",
            "shipments", "pipeline_events",
        ]
        stats = {}
        with _conn(self.db_path) as con:
            for table in tables:
                row = con.execute(
                    f"SELECT COUNT(*) AS n FROM {table}"
                ).fetchone()
                stats[table] = row["n"]
        return stats


# ═════════════════════════════════════════════════════════════════════
# MODULE-LEVEL SINGLETON
# ═════════════════════════════════════════════════════════════════════

_db_instance: Optional[WISDatabase] = None


def get_db(db_path: Path = DB_PATH) -> WISDatabase:
    """Return the module-level WISDatabase singleton."""
    global _db_instance
    if _db_instance is None:
        _db_instance = WISDatabase(db_path)
    return _db_instance


def reset_db(db_path: Path = DB_PATH):
    """
    Wipe all data from every table and reset the module singleton.
    Called by the UI reset button so data does not bleed across sessions.
    Preserves the database file and schema — only truncates rows.
    """
    global _db_instance
    tables = [
        "pipeline_events",
        "shipments",
        "route_steps",
        "order_items",
        "picker_orders",
        "asset_stickers",
        "do_comparisons",
        "delivery_order_items",
        "delivery_orders",
        "detections",
        "scan_sessions",
        "inventory",          # re-seeded on next auto_picker init
    ]
    with _conn(db_path) as con:
        con.execute("PRAGMA foreign_keys=OFF")
        for table in tables:
            con.execute(f"DELETE FROM {table}")
        con.execute("PRAGMA foreign_keys=ON")
    # Reset the singleton so next get_db() gets a fresh instance
    _db_instance = None
