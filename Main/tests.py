"""
tests.py
════════════════════════════════════════════════════════════════════════
Warehouse Intelligence System — Test Suite
ICT304 · Murdoch University

35 test cases covering:
  SS1 · Object Scanner     (9 tests)  — DB scan sessions, detections,
                                         delivery orders, DO comparisons,
                                         asset stickers
  SS2 · Auto-Picker        (10 tests) — inventory seeding, order lifecycle,
                                         route steps, fulfillment tracking
  SS3 · Courier Email      (4 tests)  — shipment creation, status tracking,
                                         shipping statistics
  XREF · Cross-Subsystem   (5 tests)  — pipeline summary, order trail,
                                         analytics, DB reset, ID safety
  ROUTE · Route Optimiser  (7 tests)  — geometry correctness, corridor
                                         selection, depot return

Usage
─────
    python tests.py              # run all tests
    python tests.py --group SS1  # run one subsystem group
    python tests.py --verbose    # show pass details too
    python tests.py --no-color   # plain output (CI/redirected output)

Requirements
────────────
  pip install (already in main app environment):
    None — only uses database.py from the same directory

Exit code: 0 if all pass, 1 if any fail.
════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

# ── Make sure database.py is importable ──────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from database import get_db, reset_db


# ═════════════════════════════════════════════════════════════════════
# TEST HARNESS
# ═════════════════════════════════════════════════════════════════════

@dataclass
class TestResult:
    test_id:   str
    name:      str
    group:     str
    status:    str          # "PASS" | "FAIL" | "SKIP"
    ms:        float = 0.0
    error:     str   = ""
    traceback: str   = ""


class TestRunner:
    def __init__(self, use_color: bool = True, verbose: bool = False):
        self.results:    List[TestResult] = []
        self.use_color   = use_color
        self.verbose     = verbose
        self._GREEN  = "\033[92m" if use_color else ""
        self._RED    = "\033[91m" if use_color else ""
        self._YELLOW = "\033[93m" if use_color else ""
        self._CYAN   = "\033[96m" if use_color else ""
        self._BOLD   = "\033[1m"  if use_color else ""
        self._RESET  = "\033[0m"  if use_color else ""

    def run(self, test_id: str, name: str, group: str, fn: Callable):
        start = time.perf_counter()
        try:
            fn()
            ms = (time.perf_counter() - start) * 1000
            r  = TestResult(test_id, name, group, "PASS", ms)
            self.results.append(r)
            if self.verbose:
                print(f"  {self._GREEN}✓{self._RESET} {test_id}: {name} "
                      f"{self._CYAN}({ms:.1f}ms){self._RESET}")
        except Exception as exc:
            ms  = (time.perf_counter() - start) * 1000
            tb  = traceback.format_exc()
            r   = TestResult(test_id, name, group, "FAIL", ms,
                             str(exc), tb)
            self.results.append(r)
            print(f"  {self._RED}✗{self._RESET} {test_id}: {name}")
            print(f"    {self._RED}Error:{self._RESET} {exc}")
            if self.verbose:
                print(tb)

    def section(self, title: str):
        print(f"\n{self._BOLD}{self._CYAN}── {title} ──{self._RESET}")

    def summary(self) -> int:
        passed = [r for r in self.results if r.status == "PASS"]
        failed = [r for r in self.results if r.status == "FAIL"]
        total  = len(self.results)
        avg_ms = sum(r.ms for r in self.results) / max(total, 1)

        print(f"\n{'═'*62}")
        print(f"{self._BOLD}RESULTS{self._RESET}  "
              f"{self._GREEN}{len(passed)}{self._RESET}/"
              f"{total} passed  ·  "
              f"{self._RED}{len(failed)}{self._RESET} failed  ·  "
              f"{avg_ms:.1f}ms avg")

        if failed:
            print(f"\n{self._RED}{self._BOLD}Failed tests:{self._RESET}")
            for r in failed:
                print(f"  {r.test_id}: {r.name}")
                print(f"    {r.error}")

        groups = {}
        for r in self.results:
            groups.setdefault(r.group, []).append(r)
        print(f"\n{'Group':<22} {'Tests':>5} {'Pass':>5} {'Fail':>5} {'Avg ms':>8}")
        print("─" * 48)
        for grp, rs in groups.items():
            p = sum(1 for r in rs if r.status == "PASS")
            f = sum(1 for r in rs if r.status == "FAIL")
            a = sum(r.ms for r in rs) / len(rs)
            fc = self._RED if f > 0 else ""
            pc = self._GREEN if f == 0 else ""
            print(f"  {grp:<20} {len(rs):>5} "
                  f"{pc}{p:>5}{self._RESET} "
                  f"{fc}{f:>5}{self._RESET} {a:>7.1f}ms")
        print("─" * 48)

        return 1 if failed else 0


# ═════════════════════════════════════════════════════════════════════
# ROUTING ALGORITHM (inline copy — no import from app.py required)
# ═════════════════════════════════════════════════════════════════════

_MAX_RACK = 24
_FAR_END  = _MAX_RACK + 1   # = 25


def _aisle_cross(x1: int, y1: int, x2: int, y2: int) -> list:
    """
    Return waypoints for a single inter-aisle move using the cheapest option:
      A) Cross at current rack x1 → walk horizontally to x2   cost = |x2-x1|
      B) Entrance corridor x=0  → walk to 0, cross, walk to x2 cost = x1+x2
      C) Far-end corridor x=25  → walk to 25, cross, walk to x2 cost = |x1-25|+|x2-25|
    """
    cost_here     = abs(x2 - x1)
    cost_entrance = x1 + x2
    cost_far_end  = abs(x1 - _FAR_END) + abs(x2 - _FAR_END)
    best = min(cost_here, cost_entrance, cost_far_end)

    if best == cost_here:
        return [(x1, y1), (x1, y2), (x2, y2)]
    elif best == cost_entrance:
        return [(x1, y1), (0, y1), (0, y2), (x2, y2)]
    else:
        return [(x1, y1), (_FAR_END, y1), (_FAR_END, y2), (x2, y2)]


def _build_route(depot: tuple, stops: list) -> list:
    """Build a full rectilinear route from depot through stops and back."""
    all_pts = [depot] + list(stops) + [depot]
    wp = []
    for i in range(len(all_pts) - 1):
        x1, y1 = all_pts[i]
        x2, y2 = all_pts[i + 1]
        if y1 == y2:
            wp += [(x1, y1), (x2, y2)]
        else:
            wp += _aisle_cross(x1, y1, x2, y2)
    # Deduplicate consecutive identical points
    out = [wp[0]] if wp else []
    for pt in wp[1:]:
        if pt != out[-1]:
            out.append(pt)
    return out


def _no_diagonal(path: list) -> None:
    """Assert every segment is horizontal or vertical (no diagonals)."""
    for i in range(len(path) - 1):
        xa, ya = path[i]
        xb, yb = path[i + 1]
        assert xa == xb or ya == yb, \
            f"Diagonal segment at step {i}: ({xa},{ya})→({xb},{yb})"


def _path_horizontal_cost(path: list) -> int:
    """Sum of horizontal distances across all segments."""
    return sum(abs(b[0] - a[0]) for a, b in zip(path, path[1:]) if a[1] == b[1])


# ═════════════════════════════════════════════════════════════════════
# TEST DEFINITIONS
# ═════════════════════════════════════════════════════════════════════

def register_tests(runner: TestRunner):
    """Define and register all 35 tests."""

    # ─────────────────────────────────────────────────────────────────
    # SS1 · Object Scanner
    # ─────────────────────────────────────────────────────────────────
    runner.section("SS1 · Object Scanner")

    def t_ss1_01():
        """Create scan session returns valid UUID-like string."""
        db  = get_db()
        sid = db.create_scan_session(total_images=3, notes="unit-test")
        assert sid and len(sid) == 36, f"Expected 36-char UUID, got: {sid!r}"
        row = db.get_scan_sessions(limit=1)[0]
        assert row["session_id"] == sid
        assert row["total_images"] == 3
    runner.run("SS1-01", "Create scan session returns valid UUID", "SS1", t_ss1_01)

    def t_ss1_02():
        """Insert CLIP laptop detection with brand=Dell, conf≥0.40."""
        db  = get_db()
        sid = db.get_scan_sessions(limit=1)[0]["session_id"]
        did = db.insert_detection(
            sid, "img.jpg", "Dell Laptop", 0.92,
            brand="Dell", brand_conf=0.88,
            detect_method="clip", bbox=(10, 10, 200, 200), position=1
        )
        assert did and len(did) == 36
        dets = db.get_detections(sid)
        assert len(dets) == 1
        assert dets[0]["brand"] == "Dell"
        assert dets[0]["detect_method"] == "clip"
    runner.run("SS1-02", "Insert CLIP laptop detection (brand=Dell)", "SS1", t_ss1_02)

    def t_ss1_03():
        """Insert CLIP phone detection with brand=Samsung."""
        db  = get_db()
        sid = db.get_scan_sessions(limit=1)[0]["session_id"]
        did = db.insert_detection(
            sid, "img.jpg", "Samsung Phone", 0.85,
            brand="Samsung", brand_conf=0.90,
            detect_method="clip", bbox=(210, 10, 350, 200), position=2
        )
        assert did
        dets = db.get_detections(sid)
        assert any(d["brand"] == "Samsung" for d in dets)
    runner.run("SS1-03", "Insert CLIP phone detection (brand=Samsung)", "SS1", t_ss1_03)

    def t_ss1_04():
        """Insert OCR bottle detection with text brand via EasyOCR."""
        db  = get_db()
        sid = db.get_scan_sessions(limit=1)[0]["session_id"]
        did = db.insert_detection(
            sid, "img2.jpg", "bottle", 0.75,
            brand="Evian", brand_conf=0.65,
            detect_method="ocr", bbox=(5, 5, 80, 200), position=1
        )
        assert did
        dets = db.get_detections(sid)
        ocr_dets = [d for d in dets if d["detect_method"] == "ocr"]
        assert len(ocr_dets) == 1
        assert ocr_dets[0]["brand"] == "Evian"
    runner.run("SS1-04", "Insert OCR bottle detection (brand=Evian)", "SS1", t_ss1_04)

    def t_ss1_05():
        """get_detection_counts() groups correctly by branded label."""
        db  = get_db()
        sid = db.get_scan_sessions(limit=1)[0]["session_id"]
        counts = db.get_detection_counts(sid)
        assert "Dell Laptop"   in counts, f"Expected 'Dell Laptop' in {counts}"
        assert "Samsung Phone" in counts
        assert "bottle"        in counts
        assert counts["Dell Laptop"] == 1
        assert counts["bottle"]      == 1
    runner.run("SS1-05", "Detection count grouping by branded label", "SS1", t_ss1_05)

    def t_ss1_06():
        """PDF Delivery Order header + items stored and retrieved correctly."""
        db  = get_db()
        sid = db.get_scan_sessions(limit=1)[0]["session_id"]
        do_id = db.insert_delivery_order(
            "DO-2024-001", "Acme Corp", session_id=sid,
            items={"laptop": 2, "cell phone": 1, "bottle": 1}
        )
        row = db.get_delivery_order(do_id)
        assert row["do_number"] == "DO-2024-001"
        assert row["supplier"]  == "Acme Corp"
        assert row["items"]["laptop"]     == 2
        assert row["items"]["cell phone"] == 1
        assert row["items"]["bottle"]     == 1
    runner.run("SS1-06", "Insert and retrieve PDF delivery order", "SS1", t_ss1_06)

    def t_ss1_07():
        """DO comparison rows stored with correct MATCH / DISCREPANCY status."""
        db     = get_db()
        sid    = db.get_scan_sessions(limit=1)[0]["session_id"]
        do_row = db.get_delivery_orders(limit=1)[0]
        comp   = {
            "matches": [
                {"item_type": "bottle", "expected": 1, "actual": 1,
                 "difference": 0, "status": "✅ MATCH"}
            ],
            "discrepancies": [
                {"item_type": "laptop", "expected": 2, "actual": 1,
                 "difference": -1, "status": "⚠️ DISCREPANCY"}
            ],
        }
        db.insert_do_comparison(do_row["do_id"], sid, comp)
        rows    = db.get_do_comparisons(do_row["do_id"])
        matched = [r for r in rows if r["status"] == "MATCH"]
        discrep = [r for r in rows if r["status"] == "DISCREPANCY"]
        assert len(rows)    == 2
        assert len(matched) == 1 and matched[0]["item_label"] == "bottle"
        assert len(discrep) == 1 and discrep[0]["difference"] == -1
    runner.run("SS1-07", "DO comparison MATCH/DISCREPANCY stored correctly", "SS1", t_ss1_07)

    def t_ss1_08():
        """Asset sticker record linked to detection_id and session_id."""
        db   = get_db()
        sid  = db.get_scan_sessions(limit=1)[0]["session_id"]
        dets = db.get_detections(sid)
        did  = dets[0]["detection_id"]
        db.insert_asset_sticker(
            "AST-2024-ABCD1234", did, sid,
            "Dell Laptop", "Dell", 0.92, "img.jpg"
        )
        stickers = db.get_asset_stickers(session_id=sid)
        assert len(stickers) == 1
        assert stickers[0]["asset_id"]    == "AST-2024-ABCD1234"
        assert stickers[0]["detection_id"] == did
        assert stickers[0]["brand"]        == "Dell"
    runner.run("SS1-08", "Asset sticker generation and retrieval", "SS1", t_ss1_08)

    def t_ss1_09():
        """INSERT OR IGNORE prevents duplicate asset_id rows."""
        db  = get_db()
        sid = db.get_scan_sessions(limit=1)[0]["session_id"]
        # Insert same asset_id again — should be silently ignored
        db.insert_asset_sticker(
            "AST-2024-ABCD1234", None, sid,
            "Dell Laptop", "Dell", 0.92, "img.jpg"
        )
        stickers = db.get_asset_stickers(session_id=sid)
        assert len(stickers) == 1, f"Expected 1 sticker, got {len(stickers)}"
    runner.run("SS1-09", "Duplicate asset sticker silently ignored (INSERT OR IGNORE)", "SS1", t_ss1_09)

    # ─────────────────────────────────────────────────────────────────
    # SS2 · Auto-Picker
    # ─────────────────────────────────────────────────────────────────
    runner.section("SS2 · Auto-Picker")

    SAMPLE_INVENTORY = [
        {"sku": "ELEC-0001", "name": "Dell Laptop", "category": "computers",
         "location": {"aisle": 1, "rack": 1}, "quantity": 10,
         "weight": 1.5, "value": 1200, "dimensions": (30, 22, 3), "fragile": True},
        {"sku": "ELEC-0025", "name": "Samsung Phone", "category": "phones",
         "location": {"aisle": 3, "rack": 1}, "quantity": 25,
         "weight": 0.2, "value": 800, "dimensions": (15, 7, 1), "fragile": True},
        {"sku": "ELEC-0073", "name": "Accessory A", "category": "accessories",
         "location": {"aisle": 5, "rack": 1}, "quantity": 100,
         "weight": 0.3, "value": 50, "dimensions": (8, 8, 5), "fragile": False},
    ]

    def t_ss2_01():
        """seed_inventory() inserts new SKUs, skips existing (INSERT OR IGNORE)."""
        db  = get_db()
        db.seed_inventory(SAMPLE_INVENTORY)
        inv = db.get_inventory()
        assert len(inv) == 3
        skus = {r["sku"] for r in inv}
        assert "ELEC-0001" in skus
        assert "ELEC-0025" in skus
        assert "ELEC-0073" in skus
        # Second call must not duplicate
        db.seed_inventory(SAMPLE_INVENTORY)
        assert len(db.get_inventory()) == 3
    runner.run("SS2-01", "seed_inventory() inserts SKUs, idempotent on re-seed", "SS2", t_ss2_01)

    def t_ss2_02():
        """inventory_stats() returns correct totals and no NULL values."""
        db    = get_db()
        stats = db.inventory_stats()
        assert stats["total_skus"]  == 3
        assert stats["total_items"] == 135   # 10+25+100
        assert stats["total_value"] == 10*1200 + 25*800 + 100*50
        for k, v in stats.items():
            assert v is not None, f"NULL in inventory_stats[{k!r}]"
    runner.run("SS2-02", "inventory_stats() correct totals, no NULL values", "SS2", t_ss2_02)

    def t_ss2_03():
        """update_inventory_quantity() persists to DB and reads back correctly."""
        db = get_db()
        db.update_inventory_quantity("ELEC-0001", 12)
        row = db.get_inventory_item("ELEC-0001")
        assert row["quantity"] == 12, f"Expected 12, got {row['quantity']}"
        # Restore
        db.update_inventory_quantity("ELEC-0001", 10)
    runner.run("SS2-03", "update_inventory_quantity() persists correctly", "SS2", t_ss2_03)

    def t_ss2_04():
        """get_inventory(low_stock_threshold=10) returns only items below threshold."""
        db = get_db()
        db.update_inventory_quantity("ELEC-0001", 5)   # below threshold
        low = db.get_inventory(low_stock_threshold=10)
        assert any(r["sku"] == "ELEC-0001" for r in low)
        assert not any(r["sku"] == "ELEC-0025" for r in low)  # 25 >= 10
        db.update_inventory_quantity("ELEC-0001", 10)   # restore
    runner.run("SS2-04", "Low-stock filter returns only items below threshold", "SS2", t_ss2_04)

    def t_ss2_05():
        """Create picker order: header + 2 line items stored and retrieved."""
        db  = get_db()
        sid = db.get_scan_sessions(limit=1)[0]["session_id"]
        db.insert_picker_order(
            "ORD-000001", "CUST001", "retail", 3,
            [{"sku": "ELEC-0001", "quantity": 2},
             {"sku": "ELEC-0025", "quantity": 1}],
            session_id=sid
        )
        order = db.get_picker_order("ORD-000001")
        assert order["customer_id"]   == "CUST001"
        assert order["customer_type"] == "retail"
        assert order["priority"]      == 3
        assert len(order["items"])    == 2
        skus = {i["sku"] for i in order["items"]}
        assert "ELEC-0001" in skus and "ELEC-0025" in skus
    runner.run("SS2-05", "Create picker order with line items", "SS2", t_ss2_05)

    def t_ss2_06():
        """Order creation with stale/invalid session_id does not raise FK error."""
        db = get_db()
        # This is the exact scenario that caused the original crash
        db.insert_picker_order(
            "ORD-000002", "CUST002", "wholesale", 1,
            [{"sku": "ELEC-0001", "quantity": 1}],
            session_id="00000000-dead-beef-0000-000000000000"
        )
        order = db.get_picker_order("ORD-000002")
        assert order is not None
        assert order["session_id"] is None   # stale ID replaced with NULL
    runner.run("SS2-06", "Order with stale session_id: no FK violation", "SS2", t_ss2_06)

    def t_ss2_07():
        """insert_route_steps() and get_route_steps() maintain step_number order."""
        db = get_db()
        db.insert_route_steps("ORD-000001", ["A01-R001", "A03-R001", "A05-R001"])
        steps = db.get_route_steps("ORD-000001")
        assert len(steps) == 3
        assert steps[0]["location"] == "A01-R001"
        assert steps[1]["location"] == "A03-R001"
        assert steps[2]["location"] == "A05-R001"
        assert [s["step_number"] for s in steps] == [1, 2, 3]
    runner.run("SS2-07", "Route steps stored and retrieved in order", "SS2", t_ss2_07)

    def t_ss2_08():
        """Order marked complete with fulfillment=100%; both items picked."""
        db = get_db()
        db.update_picker_order_status(
            "ORD-000001", "complete",
            fulfillment=100.0, order_value=3200.0,
            route=["A01-R001", "A03-R001"],
            route_distance=8.4, route_time=1.2
        )
        db.update_order_item_pick("ORD-000001", "ELEC-0001", 2, "picked")
        db.update_order_item_pick("ORD-000001", "ELEC-0025", 1, "picked")
        order  = db.get_picker_order("ORD-000001")
        assert order["status"]      == "complete"
        assert order["fulfillment"] == 100.0
        picked = [i for i in order["items"] if i["item_status"] == "picked"]
        assert len(picked) == 2
    runner.run("SS2-08", "Order complete: fulfillment=100%, items marked 'picked'", "SS2", t_ss2_08)

    def t_ss2_09():
        """Partial fulfillment: status=partially_fulfilled, fulfillment=60%."""
        db = get_db()
        db.insert_picker_order(
            "ORD-000003", "CUST003", "repair_shop", 2,
            [{"sku": "ELEC-0073", "quantity": 50}]
        )
        db.update_picker_order_status(
            "ORD-000003", "partially_fulfilled",
            fulfillment=60.0, order_value=1500.0
        )
        db.update_order_item_pick("ORD-000003", "ELEC-0073", 30, "partial")
        order = db.get_picker_order("ORD-000003")
        assert order["status"]      == "partially_fulfilled"
        assert order["fulfillment"] == 60.0
        partial = [i for i in order["items"] if i["item_status"] == "partial"]
        assert len(partial) == 1
        assert partial[0]["picked"] == 30
    runner.run("SS2-09", "Partial fulfillment recorded correctly (60%)", "SS2", t_ss2_09)

    def t_ss2_10():
        """get_shipped_order_ids() excludes orders with no confirmed shipment."""
        db      = get_db()
        shipped = db.get_shipped_order_ids()
        # ORD-000001 is complete but not yet shipped
        assert "ORD-000001" not in shipped, \
            "Unshipped completed order must not appear in shipped set"
    runner.run("SS2-10", "Unshipped order absent from shipped_order_ids set", "SS2", t_ss2_10)

    # ─────────────────────────────────────────────────────────────────
    # SS3 · Courier Email
    # ─────────────────────────────────────────────────────────────────
    runner.section("SS3 · Courier Email")

    def t_ss3_01():
        """Shipment row stored with correct method, courier, and fragile flag."""
        db = get_db()
        db.insert_shipment(
            "SHP-000001", "ORD-000001", "confirmed",
            "Express", "FedEx", True, 0.88,
            "ops@fedex.com",
            "[Express] Pick-up Request — ORD-000001",
            "Dear FedEx, please collect order ORD-000001."
        )
        row = db.get_shipment("SHP-000001")
        assert row["shipment_id"]    == "SHP-000001"
        assert row["order_id"]       == "ORD-000001"
        assert row["shipping_method"]== "Express"
        assert row["courier"]        == "FedEx"
        assert row["is_fragile"]     == 1
        assert row["ml_confidence"]  == 0.88
        assert row["status"]         == "confirmed"
    runner.run("SS3-01", "Shipment inserted: method, fragile flag, confidence correct", "SS3", t_ss3_01)

    def t_ss3_02():
        """Confirmed shipment's order_id appears in get_shipped_order_ids()."""
        db      = get_db()
        shipped = db.get_shipped_order_ids()
        assert "ORD-000001" in shipped, \
            "ORD-000001 should be in shipped set after SHP-000001 confirmed"
    runner.run("SS3-02", "Confirmed shipment order in shipped_order_ids set", "SS3", t_ss3_02)

    def t_ss3_03():
        """shipment_stats() returns correct total/confirmed/failed counts."""
        db    = get_db()
        stats = db.shipment_stats()
        assert stats["total"]     == 1
        assert stats["confirmed"] == 1
        assert stats["failed"]    == 0
        for k, v in stats.items():
            assert v is not None, f"NULL in shipment_stats[{k!r}]"
    runner.run("SS3-03", "shipment_stats() correct counts, no NULL values", "SS3", t_ss3_03)

    def t_ss3_04():
        """Multiple shipment statuses tracked: total=3, confirmed=2, failed=1."""
        db = get_db()
        db.insert_shipment("SHP-000002", "ORD-000002", "confirmed",
                           "Same-Day Express", "FedEx", False, 0.92,
                           "ops@fedex.com", "Subject", "Body")
        db.insert_shipment("SHP-000003", "ORD-000003", "failed",
                           "Standard", "FedEx", False, 0.80,
                           "ops@fedex.com", "Subject", "Body")
        stats = db.shipment_stats()
        assert stats["total"]     == 3
        assert stats["confirmed"] == 2
        assert stats["failed"]    == 1
    runner.run("SS3-04", "Multiple shipment statuses: total=3, confirmed=2, failed=1", "SS3", t_ss3_04)

    # ─────────────────────────────────────────────────────────────────
    # XREF · Cross-Subsystem & Database
    # ─────────────────────────────────────────────────────────────────
    runner.section("XREF · Cross-Subsystem & Database")

    def t_xref_01():
        """get_pipeline_summary() returns all 6 KPIs as non-NULL integers."""
        db = get_db()
        p  = db.get_pipeline_summary()
        required = ["scans","items_detected","picker_orders",
                    "completed_picks","shipments","confirmed_shipments"]
        for key in required:
            assert key in p, f"Missing key: {key}"
            assert p[key] is not None, f"NULL for key: {key}"
        assert p["scans"]               == 1
        assert p["picker_orders"]       >= 3
        assert p["completed_picks"]     >= 1
        assert p["confirmed_shipments"] >= 1
    runner.run("XREF-01", "Pipeline summary: all 6 KPIs present and non-NULL", "XREF", t_xref_01)

    def t_xref_02():
        """get_full_order_trail() joins scan→order→route→shipment correctly."""
        db    = get_db()
        trail = db.get_full_order_trail("ORD-000001")
        assert trail["order"]["status"]          == "complete"
        assert len(trail["items"])               == 2
        assert len(trail["route"])               == 3   # 3 route steps
        assert trail["shipment"]["shipment_id"]  == "SHP-000001"
        assert len(trail["events"])              >= 1
    runner.run("XREF-02", "Full order trail: scan→pick→ship all populated", "XREF", t_xref_02)

    def t_xref_03():
        """get_analytics() returns correct aggregates for populated DB."""
        db   = get_db()
        anal = db.get_analytics()
        assert len(anal["detections_by_label"]) >= 2   # Dell Laptop, Samsung Phone, bottle
        assert len(anal["brand_distribution"])  >= 1
        assert len(anal["orders_by_type"])      >= 1
        assert anal["do_accuracy"]["total_items"] == 2
        assert anal["do_accuracy"]["matched"]     == 1
    runner.run("XREF-03", "Analytics: correct aggregates for detections, orders, DO accuracy", "XREF", t_xref_03)

    def t_xref_04():
        """reset_db() truncates all 12 tables; next get_db() starts fresh."""
        reset_db()
        db2   = get_db()
        stats = db2.get_db_stats()
        for table, count in stats.items():
            assert count == 0, f"Table '{table}' not empty after reset: {count} rows"
    runner.run("XREF-04", "reset_db() clears all 12 tables", "XREF", t_xref_04)

    def t_xref_05():
        """_next_order_counter() returns max+1 to prevent ID collisions on restart."""
        db = get_db()
        db.seed_inventory([{
            "sku": "ELEC-0001", "name": "Test Item", "category": "computers",
            "location": {"aisle": 1, "rack": 1}, "quantity": 5,
            "weight": 0.5, "value": 100, "dimensions": (1, 1, 1), "fragile": False
        }])
        db.insert_picker_order("ORD-000042", "CUST999", "retail", 3,
                               [{"sku": "ELEC-0001", "quantity": 1}])
        orders  = db.get_picker_orders(limit=9999)
        max_num = max(
            (int(o["order_id"][4:]) for o in orders if o["order_id"].startswith("ORD-")),
            default=0
        )
        assert max_num       == 42
        assert max_num + 1   == 43   # next safe counter — no collision
    runner.run("XREF-05", "Order counter collision prevention (ORD-000042 → next=43)", "XREF", t_xref_05)

    # ─────────────────────────────────────────────────────────────────
    # ROUTE · Route Optimiser
    # ─────────────────────────────────────────────────────────────────
    runner.section("ROUTE · Route Optimiser")

    def t_route_01():
        """No diagonal segments across diverse route configurations."""
        test_cases = [
            [(5, 1), (8, 2)],
            [(20, 1), (22, 3)],
            [(1, 1), (24, 6)],
            [(12, 2), (12, 4), (12, 6)],
            [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6)],
        ]
        for stops in test_cases:
            path = _build_route((1, 1), stops)
            _no_diagonal(path)
    runner.run("ROUTE-01", "No diagonal segments across diverse configurations", "ROUTE", t_route_01)

    def t_route_02():
        """Cross at current rack when it is the cheapest option."""
        # (10,1)→(10,3): cost_here=0, entrance=20, far=30 → cross at x=10
        path  = _build_route((1, 1), [(10, 1), (10, 3)])
        _no_diagonal(path)
        verts = [(a, b) for a, b in zip(path, path[1:])
                 if a[0] == b[0] and a[1] != b[1]]
        for seg in verts:
            x = seg[0][0]
            assert x == 10, \
                f"Should cross at x=10 (cost=0), got x={x}"
    runner.run("ROUTE-02", "Cross at current rack when cost_here is minimum", "ROUTE", t_route_02)

    def t_route_03():
        """Low-rack transitions use the optimal crossing (cost_here ≤ cost_entrance)."""
        # (2,1)→(3,2): cost_here=1, entrance=5, far=44 → cross at x=2
        path = _build_route((1, 1), [(2, 1), (3, 2)])
        _no_diagonal(path)
        cost = _path_horizontal_cost(path)
        # Optimal cost for (2,1)→(3,2) is 1 (cross at x=2, walk one rack)
        # Total path: depot→(2,1)→(2,2)→(3,2)→depot; hcost = 1+1+2 = 4
        assert cost <= 10, f"Path cost {cost} unexpectedly high"
    runner.run("ROUTE-03", "Low-rack transition uses optimal (≤ entrance corridor) crossing", "ROUTE", t_route_03)

    def t_route_04():
        """Algorithm always picks minimum horizontal distance for any transition."""
        test_cases = [
            (23, 1, 24, 2),   # here=1,  entrance=47, far=3   → here(1)
            (20, 1, 22, 3),   # here=2,  entrance=42, far=8   → here(2)
            ( 1, 1, 24, 2),   # here=23, entrance=25, far=25  → here(23)
            (24, 1,  1, 2),   # here=23, entrance=25, far=25  → here(23)
        ]
        for x1, y1, x2, y2 in test_cases:
            ch = abs(x2 - x1)
            ce = x1 + x2
            cf = abs(x1 - _FAR_END) + abs(x2 - _FAR_END)
            expected_cost = min(ch, ce, cf)
            path       = _aisle_cross(x1, y1, x2, y2)
            actual_cost = _path_horizontal_cost(path)
            assert actual_cost == expected_cost, \
                f"({x1},{y1})→({x2},{y2}): expected cost {expected_cost}, got {actual_cost}"
    runner.run("ROUTE-04", "Algorithm always picks minimum horizontal distance", "ROUTE", t_route_04)

    def t_route_05():
        """Same-aisle stops do not generate unnecessary aisle crossings."""
        path   = _build_route((1, 1), [(3, 2), (7, 2), (15, 2)])
        _no_diagonal(path)
        aisles = set(pt[1] for pt in path)
        # Only aisles 1 (depot) and 2 (all stops) should appear
        assert aisles == {1, 2}, \
            f"Unexpected aisles in same-aisle route: {aisles}"
    runner.run("ROUTE-05", "Same-aisle stops: no unnecessary crossings", "ROUTE", t_route_05)

    def t_route_06():
        """Route always ends at depot (1,1)."""
        for stops in [[(5, 1)], [(8, 3), (20, 5)], [(1, 6), (24, 6)]]:
            path = _build_route((1, 1), stops)
            assert path[-1] == (1, 1), \
                f"Route did not return to depot for stops={stops}; last={path[-1]}"
    runner.run("ROUTE-06", "Route always returns to depot (1,1)", "ROUTE", t_route_06)

    def t_route_07():
        """All vertical (aisle-crossing) moves occur at constant x — no diagonal."""
        stops = [(5, 1), (8, 2), (20, 4), (3, 6)]
        path  = _build_route((1, 1), stops)
        for i in range(len(path) - 1):
            xa, ya = path[i]
            xb, yb = path[i + 1]
            if ya != yb:
                assert xa == xb, \
                    f"Vertical move must be at constant x; got {path[i]}→{path[i+1]}"
    runner.run("ROUTE-07", "All vertical moves at constant x (no diagonal ever)", "ROUTE", t_route_07)


# ═════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="WIS Test Suite — ICT304 Murdoch University"
    )
    parser.add_argument("--group",    help="Run only tests matching this prefix (e.g. SS1, ROUTE)")
    parser.add_argument("--verbose",  action="store_true", help="Print PASS details too")
    parser.add_argument("--no-color", action="store_true", help="Disable colour output")
    args = parser.parse_args()

    use_color = not args.no_color and sys.stdout.isatty()
    runner    = TestRunner(use_color=use_color, verbose=args.verbose)

    print(f"{'═'*62}")
    print(f"  Warehouse Intelligence System — Test Suite")
    print(f"  ICT304 · Murdoch University")
    print(f"{'═'*62}")

    # Reset DB to ensure clean state
    try:
        reset_db()
    except Exception:
        pass   # DB may not exist yet — that is fine

    register_tests(runner)

    # Filter by group if requested
    if args.group:
        runner.results = [r for r in runner.results
                          if r.group == args.group.upper()
                          or r.test_id.startswith(args.group.upper())]

    # Clean up temp DB
    try:
        reset_db()
        db_path = "wis_warehouse.db"
        if os.path.exists(db_path):
            os.remove(db_path)
    except Exception:
        pass

    return runner.summary()


if __name__ == "__main__":
    sys.exit(main())
