"""
db_viewer.py
════════════════════════════════════════════════════════════════════
Database Viewer — renders inside the "🗄️ Database" tab of app.py

Call:  render_db_tab()

Sub-tabs:
  📊 Overview      — row counts, pipeline KPIs, health indicators
  🔍 Browse        — table browser with per-table filters
  🔗 Order Trail   — full lifecycle for a single order
  📈 Analytics     — charts drawn from aggregate queries
  📋 Audit Log     — pipeline_events stream
  📤 Export        — download full DB as JSON
════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Dict, List

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from database import get_db, DB_PATH


# ─────────────────────────────────────────────────────────────────────
# HUMAN-READABLE TABLE NAMES
# ─────────────────────────────────────────────────────────────────────

TABLE_META = {
    # name                  label                  subsystem  icon
    "scan_sessions":       ("Scan Sessions",        "SS1",    "📷"),
    "detections":          ("Detections",            "SS1",    "🔍"),
    "delivery_orders":     ("Delivery Orders",       "SS1",    "📄"),
    "delivery_order_items":("DO Line Items",         "SS1",    "📋"),
    "do_comparisons":      ("DO Comparisons",        "SS1",    "⚖️"),
    "asset_stickers":      ("Asset Stickers",        "SS1",    "🏷️"),
    "inventory":           ("Inventory",             "SS2",    "📦"),
    "picker_orders":       ("Picker Orders",         "SS2",    "🛒"),
    "order_items":         ("Order Line Items",      "SS2",    "📝"),
    "route_steps":         ("Route Steps",           "SS2",    "🗺️"),
    "shipments":           ("Shipments / Emails",    "SS3",    "📧"),
    "pipeline_events":     ("Audit Log",             "XREF",   "🔗"),
}

SS_COLOUR = {
    "SS1":  "#667eea",
    "SS2":  "#11998e",
    "SS3":  "#eb3349",
    "XREF": "#f4a261",
}


# ─────────────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────────────

def render_db_tab():
    db = get_db()

    st.markdown("## 🗄️ Database Viewer")
    st.caption(f"SQLite · `{DB_PATH.resolve()}`")

    sub = st.tabs([
        "📊 Overview",
        "🔍 Browse Tables",
        "🔗 Order Trail",
        "📈 Analytics",
        "📋 Audit Log",
        "📤 Export",
    ])

    with sub[0]:  _overview(db)
    with sub[1]:  _browse(db)
    with sub[2]:  _order_trail(db)
    with sub[3]:  _analytics(db)
    with sub[4]:  _audit_log(db)
    with sub[5]:  _export(db)


# ─────────────────────────────────────────────────────────────────────
# SUB-TAB 0 — OVERVIEW
# ─────────────────────────────────────────────────────────────────────

def _overview(db):
    st.subheader("📊 Database Overview")

    stats    = db.get_db_stats()
    pipeline = db.get_pipeline_summary()
    inv      = db.inventory_stats()
    ship     = db.shipment_stats()

    # ── Pipeline KPIs ─────────────────────────────────────────────────
    st.markdown("#### 🔄 Pipeline KPIs")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("📷 Scan Sessions",   pipeline.get("scans", 0))
    c2.metric("🔍 Items Detected",  pipeline.get("items_detected", 0))
    c3.metric("🛒 Picker Orders",   pipeline.get("picker_orders", 0))
    c4.metric("✅ Picks Complete",  pipeline.get("completed_picks", 0))
    c5.metric("📧 Shipments",       pipeline.get("shipments", 0))
    c6.metric("🚚 Confirmed",       pipeline.get("confirmed_shipments", 0))

    st.markdown("---")

    # ── Table row counts ──────────────────────────────────────────────
    st.markdown("#### 🗂️ Table Row Counts")

    col_ss1, col_ss2, col_ss3, col_xref = st.columns(4)

    groups = {"SS1": col_ss1, "SS2": col_ss2, "SS3": col_ss3, "XREF": col_xref}

    for table, (label, ss, icon) in TABLE_META.items():
        count = stats.get(table, 0)
        col   = groups[ss]
        with col:
            colour = SS_COLOUR[ss]
            st.markdown(
                f"""<div style="background:{colour}18;border-left:4px solid {colour};
                    padding:.5rem .8rem;border-radius:0 6px 6px 0;margin-bottom:.4rem">
                    <span style="font-size:.8rem;color:{colour};font-weight:600">
                    {icon} {label}</span><br>
                    <span style="font-size:1.4rem;font-weight:700">{count:,}</span>
                    </div>""",
                unsafe_allow_html=True,
            )

    st.markdown("---")

    # ── Inventory + Shipment snapshots ────────────────────────────────
    c1b, c2b = st.columns(2)

    with c1b:
        st.markdown("#### 📦 Inventory Snapshot")
        if inv:
            i1, i2, i3, i4 = st.columns(4)
            i1.metric("SKUs",       inv.get("total_skus") or 0)
            i2.metric("Total Units",f"{inv.get('total_items') or 0:,}")
            i3.metric("Total Value",f"${inv.get('total_value') or 0:,.0f}")
            i4.metric("Low Stock",  inv.get("low_stock_count") or 0)
        else:
            st.info("No inventory data yet.")

    with c2b:
        st.markdown("#### 📧 Shipment Snapshot")
        if ship and ship.get("total"):
            s1, s2, s3 = st.columns(3)
            total = ship.get("total", 0)
            conf  = ship.get("confirmed", 0)
            fail  = ship.get("failed", 0)
            s1.metric("Total",     total)
            s2.metric("Confirmed", conf)
            s3.metric("Failed",    fail)
            if total:
                rate = conf / total * 100
                st.progress(int(rate), text=f"Success rate {rate:.0f}%")
        else:
            st.info("No shipments yet.")


# ─────────────────────────────────────────────────────────────────────
# SUB-TAB 1 — BROWSE TABLES
# ─────────────────────────────────────────────────────────────────────

def _browse(db):
    st.subheader("🔍 Browse Tables")

    # Build display options
    options = {
        f"{icon} {label} ({ss})": table
        for table, (label, ss, icon) in TABLE_META.items()
    }
    chosen_label = st.selectbox("Select table", list(options.keys()))
    chosen_table = options[chosen_label]

    # Load rows
    limit = st.slider("Rows to show", 10, 500, 50, 10)

    try:
        loader = _get_table_rows(db, chosen_table, limit) or []
        if not loader:
            st.info(f"No rows in `{chosen_table}` yet.")
            return

        df = pd.DataFrame(loader)

        # Column search
        search = st.text_input("🔎 Filter rows (any column)", "")
        if search:
            mask = df.apply(
                lambda col: col.astype(str).str.contains(search, case=False, na=False)
            ).any(axis=1)
            df = df[mask]

        st.markdown(f"**{len(df):,} rows** shown")
        st.dataframe(df, use_container_width=True, hide_index=True)

    except Exception as e:
        st.error(f"Could not load table: {e}")


def _get_table_rows(db, table: str, limit: int) -> List[Dict]:
    """Route to the correct DB method for each table."""
    dispatch = {
        "scan_sessions":        lambda: db.get_scan_sessions(limit),
        "detections":           lambda: db.get_all_detections(limit),
        "delivery_orders":      lambda: db.get_delivery_orders(limit),
        "delivery_order_items": lambda: _doi_rows(db, limit),
        "do_comparisons":       lambda: _do_comp_rows(db, limit),
        "asset_stickers":       lambda: db.get_asset_stickers(limit=limit),
        "inventory":            lambda: db.get_inventory(),
        "picker_orders":        lambda: db.get_picker_orders(limit=limit),
        "order_items":          lambda: _order_item_rows(db, limit),
        "route_steps":          lambda: _route_step_rows(db, limit),
        "shipments":            lambda: db.get_shipments(limit=limit),
        "pipeline_events":      lambda: db.get_pipeline_events(limit),
    }
    fn = dispatch.get(table)
    return fn() if fn else []


def _doi_rows(db, limit):
    from database import _conn, DB_PATH
    with _conn(DB_PATH) as con:
        rows = con.execute(
            "SELECT * FROM delivery_order_items LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]


def _do_comp_rows(db, limit):
    from database import _conn, DB_PATH
    with _conn(DB_PATH) as con:
        rows = con.execute(
            "SELECT * FROM do_comparisons ORDER BY compared_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]


def _order_item_rows(db, limit):
    from database import _conn, DB_PATH
    with _conn(DB_PATH) as con:
        rows = con.execute(
            "SELECT oi.*, i.name FROM order_items oi "
            "LEFT JOIN inventory i ON oi.sku=i.sku LIMIT ?",
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]


def _route_step_rows(db, limit):
    from database import _conn, DB_PATH
    with _conn(DB_PATH) as con:
        rows = con.execute(
            "SELECT * FROM route_steps ORDER BY order_id, step_number LIMIT ?",
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]


# ─────────────────────────────────────────────────────────────────────
# SUB-TAB 2 — ORDER TRAIL
# ─────────────────────────────────────────────────────────────────────

def _order_trail(db):
    st.subheader("🔗 Full Order Lifecycle")
    st.caption("Select an order to see its complete journey: scan → pick → ship.")

    orders = db.get_picker_orders(limit=100)
    if not orders:
        st.info("No picker orders in the database yet.")
        return

    options = {
        f"{o['order_id']} — {o['customer_id']} [{o['status']}]": o["order_id"]
        for o in orders
    }
    sel = st.selectbox("Select Order", list(options.keys()))
    order_id = options[sel]

    trail = db.get_full_order_trail(order_id)
    if not trail:
        st.error("Could not load trail for this order.")
        return

    o = trail["order"]

    # Header metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Status",      o["status"].upper())
    c2.metric("Fulfillment", f"{o.get('fulfillment', 0):.1f}%")
    c3.metric("Value",       f"${o.get('order_value', 0):,.2f}")
    c4.metric("Priority",    o.get("priority", "—"))

    st.markdown("---")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**📝 Order Lines**")
        if trail["items"]:
            raw = trail["items"]
            # Safely build display dataframe — some columns may be missing
            df_items = pd.DataFrame([{
                "SKU":     r.get("sku", ""),
                "Name":    r.get("name") or r.get("sku", ""),
                "Ordered": r.get("quantity", 0),
                "Picked":  r.get("picked", 0),
                "Status":  r.get("item_status", "—"),
            } for r in raw])
            st.dataframe(df_items, hide_index=True, use_container_width=True)
        else:
            st.caption("No items found.")

        if trail["shipment"]:
            st.markdown("**📧 Shipment**")
            s = trail["shipment"]
            st.write(f"ID: `{s['shipment_id']}`")
            st.write(f"Method: **{s['shipping_method']}** via {s['courier']}")
            st.write(f"Fragile: {'⚠️ Yes' if s['is_fragile'] else '✅ No'}")
            st.write(f"Confidence: {s.get('ml_confidence', 0):.0%}")
            with st.expander("📧 View Email"):
                st.code(s["email_body"], language="text")

    with col_b:
        st.markdown("**🗺️ Route Steps**")
        if trail["route"]:
            df_route = pd.DataFrame([{
                "Stop":     r.get("step_number", i+1),
                "Location": r.get("location", ""),
            } for i, r in enumerate(trail["route"])])
            st.dataframe(df_route, hide_index=True, use_container_width=True)
        else:
            st.caption("No route steps stored.")

        st.markdown("**🔗 Pipeline Events**")
        if trail["events"]:
            for ev in trail["events"]:
                icon = {"SS1": "📷", "SS2": "🤖", "SS3": "📧"}.get(ev["subsystem"], "🔗")
                ts   = ev["occurred_at"][:16]
                st.markdown(
                    f"`{ts}` {icon} **{ev['event_type']}** — {ev.get('description', '')}"
                )
        else:
            st.caption("No events logged.")


# ─────────────────────────────────────────────────────────────────────
# SUB-TAB 3 — ANALYTICS
# ─────────────────────────────────────────────────────────────────────

def _analytics(db):
    st.subheader("📈 Analytics")

    data = db.get_analytics()

    # ── Detections by label ───────────────────────────────────────────
    if data.get("detections_by_label"):
        st.markdown("#### 🔍 Detections by Item Type")
        df = pd.DataFrame(data["detections_by_label"])
        df["avg_conf"] = df["avg_conf"].round(3)
        fig = px.bar(df, x="item_label", y="cnt",
                     color="cnt", color_continuous_scale="Blues",
                     labels={"item_label": "Item", "cnt": "Count"},
                     title="Total Detections by Item Type")
        st.plotly_chart(fig, use_container_width=True)

    # ── Brand distribution ────────────────────────────────────────────
    if data.get("brand_distribution"):
        st.markdown("#### 🏷️ Brand Distribution")
        df_b = pd.DataFrame(data["brand_distribution"])
        fig_b = px.pie(df_b, names="brand", values="cnt",
                       title="Top Brands Detected")
        st.plotly_chart(fig_b, use_container_width=True)

    col1, col2 = st.columns(2)

    # ── Orders by customer type ────────────────────────────────────────
    with col1:
        if data.get("orders_by_type"):
            st.markdown("#### 🛒 Orders by Customer Type")
            df_o = pd.DataFrame(data["orders_by_type"])
            fig_o = px.bar(df_o, x="customer_type", y="orders",
                           color="customer_type",
                           labels={"customer_type": "Type", "orders": "Orders"},
                           title="Orders per Customer Type")
            st.plotly_chart(fig_o, use_container_width=True)

    # ── Shipping method breakdown ──────────────────────────────────────
    with col2:
        if data.get("shipping_breakdown"):
            st.markdown("#### 📦 Shipping Method Breakdown")
            df_s = pd.DataFrame(data["shipping_breakdown"])
            fig_s = px.pie(df_s, names="shipping_method", values="cnt",
                           title="Shipments by Method")
            st.plotly_chart(fig_s, use_container_width=True)

    # ── Daily activity ─────────────────────────────────────────────────
    if data.get("daily_activity"):
        st.markdown("#### 📅 Daily Pipeline Activity (last 14 days)")
        df_d = pd.DataFrame(data["daily_activity"])
        if "day" in df_d.columns and "events" in df_d.columns:
         df_d["subsystem"] = df_d.get("subsystem", "Unknown")
        fig_d = px.bar(df_d, x="day", y="events",
                       color="subsystem" if "subsystem" in df_d.columns else None,
                       barmode="group",
                       color_discrete_map=SS_COLOUR,
                       labels={"day": "Date", "events": "Events"},
                       title="Events per Day by Subsystem")
        st.plotly_chart(fig_d, use_container_width=True)

    # ── DO accuracy ────────────────────────────────────────────────────
    if data.get("do_accuracy") and data["do_accuracy"].get("total_items"):
        st.markdown("#### ⚖️ Delivery Order Accuracy")
        acc = data["do_accuracy"]
        total   = acc.get("total_items", 0) or 1
        matched = acc.get("matched", 0)
        rate    = matched / total * 100
        c1, c2, c3 = st.columns(3)
        c1.metric("Items Compared", total)
        c2.metric("Matched",        matched)
        c3.metric("Accuracy",       f"{rate:.1f}%")
        st.progress(int(rate))


# ─────────────────────────────────────────────────────────────────────
# SUB-TAB 4 — AUDIT LOG
# ─────────────────────────────────────────────────────────────────────

def _audit_log(db):
    st.subheader("📋 Pipeline Audit Log")

    events = db.get_pipeline_events(limit=200)
    if not events:
        st.info("No events logged yet.")
        return

    df = pd.DataFrame(events)
    df["occurred_at"] = pd.to_datetime(df["occurred_at"]).dt.strftime("%Y-%m-%d %H:%M:%S")

    # Filter
    ss_filter = st.multiselect(
        "Filter by subsystem",
        options=["SS1", "SS2", "SS3", "XREF"],
        default=[],
    )
    if ss_filter:
        df = df[df["subsystem"].isin(ss_filter)]

    st.dataframe(
        df[["occurred_at", "subsystem", "event_type", "reference_id", "description"]],
        use_container_width=True,
        hide_index=True,
        column_config={
            "subsystem":   st.column_config.TextColumn(width="small"),
            "event_type":  st.column_config.TextColumn(width="medium"),
            "occurred_at": st.column_config.TextColumn("Time", width="medium"),
        },
    )


# ─────────────────────────────────────────────────────────────────────
# SUB-TAB 5 — EXPORT
# ─────────────────────────────────────────────────────────────────────

def _export(db):
    st.subheader("📤 Export Database")

    st.markdown("""
    Download the **entire database** as a JSON file — every table, every row.
    Useful for backups, audit trails, or importing into another tool.
    """)

    col1, col2 = st.columns(2)

    with col1:
        if st.button("📦 Generate Full Export", type="primary",
                     use_container_width=True):
            with st.spinner("Exporting…"):
                dump = db.export_full_db()
                blob = json.dumps(dump, indent=2, default=str).encode()
                ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
                st.session_state["_db_export"] = (blob, ts)

        if "_db_export" in st.session_state:
            blob, ts = st.session_state["_db_export"]
            st.download_button(
                "⬇️ Download JSON",
                data=blob,
                file_name=f"wis_db_export_{ts}.json",
                mime="application/json",
                use_container_width=True,
            )
            rows = sum(
                len(v) for v in json.loads(blob).values()
                if isinstance(v, list)
            )
            st.success(f"✅ Export ready — {rows:,} total rows across all tables.")

    with col2:
        st.markdown("**📌 Maintenance**")
        keep_days = st.number_input("Purge scan sessions older than (days)",
                                    min_value=7, max_value=365, value=30)
        if st.button("🗑️ Purge Old Sessions", use_container_width=True):
            db.purge_old_sessions(keep_days=keep_days)
            st.success(f"Purged sessions older than {keep_days} days.")
            st.rerun()
