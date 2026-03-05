#!/usr/bin/env python3
"""
STREAMLIT DASHBOARD
Subsystem 3 — Intelligent Auto Courier Email System
ICT304 — Warehouse Intelligence System

Tabs:
  📦 Barcode Scanner — simulate scan & run full pipeline
  📋 Activity Log — all processed shipments
  📧 Email Preview — view generated emails
  📊 Model Analytics — Random Forest performance
"""

import streamlit as st
import pandas as pd
from datetime import datetime

# ── Local imports ──
from order_database import OrderDatabase, WAREHOUSE_INFO
from feature_engineering import FeatureEngineer
from ai_decision_engine import AIDecisionEngine, SHIPPING_METHODS, COURIER
from email_generator import EmailGenerator, EmailSender, EMAIL_CONFIG
from courier_system import CourierSystem


# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Auto Courier Email System",
    page_icon="📧",
    layout="wide"
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1a1a2e;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.2rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    .metric-card h3 { font-size: 2rem; margin: 0; font-weight: 700; }
    .metric-card p { font-size: 0.85rem; margin: 0; opacity: 0.9; }
    .success-card {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
    }
    .fail-card {
        background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);
    }
    .rate-card {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
    }
    .email-preview {
        background: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 1.5rem;
        font-family: 'Courier New', monospace;
        font-size: 0.82rem;
        white-space: pre-wrap;
        max-height: 500px;
        overflow-y: auto;
    }
    .pipeline-step {
        padding: 0.6rem 1rem;
        border-left: 3px solid #667eea;
        margin-bottom: 0.5rem;
        background: #f8f9ff;
        border-radius: 0 6px 6px 0;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
if "system" not in st.session_state:
    st.session_state.system = CourierSystem(
        num_orders=10, training_samples=600
    )
if "results" not in st.session_state:
    st.session_state.results = []
if "last_email" not in st.session_state:
    st.session_state.last_email = None
if "pending_shipment" not in st.session_state:
    st.session_state.pending_shipment = None

system: CourierSystem = st.session_state.system


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.title("📧 Auto Courier Email")
    st.caption("Subsystem 3 — ICT304 AI System")

    st.divider()

    # Model status
    analytics = system.ai_engine.get_analytics()
    st.subheader("🤖 AI Model")
    st.metric("Random Forest Accuracy", f"{analytics['accuracy']:.1%}")
    st.caption(f"Training samples: {analytics['training_samples']}")

    st.divider()

    # Warehouse Info
    st.subheader("🏭 Warehouse")
    st.caption(f"📍 {WAREHOUSE_INFO['address']}")
    st.caption(f"📞 {WAREHOUSE_INFO['contact']}")
    st.caption(f"🕐 {WAREHOUSE_INFO['operating_hours']}")
    st.caption(f"🚪 {WAREHOUSE_INFO['dock_number']}")

    st.divider()

    # System controls
    st.subheader("⚡ Controls")
    if st.button("🔄 Reset System", use_container_width=True):
        st.session_state.system = CourierSystem(
            num_orders=10, training_samples=600
        )
        st.session_state.results = []
        st.session_state.last_email = None
        st.rerun()

    # Simulation mode toggle
    sim = st.toggle("📡 Simulation Mode", value=True)
    EMAIL_CONFIG["simulation_mode"] = sim


# ─────────────────────────────────────────────
# HEADER + METRICS
# ─────────────────────────────────────────────
st.markdown('<p class="main-header">📧 Intelligent Auto Courier Email System</p>',
            unsafe_allow_html=True)
st.markdown('<p class="sub-header">Subsystem 3 — Autonomous Barcode-to-Email Pipeline</p>',
            unsafe_allow_html=True)

dash = system.get_dashboard_data()
s = dash["summary"]

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f"""
    <div class="metric-card">
        <h3>{s['total']}</h3>
        <p>Total Processed</p>
    </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown(f"""
    <div class="metric-card success-card">
        <h3>{s['confirmed']}</h3>
        <p>Confirmed</p>
    </div>
    """, unsafe_allow_html=True)
with col3:
    st.markdown(f"""
    <div class="metric-card fail-card">
        <h3>{s['failed']}</h3>
        <p>Failed</p>
    </div>
    """, unsafe_allow_html=True)
with col4:
    st.markdown(f"""
    <div class="metric-card rate-card">
        <h3>{s['success_rate']}%</h3>
        <p>Success Rate</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("")

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📦 Barcode Scanner",
    "📋 Activity Log",
    "📧 Email Preview",
    "📊 Model Analytics"
])


# ── TAB 1: BARCODE SCANNER ──────────────────
with tab1:
    st.subheader("📦 Simulate Barcode Scan")
    st.caption(
        "Select an Order ID to simulate a barcode scan and trigger "
        "the full autonomous pipeline."
    )

    col_scan, col_info = st.columns([1, 1])

    with col_scan:
        order_ids = system.database.get_all_order_ids()
        selected_id = st.selectbox(
            "Select Order ID (Barcode)",
            options=order_ids,
            index=0,
            help="Simulates scanning a barcode on the package"
        )

        # Show order preview
        preview_order = system.database.scan_barcode(selected_id)
        if preview_order:
            st.markdown("##### 📋 Order Preview")
            preview_cols = st.columns(2)
            with preview_cols[0]:
                st.write(f"**Product:** {preview_order.product_name}")
                st.write(f"**Category:** {preview_order.product_category}")
                st.write(f"**Weight:** {preview_order.weight_kg} kg")
                st.write(f"**Dimensions:** {preview_order.dimensions_str}")
            with preview_cols[1]:
                st.write(f"**Destination:** {preview_order.destination_country}")
                st.write(f"**Zone:** {preview_order.destination_zone}")
                st.write(f"**Priority:** {preview_order.order_priority}")
                st.write(f"**Value:** SGD {preview_order.declared_value:,.2f}")

            if preview_order.is_fragile:
                st.warning("⚠️ This item is FRAGILE (glass/ceramics)")

    with col_info:
        st.markdown("##### 🔄 Pipeline Architecture")
        st.markdown("""
        ```
        [Barcode Scanner]
                ↓
        [Database Retrieval]
                ↓
        [Feature Extraction]
            weight, volume, distance, priority
                ↓
        [Random Forest ML Model]
            → Shipping Method
                ↓
        [Rule-Based: Fragility]
                ↓
        [Fixed-Template Email → FedEx]
                ↓
        [Send Email]
        ```
        """)

    st.markdown("---")

    cols = st.columns(2)
    with cols[0]:
        if st.button("🔍 Preview Shipment", use_container_width=True):
            with st.spinner("Analyzing shipment..."):
                result = system.preview_shipment(selected_id)
                st.session_state.pending_shipment = result

    # Show preview if pending
    if st.session_state.pending_shipment and \
       st.session_state.pending_shipment["order_id"] == selected_id:
        
        result = st.session_state.pending_shipment
        
        st.info("ℹ️ Shipment Preview Generated — Review details below before sending.")

        if result.get("decision"):
            d = result["decision"]
            
            # AI & Rule Decisions
            st.markdown("##### 🤖 AI & Rule-Based Decisions")
            step_cols = st.columns(2)
            with step_cols[0]:
                st.metric(
                    "📦 Shipping Method (ML)",
                    d["shipping_method"],
                    f"{d['shipping_confidence']:.0%} confidence"
                )
            with step_cols[1]:
                frag = "⚠️ FRAGILE" if d["is_fragile"] else "✅ Non-Fragile"
                st.metric("🔍 Fragility (Rule)", frag)

            # Email Draft Preview
            st.markdown("##### 📧 Email Draft Preview")
            st.markdown(f"**To:** `{result['email_to']}`")
            st.markdown(f"**Subject:** {result['email_subject']}")
            st.markdown(
                f'<div class="email-preview">{result["email_body"]}</div>',
                unsafe_allow_html=True
            )

            st.markdown("---")

            # CONFIRM BUTTON
            with cols[1]:
                if st.button("✅ Send Email & Confirm", type="primary", use_container_width=True):
                    with st.spinner("Sending email..."):
                         final_result = system.confirm_shipment(result["request_id"])
                         st.session_state.results.insert(0, final_result)
                         st.session_state.pending_shipment = None
                         st.session_state.last_email = {
                             "subject": final_result["email_subject"],
                             "body": final_result["email_body"],
                             "to": final_result["email_to"],
                             "order_id": final_result["order_id"]
                         }
                         st.rerun()

    # Success/Error Messages for finalized results
    if st.session_state.results:
        latest = st.session_state.results[0]
        if latest["order_id"] == selected_id:
             if latest["status"] == "confirmed":
                 st.success("✅ Email Sent Successfully!")
             elif latest["status"] == "failed":
                 st.error(f"❌ Failed: {latest.get('error')}")


# ── TAB 2: ACTIVITY LOG ─────────────────────
with tab2:
    st.subheader("📋 Processed Shipments")

    if system.shipments:
        log_data = []
        for rid, rec in sorted(
            system.shipments.items(),
            key=lambda x: x[1].created_at, reverse=True
        ):
            d = rec.decision or {}
            log_data.append({
                "Request ID": rid,
                "Order ID": rec.order_id,
                "Status": "✅ " + rec.status if rec.status == "confirmed"
                         else "❌ " + rec.status,
                "Shipping": d.get("shipping_method", "—"),
                "Confidence": f"{d.get('shipping_confidence', 0):.0%}"
                              if d.get("shipping_confidence") else "—",
                "Fragile": "⚠️ Yes" if d.get("is_fragile") else "No",
                "Time": rec.created_at[:19]
            })

        st.dataframe(
            pd.DataFrame(log_data),
            use_container_width=True,
            hide_index=True,
            height=400
        )

        st.markdown("---")
        st.markdown("##### 📊 Summary by Shipping Method")
        method_counts = {}
        for rec in system.shipments.values():
            d = rec.decision or {}
            m = d.get("shipping_method", "Unknown")
            method_counts[m] = method_counts.get(m, 0) + 1

        st.bar_chart(pd.DataFrame(
            {"Count": method_counts},
            index=list(method_counts.keys())
        ))
    else:
        st.info("📋 No shipments processed yet. Use the Barcode Scanner tab.")


# ── TAB 3: EMAIL PREVIEW ────────────────────
with tab3:
    st.subheader("📧 Generated Email Preview")

    if st.session_state.last_email:
        email = st.session_state.last_email
        st.markdown(f"**To:** `{email['to']}`")
        st.markdown(f"**Subject:** {email['subject']}")
        st.markdown(f"**Order:** {email['order_id']}")
        st.markdown("---")
        st.markdown(
            f'<div class="email-preview">{email["body"]}</div>',
            unsafe_allow_html=True
        )
    elif system.shipments:
        # Show the most recent email
        for rid, rec in sorted(
            system.shipments.items(),
            key=lambda x: x[1].created_at, reverse=True
        ):
            if rec.email_body:
                st.markdown(f"**To:** `{rec.email_to}`")
                st.markdown(f"**Subject:** {rec.email_subject}")
                st.markdown(f"**Order:** {rec.order_id}")
                st.markdown("---")
                st.markdown(
                    f'<div class="email-preview">{rec.email_body}</div>',
                    unsafe_allow_html=True
                )
                break
        else:
            st.info("📧 No emails generated yet.")
    else:
        st.info("📧 No emails generated yet. Process a shipment first.")


# ── TAB 4: MODEL ANALYTICS ──────────────────
with tab4:
    st.subheader("📊 Random Forest Model Analytics")

    analytics = system.ai_engine.get_analytics()

    # Model overview
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("🎯 Accuracy", f"{analytics['accuracy']:.1%}")
    with m2:
        st.metric("🌲 Estimators", analytics["n_estimators"])
    with m3:
        st.metric("📏 Max Depth", analytics["max_depth"])

    st.markdown("---")

    # Feature importance
    st.markdown("##### 🔑 Feature Importance")
    fi = analytics["feature_importances"]
    fi_df = pd.DataFrame({
        "Feature": list(fi.keys()),
        "Importance": list(fi.values())
    }).sort_values("Importance", ascending=False)

    st.bar_chart(fi_df.set_index("Feature"))
    st.dataframe(fi_df, use_container_width=True, hide_index=True)

    st.markdown("---")

    # Classification report
    st.markdown("##### 📋 Classification Report")
    st.code(analytics["classification_report"], language="text")

    st.markdown("---")

    # Rules summary
    st.markdown("##### 📐 Rule-Based Logic")

    st.markdown("**Fragility (Rule-Based):**")
    st.code(
        "IF product_category IN ['glass', 'ceramics'] → FRAGILE\n"
        "ELSE → Non-Fragile",
        language="text"
    )

    st.markdown("**Courier Partner:**")
    st.code(
        f"Single delivery partner: {COURIER}\n"
        f"All shipments are sent to FedEx regardless of shipping method.",
        language="text"
    )

    st.markdown("---")
    st.markdown("##### 🏗️ System Architecture")
    st.code("""
[Barcode Scanner]
        ↓
[Database Retrieval]
        ↓
[Feature Extraction]
    weight, volume, distance, priority
        ↓
[Random Forest ML Model]  ← Only AI component
    → Shipping Method (Same-Day / Express / Standard)
        ↓
[Rule-Based: Fragility]
    → category == glass/ceramics
        ↓
[Fixed-Template Email → FedEx]
        ↓
[SMTP / Simulated Send]
    """, language="text")


# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("---")
st.caption(
    "📧 Intelligent Auto Courier Email System — Subsystem 3 | "
    "ICT304 AI System Design — Warehouse Intelligence System | "
    f"Session: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
)
