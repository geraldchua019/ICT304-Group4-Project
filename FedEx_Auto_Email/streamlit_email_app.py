import streamlit as st
import pandas as pd
from datetime import datetime
import json

# Import the core FedEx Auto Email System
from fedex_auto_email import (
    FedExAutoEmailSystem,
    ShippingPackage,
    PackageStatus,
    EmailStatus,
    EMAIL_CONFIG,
    CARRIER_CONFIG,
    WAREHOUSE_INFO
)

# ─────────────────────────────────────────────
# PAGE CONFIGURATION
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="FedEx Auto Email System",
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
    .metric-card h3 {
        font-size: 2rem;
        margin: 0;
        font-weight: 700;
    }
    .metric-card p {
        font-size: 0.85rem;
        margin: 0;
        opacity: 0.9;
    }
    .success-card {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
    }
    .fail-card {
        background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);
    }
    .pending-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    }
    .rate-card {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
    }
    .step-success {
        color: #27ae60;
        font-weight: 600;
    }
    .step-fail {
        color: #e74c3c;
        font-weight: 600;
    }
    .email-preview {
        background: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 1.5rem;
        font-family: 'Courier New', monospace;
        font-size: 0.85rem;
        white-space: pre-wrap;
        max-height: 500px;
        overflow-y: auto;
    }
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .badge-sent { background: #d4edda; color: #155724; }
    .badge-failed { background: #f8d7da; color: #721c24; }
    .badge-pending { background: #fff3cd; color: #856404; }
    .section-divider {
        border-top: 2px solid #eee;
        margin: 2rem 0;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
if "email_system" not in st.session_state:
    st.session_state.email_system = FedExAutoEmailSystem()

if "processing_results" not in st.session_state:
    st.session_state.processing_results = []

if "last_email_preview" not in st.session_state:
    st.session_state.last_email_preview = None

system = st.session_state.email_system


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/email.png", width=60)
    st.title("📧 FedEx Auto Email")
    st.caption("Subsystem 3 — Warehouse Intelligence")
    
    st.divider()
    
    # System Status
    st.subheader("⚙️ System Configuration")
    
    st.markdown(f"**SMTP Server:** `{EMAIL_CONFIG['smtp_server']}`")
    st.markdown(f"**TLS Encryption:** {'✅ Enabled' if EMAIL_CONFIG['use_tls'] else '❌ Disabled'}")
    st.markdown(f"**Max Retries:** {EMAIL_CONFIG['max_retries']}")
    
    sim_mode = st.toggle("Simulation Mode", value=True,
                         help="When ON, emails are simulated (not actually sent)")
    EMAIL_CONFIG["simulation_mode"] = sim_mode
    
    if sim_mode:
        st.info("📌 Simulation mode — No real emails sent")
    else:
        st.warning("⚠️ Live mode — Emails will be sent via SMTP")
    
    st.divider()
    
    # Warehouse Info
    st.subheader("🏭 Warehouse")
    st.markdown(f"**{WAREHOUSE_INFO['name']}**")
    st.caption(WAREHOUSE_INFO['address'])
    st.caption(f"📞 {WAREHOUSE_INFO['contact']}")
    st.caption(f"🕐 {WAREHOUSE_INFO['operating_hours']}")
    st.caption(f"🚪 {WAREHOUSE_INFO['dock_number']}")
    
    st.divider()
    
    # Quick Actions
    st.subheader("⚡ Quick Actions")
    if st.button("🔄 Reset System", use_container_width=True):
        st.session_state.email_system = FedExAutoEmailSystem()
        st.session_state.processing_results = []
        st.session_state.last_email_preview = None
        st.rerun()
    
    if st.button("💾 Export Report", use_container_width=True):
        if system.pickup_requests:
            report = system.save_shipping_report()
            st.success("Report saved!")
            st.json(report["dashboard"])
        else:
            st.warning("No data to export")


# ─────────────────────────────────────────────
# MAIN CONTENT
# ─────────────────────────────────────────────
st.markdown('<p class="main-header">📧 FedEx Auto Email System</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Subsystem 3 — Automated Carrier Pickup Request System | ICT304 Warehouse Intelligence</p>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# DASHBOARD METRICS
# ─────────────────────────────────────────────
dashboard = system.get_dashboard_data()
summary = dashboard["summary"]

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <h3>{summary['total_requests']}</h3>
        <p>Total Requests</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card success-card">
        <h3>{summary['confirmed']}</h3>
        <p>Confirmed</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card fail-card">
        <h3>{summary['failed']}</h3>
        <p>Failed</p>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="metric-card rate-card">
        <h3>{summary['success_rate']}%</h3>
        <p>Success Rate</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("")

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📦 Shipping Zone Trigger",
    "📋 Activity Log",
    "📧 Email Preview",
    "📊 System Analytics"
])


# ── TAB 1: SHIPPING ZONE TRIGGER ──
with tab1:
    st.subheader("📦 Trigger Pickup Request")
    st.caption("Simulate a package entering the shipping zone to trigger automated pickup request")
    
    with st.form("shipping_form", clear_on_submit=False):
        st.markdown("##### 📋 Order Information")
        col_a, col_b = st.columns(2)
        with col_a:
            order_id = st.text_input("Order ID *", value="ORD-10234",
                                     help="Unique order identifier")
        with col_b:
            contents_desc = st.text_input("Contents Description",
                                          value="Electronics - Laptop x1")
        
        st.markdown("##### 📐 Package Details")
        col_c, col_d, col_e = st.columns(3)
        with col_c:
            weight = st.number_input("Weight (kg) *", min_value=0.1,
                                     max_value=100.0, value=3.2, step=0.1)
        with col_d:
            item_count = st.number_input("Item Count *", min_value=1,
                                         max_value=1000, value=1)
        with col_e:
            declared_value = st.number_input("Declared Value (SGD)",
                                             min_value=0.0, value=1200.00, step=50.0)
        
        col_f, col_g, col_h = st.columns(3)
        with col_f:
            dim_l = st.number_input("Length (cm)", min_value=1.0, value=30.0, step=1.0)
        with col_g:
            dim_w = st.number_input("Width (cm)", min_value=1.0, value=20.0, step=1.0)
        with col_h:
            dim_h = st.number_input("Height (cm)", min_value=1.0, value=15.0, step=1.0)
        
        fragile = st.checkbox("⚠️ Fragile — Handle with care", value=True)
        
        st.markdown("##### 📍 Destination")
        col_i, col_j = st.columns(2)
        with col_i:
            dest_address = st.text_input("Destination Address *",
                                          value="123 Orchard Road, #05-01, Singapore 238858")
        with col_j:
            dest_country = st.text_input("Destination Country *",
                                          value="Singapore")
        
        st.markdown("##### 👤 Contact Details")
        col_k, col_l, col_m = st.columns(3)
        with col_k:
            contact_name = st.text_input("Recipient Name *", value="John Tan")
        with col_l:
            contact_email = st.text_input("Recipient Email *",
                                           value="john.tan@example.com")
        with col_m:
            contact_phone = st.text_input("Recipient Phone",
                                           value="+65 9123 4567")
        
        st.markdown("##### 🚚 Carrier Selection")
        col_n, col_o = st.columns(2)
        with col_n:
            carrier = st.selectbox("Carrier *",
                                   options=list(CARRIER_CONFIG.keys()),
                                   index=0)
        with col_o:
            carrier_services = CARRIER_CONFIG[carrier]["service_types"]
            service_type = st.selectbox("Service Type *",
                                        options=carrier_services,
                                        index=0)
        
        submitted = st.form_submit_button(
            "🚀 Trigger Pickup Request",
            use_container_width=True,
            type="primary"
        )
    
    if submitted:
        # Create package
        package = ShippingPackage(
            order_id=order_id,
            weight_kg=weight,
            dimensions_cm=(dim_l, dim_w, dim_h),
            destination_address=dest_address,
            destination_country=dest_country,
            contact_name=contact_name,
            contact_email=contact_email,
            contact_phone=contact_phone,
            contents_description=contents_desc,
            item_count=item_count,
            declared_value=declared_value,
            fragile=fragile
        )
        
        # Process
        with st.spinner("🔄 Processing pickup request..."):
            result = system.trigger_shipping(package, carrier=carrier,
                                              service_type=service_type)
        
        st.session_state.processing_results.append(result)
        
        # Show result
        st.markdown("---")
        st.subheader("📋 Processing Results")
        
        # Step-by-step results
        steps = result.get("steps", {})
        
        # Validation
        validation = steps.get("validation", {})
        if validation.get("valid"):
            st.markdown(f'<span class="step-success">✅ Step 1: Validation — PASSED</span>',
                        unsafe_allow_html=True)
            if validation.get("warnings"):
                for w in validation["warnings"]:
                    st.warning(f"⚠️ {w}")
        else:
            st.markdown(f'<span class="step-fail">❌ Step 1: Validation — FAILED</span>',
                        unsafe_allow_html=True)
            for e in validation.get("errors", []):
                st.error(f"❌ {e}")
        
        # Request Generation
        if "request_generation" in steps:
            rg = steps["request_generation"]
            st.markdown(f'<span class="step-success">✅ Step 2: Request Generated — {rg["request_id"]}</span>',
                        unsafe_allow_html=True)
            st.caption(f"Pickup Time: {rg['pickup_time']}")
        
        # Email Formatting
        if "email_formatting" in steps:
            ef = steps["email_formatting"]
            st.markdown(f'<span class="step-success">✅ Step 3: Email Formatted — {ef["body_length"]} chars</span>',
                        unsafe_allow_html=True)
            
            # Store email preview
            if result.get("request_id") and result["request_id"] in system.pickup_requests:
                req = system.pickup_requests[result["request_id"]]
                st.session_state.last_email_preview = {
                    "subject": req.email_subject,
                    "body": req.email_body,
                    "carrier": carrier,
                    "request_id": result["request_id"]
                }
        
        # Transmission
        if "transmission" in steps:
            tx = steps["transmission"]
            if tx["success"]:
                st.markdown(f'<span class="step-success">✅ Step 4: Email Sent — {tx["message"]}</span>',
                            unsafe_allow_html=True)
            else:
                st.markdown(f'<span class="step-fail">❌ Step 4: Email Failed — {tx["message"]}</span>',
                            unsafe_allow_html=True)
        
        # Confirmation
        if "confirmation" in steps:
            conf = steps["confirmation"]
            if conf["status"] == "pickup_confirmed":
                st.markdown(f'<span class="step-success">✅ Step 5: Pickup Confirmed</span>',
                            unsafe_allow_html=True)
                st.success(f"🎉 Pickup request confirmed at {conf.get('confirmed_at', 'N/A')}")
            elif conf["status"] == "failed":
                st.markdown(f'<span class="step-fail">❌ Step 5: Confirmation Failed</span>',
                            unsafe_allow_html=True)
                st.error(f"❌ {conf.get('error', 'Unknown error')}")
        
        # Final Status
        final = result.get("final_status", "unknown")
        if final == "pickup_confirmed":
            st.balloons()
            st.success(f"✅ **PICKUP CONFIRMED** — Order {order_id} via {carrier} {service_type}")
        elif final == "validation_failed":
            st.error(f"❌ **VALIDATION FAILED** — Please fix errors above")
        else:
            st.warning(f"⚠️ **Status: {final.upper()}** — Check email logs for details")


# ── TAB 2: ACTIVITY LOG ──
with tab2:
    st.subheader("📋 Pickup Request Activity Log")
    
    if system.pickup_requests:
        # Build table data
        log_data = []
        for req_id, req in system.pickup_requests.items():
            status_emoji = {
                "sent": "✅",
                "failed": "❌",
                "pending": "⏳",
                "retrying": "🔄"
            }.get(req.status.value, "❓")
            
            log_data.append({
                "Status": f"{status_emoji} {req.status.value.upper()}",
                "Request ID": req_id,
                "Order ID": req.package.order_id,
                "Carrier": req.carrier,
                "Service": req.service_type,
                "Destination": req.package.destination_country,
                "Weight (kg)": req.package.weight_kg,
                "Value": f"SGD {req.package.declared_value:,.2f}",
                "Pickup Time": req.pickup_time,
                "Attempts": req.retry_count,
                "Created": req.created_at[:19] if req.created_at else "—"
            })
        
        df = pd.DataFrame(log_data)
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        # Email Transmission Logs
        st.markdown("---")
        st.subheader("📬 Email Transmission Logs")
        
        email_logs = system.email_sender.get_logs()
        if email_logs:
            log_display = []
            for log in email_logs:
                action_emoji = {
                    "send_success": "✅",
                    "send_failed": "❌",
                    "send_attempt": "📤",
                    "retry": "🔄"
                }.get(log["action"], "📝")
                
                log_display.append({
                    "": action_emoji,
                    "Log ID": log["log_id"],
                    "Request ID": log["request_id"],
                    "Action": log["action"],
                    "Status": log["status"],
                    "Details": log.get("details", ""),
                    "Error": log.get("error", ""),
                    "Timestamp": log["timestamp"][:19]
                })
            
            df_logs = pd.DataFrame(log_display)
            st.dataframe(df_logs, use_container_width=True, hide_index=True)
        else:
            st.info("No email transmission logs yet")
    else:
        st.info("📭 No pickup requests yet. Use the **Shipping Zone Trigger** tab to create one.")


# ── TAB 3: EMAIL PREVIEW ──
with tab3:
    st.subheader("📧 Email Preview")
    
    if st.session_state.last_email_preview:
        preview = st.session_state.last_email_preview
        
        col_p1, col_p2 = st.columns([3, 1])
        with col_p1:
            st.markdown(f"**Request ID:** `{preview['request_id']}`")
            st.markdown(f"**Carrier:** {preview['carrier']}")
        with col_p2:
            st.markdown(f"**Generated:** {datetime.now().strftime('%H:%M:%S')}")
        
        st.markdown("---")
        
        st.markdown(f"**Subject:** `{preview['subject']}`")
        st.markdown(f"**To:** `{CARRIER_CONFIG[preview['carrier']]['email']}`")
        st.markdown(f"**From:** `{EMAIL_CONFIG['sender_email']}`")
        
        st.markdown("---")
        st.markdown("**Email Body:**")
        st.markdown(f'<div class="email-preview">{preview["body"]}</div>',
                    unsafe_allow_html=True)
    else:
        st.info("📭 No email preview available. Trigger a pickup request first.")
    
    # Show all past email bodies
    if system.pickup_requests:
        st.markdown("---")
        st.subheader("📜 All Generated Emails")
        
        for req_id, req in system.pickup_requests.items():
            if req.email_body:
                with st.expander(f"📧 {req_id} — {req.package.order_id} ({req.carrier})"):
                    st.markdown(f"**Subject:** `{req.email_subject}`")
                    st.code(req.email_body, language=None)


# ── TAB 4: SYSTEM ANALYTICS ──
with tab4:
    st.subheader("📊 System Analytics")
    
    if system.pickup_requests:
        # Carrier breakdown
        col_an1, col_an2 = st.columns(2)
        
        with col_an1:
            st.markdown("##### 🚚 By Carrier")
            carrier_counts = {}
            for req in system.pickup_requests.values():
                carrier_counts[req.carrier] = carrier_counts.get(req.carrier, 0) + 1
            
            carrier_df = pd.DataFrame([
                {"Carrier": k, "Requests": v}
                for k, v in carrier_counts.items()
            ])
            st.bar_chart(carrier_df.set_index("Carrier"))
        
        with col_an2:
            st.markdown("##### 📍 By Destination")
            dest_counts = {}
            for req in system.pickup_requests.values():
                country = req.package.destination_country
                dest_counts[country] = dest_counts.get(country, 0) + 1
            
            dest_df = pd.DataFrame([
                {"Country": k, "Shipments": v}
                for k, v in dest_counts.items()
            ])
            st.bar_chart(dest_df.set_index("Country"))
        
        # Status breakdown
        st.markdown("---")
        st.markdown("##### 📈 Status Breakdown")
        
        status_counts = {}
        for req in system.pickup_requests.values():
            s = req.status.value
            status_counts[s] = status_counts.get(s, 0) + 1
        
        col_s1, col_s2, col_s3 = st.columns(3)
        for i, (status, count) in enumerate(status_counts.items()):
            col = [col_s1, col_s2, col_s3][i % 3]
            with col:
                emoji = {"sent": "✅", "failed": "❌", "pending": "⏳"}.get(status, "📝")
                st.metric(f"{emoji} {status.upper()}", count)
        
        # Value summary
        st.markdown("---")
        st.markdown("##### 💰 Shipment Value Summary")
        total_value = sum(r.package.declared_value for r in system.pickup_requests.values())
        total_weight = sum(r.package.weight_kg for r in system.pickup_requests.values())
        total_items = sum(r.package.item_count for r in system.pickup_requests.values())
        
        col_v1, col_v2, col_v3 = st.columns(3)
        with col_v1:
            st.metric("💰 Total Declared Value", f"SGD {total_value:,.2f}")
        with col_v2:
            st.metric("⚖️ Total Weight", f"{total_weight:,.1f} kg")
        with col_v3:
            st.metric("📦 Total Items", f"{total_items:,}")
        
        # Carrier configuration reference
        st.markdown("---")
        st.markdown("##### 🔧 Carrier Configuration")
        config_data = []
        for name, config in CARRIER_CONFIG.items():
            config_data.append({
                "Carrier": config["name"],
                "Email": config["email"],
                "Max Weight (kg)": config["max_weight_kg"],
                "Services": ", ".join(config["service_types"])
            })
        st.dataframe(pd.DataFrame(config_data), use_container_width=True, hide_index=True)
    else:
        st.info("📊 No data to analyze yet. Trigger some pickup requests first!")


# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("---")
st.caption(
    "📧 FedEx Auto Email System — Subsystem 3 | "
    "ICT304 AI System Design — Warehouse Intelligence System | "
    f"Session: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
)
