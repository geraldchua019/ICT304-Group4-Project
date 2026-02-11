#!/usr/bin/env python3
"""
FEDEX AUTO EMAIL SYSTEM (Subsystem 3)
For ICT304 Assignment - Warehouse Intelligence System

Purpose:
- Automates outbound logistics by generating and sending carrier pickup requests
- Triggered when a packed package enters the shipping zone
- Validates package data, generates email, sends via SMTP, logs confirmation

Assumptions:
- Package data comes from inventory/order management system
- SMTP email sending (simulated for prototype)
- Supports multiple carriers (FedEx, DHL, UPS)
- Retry mechanism for failed transmissions
"""

import json
import smtplib
import random
import string
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Tuple, Optional
from enum import Enum
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart


# ========== SYSTEM CONFIGURATION ==========
EMAIL_CONFIG = {
    "smtp_server": "smtp.gmail.com",
    "smtp_port": 587,
    "use_tls": True,
    "sender_email": "warehouse.system@smartwarehouse.com",
    "sender_name": "Smart Inventory System",
    "max_retries": 3,
    "retry_delay_seconds": 5,
    "simulation_mode": True  # Set to False for real SMTP
}

CARRIER_CONFIG = {
    "FedEx": {
        "email": "pickup@fedex.com",
        "name": "FedEx Express",
        "service_types": ["Priority Overnight", "Standard Overnight", "2Day", "Ground"],
        "max_weight_kg": 68.0,
        "max_dimensions_cm": (274, 330, 330)  # L x W x H
    },
    "DHL": {
        "email": "pickup@dhl.com",
        "name": "DHL Express",
        "service_types": ["Express Worldwide", "Express 12:00", "Economy Select"],
        "max_weight_kg": 70.0,
        "max_dimensions_cm": (300, 300, 300)
    },
    "UPS": {
        "email": "pickup@ups.com",
        "name": "UPS",
        "service_types": ["Next Day Air", "2nd Day Air", "Ground", "Express Saver"],
        "max_weight_kg": 68.0,
        "max_dimensions_cm": (274, 330, 330)
    }
}

WAREHOUSE_INFO = {
    "name": "Smart Electronics Warehouse",
    "address": "50 Nanyang Avenue, Singapore 639798",
    "contact": "+65 6791 2000",
    "operating_hours": "08:00 - 18:00 SGT",
    "dock_number": "Dock B-3"
}


# ========== DATA MODELS ==========
class PackageStatus(Enum):
    PACKED = "packed"
    VALIDATED = "validated"
    PICKUP_REQUESTED = "pickup_requested"
    PICKUP_CONFIRMED = "pickup_confirmed"
    SHIPPED = "shipped"
    FAILED = "failed"


class EmailStatus(Enum):
    PENDING = "pending"
    SENT = "sent"
    FAILED = "failed"
    RETRYING = "retrying"


@dataclass
class ShippingPackage:
    """Represents a packed package ready for shipping"""
    order_id: str
    weight_kg: float
    dimensions_cm: Tuple[float, float, float]  # L x W x H
    destination_address: str
    destination_country: str
    contact_name: str
    contact_email: str
    contact_phone: str
    contents_description: str
    item_count: int
    declared_value: float
    currency: str = "SGD"
    fragile: bool = False
    status: PackageStatus = PackageStatus.PACKED
    packed_at: str = ""
    
    def __post_init__(self):
        if not self.packed_at:
            self.packed_at = datetime.now().isoformat()
    
    def volume_cm3(self) -> float:
        """Calculate package volume"""
        return self.dimensions_cm[0] * self.dimensions_cm[1] * self.dimensions_cm[2]
    
    def dimensions_str(self) -> str:
        """Format dimensions as string"""
        return f"{self.dimensions_cm[0]}x{self.dimensions_cm[1]}x{self.dimensions_cm[2]} cm"


@dataclass
class PickupRequest:
    """Represents a carrier pickup request"""
    request_id: str
    package: ShippingPackage
    carrier: str
    service_type: str
    pickup_time: str
    email_subject: str = ""
    email_body: str = ""
    status: EmailStatus = EmailStatus.PENDING
    retry_count: int = 0
    created_at: str = ""
    sent_at: str = ""
    confirmed_at: str = ""
    error_message: str = ""
    
    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()


@dataclass
class EmailLog:
    """Log entry for email transmission attempts"""
    log_id: str
    request_id: str
    timestamp: str
    action: str  # "send_attempt", "send_success", "send_failed", "retry"
    status: str
    details: str = ""
    error: str = ""


# ========== PACKAGE VALIDATOR ==========
class PackageValidator:
    """Validates package data before generating pickup request"""
    
    def __init__(self):
        self.errors: List[str] = []
        self.warnings: List[str] = []
    
    def validate(self, package: ShippingPackage) -> Tuple[bool, List[str], List[str]]:
        """
        Validate all package fields.
        Returns: (is_valid, errors, warnings)
        """
        self.errors = []
        self.warnings = []
        
        # Required field checks
        self._validate_order_id(package.order_id)
        self._validate_weight(package.weight_kg)
        self._validate_dimensions(package.dimensions_cm)
        self._validate_address(package.destination_address, package.destination_country)
        self._validate_contact(package.contact_name, package.contact_email, package.contact_phone)
        self._validate_contents(package.contents_description, package.item_count)
        self._validate_value(package.declared_value)
        
        is_valid = len(self.errors) == 0
        return is_valid, self.errors.copy(), self.warnings.copy()
    
    def _validate_order_id(self, order_id: str):
        if not order_id or not order_id.strip():
            self.errors.append("Order ID is required")
        elif len(order_id) < 3:
            self.errors.append("Order ID must be at least 3 characters")
    
    def _validate_weight(self, weight: float):
        if weight <= 0:
            self.errors.append("Weight must be greater than 0 kg")
        elif weight > 100:
            self.warnings.append(f"Weight {weight}kg exceeds typical limits — verify with carrier")
    
    def _validate_dimensions(self, dimensions: Tuple):
        if len(dimensions) != 3:
            self.errors.append("Dimensions must have 3 values (L x W x H)")
            return
        for i, dim in enumerate(dimensions):
            label = ["Length", "Width", "Height"][i]
            if dim <= 0:
                self.errors.append(f"{label} must be greater than 0 cm")
        
        volume = dimensions[0] * dimensions[1] * dimensions[2]
        if volume > 1_000_000:
            self.warnings.append(f"Volume {volume:,.0f} cm³ is very large — verify dimensions")
    
    def _validate_address(self, address: str, country: str):
        if not address or not address.strip():
            self.errors.append("Destination address is required")
        elif len(address) < 10:
            self.errors.append("Destination address appears too short")
        
        if not country or not country.strip():
            self.errors.append("Destination country is required")
    
    def _validate_contact(self, name: str, email: str, phone: str):
        if not name or not name.strip():
            self.errors.append("Contact name is required")
        
        if not email or not email.strip():
            self.errors.append("Contact email is required")
        elif "@" not in email or "." not in email:
            self.errors.append("Contact email format is invalid")
        
        if not phone or not phone.strip():
            self.warnings.append("Contact phone is missing — recommended for delivery")
    
    def _validate_contents(self, description: str, count: int):
        if not description or not description.strip():
            self.warnings.append("Contents description is empty — may delay customs")
        if count <= 0:
            self.errors.append("Item count must be at least 1")
    
    def _validate_value(self, value: float):
        if value < 0:
            self.errors.append("Declared value cannot be negative")
        elif value == 0:
            self.warnings.append("Declared value is $0 — verify before shipping")


# ========== EMAIL TEMPLATE GENERATOR ==========
class EmailTemplateGenerator:
    """Generates standardized pickup request emails"""
    
    def generate_subject(self, request: PickupRequest) -> str:
        """Generate email subject line"""
        return (
            f"Pickup Request – Order {request.package.order_id} | "
            f"{request.carrier} {request.service_type}"
        )
    
    def generate_body(self, request: PickupRequest) -> str:
        """Generate professional email body"""
        pkg = request.package
        carrier_info = CARRIER_CONFIG.get(request.carrier, {})
        carrier_name = carrier_info.get("name", request.carrier)
        
        fragile_note = "\n⚠️  FRAGILE — Handle with care\n" if pkg.fragile else ""
        
        body = f"""Dear {carrier_name} Team,

We request a pickup for the following shipment from {WAREHOUSE_INFO['name']}.

════════════════════════════════════════
  SHIPMENT DETAILS
════════════════════════════════════════

  Order ID:        {pkg.order_id}
  Service Type:    {request.service_type}
  Request ID:      {request.request_id}

────────────────────────────────────────
  PACKAGE INFORMATION
────────────────────────────────────────

  Weight:          {pkg.weight_kg} kg
  Dimensions:      {pkg.dimensions_str()}
  Volume:          {pkg.volume_cm3():,.0f} cm³
  Contents:        {pkg.contents_description}
  Item Count:      {pkg.item_count}
  Declared Value:  {pkg.currency} {pkg.declared_value:,.2f}
{fragile_note}
────────────────────────────────────────
  DESTINATION
────────────────────────────────────────

  Recipient:       {pkg.contact_name}
  Address:         {pkg.destination_address}
  Country:         {pkg.destination_country}
  Email:           {pkg.contact_email}
  Phone:           {pkg.contact_phone}

────────────────────────────────────────
  PICKUP DETAILS
────────────────────────────────────────

  Pickup Location: {WAREHOUSE_INFO['address']}
  Dock Number:     {WAREHOUSE_INFO['dock_number']}
  Requested Time:  {request.pickup_time}
  Operating Hours: {WAREHOUSE_INFO['operating_hours']}
  Contact:         {WAREHOUSE_INFO['contact']}

════════════════════════════════════════

Kindly confirm the pickup schedule at your earliest convenience.

Regards,
{EMAIL_CONFIG['sender_name']}
{WAREHOUSE_INFO['name']}
{WAREHOUSE_INFO['address']}
Tel: {WAREHOUSE_INFO['contact']}

---
This is an automated message generated by the Smart Warehouse Management System.
Request ID: {request.request_id} | Generated: {request.created_at}
"""
        return body


# ========== EMAIL SENDER ==========
class EmailSender:
    """Handles email transmission with retry mechanism"""
    
    def __init__(self, simulation_mode: bool = True):
        self.simulation_mode = simulation_mode
        self.logs: List[EmailLog] = []
        self._log_counter = 0
    
    def _generate_log_id(self) -> str:
        self._log_counter += 1
        return f"LOG-{self._log_counter:06d}"
    
    def send(self, request: PickupRequest) -> Tuple[bool, str]:
        """
        Send pickup request email with retry mechanism.
        Returns: (success, message)
        """
        carrier_info = CARRIER_CONFIG.get(request.carrier)
        if not carrier_info:
            error_msg = f"Unknown carrier: {request.carrier}"
            self._add_log(request.request_id, "send_failed", "failed", error=error_msg)
            return False, error_msg
        
        recipient = carrier_info["email"]
        
        for attempt in range(1, EMAIL_CONFIG["max_retries"] + 1):
            request.retry_count = attempt
            
            if attempt > 1:
                request.status = EmailStatus.RETRYING
                self._add_log(
                    request.request_id, "retry",
                    f"retrying (attempt {attempt}/{EMAIL_CONFIG['max_retries']})",
                    details=f"Retrying after previous failure"
                )
                print(f"     ⟳ Retry attempt {attempt}/{EMAIL_CONFIG['max_retries']}...")
            
            try:
                if self.simulation_mode:
                    success = self._simulate_send(request, recipient)
                else:
                    success = self._real_send(request, recipient)
                
                if success:
                    request.status = EmailStatus.SENT
                    request.sent_at = datetime.now().isoformat()
                    self._add_log(
                        request.request_id, "send_success", "sent",
                        details=f"Email sent to {recipient} (attempt {attempt})"
                    )
                    return True, f"Email sent successfully to {recipient} on attempt {attempt}"
                else:
                    raise Exception("Simulated transmission failure")
                    
            except Exception as e:
                error_msg = str(e)
                self._add_log(
                    request.request_id, "send_attempt", "failed",
                    error=error_msg,
                    details=f"Attempt {attempt}/{EMAIL_CONFIG['max_retries']}"
                )
                print(f"     ✗ Attempt {attempt} failed: {error_msg}")
        
        # All retries exhausted
        request.status = EmailStatus.FAILED
        final_msg = f"Failed after {EMAIL_CONFIG['max_retries']} attempts"
        self._add_log(request.request_id, "send_failed", "failed", error=final_msg)
        return False, final_msg
    
    def _simulate_send(self, request: PickupRequest, recipient: str) -> bool:
        """Simulate email sending (prototype mode)"""
        # 90% success rate simulation
        success = random.random() < 0.90
        
        if success:
            print(f"\n     ┌─────────────────────────────────────────┐")
            print(f"     │  📧 SIMULATED EMAIL SENT                │")
            print(f"     ├─────────────────────────────────────────┤")
            print(f"     │  To:      {recipient:<30}│")
            print(f"     │  Subject: {request.email_subject[:30]:<30}│")
            print(f"     │  Status:  ✅ Delivered                   │")
            print(f"     └─────────────────────────────────────────┘")
        
        return success
    
    def _real_send(self, request: PickupRequest, recipient: str) -> bool:
        """Send email via real SMTP server"""
        try:
            msg = MIMEMultipart()
            msg['From'] = f"{EMAIL_CONFIG['sender_name']} <{EMAIL_CONFIG['sender_email']}>"
            msg['To'] = recipient
            msg['Subject'] = request.email_subject
            msg.attach(MIMEText(request.email_body, 'plain'))
            
            server = smtplib.SMTP(EMAIL_CONFIG['smtp_server'], EMAIL_CONFIG['smtp_port'])
            if EMAIL_CONFIG['use_tls']:
                server.starttls()
            # server.login(username, password)  # Uncomment for real credentials
            server.send_message(msg)
            server.quit()
            return True
        except Exception as e:
            print(f"     SMTP Error: {e}")
            return False
    
    def _add_log(self, request_id: str, action: str, status: str,
                 details: str = "", error: str = ""):
        log = EmailLog(
            log_id=self._generate_log_id(),
            request_id=request_id,
            timestamp=datetime.now().isoformat(),
            action=action,
            status=status,
            details=details,
            error=error
        )
        self.logs.append(log)
    
    def get_logs(self, request_id: Optional[str] = None) -> List[Dict]:
        """Get transmission logs, optionally filtered by request ID"""
        logs = self.logs
        if request_id:
            logs = [l for l in logs if l.request_id == request_id]
        return [asdict(l) for l in logs]


# ========== MAIN SYSTEM ==========
class FedExAutoEmailSystem:
    """
    FedEx Auto Email System — Main Controller
    
    Orchestrates the full shipping zone → carrier pickup workflow:
    1. Trigger (package enters shipping zone)
    2. Validate package data
    3. Generate pickup request object
    4. Format email from template
    5. Send email via SMTP (with retry)
    6. Handle confirmation / failure
    """
    
    def __init__(self):
        self.validator = PackageValidator()
        self.template_gen = EmailTemplateGenerator()
        self.email_sender = EmailSender(simulation_mode=EMAIL_CONFIG["simulation_mode"])
        self.pickup_requests: Dict[str, PickupRequest] = {}
        self.packages: Dict[str, ShippingPackage] = {}
        self._request_counter = 0
    
    def _generate_request_id(self) -> str:
        self._request_counter += 1
        suffix = ''.join(random.choices(string.ascii_uppercase + string.digits, k=4))
        return f"PKR-{self._request_counter:06d}-{suffix}"
    
    def trigger_shipping(self, package: ShippingPackage,
                         carrier: str = "FedEx",
                         service_type: str = "Standard Overnight") -> Dict:
        """
        Main entry point — triggered when a packed package enters the shipping zone.
        
        Executes the full pipeline:
        Step 1: Validate → Step 2: Generate Request → Step 3: Format Email →
        Step 4: Send Email → Step 5: Handle Confirmation
        
        Returns: Complete result dictionary with status and details
        """
        print(f"\n{'='*65}")
        print(f"  📦 FEDEX AUTO EMAIL SYSTEM — SHIPPING TRIGGER")
        print(f"{'='*65}")
        print(f"  Order:   {package.order_id}")
        print(f"  Carrier: {carrier}")
        print(f"  Service: {service_type}")
        print(f"{'='*65}")
        
        result = {
            "order_id": package.order_id,
            "carrier": carrier,
            "service_type": service_type,
            "steps": {},
            "final_status": "pending",
            "timestamp": datetime.now().isoformat()
        }
        
        # ── Step 1: Data Validation ──
        print(f"\n  ┌── Step 1: Data Validation")
        is_valid, errors, warnings = self._validate_package(package)
        
        result["steps"]["validation"] = {
            "valid": is_valid,
            "errors": errors,
            "warnings": warnings
        }
        
        if not is_valid:
            print(f"  │   ✗ FAILED — {len(errors)} error(s)")
            for err in errors:
                print(f"  │     • {err}")
            print(f"  └── Pipeline halted\n")
            result["final_status"] = "validation_failed"
            package.status = PackageStatus.FAILED
            self.packages[package.order_id] = package
            return result
        
        print(f"  │   ✓ Passed — {len(warnings)} warning(s)")
        for warn in warnings:
            print(f"  │     ⚠ {warn}")
        package.status = PackageStatus.VALIDATED
        
        # ── Step 2: Generate Pickup Request ──
        print(f"  ├── Step 2: Generate Pickup Request")
        request = self._generate_pickup_request(package, carrier, service_type)
        result["request_id"] = request.request_id
        result["steps"]["request_generation"] = {
            "request_id": request.request_id,
            "pickup_time": request.pickup_time
        }
        print(f"  │   ✓ Request ID: {request.request_id}")
        print(f"  │   ✓ Pickup Time: {request.pickup_time}")
        
        # ── Step 3: Format Email ──
        print(f"  ├── Step 3: Format Email Template")
        self._format_email(request)
        result["steps"]["email_formatting"] = {
            "subject": request.email_subject,
            "body_length": len(request.email_body)
        }
        print(f"  │   ✓ Subject: {request.email_subject}")
        print(f"  │   ✓ Body: {len(request.email_body)} characters")
        
        # ── Step 4: Send Email ──
        print(f"  ├── Step 4: Email Transmission")
        success, message = self._send_email(request)
        result["steps"]["transmission"] = {
            "success": success,
            "message": message,
            "attempts": request.retry_count
        }
        
        # ── Step 5: Handle Confirmation ──
        print(f"  └── Step 5: Confirmation Handling")
        confirmation = self._handle_confirmation(request, success)
        result["steps"]["confirmation"] = confirmation
        result["final_status"] = confirmation["status"]
        
        # Store in system
        self.pickup_requests[request.request_id] = request
        self.packages[package.order_id] = package
        
        # Print final summary
        status_icon = "✅" if success else "❌"
        print(f"\n  {status_icon} Final Status: {confirmation['status'].upper()}")
        print(f"{'='*65}\n")
        
        return result
    
    def _validate_package(self, package: ShippingPackage) -> Tuple[bool, List[str], List[str]]:
        """Step 1: Validate package data"""
        return self.validator.validate(package)
    
    def _generate_pickup_request(self, package: ShippingPackage,
                                  carrier: str, service_type: str) -> PickupRequest:
        """Step 2: Create structured pickup request object"""
        # Calculate next available pickup slot
        now = datetime.now()
        if now.hour < 16:
            pickup_time = now.replace(hour=16, minute=0, second=0).strftime("%Y-%m-%d %H:%M")
        else:
            next_day = now + timedelta(days=1)
            pickup_time = next_day.replace(hour=9, minute=0, second=0).strftime("%Y-%m-%d %H:%M")
        
        request = PickupRequest(
            request_id=self._generate_request_id(),
            package=package,
            carrier=carrier,
            service_type=service_type,
            pickup_time=pickup_time
        )
        
        return request
    
    def _format_email(self, request: PickupRequest):
        """Step 3: Generate email subject and body from template"""
        request.email_subject = self.template_gen.generate_subject(request)
        request.email_body = self.template_gen.generate_body(request)
    
    def _send_email(self, request: PickupRequest) -> Tuple[bool, str]:
        """Step 4: Transmit email with retry mechanism"""
        package = request.package
        package.status = PackageStatus.PICKUP_REQUESTED
        return self.email_sender.send(request)
    
    def _handle_confirmation(self, request: PickupRequest, success: bool) -> Dict:
        """Step 5: Handle post-transmission confirmation"""
        if success:
            request.confirmed_at = datetime.now().isoformat()
            request.package.status = PackageStatus.PICKUP_CONFIRMED
            
            print(f"       ✓ Confirmation logged at {request.confirmed_at}")
            print(f"       ✓ Order status → PICKUP_CONFIRMED")
            print(f"       ✓ Dashboard updated")
            
            return {
                "status": "pickup_confirmed",
                "confirmed_at": request.confirmed_at,
                "order_status": PackageStatus.PICKUP_CONFIRMED.value
            }
        else:
            request.package.status = PackageStatus.FAILED
            
            print(f"       ✗ Pickup request FAILED")
            print(f"       ✗ Alert sent to warehouse manager")
            print(f"       ✗ Error logged in system")
            
            return {
                "status": "failed",
                "error": request.error_message or "Email transmission failed after retries",
                "order_status": PackageStatus.FAILED.value,
                "admin_alert": True
            }
    
    def get_dashboard_data(self) -> Dict:
        """Get real-time dashboard data"""
        total = len(self.pickup_requests)
        confirmed = sum(1 for r in self.pickup_requests.values()
                        if r.status == EmailStatus.SENT)
        failed = sum(1 for r in self.pickup_requests.values()
                     if r.status == EmailStatus.FAILED)
        pending = total - confirmed - failed
        
        success_rate = (confirmed / total * 100) if total > 0 else 0
        
        recent_requests = []
        for req_id, req in sorted(self.pickup_requests.items(),
                                   key=lambda x: x[1].created_at, reverse=True)[:10]:
            recent_requests.append({
                "request_id": req_id,
                "order_id": req.package.order_id,
                "carrier": req.carrier,
                "destination": req.package.destination_country,
                "status": req.status.value,
                "created_at": req.created_at,
                "weight_kg": req.package.weight_kg,
                "value": req.package.declared_value
            })
        
        return {
            "summary": {
                "total_requests": total,
                "confirmed": confirmed,
                "failed": failed,
                "pending": pending,
                "success_rate": round(success_rate, 1)
            },
            "recent_requests": recent_requests,
            "email_logs": self.email_sender.get_logs(),
            "timestamp": datetime.now().isoformat()
        }
    
    def save_shipping_report(self) -> Dict:
        """Save comprehensive shipping report as JSON"""
        dashboard = self.get_dashboard_data()
        
        report = {
            "report_type": "FedEx Auto Email System Report",
            "generated_at": datetime.now().isoformat(),
            "warehouse": WAREHOUSE_INFO,
            "email_config": {
                "smtp_server": EMAIL_CONFIG["smtp_server"],
                "simulation_mode": EMAIL_CONFIG["simulation_mode"],
                "max_retries": EMAIL_CONFIG["max_retries"]
            },
            "carriers_configured": list(CARRIER_CONFIG.keys()),
            "dashboard": dashboard["summary"],
            "requests": [],
            "email_logs": dashboard["email_logs"]
        }
        
        for req_id, req in self.pickup_requests.items():
            report["requests"].append({
                "request_id": req_id,
                "order_id": req.package.order_id,
                "carrier": req.carrier,
                "service_type": req.service_type,
                "destination": req.package.destination_country,
                "weight_kg": req.package.weight_kg,
                "dimensions": req.package.dimensions_str(),
                "declared_value": f"{req.package.currency} {req.package.declared_value:,.2f}",
                "status": req.status.value,
                "pickup_time": req.pickup_time,
                "retry_count": req.retry_count,
                "created_at": req.created_at,
                "sent_at": req.sent_at,
                "confirmed_at": req.confirmed_at
            })
        
        filename = f"shipping_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"\n  📄 Shipping report saved to: {filename}")
        return report


# ========== DEMONSTRATION ==========
def run_fedex_demo():
    """Run demonstration of FedEx Auto Email System"""
    print("=" * 70)
    print("  FEDEX AUTO EMAIL SYSTEM — DEMONSTRATION")
    print("  ICT304 - Warehouse Intelligence System (Subsystem 3)")
    print("=" * 70)
    
    # Initialize system
    print("\n1. 🔧 Initializing FedEx Auto Email System...")
    system = FedExAutoEmailSystem()
    print(f"   ✓ System initialized")
    print(f"   ✓ Simulation mode: {EMAIL_CONFIG['simulation_mode']}")
    print(f"   ✓ Carriers: {', '.join(CARRIER_CONFIG.keys())}")
    
    # ── Demo 1: Standard FedEx Pickup ──
    print(f"\n2. 📦 Demo 1: Standard FedEx Pickup Request")
    
    package1 = ShippingPackage(
        order_id="ORD-10234",
        weight_kg=3.2,
        dimensions_cm=(30, 20, 15),
        destination_address="123 Orchard Road, #05-01, Singapore 238858",
        destination_country="Singapore",
        contact_name="John Tan",
        contact_email="john.tan@example.com",
        contact_phone="+65 9123 4567",
        contents_description="Electronics - Laptop x1",
        item_count=1,
        declared_value=1200.00,
        fragile=True
    )
    
    result1 = system.trigger_shipping(package1, carrier="FedEx", service_type="Standard Overnight")
    
    # ── Demo 2: DHL International Shipment ──
    print(f"\n3. 📦 Demo 2: DHL International Shipment")
    
    package2 = ShippingPackage(
        order_id="ORD-10235",
        weight_kg=8.5,
        dimensions_cm=(50, 40, 30),
        destination_address="1-1-2 Oshiage, Sumida-ku, Tokyo 131-8634",
        destination_country="Japan",
        contact_name="Yuki Tanaka",
        contact_email="yuki.tanaka@example.jp",
        contact_phone="+81 3-1234-5678",
        contents_description="Electronics - Smartphones x5, Accessories x3",
        item_count=8,
        declared_value=4500.00,
        currency="SGD"
    )
    
    result2 = system.trigger_shipping(package2, carrier="DHL", service_type="Express Worldwide")
    
    # ── Demo 3: UPS Bulk Order ──
    print(f"\n4. 📦 Demo 3: UPS Bulk Wholesale Order")
    
    package3 = ShippingPackage(
        order_id="ORD-10236",
        weight_kg=25.0,
        dimensions_cm=(80, 60, 50),
        destination_address="Level 3, 50 Bourke Street, Melbourne VIC 3000",
        destination_country="Australia",
        contact_name="Sarah Chen",
        contact_email="sarah.c@techstore.com.au",
        contact_phone="+61 3 9876 5432",
        contents_description="Electronics - Laptops x10, Chargers x10",
        item_count=20,
        declared_value=12000.00,
        currency="SGD",
        fragile=True
    )
    
    result3 = system.trigger_shipping(package3, carrier="UPS", service_type="Express Saver")
    
    # ── Demo 4: Validation Failure ──
    print(f"\n5. ⚠️  Demo 4: Invalid Package (Validation Failure)")
    
    invalid_package = ShippingPackage(
        order_id="",
        weight_kg=-1.0,
        dimensions_cm=(0, 0, 0),
        destination_address="",
        destination_country="",
        contact_name="",
        contact_email="not-an-email",
        contact_phone="",
        contents_description="",
        item_count=0,
        declared_value=-50.00
    )
    
    result4 = system.trigger_shipping(invalid_package, carrier="FedEx")
    
    # ── Dashboard Summary ──
    print(f"\n6. 📊 Dashboard Summary")
    dashboard = system.get_dashboard_data()
    summary = dashboard["summary"]
    
    print(f"   ┌─────────────────────────────────────────┐")
    print(f"   │  SHIPPING DASHBOARD                     │")
    print(f"   ├─────────────────────────────────────────┤")
    print(f"   │  Total Requests:  {summary['total_requests']:<22}│")
    print(f"   │  Confirmed:       {summary['confirmed']:<22}│")
    print(f"   │  Failed:          {summary['failed']:<22}│")
    print(f"   │  Pending:         {summary['pending']:<22}│")
    print(f"   │  Success Rate:    {summary['success_rate']}%{' '*(20-len(str(summary['success_rate'])))}│")
    print(f"   └─────────────────────────────────────────┘")
    
    print(f"\n   Recent Requests:")
    for req in dashboard["recent_requests"]:
        status_icon = "✅" if req["status"] == "sent" else "❌" if req["status"] == "failed" else "⏳"
        print(f"   {status_icon} {req['request_id']} | {req['order_id']} | "
              f"{req['carrier']} → {req['destination']} | {req['status']}")
    
    # ── Save Report ──
    print(f"\n7. 💾 Saving Shipping Report...")
    report = system.save_shipping_report()
    
    print(f"\n{'='*70}")
    print(f"  ✅ FEDEX AUTO EMAIL SYSTEM DEMONSTRATION COMPLETE")
    print(f"{'='*70}")
    
    return system


if __name__ == "__main__":
    import sys
    
    print("FEDEX AUTO EMAIL SYSTEM")
    print("ICT304 Assignment — Warehouse Intelligence System (Subsystem 3)")
    print("Automated Carrier Pickup Request System\n")
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--demo":
            system = run_fedex_demo()
        elif sys.argv[1] == "--quick-test":
            system = FedExAutoEmailSystem()
            pkg = ShippingPackage(
                order_id="TEST-001",
                weight_kg=2.0,
                dimensions_cm=(20, 15, 10),
                destination_address="123 Test Street, Singapore 123456",
                destination_country="Singapore",
                contact_name="Test User",
                contact_email="test@example.com",
                contact_phone="+65 9999 0000",
                contents_description="Test Package",
                item_count=1,
                declared_value=100.00
            )
            result = system.trigger_shipping(pkg)
            print(json.dumps(result, indent=2, default=str))
        else:
            print("Usage: python fedex_auto_email.py [--demo|--quick-test]")
    else:
        system = run_fedex_demo()
