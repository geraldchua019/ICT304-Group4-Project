#!/usr/bin/env python3
"""
AUTO-PICKER SUBSYSTEM - COMPLETE WITH ERROR HANDLING
For ICT304 Assignment - Warehouse Intelligence System

Features:
✅ Comprehensive input validation
✅ Error logging and recovery
✅ Inventory integrity protection
✅ Graceful degradation
✅ Production-ready error handling
✅ Type checking and bounds validation
✅ File operation safety
✅ Performance monitoring

Assumptions:
- Electronics items only
- Single-layer grid layout (6×24 configuration)
- 2D X-Y arrangement, no vertical stacking
- Known SKUs with fixed locations
- Top-down camera for item counting
"""

import math
import json
import os
import sys
import traceback
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional, Any, Union
from enum import Enum
from uuid import uuid4
import numpy as np

# ============================================================================
# CONFIGURATION & CONSTANTS
# ============================================================================

WAREHOUSE_CONFIG = {
    "grid_size": (6, 24),           # 6 aisles × 24 racks per aisle
    "aisle_width": 3.0,            # meters between aisles
    "rack_depth": 1.2,            # meters per rack
    "shelf_height": 0.0,          # Single layer = all items at same height
    "camera_positions": [(0, 0), (12, 0), (24, 0)],  # Camera coordinates for monitoring
    "max_order_quantity": 1000,    # Maximum items per order
    "max_order_items": 50,         # Maximum unique SKUs per order
    "min_customer_id_length": 3,   # Minimum customer ID length
    "max_customer_id_length": 100, # Maximum customer ID length
    "low_stock_threshold": 10,     # Threshold for low stock warning
    "walking_speed": 80,          # meters per minute
    "picking_time_per_item": 0.5,  # minutes per item
    "demo_item_cap": 5,           # Cap items in demo mode
}

VALID_CUSTOMER_TYPES = {"retail", "wholesale", "repair_shop"}
VALID_CATEGORIES = {"computers", "phones", "accessories", "components"}
VALID_ORDER_STATUSES = {"pending", "processing", "picking", "partially_fulfilled", "complete", "failed"}

# ============================================================================
# ERROR HANDLER CLASS
# ============================================================================

class ErrorHandler:
    """
    Comprehensive error handling and logging system.
    Tracks all errors, provides recovery mechanisms, and maintains error logs.
    """
    
    def __init__(self, log_file: str = "auto_picker_errors.log"):
        self.log_file = log_file
        self.error_counts = {
            "validation": 0,
            "inventory": 0,
            "system": 0,
            "file_io": 0,
            "recovered": 0
        }
        self.error_log = []
        self.warning_log = []
        self._initialize_log_file()
    
    def _initialize_log_file(self):
        """Initialize error log file with header"""
        try:
            with open(self.log_file, "w", encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write(f"AUTO-PICKER ERROR LOG\n")
                f.write(f"Started: {datetime.now().isoformat()}\n")
                f.write(f"=" * 80 + "\n\n")
        except Exception as e:
            print(f"⚠️  Warning: Could not initialize log file: {e}")
    
    def log_error(self, component: str, error: Exception, context: Dict = None, 
                  severity: str = "medium") -> Dict:
        """
        Log an error with full context and stack trace.
        
        Args:
            component: Component where error occurred
            error: The exception object
            context: Additional context dictionary
            severity: 'low', 'medium', 'high', or 'critical'
        
        Returns:
            Error entry dictionary
        """
        timestamp = datetime.now().isoformat()
        
        error_entry = {
            "timestamp": timestamp,
            "component": component,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "severity": severity,
            "stack_trace": traceback.format_exc(),
            "context": context or {}
        }
        
        # Update counts
        if severity == "critical":
            self.error_counts["system"] += 1
        else:
            self.error_counts[severity] = self.error_counts.get(severity, 0) + 1
        
        # Store in memory
        self.error_log.append(error_entry)
        
        # Write to file
        try:
            with open(self.log_file, "a", encoding='utf-8') as f:
                f.write("\n" + "=" * 60 + "\n")
                f.write(f"ERROR at {timestamp}\n")
                f.write(f"Component: {component}\n")
                f.write(f"Type: {type(error).__name__}\n")
                f.write(f"Message: {str(error)}\n")
                f.write(f"Severity: {severity}\n")
                
                if context:
                    f.write("\nContext:\n")
                    for key, value in context.items():
                        f.write(f"  {key}: {value}\n")
                
                f.write("\nStack Trace:\n")
                f.write(traceback.format_exc())
                f.write("=" * 60 + "\n")
        except Exception as e:
            print(f"⚠️  Could not write to log file: {e}")
        
        # Console output based on severity
        if severity == "critical":
            print(f"💥 CRITICAL ERROR in {component}: {error}")
        elif severity == "high":
            print(f"🚨 ERROR in {component}: {error}")
        elif severity == "medium":
            print(f"⚠️  Error in {component}: {error}")
        
        return error_entry
    
    def log_warning(self, component: str, message: str, context: Dict = None) -> Dict:
        """Log a warning (non-critical issue)"""
        warning_entry = {
            "timestamp": datetime.now().isoformat(),
            "component": component,
            "message": message,
            "context": context or {}
        }
        self.warning_log.append(warning_entry)
        print(f"⚠️  Warning in {component}: {message}")
        return warning_entry
    
    def create_error_response(self, error_type: str, message: str, 
                            details: Dict = None, recoverable: bool = False,
                            suggestion: str = None) -> Dict:
        """Create standardized error response dictionary"""
        if recoverable:
            self.error_counts["recovered"] += 1
        
        response = {
            "success": False,
            "error": {
                "type": error_type,
                "message": message,
                "details": details or {},
                "timestamp": datetime.now().isoformat(),
                "recoverable": recoverable
            }
        }
        
        if suggestion:
            response["error"]["suggestion"] = suggestion
        
        return response
    
    def create_success_response(self, data: Dict = None, message: str = None,
                              warnings: List = None) -> Dict:
        """Create standardized success response dictionary"""
        response = {
            "success": True,
            "timestamp": datetime.now().isoformat()
        }
        
        if data:
            response.update(data)
        if message:
            response["message"] = message
        if warnings:
            response["warnings"] = warnings
        
        return response
    
    def get_error_report(self) -> Dict:
        """Generate comprehensive error statistics report"""
        total_errors = sum(self.error_counts.values())
        
        # Count by severity
        severity_counts = {}
        for error in self.error_log[-100:]:  # Last 100 errors
            severity = error.get('severity', 'unknown')
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        return {
            "timestamp": datetime.now().isoformat(),
            "total_errors": total_errors,
            "error_counts": self.error_counts,
            "severity_breakdown": severity_counts,
            "recovery_rate": (
                (self.error_counts["recovered"] / max(total_errors, 1)) * 100
            ),
            "recent_errors": self.error_log[-5:] if self.error_log else [],
            "recent_warnings": self.warning_log[-5:] if self.warning_log else [],
            "log_file": self.log_file,
            "system_status": "healthy" if total_errors == 0 else "degraded"
        }
    
    def cleanup_old_logs(self, days_to_keep: int = 7):
        """Clean up log entries older than days_to_keep"""
        cutoff_time = datetime.now() - timedelta(days=days_to_keep)
        
        # Clean memory logs
        self.error_log = [
            log for log in self.error_log
            if datetime.fromisoformat(log['timestamp'].replace('Z', '+00:00')) > cutoff_time
        ]
        
        self.warning_log = [
            log for log in self.warning_log
            if datetime.fromisoformat(log['timestamp'].replace('Z', '+00:00')) > cutoff_time
        ]


# ============================================================================
# DATA MODELS WITH BUILT-IN VALIDATION
# ============================================================================

class OrderStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    PICKING = "picking"
    PARTIAL = "partially_fulfilled"
    COMPLETE = "complete"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class GridLocation:
    """Optimized for single-layer grid warehouse with built-in validation"""
    aisle: int
    rack: int
    x: float = 0.0
    y: float = 0.0
    
    def __post_init__(self):
        """Validate and calculate coordinates"""
        # Validate aisle range
        if not isinstance(self.aisle, int):
            raise TypeError(f"Aisle must be integer, got {type(self.aisle).__name__}")
        
        if not (1 <= self.aisle <= WAREHOUSE_CONFIG["grid_size"][0]):
            raise ValueError(
                f"Aisle must be between 1 and {WAREHOUSE_CONFIG['grid_size'][0]}, "
                f"got {self.aisle}"
            )
        
        # Validate rack range
        if not isinstance(self.rack, int):
            raise TypeError(f"Rack must be integer, got {type(self.rack).__name__}")
        
        if not (1 <= self.rack <= WAREHOUSE_CONFIG["grid_size"][1]):
            raise ValueError(
                f"Rack must be between 1 and {WAREHOUSE_CONFIG['grid_size'][1]}, "
                f"got {self.rack}"
            )
        
        # Calculate coordinates
        self.x = (self.aisle - 1) * WAREHOUSE_CONFIG["aisle_width"]
        self.y = (self.rack - 1) * WAREHOUSE_CONFIG["rack_depth"]
    
    def __str__(self):
        return f"A{self.aisle:02d}-R{self.rack:03d}"
    
    def distance_to(self, other: 'GridLocation') -> float:
        """Manhattan distance for grid movement"""
        if not isinstance(other, GridLocation):
            raise TypeError(f"Expected GridLocation, got {type(other).__name__}")
        
        if self.aisle == other.aisle:
            return abs(self.y - other.y)
        else:
            aisle_end_distance = min(self.y, WAREHOUSE_CONFIG["rack_depth"] * 12 - self.y)
            other_aisle_end = min(other.y, WAREHOUSE_CONFIG["rack_depth"] * 12 - other.y)
            aisle_cross = abs(self.aisle - other.aisle) * WAREHOUSE_CONFIG["aisle_width"]
            return aisle_end_distance + aisle_cross + other_aisle_end + abs(self.y - other.y)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            "aisle": self.aisle,
            "rack": self.rack,
            "x": self.x,
            "y": self.y
        }

@dataclass
class ElectronicsItem:
    """Electronics warehouse item with validation"""
    sku: str
    name: str
    category: str
    location: GridLocation
    quantity: int
    weight: float
    value: float
    dimensions: Tuple[int, int, int]
    fragile: bool = False
    
    def __post_init__(self):
        """Validate all item attributes"""
        # SKU validation
        if not isinstance(self.sku, str):
            raise TypeError(f"SKU must be string, got {type(self.sku).__name__}")
        
        if not self.sku.startswith("ELEC-"):
            raise ValueError(f"SKU must start with 'ELEC-', got '{self.sku}'")
        
        if len(self.sku) > 20:
            raise ValueError(f"SKU too long (max 20 chars): {self.sku}")
        
        # Name validation
        if not self.name or not isinstance(self.name, str):
            raise ValueError("Item name must be non-empty string")
        
        # Category validation
        if self.category not in VALID_CATEGORIES:
            raise ValueError(f"Invalid category '{self.category}'. Must be one of: {VALID_CATEGORIES}")
        
        # Quantity validation
        if not isinstance(self.quantity, int):
            raise TypeError(f"Quantity must be integer, got {type(self.quantity).__name__}")
        
        if self.quantity < 0:
            raise ValueError(f"Quantity cannot be negative: {self.quantity}")
        
        # Weight validation
        if not isinstance(self.weight, (int, float)):
            raise TypeError(f"Weight must be number, got {type(self.weight).__name__}")
        
        if self.weight <= 0:
            raise ValueError(f"Weight must be positive: {self.weight}")
        
        # Value validation
        if not isinstance(self.value, (int, float)):
            raise TypeError(f"Value must be number, got {type(self.value).__name__}")
        
        if self.value < 0:
            raise ValueError(f"Value cannot be negative: {self.value}")
        
        # Dimensions validation
        if not isinstance(self.dimensions, tuple) or len(self.dimensions) != 3:
            raise ValueError("Dimensions must be tuple of 3 integers")
        
        for dim in self.dimensions:
            if not isinstance(dim, int) or dim <= 0:
                raise ValueError(f"Dimensions must be positive integers: {self.dimensions}")
    
    def __str__(self):
        return f"{self.sku}: {self.name} ({self.quantity} units)"
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            "sku": self.sku,
            "name": self.name,
            "category": self.category,
            "location": str(self.location),
            "quantity": self.quantity,
            "weight": self.weight,
            "value": self.value,
            "dimensions": self.dimensions,
            "fragile": self.fragile
        }

@dataclass
class OrderItem:
    """Order item with validation and tracking"""
    sku: str
    quantity: int
    picked: int = 0
    status: str = "pending"
    
    def __post_init__(self):
        """Validate order item"""
        if not isinstance(self.sku, str):
            raise TypeError(f"SKU must be string, got {type(self.sku).__name__}")
        
        if not self.sku:
            raise ValueError("SKU cannot be empty")
        
        if not isinstance(self.quantity, int):
            raise TypeError(f"Quantity must be integer, got {type(self.quantity).__name__}")
        
        if self.quantity <= 0:
            raise ValueError(f"Quantity must be positive: {self.quantity}")
        
        if self.quantity > WAREHOUSE_CONFIG["max_order_quantity"]:
            raise ValueError(
                f"Quantity exceeds maximum ({WAREHOUSE_CONFIG['max_order_quantity']}): {self.quantity}"
            )
        
        if not isinstance(self.picked, int):
            raise TypeError(f"Picked must be integer, got {type(self.picked).__name__}")
        
        if self.picked < 0:
            raise ValueError(f"Picked cannot be negative: {self.picked}")
        
        valid_statuses = {"pending", "picking", "picked", "partial", "failed"}
        if self.status not in valid_statuses:
            raise ValueError(f"Invalid status '{self.status}'. Must be one of: {valid_statuses}")
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            "sku": self.sku,
            "quantity": self.quantity,
            "picked": self.picked,
            "status": self.status
        }

@dataclass
class CustomerOrder:
    """Customer order with validation and tracking"""
    order_id: str
    customer_id: str
    customer_type: str
    items: List[OrderItem]
    status: OrderStatus
    created_at: datetime
    priority: int = 3
    completed_at: Optional[datetime] = None
    picker_id: Optional[str] = None
    
    def __post_init__(self):
        """Validate order data"""
        # Order ID validation
        if not self.order_id or not isinstance(self.order_id, str):
            raise ValueError("Order ID must be non-empty string")
        
        # Customer ID validation
        if not self.customer_id or not isinstance(self.customer_id, str):
            raise ValueError("Customer ID must be non-empty string")
        
        if len(self.customer_id.strip()) < WAREHOUSE_CONFIG["min_customer_id_length"]:
            raise ValueError(
                f"Customer ID must be at least {WAREHOUSE_CONFIG['min_customer_id_length']} characters"
            )
        
        # Customer type validation
        if self.customer_type not in VALID_CUSTOMER_TYPES:
            raise ValueError(f"Invalid customer type '{self.customer_type}'")
        
        # Priority validation
        if not isinstance(self.priority, int):
            raise TypeError(f"Priority must be integer, got {type(self.priority).__name__}")
        
        if self.priority not in [1, 2, 3]:
            raise ValueError(f"Priority must be 1, 2, or 3, got {self.priority}")
        
        # Items validation
        if not self.items:
            raise ValueError("Order must have at least one item")
        
        if len(self.items) > WAREHOUSE_CONFIG["max_order_items"]:
            raise ValueError(
                f"Too many items (max {WAREHOUSE_CONFIG['max_order_items']}): {len(self.items)}"
            )
    
    def fulfillment_rate(self) -> float:
        """Calculate order fulfillment percentage"""
        try:
            total_requested = sum(item.quantity for item in self.items)
            total_picked = sum(item.picked for item in self.items)
            
            if total_requested == 0:
                return 0.0
            
            return (total_picked / total_requested) * 100
        except Exception:
            return 0.0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            "order_id": self.order_id,
            "customer_id": self.customer_id,
            "customer_type": self.customer_type,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "priority": self.priority,
            "picker_id": self.picker_id,
            "fulfillment_rate": self.fulfillment_rate(),
            "items": [item.to_dict() for item in self.items]
        }


# ============================================================================
# GRID-BASED ROUTE OPTIMIZER WITH ERROR HANDLING
# ============================================================================

class GridRouteOptimizer:
    """Grid-based route optimizer with comprehensive error handling"""
    
    def __init__(self, error_handler: Optional[ErrorHandler] = None):
        self.start_location = GridLocation(aisle=1, rack=1)
        self.error_handler = error_handler or ErrorHandler()
        self.route_cache = {}
        self.stats = {
            "routes_generated": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "total_distance": 0
        }
    
    def validate_location(self, location: GridLocation) -> Tuple[bool, str]:
        """Validate a grid location"""
        try:
            if not isinstance(location, GridLocation):
                return False, f"Expected GridLocation, got {type(location).__name__}"
            
            # Already validated in GridLocation.__post_init__
            return True, ""
        except Exception as e:
            return False, str(e)
    
    def manhattan_distance(self, loc1: GridLocation, loc2: GridLocation) -> float:
        """Calculate Manhattan distance with validation"""
        try:
            # Validate inputs
            valid1, msg1 = self.validate_location(loc1)
            valid2, msg2 = self.validate_location(loc2)
            
            if not valid1:
                raise ValueError(f"Invalid location 1: {msg1}")
            if not valid2:
                raise ValueError(f"Invalid location 2: {msg2}")
            
            return loc1.distance_to(loc2)
            
        except Exception as e:
            self.error_handler.log_error(
                "manhattan_distance", e,
                {
                    "loc1": str(loc1) if hasattr(loc1, '__str__') else str(type(loc1)),
                    "loc2": str(loc2) if hasattr(loc2, '__str__') else str(type(loc2))
                },
                severity="medium"
            )
            raise
    
    def generate_grid_route(self, items: List[ElectronicsItem]) -> List[GridLocation]:
        """
        Generate optimal picking route for single-layer grid.
        Strategy: Process by aisle, then by rack number within aisle.
        """
        try:
            # Validate input
            if not items:
                return []
            
            if not isinstance(items, list):
                raise TypeError(f"Expected list, got {type(items).__name__}")
            
            # Filter out invalid items
            valid_items = []
            for item in items:
                if not isinstance(item, ElectronicsItem):
                    self.error_handler.log_warning(
                        "generate_grid_route",
                        f"Skipping invalid item: {type(item).__name__}"
                    )
                    continue
                
                valid, msg = self.validate_location(item.location)
                if not valid:
                    self.error_handler.log_warning(
                        "generate_grid_route",
                        f"Skipping item {item.sku}: {msg}"
                    )
                    continue
                
                valid_items.append(item)
            
            if not valid_items:
                return []
            
            # Check cache
            cache_key = hash(tuple(sorted([item.sku for item in valid_items])))
            if cache_key in self.route_cache:
                self.stats["cache_hits"] += 1
                return self.route_cache[cache_key]
            
            self.stats["cache_misses"] += 1
            
            # Group items by aisle
            aisle_groups = {}
            for item in valid_items:
                aisle = item.location.aisle
                if aisle not in aisle_groups:
                    aisle_groups[aisle] = []
                aisle_groups[aisle].append(item)
            
            # Sort aisles
            sorted_aisles = sorted(aisle_groups.keys())
            route = []
            
            for aisle in sorted_aisles:
                # Sort items in this aisle by rack number
                aisle_items = sorted(
                    aisle_groups[aisle],
                    key=lambda x: x.location.rack
                )
                
                for item in aisle_items:
                    route.append(item.location)
            
            # Cache the result
            self.route_cache[cache_key] = route
            self.stats["routes_generated"] += 1
            
            return route
            
        except Exception as e:
            self.error_handler.log_error(
                "generate_grid_route", e,
                {"item_count": len(items) if items else 0},
                severity="high"
            )
            return []
    
    def calculate_route_metrics(self, route: List[GridLocation]) -> Dict:
        """Calculate time and distance metrics for a route"""
        try:
            if not route:
                return {
                    "total_distance_m": 0,
                    "travel_time_min": 0,
                    "picking_time_min": 0,
                    "total_time_min": 0,
                    "items_count": 0,
                    "valid": True
                }
            
            if not isinstance(route, list):
                raise TypeError(f"Expected list, got {type(route).__name__}")
            
            total_distance = 0
            current = self.start_location
            
            # Calculate distance between consecutive locations
            for i, location in enumerate(route):
                valid, msg = self.validate_location(location)
                if not valid:
                    self.error_handler.log_warning(
                        "calculate_route_metrics",
                        f"Skipping invalid location at position {i}: {msg}"
                    )
                    continue
                
                distance = self.manhattan_distance(current, location)
                total_distance += distance
                current = location
            
            # Return to start
            total_distance += self.manhattan_distance(current, self.start_location)
            
            # Update stats
            self.stats["total_distance"] += total_distance
            
            # Time calculations
            travel_time = total_distance / WAREHOUSE_CONFIG["walking_speed"]
            picking_time = len(route) * WAREHOUSE_CONFIG["picking_time_per_item"]
            total_time = travel_time + picking_time
            
            return {
                "total_distance_m": round(total_distance, 2),
                "travel_time_min": round(travel_time, 2),
                "picking_time_min": round(picking_time, 2),
                "total_time_min": round(total_time, 2),
                "items_count": len(route),
                "valid": True
            }
            
        except Exception as e:
            self.error_handler.log_error(
                "calculate_route_metrics", e,
                {"route_length": len(route) if route else 0},
                severity="medium"
            )
            
            return {
                "total_distance_m": 0,
                "travel_time_min": 0,
                "picking_time_min": 0,
                "total_time_min": 0,
                "items_count": 0,
                "valid": False,
                "error": str(e)
            }
    
    def get_stats(self) -> Dict:
        """Get optimizer statistics"""
        return {
            **self.stats,
            "cache_size": len(self.route_cache)
        }


# ============================================================================
# ELECTRONICS AUTO-PICKER WITH COMPREHENSIVE ERROR HANDLING
# ============================================================================

class ElectronicsAutoPicker:
    """Auto-picker with comprehensive error handling and recovery"""
    
    def __init__(self, enable_error_handling: bool = True):
        self.error_handler = ErrorHandler() if enable_error_handling else None
        self.optimizer = GridRouteOptimizer(self.error_handler)
        
        # Core data structures
        self.orders: Dict[str, CustomerOrder] = {}
        self.inventory: Dict[str, ElectronicsItem] = {}
        self.order_counter = 1
        
        # Performance metrics
        self.metrics = {
            "start_time": datetime.now().isoformat(),
            "total_orders_created": 0,
            "total_orders_processed": 0,
            "total_items_picked": 0,
            "total_order_value": 0.0,
            "successful_picks": 0,
            "failed_picks": 0,
            "partial_picks": 0
        }
        
        # System state
        self.system_state = "initializing"
        self.last_maintenance = datetime.now()
        self.maintenance_interval = timedelta(hours=24)
        
        # Initialize inventory
        try:
            self._init_electronics_inventory()
            self.system_state = "running"
            print(f"✅ ElectronicsAutoPicker initialized successfully")
            print(f"   📦 Inventory: {len(self.inventory)} items")
            print(f"   🗺️  Grid: {WAREHOUSE_CONFIG['grid_size'][0]}×{WAREHOUSE_CONFIG['grid_size'][1]}")
        except Exception as e:
            self.system_state = "failed"
            if self.error_handler:
                self.error_handler.log_error(
                    "__init__", e, severity="critical"
                )
            raise
    
    def _init_electronics_inventory(self):
        """Initialize inventory with validation"""
        sku_counter = 1
        items_created = 0
        
        for aisle in range(1, WAREHOUSE_CONFIG["grid_size"][0] + 1):
            for rack in range(1, WAREHOUSE_CONFIG["grid_size"][1] + 1):
                try:
                    # Assign categories
                    if aisle <= 2:
                        category = "computers"
                    elif aisle <= 4:
                        category = "phones"
                    else:
                        category = "accessories" if rack <= 12 else "components"
                    
                    # Generate SKU
                    sku = f"ELEC-{sku_counter:04d}"
                    
                    # Create location
                    location = GridLocation(aisle=aisle, rack=rack)
                    
                    # Item properties based on category
                    if category == "computers":
                        name = f"Laptop Model {sku_counter}"
                        weight = 2.5
                        value = 1200.00
                        dimensions = (35, 25, 2)
                        fragile = True
                        quantity = 10 + (rack % 5)
                    elif category == "phones":
                        name = f"Smartphone {sku_counter}"
                        weight = 0.2
                        value = 800.00
                        dimensions = (15, 7, 1)
                        fragile = True
                        quantity = 25 + (rack % 10)
                    elif category == "accessories":
                        name = f"Accessory Kit {sku_counter}"
                        weight = 0.5
                        value = 50.00
                        dimensions = (20, 15, 5)
                        fragile = False
                        quantity = 100 + (rack % 20)
                    else:  # components
                        name = f"Component {sku_counter}"
                        weight = 0.1
                        value = 150.00
                        dimensions = (10, 10, 2)
                        fragile = True
                        quantity = 50 + (rack % 15)
                    
                    # Create and validate item
                    item = ElectronicsItem(
                        sku=sku,
                        name=name,
                        category=category,
                        location=location,
                        quantity=quantity,
                        weight=weight,
                        value=value,
                        dimensions=dimensions,
                        fragile=fragile
                    )
                    
                    self.inventory[sku] = item
                    items_created += 1
                    sku_counter += 1
                    
                except Exception as e:
                    if self.error_handler:
                        self.error_handler.log_warning(
                            "_init_electronics_inventory",
                            f"Failed to create item at A{aisle}-R{rack}: {e}"
                        )
                    continue
        
        # Validate inventory after creation
        self._validate_inventory_state()
        
        return items_created
    
    def _validate_inventory_state(self) -> List[Dict]:
        """Check inventory for corruption and fix issues"""
        issues = []
        
        for sku, item in self.inventory.items():
            # Check for negative quantities
            if item.quantity < 0:
                issues.append({
                    "sku": sku,
                    "issue": "negative_quantity",
                    "value": item.quantity,
                    "action": "reset_to_zero"
                })
                item.quantity = 0
            
            # Check for unrealistic quantities
            if item.quantity > 10000:
                issues.append({
                    "sku": sku,
                    "issue": "unrealistic_quantity",
                    "value": item.quantity,
                    "action": "cap_at_10000"
                })
                item.quantity = 10000
            
            # Check for missing location
            if not item.location:
                issues.append({
                    "sku": sku,
                    "issue": "missing_location",
                    "action": "assign_default"
                })
                item.location = GridLocation(aisle=1, rack=1)
        
        if issues and self.error_handler:
            self.error_handler.log_warning(
                "_validate_inventory_state",
                f"Fixed {len(issues)} inventory issues",
                {"issues": issues[:5]}
            )
        
        return issues
    
    def _validate_order_input(self, customer_id: str, customer_type: str,
                            items: List[Tuple[str, int]]) -> Tuple[bool, List[str]]:
        """Validate order input parameters"""
        errors = []
        
        # Customer ID validation
        if not customer_id or not isinstance(customer_id, str):
            errors.append("Customer ID must be a non-empty string")
        else:
            customer_id = customer_id.strip()
            if len(customer_id) < WAREHOUSE_CONFIG["min_customer_id_length"]:
                errors.append(
                    f"Customer ID must be at least "
                    f"{WAREHOUSE_CONFIG['min_customer_id_length']} characters"
                )
            if len(customer_id) > WAREHOUSE_CONFIG["max_customer_id_length"]:
                errors.append(
                    f"Customer ID too long (max {WAREHOUSE_CONFIG['max_customer_id_length']})"
                )
        
        # Customer type validation
        if customer_type not in VALID_CUSTOMER_TYPES:
            errors.append(
                f"Invalid customer type '{customer_type}'. "
                f"Must be one of: {VALID_CUSTOMER_TYPES}"
            )
        
        # Items validation
        if not items:
            errors.append("Order must contain at least one item")
        elif not isinstance(items, list):
            errors.append(f"Items must be a list, got {type(items).__name__}")
        elif len(items) > WAREHOUSE_CONFIG["max_order_items"]:
            errors.append(
                f"Too many items (max {WAREHOUSE_CONFIG['max_order_items']}, "
                f"got {len(items)})"
            )
        else:
            for idx, (sku, quantity) in enumerate(items, 1):
                item_errors = []
                
                # SKU validation
                if not isinstance(sku, str):
                    item_errors.append(f"SKU must be string, got {type(sku).__name__}")
                elif not sku:
                    item_errors.append("SKU cannot be empty")
                elif not sku.startswith("ELEC-"):
                    item_errors.append(f"SKU must start with 'ELEC-', got '{sku}'")
                elif len(sku) > 20:
                    item_errors.append(f"SKU too long (max 20 chars): {sku}")
                
                # Quantity validation
                if not isinstance(quantity, int):
                    item_errors.append(f"Quantity must be integer, got {type(quantity).__name__}")
                elif quantity <= 0:
                    item_errors.append(f"Quantity must be positive, got {quantity}")
                elif quantity > WAREHOUSE_CONFIG["max_order_quantity"]:
                    item_errors.append(
                        f"Quantity exceeds maximum ({WAREHOUSE_CONFIG['max_order_quantity']}): {quantity}"
                    )
                
                if item_errors:
                    errors.append(f"Item {idx} ({sku}): {'; '.join(item_errors)}")
        
        return len(errors) == 0, errors
    
    def create_electronics_order(self, customer_id: str, customer_type: str,
                               items: List[Tuple[str, int]]) -> str:
        """Create electronics order with full validation"""
        try:
            # Validate inputs
            is_valid, validation_errors = self._validate_order_input(
                customer_id, customer_type, items
            )
            
            if not is_valid:
                error_msg = "\n".join(validation_errors)
                raise ValueError(f"Order validation failed:\n{error_msg}")
            
            # Process items
            order_items = []
            missing_skus = []
            insufficient_stock = []
            
            for sku, requested_qty in items:
                if sku not in self.inventory:
                    missing_skus.append(sku)
                    continue
                
                inv_item = self.inventory[sku]
                available_qty = inv_item.quantity
                
                if requested_qty > available_qty:
                    insufficient_stock.append({
                        "sku": sku,
                        "requested": requested_qty,
                        "available": available_qty
                    })
                    continue
                
                order_items.append(OrderItem(sku=sku, quantity=requested_qty))
            
            # Handle missing items
            if missing_skus and self.error_handler:
                self.error_handler.log_warning(
                    "create_electronics_order",
                    f"SKUs not found: {', '.join(missing_skus)}"
                )
            
            if insufficient_stock and self.error_handler:
                self.error_handler.log_warning(
                    "create_electronics_order",
                    f"Insufficient stock for {len(insufficient_stock)} items"
                )
            
            if not order_items:
                raise ValueError("No valid items in order")
            
            # Create order
            order_id = f"ELEC-ORD-{self.order_counter:06d}"
            self.order_counter += 1
            
            # Set priority based on customer type
            priority_map = {"wholesale": 1, "repair_shop": 2, "retail": 3}
            priority = priority_map.get(customer_type, 3)
            
            order = CustomerOrder(
                order_id=order_id,
                customer_id=customer_id.strip(),
                customer_type=customer_type,
                items=order_items,
                status=OrderStatus.PENDING,
                created_at=datetime.now(),
                priority=priority
            )
            
            self.orders[order_id] = order
            self.metrics["total_orders_created"] += 1
            
            # Log warnings
            if missing_skus or insufficient_stock:
                warnings = []
                if missing_skus:
                    warnings.append(f"Skipped {len(missing_skus)} missing SKUs")
                if insufficient_stock:
                    warnings.append(f"Skipped {len(insufficient_stock)} items with insufficient stock")
                
                print(f"⚠️  Order {order_id} created with warnings: {', '.join(warnings)}")
            
            return order_id
            
        except Exception as e:
            if self.error_handler:
                self.error_handler.log_error(
                    "create_electronics_order", e,
                    {
                        "customer_id": customer_id,
                        "customer_type": customer_type,
                        "item_count": len(items) if items else 0
                    },
                    severity="high"
                )
            raise
    
    def process_grid_order(self, order_id: str, demo_mode: bool = True) -> Dict:
        """Process order with grid-optimized routing and error handling"""
        try:
            # Validate order exists
            if order_id not in self.orders:
                return self.error_handler.create_error_response(
                    "order_not_found",
                    f"Order {order_id} not found",
                    recoverable=False
                ) if self.error_handler else {
                    "success": False,
                    "error": f"Order {order_id} not found"
                }
            
            order = self.orders[order_id]
            
            # Check order status
            if order.status != OrderStatus.PENDING:
                return self.error_handler.create_error_response(
                    "invalid_status",
                    f"Order already in status: {order.status.value}",
                    {"current_status": order.status.value},
                    recoverable=False
                ) if self.error_handler else {
                    "success": False,
                    "error": f"Order already in status: {order.status.value}"
                }
            
            # Check if order has items
            if not order.items:
                return self.error_handler.create_error_response(
                    "empty_order",
                    "Order has no items",
                    recoverable=False
                ) if self.error_handler else {
                    "success": False,
                    "error": "Order has no items"
                }
            
            order.status = OrderStatus.PROCESSING
            
            # Get inventory items for this order
            order_items = []
            unavailable_items = []
            
            for order_item in order.items:
                if order_item.sku not in self.inventory:
                    unavailable_items.append({
                        "sku": order_item.sku,
                        "reason": "not_found"
                    })
                    continue
                
                inv_item = self.inventory[order_item.sku]
                
                # Cap quantity for demo mode
                pick_qty = order_item.quantity
                if demo_mode:
                    pick_qty = min(pick_qty, WAREHOUSE_CONFIG["demo_item_cap"])
                    if pick_qty < order_item.quantity:
                        print(f"   ⚠️  Demo mode: Capping {order_item.sku} from {order_item.quantity} to {pick_qty}")
                
                for _ in range(pick_qty):
                    order_items.append(inv_item)
            
            if not order_items:
                return self.error_handler.create_error_response(
                    "no_inventory",
                    "No items available to pick",
                    {"unavailable": unavailable_items},
                    recoverable=True,
                    suggestion="Check inventory levels"
                ) if self.error_handler else {
                    "success": False,
                    "error": "No items available to pick"
                }
            
            # Generate route
            route = self.optimizer.generate_grid_route(order_items)
            metrics = self.optimizer.calculate_route_metrics(route)
            
            self.metrics["total_orders_processed"] += 1
            
            response = {
                "success": True,
                "order_id": order_id,
                "status": "route_generated",
                "route_locations": [str(loc) for loc in route],
                "metrics": metrics,
                "total_value": self._calculate_order_value(order),
                "demo_mode": demo_mode
            }
            
            if unavailable_items:
                response["warnings"] = {
                    "unavailable_items": unavailable_items
                }
            
            return response
            
        except Exception as e:
            if self.error_handler:
                self.error_handler.log_error(
                    "process_grid_order", e,
                    {"order_id": order_id},
                    severity="high"
                )
            
            return self.error_handler.create_error_response(
                "system_error",
                f"Failed to process order: {str(e)}",
                recoverable=False
            ) if self.error_handler else {
                "success": False,
                "error": f"System error: {str(e)}"
            }
    
    def execute_grid_picking(self, order_id: str, picker_id: str = "PICKER-01") -> Dict:
        """Execute picking with comprehensive error handling"""
        try:
            # Validate inputs
            if not order_id or not isinstance(order_id, str):
                raise ValueError("Invalid order ID")
            
            if not picker_id or not isinstance(picker_id, str):
                raise ValueError("Invalid picker ID")
            
            # Validate order exists
            if order_id not in self.orders:
                return self.error_handler.create_error_response(
                    "order_not_found",
                    f"Order {order_id} not found",
                    recoverable=False
                ) if self.error_handler else {
                    "success": False,
                    "error": f"Order {order_id} not found"
                }
            
            order = self.orders[order_id]
            
            # Check order can be picked
            if order.status not in [OrderStatus.PROCESSING, OrderStatus.PENDING]:
                return self.error_handler.create_error_response(
                    "invalid_status",
                    f"Cannot pick order in status: {order.status.value}",
                    {"current_status": order.status.value},
                    recoverable=False
                ) if self.error_handler else {
                    "success": False,
                    "error": f"Cannot pick order in status: {order.status.value}"
                }
            
            # Validate inventory state before picking
            inventory_issues = self._validate_inventory_state()
            if inventory_issues and self.error_handler:
                self.error_handler.log_warning(
                    "execute_grid_picking",
                    f"Found {len(inventory_issues)} inventory issues before picking"
                )
            
            start_time = datetime.now()
            order.status = OrderStatus.PICKING
            order.picker_id = picker_id
            
            print(f"\n{'='*60}")
            print(f"🚀 EXECUTING GRID-OPTIMIZED PICKING")
            print(f"{'='*60}")
            print(f"Order: {order_id}")
            print(f"Picker: {picker_id}")
            print(f"Start: {start_time.strftime('%H:%M:%S')}")
            print(f"{'='*60}")
            
            picked_items = []
            failed_items = []
            
            for order_item in order.items:
                try:
                    if order_item.sku not in self.inventory:
                        print(f"   ❌ SKU {order_item.sku} not in inventory")
                        order_item.status = "failed"
                        failed_items.append({
                            "sku": order_item.sku,
                            "reason": "not_found"
                        })
                        self.metrics["failed_picks"] += 1
                        continue
                    
                    inv_item = self.inventory[order_item.sku]
                    
                    print(f"\n   📍 Location: {inv_item.location}")
                    print(f"      Item: {inv_item.name}")
                    print(f"      Category: {inv_item.category}")
                    print(f"      Requested: {order_item.quantity}")
                    print(f"      Available: {inv_item.quantity}")
                    
                    if inv_item.quantity >= order_item.quantity:
                        # Successful pick
                        pick_qty = order_item.quantity
                        inv_item.quantity -= pick_qty
                        order_item.picked = pick_qty
                        order_item.status = "picked"
                        
                        print(f"      ✅ Picked: {pick_qty}")
                        print(f"      Remaining: {inv_item.quantity}")
                        
                        if inv_item.fragile:
                            print(f"      ⚠️  Fragile item - Handle with care")
                        
                        picked_items.append({
                            "sku": inv_item.sku,
                            "quantity": pick_qty,
                            "location": str(inv_item.location),
                            "value": inv_item.value * pick_qty,
                            "fragile": inv_item.fragile
                        })
                        
                        self.metrics["successful_picks"] += 1
                        self.metrics["total_items_picked"] += pick_qty
                        self.metrics["total_order_value"] += inv_item.value * pick_qty
                        
                    else:
                        # Partial or failed pick
                        available = inv_item.quantity
                        if available > 0:
                            # Partial fulfillment
                            pick_qty = available
                            inv_item.quantity = 0
                            order_item.picked = pick_qty
                            order_item.status = "partial"
                            
                            print(f"      ⚠️  Partial: {pick_qty}/{order_item.quantity}")
                            
                            picked_items.append({
                                "sku": inv_item.sku,
                                "quantity": pick_qty,
                                "location": str(inv_item.location),
                                "value": inv_item.value * pick_qty,
                                "partial": True
                            })
                            
                            self.metrics["partial_picks"] += 1
                            self.metrics["total_items_picked"] += pick_qty
                            self.metrics["total_order_value"] += inv_item.value * pick_qty
                            
                            failed_items.append({
                                "sku": inv_item.sku,
                                "reason": "insufficient_stock",
                                "requested": order_item.quantity,
                                "available": available,
                                "picked": pick_qty
                            })
                        else:
                            # Complete failure
                            print(f"      ❌ Out of stock")
                            order_item.status = "failed"
                            failed_items.append({
                                "sku": inv_item.sku,
                                "reason": "out_of_stock",
                                "requested": order_item.quantity
                            })
                            self.metrics["failed_picks"] += 1
                
                except Exception as e:
                    error_msg = f"Error processing {order_item.sku}: {str(e)}"
                    print(f"      💥 {error_msg}")
                    failed_items.append({
                        "sku": order_item.sku,
                        "reason": "system_error",
                        "error": str(e)
                    })
                    self.metrics["failed_picks"] += 1
                    
                    if self.error_handler:
                        self.error_handler.log_error(
                            "execute_grid_picking", e,
                            {"order_id": order_id, "sku": order_item.sku},
                            severity="medium"
                        )
            
            # Update order status
            fulfillment_rate = order.fulfillment_rate()
            
            if fulfillment_rate == 100:
                order.status = OrderStatus.COMPLETE
                final_status = "COMPLETE"
            elif fulfillment_rate > 0:
                order.status = OrderStatus.PARTIAL
                final_status = "PARTIALLY_FULFILLED"
            else:
                order.status = OrderStatus.FAILED
                final_status = "FAILED"
            
            order.completed_at = datetime.now()
            end_time = order.completed_at
            duration = (end_time - start_time).total_seconds() / 60
            
            # Validate inventory after picking
            self._validate_inventory_state()
            
            # Print summary
            print(f"\n{'='*60}")
            print(f"📊 PICKING COMPLETE - SUMMARY")
            print(f"{'='*60}")
            print(f"Order: {order_id}")
            print(f"Status: {final_status}")
            print(f"Fulfillment: {fulfillment_rate:.1f}%")
            print(f"Duration: {duration:.1f} minutes")
            print(f"Items Picked: {len(picked_items)}")
            print(f"Items Failed: {len(failed_items)}")
            
            if failed_items:
                print(f"\n   Failed Items Summary:")
                failure_counts = {}
                for item in failed_items:
                    reason = item["reason"]
                    failure_counts[reason] = failure_counts.get(reason, 0) + 1
                
                for reason, count in failure_counts.items():
                    print(f"      • {reason}: {count}")
            
            total_value = sum(item.get('value', 0) for item in picked_items)
            print(f"\n   Total Order Value: ${total_value:,.2f} AUD")
            
            return {
                "success": True,
                "order_id": order_id,
                "status": final_status,
                "fulfillment_rate": fulfillment_rate,
                "picked_items": len(picked_items),
                "failed_items": len(failed_items),
                "failure_breakdown": failure_counts if failed_items else None,
                "total_value": total_value,
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "duration_minutes": round(duration, 2),
                "grid_efficiency": "optimized"
            }
            
        except Exception as e:
            if self.error_handler:
                self.error_handler.log_error(
                    "execute_grid_picking", e,
                    {"order_id": order_id, "picker_id": picker_id},
                    severity="critical"
                )
            
            return self.error_handler.create_error_response(
                "system_error",
                f"Critical failure during picking: {str(e)}",
                {"order_id": order_id},
                recoverable=False
            ) if self.error_handler else {
                "success": False,
                "error": f"System error: {str(e)}"
            }
    
    def batch_process_orders(self, order_ids: List[str]) -> Dict:
        """Process multiple orders in batch with error handling"""
        try:
            if not order_ids:
                return {
                    "success": False,
                    "error": "No order IDs provided"
                }
            
            if len(order_ids) > 20:
                return {
                    "success": False,
                    "error": f"Too many orders (max 20, got {len(order_ids)})"
                }
            
            print(f"\n📦 BATCH PROCESSING {len(order_ids)} ORDERS")
            print(f"{'='*60}")
            
            all_items = []
            valid_orders = []
            invalid_orders = []
            
            # Collect items from valid orders
            for order_id in order_ids:
                if order_id not in self.orders:
                    invalid_orders.append(order_id)
                    continue
                
                order = self.orders[order_id]
                if order.status != OrderStatus.PENDING:
                    invalid_orders.append(f"{order_id} (status: {order.status.value})")
                    continue
                
                for order_item in order.items:
                    if order_item.sku in self.inventory:
                        inv_item = self.inventory[order_item.sku]
                        all_items.append({
                            "order_id": order_id,
                            "item": inv_item,
                            "quantity": order_item.quantity
                        })
                
                valid_orders.append(order_id)
            
            if invalid_orders:
                print(f"⚠️  Skipping invalid orders: {len(invalid_orders)}")
            
            if not all_items:
                return {
                    "success": False,
                    "error": "No valid items found in any orders",
                    "invalid_orders": invalid_orders
                }
            
            # Group by location
            location_groups = {}
            for item_data in all_items:
                loc_str = str(item_data["item"].location)
                if loc_str not in location_groups:
                    location_groups[loc_str] = []
                location_groups[loc_str].append(item_data)
            
            unique_locations = list(location_groups.keys())
            
            # Calculate efficiency
            single_order_trips = len(valid_orders) * len(unique_locations)  # Rough estimate
            batch_trip = len(unique_locations)
            efficiency_gain = ((single_order_trips - batch_trip) / max(single_order_trips, 1) * 100)
            
            print(f"\n   Batch Optimization Results:")
            print(f"      Valid orders: {len(valid_orders)}")
            print(f"      Unique locations: {len(unique_locations)}")
            print(f"      Total items: {len(all_items)}")
            print(f"      Efficiency gain: {efficiency_gain:.1f}%")
            
            return {
                "success": True,
                "batch_size": len(valid_orders),
                "unique_locations": len(unique_locations),
                "total_items": len(all_items),
                "estimated_single_trips": single_order_trips,
                "estimated_batch_trips": batch_trip,
                "efficiency_gain": round(efficiency_gain, 1),
                "locations": unique_locations[:10],  # First 10 only
                "invalid_orders": invalid_orders if invalid_orders else None
            }
            
        except Exception as e:
            if self.error_handler:
                self.error_handler.log_error(
                    "batch_process_orders", e,
                    {"order_count": len(order_ids) if order_ids else 0},
                    severity="high"
                )
            
            return {
                "success": False,
                "error": f"Batch processing failed: {str(e)}"
            }
    
    def get_grid_visualization(self) -> Dict:
        """Generate grid visualization data with error handling"""
        try:
            grid = np.zeros(WAREHOUSE_CONFIG["grid_size"], dtype=int)
            
            # Fill grid with inventory counts
            for item in self.inventory.values():
                try:
                    if 1 <= item.location.aisle <= WAREHOUSE_CONFIG["grid_size"][0] and \
                       1 <= item.location.rack <= WAREHOUSE_CONFIG["grid_size"][1]:
                        grid[item.location.aisle-1, item.location.rack-1] = item.quantity
                except Exception:
                    continue
            
            # Find low stock locations
            low_stock_locations = []
            for aisle in range(WAREHOUSE_CONFIG["grid_size"][0]):
                for rack in range(WAREHOUSE_CONFIG["grid_size"][1]):
                    if grid[aisle, rack] < WAREHOUSE_CONFIG["low_stock_threshold"]:
                        low_stock_locations.append(f"A{aisle+1:02d}-R{rack+1:03d}")
            
            # Calculate statistics
            occupied_cells = np.count_nonzero(grid)
            total_items = int(np.sum(grid))
            
            # Get top 20 low stock locations
            low_stock_locations = low_stock_locations[:20]
            
            return {
                "grid_size": WAREHOUSE_CONFIG["grid_size"],
                "total_cells": WAREHOUSE_CONFIG["grid_size"][0] * WAREHOUSE_CONFIG["grid_size"][1],
                "occupied_cells": occupied_cells,
                "total_items": total_items,
                "average_stock_per_cell": round(np.mean(grid[grid > 0]), 1) if occupied_cells > 0 else 0,
                "low_stock_locations": low_stock_locations,
                "grid_sample": grid.tolist()[:3] if grid.size > 0 else []
            }
            
        except Exception as e:
            if self.error_handler:
                self.error_handler.log_error(
                    "get_grid_visualization", e,
                    severity="low"
                )
            
            return {
                "grid_size": WAREHOUSE_CONFIG["grid_size"],
                "total_cells": WAREHOUSE_CONFIG["grid_size"][0] * WAREHOUSE_CONFIG["grid_size"][1],
                "occupied_cells": 0,
                "total_items": 0,
                "average_stock_per_cell": 0,
                "low_stock_locations": [],
                "grid_sample": [],
                "error": "Visualization generation failed"
            }
    
    def _calculate_order_value(self, order: CustomerOrder) -> float:
        """Calculate total value of an order"""
        try:
            total = 0.0
            for order_item in order.items:
                if order_item.sku in self.inventory:
                    total += self.inventory[order_item.sku].value * order_item.quantity
            return round(total, 2)
        except Exception:
            return 0.0
    
    def save_grid_report(self) -> Optional[str]:
        """Save comprehensive grid warehouse report with error handling"""
        try:
            report = {
                "timestamp": datetime.now().isoformat(),
                "warehouse_config": WAREHOUSE_CONFIG,
                "system_state": self.system_state,
                "inventory_summary": {
                    "total_items": len(self.inventory),
                    "total_quantity": sum(item.quantity for item in self.inventory.values()),
                    "total_value": round(
                        sum(item.quantity * item.value for item in self.inventory.values()), 2
                    ),
                    "by_category": self._get_inventory_by_category()
                },
                "order_summary": self._get_order_summary(),
                "grid_analysis": self.get_grid_visualization(),
                "performance_metrics": self.metrics,
                "optimizer_stats": self.optimizer.get_stats(),
                "error_stats": self.error_handler.get_error_report() if self.error_handler else None
            }
            
            # Generate filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"grid_warehouse_report_{timestamp}.json"
            temp_filename = f"{filename}.tmp"
            
            # Write to temp file first
            with open(temp_filename, "w", encoding='utf-8') as f:
                json.dump(report, f, indent=2, default=str)
            
            # Rename to final filename
            if os.path.exists(filename):
                os.remove(filename)
            os.rename(temp_filename, filename)
            
            print(f"\n✅ Grid warehouse report saved to: {filename}")
            return filename
            
        except PermissionError:
            print(f"\n❌ Permission denied: Cannot write to {filename}")
            return None
        except json.JSONEncodeError as e:
            print(f"\n❌ JSON encoding error: {e}")
            return None
        except Exception as e:
            print(f"\n❌ Failed to save report: {e}")
            if self.error_handler:
                self.error_handler.log_error(
                    "save_grid_report", e,
                    severity="high"
                )
            return None
    
    def _get_inventory_by_category(self) -> Dict:
        """Get inventory breakdown by category"""
        by_category = {}
        
        for item in self.inventory.values():
            cat = item.category
            if cat not in by_category:
                by_category[cat] = {
                    "count": 0,
                    "quantity": 0,
                    "value": 0.0
                }
            
            by_category[cat]["count"] += 1
            by_category[cat]["quantity"] += item.quantity
            by_category[cat]["value"] += item.quantity * item.value
        
        # Round values
        for cat in by_category:
            by_category[cat]["value"] = round(by_category[cat]["value"], 2)
        
        return by_category
    
    def _get_order_summary(self) -> Dict:
        """Get order summary statistics"""
        summary = {
            "total_orders": len(self.orders),
            "by_status": {},
            "by_customer_type": {},
            "total_value": 0.0
        }
        
        # Count by status
        for status in OrderStatus:
            count = sum(1 for o in self.orders.values() if o.status == status)
            if count > 0:
                summary["by_status"][status.value] = count
        
        # Count by customer type
        for ct in VALID_CUSTOMER_TYPES:
            count = sum(1 for o in self.orders.values() if o.customer_type == ct)
            if count > 0:
                summary["by_customer_type"][ct] = count
        
        # Calculate total value
        summary["total_value"] = round(
            sum(self._calculate_order_value(o) for o in self.orders.values()), 2
        )
        
        return summary
    
    def get_system_health(self) -> Dict:
        """Get comprehensive system health report"""
        # Check if maintenance is needed
        time_since_maintenance = datetime.now() - self.last_maintenance
        maintenance_needed = time_since_maintenance > self.maintenance_interval
        
        # Validate inventory
        inventory_issues = self._validate_inventory_state()
        
        # Calculate success rate
        total_ops = self.metrics["successful_picks"] + self.metrics["failed_picks"]
        success_rate = (
            (self.metrics["successful_picks"] / max(total_ops, 1)) * 100
        )
        
        # Determine system status
        if len(inventory_issues) > 10:
            status = "critical"
        elif len(inventory_issues) > 0 or maintenance_needed:
            status = "degraded"
        else:
            status = "healthy"
        
        return {
            "status": status,
            "uptime": (datetime.now() - datetime.fromisoformat(self.metrics["start_time"])).total_seconds() / 3600,
            "maintenance_needed": maintenance_needed,
            "last_maintenance": self.last_maintenance.isoformat(),
            "performance": {
                "success_rate": round(success_rate, 1),
                "total_items_picked": self.metrics["total_items_picked"],
                "total_orders_processed": self.metrics["total_orders_processed"]
            },
            "inventory_health": {
                "issues_count": len(inventory_issues),
                "issues": inventory_issues[:5]  # First 5 issues
            },
            "error_stats": self.error_handler.get_error_report() if self.error_handler else None,
            "optimizer_stats": self.optimizer.get_stats(),
            "recommendations": self._generate_recommendations(inventory_issues, maintenance_needed)
        }
    
    def _generate_recommendations(self, inventory_issues: List[Dict], 
                                 maintenance_needed: bool) -> List[str]:
        """Generate system recommendations based on health"""
        recommendations = []
        
        if maintenance_needed:
            recommendations.append("Schedule system maintenance - 24 hours since last check")
        
        if len(inventory_issues) > 0:
            recommendations.append(f"Review {len(inventory_issues)} inventory issues")
        
        # Check low stock
        low_stock_count = sum(1 for i in self.inventory.values() 
                            if i.quantity < WAREHOUSE_CONFIG["low_stock_threshold"])
        if low_stock_count > 10:
            recommendations.append(f"Restock items - {low_stock_count} SKUs below threshold")
        
        # Check performance
        if self.metrics["failed_picks"] > self.metrics["successful_picks"]:
            recommendations.append("High failure rate - Review inventory levels")
        
        if not recommendations:
            recommendations.append("System operating within normal parameters")
        
        return recommendations
    
    def perform_maintenance(self) -> Dict:
        """Perform system maintenance"""
        self.last_maintenance = datetime.now()
        
        # Validate and fix inventory
        inventory_issues = self._validate_inventory_state()
        
        # Clean up old error logs
        if self.error_handler:
            self.error_handler.cleanup_old_logs()
        
        # Clear route cache
        self.optimizer.route_cache.clear()
        
        return {
            "timestamp": self.last_maintenance.isoformat(),
            "inventory_issues_fixed": len(inventory_issues),
            "cache_cleared": True,
            "logs_cleaned": True
        }


# ============================================================================
# DEMONSTRATION FUNCTIONS
# ============================================================================

def run_grid_demo():
    """Run demonstration of grid-optimized auto-picker"""
    print("=" * 80)
    print("🚀 GRID-OPTIMIZED AUTO-PICKER DEMONSTRATION")
    print("   ICT304 - Warehouse Intelligence System")
    print(f"   Warehouse: {WAREHOUSE_CONFIG['grid_size'][0]}×{WAREHOUSE_CONFIG['grid_size'][1]} grid")
    print("=" * 80)
    
    try:
        # Initialize picker
        print("\n1. 🏭 Initializing Electronics Warehouse...")
        picker = ElectronicsAutoPicker(enable_error_handling=True)
        
        # Show grid visualization
        print("\n2. 📊 Grid Warehouse Visualization...")
        grid_data = picker.get_grid_visualization()
        print(f"   • Grid Size: {grid_data['grid_size'][0]}×{grid_data['grid_size'][1]}")
        print(f"   • Total Items: {grid_data['total_items']:,}")
        print(f"   • Occupied Cells: {grid_data['occupied_cells']}")
        print(f"   • Low Stock Locations: {len(grid_data['low_stock_locations'])}")
        
        # Create sample orders
        print("\n3. 📝 Creating Electronics Orders...")
        
        # Wholesale order (high priority)
        wholesale_order = picker.create_electronics_order(
            "WHOLESALE-001",
            "wholesale",
            [
                ("ELEC-0010", 5),   # Computers
                ("ELEC-0050", 10),  # Phones
                ("ELEC-0100", 20),  # Accessories
            ]
        )
        print(f"   • Wholesale Order: {wholesale_order} (Priority: 1)")
        
        # Retail orders
        retail_orders = []
        for i in range(3):
            try:
                order_id = picker.create_electronics_order(
                    f"RETAIL-{i+1:03d}",
                    "retail",
                    [
                        (f"ELEC-{(i*50)+1:04d}", 2),
                        (f"ELEC-{(i*50)+25:04d}", 1),
                    ]
                )
                retail_orders.append(order_id)
                print(f"   • Retail Order {i+1}: {order_id} (Priority: 3)")
            except Exception as e:
                print(f"   ⚠️  Failed to create retail order {i+1}: {e}")
        
        # Repair shop order
        repair_order = picker.create_electronics_order(
            "REPAIR-001",
            "repair_shop",
            [
                ("ELEC-0001", 1),   # Specific component
                ("ELEC-0050", 2),   # Phone for parts
                ("ELEC-0120", 5),   # Accessories
            ]
        )
        print(f"   • Repair Shop Order: {repair_order} (Priority: 2)")
        
        # Process wholesale order
        print(f"\n4. 🗺️  Processing Wholesale Order with Grid Optimization...")
        wholesale_result = picker.process_grid_order(wholesale_order, demo_mode=True)
        
        if wholesale_result.get("success", False):
            print(f"   • Order: {wholesale_result['order_id']}")
            print(f"   • Route Locations: {len(wholesale_result['route_locations'])}")
            print(f"   • Total Distance: {wholesale_result['metrics']['total_distance_m']}m")
            print(f"   • Estimated Time: {wholesale_result['metrics']['total_time_min']}min")
            print(f"   • Order Value: ${wholesale_result['total_value']:,.2f}")
            
            # Execute picking
            print(f"\n5. 🤖 Executing Grid-Optimized Picking...")
            execution_result = picker.execute_grid_picking(wholesale_order)
            
            if execution_result.get("success", False):
                print(f"   • Status: {execution_result['status']}")
                print(f"   • Fulfillment: {execution_result['fulfillment_rate']:.1f}%")
                print(f"   • Duration: {execution_result['duration_minutes']} minutes")
        
        # Demonstrate batch processing
        print(f"\n6. 📦 Demonstrating Batch Processing...")
        batch_result = picker.batch_process_orders(retail_orders + [repair_order])
        
        if batch_result.get("success", False):
            print(f"   • Batch Size: {batch_result['batch_size']} orders")
            print(f"   • Unique Locations: {batch_result['unique_locations']}")
            print(f"   • Efficiency Gain: {batch_result['efficiency_gain']}%")
            print(f"   • Sample Locations: {', '.join(batch_result['locations'][:3])}...")
        
        # Save report
        print(f"\n7. 💾 Saving Warehouse Report...")
        report_file = picker.save_grid_report()
        
        if report_file:
            print(f"   ✅ Report saved successfully")
        
        # Show system health
        print(f"\n8. 🏥 System Health Check...")
        health = picker.get_system_health()
        print(f"   • Status: {health['status'].upper()}")
        print(f"   • Success Rate: {health['performance']['success_rate']}%")
        print(f"   • Items Picked: {health['performance']['total_items_picked']}")
        
        print(f"\n{'='*80}")
        print(f"✅ GRID-OPTIMIZED DEMONSTRATION COMPLETE")
        print(f"{'='*80}")
        
        return picker, wholesale_order
        
    except Exception as e:
        print(f"\n❌ Demonstration failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def demonstrate_error_handling():
    """Demonstrate error handling capabilities"""
    print("\n" + "=" * 80)
    print("🛡️  ERROR HANDLING DEMONSTRATION")
    print("=" * 80)
    
    picker = ElectronicsAutoPicker(enable_error_handling=True)
    
    # Test 1: Invalid order (empty customer ID)
    print("\n1. Testing: Empty Customer ID")
    try:
        order_id = picker.create_electronics_order(
            "", "retail", [("ELEC-0001", 1)]
        )
    except ValueError as e:
        print(f"   ✅ Caught expected error: {e}")
    
    # Test 2: Invalid order (negative quantity)
    print("\n2. Testing: Negative Quantity")
    try:
        order_id = picker.create_electronics_order(
            "CUST-001", "retail", [("ELEC-0001", -5)]
        )
    except ValueError as e:
        print(f"   ✅ Caught expected error: {e}")
    
    # Test 3: Non-existent SKU
    print("\n3. Testing: Non-existent SKU")
    try:
        order_id = picker.create_electronics_order(
            "CUST-001", "retail", [("ELEC-9999", 1)]
        )
        print(f"   ✅ Created order with warning (missing SKU): {order_id}")
    except Exception as e:
        print(f"   ❌ Unexpected error: {e}")
    
    # Test 4: Process non-existent order
    print("\n4. Testing: Process Non-existent Order")
    result = picker.process_grid_order("INVALID-ORDER")
    print(f"   ✅ Got error response: {result.get('error', {}).get('message', 'Unknown error')}")
    
    # Test 5: Inventory corruption recovery
    print("\n5. Testing: Inventory Corruption Recovery")
    if "ELEC-0001" in picker.inventory:
        picker.inventory["ELEC-0001"].quantity = -10
        issues = picker._validate_inventory_state()
        print(f"   ✅ Fixed {len(issues)} inventory issues")
        print(f"   • Quantity now: {picker.inventory['ELEC-0001'].quantity}")
    
    # Show error report
    print("\n6. 📊 Error Handler Report:")
    if picker.error_handler:
        report = picker.error_handler.get_error_report()
        print(f"   • Total Errors: {report['total_errors']}")
        print(f"   • Recovery Rate: {report['recovery_rate']:.1f}%")
        print(f"   • System Status: {report['system_status']}")
    
    print("\n✅ Error handling demonstration complete")
    return picker


def main():
    """Main entry point"""
    import sys
    
    print("\n" + "=" * 80)
    print("ELECTRONICS WAREHOUSE AUTO-PICKER")
    print("ICT304 Assignment - Warehouse Intelligence System")
    print("Optimized for single-layer 6×24 grid configuration")
    print("=" * 80)
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--demo":
            picker, order_id = run_grid_demo()
        elif sys.argv[1] == "--error-test":
            picker = demonstrate_error_handling()
        elif sys.argv[1] == "--quick-test":
            picker = ElectronicsAutoPicker()
            order_id = picker.create_electronics_order(
                "TEST-001",
                "retail",
                [("ELEC-0001", 2), ("ELEC-0050", 1)]
            )
            result = picker.process_grid_order(order_id)
            print(json.dumps(result, indent=2))
        elif sys.argv[1] == "--health":
            picker = ElectronicsAutoPicker()
            health = picker.get_system_health()
            print(json.dumps(health, indent=2))
        else:
            print("\nUsage:")
            print("  python auto_picker.py --demo        # Run full demonstration")
            print("  python auto_picker.py --error-test  # Test error handling")
            print("  python auto_picker.py --quick-test  # Quick system test")
            print("  python auto_picker.py --health      # Show system health")
    else:
        # Run full demo by default
        picker, order_id = run_grid_demo()
        
        # Ask about error handling demo
        response = input("\nShow error handling demonstration? (y/n): ")
        if response.lower() == 'y':
            demonstrate_error_handling()
    
    print("\n📁 Generated files:")
    print("   • grid_warehouse_report_*.json")
    print("   • auto_picker_errors.log")
    print("\n✅ Done!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Operation cancelled by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        traceback.print_exc()
        sys.exit(1)
