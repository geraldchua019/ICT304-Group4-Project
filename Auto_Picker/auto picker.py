#!/usr/bin/env python3
"""
AUTO-PICKER 
For ICT304 Assignment - Warehouse Intelligence System

Assumptions:
- Electronics items only
- Single-layer grid layout (6×24 configuration)
- 2D X-Y arrangement, no vertical stacking
- Known SKUs with fixed locations
- Top-down camera for item counting
"""

import math
import json
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional
from enum import Enum
import numpy as np

# ========== WAREHOUSE CONFIGURATION ==========
WAREHOUSE_CONFIG = {
    "grid_size": (6, 24),  # 6 aisles × 24 racks per aisle
    "aisle_width": 3.0,    # meters between aisles
    "rack_depth": 1.2,     # meters per rack
    "shelf_height": 0.0,   # Single layer = all items at same height
    "camera_positions": [(0, 0), (12, 0), (24, 0)]  # Camera coordinates for monitoring
}

# ========== OPTIMIZED DATA MODELS ==========
class OrderStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    PICKING = "picking"
    PARTIAL = "partially_fulfilled"
    COMPLETE = "complete"
    FAILED = "failed"

@dataclass
class GridLocation:
    """Optimized for single-layer grid warehouse"""
    aisle: int          # 1-6 (from WAREHOUSE_CONFIG)
    rack: int           # 1-24 (from WAREHOUSE_CONFIG)
    x: float = 0.0      # Auto-calculated from grid
    y: float = 0.0      # Auto-calculated from grid
    
    def __post_init__(self):
        """Calculate X-Y coordinates from grid position"""
        # Calculate X coordinate: aisle * aisle_width
        self.x = (self.aisle - 1) * WAREHOUSE_CONFIG["aisle_width"]
        # Calculate Y coordinate: rack * rack_depth
        self.y = (self.rack - 1) * WAREHOUSE_CONFIG["rack_depth"]
    
    def __str__(self):
        return f"A{self.aisle:02d}-R{self.rack:03d}"
    
    def distance_to(self, other: 'GridLocation') -> float:
        """Manhattan distance for grid movement (aisles are separate)"""
        if self.aisle == other.aisle:
            # Same aisle: only horizontal movement
            return abs(self.y - other.y)
        else:
            # Different aisles: move to aisle end + cross aisle + move to location
            aisle_end_distance = min(self.y, WAREHOUSE_CONFIG["rack_depth"] * 12 - self.y)
            other_aisle_end = min(other.y, WAREHOUSE_CONFIG["rack_depth"] * 12 - other.y)
            aisle_cross = abs(self.aisle - other.aisle) * WAREHOUSE_CONFIG["aisle_width"]
            return aisle_end_distance + aisle_cross + other_aisle_end + abs(self.y - other.y)

@dataclass
class ElectronicsItem:
    """Optimized for electronics warehouse items"""
    sku: str
    name: str
    category: str  # "computers", "phones", "accessories", "components"
    location: GridLocation
    quantity: int
    weight: float          # kg, for picking priority
    value: float           # AUD, for security/priority
    dimensions: Tuple      # (length, width, height) in cm
    fragile: bool = False  # Needs careful handling
    
    def __str__(self):
        return f"{self.sku}: {self.name} ({self.quantity} units)"

@dataclass
class OrderItem:
    sku: str
    quantity: int
    picked: int = 0
    status: str = "pending"  # pending, picking, picked, failed

@dataclass
class CustomerOrder:
    order_id: str
    customer_id: str
    customer_type: str  # "retail", "wholesale", "repair_shop"
    items: List[OrderItem]
    status: OrderStatus
    created_at: datetime
    priority: int = 3  # 1=High (wholesale), 2=Medium, 3=Low (retail)
    
    def fulfillment_rate(self) -> float:
        """Calculate order fulfillment percentage"""
        total_requested = sum(item.quantity for item in self.items)
        total_picked = sum(item.picked for item in self.items)
        return (total_picked / total_requested * 100) if total_requested > 0 else 0.0

# ========== GRID-BASED ROUTE OPTIMIZER ==========
class GridRouteOptimizer:
    """Optimized for single-layer grid warehouse layout"""
    
    def __init__(self):
        self.start_location = GridLocation(aisle=1, rack=1)  # Entry point
        
    def manhattan_distance(self, loc1: GridLocation, loc2: GridLocation) -> float:
        """Optimized Manhattan distance for grid movement"""
        return loc1.distance_to(loc2)
    
    def generate_grid_route(self, items: List[ElectronicsItem]) -> List[GridLocation]:
        """
        Generate optimal picking route for single-layer grid.
        Strategy: Process by aisle, then by rack number within aisle.
        """
        if not items:
            return []
        
        # Group items by aisle
        aisle_groups = {}
        for item in items:
            aisle = item.location.aisle
            if aisle not in aisle_groups:
                aisle_groups[aisle] = []
            aisle_groups[aisle].append(item)
        
        # Sort aisles
        sorted_aisles = sorted(aisle_groups.keys())
        route = []
        current_location = self.start_location
        
        for aisle in sorted_aisles:
            # Sort items in this aisle by rack number
            aisle_items = sorted(aisle_groups[aisle], 
                               key=lambda x: x.location.rack)
            
            for item in aisle_items:
                route.append(item.location)
                current_location = item.location
        
        return route
    
    def calculate_route_metrics(self, route: List[GridLocation]) -> Dict:
        """Calculate time and distance for the route"""
        if len(route) < 2:
            return {"distance": 0, "time": 0, "pick_count": 0}
        
        total_distance = 0
        current = self.start_location
        
        for location in route:
            total_distance += self.manhattan_distance(current, location)
            current = location
        
        # Return to start if needed
        total_distance += self.manhattan_distance(current, self.start_location)
        
        # Time calculation (meters/minute at walking speed)
        walking_speed = 80  # meters per minute
        picking_time_per_item = 0.5  # minutes per item (electronics are quick)
        
        travel_time = total_distance / walking_speed
        picking_time = len(route) * picking_time_per_item
        total_time = travel_time + picking_time
        
        return {
            "total_distance_m": round(total_distance, 2),
            "travel_time_min": round(travel_time, 2),
            "picking_time_min": round(picking_time, 2),
            "total_time_min": round(total_time, 2),
            "items_count": len(route)
        }

# ========== ELECTRONICS-SPECIFIC AUTO-PICKER ==========
class ElectronicsAutoPicker:
    """Auto-picker optimized for electronics warehouse"""
    
    def __init__(self):
        self.orders: Dict[str, CustomerOrder] = {}
        self.inventory: Dict[str, ElectronicsItem] = {}
        self.optimizer = GridRouteOptimizer()
        self.order_counter = 1
        
        # Initialize with electronics-specific inventory
        self._init_electronics_inventory()
    
    def _init_electronics_inventory(self):
        """Initialize with realistic electronics inventory for 6×24 grid"""
        
        # Common electronics categories
        categories = {
            "computers": ["laptops", "desktops", "tablets"],
            "phones": ["smartphones", "feature phones", "smartwatches"],
            "accessories": ["cables", "chargers", "cases", "headphones"],
            "components": ["memory", "storage", "processors", "motherboards"]
        }
        
        # Populate grid systematically
        sku_counter = 1
        
        for aisle in range(1, WAREHOUSE_CONFIG["grid_size"][0] + 1):
            for rack in range(1, WAREHOUSE_CONFIG["grid_size"][1] + 1):
                # Assign categories to different sections of warehouse
                if aisle <= 2:
                    category = "computers"
                elif aisle <= 4:
                    category = "phones"
                else:
                    category = "accessories" if rack <= 12 else "components"
                
                # Generate item
                sku = f"ELEC-{sku_counter:04d}"
                location = GridLocation(aisle=aisle, rack=rack)
                
                # Item properties based on category
                if category == "computers":
                    name = f"Laptop Model {sku_counter}"
                    weight = 2.5
                    value = 1200.00
                    dimensions = (35, 25, 2)
                    fragile = True
                    quantity = 10 + (rack % 5)  # Vary stock
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
                sku_counter += 1
        
        print(f"Initialized {len(self.inventory)} electronics items in {WAREHOUSE_CONFIG['grid_size'][0]}×{WAREHOUSE_CONFIG['grid_size'][1]} grid")
    
    def create_electronics_order(self, customer_id: str, customer_type: str, 
                                items: List[Tuple[str, int]]) -> str:
        """Create optimized electronics order"""
        from uuid import uuid4
        
        order_id = f"ELEC-ORD-{self.order_counter:06d}"
        self.order_counter += 1
        
        order_items = []
        for sku, quantity in items:
            if sku in self.inventory:
                order_items.append(OrderItem(sku=sku, quantity=quantity))
            else:
                print(f"Warning: SKU {sku} not found, skipping")
        
        # Set priority based on customer type
        priority_map = {"wholesale": 1, "repair_shop": 2, "retail": 3}
        priority = priority_map.get(customer_type, 3)
        
        order = CustomerOrder(
            order_id=order_id,
            customer_id=customer_id,
            customer_type=customer_type,
            items=order_items,
            status=OrderStatus.PENDING,
            created_at=datetime.now(),
            priority=priority
        )
        
        self.orders[order_id] = order
        return order_id
    
    def process_grid_order(self, order_id: str) -> Dict:
        """Process order with grid-optimized routing"""
        if order_id not in self.orders:
            return {"error": "Order not found"}
        
        order = self.orders[order_id]
        order.status = OrderStatus.PROCESSING
        
        print(f"\n Processing Electronics Order: {order_id}")
        print(f"   Customer: {order.customer_id} ({order.customer_type})")
        print(f"   Priority: {order.priority}")
        print(f"   Items: {len(order.items)}")
        
        # Get inventory items for this order
        order_items = []
        for order_item in order.items:
            if order_item.sku in self.inventory:
                inv_item = self.inventory[order_item.sku]
                # Add item for each quantity unit (simplified)
                for _ in range(min(order_item.quantity, 5)):  # Cap for demo
                    order_items.append(inv_item)
        
        # Generate grid-optimized route
        route = self.optimizer.generate_grid_route(order_items)
        metrics = self.optimizer.calculate_route_metrics(route)
        
        # Create picking plan
        picking_plan = []
        for location in route:
            # Find which items are at this location
            items_at_loc = [item for item in order_items if item.location == location]
            if items_at_loc:
                picking_plan.append({
                    "location": str(location),
                    "items": [{"sku": item.sku, "name": item.name} 
                             for item in items_at_loc[:1]]  # First item only for demo
                })
        
        return {
            "order_id": order_id,
            "status": "route_generated",
            "route_locations": [str(loc) for loc in route],
            "picking_plan": picking_plan,
            "metrics": metrics,
            "grid_optimization": True,
            "total_value": self._calculate_order_value(order)
        }
    
    def execute_grid_picking(self, order_id: str, picker_id: str = "PICKER-01") -> Dict:
        """Execute picking with grid optimization"""
        if order_id not in self.orders:
            return {"error": "Order not found"}
        
        order = self.orders[order_id]
        start_time = datetime.now()
        
        print(f"\n{'='*60}")
        print(f" EXECUTING GRID-OPTIMIZED PICKING")
        print(f"{'='*60}")
        print(f"Order: {order_id}")
        print(f"Picker: {picker_id}")
        print(f"Start: {start_time.strftime('%H:%M:%S')}")
        print(f"{'='*60}")
        
        picked_items = []
        failed_items = []
        
        for order_item in order.items:
            if order_item.sku not in self.inventory:
                print(f" SKU {order_item.sku} not in inventory")
                order_item.status = "failed"
                failed_items.append(order_item.sku)
                continue
            
            inv_item = self.inventory[order_item.sku]
            
            print(f"\n Location: {inv_item.location}")
            print(f"   Item: {inv_item.name}")
            print(f"   Category: {inv_item.category}")
            print(f"   Requested: {order_item.quantity}")
            print(f"   Available: {inv_item.quantity}")
            
            if inv_item.quantity >= order_item.quantity:
                # Successful pick
                pick_qty = order_item.quantity
                inv_item.quantity -= pick_qty
                order_item.picked = pick_qty
                order_item.status = "picked"
                
                print(f"    Picked: {pick_qty}")
                print(f"   Remaining: {inv_item.quantity}")
                
                if inv_item.fragile:
                    print(f"     Fragile item - Handle with care")
                
                picked_items.append({
                    "sku": inv_item.sku,
                    "quantity": pick_qty,
                    "location": str(inv_item.location),
                    "value": inv_item.value * pick_qty
                })
            else:
                # Partial or failed pick
                available = inv_item.quantity
                if available > 0:
                    # Partial fulfillment
                    pick_qty = available
                    inv_item.quantity = 0
                    order_item.picked = pick_qty
                    order_item.status = "partial"
                    
                    print(f"     Partial: {pick_qty}/{order_item.quantity}")
                    picked_items.append({
                        "sku": inv_item.sku,
                        "quantity": pick_qty,
                        "location": str(inv_item.location),
                        "value": inv_item.value * pick_qty,
                        "partial": True
                    })
                else:
                    # Complete failure
                    print(f"    Out of stock")
                    order_item.status = "failed"
                    failed_items.append(inv_item.sku)
        
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
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds() / 60  # minutes
        
        # Print summary
        print(f"\n{'='*60}")
        print(f" PICKING COMPLETE - SUMMARY")
        print(f"{'='*60}")
        print(f"Order: {order_id}")
        print(f"Status: {final_status}")
        print(f"Fulfillment: {fulfillment_rate:.1f}%")
        print(f"Duration: {duration:.1f} minutes")
        print(f"Items Picked: {len(picked_items)}")
        print(f"Items Failed: {len(failed_items)}")
        
        if failed_items:
            print(f"\nFailed Items:")
            for sku in failed_items[:5]:  # Show first 5 only
                print(f"  • {sku}")
        
        # Calculate order value
        total_value = sum(item.get('value', 0) for item in picked_items)
        print(f"\nTotal Order Value: ${total_value:,.2f} AUD")
        
        return {
            "order_id": order_id,
            "status": final_status,
            "fulfillment_rate": fulfillment_rate,
            "picked_items": len(picked_items),
            "failed_items": len(failed_items),
            "total_value": total_value,
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "duration_minutes": round(duration, 2),
            "grid_efficiency": "optimized"
        }
    
    def batch_process_orders(self, order_ids: List[str]) -> Dict:
        """Process multiple orders in batch (optimized for grid)"""
        print(f"\n BATCH PROCESSING {len(order_ids)} ORDERS")
        print(f"{'='*60}")
        
        all_items = []
        order_info = {}
        
        # Collect all items from all orders
        for order_id in order_ids:
            if order_id in self.orders:
                order = self.orders[order_id]
                for order_item in order.items:
                    if order_item.sku in self.inventory:
                        inv_item = self.inventory[order_item.sku]
                        all_items.append({
                            "order_id": order_id,
                            "item": inv_item,
                            "quantity": order_item.quantity
                        })
                order_info[order_id] = order
        
        # Group by location for batch picking
        location_groups = {}
        for item_data in all_items:
            loc_str = str(item_data["item"].location)
            if loc_str not in location_groups:
                location_groups[loc_str] = []
            location_groups[loc_str].append(item_data)
        
        # Generate optimized batch route
        unique_locations = list(location_groups.keys())
        
        print(f"\nBatch Optimization:")
        print(f"  Total unique locations: {len(unique_locations)}")
        print(f"  Total items to pick: {len(all_items)}")
        
        # Simulate batch picking efficiency
        single_order_trips = len(order_ids) * 15  # Estimated
        batch_trip = len(unique_locations)
        efficiency_gain = ((single_order_trips - batch_trip) / single_order_trips * 100)
        
        return {
            "batch_size": len(order_ids),
            "unique_locations": len(unique_locations),
            "total_items": len(all_items),
            "estimated_single_trips": single_order_trips,
            "estimated_batch_trips": batch_trip,
            "efficiency_gain": round(efficiency_gain, 1),
            "locations": unique_locations[:10]  # First 10 only
        }
    
    def get_grid_visualization(self) -> Dict:
        """Generate grid visualization data"""
        grid = np.zeros(WAREHOUSE_CONFIG["grid_size"], dtype=int)
        
        # Fill grid with inventory counts
        for item in self.inventory.values():
            if 1 <= item.location.aisle <= WAREHOUSE_CONFIG["grid_size"][0] and \
               1 <= item.location.rack <= WAREHOUSE_CONFIG["grid_size"][1]:
                grid[item.location.aisle-1, item.location.rack-1] = item.quantity
        
        # Find high-traffic areas (low stock)
        low_stock_locations = []
        for aisle in range(WAREHOUSE_CONFIG["grid_size"][0]):
            for rack in range(WAREHOUSE_CONFIG["grid_size"][1]):
                if grid[aisle, rack] < 10:  # Threshold for low stock
                    low_stock_locations.append(f"A{aisle+1:02d}-R{rack+1:03d}")
        
        return {
            "grid_size": WAREHOUSE_CONFIG["grid_size"],
            "total_cells": WAREHOUSE_CONFIG["grid_size"][0] * WAREHOUSE_CONFIG["grid_size"][1],
            "occupied_cells": np.count_nonzero(grid),
            "total_items": int(np.sum(grid)),
            "average_stock_per_cell": round(np.mean(grid[grid > 0]), 1),
            "low_stock_locations": low_stock_locations[:20],  # First 20 only
            "grid_sample": grid.tolist()[:3]  # First 3 rows only
        }
    
    def _calculate_order_value(self, order: CustomerOrder) -> float:
        """Calculate total value of an order"""
        total = 0.0
        for order_item in order.items:
            if order_item.sku in self.inventory:
                total += self.inventory[order_item.sku].value * order_item.quantity
        return total
    
    def save_grid_report(self):
        """Save comprehensive grid warehouse report"""
        import json
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "warehouse_config": WAREHOUSE_CONFIG,
            "inventory_summary": {
                "total_items": len(self.inventory),
                "total_quantity": sum(item.quantity for item in self.inventory.values()),
                "total_value": sum(item.quantity * item.value for item in self.inventory.values()),
                "by_category": {}
            },
            "order_summary": {
                "total_orders": len(self.orders),
                "by_status": {},
                "by_customer_type": {}
            },
            "grid_analysis": self.get_grid_visualization()
        }
        
        # Category breakdown
        for item in self.inventory.values():
            cat = item.category
            if cat not in report["inventory_summary"]["by_category"]:
                report["inventory_summary"]["by_category"][cat] = {
                    "count": 0,
                    "quantity": 0,
                    "value": 0.0
                }
            report["inventory_summary"]["by_category"][cat]["count"] += 1
            report["inventory_summary"]["by_category"][cat]["quantity"] += item.quantity
            report["inventory_summary"]["by_category"][cat]["value"] += item.quantity * item.value
        
        # Order status breakdown
        for status in OrderStatus:
            count = sum(1 for o in self.orders.values() if o.status == status)
            report["order_summary"]["by_status"][status.value] = count
        
        # Customer type breakdown
        customer_types = set(o.customer_type for o in self.orders.values())
        for ct in customer_types:
            count = sum(1 for o in self.orders.values() if o.customer_type == ct)
            report["order_summary"]["by_customer_type"][ct] = count
        
        filename = f"grid_warehouse_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, "w", encoding='utf-8') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"\n Grid warehouse report saved to: {filename}")
        return report

# ========== DEMONSTRATION ==========
def run_grid_demo():
    """Run demonstration of grid-optimized auto-picker"""
    print("=" * 70)
    print("GRID-OPTIMIZED AUTO-PICKER DEMONSTRATION")
    print("ICT304 - Warehouse Intelligence System")
    print(f"Warehouse: {WAREHOUSE_CONFIG['grid_size'][0]}×{WAREHOUSE_CONFIG['grid_size'][1]} grid")
    print("=" * 70)
    
    # Initialize grid-optimized picker
    print("\n1.  Initializing Electronics Warehouse...")
    picker = ElectronicsAutoPicker()
    
    # Show grid visualization
    print("\n2.  Grid Warehouse Visualization...")
    grid_data = picker.get_grid_visualization()
    print(f"   Grid Size: {grid_data['grid_size'][0]}×{grid_data['grid_size'][1]}")
    print(f"   Total Items: {grid_data['total_items']:,}")
    print(f"   Occupied Cells: {grid_data['occupied_cells']}")
    print(f"   Low Stock Locations: {len(grid_data['low_stock_locations'])}")
    
    # Create sample electronics orders
    print("\n3.  Creating Electronics Orders...")
    
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
        order_id = picker.create_electronics_order(
            f"RETAIL-{i+1:03d}",
            "retail",
            [
                (f"ELCE-{(i*50)+1:04d}", 2),
                (f"ELCE-{(i*50)+25:04d}", 1),
            ]
        )
        retail_orders.append(order_id)
        print(f"   • Retail Order {i+1}: {order_id} (Priority: 3)")
    
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
    
    # Process wholesale order with grid optimization
    print(f"\n4.  Processing Wholesale Order with Grid Optimization...")
    wholesale_result = picker.process_grid_order(wholesale_order)
    
    print(f"   Order: {wholesale_result['order_id']}")
    print(f"   Route Locations: {len(wholesale_result['route_locations'])}")
    print(f"   Total Distance: {wholesale_result['metrics']['total_distance_m']}m")
    print(f"   Estimated Time: {wholesale_result['metrics']['total_time_min']}min")
    print(f"   Order Value: ${wholesale_result['total_value']:,.2f}")
    
    # Execute picking
    print(f"\n5.  Executing Grid-Optimized Picking...")
    execution_result = picker.execute_grid_picking(wholesale_order)
    
    # Demonstrate batch processing
    print(f"\n6.  Demonstrating Batch Processing...")
    batch_result = picker.batch_process_orders(retail_orders + [repair_order])
    
    print(f"   Batch Size: {batch_result['batch_size']} orders")
    print(f"   Unique Locations: {batch_result['unique_locations']}")
    print(f"   Efficiency Gain: {batch_result['efficiency_gain']}%")
    print(f"   Sample Locations: {', '.join(batch_result['locations'][:3])}...")
    
    # Save comprehensive report
    print(f"\n7.  Saving Warehouse Report...")
    report = picker.save_grid_report()
    
    print(f"\n{'='*70}")
    print(f" GRID-OPTIMIZED DEMONSTRATION COMPLETE")
    print(f"{'='*70}")
    
    # Quick statistics
    print(f"\n DEMONSTRATION STATISTICS:")
    print(f"   • Total Inventory Items: {report['inventory_summary']['total_items']}")
    print(f"   • Total Inventory Value: ${report['inventory_summary']['total_value']:,.2f}")
    print(f"   • Total Orders Processed: {report['order_summary']['total_orders']}")
    print(f"   • Warehouse Utilization: {(report['grid_analysis']['occupied_cells']/report['grid_analysis']['total_cells']*100):.1f}%")
    
    return picker, wholesale_order

def demonstrate_grid_features():
    """Demonstrate specific grid optimization features"""
    print("\n" + "=" * 70)
    print("GRID OPTIMIZATION FEATURES")
    print("=" * 70)
    
    picker = ElectronicsAutoPicker()
    
    # Create orders in same aisle to show aisle optimization
    print("\n Aisle-Based Optimization Example:")
    
    # Create orders with items in same aisle
    aisle_orders = []
    for aisle in [1, 3, 6]:  # Different aisles
        items = []
        for rack in [1, 5, 10, 15, 20]:  # Spread in same aisle
            # Find an item in this aisle
            for sku, item in picker.inventory.items():
                if item.location.aisle == aisle and item.location.rack == rack:
                    items.append((sku, 1))
                    break
        
        if items:
            order_id = picker.create_electronics_order(
                f"AISLE-{aisle}",
                "retail",
                items
            )
            aisle_orders.append(order_id)
            print(f"   Created order {order_id} in Aisle {aisle} with {len(items)} items")
    
    # Process and compare
    for order_id in aisle_orders[:2]:  # Process first 2
        result = picker.process_grid_order(order_id)
        print(f"\n   Order {order_id}:")
        print(f"     Route length: {len(result['route_locations'])} locations")
        print(f"     Distance: {result['metrics']['total_distance_m']}m")
        print(f"     Time: {result['metrics']['total_time_min']}min")
        
        # Check if route stays in same aisle
        aisles_in_route = set(loc.split('-')[0] for loc in result['route_locations'])
        print(f"     Aisles visited: {', '.join(sorted(aisles_in_route))}")
    
    return picker

if __name__ == "__main__":
    import sys
    
    print("ELECTRONICS WAREHOUSE AUTO-PICKER")
    print("ICT304 Assignment - Warehouse Intelligence System")
    print("Optimized for single-layer 6×24 grid configuration\n")
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--grid-demo":
            picker, order_id = run_grid_demo()
        elif sys.argv[1] == "--grid-features":
            picker = demonstrate_grid_features()
        elif sys.argv[1] == "--quick-test":
            # Quick test mode
            picker = ElectronicsAutoPicker()
            order_id = picker.create_electronics_order(
                "TEST-001",
                "retail",
                [("ELEC-0001", 2), ("ELEC-0050", 1)]
            )
            result = picker.process_grid_order(order_id)
            print(json.dumps(result, indent=2))
        else:
            print("Usage: python grid_auto_picker.py [--grid-demo|--grid-features|--quick-test]")
    else:
        # Run full grid demo by default
        picker, order_id = run_grid_demo()
        
        # Ask about additional features
        response = input("\nShow grid optimization features? (y/n): ")
        if response.lower() == 'y':
            demonstrate_grid_features()
    
    print("\n Generated files:")
    print("   - grid_warehouse_report_*.json")
    print("   - (Run with --grid-demo for full demonstration)")
