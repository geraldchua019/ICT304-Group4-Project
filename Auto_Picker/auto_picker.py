#!/usr/bin/env python3
"""
SIMPLE AUTO-PICKER
For ICT304 Assignment - Warehouse Intelligence System
"""

import math
import heapq
from datetime import datetime
from dataclasses import dataclass
from typing import List, Dict, Optional
from enum import Enum

# ========== DATA MODELS ==========
class OrderStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    PICKING = "picking"
    COMPLETE = "complete"

@dataclass
class WarehouseLocation:
    zone: str
    aisle: int
    rack: int
    shelf: int
    x: float
    y: float
    
    def __str__(self):
        return f"{self.zone}-{self.aisle:02d}-{self.rack:02d}-{self.shelf}"

@dataclass
class InventoryItem:
    sku: str
    name: str
    location: WarehouseLocation
    quantity: int

@dataclass
class OrderItem:
    sku: str
    quantity: int

@dataclass
class CustomerOrder:
    order_id: str
    customer_id: str
    items: List[OrderItem]
    status: OrderStatus

# ========== ROUTE OPTIMIZER ==========
class RouteOptimizer:
    """Simple route optimizer using A* algorithm"""
    
    @staticmethod
    def distance(loc1: WarehouseLocation, loc2: WarehouseLocation) -> float:
        """Calculate Euclidean distance between two locations"""
        return math.sqrt((loc1.x - loc2.x)**2 + (loc1.y - loc2.y)**2)
    
    def find_optimal_route(self, start: WarehouseLocation, 
                          items: List[InventoryItem]) -> List[WarehouseLocation]:
        """Find optimal route using nearest neighbor algorithm"""
        if not items:
            return []
        
        route = []
        current = start
        unvisited = items.copy()
        
        while unvisited:
            # Find nearest item
            nearest = min(unvisited, key=lambda item: self.distance(current, item.location))
            route.append(nearest.location)
            current = nearest.location
            unvisited.remove(nearest)
        
        return route

# ========== AUTO-PICKER CONTROLLER ==========
class SimpleAutoPicker:
    """Simplified auto-picker controller"""
    
    def __init__(self):
        self.orders: Dict[str, CustomerOrder] = {}
        self.inventory: Dict[str, InventoryItem] = {}
        self.optimizer = RouteOptimizer()
        
        # Initialize sample data
        self._init_sample_data()
    
    def _init_sample_data(self):
        """Initialize with sample warehouse data"""
        # Sample inventory
        sample_items = [
            InventoryItem("ITEM-001", "Wireless Mouse", 
                         WarehouseLocation("A", 1, 1, 1, 10.0, 5.0), 50),
            InventoryItem("ITEM-002", "USB-C Cable", 
                         WarehouseLocation("A", 1, 2, 2, 12.0, 5.0), 200),
            InventoryItem("ITEM-003", "Notebook", 
                         WarehouseLocation("B", 2, 1, 1, 20.0, 5.0), 100),
        ]
        
        for item in sample_items:
            self.inventory[item.sku] = item
        
        # Starting location for pickers
        self.start_location = WarehouseLocation("START", 0, 0, 0, 0.0, 0.0)
    
    def create_order(self, customer_id: str, items: List[OrderItem]) -> str:
        """Create a new customer order"""
        import uuid
        order_id = f"ORD-{uuid.uuid4().hex[:6].upper()}"
        
        order = CustomerOrder(
            order_id=order_id,
            customer_id=customer_id,
            items=items,
            status=OrderStatus.PENDING
        )
        
        self.orders[order_id] = order
        return order_id
    
    def process_order(self, order_id: str) -> Dict:
        """Process an order and generate picking route"""
        if order_id not in self.orders:
            return {"error": "Order not found"}
        
        order = self.orders[order_id]
        
        # Get inventory items for this order
        order_items = []
        for order_item in order.items:
            if order_item.sku in self.inventory:
                inv_item = self.inventory[order_item.sku]
                order_items.append(inv_item)
        
        # Generate optimal route
        route = self.optimizer.find_optimal_route(self.start_location, order_items)
        
        # Calculate metrics
        total_distance = 0
        if route:
            # Distance from start to first location
            total_distance += self.optimizer.distance(self.start_location, route[0])
            
            # Distance between route locations
            for i in range(len(route) - 1):
                total_distance += self.optimizer.distance(route[i], route[i + 1])
        
        # Update order status
        order.status = OrderStatus.PROCESSING
        
        return {
            "order_id": order_id,
            "route": [str(loc) for loc in route],
            "total_locations": len(route),
            "estimated_distance": round(total_distance, 2),
            "estimated_time": round(total_distance / 1.4 / 60, 2),  # Convert to minutes
            "status": "ready_for_picking"
        }
    
    def execute_picking(self, order_id: str) -> Dict:
        """Execute picking for an order"""
        if order_id not in self.orders:
            return {"error": "Order not found"}
        
        order = self.orders[order_id]
        
        print(f"\n Executing picking for order {order_id}")
        print("-" * 40)
        
        # Simulate picking process
        for i, order_item in enumerate(order.items, 1):
            if order_item.sku in self.inventory:
                item = self.inventory[order_item.sku]
                
                print(f"Step {i}: Picking {order_item.quantity}x {item.name}")
                print(f"    Location: {item.location}")
                print(f"    SKU: {item.sku}")
                
                # Update inventory
                if item.quantity >= order_item.quantity:
                    item.quantity -= order_item.quantity
                    print(f"     Success - Remaining: {item.quantity}")
                else:
                    print(f"    ️ Insufficient stock!")
        
        # Update order status
        order.status = OrderStatus.COMPLETE
        
        print("-" * 40)
        print(f" Order {order_id} picking complete!")
        
        return {
            "order_id": order_id,
            "status": "complete",
            "completed_at": datetime.now().isoformat()
        }
    
    def get_status(self) -> Dict:
        """Get system status"""
        pending = sum(1 for o in self.orders.values() if o.status == OrderStatus.PENDING)
        processing = sum(1 for o in self.orders.values() if o.status == OrderStatus.PROCESSING)
        
        return {
            "total_orders": len(self.orders),
            "pending_orders": pending,
            "processing_orders": processing,
            "total_inventory_items": len(self.inventory),
            "timestamp": datetime.now().isoformat()
        }

# ========== MAIN DEMONSTRATION ==========
def run_demo():
    """Run a complete demonstration"""
    print("=" * 60)
    print("AUTO-PICKER SUBSYSTEM DEMONSTRATION")
    print("ICT304 - Warehouse Intelligence System")
    print("=" * 60)
    
    # Initialize system
    print("\n1. Initializing Auto-Picker System...")
    picker = SimpleAutoPicker()
    
    # Show initial status
    status = picker.get_status()
    print(f"   • Inventory items: {status['total_inventory_items']}")
    print(f"   • Active orders: {status['total_orders']}")
    
    # Create sample orders
    print("\n2. Creating Sample Orders...")
    
    order1_id = picker.create_order(
        "CUST-001",
        [
            OrderItem("ITEM-001", 2),
            OrderItem("ITEM-002", 3)
        ]
    )
    print(f"   • Created Order 1: {order1_id}")
    
    order2_id = picker.create_order(
        "CUST-002",
        [
            OrderItem("ITEM-003", 1),
            OrderItem("ITEM-001", 1)
        ]
    )
    print(f"   • Created Order 2: {order2_id}")
    
    # Process orders
    print("\n3. Processing Orders and Generating Routes...")
    
    for i, order_id in enumerate([order1_id, order2_id], 1):
        result = picker.process_order(order_id)
        print(f"\n   Order {i} ({order_id}):")
        print(f"     Route: {result['route']}")
        print(f"     Locations: {result['total_locations']}")
        print(f"     Distance: {result['estimated_distance']}m")
        print(f"     Time: {result['estimated_time']}min")
    
    # Execute picking
    print("\n4. Executing Picking...")
    picker.execute_picking(order1_id)
    
    # Final status
    print("\n5. Final System Status:")
    final_status = picker.get_status()
    for key, value in final_status.items():
        if key != "timestamp":
            print(f"   • {key.replace('_', ' ').title()}: {value}")
    
    print("\n" + "=" * 60)
    print(" DEMONSTRATION COMPLETE")
    print("=" * 60)

def run_api_demo():
    """Demonstrate API-like functionality"""
    print("\n" + "=" * 60)
    print("API DEMONSTRATION")
    print("=" * 60)
    
    picker = SimpleAutoPicker()
    
    # Create order via "API"
    print("\n Creating order via API call...")
    order_data = {
        "customer_id": "API-CUSTOMER",
        "items": [
            {"sku": "ITEM-001", "quantity": 1},
            {"sku": "ITEM-002", "quantity": 2},
            {"sku": "ITEM-003", "quantity": 1}
        ]
    }
    
    # Convert to OrderItem objects
    order_items = [OrderItem(**item) for item in order_data["items"]]
    order_id = picker.create_order(order_data["customer_id"], order_items)
    
    print(f"Order created: {order_id}")
    
    # Process order
    print("\n️ Generating picking route...")
    route_info = picker.process_order(order_id)
    
    print(f"Route generated:")
    print(f"  Path: {' → '.join(route_info['route'])}")
    print(f"  Estimated: {route_info['estimated_time']} minutes")
    
    return picker, order_id

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--api":
        picker, order_id = run_api_demo()
        
        # Ask if user wants to execute picking
        response = input("\nExecute picking? (y/n): ")
        if response.lower() == 'y':
            picker.execute_picking(order_id)
    else:
        run_demo()
    
    # Save demonstration data
    print("\n Demonstration data saved to 'picker_demo_log.txt'")
    with open("picker_demo_log.txt", "w") as f:
        f.write(f"Auto-Picker Demo - {datetime.now()}\n")
        f.write("=" * 40 + "\n")
        f.write("This file demonstrates the functionality of the Auto-Picker subsystem.\n")
        f.write("Created for ICT304 Assignment - Warehouse Intelligence System\n")
