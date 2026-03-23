#!/usr/bin/env python3
"""
ORDER DATABASE MODULE
Subsystem 3 — Intelligent Auto Courier Email System
ICT304 — Warehouse Intelligence System

Simulates an order management database with synthetic data.
Provides barcode scan → order lookup.
"""

import random
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, Optional, List


# ============================================================================
# CONFIGURATION
# ============================================================================

PRODUCT_CATEGORIES = {
    "electronics": {
        "items": ["Laptop", "Smartphone", "Tablet", "Monitor", "Headphones",
                  "Keyboard", "Mouse", "Webcam", "Speaker", "Smartwatch"],
        "weight_range": (0.2, 15.0),
        "value_range": (50, 3000),
        "fragile": False
    },
    "glass": {
        "items": ["Glass Vase", "Wine Glasses", "Crystal Display",
                  "Mirror", "Chandelier", "Glass Table Top"],
        "weight_range": (0.5, 20.0),
        "value_range": (30, 500),
        "fragile": True
    },
    "ceramics": {
        "items": ["Ceramic Set", "Porcelain Bowl", "Clay Pot",
                  "Ceramic Tiles Pack", "Art Sculpture"],
        "weight_range": (1.0, 15.0),
        "value_range": (20, 400),
        "fragile": True
    },
    "clothing": {
        "items": ["T-Shirt Pack", "Jacket", "Shoes", "Dress", "Suit",
                  "Accessories Set", "Sportswear", "Winter Coat"],
        "weight_range": (0.3, 5.0),
        "value_range": (20, 300),
        "fragile": False
    },
    "machinery": {
        "items": ["Motor Part", "Pump Assembly", "Gear Set",
                  "Industrial Valve", "Compressor Unit", "Bearing Kit"],
        "weight_range": (5.0, 50.0),
        "value_range": (100, 5000),
        "fragile": False
    },
    "food": {
        "items": ["Frozen Seafood", "Fresh Produce", "Dairy Products",
                  "Imported Chocolates", "Health Supplements"],
        "weight_range": (1.0, 25.0),
        "value_range": (20, 200),
        "fragile": False
    },
    "documents": {
        "items": ["Legal Documents", "Contract Package",
                  "Certificate Set", "Blueprint Roll"],
        "weight_range": (0.1, 3.0),
        "value_range": (5, 50),
        "fragile": False
    }
}

DESTINATION_ZONES = {
    "local": {
        "postcodes": ["560123", "530210", "680456", "520789", "310654",
                      "730891", "460234", "640567", "510890", "380345"],
        "distance_range": (2, 15),
        "country": "Singapore"
    },
    "regional": {
        "postcodes": ["50000", "10110", "1010", "50450", "12000"],
        "distance_range": (300, 3000),
        "country_options": ["Malaysia", "Thailand", "Indonesia",
                            "Philippines", "Vietnam"]
    },
    "international": {
        "postcodes": ["SW1A 1AA", "10001", "100-0001", "2000", "M5V 3L9"],
        "distance_range": (5000, 15000),
        "country_options": ["United Kingdom", "United States", "Japan",
                            "Australia", "Canada"]
    }
}

WAREHOUSE_INFO = {
    "name": "Smart Electronics Warehouse",
    "address": "50 Nanyang Avenue, Singapore 639798",
    "contact": "+65 6791 2000",
    "operating_hours": "08:00 - 18:00 SGT",
    "dock_number": "Dock B-3"
}


# ============================================================================
# DATA MODEL
# ============================================================================

@dataclass
class Order:
    """Represents a complete order retrieved from the database."""
    order_id: str
    product_name: str
    product_category: str
    weight_kg: float
    length_cm: float
    width_cm: float
    height_cm: float
    destination_postcode: str
    destination_address: str
    destination_country: str
    destination_zone: str        # local / regional / international
    distance_km: float
    order_priority: str          # "Urgent" / "High" / "Standard"
    declared_value: float
    currency: str
    item_count: int
    contact_name: str
    contact_email: str
    contact_phone: str
    created_at: str = ""

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()

    @property
    def volume_cm3(self) -> float:
        return self.length_cm * self.width_cm * self.height_cm

    @property
    def dimensions_str(self) -> str:
        return f"{self.length_cm}x{self.width_cm}x{self.height_cm} cm"

    @property
    def is_fragile(self) -> bool:
        """Rule-based: glass and ceramics are fragile."""
        return PRODUCT_CATEGORIES.get(
            self.product_category, {}
        ).get("fragile", False)

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["volume_cm3"] = self.volume_cm3
        d["dimensions_str"] = self.dimensions_str
        d["is_fragile"] = self.is_fragile
        return d


# ============================================================================
# ORDER DATABASE
# ============================================================================

class OrderDatabase:
    """Simulated order management database."""

    def __init__(self, num_orders: int = 100):
        self.orders: Dict[str, Order] = {}
        self._generate_synthetic_orders(num_orders)

    def scan_barcode(self, order_id: str) -> Optional[Order]:
        """
        Simulate barcode scanning → order lookup.
        This is the entry point for the full pipeline.
        """
        return self.orders.get(order_id)

    def get_all_order_ids(self) -> List[str]:
        return list(self.orders.keys())

    def get_random_order_ids(self, n: int = 5) -> List[str]:
        ids = list(self.orders.keys())
        return random.sample(ids, min(n, len(ids)))

    def _generate_synthetic_orders(self, count: int):
        """Generate realistic synthetic orders."""
        names = [
            "John Tan", "Sarah Lee", "Ahmad Hassan", "Yuki Tanaka",
            "Wei Lin Chen", "Priya Sharma", "David Kim", "Maria Santos",
            "Robert Smith", "Fatimah Ali", "James Wilson", "Aiko Suzuki",
            "Mark Johnson", "Lisa Wong", "Ravi Patel", "Emma Brown",
            "Kenji Nakamura", "Sofia Garcia", "Peter Lim", "Hannah Ong"
        ]
        priorities = ["Urgent", "High", "Standard"]

        for i in range(1, count + 1):
            order_id = f"ORD-{i:05d}"
            category = random.choice(list(PRODUCT_CATEGORIES.keys()))
            cat = PRODUCT_CATEGORIES[category]

            weight = round(random.uniform(*cat["weight_range"]), 2)
            length = round(random.uniform(10, 120), 1)
            width = round(random.uniform(8, 80), 1)
            height = round(random.uniform(5, 60), 1)

            zone = random.choices(
                ["local", "regional", "international"],
                weights=[0.5, 0.3, 0.2]
            )[0]
            zc = DESTINATION_ZONES[zone]
            postcode = random.choice(zc["postcodes"])
            distance = round(random.uniform(*zc["distance_range"]), 1)

            if zone == "local":
                country = "Singapore"
                address = f"Blk {random.randint(1,999)} " \
                          f"{random.choice(['Ang Mo Kio','Bedok','Clementi','Tampines','Jurong'])} " \
                          f"Ave {random.randint(1,10)}, Singapore {postcode}"
            else:
                country = random.choice(zc["country_options"])
                address = f"{random.randint(1,500)} " \
                          f"{random.choice(['Main St','High St','Park Ave','Central Rd'])}, " \
                          f"{country} {postcode}"

            priority = random.choices(
                priorities, weights=[0.2, 0.3, 0.5]
            )[0]
            value = round(random.uniform(*cat["value_range"]), 2)
            contact = random.choice(names)

            self.orders[order_id] = Order(
                order_id=order_id,
                product_name=random.choice(cat["items"]),
                product_category=category,
                weight_kg=weight,
                length_cm=length, width_cm=width, height_cm=height,
                destination_postcode=postcode,
                destination_address=address,
                destination_country=country,
                destination_zone=zone,
                distance_km=distance,
                order_priority=priority,
                declared_value=value,
                currency="SGD",
                item_count=random.randint(1, 10),
                contact_name=contact,
                contact_email=contact.lower().replace(" ", ".") + "@example.com",
                contact_phone=f"+65 {random.randint(8000,9999)} {random.randint(1000,9999)}"
            )

    def get_summary(self) -> Dict:
        orders = list(self.orders.values())
        categories = {}
        zones = {}
        for o in orders:
            categories[o.product_category] = categories.get(o.product_category, 0) + 1
            zones[o.destination_zone] = zones.get(o.destination_zone, 0) + 1
        return {
            "total_orders": len(orders),
            "by_category": categories,
            "by_zone": zones
        }
