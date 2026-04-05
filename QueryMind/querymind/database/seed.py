"""
Seed script to populate a realistic e-commerce SQLite database for QueryMind.
Runnable via: python -m querymind.database.seed
"""

import asyncio
import os

import aiosqlite

from querymind.config import settings

# E-commerce Database Schema
SCHEMA_SQL = """
PRAGMA foreign_keys = ON;

DROP TABLE IF EXISTS order_items;
DROP TABLE IF EXISTS orders;
DROP TABLE IF EXISTS products;
DROP TABLE IF EXISTS users;

CREATE TABLE users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    email TEXT UNIQUE NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE products (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    category TEXT NOT NULL,
    price REAL NOT NULL,
    stock_quantity INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE orders (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    status TEXT NOT NULL,
    total_amount REAL NOT NULL,
    ordered_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

CREATE TABLE order_items (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    order_id INTEGER NOT NULL,
    product_id INTEGER NOT NULL,
    quantity INTEGER NOT NULL,
    unit_price REAL NOT NULL,
    FOREIGN KEY (order_id) REFERENCES orders(id) ON DELETE CASCADE,
    FOREIGN KEY (product_id) REFERENCES products(id) ON DELETE CASCADE
);
"""

# Seed Data
USERS = [
    ("Alice Smith", "alice@example.com"),
    ("Bob Johnson", "bob@example.com"),
    ("Charlie Brown", "charlie@example.com")
]

PRODUCTS = [
    ("Laptop Pro", "Electronics", 1299.99, 50),
    ("Wireless Mouse", "Electronics", 49.99, 200),
    ("Mechanical Keyboard", "Electronics", 149.99, 150),
    ("Coffee Mug", "Home", 12.50, 500),
    ("Office Chair", "Furniture", 249.00, 30)
]

ORDERS = [
    (1, "DELIVERED", 1349.98),
    (2, "SHIPPED", 249.00),
    (1, "PENDING", 12.50)
]

ORDER_ITEMS = [
    (1, 1, 1, 1299.99), # Order 1: 1 Laptop Pro
    (1, 2, 1, 49.99),   # Order 1: 1 Wireless Mouse
    (2, 5, 1, 249.00),  # Order 2: 1 Office Chair
    (3, 4, 1, 12.50)    # Order 3: 1 Coffee Mug
]


async def seed_database() -> None:
    """Create tables and insert initial seed data."""
    print(f"Creating database at {settings.db_path}...")
    
    # Ensure any required parent directories exist
    db_dir = os.path.dirname(settings.db_path)
    if db_dir and not os.path.exists(db_dir):
        os.makedirs(db_dir)

    async with aiosqlite.connect(settings.db_path) as db:
        print("Executing schema...")
        await db.executescript(SCHEMA_SQL)
        
        print("Inserting users...")
        await db.executemany(
            "INSERT INTO users (name, email) VALUES (?, ?)", USERS
        )
        
        print("Inserting products...")
        await db.executemany(
            "INSERT INTO products (name, category, price, stock_quantity) VALUES (?, ? , ?, ?)", PRODUCTS
        )
        
        print("Inserting orders...")
        await db.executemany(
            "INSERT INTO orders (user_id, status, total_amount) VALUES (?, ?, ?)", ORDERS
        )
        
        print("Inserting order items...")
        await db.executemany(
            "INSERT INTO order_items (order_id, product_id, quantity, unit_price) VALUES (?, ?, ?, ?)", ORDER_ITEMS
        )
        
        await db.commit()
        print("Database seeding complete!")


if __name__ == "__main__":
    asyncio.run(seed_database())
