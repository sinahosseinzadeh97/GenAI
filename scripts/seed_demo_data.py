#!/usr/bin/env python3
"""
Seed QueryMind with demo data for presentations and testing.
Run: python scripts/seed_demo_data.py
"""

import asyncio
import aiosqlite
import os
import sys

DB_PATH = os.getenv("DB_PATH", "data/querymind.db")


async def seed():
    print(f"Seeding demo data into {DB_PATH}...")

    async with aiosqlite.connect(DB_PATH) as db:

        # Create tables
        await db.executescript("""
            CREATE TABLE IF NOT EXISTS customers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT UNIQUE NOT NULL,
                plan TEXT NOT NULL,
                country TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS products (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                category TEXT NOT NULL,
                price REAL NOT NULL,
                stock INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS orders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                customer_id INTEGER NOT NULL,
                product_id INTEGER NOT NULL,
                quantity INTEGER NOT NULL,
                status TEXT NOT NULL,
                total REAL NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (customer_id) REFERENCES customers(id),
                FOREIGN KEY (product_id) REFERENCES products(id)
            );
        """)

        # Seed customers
        customers = [
            ("Alice Rossi", "alice@example.com", "enterprise", "Italy", "2024-01-15"),
            ("Marco Bianchi", "marco@example.com", "pro", "Italy", "2024-02-03"),
            ("Sarah Chen", "sarah@example.com", "enterprise", "USA", "2024-02-14"),
            ("Luca Ferrari", "luca@example.com", "starter", "Italy", "2024-03-01"),
            ("Emma Wilson", "emma@example.com", "pro", "UK", "2024-03-22"),
            ("James Park", "james@example.com", "enterprise", "USA", "2024-04-10"),
            ("Sofia Greco", "sofia@example.com", "starter", "Italy", "2024-05-05"),
            ("David Kim", "david@example.com", "pro", "Korea", "2024-06-18"),
            ("Giulia Marino", "giulia@example.com", "enterprise", "Italy", "2024-07-30"),
            ("Thomas Muller", "thomas@example.com", "pro", "Germany", "2024-08-12"),
        ]

        await db.executemany(
            "INSERT OR IGNORE INTO customers (name, email, plan, country, created_at) VALUES (?,?,?,?,?)",
            customers
        )

        # Seed products
        products = [
            ("QueryMind Starter", "SaaS", 29.00, 999),
            ("QueryMind Pro", "SaaS", 99.00, 999),
            ("QueryMind Enterprise", "SaaS", 499.00, 999),
            ("RAG Module Add-on", "Add-on", 49.00, 500),
            ("Agent Module Add-on", "Add-on", 79.00, 500),
            ("Support Package", "Service", 199.00, 100),
            ("Custom Integration", "Service", 999.00, 50),
        ]

        await db.executemany(
            "INSERT OR IGNORE INTO products (name, category, price, stock) VALUES (?,?,?,?)",
            products
        )

        # Seed orders
        orders = [
            (1, 3, 1, "delivered", 499.00, "2024-01-20"),
            (2, 2, 1, "delivered", 99.00, "2024-02-10"),
            (3, 3, 1, "delivered", 499.00, "2024-02-20"),
            (3, 4, 2, "delivered", 98.00, "2024-03-01"),
            (4, 1, 1, "delivered", 29.00, "2024-03-15"),
            (5, 2, 1, "delivered", 99.00, "2024-04-01"),
            (6, 3, 1, "delivered", 499.00, "2024-04-15"),
            (6, 5, 1, "delivered", 79.00, "2024-04-16"),
            (7, 1, 1, "pending", 29.00, "2024-05-10"),
            (8, 2, 1, "delivered", 99.00, "2024-06-20"),
            (9, 3, 1, "delivered", 499.00, "2024-08-01"),
            (9, 6, 1, "delivered", 199.00, "2024-08-05"),
            (10, 2, 1, "delivered", 99.00, "2024-08-15"),
            (1, 7, 1, "in_progress", 999.00, "2024-09-01"),
            (3, 5, 1, "delivered", 79.00, "2024-09-10"),
        ]

        await db.executemany(
            "INSERT OR IGNORE INTO orders (customer_id, product_id, quantity, status, total, created_at) VALUES (?,?,?,?,?,?)",
            orders
        )

        await db.commit()

    print("✅ Demo data seeded successfully!")
    print("")
    print("Try these queries in SQL Mode:")
    print("  → How many customers do we have per country?")
    print("  → What is our total revenue from delivered orders?")
    print("  → Which customers are on the enterprise plan?")
    print("  → What are the top selling products?")
    print("  → Show me all pending orders")


if __name__ == "__main__":
    asyncio.run(seed())
