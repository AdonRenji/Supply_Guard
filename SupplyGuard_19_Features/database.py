"""
SupplyGuard Database Module
Handles SQLite schema, connection lifecycle, seed data, and transactions
for customer orders, order items, product catalogue, and prediction history.
"""

from pathlib import Path
import sqlite3
import random
import string
from datetime import datetime
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "supplyguard.db"


def get_db_connection():
    """Establish and return an SQLite connection with row factories and foreign keys enabled."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def init_db(dataset_path=None):
    """
    Initialize SQLite database tables and seed products from the DataCo dataset
    if the catalogue is currently empty.
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()

        # 1. Products table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS products (
                product_id INTEGER PRIMARY KEY,
                product_name TEXT NOT NULL,
                category_name TEXT,
                price REAL NOT NULL CHECK (price >= 0),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)

        # 2. Orders table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS orders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                order_id TEXT UNIQUE NOT NULL,
                customer_name TEXT NOT NULL,
                customer_email TEXT NOT NULL,
                shipping_mode TEXT NOT NULL,
                days_scheduled REAL NOT NULL CHECK (days_scheduled >= 0),
                market TEXT NOT NULL,
                order_region TEXT NOT NULL,
                order_country TEXT NOT NULL,
                order_state TEXT NOT NULL,
                order_type TEXT NOT NULL,
                order_date TEXT NOT NULL,
                order_time TEXT NOT NULL,
                total_amount REAL NOT NULL CHECK (total_amount >= 0),
                status TEXT DEFAULT 'CONFIRMED',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)

        # 3. Order Items table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS order_items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                order_id INTEGER NOT NULL,
                product_id INTEGER NOT NULL,
                product_name TEXT NOT NULL,
                quantity INTEGER NOT NULL CHECK (quantity >= 1),
                product_price REAL NOT NULL CHECK (product_price >= 0),
                discount_rate REAL NOT NULL DEFAULT 0.0 CHECK (discount_rate >= 0.0 AND discount_rate <= 1.0),
                sales REAL NOT NULL CHECK (sales >= 0),
                FOREIGN KEY (order_id) REFERENCES orders(id) ON DELETE CASCADE,
                FOREIGN KEY (product_id) REFERENCES products(product_id)
            );
        """)

        # 4. Predictions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prediction_id TEXT UNIQUE NOT NULL,
                order_id INTEGER NOT NULL,
                order_item_id INTEGER,
                is_late INTEGER NOT NULL CHECK (is_late IN (0, 1)),
                late_probability REAL NOT NULL,
                risk_label TEXT NOT NULL,
                key_insight TEXT,
                recommendations TEXT,
                model_version TEXT DEFAULT 'supplyguard_19_model_rf_v1',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (order_id) REFERENCES orders(id) ON DELETE CASCADE,
                FOREIGN KEY (order_item_id) REFERENCES order_items(id) ON DELETE SET NULL
            );
        """)

        # Seed products if empty
        cursor.execute("SELECT COUNT(*) AS cnt FROM products;")
        if cursor.fetchone()["cnt"] == 0:
            if dataset_path is None:
                dataset_path = BASE_DIR / "DataCoSupplyChainDataset.csv"

            if Path(dataset_path).exists():
                try:
                    df = pd.read_csv(
                        dataset_path,
                        encoding="cp1252",
                        usecols=["Product Card Id", "Product Name", "Category Name", "Product Price"]
                    )
                    prods = df.drop_duplicates(subset=["Product Card Id"]).sort_values("Product Card Id")

                    records = []
                    for _, row in prods.iterrows():
                        p_id = int(row["Product Card Id"])
                        name = str(row["Product Name"]).strip()
                        category = str(row["Category Name"]).strip() if pd.notna(row["Category Name"]) else "General"
                        price = round(float(row["Product Price"]), 2) if pd.notna(row["Product Price"]) else 0.0
                        records.append((p_id, name, category, price))

                    cursor.executemany("""
                        INSERT INTO products (product_id, product_name, category_name, price)
                        VALUES (?, ?, ?, ?);
                    """, records)
                except Exception as exc:
                    print(f"Warning: Could not seed products from dataset: {exc}")


def generate_order_id():
    """
    Generate a collision-safe, unique human-readable Order ID.
    Example format: SG-ORD-20261020-A4B7
    """
    date_str = datetime.now().strftime("%Y%m%d")
    with get_db_connection() as conn:
        for _ in range(20):
            suffix = "".join(random.choices(string.ascii_uppercase + string.digits, k=4))
            candidate_id = f"SG-ORD-{date_str}-{suffix}"
            cursor = conn.execute("SELECT 1 FROM orders WHERE order_id = ? LIMIT 1;", (candidate_id,))
            if cursor.fetchone() is None:
                return candidate_id

    # Fallback with timestamp in microsecond
    return f"SG-ORD-{date_str}-{datetime.now().strftime('%H%M%S%f')[:8]}"


def get_products_catalogue():
    """Return all products available in the catalogue, ordered by category and name."""
    with get_db_connection() as conn:
        cursor = conn.execute("""
            SELECT product_id, product_name, category_name, price
            FROM products
            ORDER BY category_name ASC, product_name ASC;
        """)
        return [dict(row) for row in cursor.fetchall()]


def get_product_by_id(product_id):
    """Retrieve single product by product_id."""
    with get_db_connection() as conn:
        cursor = conn.execute("""
            SELECT product_id, product_name, category_name, price
            FROM products
            WHERE product_id = ?;
        """, (product_id,))
        row = cursor.fetchone()
        return dict(row) if row else None


def save_customer_order(order_data, items_list):
    """
    Save customer order and its associated order items atomically inside a transaction.
    Returns:
        order_id (str): Generated human-readable Order ID
        internal_id (int): Primary key ID of the order row
    """
    if not items_list:
        raise ValueError("Order must contain at least one product.")

    order_id = generate_order_id()

    with get_db_connection() as conn:
        cursor = conn.cursor()

        # Insert parent order
        cursor.execute("""
            INSERT INTO orders (
                order_id, customer_name, customer_email, shipping_mode,
                days_scheduled, market, order_region, order_country,
                order_state, order_type, order_date, order_time, total_amount, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """, (
            order_id,
            order_data["customer_name"].strip(),
            order_data["customer_email"].strip().lower(),
            order_data["shipping_mode"].strip(),
            float(order_data["days_scheduled"]),
            order_data["market"].strip(),
            order_data["order_region"].strip(),
            order_data["order_country"].strip(),
            order_data["order_state"].strip(),
            order_data["order_type"].strip(),
            order_data["order_date"].strip(),
            order_data["order_time"].strip(),
            float(order_data["total_amount"]),
            order_data.get("status", "CONFIRMED")
        ))

        internal_id = cursor.lastrowid

        # Insert order items
        item_rows = []
        for item in items_list:
            item_rows.append((
                internal_id,
                int(item["product_id"]),
                item["product_name"].strip(),
                int(item["quantity"]),
                float(item["product_price"]),
                float(item.get("discount_rate", 0.0)),
                float(item["sales"])
            ))

        cursor.executemany("""
            INSERT INTO order_items (
                order_id, product_id, product_name, quantity,
                product_price, discount_rate, sales
            ) VALUES (?, ?, ?, ?, ?, ?, ?);
        """, item_rows)

        return order_id, internal_id


def get_order_by_order_id(order_id_query):
    """
    Look up an order by its human-readable Order ID (case-insensitive).
    Returns order dictionary with nested 'items' list, or None if not found.
    """
    if not order_id_query:
        return None

    clean_id = str(order_id_query).strip()

    with get_db_connection() as conn:
        cursor = conn.execute("""
            SELECT * FROM orders
            WHERE UPPER(order_id) = UPPER(?)
            LIMIT 1;
        """, (clean_id,))
        order_row = cursor.fetchone()

        if not order_row:
            return None

        order_dict = dict(order_row)

        # Retrieve items
        item_cursor = conn.execute("""
            SELECT * FROM order_items
            WHERE order_id = ?
            ORDER BY id ASC;
        """, (order_dict["id"],))

        order_dict["items"] = [dict(row) for row in item_cursor.fetchall()]
        order_dict["order_items"] = order_dict["items"]

        # Retrieve latest prediction if available
        pred_cursor = conn.execute("""
            SELECT * FROM predictions
            WHERE order_id = ?
            ORDER BY id DESC
            LIMIT 1;
        """, (order_dict["id"],))
        pred_row = pred_cursor.fetchone()
        order_dict["latest_prediction"] = dict(pred_row) if pred_row else None

        return order_dict


def get_recent_orders(limit=10):
    """Retrieve list of recent orders for convenient lookup on the manager dashboard."""
    with get_db_connection() as conn:
        cursor = conn.execute("""
            SELECT o.id, o.order_id, o.customer_name, o.customer_email,
                   o.order_date, o.order_time, o.total_amount, o.status,
                   o.shipping_mode, o.market, o.order_country, o.created_at,
                   COUNT(i.id) AS item_count
            FROM orders o
            LEFT JOIN order_items i ON o.id = i.order_id
            GROUP BY o.id
            ORDER BY o.id DESC
            LIMIT ?;
        """, (limit,))
        return [dict(row) for row in cursor.fetchall()]


def save_order_prediction(internal_order_id, prediction_id, is_late, late_probability,
                          risk_label, key_insight, recommendations_json, order_item_id=None):
    """Save delivery risk prediction result for a specific order into SQLite."""
    with get_db_connection() as conn:
        conn.execute("""
            INSERT INTO predictions (
                prediction_id, order_id, order_item_id, is_late,
                late_probability, risk_label, key_insight, recommendations
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?);
        """, (
            prediction_id,
            internal_order_id,
            order_item_id,
            1 if is_late else 0,
            float(late_probability),
            risk_label,
            key_insight,
            recommendations_json
        ))
