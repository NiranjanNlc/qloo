import sqlite3
import os


def setup_database():
    """
    Sets up the associations.db database.
    Creates the products and associations tables if they don't exist.
    """
    # Ensure the data directory exists
    if not os.path.exists("data"):
        os.makedirs("data")

    conn = sqlite3.connect("data/associations.db")
    cursor = conn.cursor()

    # Create a table to store products from the catalog
    # This can be populated from the grocery_catalog.csv
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS products (
            product_id INTEGER PRIMARY KEY,
            product_name TEXT NOT NULL,
            category TEXT
        )
    """
    )

    # Create a table to store the discovered association rules
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS associations (
            rule_id INTEGER PRIMARY KEY AUTOINCREMENT,
            antecedent_product_id INTEGER,
            consequent_product_id INTEGER,
            confidence REAL NOT NULL,
            support REAL NOT NULL,
            lift REAL NOT NULL,
            FOREIGN KEY (antecedent_product_id) REFERENCES products(product_id),
            FOREIGN KEY (consequent_product_id) REFERENCES products(product_id)
        )
    """
    )

    print(
        "Database 'associations.db' with 'products' and 'associations' tables set up successfully."
    )

    conn.commit()
    conn.close()


if __name__ == "__main__":
    setup_database()
