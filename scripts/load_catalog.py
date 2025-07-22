#!/usr/bin/env python3
"""
ETL Script: Load Grocery Catalog into SQLite Database

This script imports the grocery catalog CSV file into the SQLite database,
setting up the products table with proper data validation and error handling.
"""

import os
import sys
import sqlite3
import pandas as pd
import logging
from pathlib import Path
from typing import Optional, Tuple

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from database_setup import setup_database


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """
    Set up logging configuration.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)

    Returns:
        Configured logger instance
    """
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger(__name__)


def validate_catalog_data(df: pd.DataFrame) -> Tuple[bool, list]:
    """
    Validate the grocery catalog data structure and content.

    Args:
        df: DataFrame containing catalog data

    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []

    # Check required columns
    required_columns = ["product_id", "product_name", "category"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        errors.append(f"Missing required columns: {missing_columns}")

    if errors:
        return False, errors

    # Check for empty DataFrame
    if df.empty:
        errors.append("Catalog data is empty")
        return False, errors

    # Check for missing values in critical columns
    if df["product_id"].isnull().any():
        errors.append("Found null values in product_id column")

    if df["product_name"].isnull().any():
        errors.append("Found null values in product_name column")

    # Check for duplicate product IDs
    duplicates = df["product_id"].duplicated()
    if duplicates.any():
        duplicate_ids = df.loc[duplicates, "product_id"].tolist()
        errors.append(f"Found duplicate product IDs: {duplicate_ids}")

    # Check data types
    try:
        pd.to_numeric(df["product_id"], errors="raise")
    except (ValueError, TypeError):
        errors.append("product_id column contains non-numeric values")

    # Check for reasonable product name lengths
    long_names = df[df["product_name"].str.len() > 100]
    if not long_names.empty:
        errors.append(
            f"Found {len(long_names)} product names longer than 100 characters"
        )

    return len(errors) == 0, errors


def clean_catalog_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and normalize the catalog data.

    Args:
        df: Raw catalog DataFrame

    Returns:
        Cleaned DataFrame
    """
    logger = logging.getLogger(__name__)
    df_cleaned = df.copy()

    # Convert product_id to integer
    df_cleaned["product_id"] = pd.to_numeric(
        df_cleaned["product_id"], errors="coerce"
    ).astype("Int64")

    # Clean product names
    df_cleaned["product_name"] = df_cleaned["product_name"].str.strip()
    df_cleaned["product_name"] = df_cleaned["product_name"].str.title()

    # Clean category names
    df_cleaned["category"] = df_cleaned["category"].fillna("Uncategorized")
    df_cleaned["category"] = df_cleaned["category"].str.strip()
    df_cleaned["category"] = df_cleaned["category"].str.title()

    # Remove any rows with null product_id or product_name after cleaning
    initial_count = len(df_cleaned)
    df_cleaned = df_cleaned.dropna(subset=["product_id", "product_name"])
    final_count = len(df_cleaned)

    if initial_count != final_count:
        logger.warning(
            f"Removed {initial_count - final_count} rows with missing critical data"
        )

    return df_cleaned


def load_catalog_to_database(
    csv_path: str, db_path: str, replace_existing: bool = True
) -> int:
    """
    Load grocery catalog from CSV into SQLite database.

    Args:
        csv_path: Path to the grocery catalog CSV file
        db_path: Path to the SQLite database file
        replace_existing: Whether to replace existing data (default: True)

    Returns:
        Number of products loaded

    Raises:
        FileNotFoundError: If CSV file doesn't exist
        sqlite3.Error: If database operations fail
        ValueError: If data validation fails
    """
    logger = logging.getLogger(__name__)

    # Validate file paths
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Catalog CSV file not found: {csv_path}")

    # Ensure database directory exists
    db_dir = os.path.dirname(db_path)
    if db_dir and not os.path.exists(db_dir):
        os.makedirs(db_dir)
        logger.info(f"Created database directory: {db_dir}")

    # Read and validate catalog data
    logger.info(f"Reading catalog from: {csv_path}")
    try:
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(df)} rows from CSV")
    except Exception as e:
        raise ValueError(f"Failed to read CSV file: {e}")

    # Validate data structure
    is_valid, errors = validate_catalog_data(df)
    if not is_valid:
        error_msg = "Data validation failed:\n" + "\n".join(
            f"- {error}" for error in errors
        )
        raise ValueError(error_msg)

    # Clean the data
    df_cleaned = clean_catalog_data(df)
    logger.info(f"Data cleaning complete. {len(df_cleaned)} products ready for import")

    # Setup database if needed
    logger.info("Setting up database schema...")
    setup_database()

    # Connect to database and load data
    logger.info(f"Connecting to database: {db_path}")
    conn = sqlite3.connect(db_path)

    try:
        cursor = conn.cursor()

        # Clear existing data if requested
        if replace_existing:
            cursor.execute("DELETE FROM products")
            logger.info("Cleared existing products data")

        # Insert products
        products_loaded = 0
        for _, row in df_cleaned.iterrows():
            try:
                cursor.execute(
                    "INSERT OR REPLACE INTO products (product_id, product_name, category) VALUES (?, ?, ?)",
                    (int(row["product_id"]), row["product_name"], row["category"]),
                )
                products_loaded += 1
            except Exception as e:
                logger.warning(f"Failed to insert product {row['product_id']}: {e}")

        conn.commit()
        logger.info(f"Successfully loaded {products_loaded} products into database")

        # Verify the data was loaded
        cursor.execute("SELECT COUNT(*) FROM products")
        total_count = cursor.fetchone()[0]
        logger.info(f"Database now contains {total_count} products total")

        # Show sample of loaded data
        cursor.execute("SELECT * FROM products LIMIT 5")
        sample_products = cursor.fetchall()
        logger.info("Sample of loaded products:")
        for product in sample_products:
            logger.info(
                f"  ID: {product[0]}, Name: {product[1]}, Category: {product[2]}"
            )

        return products_loaded

    except sqlite3.Error as e:
        conn.rollback()
        raise sqlite3.Error(f"Database operation failed: {e}")
    finally:
        conn.close()


def get_catalog_statistics(db_path: str) -> dict:
    """
    Get statistics about the loaded catalog data.

    Args:
        db_path: Path to the SQLite database

    Returns:
        Dictionary with catalog statistics
    """
    logger = logging.getLogger(__name__)

    if not os.path.exists(db_path):
        return {"error": "Database file not found"}

    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.cursor()

        # Total products
        cursor.execute("SELECT COUNT(*) FROM products")
        total_products = cursor.fetchone()[0]

        # Products by category
        cursor.execute(
            """
            SELECT category, COUNT(*) as count 
            FROM products 
            GROUP BY category 
            ORDER BY count DESC
        """
        )
        categories = dict(cursor.fetchall())

        # Sample products
        cursor.execute("SELECT product_name FROM products LIMIT 10")
        sample_names = [row[0] for row in cursor.fetchall()]

        stats = {
            "total_products": total_products,
            "categories": categories,
            "sample_products": sample_names,
        }

        logger.info(
            f"Catalog statistics: {total_products} products across {len(categories)} categories"
        )
        return stats

    except sqlite3.Error as e:
        logger.error(f"Failed to get statistics: {e}")
        return {"error": str(e)}
    finally:
        conn.close()


def main():
    """Main entry point for the ETL script."""
    logger = setup_logging()

    # Default paths (can be overridden via command line arguments)
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    csv_path = project_root / "data" / "grocery_catalog.csv"
    db_path = project_root / "data" / "associations.db"

    # Handle command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] in ["-h", "--help"]:
            print(
                f"""
Usage: python {sys.argv[0]} [csv_path] [db_path]

Load grocery catalog from CSV into SQLite database.

Arguments:
    csv_path    Path to CSV file (default: {csv_path})
    db_path     Path to database file (default: {db_path})

Options:
    -h, --help  Show this help message
    --stats     Show catalog statistics only (don't reload)
    --validate  Validate CSV file only (don't load)

Examples:
    python {sys.argv[0]}
    python {sys.argv[0]} --stats
    python {sys.argv[0]} --validate
    python {sys.argv[0]} my_catalog.csv my_db.db
"""
            )
            return

        elif sys.argv[1] == "--stats":
            # Show statistics only
            logger.info("Retrieving catalog statistics...")
            stats = get_catalog_statistics(str(db_path))

            if "error" in stats:
                logger.error(f"Error getting statistics: {stats['error']}")
                sys.exit(1)

            print(f"\n📊 Catalog Statistics:")
            print(f"Total Products: {stats['total_products']}")
            print(f"\nProducts by Category:")
            for category, count in stats["categories"].items():
                print(f"  {category}: {count}")
            print(f"\nSample Products: {', '.join(stats['sample_products'][:5])}")
            return

        elif sys.argv[1] == "--validate":
            # Validate only
            logger.info("Validating catalog data...")
            try:
                df = pd.read_csv(csv_path)
                is_valid, errors = validate_catalog_data(df)
                if is_valid:
                    logger.info("✅ Catalog data validation passed")
                    print("✅ Catalog data is valid")
                else:
                    logger.error("❌ Catalog data validation failed")
                    print("❌ Validation errors:")
                    for error in errors:
                        print(f"  - {error}")
                    sys.exit(1)
            except Exception as e:
                logger.error(f"Validation failed: {e}")
                sys.exit(1)
            return

        else:
            # Custom paths provided
            if len(sys.argv) >= 2:
                csv_path = Path(sys.argv[1])
            if len(sys.argv) >= 3:
                db_path = Path(sys.argv[2])

    try:
        logger.info("🚀 Starting catalog ETL process...")
        logger.info(f"Source CSV: {csv_path}")
        logger.info(f"Target DB: {db_path}")

        # Load catalog
        products_loaded = load_catalog_to_database(str(csv_path), str(db_path))

        # Show final statistics
        stats = get_catalog_statistics(str(db_path))

        logger.info("✅ ETL process completed successfully!")
        print(f"\n🎉 Success! Loaded {products_loaded} products into database")
        print(
            f"📊 Total products in database: {stats.get('total_products', 'Unknown')}"
        )
        print(f"📂 Categories: {', '.join(stats.get('categories', {}).keys())}")

    except Exception as e:
        logger.error(f"❌ ETL process failed: {e}")
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
