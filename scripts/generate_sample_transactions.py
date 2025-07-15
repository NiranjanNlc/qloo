#!/usr/bin/env python3
"""
Generate Sample Transaction Data

This script generates realistic sample transaction data for testing
the association engine and demonstrating the supermarket layout optimizer.
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
from typing import List, Dict, Set


def load_product_catalog() -> pd.DataFrame:
    """Load the product catalog."""
    return pd.read_csv('data/grocery_catalog.csv')


def create_shopping_patterns() -> Dict[str, Dict]:
    """
    Define realistic shopping patterns and product associations.
    
    Returns:
        Dictionary of shopping patterns with product groups and probabilities.
    """
    return {
        'breakfast_combo': {
            'products': [2, 3, 4, 14],  # Bread, Milk, Eggs, Orange Juice
            'probability': 0.25,
            'min_items': 2,
            'max_items': 4
        },
        'mexican_night': {
            'products': [8, 10, 11],  # Ground Beef, Salsa, Tortilla Chips
            'probability': 0.20,
            'min_items': 2,
            'max_items': 3
        },
        'healthy_snack': {
            'products': [1, 5, 13],  # Bananas, Avocado, Apples
            'probability': 0.30,
            'min_items': 1,
            'max_items': 3
        },
        'party_snacks': {
            'products': [6, 7, 11],  # Potato Chips, Cola, Tortilla Chips
            'probability': 0.15,
            'min_items': 2,
            'max_items': 3
        },
        'dairy_run': {
            'products': [3, 4, 9, 15],  # Milk, Eggs, Cheese, Yogurt
            'probability': 0.35,
            'min_items': 2,
            'max_items': 4
        },
        'meat_prep': {
            'products': [8, 12],  # Ground Beef, Chicken Breast
            'probability': 0.20,
            'min_items': 1,
            'max_items': 2
        },
        'sandwich_making': {
            'products': [2, 3, 9],  # Bread, Milk, Cheese
            'probability': 0.25,
            'min_items': 2,
            'max_items': 3
        }
    }


def generate_base_transaction(customer_type: str, patterns: Dict) -> Set[int]:
    """
    Generate a base transaction based on customer type and patterns.
    
    Args:
        customer_type: Type of customer (affects shopping behavior)
        patterns: Shopping patterns dictionary
        
    Returns:
        Set of product IDs for the transaction
    """
    transaction = set()
    
    # Apply patterns based on probabilities
    for pattern_name, pattern_info in patterns.items():
        if random.random() < pattern_info['probability']:
            # Decide how many items from this pattern to include
            num_items = random.randint(pattern_info['min_items'], pattern_info['max_items'])
            selected_products = random.sample(pattern_info['products'], 
                                            min(num_items, len(pattern_info['products'])))
            transaction.update(selected_products)
    
    # Add some random individual items
    all_products = list(range(1, 16))  # Product IDs 1-15
    num_random = random.randint(0, 3)
    random_products = random.sample(all_products, num_random)
    transaction.update(random_products)
    
    return transaction


def generate_customer_types() -> Dict[str, Dict]:
    """Define different customer types and their behaviors."""
    return {
        'family_shopper': {
            'avg_items': 8,
            'pattern_multiplier': 1.5,
            'probability': 0.40
        },
        'single_shopper': {
            'avg_items': 4,
            'pattern_multiplier': 0.8,
            'probability': 0.30
        },
        'bulk_shopper': {
            'avg_items': 12,
            'pattern_multiplier': 2.0,
            'probability': 0.20
        },
        'convenience_shopper': {
            'avg_items': 3,
            'pattern_multiplier': 0.6,
            'probability': 0.10
        }
    }


def generate_transactions(num_transactions: int = 1000, 
                         start_date: datetime = None,
                         end_date: datetime = None) -> pd.DataFrame:
    """
    Generate sample transaction data.
    
    Args:
        num_transactions: Number of transactions to generate
        start_date: Start date for transactions (default: 30 days ago)
        end_date: End date for transactions (default: today)
        
    Returns:
        DataFrame with transaction data
    """
    if start_date is None:
        start_date = datetime.now() - timedelta(days=30)
    if end_date is None:
        end_date = datetime.now()
    
    patterns = create_shopping_patterns()
    customer_types = generate_customer_types()
    
    transactions_data = []
    
    for transaction_id in range(1, num_transactions + 1):
        # Determine customer type
        customer_type = random.choices(
            list(customer_types.keys()),
            weights=[ct['probability'] for ct in customer_types.values()],
            k=1
        )[0]
        
        # Generate transaction timestamp
        time_delta = end_date - start_date
        random_seconds = random.randint(0, int(time_delta.total_seconds()))
        transaction_time = start_date + timedelta(seconds=random_seconds)
        
        # Generate base transaction
        transaction_products = generate_base_transaction(customer_type, patterns)
        
        # Adjust based on customer type
        customer_info = customer_types[customer_type]
        pattern_mult = customer_info['pattern_multiplier']
        
        # Modify patterns based on customer type
        if pattern_mult > 1.0:
            # Add more items for bulk shoppers
            additional_items = random.randint(0, int(pattern_mult * 2))
            all_products = list(range(1, 16))
            extra_products = random.sample(all_products, 
                                         min(additional_items, len(all_products)))
            transaction_products.update(extra_products)
        elif pattern_mult < 1.0:
            # Remove some items for convenience shoppers
            if len(transaction_products) > 2:
                items_to_remove = random.randint(0, len(transaction_products) // 2)
                products_list = list(transaction_products)
                items_to_remove = random.sample(products_list, items_to_remove)
                transaction_products -= set(items_to_remove)
        
        # Ensure minimum transaction size
        if len(transaction_products) == 0:
            transaction_products.add(random.randint(1, 15))
        
        # Add each product in the transaction as a separate row
        for product_id in transaction_products:
            transactions_data.append({
                'transaction_id': transaction_id,
                'customer_type': customer_type,
                'product_id': product_id,
                'timestamp': transaction_time,
                'day_of_week': transaction_time.strftime('%A'),
                'hour': transaction_time.hour
            })
    
    return pd.DataFrame(transactions_data)


def save_transaction_data(transactions_df: pd.DataFrame, filename: str = 'data/sample_transactions.csv'):
    """Save transaction data to CSV file."""
    transactions_df.to_csv(filename, index=False)
    print(f"Saved {len(transactions_df)} transaction records to {filename}")
    
    # Print some statistics
    unique_transactions = transactions_df['transaction_id'].nunique()
    unique_products = transactions_df['product_id'].nunique()
    avg_items_per_transaction = len(transactions_df) / unique_transactions
    
    print(f"Statistics:")
    print(f"  - Unique transactions: {unique_transactions}")
    print(f"  - Unique products: {unique_products}")
    print(f"  - Average items per transaction: {avg_items_per_transaction:.2f}")
    print(f"  - Date range: {transactions_df['timestamp'].min()} to {transactions_df['timestamp'].max()}")


def main():
    """Main function to generate and save sample transaction data."""
    print("Generating sample transaction data...")
    
    # Generate transactions
    transactions_df = generate_transactions(num_transactions=1000)
    
    # Save to file
    save_transaction_data(transactions_df)
    
    # Show sample data
    print("\nSample transactions:")
    print(transactions_df.head(20))
    
    # Show transaction size distribution
    transaction_sizes = transactions_df.groupby('transaction_id').size()
    print(f"\nTransaction size distribution:")
    print(f"  - Min items: {transaction_sizes.min()}")
    print(f"  - Max items: {transaction_sizes.max()}")
    print(f"  - Mean items: {transaction_sizes.mean():.2f}")
    print(f"  - Median items: {transaction_sizes.median():.2f}")
    
    # Show customer type distribution
    customer_dist = transactions_df.groupby('customer_type')['transaction_id'].nunique()
    print(f"\nCustomer type distribution:")
    for customer_type, count in customer_dist.items():
        print(f"  - {customer_type}: {count} transactions")


if __name__ == "__main__":
    main() 