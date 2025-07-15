#!/usr/bin/env python3
"""
Working Qloo API Demonstration

This script demonstrates how to properly use the Qloo API for supermarket
layout optimization with the correct parameters and endpoints.
"""

import sys
import os
import json
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from qloo_client import create_qloo_client


def demo_product_search():
    """Demonstrate basic product search functionality."""
    print("🛍️  Product Search Demo")
    print("=" * 40)
    
    client = create_qloo_client()
    
    # Test different product searches
    products_to_search = ["apple", "milk", "bread", "cheese", "yogurt"]
    
    for product in products_to_search:
        try:
            results = client.get_product_recommendations(product, limit=5)
            print(f"\n📦 Search: '{product}'")
            print(f"🔢 Results: {len(results)} items found")
            
            if results:
                print("🎯 Top results:")
                for i, result in enumerate(results[:3], 1):
                    name = result.get('name', 'Unknown')
                    result_id = result.get('id', 'N/A')
                    print(f"   {i}. {name} (ID: {result_id})")
            
        except Exception as e:
            print(f"❌ Error searching for {product}: {e}")


def demo_category_analysis():
    """Demonstrate category analysis for store layout."""
    print("\n🏪 Category Analysis Demo")
    print("=" * 40)
    
    client = create_qloo_client()
    
    # Analyze different store categories
    categories = ["beverages", "dairy", "produce", "snacks", "frozen"]
    
    for category in categories:
        try:
            insights = client.get_category_insights(category)
            print(f"\n📂 Category: {category}")
            print(f"📊 Total results: {insights.get('total_results', 0)}")
            
            top_products = insights.get('top_products', [])
            if top_products:
                print(f"🏆 Top items in {category}:")
                for i, product in enumerate(top_products[:3], 1):
                    name = product.get('name', 'Unknown')
                    score = product.get('relevance_score', 0)
                    print(f"   {i}. {name} (score: {score:.2f})")
            
        except Exception as e:
            print(f"❌ Error analyzing {category}: {e}")


def demo_product_associations():
    """Demonstrate product association analysis."""
    print("\n🔗 Product Association Demo")
    print("=" * 40)
    
    client = create_qloo_client()
    
    # Analyze products that go together
    product_groups = [
        ["milk", "cereal"],
        ["bread", "butter"],
        ["pasta", "sauce"],
    ]
    
    for products in product_groups:
        try:
            associations = client.get_product_associations(products)
            print(f"\n🛒 Product group: {', '.join(products)}")
            
            for product, assocs in associations.items():
                print(f"📦 {product} associations:")
                for i, assoc in enumerate(assocs[:2], 1):
                    related_id = assoc.get('associated_product_id', 'Unknown')
                    strength = assoc.get('association_strength', 0)
                    print(f"   {i}. {related_id} (strength: {strength:.2f})")
            
        except Exception as e:
            print(f"❌ Error analyzing associations for {products}: {e}")


def demo_store_layout_recommendations():
    """Demonstrate how to use API data for store layout optimization."""
    print("\n🏗️  Store Layout Optimization Demo")
    print("=" * 50)
    
    client = create_qloo_client()
    
    # Get data for layout decisions
    print("📋 Gathering data for layout optimization...")
    
    # 1. Popular products in each category
    categories = ["dairy", "produce", "beverages"]
    category_data = {}
    
    for category in categories:
        insights = client.get_category_insights(category)
        category_data[category] = insights
        print(f"✅ Analyzed {category}: {insights.get('total_results', 0)} products")
    
    # 2. Product associations for placement decisions
    key_products = ["milk", "bread", "apple"]
    association_data = client.get_product_associations(key_products)
    print(f"✅ Analyzed associations for {len(key_products)} key products")
    
    # 3. Generate layout recommendations
    print("\n🎯 Layout Recommendations:")
    print("1. High-traffic items (place near entrance):")
    for category, data in category_data.items():
        top_items = data.get('top_products', [])[:2]
        for item in top_items:
            name = item.get('name', 'Unknown')
            print(f"   • {name} from {category}")
    
    print("\n2. Product placement suggestions:")
    for product, assocs in association_data.items():
        if assocs:
            related = assocs[0].get('associated_product_id', 'Unknown')
            print(f"   • Place {product} near {related}")
    
    print("\n3. Category organization:")
    print("   • Dairy section: High association strength with breakfast items")
    print("   • Produce section: Place near entrance for freshness appeal")
    print("   • Beverages: Consider end-cap placement for visibility")


def main():
    """Run all API demonstrations."""
    print("🚀 Qloo API Working Demo")
    print("=" * 50)
    print("This demo shows how to properly use the Qloo API")
    print("for supermarket layout optimization.\n")
    
    try:
        # Test connection first
        client = create_qloo_client()
        if not client.health_check():
            print("❌ API connection failed")
            return
        
        print("✅ API connection successful!\n")
        
        # Run demonstrations
        demo_product_search()
        demo_category_analysis()
        demo_product_associations()
        demo_store_layout_recommendations()
        
        print("\n🎉 Demo completed successfully!")
        print("\n💡 Key takeaways:")
        print("   • Use search endpoint with 'query' parameter")
        print("   • Don't use 'type' parameter (causes 403 errors)")
        print("   • API returns results in 'results' key")
        print("   • All methods now work with real API data")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")


if __name__ == "__main__":
    main() 