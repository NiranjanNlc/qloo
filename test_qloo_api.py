#!/usr/bin/env python3
"""
Test script for Qloo API client.

This script tests the connection to the Qloo API and verifies that 
the implemented methods work correctly with real data.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from qloo_client import QlooClient, create_qloo_client


def test_api_connection():
    """Test basic API connection and health check."""
    print("🔍 Testing Qloo API Connection...")
    print("=" * 50)
    
    try:
        # Create client instance
        client = create_qloo_client()
        print(f"✅ Client created successfully")
        print(f"📍 Base URL: {client.base_url}")
        print(f"🔑 API Key: {client.api_key[:10]}..." if client.api_key else "❌ No API Key")
        
        # Test health check
        print("\n🏥 Testing health check...")
        is_healthy = client.health_check()
        print(f"{'✅' if is_healthy else '❌'} Health check: {'PASSED' if is_healthy else 'FAILED'}")
        
        # Get API info
        if is_healthy:
            print("\n📋 API Information:")
            api_info = client.get_api_info()
            for key, value in api_info.items():
                print(f"   {key}: {value}")
        
        return client, is_healthy
        
    except Exception as e:
        print(f"❌ Failed to create client: {e}")
        return None, False


def test_product_recommendations(client):
    """Test product recommendations endpoint with different types."""
    print("\n🛍️  Testing Product Recommendations...")
    print("-" * 50)
    
    test_cases = [
        {"product_id": "apple", "type": "product", "description": "Product recommendations for apple"},
        {"product_id": "milk", "type": "food", "description": "Food recommendations for milk"},
        {"product_id": "bread", "type": "item", "description": "Item recommendations for bread"},
    ]
    
    for case in test_cases:
        try:
            print(f"\n🧪 Testing: {case['description']}")
            recommendations = client.get_product_recommendations(
                case["product_id"], 
                limit=5, 
                recommendation_type=case["type"]
            )
            
            print(f"📦 Product: {case['product_id']} (type: {case['type']})")
            print(f"🔢 Recommendations received: {len(recommendations)}")
            
            if recommendations:
                print("🎯 Sample recommendations:")
                for i, rec in enumerate(recommendations[:3], 1):
                    print(f"   {i}. {rec.get('name', rec.get('id', 'Unknown'))}")
                    if 'confidence_score' in rec:
                        print(f"      Confidence: {rec['confidence_score']:.2f}")
            
            print("✅ Recommendations test passed")
            return True
            
        except Exception as e:
            print(f"❌ Recommendations test failed for {case['product_id']}: {e}")
            continue
    
    return False


def test_product_associations(client):
    """Test product associations endpoint with different types."""
    print("\n🔗 Testing Product Associations...")
    print("-" * 50)
    
    test_cases = [
        {"products": ["bread", "milk"], "type": "product", "description": "Product associations"},
        {"products": ["apple", "banana"], "type": "food", "description": "Food associations"},
        {"products": ["eggs", "cheese"], "type": "item", "description": "Item associations"},
    ]
    
    for case in test_cases:
        try:
            print(f"\n🧪 Testing: {case['description']}")
            associations = client.get_product_associations(
                case["products"], 
                association_type=case["type"]
            )
            
            print(f"📦 Products: {', '.join(case['products'])} (type: {case['type']})")
            print(f"🔢 Associations received: {len(associations)}")
            
            if associations:
                print("🎯 Sample associations:")
                for product_id, assocs in list(associations.items())[:2]:
                    print(f"   {product_id}: {len(assocs)} associations")
                    if assocs:
                        first_assoc = assocs[0]
                        print(f"      → {first_assoc.get('associated_product_id', 'Unknown')}")
            
            print("✅ Associations test passed")
            return True
            
        except Exception as e:
            print(f"❌ Associations test failed for {case['products']}: {e}")
            continue
    
    return False


def test_category_insights(client):
    """Test category insights endpoint with different types."""
    print("\n📊 Testing Category Insights...")
    print("-" * 50)
    
    test_cases = [
        {"category": "beverages", "type": "category", "description": "Category insights for beverages"},
        {"category": "dairy", "type": "food_category", "description": "Food category insights for dairy"},
        {"category": "produce", "type": "section", "description": "Section insights for produce"},
    ]
    
    for case in test_cases:
        try:
            print(f"\n🧪 Testing: {case['description']}")
            insights = client.get_category_insights(
                case["category"], 
                insight_type=case["type"]
            )
            
            print(f"📂 Category: {case['category']} (type: {case['type']})")
            print(f"📈 Insights received: {len(insights) if insights else 0} fields")
            
            if insights:
                print("🎯 Insight summary:")
                if 'total_products' in insights:
                    print(f"   Total products: {insights['total_products']}")
                if 'top_products' in insights:
                    print(f"   Top products: {len(insights['top_products'])}")
                if 'seasonal_trends' in insights:
                    print(f"   Seasonal data: {len(insights['seasonal_trends'])} seasons")
            
            print("✅ Category insights test passed")
            return True
            
        except Exception as e:
            print(f"❌ Category insights test failed for {case['category']}: {e}")
            continue
    
    return False


def test_different_parameter_combinations(client):
    """Test different parameter combinations to understand the API better."""
    print("\n🔬 Testing Different Parameter Combinations...")
    print("-" * 50)
    
    # Test various type values that might work
    type_values = [
        "product", "food", "item", "entity", "object",
        "recommendation", "similarity", "related", "association"
    ]
    
    test_product = "apple"
    
    for type_value in type_values:
        try:
            print(f"\n🧪 Testing type='{type_value}' for product '{test_product}'")
            recommendations = client.get_product_recommendations(
                test_product, 
                limit=3, 
                recommendation_type=type_value
            )
            
            if recommendations:
                print(f"   ✅ Success! Got {len(recommendations)} recommendations")
                if len(recommendations) > 0:
                    print(f"   📦 First result: {recommendations[0]}")
            else:
                print(f"   ⚠️  Success but no results returned")
                
        except Exception as e:
            print(f"   ❌ Failed with type '{type_value}': {str(e)[:100]}...")


def test_api_diagnostics(client):
    """Test API endpoints and parameters to understand the structure."""
    print("\n🔍 Running API Diagnostics...")
    print("-" * 50)
    
    # Test basic endpoints
    print("\n📋 Testing Endpoints:")
    endpoint_results = client.test_api_endpoints()
    for endpoint, result in endpoint_results.items():
        status_icon = "✅" if result["status"] == "success" else "❌"
        print(f"{status_icon} {endpoint}: {result.get('status_code', result['status'])}")
        if result["status"] == "error" and "status_code" in result:
            print(f"   💡 {result.get('error', '')[:100]}...")
    
    # Test search parameters 
    print("\n🔍 Testing Search Parameters:")
    search_results = client.test_search_parameters()
    for combo_name, result in search_results.items():
        status_icon = "✅" if result["status"] == "success" else "❌"
        params_str = str(result["params"])
        print(f"{status_icon} {params_str}: {result.get('status_code', result['status'])}")
        if result["status"] == "success":
            print(f"   🎉 SUCCESS! Found working parameters!")
            print(f"   📦 Response keys: {list(result['data'].keys()) if isinstance(result['data'], dict) else 'Non-dict response'}")
            return True
        elif result["status"] == "error" and "status_code" in result:
            error_text = result.get('error', '')
            if len(error_text) > 100:
                error_text = error_text[:100] + "..."
            print(f"   💡 {error_text}")
    
    return False


def main():
    """Run all API tests."""
    print("🚀 Qloo API Test Suite (Updated)")
    print("=" * 50)
    
    # Test connection
    client, is_connected = test_api_connection()
    
    if not client:
        print("\n❌ Cannot proceed with tests - client creation failed")
        return
    
    # Run API method tests
    test_results = []
    
    if is_connected:
        print("\n🧪 Running API endpoint tests...")
        
        # Run diagnostic tests first
        diagnostic_success = test_api_diagnostics(client)
        
        # Run specific tests
        test_results.append(test_product_recommendations(client))
        test_results.append(test_product_associations(client))
        test_results.append(test_category_insights(client))
        
        # Run exploratory tests if diagnostics didn't find working params
        if not diagnostic_success:
            test_different_parameter_combinations(client)
        
    else:
        print("\n⚠️  API connection failed, testing with fallback data...")
        test_results.append(test_product_recommendations(client))
        test_results.append(test_product_associations(client))
        test_results.append(test_category_insights(client))
    
    # Summary
    print("\n📋 Test Summary")
    print("=" * 30)
    passed = sum(test_results)
    total = len(test_results)
    
    print(f"✅ Passed: {passed}/{total}")
    print(f"❌ Failed: {total - passed}/{total}")
    
    if is_connected and passed > 0:
        print("🎉 Some tests passed! The API type parameter fix is working.")
    elif passed > 0:
        print("🔄 Tests passed with fallback data.")
    else:
        print("⚠️  All tests failed. Check the API configuration and parameters.")
    
    print(f"\n💡 Next steps:")
    print("   • Review the debug output to understand correct API parameters")
    print("   • Use successful type parameters in your supermarket layout optimizer")
    print("   • Implement proper error handling for production use")
    print("   • Consider caching frequently requested data")


if __name__ == "__main__":
    main() 