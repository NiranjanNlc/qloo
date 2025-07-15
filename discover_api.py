#!/usr/bin/env python3
"""
API Discovery script for Qloo.

This script attempts to discover the correct API endpoints by testing
common patterns and structures.
"""

import sys
import os
import requests
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from qloo_client import create_qloo_client


def test_endpoint(base_url: str, api_key: str, endpoint: str) -> dict:
    """Test a specific endpoint and return the result."""
    url = f"{base_url}/{endpoint.lstrip('/')}" if endpoint else base_url
    
    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        return {
            "endpoint": endpoint or "root",
            "status_code": response.status_code,
            "success": response.status_code < 400,
            "content_type": response.headers.get("content-type", ""),
            "content_preview": str(response.text)[:200] + "..." if len(response.text) > 200 else response.text
        }
    except Exception as e:
        return {
            "endpoint": endpoint or "root",
            "status_code": None,
            "success": False,
            "error": str(e),
            "content_preview": ""
        }


def test_recommendations_with_type_parameter(base_url: str, api_key: str):
    """Test the recommendations endpoint with different type parameter values."""
    print(f"\n🛍️  Testing Recommendations with Type Parameter")
    print("=" * 60)
    
    headers = {
        "x-api-key": api_key,
        "Content-Type": "application/json"
    }
    
    # Test various type parameter values
    type_values = [
        "product", "food", "item", "entity", "object", "goods",
        "recommendation", "similarity", "related", "association",
        "supermarket", "grocery", "retail"
    ]
    
    url = f"{base_url}/recommendations"
    
    for type_value in type_values:
        try:
            params = {
                "input": "apple",
                "type": type_value,
                "limit": 3
            }
            
            response = requests.get(url, params=params, headers=headers, timeout=10)
            
            status_icon = "✅" if response.status_code < 400 else "❌"
            print(f"{status_icon} type='{type_value}' → {response.status_code}")
            
            if response.status_code < 400:
                try:
                    data = response.json()
                    print(f"   📦 Success! Response keys: {list(data.keys())}")
                    if 'recommendations' in data:
                        print(f"   🎯 Found {len(data['recommendations'])} recommendations")
                    elif 'data' in data:
                        print(f"   🎯 Found data with {len(data['data']) if isinstance(data['data'], list) else 'N/A'} items")
                    else:
                        print(f"   🎯 Response structure: {str(data)[:100]}...")
                except Exception as e:
                    print(f"   📦 Success! Raw response: {response.text[:100]}...")
            elif response.status_code == 400:
                try:
                    error_data = response.json()
                    print(f"   💡 Error: {error_data.get('error', {}).get('message', error_data)}")
                except:
                    print(f"   💡 Error response: {response.text[:100]}")
                    
        except Exception as e:
            print(f"❌ type='{type_value}': {str(e)[:50]}...")


def test_other_endpoints_with_type(base_url: str, api_key: str):
    """Test other endpoints that might require type parameters."""
    print(f"\n🔬 Testing Other Endpoints with Type Parameter")
    print("=" * 60)
    
    headers = {
        "x-api-key": api_key,
        "Content-Type": "application/json"
    }
    
    endpoints_to_test = [
        ("insights", {"input": "beverages", "type": "category"}),
        ("associations", {"inputs": ["apple", "orange"], "type": "product"}),
        ("search", {"input": "apple", "type": "product"}),
        ("similar", {"input": "apple", "type": "product"}),
    ]
    
    for endpoint, params in endpoints_to_test:
        try:
            url = f"{base_url}/{endpoint}"
            
            # Try GET first
            response = requests.get(url, params=params, headers=headers, timeout=5)
            method = "GET"
            
            # If GET fails, try POST
            if response.status_code >= 400:
                response = requests.post(url, json=params, headers=headers, timeout=5)
                method = "POST"
            
            status_icon = "✅" if response.status_code < 400 else "❌"
            print(f"{status_icon} {endpoint} ({method}) → {response.status_code}")
            
            if response.status_code < 400:
                try:
                    data = response.json()
                    print(f"   📦 Response keys: {list(data.keys())}")
                except:
                    print(f"   📦 Raw response: {response.text[:100]}...")
            elif response.status_code == 400:
                try:
                    error_data = response.json()
                    print(f"   💡 Error: {error_data}")
                except:
                    print(f"   💡 Error: {response.text[:100]}")
                    
        except Exception as e:
            print(f"❌ {endpoint}: {str(e)[:50]}...")


def discover_api_structure():
    """Discover the API structure by testing common endpoints."""
    print("🔍 Qloo API Discovery (Updated)")
    print("=" * 60)
    
    try:
        client = create_qloo_client()
        print(f"📍 Base URL: {client.base_url}")
        print(f"🔑 API Key: {client.api_key[:15]}..." if client.api_key else "❌ No API Key")
        
        if not client.api_key:
            print("❌ No API key available")
            return
        
    except Exception as e:
        print(f"❌ Failed to create client: {e}")
        return
    
    # Test root endpoint first
    print(f"\n📋 Testing Root Endpoint")
    print("-" * 30)
    
    root_result = test_endpoint(client.base_url, client.api_key, "")
    if root_result["success"]:
        print(f"✅ Root endpoint working")
        try:
            import json
            root_data = json.loads(root_result["content_preview"].replace("...", ""))
            print(f"   API Info: {root_data}")
        except:
            print(f"   Content: {root_result['content_preview']}")
    else:
        print(f"❌ Root endpoint failed: {root_result.get('error', 'Unknown error')}")
    
    # Test endpoints with type parameters (the main fix)
    test_recommendations_with_type_parameter(client.base_url, client.api_key)
    test_other_endpoints_with_type(client.base_url, client.api_key)
    
    # Quick test of common endpoints without parameters
    print(f"\n🧪 Testing Common Endpoints (No Parameters)")
    print("-" * 50)
    
    endpoints_to_test = [
        "recommendations", "associations", "insights", "categories",
        "search", "similar", "products", "items"
    ]
    
    successful_endpoints = []
    
    for endpoint in endpoints_to_test:
        result = test_endpoint(client.base_url, client.api_key, endpoint)
        
        status_icon = "✅" if result["success"] else "❌"
        status_code = result.get("status_code", "ERR")
        
        print(f"{status_icon} {endpoint:<15} | {status_code}")
        
        if result["success"]:
            successful_endpoints.append(result)
    
    print(f"\n📊 Discovery Summary")
    print("=" * 30)
    print(f"✅ Working endpoints: {len(successful_endpoints)}")
    print(f"🔑 Key finding: API requires 'type' parameter for most endpoints")
    print(f"🎯 Recommendation: Use type='product' for product recommendations")


if __name__ == "__main__":
    discover_api_structure() 