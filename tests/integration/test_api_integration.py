"""Integration tests for API endpoints."""

import pytest
import requests
import time
from typing import Dict, Any


class TestAPIIntegration:
    """Test API integration with external services."""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test environment."""
        self.base_url = "http://localhost:8000"
        self.timeout = 30
        
        # Wait for API to be ready
        max_retries = 10
        for i in range(max_retries):
            try:
                response = requests.get(f"{self.base_url}/health", timeout=5)
                if response.status_code == 200:
                    break
            except requests.RequestException:
                if i == max_retries - 1:
                    pytest.fail("API not ready after maximum retries")
                time.sleep(3)
    
    def test_health_endpoint(self):
        """Test health endpoint returns 200."""
        response = requests.get(f"{self.base_url}/health")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert data["status"] == "healthy"
    
    def test_combos_endpoint_returns_data(self):
        """Test combos endpoint returns valid data."""
        response = requests.get(f"{self.base_url}/combos?limit=5")
        assert response.status_code == 200
        
        data = response.json()
        assert "combos" in data
        assert "total" in data
        assert "page" in data
        assert isinstance(data["combos"], list)
        assert len(data["combos"]) <= 5
    
    def test_combos_endpoint_with_filters(self):
        """Test combos endpoint with various filters."""
        # Test with confidence filter
        response = requests.get(f"{self.base_url}/combos?min_confidence=0.8&limit=3")
        assert response.status_code == 200
        
        data = response.json()
        combos = data["combos"]
        for combo in combos:
            assert combo["confidence"] >= 0.8
    
    def test_associations_endpoint(self):
        """Test associations endpoint."""
        # First get a product to test with
        response = requests.get(f"{self.base_url}/products?query=milk&limit=1")
        assert response.status_code == 200
        
        products = response.json().get("products", [])
        if not products:
            pytest.skip("No products available for association testing")
        
        product_id = products[0].get("id") or "milk"
        
        # Test associations for this product
        response = requests.get(f"{self.base_url}/associations/{product_id}")
        assert response.status_code == 200
        
        data = response.json()
        assert "associations" in data
        assert isinstance(data["associations"], list)
    
    def test_products_search_endpoint(self):
        """Test products search endpoint."""
        response = requests.get(f"{self.base_url}/products?query=apple&limit=5")
        assert response.status_code == 200
        
        data = response.json()
        assert "products" in data
        assert isinstance(data["products"], list)
    
    def test_api_error_handling(self):
        """Test API error handling."""
        # Test invalid endpoint
        response = requests.get(f"{self.base_url}/invalid-endpoint")
        assert response.status_code == 404
        
        # Test invalid parameters
        response = requests.get(f"{self.base_url}/combos?limit=invalid")
        assert response.status_code == 422  # Validation error
    
    def test_api_performance(self):
        """Test API response times."""
        endpoints = [
            "/health",
            "/combos?limit=10",
            "/products?query=test&limit=5"
        ]
        
        for endpoint in endpoints:
            start_time = time.time()
            response = requests.get(f"{self.base_url}{endpoint}")
            response_time = time.time() - start_time
            
            assert response.status_code == 200
            assert response_time < 5.0  # Should respond within 5 seconds
    
    def test_concurrent_requests(self):
        """Test API handles concurrent requests."""
        import concurrent.futures
        import threading
        
        def make_request():
            response = requests.get(f"{self.base_url}/combos?limit=5")
            return response.status_code == 200
        
        # Make 10 concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_request) for _ in range(10)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # All requests should succeed
        assert all(results)


class TestDatabaseIntegration:
    """Test database integration."""
    
    def test_database_connection(self):
        """Test database connection through API."""
        response = requests.get("http://localhost:8000/health")
        assert response.status_code == 200
        
        data = response.json()
        # Health endpoint should verify database connectivity
        assert data.get("database", {}).get("status") == "connected"
    
    def test_data_persistence(self):
        """Test data persistence across requests."""
        # This test assumes that combo data is consistent
        response1 = requests.get("http://localhost:8000/combos?limit=5")
        response2 = requests.get("http://localhost:8000/combos?limit=5")
        
        assert response1.status_code == 200
        assert response2.status_code == 200
        
        # Should return same data for same query
        data1 = response1.json()
        data2 = response2.json()
        assert data1["total"] == data2["total"]


class TestExternalAPIIntegration:
    """Test integration with external APIs."""
    
    def test_qloo_api_integration(self):
        """Test Qloo API integration through our API."""
        # Test that our API can successfully call Qloo API
        response = requests.get("http://localhost:8000/products?query=apple&limit=3")
        assert response.status_code == 200
        
        data = response.json()
        products = data.get("products", [])
        
        # Should return products from Qloo API
        assert len(products) > 0
        
        # Verify product structure
        for product in products:
            assert "name" in product or "id" in product
    
    @pytest.mark.slow
    def test_qloo_api_timeout_handling(self):
        """Test handling of Qloo API timeouts."""
        # This test would need to simulate timeout conditions
        # For now, just test that the endpoint handles errors gracefully
        response = requests.get("http://localhost:8000/products?query=nonexistent_product_xyz123")
        assert response.status_code in [200, 404, 500]  # Should handle gracefully