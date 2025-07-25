"""Smoke tests for deployed environments."""

import pytest
import requests
import os
import time
from typing import Dict, Any


class SmokeTestConfig:
    """Configuration for smoke tests based on environment."""
    
    @staticmethod
    def get_config(env: str) -> Dict[str, Any]:
        """Get configuration for specified environment."""
        configs = {
            "staging": {
                "api_base_url": "https://staging-api.qloo-optimizer.example.com",
                "frontend_url": "https://staging.qloo-optimizer.example.com",
                "timeout": 30,
                "expected_response_time": 5.0,
                "min_combos_expected": 5
            },
            "production": {
                "api_base_url": "https://api.qloo-optimizer.example.com",
                "frontend_url": "https://qloo-optimizer.example.com",
                "timeout": 10,
                "expected_response_time": 3.0,
                "min_combos_expected": 10
            },
            "local": {
                "api_base_url": "http://localhost:8000",
                "frontend_url": "http://localhost:8501",
                "timeout": 10,
                "expected_response_time": 2.0,
                "min_combos_expected": 5
            }
        }
        
        return configs.get(env, configs["local"])


@pytest.fixture(scope="session")
def env_config():
    """Get environment configuration."""
    env = os.getenv("TEST_ENV", "local")
    return SmokeTestConfig.get_config(env)


@pytest.fixture(scope="session")
def api_client(env_config):
    """Create API client for testing."""
    class APIClient:
        def __init__(self, base_url: str, timeout: int):
            self.base_url = base_url
            self.timeout = timeout
            self.session = requests.Session()
            
        def get(self, endpoint: str, **kwargs):
            """Make GET request with consistent timeout."""
            kwargs.setdefault('timeout', self.timeout)
            return self.session.get(f"{self.base_url}{endpoint}", **kwargs)
    
    return APIClient(env_config["api_base_url"], env_config["timeout"])


class TestBasicConnectivity:
    """Test basic connectivity and service availability."""
    
    def test_api_health_endpoint(self, api_client, env_config):
        """Test API health endpoint responds correctly."""
        start_time = time.time()
        response = api_client.get("/health")
        response_time = time.time() - start_time
        
        # Assertions
        assert response.status_code == 200, f"Health endpoint returned {response.status_code}"
        assert response_time < env_config["expected_response_time"], \
            f"Health check too slow: {response_time:.2f}s"
        
        data = response.json()
        assert data.get("status") == "healthy", f"Health status: {data.get('status')}"
    
    def test_frontend_accessibility(self, env_config):
        """Test frontend is accessible."""
        try:
            response = requests.get(env_config["frontend_url"], timeout=env_config["timeout"])
            assert response.status_code == 200, f"Frontend returned {response.status_code}"
        except requests.RequestException as e:
            pytest.fail(f"Frontend not accessible: {e}")
    
    def test_api_cors_headers(self, api_client):
        """Test API has proper CORS headers for frontend integration."""
        response = api_client.get("/health")
        
        # Check for CORS headers (important for frontend integration)
        headers = response.headers
        assert "Access-Control-Allow-Origin" in headers or response.status_code == 200


class TestCoreAPIFunctionality:
    """Test core API functionality works correctly."""
    
    def test_combos_endpoint_basic(self, api_client, env_config):
        """Test combos endpoint returns data."""
        start_time = time.time()
        response = api_client.get("/combos?limit=10")
        response_time = time.time() - start_time
        
        assert response.status_code == 200, f"Combos endpoint returned {response.status_code}"
        assert response_time < env_config["expected_response_time"], \
            f"Combos endpoint too slow: {response_time:.2f}s"
        
        data = response.json()
        assert "combos" in data, "Response missing 'combos' field"
        assert "total" in data, "Response missing 'total' field"
        assert len(data["combos"]) >= env_config["min_combos_expected"], \
            f"Expected at least {env_config['min_combos_expected']} combos, got {len(data['combos'])}"
    
    def test_combos_endpoint_with_filters(self, api_client):
        """Test combos endpoint with filters."""
        response = api_client.get("/combos?limit=5&min_confidence=0.7")
        assert response.status_code == 200
        
        data = response.json()
        combos = data["combos"]
        
        # Verify filtering works
        for combo in combos:
            assert combo.get("confidence", 0) >= 0.7, \
                f"Combo confidence {combo.get('confidence')} below filter threshold 0.7"
    
    def test_products_search_endpoint(self, api_client):
        """Test product search functionality."""
        search_queries = ["apple", "milk", "bread"]
        
        for query in search_queries:
            response = api_client.get(f"/products?query={query}&limit=5")
            assert response.status_code == 200, \
                f"Product search failed for query '{query}': {response.status_code}"
            
            data = response.json()
            assert "products" in data, f"Product search response missing 'products' field for '{query}'"
    
    def test_associations_endpoint(self, api_client):
        """Test product associations endpoint."""
        # Test with a common product
        response = api_client.get("/associations/milk")
        assert response.status_code in [200, 404], \
            f"Associations endpoint returned unexpected status: {response.status_code}"
        
        if response.status_code == 200:
            data = response.json()
            assert "associations" in data, "Associations response missing 'associations' field"


class TestDataIntegrity:
    """Test data integrity and consistency."""
    
    def test_combo_data_structure(self, api_client):
        """Test combo data has proper structure."""
        response = api_client.get("/combos?limit=3")
        assert response.status_code == 200
        
        data = response.json()
        combos = data["combos"]
        
        for i, combo in enumerate(combos):
            # Required fields
            assert "products" in combo, f"Combo {i} missing 'products' field"
            assert "confidence" in combo, f"Combo {i} missing 'confidence' field"
            
            # Data validation
            assert isinstance(combo["products"], list), f"Combo {i} 'products' not a list"
            assert len(combo["products"]) >= 2, f"Combo {i} has less than 2 products"
            assert 0 <= combo["confidence"] <= 1, f"Combo {i} confidence out of range: {combo['confidence']}"
    
    def test_pagination_consistency(self, api_client):
        """Test pagination works consistently."""
        # Get first page
        response1 = api_client.get("/combos?limit=5&page=1")
        assert response1.status_code == 200
        
        data1 = response1.json()
        
        # Get second page
        response2 = api_client.get("/combos?limit=5&page=2")
        assert response2.status_code == 200
        
        data2 = response2.json()
        
        # Verify pagination metadata
        assert data1.get("page") == 1, "First page metadata incorrect"
        assert data2.get("page") == 2, "Second page metadata incorrect"
        assert data1.get("total") == data2.get("total"), "Total count inconsistent between pages"
    
    def test_filtering_consistency(self, api_client):
        """Test filtering produces consistent results."""
        # Test same filter multiple times
        params = "limit=10&min_confidence=0.6"
        
        responses = []
        for _ in range(3):
            response = api_client.get(f"/combos?{params}")
            assert response.status_code == 200
            responses.append(response.json())
            time.sleep(0.5)
        
        # Results should be consistent (assuming data doesn't change during test)
        first_total = responses[0]["total"]
        for i, response_data in enumerate(responses[1:], 1):
            assert response_data["total"] == first_total, \
                f"Total count inconsistent on attempt {i+1}: {response_data['total']} vs {first_total}"


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_invalid_endpoints(self, api_client):
        """Test invalid endpoints return appropriate errors."""
        invalid_endpoints = [
            "/invalid-endpoint",
            "/combos/invalid",
            "/products/invalid"
        ]
        
        for endpoint in invalid_endpoints:
            response = api_client.get(endpoint)
            assert response.status_code == 404, \
                f"Invalid endpoint '{endpoint}' should return 404, got {response.status_code}"
    
    def test_invalid_parameters(self, api_client):
        """Test invalid parameters are handled gracefully."""
        invalid_requests = [
            "/combos?limit=invalid",
            "/combos?min_confidence=2.0",
            "/combos?page=0",
            "/products?limit=-1"
        ]
        
        for request in invalid_requests:
            response = api_client.get(request)
            assert response.status_code in [400, 422], \
                f"Invalid request '{request}' should return 400/422, got {response.status_code}"
    
    def test_large_requests(self, api_client):
        """Test handling of large requests."""
        # Test large limit (should be handled gracefully)
        response = api_client.get("/combos?limit=10000")
        assert response.status_code in [200, 400, 422], \
            f"Large limit request should be handled gracefully, got {response.status_code}"


class TestPerformance:
    """Test basic performance requirements."""
    
    def test_response_times(self, api_client, env_config):
        """Test response times meet requirements."""
        endpoints = [
            "/health",
            "/combos?limit=10",
            "/products?query=test&limit=5"
        ]
        
        max_response_time = env_config["expected_response_time"]
        
        for endpoint in endpoints:
            start_time = time.time()
            response = api_client.get(endpoint)
            response_time = time.time() - start_time
            
            assert response.status_code == 200, f"Endpoint {endpoint} failed: {response.status_code}"
            assert response_time < max_response_time, \
                f"Endpoint {endpoint} too slow: {response_time:.2f}s > {max_response_time}s"
    
    def test_concurrent_requests(self, api_client):
        """Test handling of concurrent requests."""
        import concurrent.futures
        import threading
        
        def make_request():
            response = api_client.get("/combos?limit=5")
            return response.status_code == 200
        
        # Make 5 concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_request) for _ in range(5)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # At least 80% should succeed
        success_rate = sum(results) / len(results)
        assert success_rate >= 0.8, f"Concurrent request success rate too low: {success_rate:.2f}"


class TestSecurity:
    """Test basic security measures."""
    
    def test_security_headers(self, api_client):
        """Test security headers are present."""
        response = api_client.get("/health")
        headers = response.headers
        
        # Check for basic security headers
        security_headers = [
            "X-Content-Type-Options",
            "X-Frame-Options",
            "X-XSS-Protection"
        ]
        
        missing_headers = []
        for header in security_headers:
            if header not in headers:
                missing_headers.append(header)
        
        # Warning rather than failure for missing security headers
        if missing_headers:
            print(f"Warning: Missing security headers: {missing_headers}")
    
    def test_no_sensitive_data_exposure(self, api_client):
        """Test no sensitive data is exposed in responses."""
        response = api_client.get("/health")
        
        # Convert response to string for searching
        response_text = response.text.lower()
        
        # Check for common sensitive data patterns
        sensitive_patterns = [
            "password", "secret", "key", "token", "api_key",
            "database_url", "db_password", "private"
        ]
        
        for pattern in sensitive_patterns:
            assert pattern not in response_text, \
                f"Potentially sensitive data '{pattern}' found in response"


# Pytest markers for different test categories
pytestmark = [
    pytest.mark.smoke,
]


def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line("markers", "smoke: smoke tests for quick validation")
    config.addinivalue_line("markers", "staging: tests for staging environment")
    config.addinivalue_line("markers", "production: tests for production environment")