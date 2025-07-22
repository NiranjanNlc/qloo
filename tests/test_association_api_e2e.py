"""
End-to-End Tests for Association API

This module contains comprehensive tests to validate the association API
endpoints, ensuring they return the expected number of offers with proper
JSON schema validation.
"""

import pytest
import requests
import json
from datetime import datetime
from typing import Dict, List, Any
import time

# Test configuration
API_BASE_URL = "http://localhost:8000"
TIMEOUT = 30


class TestAssociationAPIE2E:
    """End-to-end tests for the Association API."""

    @pytest.fixture(scope="class")
    def api_client(self):
        """Create an API client for testing."""
        # Wait for API to be ready
        max_retries = 10
        for _ in range(max_retries):
            try:
                response = requests.get(f"{API_BASE_URL}/health", timeout=5)
                if response.status_code == 200:
                    break
            except requests.exceptions.RequestException:
                time.sleep(2)
        else:
            pytest.fail("API did not become ready within timeout")

        return APIClient(API_BASE_URL)

    def test_health_endpoint(self, api_client):
        """Test the health check endpoint."""
        response = api_client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "version" in data

    def test_root_endpoint(self, api_client):
        """Test the root endpoint."""
        response = api_client.get("/")
        assert response.status_code == 200

        data = response.json()
        assert "message" in data
        assert "version" in data
        assert "docs" in data

    def test_combos_endpoint_basic(self, api_client):
        """Test the basic functionality of the combos endpoint."""
        response = api_client.get("/combos")
        assert response.status_code == 200

        data = response.json()
        self._validate_combos_response_schema(data)

        # Ensure we get at least 20 offers as required
        assert (
            data["total_combos"] >= 20
        ), f"Expected at least 20 combos, got {data['total_combos']}"
        assert (
            len(data["combos"]) >= 20
        ), f"Expected at least 20 combos in list, got {len(data['combos'])}"

    def test_combos_endpoint_with_filters(self, api_client):
        """Test the combos endpoint with various filters."""
        # Test with limit parameter
        response = api_client.get("/combos", params={"limit": 25})
        assert response.status_code == 200
        data = response.json()
        assert data["total_combos"] >= 20
        assert len(data["combos"]) >= 20

        # Test with confidence filter
        response = api_client.get(
            "/combos", params={"min_confidence": 0.7, "limit": 30}
        )
        assert response.status_code == 200
        data = response.json()

        # Validate confidence filtering
        for combo in data["combos"]:
            assert combo["confidence_score"] >= 0.7

        # Test with sorting
        response = api_client.get(
            "/combos", params={"sort_by": "lift", "sort_desc": True, "limit": 25}
        )
        assert response.status_code == 200
        data = response.json()

        # Validate sorting
        if len(data["combos"]) > 1:
            for i in range(len(data["combos"]) - 1):
                assert data["combos"][i]["lift"] >= data["combos"][i + 1]["lift"]

    def test_combos_endpoint_performance(self, api_client):
        """Test the performance of the combos endpoint."""
        start_time = time.time()
        response = api_client.get("/combos", params={"limit": 50})
        end_time = time.time()

        assert response.status_code == 200

        # Response time should be reasonable (under 5 seconds for 50 combos)
        response_time = end_time - start_time
        assert response_time < 5.0, f"Response time too slow: {response_time:.2f}s"

    def test_associations_endpoint(self, api_client):
        """Test the product associations endpoint."""
        # First get products to test with
        response = api_client.get("/products")
        assert response.status_code == 200

        products = response.json()
        assert len(products) > 0

        # Test associations for first product
        product_id = products[0]["id"]
        response = api_client.get(f"/associations/{product_id}")
        assert response.status_code == 200

        data = response.json()
        self._validate_associations_response_schema(data)

        assert data["product_id"] == product_id
        assert "associations" in data
        assert "total_associations" in data

    def test_products_endpoint(self, api_client):
        """Test the products endpoint."""
        response = api_client.get("/products")
        assert response.status_code == 200

        products = response.json()
        assert isinstance(products, list)
        assert len(products) > 0

        # Validate product structure
        for product in products:
            assert "id" in product
            assert "name" in product
            assert "category" in product
            assert "price" in product

    def test_regenerate_combos_endpoint(self, api_client):
        """Test the combo regeneration endpoint."""
        response = api_client.post("/combos/regenerate")
        assert response.status_code == 200

        data = response.json()
        assert "message" in data
        assert "status" in data
        assert data["status"] == "accepted"

    def test_combos_json_schema_validation(self, api_client):
        """Comprehensive JSON schema validation for combos response."""
        response = api_client.get("/combos", params={"limit": 30})
        assert response.status_code == 200

        data = response.json()

        # Validate response structure matches expected schema
        self._validate_combos_response_schema(data)

        # Additional detailed validation
        assert isinstance(data["total_combos"], int)
        assert data["total_combos"] >= 20
        assert isinstance(data["combos"], list)
        assert len(data["combos"]) >= 20

        # Validate each combo object
        for combo in data["combos"]:
            self._validate_combo_object_schema(combo)

    def test_api_error_handling(self, api_client):
        """Test API error handling."""
        # Test invalid product ID for associations
        response = api_client.get("/associations/99999")
        assert response.status_code == 404

        # Test invalid parameters
        response = api_client.get(
            "/combos", params={"min_confidence": 2.0}
        )  # Invalid confidence
        assert response.status_code in [400, 422]  # Bad request or validation error

        # Test invalid sort parameter
        response = api_client.get("/combos", params={"sort_by": "invalid_field"})
        assert response.status_code == 200  # Should handle gracefully

    def test_api_documentation(self, api_client):
        """Test that API documentation is accessible."""
        # Test OpenAPI schema endpoint
        response = api_client.get("/openapi.json")
        assert response.status_code == 200

        schema = response.json()
        assert "openapi" in schema
        assert "info" in schema
        assert "paths" in schema

        # Validate key endpoints are documented
        assert "/combos" in schema["paths"]
        assert "/associations/{product_id}" in schema["paths"]
        assert "/products" in schema["paths"]

    def _validate_combos_response_schema(self, data: Dict[str, Any]):
        """Validate the structure of combos response."""
        required_fields = ["total_combos", "combos", "generated_at", "filters_applied"]
        for field in required_fields:
            assert field in data, f"Missing required field: {field}"

        assert isinstance(data["total_combos"], int)
        assert isinstance(data["combos"], list)
        assert isinstance(data["filters_applied"], dict)

        # Validate timestamp format
        try:
            datetime.fromisoformat(data["generated_at"].replace("Z", "+00:00"))
        except ValueError:
            pytest.fail(f"Invalid timestamp format: {data['generated_at']}")

    def _validate_combo_object_schema(self, combo: Dict[str, Any]):
        """Validate the structure of a single combo object."""
        required_fields = [
            "combo_id",
            "name",
            "products",
            "product_names",
            "confidence_score",
            "support",
            "lift",
            "is_active",
        ]

        for field in required_fields:
            assert field in combo, f"Missing required field in combo: {field}"

        # Validate data types
        assert isinstance(combo["combo_id"], str)
        assert isinstance(combo["name"], str)
        assert isinstance(combo["products"], list)
        assert isinstance(combo["product_names"], list)
        assert isinstance(combo["confidence_score"], (int, float))
        assert isinstance(combo["support"], (int, float))
        assert isinstance(combo["lift"], (int, float))
        assert isinstance(combo["is_active"], bool)

        # Validate value ranges
        assert 0.0 <= combo["confidence_score"] <= 1.0
        assert 0.0 <= combo["support"] <= 1.0
        assert combo["lift"] > 0.0

        # Validate product lists match
        assert len(combo["products"]) == len(combo["product_names"])
        assert len(combo["products"]) >= 2  # Combos should have at least 2 products

        # Validate optional fields
        if (
            "expected_discount_percent" in combo
            and combo["expected_discount_percent"] is not None
        ):
            assert 0.0 <= combo["expected_discount_percent"] <= 100.0

    def _validate_associations_response_schema(self, data: Dict[str, Any]):
        """Validate the structure of associations response."""
        required_fields = [
            "product_id",
            "product_name",
            "associations",
            "total_associations",
        ]

        for field in required_fields:
            assert field in data, f"Missing required field: {field}"

        assert isinstance(data["product_id"], int)
        assert isinstance(data["product_name"], str)
        assert isinstance(data["associations"], list)
        assert isinstance(data["total_associations"], int)


class APIClient:
    """Simple HTTP client for API testing."""

    def __init__(self, base_url: str):
        self.base_url = base_url
        self.session = requests.Session()

    def get(self, path: str, params: Dict = None):
        """Make GET request."""
        url = f"{self.base_url}{path}"
        return self.session.get(url, params=params, timeout=TIMEOUT)

    def post(self, path: str, json_data: Dict = None, params: Dict = None):
        """Make POST request."""
        url = f"{self.base_url}{path}"
        return self.session.post(url, json=json_data, params=params, timeout=TIMEOUT)


# Performance benchmark tests
class TestAssociationAPIPerformance:
    """Performance tests for the Association API."""

    @pytest.fixture(scope="class")
    def api_client(self):
        """Create an API client for testing."""
        return APIClient(API_BASE_URL)

    def test_concurrent_requests(self, api_client):
        """Test API performance under concurrent load."""
        import concurrent.futures
        import threading

        def make_request():
            response = api_client.get("/combos", params={"limit": 25})
            return response.status_code == 200

        # Make 10 concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_request) for _ in range(10)]
            results = [
                future.result() for future in concurrent.futures.as_completed(futures)
            ]

        # All requests should succeed
        assert all(results), "Some concurrent requests failed"

    def test_large_response_performance(self, api_client):
        """Test performance with maximum response size."""
        start_time = time.time()
        response = api_client.get("/combos", params={"limit": 100})
        end_time = time.time()

        assert response.status_code == 200

        # Even large responses should be fast
        response_time = end_time - start_time
        assert response_time < 10.0, f"Large response too slow: {response_time:.2f}s"

        data = response.json()
        assert len(data["combos"]) >= 20


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v", "--tb=short"])
