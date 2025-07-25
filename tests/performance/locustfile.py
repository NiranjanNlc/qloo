"""Locust load testing configuration for Qloo Supermarket API."""

from locust import HttpUser, task, between
import random
import json


class QlooAPIUser(HttpUser):
    """Simulated user for API load testing."""
    
    wait_time = between(1, 3)  # Wait 1-3 seconds between requests
    
    def on_start(self):
        """Setup executed when user starts."""
        self.products = [
            "apple", "milk", "bread", "eggs", "cheese", "butter", 
            "orange", "banana", "yogurt", "chicken", "beef", "rice",
            "pasta", "tomato", "onion", "carrot", "potato", "lettuce"
        ]
        
        self.categories = [
            "dairy", "fruits", "vegetables", "meat", "grains", "beverages"
        ]
    
    @task(3)
    def get_health(self):
        """Test health endpoint (high frequency)."""
        with self.client.get("/health", catch_response=True) as response:
            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "healthy":
                    response.success()
                else:
                    response.failure("Health check failed")
            else:
                response.failure(f"Got status code {response.status_code}")
    
    @task(10)
    def get_combos(self):
        """Test combo offers endpoint (most frequent)."""
        params = {
            "limit": random.randint(5, 20),
            "min_confidence": random.choice([0.3, 0.5, 0.6, 0.7, 0.8]),
            "page": random.randint(1, 3)
        }
        
        # Occasionally add category filter
        if random.random() < 0.3:
            params["category"] = random.choice(self.categories)
        
        with self.client.get("/combos", params=params, catch_response=True) as response:
            if response.status_code == 200:
                data = response.json()
                if "combos" in data and "total" in data:
                    response.success()
                else:
                    response.failure("Invalid response structure")
            elif response.status_code == 422:
                # Validation error is acceptable for some random params
                response.success()
            else:
                response.failure(f"Got status code {response.status_code}")
    
    @task(7)
    def search_products(self):
        """Test product search endpoint."""
        query = random.choice(self.products)
        params = {
            "query": query,
            "limit": random.randint(3, 15)
        }
        
        with self.client.get("/products", params=params, catch_response=True) as response:
            if response.status_code == 200:
                data = response.json()
                if "products" in data:
                    response.success()
                else:
                    response.failure("Invalid response structure")
            else:
                response.failure(f"Got status code {response.status_code}")
    
    @task(5)
    def get_associations(self):
        """Test product associations endpoint."""
        product = random.choice(self.products)
        
        with self.client.get(f"/associations/{product}", catch_response=True) as response:
            if response.status_code in [200, 404]:
                # Both success and not found are acceptable
                if response.status_code == 200:
                    data = response.json()
                    if "associations" in data:
                        response.success()
                    else:
                        response.failure("Invalid response structure")
                else:
                    response.success()  # 404 is acceptable for random products
            else:
                response.failure(f"Got status code {response.status_code}")
    
    @task(2)
    def test_pagination(self):
        """Test pagination performance."""
        page = random.randint(1, 5)
        params = {
            "limit": 10,
            "page": page
        }
        
        with self.client.get("/combos", params=params, catch_response=True) as response:
            if response.status_code == 200:
                data = response.json()
                if data.get("page") == page:
                    response.success()
                else:
                    response.failure("Pagination not working correctly")
            else:
                response.failure(f"Got status code {response.status_code}")
    
    @task(1)
    def test_large_limit(self):
        """Test performance with large result sets."""
        params = {
            "limit": random.choice([50, 100, 200]),
            "min_confidence": 0.3  # Lower confidence for more results
        }
        
        with self.client.get("/combos", params=params, catch_response=True) as response:
            if response.status_code == 200:
                data = response.json()
                combo_count = len(data.get("combos", []))
                if combo_count > 0:
                    response.success()
                else:
                    response.failure("No combos returned for large limit")
            elif response.status_code == 422:
                # Validation error for too large limit is acceptable
                response.success()
            else:
                response.failure(f"Got status code {response.status_code}")


class HighVolumeUser(HttpUser):
    """Simulated high-volume user for stress testing."""
    
    wait_time = between(0.1, 0.5)  # Very aggressive timing
    
    def on_start(self):
        """Setup for high-volume testing."""
        self.common_queries = ["milk", "bread", "apple", "cheese", "eggs"]
    
    @task(1)
    def rapid_health_checks(self):
        """Rapid health check requests."""
        self.client.get("/health")
    
    @task(3)
    def rapid_combo_requests(self):
        """Rapid combo requests with common parameters."""
        params = {
            "limit": 10,
            "min_confidence": 0.5
        }
        self.client.get("/combos", params=params)
    
    @task(2)
    def rapid_search_requests(self):
        """Rapid search requests for common products."""
        query = random.choice(self.common_queries)
        params = {"query": query, "limit": 5}
        self.client.get("/products", params=params)


class APIStressUser(HttpUser):
    """User for API stress testing with edge cases."""
    
    wait_time = between(0.5, 1.5)
    
    @task(1)
    def test_invalid_parameters(self):
        """Test with invalid parameters to check error handling."""
        invalid_params = [
            {"limit": -1},
            {"limit": "invalid"},
            {"min_confidence": 2.0},  # Over 1.0
            {"min_confidence": -0.5},  # Negative
            {"page": 0},  # Should be >= 1
            {"category": "invalid_category_that_does_not_exist"}
        ]
        
        params = random.choice(invalid_params)
        
        with self.client.get("/combos", params=params, catch_response=True) as response:
            if response.status_code in [400, 422]:
                # Error responses are expected for invalid params
                response.success()
            elif response.status_code == 200:
                # Some invalid params might be handled gracefully
                response.success()
            else:
                response.failure(f"Unexpected status code {response.status_code}")
    
    @task(1)
    def test_nonexistent_products(self):
        """Test associations for non-existent products."""
        fake_products = [
            "nonexistent_product_xyz123",
            "fake_item_999",
            "invalid_product_abc",
            "",  # Empty string
            "a" * 100  # Very long product name
        ]
        
        product = random.choice(fake_products)
        
        with self.client.get(f"/associations/{product}", catch_response=True) as response:
            if response.status_code in [200, 404, 400]:
                # All these responses are acceptable for invalid products
                response.success()
            else:
                response.failure(f"Unexpected status code {response.status_code}")
    
    @task(1)
    def test_extreme_limits(self):
        """Test with extreme limit values."""
        extreme_limits = [0, 1, 1000, 9999, -1]
        limit = random.choice(extreme_limits)
        
        params = {"limit": limit}
        
        with self.client.get("/combos", params=params, catch_response=True) as response:
            if response.status_code in [200, 400, 422]:
                # Should handle extreme values gracefully
                response.success()
            else:
                response.failure(f"Unexpected status code {response.status_code}")


# Configuration for different load testing scenarios
class LoadTestScenarios:
    """Different load testing scenarios."""
    
    @staticmethod
    def normal_load():
        """Normal load configuration."""
        return {
            "users": 50,
            "spawn_rate": 5,
            "run_time": "5m",
            "user_classes": [QlooAPIUser]
        }
    
    @staticmethod
    def high_load():
        """High load configuration."""
        return {
            "users": 200,
            "spawn_rate": 10,
            "run_time": "10m",
            "user_classes": [QlooAPIUser, HighVolumeUser]
        }
    
    @staticmethod
    def stress_test():
        """Stress test configuration."""
        return {
            "users": 500,
            "spawn_rate": 20,
            "run_time": "15m",
            "user_classes": [QlooAPIUser, HighVolumeUser, APIStressUser]
        }
    
    @staticmethod
    def spike_test():
        """Spike test configuration."""
        return {
            "users": 1000,
            "spawn_rate": 50,
            "run_time": "3m",
            "user_classes": [HighVolumeUser]
        }


# Usage instructions:
"""
Run load tests with different configurations:

1. Normal Load Test:
   locust -f locustfile.py --users 50 --spawn-rate 5 --run-time 5m --host http://localhost:8000

2. High Load Test:
   locust -f locustfile.py --users 200 --spawn-rate 10 --run-time 10m --host http://localhost:8000

3. Stress Test:
   locust -f locustfile.py --users 500 --spawn-rate 20 --run-time 15m --host http://localhost:8000

4. Spike Test:
   locust -f locustfile.py --users 1000 --spawn-rate 50 --run-time 3m --host http://localhost:8000

5. Web UI Mode (for interactive testing):
   locust -f locustfile.py --host http://localhost:8000
   Then open http://localhost:8089 in your browser

6. Headless mode with HTML report:
   locust -f locustfile.py --users 100 --spawn-rate 10 --run-time 5m --host http://localhost:8000 --headless --html=report.html
"""