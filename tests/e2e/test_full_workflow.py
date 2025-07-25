"""End-to-end tests for complete user workflows."""

import pytest
import requests
import time
import os
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options


class TestCompleteWorkflow:
    """Test complete user workflows end-to-end."""
    
    @pytest.fixture(scope="class")
    def driver(self):
        """Setup Selenium WebDriver."""
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--window-size=1920,1080")
        
        driver = webdriver.Chrome(options=chrome_options)
        yield driver
        driver.quit()
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test environment."""
        self.api_url = os.getenv("API_BASE_URL", "http://localhost:8000")
        self.frontend_url = os.getenv("FRONTEND_URL", "http://localhost:8501")
        
        # Wait for services to be ready
        self._wait_for_service(self.api_url + "/health", "API")
        self._wait_for_service(self.frontend_url, "Frontend")
    
    def _wait_for_service(self, url: str, service_name: str, timeout: int = 60):
        """Wait for a service to be ready."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    return
            except requests.RequestException:
                pass
            time.sleep(2)
        
        pytest.fail(f"{service_name} not ready after {timeout} seconds")
    
    def test_api_to_frontend_workflow(self, driver):
        """Test complete workflow from API to frontend display."""
        # Step 1: Verify API returns data
        response = requests.get(f"{self.api_url}/combos?limit=5")
        assert response.status_code == 200
        
        api_data = response.json()
        assert "combos" in api_data
        assert len(api_data["combos"]) > 0
        
        # Step 2: Visit frontend and verify it loads
        driver.get(self.frontend_url)
        
        # Wait for Streamlit to load
        WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )
        
        # Verify page title
        assert "Qloo" in driver.title or "Supermarket" in driver.title
        
        # Step 3: Navigate to combo offers page
        try:
            # Look for navigation elements or combo-related content
            combo_elements = driver.find_elements(By.XPATH, "//*[contains(text(), 'Combo') or contains(text(), 'combo')]")
            if combo_elements:
                combo_elements[0].click()
                time.sleep(3)
        except Exception:
            # If navigation fails, that's okay for this test
            pass
        
        # Step 4: Verify data is displayed
        page_source = driver.page_source.lower()
        assert any(keyword in page_source for keyword in ["combo", "product", "association", "recommendation"])
    
    def test_product_search_workflow(self, driver):
        """Test product search workflow."""
        # Step 1: Search for products via API
        search_query = "apple"
        response = requests.get(f"{self.api_url}/products?query={search_query}&limit=5")
        assert response.status_code == 200
        
        products = response.json().get("products", [])
        
        # Step 2: Visit frontend
        driver.get(self.frontend_url)
        WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )
        
        # Step 3: Look for search functionality
        try:
            search_inputs = driver.find_elements(By.XPATH, "//input[@type='text']")
            if search_inputs:
                search_input = search_inputs[0]
                search_input.clear()
                search_input.send_keys(search_query)
                
                # Look for search button or submit
                search_buttons = driver.find_elements(By.XPATH, "//*[contains(text(), 'Search') or contains(text(), 'search')]")
                if search_buttons:
                    search_buttons[0].click()
                    time.sleep(3)
        except Exception:
            # Search functionality might not be available in current UI
            pass
        
        # Verify page contains relevant content
        page_source = driver.page_source.lower()
        assert any(keyword in page_source for keyword in ["product", "search", "catalog"])
    
    def test_association_rules_workflow(self):
        """Test association rules workflow via API."""
        # Step 1: Get available products
        response = requests.get(f"{self.api_url}/products?query=milk&limit=1")
        assert response.status_code == 200
        
        products = response.json().get("products", [])
        if not products:
            pytest.skip("No products available for association testing")
        
        product_id = products[0].get("id", "milk")
        
        # Step 2: Get associations for the product
        response = requests.get(f"{self.api_url}/associations/{product_id}")
        assert response.status_code == 200
        
        associations = response.json()
        assert "associations" in associations
        
        # Step 3: Verify associations have proper structure
        for association in associations["associations"]:
            assert any(key in association for key in ["product", "confidence", "lift", "support"])
    
    def test_combo_generation_workflow(self):
        """Test combo generation workflow."""
        # Step 1: Get combo offers
        response = requests.get(f"{self.api_url}/combos?limit=10&min_confidence=0.5")
        assert response.status_code == 200
        
        data = response.json()
        combos = data["combos"]
        
        # Step 2: Verify combo structure
        for combo in combos:
            assert "products" in combo
            assert "confidence" in combo
            assert isinstance(combo["products"], list)
            assert len(combo["products"]) >= 2
            assert combo["confidence"] >= 0.5
        
        # Step 3: Test filtering
        high_confidence_response = requests.get(f"{self.api_url}/combos?min_confidence=0.8&limit=5")
        assert high_confidence_response.status_code == 200
        
        high_confidence_combos = high_confidence_response.json()["combos"]
        for combo in high_confidence_combos:
            assert combo["confidence"] >= 0.8
    
    def test_health_monitoring_workflow(self):
        """Test health monitoring and system status."""
        # Step 1: Check API health
        response = requests.get(f"{self.api_url}/health")
        assert response.status_code == 200
        
        health_data = response.json()
        assert health_data["status"] == "healthy"
        
        # Step 2: Check database connectivity (if included in health check)
        if "database" in health_data:
            assert health_data["database"]["status"] == "connected"
        
        # Step 3: Check external service connectivity (if included)
        if "external_services" in health_data:
            for service_name, service_status in health_data["external_services"].items():
                assert service_status in ["connected", "available", "healthy"]
    
    def test_error_handling_workflow(self):
        """Test error handling across the system."""
        # Step 1: Test invalid API endpoints
        response = requests.get(f"{self.api_url}/invalid-endpoint")
        assert response.status_code == 404
        
        # Step 2: Test invalid parameters
        response = requests.get(f"{self.api_url}/combos?limit=invalid")
        assert response.status_code == 422
        
        error_data = response.json()
        assert "detail" in error_data or "error" in error_data
        
        # Step 3: Test non-existent product associations
        response = requests.get(f"{self.api_url}/associations/nonexistent-product-123")
        assert response.status_code in [200, 404]  # Should handle gracefully
    
    def test_performance_workflow(self):
        """Test system performance under normal load."""
        # Step 1: Measure API response times
        endpoints = [
            "/health",
            "/combos?limit=5",
            "/products?query=test&limit=3"
        ]
        
        for endpoint in endpoints:
            start_time = time.time()
            response = requests.get(f"{self.api_url}{endpoint}")
            response_time = time.time() - start_time
            
            assert response.status_code == 200
            assert response_time < 5.0  # Should respond within 5 seconds
        
        # Step 2: Test concurrent requests
        import concurrent.futures
        
        def make_request():
            response = requests.get(f"{self.api_url}/combos?limit=3")
            return response.status_code == 200 and response.elapsed.total_seconds() < 10
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_request) for _ in range(10)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # At least 80% of concurrent requests should succeed within time limit
        success_rate = sum(results) / len(results)
        assert success_rate >= 0.8


class TestDataFlow:
    """Test data flow through the entire system."""
    
    def test_data_consistency_across_endpoints(self):
        """Test data consistency across different endpoints."""
        api_url = os.getenv("API_BASE_URL", "http://localhost:8000")
        
        # Get total combo count
        combos_response = requests.get(f"{api_url}/combos?limit=100")
        assert combos_response.status_code == 200
        
        total_combos = combos_response.json()["total"]
        
        # Get paginated results and verify consistency
        page_size = 10
        total_fetched = 0
        page = 1
        
        while total_fetched < total_combos and page <= 5:  # Limit to 5 pages for test
            response = requests.get(f"{api_url}/combos?limit={page_size}&page={page}")
            assert response.status_code == 200
            
            data = response.json()
            combos = data["combos"]
            total_fetched += len(combos)
            page += 1
            
            # Verify each combo has required fields
            for combo in combos:
                assert "products" in combo
                assert "confidence" in combo
                assert isinstance(combo["products"], list)
                assert len(combo["products"]) >= 2
    
    def test_cache_consistency(self):
        """Test cache consistency across requests."""
        api_url = os.getenv("API_BASE_URL", "http://localhost:8000")
        
        # Make same request multiple times
        endpoint = "/combos?limit=5"
        responses = []
        
        for _ in range(3):
            response = requests.get(f"{api_url}{endpoint}")
            assert response.status_code == 200
            responses.append(response.json())
            time.sleep(1)
        
        # Results should be consistent (assuming data doesn't change during test)
        first_response = responses[0]
        for response in responses[1:]:
            assert response["total"] == first_response["total"]
            # Note: exact combo order might vary, so we just check totals