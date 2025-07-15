"""
Unit tests for QlooClient retry wrapper functionality.

These tests verify that the retry logic with exponential backoff works correctly
for various scenarios including happy path, timeouts, and server errors.
"""

import pytest
import requests
import time
from unittest.mock import Mock, patch, MagicMock
from src.qloo_client import QlooClient, QlooAPIError, QlooTimeoutError


class TestQlooClientRetryWrapper:
    """Test suite for QlooClient retry functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.client = QlooClient(
            api_key="test_key",
            base_url="https://test.api.com",
            max_retries=2,
            initial_delay=0.1,  # Fast tests
            max_delay=1.0,
            timeout=1.0
        )

    def test_exponential_backoff_calculation(self):
        """Test exponential backoff delay calculation."""
        # Test basic exponential backoff
        delay_0 = self.client._exponential_backoff_delay(0)
        delay_1 = self.client._exponential_backoff_delay(1)
        delay_2 = self.client._exponential_backoff_delay(2)
        
        # Should follow exponential pattern (with jitter)
        assert 0.05 <= delay_0 <= 0.15  # ~0.1 * (2^0) ± jitter
        assert 0.15 <= delay_1 <= 0.25  # ~0.1 * (2^1) ± jitter  
        assert 0.3 <= delay_2 <= 0.5   # ~0.1 * (2^2) ± jitter
        
        # Test max delay cap
        client_capped = QlooClient(
            api_key="test", 
            max_retries=3, 
            initial_delay=1.0, 
            max_delay=2.0
        )
        delay_large = client_capped._exponential_backoff_delay(10)
        assert delay_large <= 2.5  # Should be capped at max_delay + jitter

    def test_should_retry_logic(self):
        """Test retry decision logic for different exception types."""
        # Test max retries exceeded
        assert not self.client._should_retry(requests.ConnectionError(), 2)
        assert not self.client._should_retry(requests.Timeout(), 3)
        
        # Test retryable exceptions
        assert self.client._should_retry(requests.ConnectionError(), 0)
        assert self.client._should_retry(requests.Timeout(), 1)
        
        # Test HTTP errors
        mock_response_500 = Mock()
        mock_response_500.status_code = 500
        http_error_500 = requests.HTTPError()
        http_error_500.response = mock_response_500
        assert self.client._should_retry(http_error_500, 0)
        
        mock_response_429 = Mock()
        mock_response_429.status_code = 429
        http_error_429 = requests.HTTPError()
        http_error_429.response = mock_response_429
        assert self.client._should_retry(http_error_429, 0)
        
        # Test non-retryable HTTP errors
        mock_response_400 = Mock()
        mock_response_400.status_code = 400
        http_error_400 = requests.HTTPError()
        http_error_400.response = mock_response_400
        assert not self.client._should_retry(http_error_400, 0)
        
        mock_response_404 = Mock()
        mock_response_404.status_code = 404
        http_error_404 = requests.HTTPError()
        http_error_404.response = mock_response_404
        assert not self.client._should_retry(http_error_404, 0)

    @patch('src.qloo_client.time.sleep')  # Mock sleep to speed up tests
    def test_happy_path_success_first_try(self, mock_sleep):
        """Test successful API call on first attempt."""
        # Mock successful response
        mock_response = Mock()
        mock_response.json.return_value = {"results": [{"id": "test", "name": "Test Product"}]}
        mock_response.raise_for_status.return_value = None
        
        with patch.object(self.client.session, 'request', return_value=mock_response) as mock_request:
            result = self.client._make_request_with_retry("search", params={"query": "test"})
            
            # Verify success
            assert result == {"results": [{"id": "test", "name": "Test Product"}]}
            assert mock_request.call_count == 1
            assert not mock_sleep.called

    @patch('src.qloo_client.time.sleep')
    def test_retry_on_connection_error_then_success(self, mock_sleep):
        """Test retry behavior when connection fails then succeeds."""
        # Mock responses: first fails, second succeeds
        mock_success_response = Mock()
        mock_success_response.json.return_value = {"results": []}
        mock_success_response.raise_for_status.return_value = None
        
        with patch.object(self.client.session, 'request') as mock_request:
            mock_request.side_effect = [
                requests.ConnectionError("Connection failed"),  # First attempt fails
                mock_success_response  # Second attempt succeeds
            ]
            
            result = self.client._make_request_with_retry("search")
            
            # Verify retry and eventual success
            assert result == {"results": []}
            assert mock_request.call_count == 2
            assert mock_sleep.call_count == 1  # One sleep between retries

    @patch('src.qloo_client.time.sleep')
    def test_timeout_error_exhausts_retries(self, mock_sleep):
        """Test that timeout errors eventually raise QlooTimeoutError."""
        with patch.object(self.client.session, 'request') as mock_request:
            mock_request.side_effect = requests.Timeout("Request timed out")
            
            with pytest.raises(QlooTimeoutError) as exc_info:
                self.client._make_request_with_retry("search")
            
            assert "Request timed out after 3 attempts" in str(exc_info.value)
            assert mock_request.call_count == 3  # max_retries=2 + initial attempt
            assert mock_sleep.call_count == 2  # Sleep between each retry

    @patch('src.qloo_client.time.sleep')
    def test_http_server_error_retries_then_fails(self, mock_sleep):
        """Test retry behavior for 5xx server errors."""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        http_error = requests.HTTPError()
        http_error.response = mock_response
        
        with patch.object(self.client.session, 'request') as mock_request:
            mock_request.side_effect = http_error
            
            with pytest.raises(QlooAPIError) as exc_info:
                self.client._make_request_with_retry("search")
            
            assert "API error 500" in str(exc_info.value)
            assert mock_request.call_count == 3  # Retries server errors
            assert mock_sleep.call_count == 2

    @patch('src.qloo_client.time.sleep')
    def test_http_client_error_no_retries(self, mock_sleep):
        """Test that 4xx client errors don't trigger retries."""
        mock_response = Mock()
        mock_response.status_code = 400
        mock_response.text = "Bad Request"
        http_error = requests.HTTPError()
        http_error.response = mock_response
        
        with patch.object(self.client.session, 'request') as mock_request:
            mock_request.side_effect = http_error
            
            with pytest.raises(QlooAPIError) as exc_info:
                self.client._make_request_with_retry("search")
            
            assert "API error 400" in str(exc_info.value)
            assert mock_request.call_count == 1  # No retries for client errors
            assert not mock_sleep.called

    @patch('src.qloo_client.time.sleep')
    def test_rate_limit_retries(self, mock_sleep):
        """Test that 429 rate limit errors trigger retries."""
        mock_response = Mock()
        mock_response.status_code = 429
        mock_response.text = "Rate Limited"
        http_error = requests.HTTPError()
        http_error.response = mock_response
        
        with patch.object(self.client.session, 'request') as mock_request:
            mock_request.side_effect = http_error
            
            with pytest.raises(QlooAPIError) as exc_info:
                self.client._make_request_with_retry("search")
            
            assert "API error 429" in str(exc_info.value)
            assert mock_request.call_count == 3  # Should retry rate limits
            assert mock_sleep.call_count == 2

    def test_get_product_recommendations_integration(self):
        """Test that get_product_recommendations uses retry wrapper."""
        # Mock successful API response
        mock_response = Mock()
        mock_response.json.return_value = {
            "results": [
                {"id": "1", "name": "Apple"},
                {"id": "2", "name": "Banana"}
            ]
        }
        mock_response.raise_for_status.return_value = None
        
        with patch.object(self.client.session, 'request', return_value=mock_response):
            recommendations = self.client.get_product_recommendations("fruit", limit=5)
            
            # Verify recommendations returned
            assert len(recommendations) == 2
            assert recommendations[0]["name"] == "Apple"
            assert recommendations[1]["name"] == "Banana"

    def test_get_product_recommendations_fallback_on_error(self):
        """Test that get_product_recommendations falls back to mock data on persistent errors."""
        with patch.object(self.client.session, 'request') as mock_request:
            mock_request.side_effect = requests.ConnectionError("Network unreachable")
            
            # Should return fallback mock data without raising exception
            recommendations = self.client.get_product_recommendations("fruit", limit=3)
            
            # Verify fallback data structure
            assert len(recommendations) <= 3
            assert all("id" in rec and "name" in rec for rec in recommendations)

    def test_health_check_with_retries(self):
        """Test health check functionality with retry logic."""
        # Test successful health check
        mock_response = Mock()
        mock_response.json.return_value = {"status": "ok"}
        mock_response.raise_for_status.return_value = None
        
        with patch.object(self.client.session, 'request', return_value=mock_response):
            assert self.client.health_check() is True
        
        # Test failed health check
        with patch.object(self.client.session, 'request') as mock_request:
            mock_request.side_effect = requests.ConnectionError("Connection failed")
            assert self.client.health_check() is False

    @patch('src.qloo_client.time.sleep')
    def test_exponential_backoff_timing(self, mock_sleep):
        """Test that exponential backoff delays are actually applied."""
        with patch.object(self.client.session, 'request') as mock_request:
            mock_request.side_effect = requests.ConnectionError("Connection failed")
            
            with pytest.raises(QlooAPIError):
                self.client._make_request_with_retry("search")
            
            # Verify sleep was called with increasing delays
            assert mock_sleep.call_count == 2
            
            # Check that delays follow exponential pattern (approximately)
            call_args = [call[0][0] for call in mock_sleep.call_args_list]
            assert call_args[0] < call_args[1]  # Second delay should be longer


class TestQlooClientConfiguration:
    """Test QlooClient initialization and configuration."""
    
    def test_client_initialization_with_custom_retry_config(self):
        """Test client initialization with custom retry parameters."""
        client = QlooClient(
            api_key="test_key",
            max_retries=5,
            initial_delay=2.0,
            max_delay=120.0,
            timeout=30.0
        )
        
        assert client.max_retries == 5
        assert client.initial_delay == 2.0
        assert client.max_delay == 120.0
        assert client.timeout == 30.0

    def test_client_initialization_with_defaults(self):
        """Test client initialization with default retry parameters."""
        client = QlooClient(api_key="test_key")
        
        # Verify default values
        assert client.max_retries == 3
        assert client.initial_delay == 1.0
        assert client.max_delay == 60.0
        assert client.timeout == 10.0

    def test_factory_function_creates_working_client(self):
        """Test that the factory function creates a properly configured client."""
        with patch.dict('os.environ', {'QLOO_API_KEY': 'test_key'}):
            from src.qloo_client import create_qloo_client
            client = create_qloo_client()
            
            assert isinstance(client, QlooClient)
            assert client.api_key == 'test_key' 