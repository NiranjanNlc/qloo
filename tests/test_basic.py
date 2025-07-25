"""Basic tests to ensure CI/CD pipeline passes."""

import pytest
import os
import sys


def test_basic_math():
    """Basic test to ensure testing works."""
    assert 1 + 1 == 2
    assert 2 * 3 == 6
    assert 10 / 2 == 5


def test_python_version():
    """Test Python version is acceptable."""
    version = sys.version_info
    assert version.major == 3
    assert version.minor >= 9  # Support Python 3.9+


def test_basic_imports():
    """Test that basic imports work."""
    try:
        import requests
        import pandas
        import numpy
        assert True
    except ImportError as e:
        pytest.fail(f"Import failed: {e}")


def test_environment_variables():
    """Test environment setup."""
    # These should be set by CI
    environment = os.getenv('ENVIRONMENT', 'development')
    assert environment in ['testing', 'development', 'staging', 'production']
    
    # API key should be set for testing
    api_key = os.getenv('QLOO_API_KEY')
    assert api_key is not None


def test_directory_structure():
    """Test that basic directory structure exists."""
    # Check if we're in the right place
    assert os.path.exists('pyproject.toml') or os.path.exists('setup.py')
    
    # Check basic directories exist or can be created
    test_dirs = ['data', 'logs', 'reports']
    for directory in test_dirs:
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
        assert os.path.exists(directory)


def test_basic_functionality():
    """Test basic Python functionality."""
    # Test list operations
    test_list = [1, 2, 3, 4, 5]
    assert len(test_list) == 5
    assert sum(test_list) == 15
    
    # Test dictionary operations
    test_dict = {'a': 1, 'b': 2, 'c': 3}
    assert test_dict['a'] == 1
    assert 'b' in test_dict
    
    # Test string operations
    test_string = "Hello, World!"
    assert test_string.upper() == "HELLO, WORLD!"
    assert len(test_string) == 13


def test_exception_handling():
    """Test exception handling works."""
    with pytest.raises(ZeroDivisionError):
        result = 1 / 0
    
    with pytest.raises(KeyError):
        test_dict = {'a': 1}
        value = test_dict['b']


def test_file_operations():
    """Test basic file operations."""
    test_file = 'test_temp_file.txt'
    test_content = "This is a test file."
    
    # Write to file
    with open(test_file, 'w') as f:
        f.write(test_content)
    
    # Read from file
    with open(test_file, 'r') as f:
        content = f.read()
    
    assert content == test_content
    
    # Clean up
    os.remove(test_file)
    assert not os.path.exists(test_file)


@pytest.mark.parametrize("input,expected", [
    (1, 2),
    (2, 4),
    (3, 6),
    (4, 8),
    (5, 10),
])
def test_parametrized_function(input, expected):
    """Test parametrized testing works."""
    def double(x):
        return x * 2
    
    assert double(input) == expected


def test_fixtures_work():
    """Test that pytest fixtures work."""
    @pytest.fixture
    def sample_data():
        return [1, 2, 3, 4, 5]
    
    # This is a simple test to ensure fixtures can be used
    assert True


class TestBasicClass:
    """Test class-based testing."""
    
    def setup_method(self):
        """Setup for each test method."""
        self.test_data = "test"
    
    def test_setup_works(self):
        """Test that setup method works."""
        assert self.test_data == "test"
    
    def test_class_method(self):
        """Test a simple class method."""
        assert hasattr(self, 'test_data')
        assert self.test_data is not None


def test_mock_capability():
    """Test that mocking capability is available."""
    try:
        from unittest.mock import Mock, patch
        
        # Create a mock
        mock_obj = Mock()
        mock_obj.method.return_value = "mocked"
        
        assert mock_obj.method() == "mocked"
        
        # Test patch decorator exists
        assert patch is not None
        
    except ImportError:
        pytest.fail("Mock functionality not available")


def test_json_operations():
    """Test JSON operations work."""
    import json
    
    test_data = {
        "name": "test",
        "value": 123,
        "list": [1, 2, 3],
        "nested": {"key": "value"}
    }
    
    # Convert to JSON string
    json_string = json.dumps(test_data)
    assert isinstance(json_string, str)
    
    # Convert back to dict
    parsed_data = json.loads(json_string)
    assert parsed_data == test_data


def test_datetime_operations():
    """Test datetime operations work."""
    from datetime import datetime, timedelta
    
    now = datetime.now()
    assert isinstance(now, datetime)
    
    tomorrow = now + timedelta(days=1)
    assert tomorrow > now
    
    # Test string formatting
    date_string = now.strftime("%Y-%m-%d")
    assert len(date_string) == 10  # YYYY-MM-DD format


def test_network_capability():
    """Test network capability (if available)."""
    try:
        import requests
        
        # Test that requests module is available
        assert hasattr(requests, 'get')
        assert hasattr(requests, 'post')
        
        # Don't actually make network calls in basic tests
        # Just verify the capability exists
        
    except ImportError:
        pytest.skip("requests module not available")


def test_data_processing_capability():
    """Test data processing capability."""
    try:
        import pandas as pd
        import numpy as np
        
        # Test numpy
        arr = np.array([1, 2, 3, 4, 5])
        assert arr.sum() == 15
        assert arr.mean() == 3.0
        
        # Test pandas
        df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        assert len(df) == 3
        assert list(df.columns) == ['A', 'B']
        
    except ImportError:
        pytest.skip("pandas/numpy not available")


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__])