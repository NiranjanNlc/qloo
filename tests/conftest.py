"""Pytest configuration and fixtures."""

import pytest
import os
import sys
from pathlib import Path

# Add src to Python path
src_path = Path(__file__).parent.parent / "src"
if src_path.exists():
    sys.path.insert(0, str(src_path))


@pytest.fixture(scope="session")
def test_environment():
    """Set up test environment variables."""
    os.environ.setdefault("ENVIRONMENT", "testing")
    os.environ.setdefault("QLOO_API_KEY", "test-key")
    os.environ.setdefault("DEBUG", "false")
    return os.environ


@pytest.fixture(scope="session")
def test_directories():
    """Ensure test directories exist."""
    dirs = ["data", "logs", "reports", "tests/fixtures"]
    for directory in dirs:
        Path(directory).mkdir(parents=True, exist_ok=True)
    return dirs


@pytest.fixture
def sample_data():
    """Provide sample data for tests."""
    return {
        "products": ["milk", "bread", "eggs", "cheese"],
        "transactions": [
            ["milk", "bread"],
            ["milk", "eggs"],
            ["bread", "cheese"],
            ["milk", "cheese", "eggs"]
        ],
        "associations": [
            {"product_a": "milk", "product_b": "bread", "confidence": 0.8},
            {"product_a": "milk", "product_b": "eggs", "confidence": 0.7},
        ]
    }


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "slow: Slow tests")
    config.addinivalue_line("markers", "network: Tests requiring network access")


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers."""
    for item in items:
        # Add unit marker to all tests by default
        if not any(item.iter_markers()):
            item.add_marker(pytest.mark.unit)