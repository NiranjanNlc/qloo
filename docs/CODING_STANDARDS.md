# 🎯 Qloo Supermarket Layout Optimizer - Coding Standards

This document outlines the coding standards and best practices for the Qloo Supermarket Layout Optimizer project. Following these standards ensures code consistency, maintainability, and quality across our codebase.

## 📋 Table of Contents

1. [Python Coding Standards](#python-coding-standards)
2. [Code Organization](#code-organization)
3. [Documentation Standards](#documentation-standards)
4. [Testing Standards](#testing-standards)
5. [API Design Guidelines](#api-design-guidelines)
6. [Database Standards](#database-standards)
7. [Security Guidelines](#security-guidelines)
8. [Performance Best Practices](#performance-best-practices)

## 🐍 Python Coding Standards

### Style Guide
We follow **PEP 8** with the following specific configurations:

#### Line Length
- **Maximum line length**: 88 characters (Black formatter default)
- **Docstring line length**: 72 characters

#### Naming Conventions
```python
# Variables and functions: snake_case
def calculate_product_score(product_id: str) -> float:
    association_strength = 0.85
    return association_strength

# Classes: PascalCase
class ProductAssociationEngine:
    pass

# Constants: UPPER_CASE
MAX_RECOMMENDATIONS_PER_PRODUCT = 50
DEFAULT_CONFIDENCE_THRESHOLD = 0.6

# Private methods and attributes: _leading_underscore
class LayoutOptimizer:
    def _calculate_internal_score(self) -> float:
        pass

# Protected methods: _leading_underscore
def _validate_transaction_data(transactions: List[Dict]) -> bool:
    pass
```

#### Import Organization
Use **isort** with the following order:
```python
# 1. Standard library imports
import os
import sys
from datetime import datetime
from typing import List, Dict, Optional

# 2. Third-party library imports
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# 3. Local application imports
from src.qloo_client import QlooClient
from src.models import Product, AssociationRule
from src.algorithms.scoring import calculate_lift
```

#### Type Hints
Always use type hints for function signatures:
```python
from typing import List, Dict, Optional, Union

def mine_associations(
    transactions: List[List[str]], 
    min_support: float = 0.03,
    min_confidence: float = 0.6
) -> List[AssociationRule]:
    """Mine association rules from transaction data."""
    pass

# For complex types, use type aliases
ProductID = str
TransactionData = List[List[ProductID]]
AssociationMatrix = Dict[ProductID, List[ProductID]]
```

### Code Quality Tools

#### Required Tools
```bash
# Code formatting
black --line-length 88 src/

# Import sorting
isort src/

# Linting
flake8 src/

# Type checking
mypy src/

# Security scanning
bandit -r src/
```

#### Pre-commit Configuration
```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/psf/black
    rev: 22.3.0
    hooks:
      - id: black
        args: [--line-length=88]
  
  - repo: https://github.com/pycqa/isort
    rev: 5.10.1
    hooks:
      - id: isort
        args: [--profile=black]
  
  - repo: https://github.com/pycqa/flake8
    rev: 4.0.1
    hooks:
      - id: flake8
```

## 🗂️ Code Organization

### Project Structure
```
src/
├── __init__.py
├── main.py                    # Application entry point
├── models.py                  # Pydantic models and data classes
├── config.py                  # Configuration management
├── exceptions.py              # Custom exceptions
├── utils.py                   # Utility functions
├── qloo_client.py            # External API client
├── association_engine.py     # Core business logic
├── layout_optimizer.py       # Optimization algorithms
├── algorithms/               # Algorithm implementations
│   ├── __init__.py
│   ├── scoring.py
│   └── optimization.py
└── data/                     # Data access layer
    ├── __init__.py
    ├── database.py
    └── repositories.py
```

### Module Organization Principles

#### Single Responsibility
Each module should have a single, well-defined purpose:
```python
# ❌ Bad: Mixed responsibilities
class DataProcessor:
    def load_data(self): pass
    def validate_data(self): pass
    def mine_associations(self): pass
    def optimize_layout(self): pass
    def generate_report(self): pass

# ✅ Good: Single responsibility
class DataLoader:
    def load_transactions(self): pass
    def load_product_catalog(self): pass

class AssociationMiner:
    def mine_frequent_itemsets(self): pass
    def generate_rules(self): pass

class LayoutOptimizer:
    def optimize_placement(self): pass
    def calculate_scores(self): pass
```

#### Clear Interfaces
Define clear interfaces between components:
```python
from abc import ABC, abstractmethod

class OptimizationAlgorithm(ABC):
    @abstractmethod
    def optimize(self, data: OptimizationData) -> OptimizationResult:
        """Optimize layout based on provided data."""
        pass

class GeneticAlgorithm(OptimizationAlgorithm):
    def optimize(self, data: OptimizationData) -> OptimizationResult:
        # Implementation
        pass
```

## 📚 Documentation Standards

### Docstring Format
Use **Google-style docstrings**:

```python
def calculate_association_lift(
    item_a_support: float,
    item_b_support: float,
    combined_support: float
) -> float:
    """Calculate the lift value for an association rule.
    
    Lift measures how much more likely item B is purchased when item A
    is purchased, compared to the general probability of purchasing item B.
    
    Args:
        item_a_support: Support value for item A (probability of occurrence)
        item_b_support: Support value for item B (probability of occurrence)
        combined_support: Support value for items A and B together
        
    Returns:
        Lift value. Values > 1 indicate positive association,
        values < 1 indicate negative association.
        
    Raises:
        ValueError: If any support value is negative or zero.
        
    Example:
        >>> lift = calculate_association_lift(0.3, 0.2, 0.1)
        >>> print(f"Lift: {lift:.2f}")
        Lift: 1.67
    """
    if item_a_support <= 0 or item_b_support <= 0:
        raise ValueError("Support values must be positive")
        
    return combined_support / (item_a_support * item_b_support)
```

### Class Documentation
```python
class ProductAssociationEngine:
    """Engine for mining product associations from transaction data.
    
    This class implements association rule mining algorithms to discover
    relationships between products based on customer purchase patterns.
    
    Attributes:
        min_support: Minimum support threshold for frequent itemsets
        min_confidence: Minimum confidence threshold for association rules
        transactions: List of transaction data
        
    Example:
        >>> engine = ProductAssociationEngine(min_support=0.03)
        >>> engine.load_transactions(transaction_data)
        >>> rules = engine.mine_associations()
        >>> print(f"Found {len(rules)} association rules")
    """
    
    def __init__(self, min_support: float = 0.03, min_confidence: float = 0.6):
        """Initialize the association engine.
        
        Args:
            min_support: Minimum support threshold (default: 0.03)
            min_confidence: Minimum confidence threshold (default: 0.6)
        """
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.transactions: List[List[str]] = []
```

### README Documentation
Each module should have comprehensive README documentation:
- Purpose and overview
- Installation instructions
- Usage examples
- API reference
- Contributing guidelines

## 🧪 Testing Standards

### Test Organization
```
tests/
├── unit/                     # Unit tests
│   ├── test_association_engine.py
│   ├── test_layout_optimizer.py
│   └── test_models.py
├── integration/              # Integration tests
│   ├── test_api_endpoints.py
│   └── test_database_operations.py
├── e2e/                     # End-to-end tests
│   └── test_full_workflow.py
├── performance/             # Performance tests
│   └── test_algorithm_performance.py
└── fixtures/                # Test data and fixtures
    ├── sample_transactions.json
    └── mock_responses.py
```

### Test Naming and Structure
```python
import pytest
from unittest.mock import Mock, patch
from src.association_engine import AssociationEngine

class TestAssociationEngine:
    """Test suite for AssociationEngine class."""
    
    @pytest.fixture
    def engine(self):
        """Create AssociationEngine instance for testing."""
        return AssociationEngine(min_support=0.1, min_confidence=0.5)
    
    @pytest.fixture
    def sample_transactions(self):
        """Sample transaction data for testing."""
        return [
            ["milk", "bread", "eggs"],
            ["milk", "bread"],
            ["bread", "butter"],
            ["milk", "eggs", "cheese"]
        ]
    
    def test_mine_frequent_itemsets_returns_correct_itemsets(
        self, engine, sample_transactions
    ):
        """Test that frequent itemset mining returns expected results."""
        # Arrange
        engine.load_transactions(sample_transactions)
        expected_itemsets = [{"milk"}, {"bread"}, {"milk", "bread"}]
        
        # Act
        itemsets = engine.mine_frequent_itemsets()
        
        # Assert
        assert len(itemsets) >= len(expected_itemsets)
        for expected in expected_itemsets:
            assert expected in itemsets
    
    def test_mine_associations_with_insufficient_data_returns_empty_list(self, engine):
        """Test that mining with insufficient data returns empty list."""
        # Arrange
        insufficient_data = [["item1"], ["item2"]]
        engine.load_transactions(insufficient_data)
        
        # Act
        rules = engine.mine_associations()
        
        # Assert
        assert rules == []
    
    @pytest.mark.parametrize("min_support,expected_count", [
        (0.1, 5),
        (0.3, 2),
        (0.5, 1),
    ])
    def test_mine_associations_with_different_support_thresholds(
        self, engine, sample_transactions, min_support, expected_count
    ):
        """Test mining with different support thresholds."""
        # Arrange
        engine.min_support = min_support
        engine.load_transactions(sample_transactions)
        
        # Act
        rules = engine.mine_associations()
        
        # Assert
        assert len(rules) == expected_count
```

### Test Coverage Requirements
- **Minimum coverage**: 80% overall
- **Critical paths**: 95% coverage
- **New code**: 90% coverage

### Performance Testing
```python
import pytest
import time
from src.layout_optimizer import LayoutOptimizer

@pytest.mark.performance
class TestLayoutOptimizerPerformance:
    """Performance tests for LayoutOptimizer."""
    
    def test_optimization_completes_within_time_limit(self):
        """Test that optimization completes within acceptable time."""
        # Arrange
        optimizer = LayoutOptimizer()
        large_dataset = generate_large_test_dataset(10000)
        max_time_seconds = 30
        
        # Act
        start_time = time.time()
        result = optimizer.optimize(large_dataset)
        execution_time = time.time() - start_time
        
        # Assert
        assert execution_time < max_time_seconds
        assert result is not None
        assert result.is_valid()
```

## 🚀 API Design Guidelines

### RESTful API Design

#### URL Structure
```python
# ✅ Good: Clear, hierarchical URLs
GET /api/v1/products                    # List products
GET /api/v1/products/{product_id}       # Get specific product
GET /api/v1/products/{product_id}/associations  # Get product associations

# ❌ Bad: Unclear, non-hierarchical URLs
GET /api/getProducts
GET /api/productDetail?id=123
```

#### HTTP Methods
```python
# Use appropriate HTTP methods
GET /api/v1/combos          # Retrieve combo offers
POST /api/v1/combos         # Create new combo
PUT /api/v1/combos/{id}     # Update entire combo
PATCH /api/v1/combos/{id}   # Partial update
DELETE /api/v1/combos/{id}  # Delete combo
```

#### Response Format
```python
from pydantic import BaseModel
from typing import List, Optional

class ComboOffer(BaseModel):
    """Model for combo offer response."""
    id: str
    products: List[str]
    confidence: float
    support: float
    lift: float
    discount_percent: float
    created_at: str
    expires_at: Optional[str] = None

class ComboOfferResponse(BaseModel):
    """Response model for combo offers endpoint."""
    combos: List[ComboOffer]
    total: int
    page: int
    page_size: int
    has_next: bool

class ErrorResponse(BaseModel):
    """Standard error response format."""
    error: str
    message: str
    details: Optional[dict] = None
    timestamp: str
```

#### Error Handling
```python
from fastapi import HTTPException

class ValidationError(HTTPException):
    """Custom validation error."""
    def __init__(self, detail: str):
        super().__init__(status_code=400, detail=detail)

class NotFoundError(HTTPException):
    """Custom not found error."""
    def __init__(self, resource: str, identifier: str):
        super().__init__(
            status_code=404, 
            detail=f"{resource} with id '{identifier}' not found"
        )

# Usage in endpoints
@app.get("/api/v1/products/{product_id}")
async def get_product(product_id: str):
    product = await product_repository.get_by_id(product_id)
    if not product:
        raise NotFoundError("Product", product_id)
    return product
```

## 🗄️ Database Standards

### Schema Design
```sql
-- Use clear, descriptive table names
CREATE TABLE products (
    id VARCHAR(50) PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    category VARCHAR(100) NOT NULL,
    price DECIMAL(10, 2) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);

-- Use proper indexes
CREATE INDEX idx_products_category ON products(category);
CREATE INDEX idx_products_name ON products(name);
```

### Migration Scripts
```python
# migrations/001_create_products_table.py
from alembic import op
import sqlalchemy as sa

def upgrade():
    """Create products table."""
    op.create_table(
        'products',
        sa.Column('id', sa.String(50), primary_key=True),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('category', sa.String(100), nullable=False),
        sa.Column('price', sa.Numeric(10, 2), nullable=False),
        sa.Column('created_at', sa.DateTime, default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime, default=sa.func.now())
    )
    
    # Create indexes
    op.create_index('idx_products_category', 'products', ['category'])
    op.create_index('idx_products_name', 'products', ['name'])

def downgrade():
    """Drop products table."""
    op.drop_table('products')
```

## 🔒 Security Guidelines

### Input Validation
```python
from pydantic import BaseModel, validator
from typing import List

class ProductQuery(BaseModel):
    """Model for product query with validation."""
    query: str
    limit: int = 10
    category: Optional[str] = None
    
    @validator('query')
    def validate_query(cls, v):
        if len(v.strip()) < 2:
            raise ValueError('Query must be at least 2 characters long')
        if len(v) > 100:
            raise ValueError('Query must be less than 100 characters')
        return v.strip()
    
    @validator('limit')
    def validate_limit(cls, v):
        if v < 1 or v > 100:
            raise ValueError('Limit must be between 1 and 100')
        return v
```

### Secrets Management
```python
import os
from typing import Optional

class Config:
    """Configuration management with proper secret handling."""
    
    def __init__(self):
        self.qloo_api_key = self._get_secret('QLOO_API_KEY')
        self.database_url = self._get_secret('DATABASE_URL')
        
    def _get_secret(self, key: str) -> str:
        """Get secret from environment variables."""
        value = os.getenv(key)
        if not value:
            raise ValueError(f"Required environment variable {key} not set")
        return value

# ❌ Never do this
API_KEY = "hardcoded-api-key-in-source-code"

# ✅ Always use environment variables
API_KEY = os.getenv('QLOO_API_KEY')
```

### SQL Injection Prevention
```python
# ❌ Vulnerable to SQL injection
def get_product_by_name(name: str):
    query = f"SELECT * FROM products WHERE name = '{name}'"
    return execute_query(query)

# ✅ Use parameterized queries
def get_product_by_name(name: str):
    query = "SELECT * FROM products WHERE name = ?"
    return execute_query(query, (name,))
```

## ⚡ Performance Best Practices

### Database Optimization
```python
# Use efficient queries
class ProductRepository:
    def get_products_with_associations(self, limit: int = 50):
        """Get products with their associations efficiently."""
        # ❌ N+1 query problem
        # products = self.get_all_products()
        # for product in products:
        #     product.associations = self.get_associations(product.id)
        
        # ✅ Single query with JOIN
        query = """
        SELECT p.*, a.associated_product_id, a.confidence, a.lift
        FROM products p
        LEFT JOIN associations a ON p.id = a.product_id
        ORDER BY p.name
        LIMIT ?
        """
        return self.execute_query(query, (limit,))
```

### Caching Strategy
```python
from functools import lru_cache
import redis

class CacheManager:
    def __init__(self):
        self.redis_client = redis.Redis(host='localhost', port=6379, db=0)
    
    @lru_cache(maxsize=1000)
    def get_product_associations(self, product_id: str):
        """Get associations with in-memory caching."""
        # Implementation
        pass
    
    def get_cached_associations(self, product_id: str):
        """Get associations with Redis caching."""
        cache_key = f"associations:{product_id}"
        cached_result = self.redis_client.get(cache_key)
        
        if cached_result:
            return json.loads(cached_result)
            
        # Fetch from database
        result = self._fetch_associations_from_db(product_id)
        
        # Cache for 1 hour
        self.redis_client.setex(cache_key, 3600, json.dumps(result))
        return result
```

### Algorithm Optimization
```python
import numpy as np
import pandas as pd

class OptimizedAssociationMiner:
    """Optimized association mining using vectorized operations."""
    
    def mine_frequent_itemsets_vectorized(self, transactions_df: pd.DataFrame):
        """Use pandas/numpy for faster computation."""
        # ❌ Slow: nested loops
        # for transaction in transactions:
        #     for item in transaction:
        #         # process item
        
        # ✅ Fast: vectorized operations
        item_counts = transactions_df.stack().value_counts()
        min_support_count = len(transactions_df) * self.min_support
        frequent_items = item_counts[item_counts >= min_support_count]
        
        return frequent_items.index.tolist()
```

---

## 🔧 Tools and Automation

### Makefile Targets
```makefile
# Quality checks
format:
	black src/ tests/
	isort src/ tests/

lint:
	flake8 src/ tests/
	mypy src/

test:
	pytest tests/ -v --cov=src --cov-report=html

security:
	bandit -r src/
	safety check

# Combined quality check
quality: format lint test security
```

### CI/CD Pipeline
```yaml
# .github/workflows/quality.yml
name: Code Quality
on: [push, pull_request]

jobs:
  quality:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: |
          pip install -e ".[dev,test]"
      
      - name: Format check
        run: |
          black --check src/ tests/
          isort --check-only src/ tests/
      
      - name: Lint
        run: |
          flake8 src/ tests/
          mypy src/
      
      - name: Test
        run: |
          pytest tests/ --cov=src --cov-report=xml
      
      - name: Security scan
        run: |
          bandit -r src/
          safety check
```

Following these coding standards will help maintain high code quality, improve collaboration, and ensure the long-term maintainability of the Qloo Supermarket Layout Optimizer project.