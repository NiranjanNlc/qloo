# 🛒 Qloo Supermarket Layout Optimizer - Developer Onboarding Guide

Welcome to the Qloo Supermarket Layout Optimizer project! This guide will help you get up and running as a new developer on our team.

## 📋 Table of Contents

1. [Project Overview](#-project-overview)
2. [Prerequisites](#-prerequisites)
3. [Environment Setup](#-environment-setup)
4. [Project Architecture](#-project-architecture)
5. [Development Workflow](#-development-workflow)
6. [Key Components](#-key-components)
7. [API Documentation](#-api-documentation)
8. [Testing](#-testing)
9. [Deployment](#-deployment)
10. [Team Guidelines](#-team-guidelines)
11. [Resources & Troubleshooting](#-resources--troubleshooting)

## 🎯 Project Overview

The Qloo Supermarket Layout Optimizer is a Python-based application that leverages AI-powered recommendation engines to optimize supermarket product layouts based on customer behavior and product associations.

### Key Features
- **Product Association Mining**: Discovers relationships between products using transaction data
- **Layout Optimization**: Optimizes shelf placements for maximum revenue and customer experience
- **Combo Offer Generation**: Creates intelligent product bundling recommendations
- **Interactive Dashboard**: Streamlit-based web interface for visualization and management
- **Automated Reporting**: Weekly reports generated via Airflow DAGs
- **Real-time API**: FastAPI backend for serving recommendations

### Technology Stack
- **Backend**: Python 3.8+, FastAPI, SQLite/PostgreSQL
- **Frontend**: Streamlit, Plotly for visualizations
- **Data Processing**: Pandas, NumPy, Association Rules Mining
- **Orchestration**: Apache Airflow
- **Containerization**: Docker, Docker Compose
- **API Integration**: Qloo AI API
- **Testing**: Pytest, Test-driven development

## 🔧 Prerequisites

Before getting started, ensure you have the following installed:

### Required Software
- **Python 3.8+** (recommended: 3.11)
- **Git** for version control
- **Docker** and **Docker Compose** for containerization
- **Make** for build automation (optional but recommended)

### Recommended Tools
- **VS Code** or **PyCharm** for development
- **Postman** or **Insomnia** for API testing
- **DBeaver** or similar for database management

### System Requirements
- **OS**: Linux, macOS, or Windows with WSL2
- **RAM**: Minimum 8GB (16GB recommended)
- **Storage**: 5GB free space for dependencies and data

## 🚀 Environment Setup

### 1. Clone the Repository
```bash
git clone https://github.com/qloo/supermarket-optimizer.git
cd supermarket-optimizer
```

### 2. Create Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
# Install all dependencies including dev tools
make install-dev

# Or manually:
pip install -e ".[dev,test]"
```

### 4. Environment Variables
```bash
# Copy environment template
cp env.example .env

# Edit .env file with your specific values
# Required variables:
# - QLOO_API_KEY: Your Qloo API key
# - QLOO_BASE_URL: https://hackathon.api.qloo.com
```

### 5. Database Setup
```bash
# Load sample catalog data
make load_catalog

# Validate the setup
make validate_catalog
```

### 6. Verify Installation
```bash
# Run tests to verify everything works
make test

# Start the development server
make run-dev
```

## 🏗️ Project Architecture

### Directory Structure
```
qloo-supermarket-optimizer/
├── src/                          # Main source code
│   ├── qloo_client.py           # Qloo API client
│   ├── association_engine.py    # Association rules mining
│   ├── layout_optimizer.py      # Layout optimization algorithms
│   ├── models.py                # Data models and schemas
│   ├── algorithms/              # Core algorithms
│   │   └── scoring.py          # Scoring algorithms
│   └── ...
├── pages/                       # Streamlit pages
│   ├── 1_Product_Catalog.py
│   ├── 2_Association_Rules.py
│   ├── 3_Layout_Optimizer.py
│   └── ...
├── tests/                       # Test suite
├── scripts/                     # Utility scripts
├── dags/                        # Airflow DAGs
├── data/                        # Data files
├── docs/                        # Documentation
├── docker-compose.yml           # Container orchestration
├── Dockerfile                   # API container
├── Dockerfile.streamlit         # Frontend container
├── pyproject.toml              # Project configuration
├── config.yml                  # Application configuration
└── Makefile                    # Build automation
```

### Component Architecture

```mermaid
graph TB
    A[Streamlit UI] --> B[FastAPI Backend]
    B --> C[Association Engine]
    B --> D[Layout Optimizer]
    B --> E[Qloo API Client]
    C --> F[SQLite Database]
    D --> F
    E --> G[Qloo AI API]
    H[Airflow DAGs] --> C
    H --> I[Weekly Reports]
    B --> J[Redis Cache]
```

### Data Flow
1. **Transaction Data** → Association Engine → Product Associations
2. **Product Catalog** → Layout Optimizer → Optimized Layouts
3. **User Interactions** → Streamlit UI → FastAPI → Results
4. **Scheduled Jobs** → Airflow → Weekly Reports

## 🔄 Development Workflow

### Daily Development Process

1. **Pull Latest Changes**
   ```bash
   git pull origin main
   ```

2. **Create Feature Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make Changes**
   - Follow [PEP 8](https://pep8.org/) coding standards
   - Write tests for new functionality
   - Update documentation as needed

4. **Test Your Changes**
   ```bash
   # Run all tests
   make test
   
   # Run specific test file
   pytest tests/test_your_module.py
   
   # Run with coverage
   make test-coverage
   ```

5. **Code Quality Checks**
   ```bash
   # Format code
   make format
   
   # Lint code
   make lint
   
   # Type checking
   make type-check
   ```

6. **Commit and Push**
   ```bash
   git add .
   git commit -m "feat: add new feature description"
   git push origin feature/your-feature-name
   ```

7. **Create Pull Request**
   - Use the PR template
   - Ensure all checks pass
   - Request code review

### Branch Strategy
- **main**: Production-ready code
- **develop**: Integration branch for features
- **feature/***: New features and enhancements
- **bugfix/***: Bug fixes
- **hotfix/***: Emergency production fixes

## 🔑 Key Components

### 1. Qloo API Client (`src/qloo_client.py`)
**Purpose**: Interface with Qloo's AI recommendation API

**Key Methods**:
```python
client = QlooClient()

# Get product recommendations
recommendations = client.get_product_recommendations("apple", limit=10)

# Get product associations
associations = client.get_product_associations(["milk", "bread"])

# Get category insights
insights = client.get_category_insights("dairy")
```

### 2. Association Engine (`src/association_engine.py`)
**Purpose**: Mine product associations from transaction data

**Key Features**:
- Frequent itemset mining
- Association rule generation
- Support/confidence/lift calculation
- Real-time rule updates

### 3. Layout Optimizer (`src/layout_optimizer.py`)
**Purpose**: Optimize product placement for maximum revenue

**Algorithms Available**:
- Genetic Algorithm
- Simulated Annealing
- Greedy Optimization
- Heuristic-based placement

### 4. FastAPI Backend (`association_api.py`)
**Purpose**: RESTful API for serving recommendations

**Key Endpoints**:
- `GET /combos` - Product combinations
- `GET /associations` - Product associations
- `GET /products` - Product search
- `GET /health` - Health check

### 5. Streamlit Frontend (`app.py` + `pages/`)
**Purpose**: Interactive web interface

**Pages**:
- Product Catalog Management
- Association Rules Viewer
- Layout Optimization Tool
- Shelf Placement Planner
- Product Flow Analysis
- Combo Offers Generator
- DAG Status Monitor
- Interactive Store Map

## 📚 API Documentation

### FastAPI Endpoints

#### Get Combo Offers
```http
GET /combos?limit=20&min_confidence=0.6&category=dairy
```

**Response**:
```json
{
  "combos": [
    {
      "id": "combo_001",
      "products": ["milk", "bread", "eggs"],
      "confidence": 0.85,
      "support": 0.12,
      "lift": 2.3,
      "discount_percent": 15.0
    }
  ],
  "total": 25,
  "page": 1
}
```

#### Get Product Associations
```http
GET /associations/{product_id}?limit=10
```

### Qloo API Integration

The project integrates with Qloo's search API:

```python
# Correct usage pattern
response = requests.get(
    "https://hackathon.api.qloo.com/search",
    params={"query": "apple", "limit": 10},
    headers={"x-api-key": "YOUR_API_KEY"}
)
```

**Important Notes**:
- Use `/search` endpoint (not `/recommendations`)
- Use `query` parameter (not `input`)
- Don't include `type` parameter (causes 403 errors)

## 🧪 Testing

### Test Structure
```
tests/
├── test_association_api_e2e.py    # End-to-end API tests
├── test_association_engine.py     # Association mining tests
├── test_combo_generator.py        # Combo generation tests
├── test_price_api.py              # Pricing API tests
├── test_weekly_dag.py             # Airflow DAG tests
└── ...
```

### Running Tests

```bash
# Run all tests
make test

# Run specific test file
pytest tests/test_association_engine.py

# Run with coverage
make test-coverage

# Run E2E tests
pytest tests/test_association_api_e2e.py -v

# Run performance tests
pytest tests/ -m performance
```

### Test Categories

**Unit Tests**: Test individual functions and classes
```bash
pytest tests/test_association_engine.py::test_mine_frequent_itemsets
```

**Integration Tests**: Test component interactions
```bash
pytest tests/test_association_api_e2e.py::test_full_workflow
```

**Performance Tests**: Verify response times and throughput
```bash
pytest tests/ -m performance --tb=short
```

### Writing Tests

Follow these patterns:

```python
import pytest
from src.association_engine import AssociationEngine

class TestAssociationEngine:
    @pytest.fixture
    def engine(self):
        return AssociationEngine()
    
    def test_mine_associations(self, engine):
        # Arrange
        transactions = [["milk", "bread"], ["milk", "eggs"]]
        
        # Act
        rules = engine.mine_associations(transactions)
        
        # Assert
        assert len(rules) > 0
        assert all(rule.confidence > 0 for rule in rules)
```

## 🚀 Deployment

### Local Development

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f association-api

# Stop services
docker-compose down
```

### Production Deployment

```bash
# Build production images
docker-compose -f docker-compose.prod.yml build

# Deploy to production
docker-compose -f docker-compose.prod.yml up -d

# Health checks
curl http://localhost:8000/health
```

### Environment Configurations

**Development** (`.env`):
```env
ENVIRONMENT=development
DEBUG=true
LOG_LEVEL=DEBUG
```

**Production** (`.env.prod`):
```env
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=INFO
DATABASE_URL=postgresql://user:pass@host:5432/qloo_optimizer
```

## 👥 Team Guidelines

### Code Style

**Python Style Guide**:
- Follow [PEP 8](https://pep8.org/)
- Use [Black](https://black.readthedocs.io/) for code formatting
- Use [isort](https://isort.readthedocs.io/) for import sorting
- Maximum line length: 88 characters

**Naming Conventions**:
- Functions and variables: `snake_case`
- Classes: `PascalCase`
- Constants: `UPPER_CASE`
- Private methods: `_leading_underscore`

### Git Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```bash
feat: add new association mining algorithm
fix: resolve database connection timeout
docs: update API documentation
test: add unit tests for layout optimizer
refactor: simplify scoring algorithm
```

### Pull Request Process

1. **Create descriptive PR title**
2. **Fill out PR template completely**
3. **Ensure all tests pass**
4. **Request review from 2+ team members**
5. **Address all review comments**
6. **Squash commits before merging**

### Code Review Guidelines

**What to Look For**:
- Code correctness and logic
- Test coverage and quality
- Performance implications
- Security considerations
- Documentation updates
- API compatibility

**Review Checklist**:
- [ ] Code follows style guidelines
- [ ] Tests are comprehensive
- [ ] Documentation is updated
- [ ] No secrets in code
- [ ] Error handling is proper
- [ ] Performance is acceptable

## 📖 Resources & Troubleshooting

### Useful Commands

```bash
# Database operations
make load_catalog              # Load product catalog
make validate_catalog         # Validate data integrity
make backup_db               # Backup database

# Development tools
make format                  # Format code with Black
make lint                   # Run linting checks
make type-check            # Run type checking
make clean                 # Clean build artifacts

# Docker operations
make docker-build          # Build Docker images
make docker-run           # Run containers
make docker-clean         # Clean Docker resources

# Data operations
make generate-sample-data  # Generate test data
make run-weekly-report    # Generate weekly report
```

### Common Issues & Solutions

**Issue**: API returns 403 Forbidden
```bash
# Solution: Check your API key and remove 'type' parameter
# ❌ Wrong:
params = {"query": "apple", "type": "product"}
# ✅ Correct:
params = {"query": "apple", "limit": 10}
```

**Issue**: Database connection fails
```bash
# Solution: Ensure database is running and accessible
docker-compose up postgres
make validate_catalog
```

**Issue**: Tests fail with import errors
```bash
# Solution: Install in editable mode
pip install -e ".[dev,test]"
```

**Issue**: Docker build fails
```bash
# Solution: Clean Docker cache and rebuild
docker system prune -f
make docker-build
```

### Performance Tips

**Database Optimization**:
- Use indexes on frequently queried columns
- Batch insert operations for large datasets
- Use connection pooling for high concurrency

**API Optimization**:
- Implement caching for frequently accessed data
- Use async operations for I/O bound tasks
- Monitor response times and add alerting

**Algorithm Optimization**:
- Profile code to identify bottlenecks
- Use vectorized operations with NumPy/Pandas
- Consider parallel processing for large datasets

### Learning Resources

**Internal Documentation**:
- [Algorithm Parameters Guide](algorithm_parameters.md)
- [API Solution Guide](../QLOO_API_SOLUTION.md)
- [Sprint Implementation Summary](../SPRINT_IMPLEMENTATION_SUMMARY.md)

**External Resources**:
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Association Rules Mining](https://en.wikipedia.org/wiki/Association_rule_learning)
- [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)

### Getting Help

**Team Contacts**:
- **Technical Lead**: For architecture and design questions
- **DevOps Engineer**: For deployment and infrastructure issues
- **Data Engineer**: For data pipeline and Airflow questions
- **QA Lead**: For testing strategy and automation

**Communication Channels**:
- **Slack**: #qloo-dev for general discussion
- **GitHub Issues**: For bug reports and feature requests
- **Weekly Standups**: Monday, Wednesday, Friday at 9 AM
- **Sprint Planning**: Every other Friday at 2 PM

### Next Steps

1. **Complete this onboarding checklist**:
   - [ ] Environment setup complete
   - [ ] All tests passing
   - [ ] API documentation reviewed
   - [ ] First pull request submitted
   - [ ] Code review participated in

2. **Shadow a team member** for your first week
3. **Attend team meetings** and ask questions
4. **Pick up your first issue** from the backlog
5. **Schedule 1:1 with your mentor** for ongoing support

---

**Welcome to the team! 🎉**

If you have any questions during onboarding, don't hesitate to reach out to any team member. We're here to help you succeed!