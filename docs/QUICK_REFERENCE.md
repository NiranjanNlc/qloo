# 🚀 Qloo Supermarket Layout Optimizer - Quick Reference Guide

This is a quick reference for common development tasks, commands, and workflows.

## 🏃‍♂️ Quick Start

```bash
# Clone and setup
git clone https://github.com/qloo/supermarket-optimizer.git
cd supermarket-optimizer
./scripts/setup_dev_environment.sh

# Start development
source venv/bin/activate  # or venv\Scripts\activate on Windows
make run-dev
```

## 📝 Common Commands

### Environment Management
```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate          # Windows

# Install dependencies
make install-dev               # All dependencies
pip install -e ".[dev,test]"  # Manual install

# Deactivate environment
deactivate
```

### Development Server
```bash
# Start Streamlit frontend
streamlit run app.py

# Start FastAPI backend
uvicorn association_api:app --reload --port 8000

# Start with Docker
docker-compose up -d

# View logs
docker-compose logs -f association-api
```

### Testing
```bash
# Run all tests
make test
pytest tests/

# Run specific test file
pytest tests/test_association_engine.py

# Run with coverage
make test-coverage
pytest --cov=src --cov-report=html

# Run performance tests
pytest tests/ -m performance

# Run E2E tests
pytest tests/test_association_api_e2e.py -v
```

### Code Quality
```bash
# Format code
make format
black src/ tests/
isort src/ tests/

# Lint code
make lint
flake8 src/ tests/
mypy src/

# Security scan
bandit -r src/
safety check

# All quality checks
make quality
```

### Database Operations
```bash
# Load catalog data
make load_catalog

# Validate catalog
make validate_catalog

# Generate sample data
python scripts/generate_sample_transactions.py

# Backup database
make backup_db
```

### Docker Operations
```bash
# Build images
docker-compose build

# Start services
docker-compose up -d

# Stop services
docker-compose down

# View logs
docker-compose logs -f [service-name]

# Rebuild specific service
docker-compose build association-api

# Clean up
docker-compose down -v
docker system prune -f
```

## 🔧 Development Workflow

### 1. Starting a New Feature
```bash
# Pull latest changes
git pull origin main

# Create feature branch
git checkout -b feature/your-feature-name

# Start development
make run-dev
```

### 2. Making Changes
```bash
# Write code...
# Write tests...

# Check code quality
make format
make lint

# Run tests
make test

# Commit changes
git add .
git commit -m "feat: add new feature description"
```

### 3. Before Pushing
```bash
# Run full quality check
make quality

# Check if all tests pass
make test

# Push changes
git push origin feature/your-feature-name
```

### 4. Pull Request
- Create PR with template
- Ensure all checks pass
- Request reviews
- Address feedback
- Merge when approved

## 📊 API Endpoints

### Development URLs
- **Streamlit Frontend**: http://localhost:8501
- **FastAPI Backend**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **ReDoc Documentation**: http://localhost:8000/redoc

### Key Endpoints
```bash
# Health check
GET http://localhost:8000/health

# Get combo offers
GET http://localhost:8000/combos?limit=20&min_confidence=0.6

# Get product associations
GET http://localhost:8000/associations/{product_id}

# Search products
GET http://localhost:8000/products?query=apple&limit=10
```

### Testing Endpoints
```bash
# Using curl
curl "http://localhost:8000/combos?limit=10"

# Using httpie
http GET localhost:8000/combos limit==10

# Using Python requests
import requests
response = requests.get("http://localhost:8000/combos", params={"limit": 10})
```

## 🗃️ Database Quick Reference

### SQLite Commands
```bash
# Connect to database
sqlite3 data/qloo_optimizer.db

# Common queries
.schema                          # Show all tables
.tables                         # List tables
SELECT * FROM products LIMIT 5; # View sample data
.quit                           # Exit
```

### Common Data Operations
```python
# Load data programmatically
from src.database_setup import setup_database
from scripts.load_catalog import load_product_catalog

setup_database()
load_product_catalog("data/grocery_catalog.csv")
```

## 🧪 Testing Quick Reference

### Test Structure
```
tests/
├── unit/           # Unit tests (fast, isolated)
├── integration/    # Integration tests (components together)
├── e2e/           # End-to-end tests (full workflow)
├── performance/   # Performance tests
└── fixtures/      # Test data
```

### Common Test Commands
```bash
# Run tests by pattern
pytest -k "test_association"

# Run tests with verbose output
pytest -v

# Run tests in parallel
pytest -n auto

# Run tests with debugging
pytest --pdb

# Run specific test class
pytest tests/test_association_engine.py::TestAssociationEngine

# Run specific test method
pytest tests/test_association_engine.py::TestAssociationEngine::test_mine_associations
```

### Writing Tests
```python
# Basic test structure
import pytest
from src.your_module import YourClass

class TestYourClass:
    @pytest.fixture
    def instance(self):
        return YourClass()
    
    def test_method_name(self, instance):
        # Arrange
        # Act
        # Assert
        pass
```

## 🐛 Debugging Tips

### Common Issues and Solutions

#### Import Errors
```bash
# Solution: Install in editable mode
pip install -e .

# Or add src to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
```

#### API Connection Issues
```bash
# Check if services are running
docker-compose ps

# Check logs
docker-compose logs association-api

# Restart services
docker-compose restart
```

#### Database Issues
```bash
# Recreate database
rm data/qloo_optimizer.db
make load_catalog

# Check database file permissions
ls -la data/qloo_optimizer.db
```

#### Permission Issues
```bash
# Fix script permissions
chmod +x scripts/setup_dev_environment.sh

# Fix Docker permissions (Linux)
sudo usermod -aG docker $USER
```

### Debugging Code
```python
# Add debug prints
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.debug("Debug message")

# Use debugger
import pdb; pdb.set_trace()

# Use ipdb (better debugger)
import ipdb; ipdb.set_trace()
```

## 📈 Performance Monitoring

### Basic Performance Checks
```bash
# Time command execution
time make test

# Monitor resource usage
htop  # Linux/macOS
taskmgr  # Windows

# Check Python performance
python -m cProfile -s cumulative your_script.py
```

### API Performance Testing
```bash
# Using ab (Apache Bench)
ab -n 1000 -c 10 http://localhost:8000/health

# Using siege
siege -c 10 -r 100 http://localhost:8000/combos

# Using locust (if installed)
locust -f tests/performance/locustfile.py
```

## 🔍 Troubleshooting

### Log Locations
```bash
# Application logs
logs/app.log

# Docker logs
docker-compose logs

# System logs (Linux)
/var/log/

# Python logs
~/.cache/pip/log/
```

### Environment Variables
```bash
# Check current environment
printenv | grep QLOO

# Load .env file
source .env

# Set temporary variable
export QLOO_API_KEY="your-key"
```

### Network Issues
```bash
# Check port availability
netstat -tulpn | grep :8000
lsof -i :8000

# Kill process on port
kill $(lsof -t -i:8000)

# Check connectivity
curl -I http://localhost:8000/health
```

## 🔐 Security Checklist

### Before Committing
- [ ] No hardcoded secrets in code
- [ ] Environment variables used for sensitive data
- [ ] No API keys in commit history
- [ ] Dependencies are up to date and secure

### Security Commands
```bash
# Scan for secrets
git secrets --scan

# Check for vulnerabilities
safety check
bandit -r src/

# Update dependencies
pip-audit
pip install --upgrade package-name
```

## 📚 Resources

### Documentation Links
- [Developer Onboarding](DEVELOPER_ONBOARDING.md)
- [Coding Standards](CODING_STANDARDS.md)
- [API Solution Guide](../QLOO_API_SOLUTION.md)
- [Algorithm Parameters](algorithm_parameters.md)

### External Resources
- [FastAPI Docs](https://fastapi.tiangolo.com/)
- [Streamlit Docs](https://docs.streamlit.io/)
- [Pytest Docs](https://docs.pytest.org/)
- [Docker Docs](https://docs.docker.com/)

### Useful Tools
- **Code Editor**: VS Code, PyCharm
- **API Testing**: Postman, Insomnia, HTTPie
- **Database**: DBeaver, SQLite Browser
- **Monitoring**: htop, docker stats
- **Git**: GitKraken, SourceTree, CLI

---

## 🆘 Need Help?

1. **Check this guide first**
2. **Search existing documentation**
3. **Check GitHub issues**
4. **Ask on team Slack: #qloo-dev**
5. **Create a GitHub issue for bugs**
6. **Schedule 1:1 with team lead for complex issues**

Remember: There are no stupid questions! The team is here to help you succeed. 🎉