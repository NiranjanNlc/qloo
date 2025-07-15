# Qloo Supermarket Layout Optimizer - Makefile
#
# This Makefile provides convenient targets for common development and deployment tasks
# including data loading, testing, and application startup.

# Variables
PYTHON := python
PIP := pip
VENV_DIR := venv
SCRIPTS_DIR := scripts
SRC_DIR := src
TESTS_DIR := tests
DATA_DIR := data

# Default target
.PHONY: help
help:  ## Show this help message
	@echo "Qloo Supermarket Layout Optimizer - Available Commands:"
	@echo ""
	@awk 'BEGIN {FS = ":.*##"; printf "Usage: make \033[36m<target>\033[0m\n\nTargets:\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) }' $(MAKEFILE_LIST)

##@ Environment Setup
.PHONY: install
install:  ## Install dependencies
	$(PIP) install -e .

.PHONY: install-dev
install-dev:  ## Install development dependencies
	$(PIP) install -e ".[dev,test]"

.PHONY: venv
venv:  ## Create virtual environment
	$(PYTHON) -m venv $(VENV_DIR)
	@echo "Virtual environment created. Activate with:"
	@echo "  Windows: $(VENV_DIR)\\Scripts\\activate"
	@echo "  Unix/Mac: source $(VENV_DIR)/bin/activate"

##@ Data Management
.PHONY: load_catalog
load_catalog:  ## Load grocery catalog into SQLite database
	@echo "🚀 Loading grocery catalog into database..."
	$(PYTHON) $(SCRIPTS_DIR)/load_catalog.py
	@echo "✅ Catalog loading complete!"

.PHONY: validate_catalog
validate_catalog:  ## Validate catalog CSV file without loading
	@echo "🔍 Validating catalog data..."
	$(PYTHON) $(SCRIPTS_DIR)/load_catalog.py --validate

.PHONY: catalog_stats
catalog_stats:  ## Show catalog statistics from database
	@echo "📊 Retrieving catalog statistics..."
	$(PYTHON) $(SCRIPTS_DIR)/load_catalog.py --stats

.PHONY: setup_db
setup_db:  ## Initialize database schema
	@echo "🏗️  Setting up database schema..."
	$(PYTHON) $(SRC_DIR)/database_setup.py
	@echo "✅ Database schema ready!"

.PHONY: clean_data
clean_data:  ## Remove database and cache files
	@echo "🧹 Cleaning data files..."
	-rm -f $(DATA_DIR)/associations.db
	-rm -rf __pycache__ $(SRC_DIR)/__pycache__ $(TESTS_DIR)/__pycache__
	-rm -rf .pytest_cache
	@echo "✅ Data files cleaned!"

##@ Testing
.PHONY: test
test:  ## Run all tests
	@echo "🧪 Running tests..."
	pytest $(TESTS_DIR) -v --cov=$(SRC_DIR)

.PHONY: test-retry
test-retry:  ## Run retry wrapper tests specifically
	@echo "🔄 Testing retry wrapper functionality..."
	pytest $(TESTS_DIR)/test_retry_wrapper.py -v

.PHONY: test-association
test-association:  ## Run association engine tests
	@echo "🔗 Testing association engine..."
	pytest $(TESTS_DIR)/test_association_engine.py -v

.PHONY: test-api
test-api:  ## Test Qloo API integration
	@echo "🌐 Testing API integration..."
	$(PYTHON) test_qloo_api.py

##@ Code Quality
.PHONY: lint
lint:  ## Run code linting
	@echo "🔍 Running linters..."
	flake8 $(SRC_DIR) $(TESTS_DIR) $(SCRIPTS_DIR)
	mypy $(SRC_DIR)

.PHONY: format
format:  ## Format code with black and isort
	@echo "🎨 Formatting code..."
	black $(SRC_DIR) $(TESTS_DIR) $(SCRIPTS_DIR)
	isort $(SRC_DIR) $(TESTS_DIR) $(SCRIPTS_DIR)

.PHONY: check
check: lint test  ## Run linting and tests

##@ Application
.PHONY: run
run:  ## Start Streamlit application
	@echo "🚀 Starting Qloo Supermarket Layout Optimizer..."
	streamlit run app.py

.PHONY: demo
demo:  ## Run API demo script
	@echo "🎬 Running API demo..."
	$(PYTHON) working_api_demo.py

.PHONY: discover
discover:  ## Run API discovery script
	@echo "🔍 Discovering API endpoints..."
	$(PYTHON) discover_api.py

##@ Development
.PHONY: dev-setup
dev-setup: venv install-dev setup_db load_catalog  ## Complete development setup
	@echo ""
	@echo "🎉 Development environment setup complete!"
	@echo ""
	@echo "Next steps:"
	@echo "  1. Activate virtual environment:"
	@echo "     Windows: $(VENV_DIR)\\Scripts\\activate"
	@echo "     Unix/Mac: source $(VENV_DIR)/bin/activate"
	@echo "  2. Start the application: make run"
	@echo "  3. Run tests: make test"

.PHONY: dev-reset
dev-reset: clean_data setup_db load_catalog  ## Reset development environment
	@echo "🔄 Development environment reset complete!"

##@ Deployment
.PHONY: build
build:  ## Build application package
	@echo "📦 Building package..."
	$(PYTHON) -m build

.PHONY: clean
clean: clean_data  ## Clean all generated files
	@echo "🧹 Cleaning build artifacts..."
	-rm -rf build/ dist/ *.egg-info/
	-rm -rf .tox/ .coverage htmlcov/
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	@echo "✅ Clean complete!"

##@ Utilities
.PHONY: requirements
requirements:  ## Generate requirements.txt
	@echo "📋 Generating requirements.txt..."
	$(PIP) freeze > requirements.txt
	@echo "✅ Requirements saved to requirements.txt"

.PHONY: health
health:  ## Check system health and dependencies
	@echo "🩺 Checking system health..."
	@echo "Python version: $$($(PYTHON) --version)"
	@echo "Pip version: $$($(PIP) --version)"
	@echo "Virtual environment: $$(if [ -n "$$VIRTUAL_ENV" ]; then echo "Active ($$VIRTUAL_ENV)"; else echo "Not active"; fi)"
	@echo ""
	@echo "📦 Checking key dependencies..."
	@$(PYTHON) -c "import requests; print('✅ requests:', requests.__version__)" 2>/dev/null || echo "❌ requests not found"
	@$(PYTHON) -c "import pandas; print('✅ pandas:', pandas.__version__)" 2>/dev/null || echo "❌ pandas not found"
	@$(PYTHON) -c "import streamlit; print('✅ streamlit:', streamlit.__version__)" 2>/dev/null || echo "❌ streamlit not found"
	@echo ""
	@echo "🗃️  Checking data files..."
	@if [ -f "$(DATA_DIR)/grocery_catalog.csv" ]; then echo "✅ Catalog CSV found"; else echo "❌ Catalog CSV missing"; fi
	@if [ -f "$(DATA_DIR)/associations.db" ]; then echo "✅ Database found"; else echo "❌ Database missing (run 'make setup_db')"; fi

# Phony targets to avoid conflicts with files
.PHONY: all
all: install-dev test lint load_catalog  ## Run complete CI pipeline

# Make load_catalog the primary data loading target
.DEFAULT_GOAL := help 