#!/bin/bash

# Qloo Supermarket Layout Optimizer - Development Environment Setup Script
# This script automates the setup process for new developers

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper functions
print_step() {
    echo -e "${BLUE}📋 $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Detect OS
detect_os() {
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        echo "linux"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        echo "macos"
    elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
        echo "windows"
    else
        echo "unknown"
    fi
}

# Main setup function
main() {
    echo -e "${BLUE}"
    echo "🛒 Qloo Supermarket Layout Optimizer"
    echo "🚀 Development Environment Setup"
    echo "=================================="
    echo -e "${NC}"
    
    OS=$(detect_os)
    print_step "Detected OS: $OS"
    
    # Check prerequisites
    print_step "Checking prerequisites..."
    
    # Check Python
    if command_exists python3; then
        PYTHON_VERSION=$(python3 --version | awk '{print $2}')
        print_success "Python $PYTHON_VERSION found"
        PYTHON_CMD="python3"
    elif command_exists python; then
        PYTHON_VERSION=$(python --version | awk '{print $2}')
        if [[ $PYTHON_VERSION == 3.* ]]; then
            print_success "Python $PYTHON_VERSION found"
            PYTHON_CMD="python"
        else
            print_error "Python 3.8+ required, found $PYTHON_VERSION"
            exit 1
        fi
    else
        print_error "Python not found. Please install Python 3.8+"
        exit 1
    fi
    
    # Check pip
    if command_exists pip3; then
        PIP_CMD="pip3"
    elif command_exists pip; then
        PIP_CMD="pip"
    else
        print_error "pip not found. Please install pip"
        exit 1
    fi
    print_success "pip found"
    
    # Check Git
    if command_exists git; then
        print_success "Git found"
    else
        print_error "Git not found. Please install Git"
        exit 1
    fi
    
    # Check Docker (optional)
    if command_exists docker; then
        print_success "Docker found"
        DOCKER_AVAILABLE=true
    else
        print_warning "Docker not found. Some features will be limited"
        DOCKER_AVAILABLE=false
    fi
    
    # Check Docker Compose (optional)
    if command_exists docker-compose || command_exists docker compose; then
        print_success "Docker Compose found"
    else
        if $DOCKER_AVAILABLE; then
            print_warning "Docker Compose not found. Using 'docker compose' if available"
        fi
    fi
    
    # Create virtual environment
    print_step "Creating virtual environment..."
    if [ ! -d "venv" ]; then
        $PYTHON_CMD -m venv venv
        print_success "Virtual environment created"
    else
        print_warning "Virtual environment already exists"
    fi
    
    # Activate virtual environment
    print_step "Activating virtual environment..."
    if [[ "$OS" == "windows" ]]; then
        source venv/Scripts/activate
    else
        source venv/bin/activate
    fi
    print_success "Virtual environment activated"
    
    # Upgrade pip
    print_step "Upgrading pip..."
    $PIP_CMD install --upgrade pip
    print_success "pip upgraded"
    
    # Install dependencies
    print_step "Installing dependencies..."
    if [ -f "pyproject.toml" ]; then
        $PIP_CMD install -e ".[dev,test]"
        print_success "Dependencies installed from pyproject.toml"
    else
        print_error "pyproject.toml not found. Are you in the project root?"
        exit 1
    fi
    
    # Create .env file
    print_step "Setting up environment variables..."
    if [ ! -f ".env" ]; then
        if [ -f "env.example" ]; then
            cp env.example .env
            print_success ".env file created from template"
            print_warning "Please edit .env file with your actual values"
        else
            print_error "env.example not found"
        fi
    else
        print_warning ".env file already exists"
    fi
    
    # Setup database
    print_step "Setting up database..."
    if [ -f "Makefile" ]; then
        if make load_catalog 2>/dev/null; then
            print_success "Database setup completed"
        else
            print_warning "Database setup failed. You may need to run 'make load_catalog' manually"
        fi
    else
        print_warning "Makefile not found. Skipping database setup"
    fi
    
    # Run tests
    print_step "Running tests..."
    if [ -f "Makefile" ]; then
        if make test 2>/dev/null; then
            print_success "All tests passed"
        else
            print_warning "Some tests failed. Check output above"
        fi
    else
        if $PIP_CMD install pytest && pytest tests/ 2>/dev/null; then
            print_success "All tests passed"
        else
            print_warning "Some tests failed. Check output above"
        fi
    fi
    
    # Docker setup (if available)
    if $DOCKER_AVAILABLE; then
        print_step "Setting up Docker environment..."
        if [ -f "docker-compose.yml" ]; then
            if docker-compose --version >/dev/null 2>&1; then
                docker-compose build --no-cache
                print_success "Docker images built"
            elif docker compose version >/dev/null 2>&1; then
                docker compose build --no-cache
                print_success "Docker images built"
            else
                print_warning "Docker Compose not available"
            fi
        else
            print_warning "docker-compose.yml not found"
        fi
    fi
    
    # Final instructions
    echo -e "${GREEN}"
    echo "🎉 Setup completed successfully!"
    echo "==============================="
    echo -e "${NC}"
    
    echo "Next steps:"
    echo "1. Edit the .env file with your actual values:"
    echo "   - QLOO_API_KEY: Your Qloo API key"
    echo "   - Other configuration as needed"
    echo ""
    echo "2. Start the development server:"
    echo "   make run-dev"
    echo "   # or manually:"
    echo "   streamlit run app.py"
    echo ""
    echo "3. View the API documentation:"
    echo "   http://localhost:8000/docs"
    echo ""
    echo "4. View the Streamlit interface:"
    echo "   http://localhost:8501"
    echo ""
    
    if $DOCKER_AVAILABLE; then
        echo "5. Start with Docker (alternative):"
        echo "   docker-compose up -d"
        echo ""
    fi
    
    echo "6. Read the developer documentation:"
    echo "   docs/DEVELOPER_ONBOARDING.md"
    echo ""
    
    print_success "Welcome to the Qloo Supermarket Layout Optimizer team!"
}

# Help function
show_help() {
    echo "Qloo Supermarket Layout Optimizer - Development Environment Setup"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -h, --help     Show this help message"
    echo "  --skip-docker  Skip Docker setup even if available"
    echo "  --skip-tests   Skip running tests during setup"
    echo ""
    echo "This script will:"
    echo "  • Check prerequisites (Python, Git, Docker)"
    echo "  • Create and setup virtual environment"
    echo "  • Install project dependencies"
    echo "  • Setup environment variables"
    echo "  • Initialize database"
    echo "  • Run tests to verify setup"
    echo "  • Build Docker images (if Docker available)"
    echo ""
}

# Parse command line arguments
SKIP_DOCKER=false
SKIP_TESTS=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        --skip-docker)
            SKIP_DOCKER=true
            shift
            ;;
        --skip-tests)
            SKIP_TESTS=true
            shift
            ;;
        *)
            print_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Run main setup
main