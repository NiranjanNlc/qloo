# Qloo Supermarket Layout Optimizer

A comprehensive Python application that leverages Qloo's AI-powered recommendation engine to optimize supermarket product layouts, generate combo offers, and analyze customer behavior patterns. The system includes a Streamlit web interface, FastAPI backend, and automated reporting pipeline with Airflow DAG scheduling.

## 🚀 Features

- **Smart Product Layout Optimization**: AI-powered recommendations for optimal product placement
- **Association Rule Mining**: Discover product relationships and customer behavior patterns  
- **Combo Offer Generation**: Automated creation of product bundle recommendations
- **Real-time Analytics Dashboard**: Interactive visualizations and KPI monitoring
- **Automated Reporting**: Weekly performance reports with embedded heatmaps
- **Mobile-Responsive Interface**: Optimized for all screen sizes and devices
- **Production-Ready API**: FastAPI service with OpenAPI documentation
- **Containerized Deployment**: Docker-based orchestration with monitoring

## 🏗️ Architecture

```
qloo/
├── app.py                      # Main Streamlit application
├── association_api.py          # FastAPI service for combos/associations
├── src/
│   ├── qloo_client.py         # Qloo API client implementation
│   ├── association_engine.py  # Association rule mining algorithms
│   ├── layout_optimizer.py    # Store layout optimization logic
│   ├── weekly_reports.py      # Automated report generation
│   └── models.py              # Data models and schemas
├── pages/                     # Streamlit multi-page app
│   ├── 1_Product_Catalog.py   # Product management interface
│   ├── 2_Association_Rules.py # Association analysis dashboard
│   ├── 3_Layout_Optimizer.py  # Layout optimization interface
│   ├── 4_Shelf_Placements.py  # Shelf placement recommendations
│   ├── 5_Product_Flow_Analysis.py # Customer flow analysis
│   ├── 6_Combo_Offers.py      # Combo offer generation
│   ├── 7_DAG_Status.py        # Airflow DAG monitoring
│   └── 8_Interactive_Store_Map.py # Visual store layout
├── dags/
│   └── weekly_report_dag.py   # Airflow DAG for automated reporting
├── tests/                     # Comprehensive test suite
├── benchmarks/               # Performance testing
├── docs/                     # Technical documentation
├── data/                     # Sample datasets
└── scripts/                  # Database setup and utilities
```

## 🛠️ Technology Stack

**Backend & APIs**:
- FastAPI - High-performance API framework
- Qloo API - AI-powered recommendation engine
- Redis - Caching layer
- PostgreSQL - Production database

**Data Pipeline**:
- Apache Airflow - Workflow orchestration  
- Pandas - Data processing and analysis
- Scikit-learn - Machine learning algorithms
- NetworkX - Graph analysis

**Frontend & Visualization**:
- Streamlit - Web application framework
- Plotly - Interactive visualizations
- CSS3 - Responsive design

**Infrastructure**:
- Docker & Docker Compose - Containerization
- Gunicorn + Uvicorn - Production ASGI server
- Nginx - Reverse proxy (in production)

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Docker and Docker Compose
- Git

### Local Development Setup

1. **Clone the repository**:
```bash
git clone <repository-url>
cd qloo
```

2. **Set up environment variables**:
```bash
cp env.example .env
# Edit .env with your Qloo API key and other configurations
```

3. **Install dependencies**:
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Unix/macOS  
source venv/bin/activate

pip install -e .
```

4. **Start the application**:
```bash
# Option 1: Development mode
streamlit run app.py

# Option 2: Full stack with Docker
docker-compose up -d
```

5. **Access the application**:
- Streamlit UI: http://localhost:8501
- FastAPI docs: http://localhost:8000/docs
- API endpoints: http://localhost:8000

### Docker Deployment

For production deployment with full orchestration:

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

Services include:
- **Streamlit**: Web interface (port 8501)
- **FastAPI**: API service (port 8000)  
- **PostgreSQL**: Database (port 5432)
- **Redis**: Cache (port 6379)
- **Nginx**: Reverse proxy (port 80)

## 📊 API Endpoints

The FastAPI service provides the following endpoints:

### Core Endpoints
- `GET /combos` - Generate product combo offers (returns 50+ combinations)
- `GET /associations` - Get product association rules
- `GET /products` - Product catalog management
- `GET /recommendations` - AI-powered product recommendations

### Documentation
- `GET /docs` - Interactive Swagger UI
- `GET /redoc` - ReDoc documentation
- `GET /openapi.json` - OpenAPI 3.0 specification

### Example Usage

```python
import requests

# Get combo offers
response = requests.get("http://localhost:8000/combos?limit=20")
combos = response.json()

# Get product associations
response = requests.get("http://localhost:8000/associations?product_id=12345")
associations = response.json()
```

## 🔧 Configuration

### Environment Variables

Create a `.env` file with the following variables:

```bash
# Qloo API Configuration
QLOO_API_KEY=your_qloo_api_key_here
QLOO_BASE_URL=https://hackathon.api.qloo.com

# Database Configuration  
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=qloo_supermarket
POSTGRES_USER=qloo_user
POSTGRES_PASSWORD=qloo_password

# Redis Configuration
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0

# Airflow Configuration
AIRFLOW_HOME=/opt/airflow
SLACK_WEBHOOK_URL=your_slack_webhook_url

# Application Settings
DEBUG=False
LOG_LEVEL=INFO
```

### Qloo API Setup

The application uses Qloo's search endpoint for product recommendations:

```python
# Correct API usage pattern
response = requests.get(
    "https://hackathon.api.qloo.com/search",
    params={"query": "apple", "limit": 10},
    headers={"x-api-key": "YOUR_API_KEY"}
)
```

**Important**: Use the `/search` endpoint with `query` parameter, not `/recommendations` with `type` parameter.

## 🧪 Testing

### Running Tests

```bash
# Run all tests
python -m pytest

# Run specific test categories
python -m pytest tests/unit/
python -m pytest tests/integration/
python -m pytest tests/e2e/

# Run with coverage
python -m pytest --cov=src tests/

# Performance benchmarks
python benchmarks/optimizer_benchmark.py
```

### Test Coverage

- **Unit Tests**: Algorithm logic, data processing, API clients
- **Integration Tests**: Database operations, API endpoints, Qloo integration
- **E2E Tests**: Complete workflow validation, UI interactions
- **Performance Tests**: Scalability testing (up to 15k SKUs)

### Key Test Results

- ✅ API response time: <2s for 50 combos
- ✅ 10k SKU optimization: <180s (target met)
- ✅ DAG success rate: 95%+
- ✅ Mobile responsiveness: All screen sizes
- ✅ Test coverage: 85%+

## 📈 Performance Benchmarks

The system is optimized for enterprise-scale operations:

| Dataset Size | Processing Time | Memory Usage | Status |
|--------------|-----------------|--------------|--------|
| 1k SKUs | ~15s | <500MB | ✅ Optimal |
| 5k SKUs | ~75s | <1GB | ✅ Good |  
| 10k SKUs | ~160s | <2GB | ✅ **Target Met** |
| 15k SKUs | ~320s | <3GB | ⚠️ Acceptable |

**Performance Features**:
- Parallel processing for large datasets
- Memory-efficient algorithms
- Intelligent caching with Redis
- Optimized database queries
- Bundle size <1MB for frontend

## 📋 Airflow DAG

Automated weekly reporting pipeline scheduled for Fridays at 02:00:

### DAG Tasks
1. **Extract Transaction Data** - Load weekly sales data
2. **Generate Association Rules** - Mine product relationships  
3. **Generate Combo Offers** - Create product combinations
4. **Generate Weekly Report** - Build HTML/JSON reports
5. **Notify Success/Failure** - Slack notifications

### Monitoring
- Real-time DAG status in Streamlit UI (page 7)
- Automated Slack notifications
- Performance metrics tracking
- Error handling and retries

## 📱 Mobile Optimization

The application is fully responsive with mobile-first design:

- **Bundle Size**: <1MB total
- **Mobile Breakpoints**: 768px, 480px
- **Touch-Optimized**: Mobile-friendly controls
- **Progressive Loading**: Lazy loading of heavy components
- **Offline Capability**: Cached static assets

## 📚 Documentation

### API Documentation
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **Algorithm Parameters**: `docs/algorithm_parameters.md`

### Development Guides
- **Coding Standards**: `docs/CODING_STANDARDS.md`
- **Developer Onboarding**: `docs/DEVELOPER_ONBOARDING.md`
- **Quick Reference**: `docs/QUICK_REFERENCE.md`

## 🐛 Troubleshooting

### Common Issues

1. **Qloo API Connection**:
   ```bash
   # Test API connectivity
   python test_qloo_api.py
   ```

2. **Database Connection**:
   ```bash
   # Initialize database
   python scripts/migrations/001_create_oltp_schema.py
   ```

3. **Docker Issues**:
   ```bash
   # Reset containers
   docker-compose down -v
   docker-compose up -d --build
   ```

### Performance Issues

- Check system resources (CPU, memory)
- Monitor Redis cache hit rates
- Review database query performance
- Verify network connectivity to Qloo API

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For support and questions:

1. **Check Documentation**: Review the `/docs` directory
2. **Run Diagnostics**: `python test_qloo_api.py`
3. **Check Logs**: `docker-compose logs -f`
4. **Performance Issues**: `python benchmarks/optimizer_benchmark.py`

---

**Status**: ✅ Production Ready | **Performance**: ✅ Targets Met | **Documentation**: ✅ Complete 