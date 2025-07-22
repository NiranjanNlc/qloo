"""
FastAPI Association API Service

This service provides endpoints for retrieving product combinations and offers
based on association rule mining and optimization algorithms.
"""

from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import uvicorn
import sys
import os
from datetime import datetime, timedelta
import logging
from dataclasses import asdict
import statistics

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from models import Combo, ComboGenerator, Product, AssociationType
    from association_engine import AprioriAssociationEngine
    from qloo_client import create_qloo_client
    from price_api import PriceAPIStub
    from weekly_reports import WeeklyKPICalculator, WeeklyReportGenerator
except ImportError as e:
    logging.error(f"Import error: {e}")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Qloo Supermarket Association API",
    description="API for retrieving product associations, combinations, and offers",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for API
class ComboOffer(BaseModel):
    """Product combination offer model for API responses."""
    combo_id: str = Field(..., description="Unique identifier for the combo")
    name: str = Field(..., description="Human-readable name for the combo")
    products: List[int] = Field(..., description="List of product IDs in the combo")
    product_names: List[str] = Field(default_factory=list, description="Names of products in the combo")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Confidence score for the combo")
    support: float = Field(..., ge=0.0, le=1.0, description="Support metric from association rules")
    lift: float = Field(..., gt=0.0, description="Lift metric from association rules")
    expected_discount_percent: Optional[float] = Field(None, ge=0.0, le=100.0, description="Expected discount percentage")
    category_mix: Optional[List[str]] = Field(None, description="Categories included in combo")
    estimated_revenue: Optional[float] = Field(None, ge=0.0, description="Estimated revenue from combo")
    is_active: bool = Field(True, description="Whether the combo is currently active")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")

class CombosResponse(BaseModel):
    """API response model for combos endpoint."""
    total_combos: int = Field(..., ge=0, description="Total number of combos returned")
    combos: List[ComboOffer] = Field(..., description="List of combo offers")
    generated_at: datetime = Field(default_factory=datetime.now, description="Response generation timestamp")
    filters_applied: Dict[str, Any] = Field(default_factory=dict, description="Filters applied to the results")

class ProductAssociation(BaseModel):
    """Product association model for API responses."""
    product_id: int = Field(..., description="Product ID")
    product_name: str = Field(..., description="Product name")
    associated_product_id: int = Field(..., description="Associated product ID")
    associated_product_name: str = Field(..., description="Associated product name")
    association_strength: float = Field(..., ge=0.0, le=1.0, description="Strength of association")
    association_type: str = Field(..., description="Type of association")

class AssociationsResponse(BaseModel):
    """API response model for associations endpoint."""
    product_id: int = Field(..., description="Base product ID")
    product_name: str = Field(..., description="Base product name")
    associations: List[ProductAssociation] = Field(..., description="List of product associations")
    total_associations: int = Field(..., ge=0, description="Total number of associations found")

class MetricsResponse(BaseModel):
    """Response model for metrics endpoint."""
    timestamp: str = Field(description="ISO timestamp when metrics were generated")
    period: str = Field(description="Time period for metrics (e.g., 'last_7_days')")
    kpis: Dict[str, Any] = Field(description="Key performance indicators")
    performance: Dict[str, Any] = Field(description="System performance metrics")
    business: Dict[str, Any] = Field(description="Business metrics")

# Global variables for caching
_combo_generator = None
_association_engine = None
_qloo_client = None
_products_cache = {}

def get_combo_generator():
    """Get or initialize combo generator."""
    global _combo_generator
    if _combo_generator is None:
        try:
            price_api = PriceAPIStub()
            _combo_generator = ComboGenerator(
                min_confidence=0.6,
                min_support=0.01,
                price_api=price_api
            )
            logger.info("Combo generator initialized")
        except Exception as e:
            logger.error(f"Failed to initialize combo generator: {e}")
            raise HTTPException(status_code=500, detail="Failed to initialize combo generator")
    return _combo_generator

def get_association_engine():
    """Get or initialize association engine."""
    global _association_engine
    if _association_engine is None:
        try:
            _association_engine = AprioriAssociationEngine()
            # In a real implementation, load and train with actual data
            logger.info("Association engine initialized")
        except Exception as e:
            logger.error(f"Failed to initialize association engine: {e}")
            raise HTTPException(status_code=500, detail="Failed to initialize association engine")
    return _association_engine

def load_mock_products():
    """Load mock product data for demonstration."""
    global _products_cache
    if not _products_cache:
        # Mock product data
        mock_products = [
            {"id": 1, "name": "Whole Milk", "category": "Dairy", "price": 3.99},
            {"id": 2, "name": "Greek Yogurt", "category": "Dairy", "price": 5.99},
            {"id": 3, "name": "Cheddar Cheese", "category": "Dairy", "price": 4.99},
            {"id": 11, "name": "Bananas", "category": "Produce", "price": 1.99},
            {"id": 12, "name": "Strawberries", "category": "Produce", "price": 4.99},
            {"id": 13, "name": "Spinach", "category": "Produce", "price": 2.99},
            {"id": 21, "name": "Ground Beef", "category": "Meat", "price": 8.99},
            {"id": 22, "name": "Chicken Breast", "category": "Meat", "price": 7.99},
            {"id": 23, "name": "Salmon Fillet", "category": "Meat", "price": 12.99},
            {"id": 31, "name": "Orange Juice", "category": "Beverages", "price": 3.49},
            {"id": 32, "name": "Cola", "category": "Beverages", "price": 2.99},
            {"id": 33, "name": "Sparkling Water", "category": "Beverages", "price": 1.99},
            {"id": 41, "name": "Potato Chips", "category": "Snacks", "price": 2.49},
            {"id": 42, "name": "Almonds", "category": "Snacks", "price": 6.99},
            {"id": 43, "name": "Granola Bars", "category": "Snacks", "price": 4.49},
            {"id": 51, "name": "Whole Wheat Bread", "category": "Bakery", "price": 2.99},
            {"id": 52, "name": "Croissants", "category": "Bakery", "price": 3.99},
            {"id": 53, "name": "Bagels", "category": "Bakery", "price": 3.49},
        ]
        _products_cache = {p["id"]: p for p in mock_products}
    return _products_cache

def generate_mock_combos(limit: int = 50, min_confidence: float = 0.5) -> List[Combo]:
    """Generate mock combo data for demonstration."""
    import random
    
    products = load_mock_products()
    product_ids = list(products.keys())
    
    # Predefined combo templates for realistic combinations
    combo_templates = [
        # Breakfast combos
        {"products": [1, 51, 31], "name": "Morning Essentials", "theme": "breakfast"},
        {"products": [2, 43, 12], "name": "Healthy Breakfast", "theme": "breakfast"},
        {"products": [53, 3, 31], "name": "Bagel Breakfast", "theme": "breakfast"},
        
        # Dinner combos
        {"products": [21, 13, 11], "name": "Beef Dinner Kit", "theme": "dinner"},
        {"products": [22, 1, 51], "name": "Chicken Meal", "theme": "dinner"},
        {"products": [23, 13, 33], "name": "Salmon Special", "theme": "dinner"},
        
        # Snack combos
        {"products": [41, 32, 42], "name": "Movie Night Pack", "theme": "snacks"},
        {"products": [43, 33, 12], "name": "Energy Boost", "theme": "snacks"},
        
        # Random combinations
        {"products": [1, 52, 31], "name": "Continental Breakfast", "theme": "breakfast"},
        {"products": [21, 22, 1], "name": "Protein Pack", "theme": "protein"},
    ]
    
    combos = []
    random.seed(42)  # For reproducible results
    
    for i in range(limit):
        if i < len(combo_templates):
            template = combo_templates[i]
            combo_products = template["products"]
            name = template["name"]
            theme = template["theme"]
        else:
            # Generate random combinations
            combo_size = random.randint(2, 4)
            combo_products = random.sample(product_ids, combo_size)
            name = f"Combo {i+1}"
            theme = "mixed"
        
        # Generate realistic metrics
        confidence = max(min_confidence, random.uniform(0.6, 0.95))
        support = random.uniform(0.01, 0.1)
        lift = random.uniform(1.1, 2.5)
        discount = random.uniform(5.0, 25.0)
        
        # Get categories
        categories = list(set(products[pid]["category"] for pid in combo_products))
        
        combo = Combo(
            combo_id=f"combo_{i+1:03d}",
            name=name,
            products=combo_products,
            confidence_score=confidence,
            support=support,
            lift=lift,
            expected_discount_percent=discount,
            category_mix=categories,
            created_at=datetime.now() - timedelta(days=random.randint(0, 30))
        )
        combos.append(combo)
    
    return combos

@app.get("/", response_model=Dict[str, str])
async def root():
    """API root endpoint with basic information."""
    return {
        "message": "Qloo Supermarket Association API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=Dict[str, str])
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }

@app.get("/combos", response_model=CombosResponse)
async def get_combos(
    limit: int = Query(50, ge=1, le=100, description="Maximum number of combos to return"),
    min_confidence: float = Query(0.5, ge=0.0, le=1.0, description="Minimum confidence score"),
    category: Optional[str] = Query(None, description="Filter by product category"),
    active_only: bool = Query(True, description="Return only active combos"),
    sort_by: str = Query("confidence", description="Sort field (confidence, lift, support, created_at)"),
    sort_desc: bool = Query(True, description="Sort in descending order")
):
    """
    Get product combination offers.
    
    This endpoint returns a list of product combinations with their associated
    metrics, discount information, and relevance scores.
    """
    try:
        # Generate mock combos
        combos = generate_mock_combos(limit=limit * 2, min_confidence=min_confidence)
        
        # Apply filters
        filtered_combos = []
        products = load_mock_products()
        
        for combo in combos:
            # Filter by confidence
            if combo.confidence_score < min_confidence:
                continue
            
            # Filter by active status
            if active_only and not combo.is_active:
                continue
            
            # Filter by category
            if category and category not in combo.category_mix:
                continue
            
            filtered_combos.append(combo)
        
        # Sort combos
        sort_key_map = {
            "confidence": lambda x: x.confidence_score,
            "lift": lambda x: x.lift,
            "support": lambda x: x.support,
            "created_at": lambda x: x.created_at
        }
        
        if sort_by in sort_key_map:
            filtered_combos.sort(key=sort_key_map[sort_by], reverse=sort_desc)
        
        # Limit results
        filtered_combos = filtered_combos[:limit]
        
        # Convert to API response format
        combo_offers = []
        for combo in filtered_combos:
            # Get product names
            product_names = [products.get(pid, {}).get("name", f"Product {pid}") for pid in combo.products]
            
            # Calculate estimated revenue (mock calculation)
            base_revenue = sum(products.get(pid, {}).get("price", 5.0) for pid in combo.products)
            estimated_revenue = base_revenue * combo.confidence_score * 0.8  # Assume 80% of base revenue
            
            combo_offer = ComboOffer(
                combo_id=combo.combo_id,
                name=combo.name,
                products=combo.products,
                product_names=product_names,
                confidence_score=combo.confidence_score,
                support=combo.support,
                lift=combo.lift,
                expected_discount_percent=combo.expected_discount_percent,
                category_mix=combo.category_mix,
                estimated_revenue=round(estimated_revenue, 2),
                is_active=combo.is_active,
                created_at=combo.created_at
            )
            combo_offers.append(combo_offer)
        
        # Prepare response
        response = CombosResponse(
            total_combos=len(combo_offers),
            combos=combo_offers,
            generated_at=datetime.now(),
            filters_applied={
                "limit": limit,
                "min_confidence": min_confidence,
                "category": category,
                "active_only": active_only,
                "sort_by": sort_by,
                "sort_desc": sort_desc
            }
        )
        
        logger.info(f"Returned {len(combo_offers)} combos with filters: {response.filters_applied}")
        return response
        
    except Exception as e:
        logger.error(f"Error generating combos: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/associations/{product_id}", response_model=AssociationsResponse)
async def get_product_associations(
    product_id: int,
    limit: int = Query(10, ge=1, le=50, description="Maximum number of associations to return"),
    min_strength: float = Query(0.1, ge=0.0, le=1.0, description="Minimum association strength")
):
    """
    Get product associations for a specific product.
    
    Returns products that are frequently bought together with the specified product.
    """
    try:
        products = load_mock_products()
        
        if product_id not in products:
            raise HTTPException(status_code=404, detail="Product not found")
        
        base_product = products[product_id]
        
        # Generate mock associations
        import random
        random.seed(product_id)  # Consistent associations for same product
        
        other_products = [p for p in products.values() if p["id"] != product_id]
        
        associations = []
        for other_product in other_products[:limit]:
            # Generate realistic association strength
            strength = random.uniform(0.1, 0.9)
            
            if strength >= min_strength:
                association = ProductAssociation(
                    product_id=product_id,
                    product_name=base_product["name"],
                    associated_product_id=other_product["id"],
                    associated_product_name=other_product["name"],
                    association_strength=strength,
                    association_type="frequently_bought_together"
                )
                associations.append(association)
        
        # Sort by strength
        associations.sort(key=lambda x: x.association_strength, reverse=True)
        associations = associations[:limit]
        
        response = AssociationsResponse(
            product_id=product_id,
            product_name=base_product["name"],
            associations=associations,
            total_associations=len(associations)
        )
        
        logger.info(f"Returned {len(associations)} associations for product {product_id}")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting associations for product {product_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/products", response_model=List[Dict[str, Any]])
async def get_products():
    """Get list of all available products."""
    try:
        products = load_mock_products()
        return list(products.values())
    except Exception as e:
        logger.error(f"Error getting products: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/combos/regenerate")
async def regenerate_combos(background_tasks: BackgroundTasks):
    """
    Trigger regeneration of combo offers in the background.
    
    This endpoint would typically trigger a background job to recompute
    product combinations based on updated transaction data.
    """
    def regenerate_task():
        logger.info("Starting combo regeneration task")
        # In a real implementation, this would:
        # 1. Load latest transaction data
        # 2. Retrain association models
        # 3. Generate new combos
        # 4. Update database
        # 5. Send notification
        import time
        time.sleep(2)  # Simulate processing time
        logger.info("Combo regeneration task completed")
    
    background_tasks.add_task(regenerate_task)
    
    return {
        "message": "Combo regeneration task started",
        "status": "accepted",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics(
    period: str = Query("last_7_days", description="Time period for metrics"),
    include_trends: bool = Query(True, description="Include trend calculations"),
    format: str = Query("detailed", description="Response format (summary, detailed)")
):
    """
    Get comprehensive KPI metrics for dashboard.
    
    This endpoint provides real-time metrics including:
    - Business KPIs (combos generated, confidence scores, etc.)
    - System performance metrics (API response times, throughput)
    - Trend analysis and comparisons
    """
    try:
        # Generate current timestamp
        timestamp = datetime.now().isoformat()
        
        # Get current combos data for metrics calculation
        combos_data = generate_mock_combos(limit=100)
        combos = combos_data
        
        # Calculate business KPIs
        total_combos = len(combos)
        avg_confidence = statistics.mean([combo.confidence_score for combo in combos]) if combos else 0
        avg_lift = statistics.mean([combo.lift for combo in combos]) if combos else 0
        avg_discount = statistics.mean([combo.expected_discount_percent for combo in combos]) if combos else 0
        high_confidence_rules = len([c for c in combos if c.confidence_score >= 0.8])
        total_revenue_potential = sum([combo.confidence_score * 100 for combo in combos]) # Mock revenue potential
        
        # System performance metrics (mock data for now)
        performance_metrics = {
            "api_response_time_ms": 145,  # Would be calculated from actual metrics
            "requests_per_minute": 24,
            "success_rate": 99.2,
            "cache_hit_rate": 78.5,
            "database_query_time_ms": 23,
            "memory_usage_mb": 256,
            "cpu_usage_percent": 15.3
        }
        
        # Business metrics
        business_metrics = {
            "total_combos_generated": total_combos,
            "high_confidence_rules": high_confidence_rules,
            "avg_confidence_score": round(avg_confidence, 3),
            "avg_lift_score": round(avg_lift, 3),
            "avg_discount_percent": round(avg_discount, 2),
            "total_revenue_potential": round(total_revenue_potential, 2),
            "active_categories": len(set([combo.category_mix for combo in combos])),
            "combo_success_rate": round((high_confidence_rules / total_combos * 100) if total_combos > 0 else 0, 1)
        }
        
        # KPI calculations with targets and trends
        kpis = {
            "combos_generated": {
                "current": total_combos,
                "target": 50,
                "previous": total_combos - 5,  # Mock previous value
                "trend": "up" if total_combos >= 50 else "stable",
                "status": "good" if total_combos >= 50 else "warning",
                "unit": "count"
            },
            "avg_confidence": {
                "current": round(avg_confidence, 3),
                "target": 0.85,
                "previous": round(avg_confidence - 0.02, 3),
                "trend": "up" if avg_confidence >= 0.85 else "stable",
                "status": "good" if avg_confidence >= 0.80 else "warning",
                "unit": "score"
            },
            "revenue_potential": {
                "current": round(total_revenue_potential, 0),
                "target": 10000,
                "previous": round(total_revenue_potential - 500, 0),
                "trend": "up",
                "status": "good" if total_revenue_potential >= 10000 else "warning",
                "unit": "currency"
            },
            "api_performance": {
                "current": performance_metrics["api_response_time_ms"],
                "target": 200,
                "previous": 168,
                "trend": "stable",
                "status": "good" if performance_metrics["api_response_time_ms"] <= 200 else "warning",
                "unit": "ms"
            }
        }
        
        # Add trend calculations if requested
        if include_trends:
            for kpi_name, kpi_data in kpis.items():
                if kpi_data["previous"] is not None:
                    change = kpi_data["current"] - kpi_data["previous"]
                    change_percent = (change / kpi_data["previous"] * 100) if kpi_data["previous"] != 0 else 0
                    kpi_data["change"] = round(change, 2)
                    kpi_data["change_percent"] = round(change_percent, 1)
        
        # Format response based on requested format
        if format == "summary":
            # Return simplified metrics for mobile/quick view
            return MetricsResponse(
                timestamp=timestamp,
                period=period,
                kpis={k: {"current": v["current"], "status": v["status"]} for k, v in kpis.items()},
                performance={"response_time": performance_metrics["api_response_time_ms"], 
                           "success_rate": performance_metrics["success_rate"]},
                business={"total_combos": business_metrics["total_combos_generated"],
                         "revenue_potential": business_metrics["total_revenue_potential"]}
            )
        else:
            # Return detailed metrics
            return MetricsResponse(
                timestamp=timestamp,
                period=period,
                kpis=kpis,
                performance=performance_metrics,
                business=business_metrics
            )
            
    except Exception as e:
        logger.error(f"Error generating metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating metrics: {str(e)}")

if __name__ == "__main__":
    # For development only
    uvicorn.run(
        "association_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    ) 