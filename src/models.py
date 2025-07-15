"""
Pydantic Models for Qloo API Response Validation

This module contains Pydantic models for validating API responses,
data structures, and internal representations used throughout the
supermarket layout optimizer application.
"""

from typing import List, Optional, Dict, Any, Union
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field, validator, root_validator


class AssociationType(str, Enum):
    """Types of product associations."""
    FREQUENTLY_BOUGHT_TOGETHER = "frequently_bought_together"
    SEARCH_RESULT = "search_result"
    CATEGORY_RELATED = "category_related"
    RECOMMENDED = "recommended"


class CategoryType(str, Enum):
    """Product category types."""
    PRODUCE = "Produce"
    DAIRY = "Dairy"
    MEAT = "Meat"
    BAKERY = "Bakery"
    BEVERAGES = "Beverages"
    SNACKS = "Snacks"
    UNCATEGORIZED = "Uncategorized"


class QlooSearchResult(BaseModel):
    """Model for individual search result from Qloo API."""
    id: Optional[str] = Field(None, description="Product or entity ID")
    name: Optional[str] = Field(None, description="Product or entity name")
    category: Optional[str] = Field(None, description="Product category")
    confidence_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Confidence score for the result")
    relevance_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Relevance score for the result")
    
    @validator('confidence_score', 'relevance_score')
    def score_must_be_valid(cls, v):
        if v is not None and (v < 0 or v > 1):
            raise ValueError('Score must be between 0 and 1')
        return v

    class Config:
        extra = "allow"  # Allow additional fields from API


class QlooSearchResponse(BaseModel):
    """Model for Qloo API search response."""
    results: List[QlooSearchResult] = Field(default_factory=list, description="List of search results")
    total_results: Optional[int] = Field(None, ge=0, description="Total number of results available")
    query: Optional[str] = Field(None, description="Original search query")
    
    @validator('results')
    def results_must_be_list(cls, v):
        if not isinstance(v, list):
            raise ValueError('Results must be a list')
        return v

    class Config:
        extra = "allow"


class ProductAssociation(BaseModel):
    """Model for product association data."""
    associated_product_id: str = Field(..., description="ID of the associated product")
    association_strength: float = Field(..., ge=0.0, le=1.0, description="Strength of the association")
    association_type: AssociationType = Field(..., description="Type of association")
    support: Optional[float] = Field(None, ge=0.0, le=1.0, description="Support metric from association rules")
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="Confidence metric from association rules")
    lift: Optional[float] = Field(None, gt=0.0, description="Lift metric from association rules")

    @validator('association_strength', 'support', 'confidence')
    def score_range_validation(cls, v):
        if v is not None and (v < 0 or v > 1):
            raise ValueError('Score must be between 0 and 1')
        return v

    @validator('lift')
    def lift_validation(cls, v):
        if v is not None and v <= 0:
            raise ValueError('Lift must be greater than 0')
        return v


class ProductAssociationsResponse(BaseModel):
    """Model for product associations API response."""
    product_id: str = Field(..., description="ID of the base product")
    associations: List[ProductAssociation] = Field(default_factory=list, description="List of associated products")
    total_associations: Optional[int] = Field(None, ge=0, description="Total number of associations found")
    
    @validator('associations')
    def associations_must_be_list(cls, v):
        if not isinstance(v, list):
            raise ValueError('Associations must be a list')
        return v


class CategoryInsights(BaseModel):
    """Model for category insights data."""
    category: str = Field(..., description="Category name")
    total_products: int = Field(..., ge=0, description="Total number of products in category")
    total_results: Optional[int] = Field(None, ge=0, description="Total search results for category")
    top_products: List[QlooSearchResult] = Field(default_factory=list, description="Top products in category")
    seasonal_trends: Optional[Dict[str, float]] = Field(None, description="Seasonal trend data")
    search_source: Optional[str] = Field(None, description="Source of the insights data")

    @validator('seasonal_trends')
    def validate_seasonal_trends(cls, v):
        if v is not None:
            valid_seasons = {'spring', 'summer', 'fall', 'winter', 'autumn'}
            for season, score in v.items():
                if season.lower() not in valid_seasons:
                    raise ValueError(f'Invalid season: {season}')
                if not isinstance(score, (int, float)) or score < 0 or score > 1:
                    raise ValueError(f'Invalid score for season {season}: must be between 0 and 1')
        return v


class Product(BaseModel):
    """Model for product catalog data."""
    product_id: int = Field(..., ge=1, description="Unique product identifier")
    product_name: str = Field(..., min_length=1, max_length=200, description="Product name")
    category: CategoryType = Field(..., description="Product category")
    
    @validator('product_name')
    def name_must_not_be_empty(cls, v):
        if not v.strip():
            raise ValueError('Product name cannot be empty')
        return v.strip().title()

    class Config:
        use_enum_values = True


class DatabaseProduct(Product):
    """Extended product model for database operations."""
    created_at: Optional[datetime] = Field(None, description="Record creation timestamp")
    updated_at: Optional[datetime] = Field(None, description="Record update timestamp")


class AssociationRule(BaseModel):
    """Model for association rule mining results."""
    rule_id: Optional[int] = Field(None, description="Unique rule identifier")
    antecedent_product_id: int = Field(..., description="Product ID in the 'if' part of the rule")
    consequent_product_id: int = Field(..., description="Product ID in the 'then' part of the rule")
    support: float = Field(..., ge=0.0, le=1.0, description="Support metric")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence metric")
    lift: float = Field(..., gt=0.0, description="Lift metric")
    
    @validator('antecedent_product_id', 'consequent_product_id')
    def product_ids_must_be_positive(cls, v):
        if v <= 0:
            raise ValueError('Product ID must be positive')
        return v
    
    @root_validator(skip_on_failure=True)
    def antecedent_consequent_must_differ(cls, values):
        antecedent = values.get('antecedent_product_id')
        consequent = values.get('consequent_product_id')
        if antecedent == consequent:
            raise ValueError('Antecedent and consequent products must be different')
        return values


class LayoutPosition(BaseModel):
    """Model for product layout position in store."""
    x: float = Field(..., description="X coordinate in store layout")
    y: float = Field(..., description="Y coordinate in store layout") 
    zone: Optional[str] = Field(None, description="Store zone or section")
    aisle: Optional[int] = Field(None, ge=1, description="Aisle number")
    shelf: Optional[int] = Field(None, ge=1, description="Shelf number")


class LayoutRecommendation(BaseModel):
    """Model for layout optimization recommendations."""
    product_id: int = Field(..., description="Product to be positioned")
    recommended_position: LayoutPosition = Field(..., description="Recommended position")
    reasoning: str = Field(..., description="Explanation for the recommendation")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Confidence in recommendation")
    related_products: List[int] = Field(default_factory=list, description="Products that influenced this recommendation")


class OptimizationResult(BaseModel):
    """Model for complete layout optimization results."""
    total_products: int = Field(..., ge=0, description="Total number of products optimized")
    recommendations: List[LayoutRecommendation] = Field(..., description="List of layout recommendations")
    optimization_score: float = Field(..., ge=0.0, le=1.0, description="Overall optimization quality score")
    execution_time: float = Field(..., ge=0.0, description="Time taken for optimization in seconds")
    algorithm_used: str = Field(..., description="Name of optimization algorithm used")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Optimization timestamp")


class APIHealthResponse(BaseModel):
    """Model for API health check response."""
    status: str = Field(..., description="Health status")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Health check timestamp")
    version: Optional[str] = Field(None, description="API version")
    uptime: Optional[float] = Field(None, description="API uptime in seconds")
    
    @validator('status')
    def status_must_be_valid(cls, v):
        valid_statuses = {'ok', 'healthy', 'unhealthy', 'error'}
        if v.lower() not in valid_statuses:
            raise ValueError(f'Invalid status: {v}')
        return v.lower()


class ErrorResponse(BaseModel):
    """Model for API error responses."""
    error: Dict[str, Any] = Field(..., description="Error details")
    status_code: Optional[int] = Field(None, description="HTTP status code")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Error timestamp")
    
    @validator('status_code')
    def validate_status_code(cls, v):
        if v is not None and (v < 100 or v > 599):
            raise ValueError('Invalid HTTP status code')
        return v


class CatalogStatistics(BaseModel):
    """Model for catalog statistics."""
    total_products: int = Field(..., ge=0, description="Total number of products")
    categories: Dict[str, int] = Field(..., description="Count of products by category")
    sample_products: List[str] = Field(default_factory=list, description="Sample product names")
    last_updated: Optional[datetime] = Field(None, description="Last catalog update timestamp")
    
    @validator('categories')
    def categories_counts_must_be_non_negative(cls, v):
        for category, count in v.items():
            if count < 0:
                raise ValueError(f'Category count for {category} cannot be negative')
        return v


# Utility functions for model validation and parsing

def parse_qloo_search_response(data: Dict[str, Any]) -> QlooSearchResponse:
    """
    Parse raw API response data into QlooSearchResponse model.
    
    Args:
        data: Raw response data from Qloo API
        
    Returns:
        Validated QlooSearchResponse instance
        
    Raises:
        ValidationError: If data doesn't match expected structure
    """
    return QlooSearchResponse.parse_obj(data)


def parse_associations(data: Dict[str, List[Dict[str, Any]]]) -> Dict[str, ProductAssociationsResponse]:
    """
    Parse associations data into validated models.
    
    Args:
        data: Raw associations data mapping product IDs to association lists
        
    Returns:
        Dictionary mapping product IDs to ProductAssociationsResponse instances
        
    Raises:
        ValidationError: If data doesn't match expected structure
    """
    parsed_associations = {}
    for product_id, associations in data.items():
        association_models = [ProductAssociation.parse_obj(assoc) for assoc in associations]
        parsed_associations[product_id] = ProductAssociationsResponse(
            product_id=product_id,
            associations=association_models,
            total_associations=len(association_models)
        )
    return parsed_associations


def validate_catalog_data(products: List[Dict[str, Any]]) -> List[Product]:
    """
    Validate and parse catalog data into Product models.
    
    Args:
        products: List of raw product data dictionaries
        
    Returns:
        List of validated Product instances
        
    Raises:
        ValidationError: If any product data is invalid
    """
    return [Product.parse_obj(product) for product in products] 