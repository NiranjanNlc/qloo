"""
Parsing Helpers for Qloo API Responses

This module provides utility functions for parsing and validating API responses
using Pydantic models. It handles data transformation, validation, and error
handling for various API endpoints and data sources.
"""

import logging
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime
import json

from pydantic import ValidationError

from .models import (
    QlooSearchResponse,
    QlooSearchResult,
    ProductAssociationsResponse,
    ProductAssociation,
    CategoryInsights,
    Product,
    AssociationRule,
    APIHealthResponse,
    ErrorResponse,
    CatalogStatistics,
    AssociationType,
)


logger = logging.getLogger(__name__)


class ParsingError(Exception):
    """Custom exception for parsing errors."""

    pass


def parse_qloo_search_response(
    raw_data: Dict[str, Any], query: Optional[str] = None
) -> QlooSearchResponse:
    """
    Parse and validate Qloo API search response.

    Args:
        raw_data: Raw response data from Qloo search API
        query: Original search query (for metadata)

    Returns:
        Validated QlooSearchResponse instance

    Raises:
        ParsingError: If response format is invalid
        ValidationError: If data validation fails
    """
    try:
        # Ensure we have the expected structure
        if not isinstance(raw_data, dict):
            raise ParsingError(f"Expected dict, got {type(raw_data)}")

        # Extract results array
        results_data = raw_data.get("results", [])
        if not isinstance(results_data, list):
            raise ParsingError(f"Expected results to be list, got {type(results_data)}")

        # Add query metadata if provided
        if query:
            raw_data["query"] = query

        # Add total results count if not present
        if "total_results" not in raw_data:
            raw_data["total_results"] = len(results_data)

        # Validate and parse
        response = QlooSearchResponse.parse_obj(raw_data)

        logger.debug(
            f"Successfully parsed search response: {len(response.results)} results"
        )
        return response

    except ValidationError as e:
        logger.error(f"Validation error parsing search response: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error parsing search response: {e}")
        raise ParsingError(f"Failed to parse search response: {e}")


def parse_associations(
    raw_data: Dict[str, List[Dict[str, Any]]],
    association_type: AssociationType = AssociationType.SEARCH_RESULT,
) -> Dict[str, ProductAssociationsResponse]:
    """
    Parse product associations data with validation.

    This is the main parsing helper function mentioned in the task requirements.

    Args:
        raw_data: Raw associations data mapping product IDs to association lists
        association_type: Type of associations being parsed

    Returns:
        Dictionary mapping product IDs to validated ProductAssociationsResponse instances

    Raises:
        ParsingError: If associations format is invalid
        ValidationError: If data validation fails
    """
    try:
        if not isinstance(raw_data, dict):
            raise ParsingError(f"Expected dict, got {type(raw_data)}")

        parsed_associations = {}

        for product_id, associations_list in raw_data.items():
            try:
                # Validate associations list structure
                if not isinstance(associations_list, list):
                    logger.warning(
                        f"Invalid associations format for product {product_id}: expected list"
                    )
                    continue

                # Parse individual associations
                validated_associations = []
                for i, assoc_data in enumerate(associations_list):
                    try:
                        # Ensure required fields and set defaults
                        if not isinstance(assoc_data, dict):
                            logger.warning(
                                f"Skipping invalid association {i} for product {product_id}"
                            )
                            continue

                        # Set default association type if not specified
                        if "association_type" not in assoc_data:
                            assoc_data["association_type"] = association_type.value

                        # Ensure association_strength is present
                        if "association_strength" not in assoc_data:
                            # Try to use confidence_score or default
                            assoc_data["association_strength"] = assoc_data.get(
                                "confidence_score", 0.8
                            )

                        # Validate and create association model
                        association = ProductAssociation.parse_obj(assoc_data)
                        validated_associations.append(association)

                    except ValidationError as e:
                        logger.warning(
                            f"Invalid association {i} for product {product_id}: {e}"
                        )
                        continue

                # Create response model
                associations_response = ProductAssociationsResponse(
                    product_id=str(product_id),
                    associations=validated_associations,
                    total_associations=len(validated_associations),
                )

                parsed_associations[product_id] = associations_response
                logger.debug(
                    f"Parsed {len(validated_associations)} associations for product {product_id}"
                )

            except Exception as e:
                logger.error(
                    f"Error parsing associations for product {product_id}: {e}"
                )
                continue

        logger.info(
            f"Successfully parsed associations for {len(parsed_associations)} products"
        )
        return parsed_associations

    except Exception as e:
        logger.error(f"Failed to parse associations data: {e}")
        raise ParsingError(f"Failed to parse associations: {e}")


def parse_category_insights(
    raw_data: Dict[str, Any], category: str
) -> CategoryInsights:
    """
    Parse category insights data with validation.

    Args:
        raw_data: Raw insights data
        category: Category name

    Returns:
        Validated CategoryInsights instance

    Raises:
        ParsingError: If insights format is invalid
        ValidationError: If data validation fails
    """
    try:
        # Ensure category is set
        raw_data["category"] = category

        # Handle different possible structures
        if "top_products" in raw_data and isinstance(raw_data["top_products"], list):
            # Validate each product in top_products
            validated_products = []
            for product_data in raw_data["top_products"]:
                try:
                    product = QlooSearchResult.parse_obj(product_data)
                    validated_products.append(product)
                except ValidationError:
                    logger.warning(
                        f"Invalid product in top_products for category {category}"
                    )
                    continue
            raw_data["top_products"] = validated_products

        # Set defaults for missing fields
        if "total_products" not in raw_data:
            raw_data["total_products"] = raw_data.get("total_results", 0)

        insights = CategoryInsights.parse_obj(raw_data)
        logger.debug(f"Successfully parsed insights for category: {category}")
        return insights

    except ValidationError as e:
        logger.error(f"Validation error parsing category insights: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error parsing category insights: {e}")
        raise ParsingError(f"Failed to parse category insights: {e}")


def parse_catalog_products(raw_data: List[Dict[str, Any]]) -> List[Product]:
    """
    Parse and validate catalog product data.

    Args:
        raw_data: List of raw product dictionaries

    Returns:
        List of validated Product instances

    Raises:
        ParsingError: If catalog format is invalid
        ValidationError: If product data validation fails
    """
    try:
        if not isinstance(raw_data, list):
            raise ParsingError(f"Expected list, got {type(raw_data)}")

        validated_products = []
        errors = []

        for i, product_data in enumerate(raw_data):
            try:
                product = Product.parse_obj(product_data)
                validated_products.append(product)
            except ValidationError as e:
                error_msg = f"Invalid product at index {i}: {e}"
                errors.append(error_msg)
                logger.warning(error_msg)
                continue

        if errors and len(validated_products) == 0:
            raise ParsingError(f"No valid products found. Errors: {errors}")

        logger.info(
            f"Successfully parsed {len(validated_products)} products ({len(errors)} errors)"
        )
        return validated_products

    except Exception as e:
        logger.error(f"Failed to parse catalog products: {e}")
        raise ParsingError(f"Failed to parse catalog products: {e}")


def parse_association_rules(raw_data: List[Dict[str, Any]]) -> List[AssociationRule]:
    """
    Parse association rules mining results.

    Args:
        raw_data: List of raw association rule dictionaries

    Returns:
        List of validated AssociationRule instances

    Raises:
        ParsingError: If rules format is invalid
        ValidationError: If rule data validation fails
    """
    try:
        if not isinstance(raw_data, list):
            raise ParsingError(f"Expected list, got {type(raw_data)}")

        validated_rules = []
        errors = []

        for i, rule_data in enumerate(raw_data):
            try:
                rule = AssociationRule.parse_obj(rule_data)
                validated_rules.append(rule)
            except ValidationError as e:
                error_msg = f"Invalid rule at index {i}: {e}"
                errors.append(error_msg)
                logger.warning(error_msg)
                continue

        logger.info(
            f"Successfully parsed {len(validated_rules)} association rules ({len(errors)} errors)"
        )
        return validated_rules

    except Exception as e:
        logger.error(f"Failed to parse association rules: {e}")
        raise ParsingError(f"Failed to parse association rules: {e}")


def parse_api_health_response(raw_data: Dict[str, Any]) -> APIHealthResponse:
    """
    Parse API health check response.

    Args:
        raw_data: Raw health check response

    Returns:
        Validated APIHealthResponse instance

    Raises:
        ParsingError: If health response format is invalid
        ValidationError: If data validation fails
    """
    try:
        # Set default status if not present
        if "status" not in raw_data:
            raw_data["status"] = "ok"

        health_response = APIHealthResponse.parse_obj(raw_data)
        logger.debug("Successfully parsed API health response")
        return health_response

    except ValidationError as e:
        logger.error(f"Validation error parsing health response: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error parsing health response: {e}")
        raise ParsingError(f"Failed to parse health response: {e}")


def parse_error_response(
    raw_data: Union[Dict[str, Any], str], status_code: Optional[int] = None
) -> ErrorResponse:
    """
    Parse API error response.

    Args:
        raw_data: Raw error response (dict or string)
        status_code: HTTP status code if available

    Returns:
        Validated ErrorResponse instance

    Raises:
        ParsingError: If error response format is invalid
    """
    try:
        # Handle string errors
        if isinstance(raw_data, str):
            raw_data = {"error": {"message": raw_data}}

        # Ensure error field exists
        if "error" not in raw_data:
            raw_data = {"error": raw_data}

        # Add status code if provided
        if status_code:
            raw_data["status_code"] = status_code

        error_response = ErrorResponse.parse_obj(raw_data)
        logger.debug("Successfully parsed error response")
        return error_response

    except Exception as e:
        logger.error(f"Failed to parse error response: {e}")
        # Return a basic error response if parsing fails
        return ErrorResponse(
            error={"message": str(e), "original_data": str(raw_data)},
            status_code=status_code,
        )


def parse_catalog_statistics(raw_data: Dict[str, Any]) -> CatalogStatistics:
    """
    Parse catalog statistics data.

    Args:
        raw_data: Raw statistics data

    Returns:
        Validated CatalogStatistics instance

    Raises:
        ParsingError: If statistics format is invalid
        ValidationError: If data validation fails
    """
    try:
        # Ensure required fields have defaults
        if "total_products" not in raw_data:
            raw_data["total_products"] = 0

        if "categories" not in raw_data:
            raw_data["categories"] = {}

        statistics = CatalogStatistics.parse_obj(raw_data)
        logger.debug("Successfully parsed catalog statistics")
        return statistics

    except ValidationError as e:
        logger.error(f"Validation error parsing catalog statistics: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error parsing catalog statistics: {e}")
        raise ParsingError(f"Failed to parse catalog statistics: {e}")


def safe_parse_json(data: Union[str, bytes, Dict, List]) -> Union[Dict, List]:
    """
    Safely parse JSON data with error handling.

    Args:
        data: JSON string, bytes, or already parsed data

    Returns:
        Parsed JSON data

    Raises:
        ParsingError: If JSON parsing fails
    """
    try:
        if isinstance(data, (dict, list)):
            return data

        if isinstance(data, bytes):
            data = data.decode("utf-8")

        if isinstance(data, str):
            return json.loads(data)

        raise ParsingError(f"Cannot parse JSON from type: {type(data)}")

    except json.JSONDecodeError as e:
        raise ParsingError(f"Invalid JSON: {e}")
    except Exception as e:
        raise ParsingError(f"JSON parsing error: {e}")


def validate_response_structure(
    data: Any, expected_keys: List[str], response_type: str = "response"
) -> bool:
    """
    Validate that response data has expected structure.

    Args:
        data: Response data to validate
        expected_keys: List of required keys
        response_type: Type of response for error messages

    Returns:
        True if structure is valid

    Raises:
        ParsingError: If structure is invalid
    """
    if not isinstance(data, dict):
        raise ParsingError(f"{response_type} must be a dictionary, got {type(data)}")

    missing_keys = [key for key in expected_keys if key not in data]
    if missing_keys:
        raise ParsingError(f"{response_type} missing required keys: {missing_keys}")

    return True


def extract_and_validate_pagination(data: Dict[str, Any]) -> Tuple[int, int, bool]:
    """
    Extract and validate pagination information from response.

    Args:
        data: Response data containing pagination info

    Returns:
        Tuple of (current_page, total_pages, has_more)

    Raises:
        ParsingError: If pagination data is invalid
    """
    try:
        current_page = data.get("page", 1)
        total_pages = data.get("total_pages", 1)
        has_more = data.get("has_more", False)

        # Validate pagination values
        if not isinstance(current_page, int) or current_page < 1:
            raise ParsingError(f"Invalid current_page: {current_page}")

        if not isinstance(total_pages, int) or total_pages < 1:
            raise ParsingError(f"Invalid total_pages: {total_pages}")

        if current_page > total_pages:
            raise ParsingError(
                f"Current page ({current_page}) cannot exceed total pages ({total_pages})"
            )

        return current_page, total_pages, bool(has_more)

    except Exception as e:
        logger.error(f"Error extracting pagination: {e}")
        return 1, 1, False


# Utility functions for common parsing tasks


def normalize_product_id(product_id: Union[str, int]) -> str:
    """
    Normalize product ID to string format.

    Args:
        product_id: Product ID in various formats

    Returns:
        Normalized product ID as string
    """
    if isinstance(product_id, (int, float)):
        return str(int(product_id))
    return str(product_id).strip()


def normalize_category_name(category: Optional[str]) -> str:
    """
    Normalize category name to standard format.

    Args:
        category: Raw category name

    Returns:
        Normalized category name
    """
    if not category or not isinstance(category, str):
        return "Uncategorized"

    return category.strip().title()


def calculate_confidence_score(support: float, lift: float) -> float:
    """
    Calculate confidence score from support and lift metrics.

    Args:
        support: Support metric (0-1)
        lift: Lift metric (>0)

    Returns:
        Calculated confidence score (0-1)
    """
    try:
        # Simple confidence estimation: support * min(lift, 2) / 2
        confidence = support * min(lift, 2.0) / 2.0
        return max(0.0, min(1.0, confidence))  # Clamp to [0,1]
    except (TypeError, ValueError):
        return 0.0
