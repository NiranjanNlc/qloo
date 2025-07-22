"""
Price API Service for Discount Suggestions

This module provides a service for suggesting discount percentages
based on product combinations, market conditions, and configurable fallbacks.
"""

import json
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import random
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class DiscountStrategy(str, Enum):
    """Discount calculation strategies."""

    CONSERVATIVE = "conservative"
    AGGRESSIVE = "aggressive"
    BALANCED = "balanced"
    MARKET_BASED = "market_based"


@dataclass
class PricePoint:
    """Represents a product's pricing information."""

    product_id: int
    current_price: float
    cost: float
    margin_percent: float
    competitor_avg_price: Optional[float] = None
    historical_discount_avg: Optional[float] = None


@dataclass
class DiscountSuggestion:
    """Represents a discount percentage suggestion with metadata."""

    combo_id: str
    suggested_discount_percent: float
    strategy_used: DiscountStrategy
    confidence_score: float
    min_discount: float
    max_discount: float
    reason: str
    estimated_impact: Optional[str] = None


class PriceAPIStub:
    """
    Price API service stub that suggests discount percentages with configurable fallbacks.

    In production, this would integrate with external pricing services,
    but for now provides intelligent discount suggestions based on business rules.
    """

    def __init__(
        self,
        default_strategy: DiscountStrategy = DiscountStrategy.BALANCED,
        fallback_discount_min: float = 5.0,
        fallback_discount_max: float = 25.0,
    ):
        """
        Initialize the Price API service.

        Args:
            default_strategy: Default discount calculation strategy
            fallback_discount_min: Minimum fallback discount percentage
            fallback_discount_max: Maximum fallback discount percentage
        """
        self.default_strategy = default_strategy
        self.fallback_discount_min = fallback_discount_min
        self.fallback_discount_max = fallback_discount_max

        # Mock pricing data for simulation
        self._mock_pricing_data = self._generate_mock_pricing_data()

    def suggest_discount_for_combo(
        self,
        combo_id: str,
        product_ids: List[int],
        confidence_score: float,
        lift: float,
        strategy: Optional[DiscountStrategy] = None,
    ) -> DiscountSuggestion:
        """
        Suggest discount percentage for a product combination.

        Args:
            combo_id: Unique identifier for the combo
            product_ids: List of product IDs in the combo
            confidence_score: Association rule confidence (0.0-1.0)
            lift: Association rule lift value
            strategy: Discount strategy to use (defaults to instance default)

        Returns:
            DiscountSuggestion with recommended discount and metadata
        """
        try:
            strategy = strategy or self.default_strategy

            # Get pricing data for products
            price_points = self._get_price_points(product_ids)

            # Calculate base discount using strategy
            base_discount = self._calculate_base_discount(
                price_points, confidence_score, lift, strategy
            )

            # Apply business constraints
            final_discount = self._apply_constraints(base_discount, price_points)

            # Calculate confidence in the suggestion
            suggestion_confidence = self._calculate_suggestion_confidence(
                price_points, confidence_score, lift
            )

            return DiscountSuggestion(
                combo_id=combo_id,
                suggested_discount_percent=final_discount,
                strategy_used=strategy,
                confidence_score=suggestion_confidence,
                min_discount=max(self.fallback_discount_min, final_discount - 5.0),
                max_discount=min(self.fallback_discount_max, final_discount + 5.0),
                reason=self._generate_discount_reason(strategy, confidence_score, lift),
                estimated_impact=self._estimate_impact(
                    final_discount, len(product_ids)
                ),
            )

        except Exception as e:
            logger.warning(f"Error calculating discount for combo {combo_id}: {e}")
            return self._fallback_discount_suggestion(combo_id, strategy)

    def _generate_mock_pricing_data(self) -> Dict[int, PricePoint]:
        """Generate mock pricing data for simulation."""
        mock_data = {}

        # Generate pricing for 50 mock products
        for product_id in range(1, 51):
            # Random price between $2-$50
            price = round(random.uniform(2.0, 50.0), 2)
            # Cost is 60-80% of price
            cost = round(price * random.uniform(0.6, 0.8), 2)
            # Margin calculation
            margin = round(((price - cost) / price) * 100, 1)

            mock_data[product_id] = PricePoint(
                product_id=product_id,
                current_price=price,
                cost=cost,
                margin_percent=margin,
                competitor_avg_price=round(price * random.uniform(0.9, 1.1), 2),
                historical_discount_avg=round(random.uniform(5.0, 15.0), 1),
            )

        return mock_data

    def _get_price_points(self, product_ids: List[int]) -> List[PricePoint]:
        """Get pricing information for given product IDs."""
        price_points = []

        for product_id in product_ids:
            if product_id in self._mock_pricing_data:
                price_points.append(self._mock_pricing_data[product_id])
            else:
                # Generate on-demand for missing products
                price = round(random.uniform(5.0, 30.0), 2)
                cost = round(price * 0.7, 2)
                margin = round(((price - cost) / price) * 100, 1)

                price_point = PricePoint(
                    product_id=product_id,
                    current_price=price,
                    cost=cost,
                    margin_percent=margin,
                    historical_discount_avg=10.0,
                )
                price_points.append(price_point)

        return price_points

    def _calculate_base_discount(
        self,
        price_points: List[PricePoint],
        confidence_score: float,
        lift: float,
        strategy: DiscountStrategy,
    ) -> float:
        """Calculate base discount percentage using the specified strategy."""
        avg_margin = sum(pp.margin_percent for pp in price_points) / len(price_points)
        avg_historical_discount = sum(
            pp.historical_discount_avg or 10.0 for pp in price_points
        ) / len(price_points)

        if strategy == DiscountStrategy.CONSERVATIVE:
            # Lower discounts, prioritize margin protection
            base = min(avg_historical_discount * 0.8, avg_margin * 0.3)
            # Adjust based on confidence (higher confidence = lower discount needed)
            confidence_adjustment = (1.0 - confidence_score) * 3.0

        elif strategy == DiscountStrategy.AGGRESSIVE:
            # Higher discounts to drive volume
            base = min(avg_historical_discount * 1.5, avg_margin * 0.6)
            # Less sensitive to confidence
            confidence_adjustment = (1.0 - confidence_score) * 2.0

        elif strategy == DiscountStrategy.MARKET_BASED:
            # Use competitor pricing and historical data
            base = avg_historical_discount
            # Adjust based on market position
            confidence_adjustment = (1.0 - confidence_score) * 2.5

        else:  # BALANCED
            # Balance between margin protection and volume
            base = avg_historical_discount * 1.1
            confidence_adjustment = (1.0 - confidence_score) * 2.5

        # Apply lift factor (higher lift = lower discount needed)
        lift_adjustment = max(0, (2.0 - lift) * 2.0)

        return round(base + confidence_adjustment + lift_adjustment, 1)

    def _apply_constraints(
        self, base_discount: float, price_points: List[PricePoint]
    ) -> float:
        """Apply business constraints to the calculated discount."""
        # Ensure minimum profitability
        min_margin_threshold = 5.0  # Minimum 5% margin required
        avg_margin = sum(pp.margin_percent for pp in price_points) / len(price_points)

        max_allowable_discount = max(0, avg_margin - min_margin_threshold)

        # Apply global constraints
        constrained_discount = max(
            self.fallback_discount_min,
            min(base_discount, max_allowable_discount, self.fallback_discount_max),
        )

        return round(constrained_discount, 1)

    def _calculate_suggestion_confidence(
        self, price_points: List[PricePoint], confidence_score: float, lift: float
    ) -> float:
        """Calculate confidence in the discount suggestion."""
        # Base confidence from association rule
        base_confidence = confidence_score

        # Adjust based on data availability
        data_quality = len(price_points) / max(len(price_points), 1)

        # Adjust based on lift (higher lift = more confident)
        lift_factor = min(1.0, lift / 2.0)

        # Adjust based on margin consistency
        margins = [pp.margin_percent for pp in price_points]
        margin_variance = max(margins) - min(margins) if margins else 0
        margin_factor = max(0.5, 1.0 - (margin_variance / 100.0))

        final_confidence = base_confidence * data_quality * lift_factor * margin_factor
        return round(min(1.0, max(0.0, final_confidence)), 3)

    def _generate_discount_reason(
        self, strategy: DiscountStrategy, confidence_score: float, lift: float
    ) -> str:
        """Generate human-readable reason for the discount suggestion."""
        reasons = []

        if confidence_score >= 0.9:
            reasons.append("Very high association confidence")
        elif confidence_score >= 0.8:
            reasons.append("High association confidence")
        else:
            reasons.append("Moderate association confidence")

        if lift >= 2.0:
            reasons.append("strong lift indicator")
        elif lift >= 1.5:
            reasons.append("good lift indicator")
        else:
            reasons.append("moderate lift indicator")

        strategy_reason = {
            DiscountStrategy.CONSERVATIVE: "conservative pricing strategy",
            DiscountStrategy.AGGRESSIVE: "aggressive volume strategy",
            DiscountStrategy.BALANCED: "balanced approach",
            DiscountStrategy.MARKET_BASED: "market-driven pricing",
        }

        reasons.append(strategy_reason[strategy])

        return f"Based on {', '.join(reasons)}"

    def _estimate_impact(self, discount_percent: float, num_products: int) -> str:
        """Estimate the potential impact of the discount."""
        if discount_percent <= 10:
            impact = "Low"
        elif discount_percent <= 20:
            impact = "Moderate"
        else:
            impact = "High"

        volume_impact = "moderate" if num_products <= 2 else "significant"

        return f"{impact} margin impact, {volume_impact} volume potential"

    def _fallback_discount_suggestion(
        self, combo_id: str, strategy: DiscountStrategy
    ) -> DiscountSuggestion:
        """Provide fallback discount suggestion when calculation fails."""
        fallback_discount = (
            self.fallback_discount_min + self.fallback_discount_max
        ) / 2

        return DiscountSuggestion(
            combo_id=combo_id,
            suggested_discount_percent=fallback_discount,
            strategy_used=strategy,
            confidence_score=0.5,
            min_discount=self.fallback_discount_min,
            max_discount=self.fallback_discount_max,
            reason="Fallback calculation due to insufficient data",
            estimated_impact="Moderate margin impact, moderate volume potential",
        )

    def get_pricing_summary(self) -> Dict:
        """Get summary of current pricing data."""
        if not self._mock_pricing_data:
            return {"error": "No pricing data available"}

        prices = [pp.current_price for pp in self._mock_pricing_data.values()]
        margins = [pp.margin_percent for pp in self._mock_pricing_data.values()]

        return {
            "total_products": len(self._mock_pricing_data),
            "avg_price": round(sum(prices) / len(prices), 2),
            "avg_margin": round(sum(margins) / len(margins), 1),
            "price_range": {"min": min(prices), "max": max(prices)},
            "margin_range": {"min": min(margins), "max": max(margins)},
            "last_updated": datetime.now().isoformat(),
        }
