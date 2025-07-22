"""
Unit tests for Price API functionality.
"""

import pytest
from unittest.mock import Mock, patch

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from price_api import PriceAPIStub, DiscountStrategy, PricePoint, DiscountSuggestion


class TestPriceAPIStub:
    """Test cases for PriceAPIStub class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.price_api = PriceAPIStub(
            default_strategy=DiscountStrategy.BALANCED,
            fallback_discount_min=5.0,
            fallback_discount_max=25.0
        )
    
    def test_initialization(self):
        """Test PriceAPIStub initialization."""
        assert self.price_api.default_strategy == DiscountStrategy.BALANCED
        assert self.price_api.fallback_discount_min == 5.0
        assert self.price_api.fallback_discount_max == 25.0
        assert len(self.price_api._mock_pricing_data) > 0
    
    def test_suggest_discount_for_combo_basic(self):
        """Test basic discount suggestion functionality."""
        suggestion = self.price_api.suggest_discount_for_combo(
            combo_id="test_combo_1",
            product_ids=[1, 2, 3],
            confidence_score=0.85,
            lift=1.5
        )
        
        assert isinstance(suggestion, DiscountSuggestion)
        assert suggestion.combo_id == "test_combo_1"
        assert 5.0 <= suggestion.suggested_discount_percent <= 25.0
        assert 0.0 <= suggestion.confidence_score <= 1.0
        assert suggestion.strategy_used == DiscountStrategy.BALANCED
        assert suggestion.min_discount <= suggestion.suggested_discount_percent <= suggestion.max_discount
    
    def test_suggest_discount_conservative_strategy(self):
        """Test conservative discount strategy."""
        conservative_api = PriceAPIStub(default_strategy=DiscountStrategy.CONSERVATIVE)
        
        suggestion = conservative_api.suggest_discount_for_combo(
            combo_id="conservative_combo",
            product_ids=[1, 2],
            confidence_score=0.9,
            lift=2.0
        )
        
        assert suggestion.strategy_used == DiscountStrategy.CONSERVATIVE
        # Conservative strategy should generally provide lower discounts
        assert suggestion.suggested_discount_percent >= 5.0
    
    def test_suggest_discount_aggressive_strategy(self):
        """Test aggressive discount strategy."""
        aggressive_api = PriceAPIStub(default_strategy=DiscountStrategy.AGGRESSIVE)
        
        suggestion = aggressive_api.suggest_discount_for_combo(
            combo_id="aggressive_combo",
            product_ids=[1, 2],
            confidence_score=0.8,
            lift=1.2
        )
        
        assert suggestion.strategy_used == DiscountStrategy.AGGRESSIVE
        # Aggressive strategy should generally provide higher discounts
        assert suggestion.suggested_discount_percent <= 25.0
    
    def test_discount_constraints(self):
        """Test that discount suggestions respect min/max constraints."""
        # Test with very high confidence and lift (should give minimum discount)
        high_confidence_suggestion = self.price_api.suggest_discount_for_combo(
            combo_id="high_conf_combo",
            product_ids=[1, 2],
            confidence_score=0.99,
            lift=3.0
        )
        
        assert high_confidence_suggestion.suggested_discount_percent >= 5.0
        
        # Test with very low confidence and lift (should give higher discount but capped)
        low_confidence_suggestion = self.price_api.suggest_discount_for_combo(
            combo_id="low_conf_combo",
            product_ids=[1, 2],
            confidence_score=0.8,
            lift=0.8
        )
        
        assert low_confidence_suggestion.suggested_discount_percent <= 25.0
    
    def test_fallback_on_error(self):
        """Test fallback behavior when calculation fails."""
        # Mock an error in the calculation
        with patch.object(self.price_api, '_calculate_base_discount', side_effect=Exception("Test error")):
            suggestion = self.price_api.suggest_discount_for_combo(
                combo_id="error_combo",
                product_ids=[1, 2],
                confidence_score=0.85,
                lift=1.5
            )
            
            # Should get fallback values
            assert suggestion.confidence_score == 0.5
            assert "Fallback calculation" in suggestion.reason
            assert 5.0 <= suggestion.suggested_discount_percent <= 25.0
    
    def test_pricing_summary(self):
        """Test pricing summary functionality."""
        summary = self.price_api.get_pricing_summary()
        
        assert "total_products" in summary
        assert "avg_price" in summary
        assert "avg_margin" in summary
        assert "price_range" in summary
        assert "margin_range" in summary
        assert "last_updated" in summary
        
        assert summary["total_products"] > 0
        assert summary["avg_price"] > 0
        assert summary["avg_margin"] > 0
    
    def test_discount_reason_generation(self):
        """Test discount reason text generation."""
        suggestion = self.price_api.suggest_discount_for_combo(
            combo_id="reason_test_combo",
            product_ids=[1, 2],
            confidence_score=0.95,
            lift=2.5
        )
        
        assert "Very high association confidence" in suggestion.reason or "high" in suggestion.reason.lower()
        assert "strong" in suggestion.reason.lower() or "good" in suggestion.reason.lower()
        assert len(suggestion.reason) > 10  # Should be descriptive
    
    def test_impact_estimation(self):
        """Test impact estimation functionality."""
        suggestion = self.price_api.suggest_discount_for_combo(
            combo_id="impact_test_combo",
            product_ids=[1, 2, 3, 4],  # More products
            confidence_score=0.85,
            lift=1.5
        )
        
        assert suggestion.estimated_impact is not None
        assert "impact" in suggestion.estimated_impact.lower()
        assert "volume" in suggestion.estimated_impact.lower()
    
    def test_confidence_calculation(self):
        """Test suggestion confidence calculation."""
        # High confidence rule should have high suggestion confidence
        high_conf_suggestion = self.price_api.suggest_discount_for_combo(
            combo_id="high_conf_test",
            product_ids=[1, 2],
            confidence_score=0.95,
            lift=2.0
        )
        
        # Low confidence rule should have lower suggestion confidence
        low_conf_suggestion = self.price_api.suggest_discount_for_combo(
            combo_id="low_conf_test",
            product_ids=[1, 2],
            confidence_score=0.8,
            lift=1.1
        )
        
        assert high_conf_suggestion.confidence_score >= low_conf_suggestion.confidence_score
    
    def test_price_point_generation(self):
        """Test price point data generation for unknown products."""
        # Test with product IDs not in mock data
        suggestion = self.price_api.suggest_discount_for_combo(
            combo_id="unknown_products_combo",
            product_ids=[999, 1000, 1001],  # Unknown product IDs
            confidence_score=0.85,
            lift=1.5
        )
        
        # Should still work and provide reasonable discount
        assert 5.0 <= suggestion.suggested_discount_percent <= 25.0
        assert suggestion.confidence_score > 0
    
    def test_strategy_override(self):
        """Test that strategy can be overridden per request."""
        # Test overriding default strategy
        suggestion = self.price_api.suggest_discount_for_combo(
            combo_id="strategy_override_combo",
            product_ids=[1, 2],
            confidence_score=0.85,
            lift=1.5,
            strategy=DiscountStrategy.CONSERVATIVE
        )
        
        assert suggestion.strategy_used == DiscountStrategy.CONSERVATIVE
        
        # Should be different from default strategy result
        default_suggestion = self.price_api.suggest_discount_for_combo(
            combo_id="strategy_default_combo",
            product_ids=[1, 2],
            confidence_score=0.85,
            lift=1.5
        )
        
        assert default_suggestion.strategy_used == DiscountStrategy.BALANCED 