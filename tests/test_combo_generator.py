"""
Unit tests for Combo and ComboGenerator classes.
"""

import pytest
from datetime import datetime
from typing import List
from unittest.mock import Mock

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models import Combo, ComboGenerator, Product


class TestCombo:
    """Test cases for Combo dataclass."""
    
    def test_valid_combo_creation(self):
        """Test creating a valid combo."""
        combo = Combo(
            combo_id="test_combo_1",
            name="Test Combo",
            products=[1, 2, 3],
            confidence_score=0.85,
            support=0.05,
            lift=1.5,
            expected_discount_percent=15.0
        )
        
        assert combo.combo_id == "test_combo_1"
        assert combo.name == "Test Combo"
        assert combo.products == [1, 2, 3]
        assert combo.confidence_score == 0.85
        assert combo.is_active is True
        
    def test_invalid_confidence_score(self):
        """Test combo creation with invalid confidence score."""
        with pytest.raises(ValueError, match="Confidence score must be between 0.0 and 1.0"):
            Combo(
                combo_id="test_combo_1",
                name="Test Combo",
                products=[1, 2],
                confidence_score=1.5,  # Invalid
                support=0.05,
                lift=1.5
            )
    
    def test_invalid_support(self):
        """Test combo creation with invalid support."""
        with pytest.raises(ValueError, match="Support must be between 0.0 and 1.0"):
            Combo(
                combo_id="test_combo_1",
                name="Test Combo",
                products=[1, 2],
                confidence_score=0.8,
                support=-0.1,  # Invalid
                lift=1.5
            )
    
    def test_invalid_lift(self):
        """Test combo creation with invalid lift."""
        with pytest.raises(ValueError, match="Lift must be non-negative"):
            Combo(
                combo_id="test_combo_1",
                name="Test Combo",
                products=[1, 2],
                confidence_score=0.8,
                support=0.05,
                lift=-1.0  # Invalid
            )
    
    def test_insufficient_products(self):
        """Test combo creation with insufficient products."""
        with pytest.raises(ValueError, match="Combo must contain at least 2 products"):
            Combo(
                combo_id="test_combo_1",
                name="Test Combo",
                products=[1],  # Insufficient
                confidence_score=0.8,
                support=0.05,
                lift=1.5
            )
    
    def test_invalid_discount_percent(self):
        """Test combo creation with invalid discount percentage."""
        with pytest.raises(ValueError, match="Discount percent must be between 0.0 and 100.0"):
            Combo(
                combo_id="test_combo_1",
                name="Test Combo",
                products=[1, 2],
                confidence_score=0.8,
                support=0.05,
                lift=1.5,
                expected_discount_percent=150.0  # Invalid
            )


class TestComboGenerator:
    """Test cases for ComboGenerator class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.generator = ComboGenerator(min_confidence=0.8, min_support=0.01)
        
        # Mock products
        self.products = [
            Mock(id=1, category="Dairy"),
            Mock(id=2, category="Produce"),
            Mock(id=3, category="Meat"),
            Mock(id=4, category="Beverages")
        ]
        
        # Mock association rules
        self.association_rules = [
            {
                'antecedent': [1, 2],
                'consequent': [3],
                'confidence': 0.85,
                'support': 0.05,
                'lift': 1.5
            },
            {
                'antecedent': [2],
                'consequent': [4],
                'confidence': 0.75,  # Below threshold
                'support': 0.03,
                'lift': 1.2
            },
            {
                'antecedent': [1],
                'consequent': [4],
                'confidence': 0.9,
                'support': 0.02,
                'lift': 2.0
            }
        ]
    
    def test_generate_weekly_combos_filters_by_confidence(self):
        """Test that only high-confidence rules are included."""
        combos = self.generator.generate_weekly_combos(self.association_rules, self.products)
        
        # Should only include rules with confidence >= 0.8
        assert len(combos) == 2
        
        # Check first combo
        combo1 = combos[0]
        assert combo1.confidence_score == 0.85
        assert combo1.products == [1, 2, 3]
        assert set(combo1.category_mix) == {"Dairy", "Produce", "Meat"}
        
        # Check second combo
        combo2 = combos[1]
        assert combo2.confidence_score == 0.9
        assert combo2.products == [1, 4]
        assert set(combo2.category_mix) == {"Dairy", "Beverages"}
    
    def test_discount_calculation(self):
        """Test discount percentage calculation."""
        # High confidence, high lift should result in lower discount
        high_discount = self.generator._calculate_discount_suggestion(0.95, 2.5)
        
        # Low confidence, low lift should result in higher discount
        low_discount = self.generator._calculate_discount_suggestion(0.80, 1.0)
        
        assert high_discount < low_discount
        assert 5.0 <= high_discount <= 25.0
        assert 5.0 <= low_discount <= 25.0
    
    def test_empty_rules_returns_empty_combos(self):
        """Test that empty association rules return empty combos."""
        combos = self.generator.generate_weekly_combos([], self.products)
        assert len(combos) == 0
    
    def test_combo_id_generation(self):
        """Test that combo IDs are generated correctly."""
        combos = self.generator.generate_weekly_combos(self.association_rules, self.products)
        
        assert combos[0].combo_id == "combo_0_85"
        assert combos[1].combo_id == "combo_2_90"
    
    def test_constructor_parameters(self):
        """Test ComboGenerator initialization with custom parameters."""
        custom_generator = ComboGenerator(min_confidence=0.7, min_support=0.02)
        
        assert custom_generator.min_confidence == 0.7
        assert custom_generator.min_support == 0.02 