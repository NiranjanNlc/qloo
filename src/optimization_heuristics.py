"""
Optimization Heuristics Module

This module implements advanced heuristics for layout optimization including:
- Crowding penalty calculations
- Adjacency reward systems
- Popularity-based placement bonuses
- Cross-selling optimization
- Configuration-driven parameter management
"""

import os
import logging
from typing import Dict, List, Optional, Tuple, Any, cast
import numpy as np
import pandas as pd
import yaml
from datetime import datetime, timedelta
from dataclasses import dataclass
from pathlib import Path

from models import Product
from layout_optimizer import ProductLocation, StoreSection, SupermarketLayoutOptimizer

logger = logging.getLogger(__name__)


@dataclass
class HeuristicScore:
    """Represents a heuristic scoring result."""

    product_id: int
    base_score: float
    crowding_penalty: float
    adjacency_reward: float
    popularity_bonus: float
    cross_selling_bonus: float
    final_score: float
    components: Dict[str, float]


@dataclass
class PlacementSuggestion:
    """Represents a product placement suggestion with heuristic reasoning."""

    product_id: int
    current_section: Optional[str]
    suggested_section: str
    confidence_score: float
    expected_improvement: float
    heuristic_breakdown: Dict[str, float]
    reasoning: str


class ConfigurationManager:
    """Manages configuration loading and parameter access."""

    def __init__(self, config_path: str = "config.yml"):
        """
        Initialize configuration manager.

        Args:
            config_path: Path to YAML configuration file
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, "r") as f:
                return cast(Dict[str, Any], yaml.safe_load(f))
        except FileNotFoundError:
            logger.warning(f"Config file {self.config_path} not found. Using defaults.")
            return self._get_default_config()
        except yaml.YAMLError as e:
            logger.error(f"Error parsing config file: {e}")
            return self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """Return default configuration if file is not available."""
        return {
            "optimization": {
                "heuristics": {
                    "crowding_penalty": {
                        "enabled": True,
                        "base_penalty": 0.15,
                        "utilization_threshold": 0.9,
                        "max_penalty": 0.5,
                        "exponential_scaling": 2.0,
                    },
                    "adjacency_reward": {
                        "enabled": True,
                        "base_reward": 0.1,
                        "association_threshold": 1.5,
                        "max_reward": 0.3,
                        "distance_decay": 0.8,
                        "category_bonus": 0.05,
                    },
                    "popularity_bonus": {
                        "enabled": True,
                        "high_traffic_multiplier": 1.2,
                        "popularity_threshold": 0.3,
                        "endcap_bonus": 1.5,
                    },
                    "cross_selling": {
                        "enabled": True,
                        "min_confidence": 0.6,
                        "complementary_bonus": 0.2,
                        "basket_affinity_weight": 1.0,
                    },
                }
            },
            "performance": {
                "thresholds": {
                    "min_improvement_threshold": 0.05,
                    "confidence_threshold": 0.7,
                }
            },
        }

    def get_heuristic_config(self, heuristic_name: str) -> Dict[str, Any]:
        """Get configuration for a specific heuristic."""
        return cast(Dict[str, Any], (
            self.config.get("optimization", {})
            .get("heuristics", {})
            .get(heuristic_name, {})
        ))

    def get_performance_config(self) -> Dict[str, Any]:
        """Get performance configuration."""
        return cast(Dict[str, Any], self.config.get("performance", {}))

    def reload_config(self) -> None:
        """Reload configuration from file."""
        self.config = self._load_config()


class HeuristicOptimizer:
    """Main class for applying heuristic optimizations to store layouts."""

    def __init__(
        self,
        layout_optimizer: SupermarketLayoutOptimizer,
        config_manager: Optional[ConfigurationManager] = None,
    ):
        """
        Initialize heuristic optimizer.

        Args:
            layout_optimizer: The main layout optimizer instance
            config_manager: Optional configuration manager (creates default if None)
        """
        self.layout_optimizer = layout_optimizer
        self.config_manager = config_manager or ConfigurationManager()

        # Cache for performance
        self._section_utilization_cache: dict[str, float] = {}
        self._product_popularity_cache: dict[int, float] = {}
        self._association_cache: dict[tuple, float] = {}

    def calculate_heuristic_scores(
        self, target_products: Optional[List[int]] = None
    ) -> List[HeuristicScore]:
        """
        Calculate comprehensive heuristic scores for products.

        Args:
            target_products: List of product IDs to score (None for all)

        Returns:
            List of HeuristicScore objects
        """
        if target_products is None:
            target_products = list(self.layout_optimizer.current_layout.keys())

        scores = []

        for product_id in target_products:
            current_location = self.layout_optimizer.current_layout.get(product_id)
            if not current_location:
                continue

            # Calculate individual heuristic components
            base_score = self._calculate_base_score(product_id, current_location)
            crowding_penalty = self._calculate_crowding_penalty(
                product_id, current_location
            )
            adjacency_reward = self._calculate_adjacency_reward(
                product_id, current_location
            )
            popularity_bonus = self._calculate_popularity_bonus(
                product_id, current_location
            )
            cross_selling_bonus = self._calculate_cross_selling_bonus(
                product_id, current_location
            )

            # Combine components into final score
            final_score = (
                base_score
                - crowding_penalty
                + adjacency_reward
                + popularity_bonus
                + cross_selling_bonus
            )

            # Create detailed component breakdown
            components = {
                "base_score": base_score,
                "crowding_penalty": -crowding_penalty,
                "adjacency_reward": adjacency_reward,
                "popularity_bonus": popularity_bonus,
                "cross_selling_bonus": cross_selling_bonus,
            }

            score = HeuristicScore(
                product_id=product_id,
                base_score=base_score,
                crowding_penalty=crowding_penalty,
                adjacency_reward=adjacency_reward,
                popularity_bonus=popularity_bonus,
                cross_selling_bonus=cross_selling_bonus,
                final_score=max(0.0, final_score),  # Ensure non-negative
                components=components,
            )
            scores.append(score)

        return sorted(scores, key=lambda x: x.final_score, reverse=True)

    def _calculate_base_score(
        self, product_id: int, location: ProductLocation
    ) -> float:
        """Calculate base score for product placement."""
        # Base score considering zone traffic and category fit
        zone = self.layout_optimizer.store_zones.get(location.zone)
        if not zone:
            return 0.5

        # Category-zone fit score
        category_fit = self._get_category_zone_fit(location.category, location.zone)

        # Traffic score
        traffic_score = zone.foot_traffic_score

        # Shelf priority score (eye level is best)
        shelf_score = {1: 1.0, 2: 0.8, 3: 0.6}.get(location.shelf_priority, 0.5)

        return category_fit * 0.4 + traffic_score * 0.4 + shelf_score * 0.2

    def _calculate_crowding_penalty(
        self, product_id: int, location: ProductLocation
    ) -> float:
        """Calculate penalty for overcrowded sections."""
        config = self.config_manager.get_heuristic_config("crowding_penalty")
        if not config.get("enabled", True):
            return 0.0

        # Get section utilization
        section_id = self._get_section_for_location(location)
        utilization = self._get_section_utilization(section_id)

        # Calculate penalty if over threshold
        threshold = config.get("utilization_threshold", 0.9)
        if utilization <= threshold:
            return 0.0

        # Calculate exponential penalty for overcrowding
        base_penalty = config.get("base_penalty", 0.15)
        max_penalty = config.get("max_penalty", 0.5)
        scaling = config.get("exponential_scaling", 2.0)

        # Exponential scaling beyond threshold
        excess_utilization = utilization - threshold
        penalty_factor = (excess_utilization / (1.0 - threshold)) ** scaling

        penalty = min(base_penalty * penalty_factor, max_penalty)

        logger.debug(
            f"Crowding penalty for product {product_id}: {penalty:.3f} "
            f"(utilization: {utilization:.2f})"
        )

        return float(penalty)

    def _calculate_adjacency_reward(
        self, product_id: int, location: ProductLocation
    ) -> float:
        """Calculate reward for beneficial product adjacencies."""
        config = self.config_manager.get_heuristic_config("adjacency_reward")
        if not config.get("enabled", True):
            return 0.0

        if not self.layout_optimizer.association_graph:
            return 0.0

        reward = 0.0
        base_reward = config.get("base_reward", 0.1)
        max_reward = config.get("max_reward", 0.3)
        distance_decay = config.get("distance_decay", 0.8)
        category_bonus = config.get("category_bonus", 0.05)
        association_threshold = config.get("association_threshold", 1.5)

        # Check associations with nearby products
        neighbors = list(self.layout_optimizer.association_graph.neighbors(product_id))

        for neighbor_id in neighbors:
            neighbor_location = self.layout_optimizer.current_layout.get(neighbor_id)
            if not neighbor_location:
                continue

            # Get association strength
            edge_data = self.layout_optimizer.association_graph[product_id][neighbor_id]
            lift = edge_data.get("weight", 0.0)

            if lift < association_threshold:
                continue

            # Calculate distance between products
            distance = self._calculate_distance(location, neighbor_location)

            # Distance-based reward decay
            distance_factor = distance_decay**distance

            # Association strength factor
            association_factor = min(lift / 3.0, 1.0)  # Normalize to max 1.0

            # Category bonus for same-category products
            category_factor = 1.0
            if location.category == neighbor_location.category:
                category_factor += category_bonus

            # Calculate reward for this association
            association_reward = (
                base_reward * association_factor * distance_factor * category_factor
            )
            reward += association_reward

        # Cap total reward
        reward = min(reward, max_reward)

        logger.debug(f"Adjacency reward for product {product_id}: {reward:.3f}")
        return float(reward)

    def _calculate_popularity_bonus(
        self, product_id: int, location: ProductLocation
    ) -> float:
        """Calculate bonus for placing popular products in high-traffic areas."""
        config = self.config_manager.get_heuristic_config("popularity_bonus")
        if not config.get("enabled", True):
            return 0.0

        # Get product popularity
        popularity = self._get_product_popularity(product_id)
        popularity_threshold = config.get("popularity_threshold", 0.3)

        if popularity < popularity_threshold:
            return 0.0

        # Get zone traffic score
        zone = self.layout_optimizer.store_zones.get(location.zone)
        if not zone:
            return 0.0

        traffic_score = zone.foot_traffic_score
        high_traffic_multiplier = config.get("high_traffic_multiplier", 1.2)

        # Base bonus for popular products in high-traffic areas
        bonus = popularity * traffic_score * 0.1

        # Extra bonus for high-traffic zones
        if traffic_score > 0.8:
            bonus *= high_traffic_multiplier

        # Extra bonus for endcap placement
        section_id = self._get_section_for_location(location)
        if section_id and "endcap" in section_id:
            endcap_bonus = config.get("endcap_bonus", 1.5)
            bonus *= endcap_bonus

        logger.debug(
            f"Popularity bonus for product {product_id}: {bonus:.3f} "
            f"(popularity: {popularity:.2f})"
        )
        return float(bonus)

    def _calculate_cross_selling_bonus(
        self, product_id: int, location: ProductLocation
    ) -> float:
        """Calculate bonus for cross-selling opportunities."""
        config = self.config_manager.get_heuristic_config("cross_selling")
        if not config.get("enabled", True):
            return 0.0

        if not self.layout_optimizer.association_engine:
            return 0.0

        bonus = 0.0
        min_confidence = config.get("min_confidence", 0.6)
        complementary_bonus = config.get("complementary_bonus", 0.2)
        basket_affinity_weight = config.get("basket_affinity_weight", 1.0)

        # Find high-confidence associations
        for rule in self.layout_optimizer.association_engine.association_rules:
            if rule["confidence"] < min_confidence:
                continue

            # Check if current product is in the rule
            antecedent = list(rule["antecedent"])
            consequent = list(rule["consequent"])

            if product_id not in (antecedent + consequent):
                continue

            # Find complementary products in the same section
            section_id = self._get_section_for_location(location)
            section_products = self._get_products_in_section(section_id)

            # Calculate cross-selling potential
            for other_product in antecedent + consequent:
                if other_product != product_id and other_product in section_products:
                    # Bonus for having complementary products nearby
                    rule_bonus = (
                        rule["confidence"]
                        * rule["lift"]
                        * complementary_bonus
                        * basket_affinity_weight
                    )
                    bonus += rule_bonus

        logger.debug(f"Cross-selling bonus for product {product_id}: {bonus:.3f}")
        return float(bonus)

    def generate_placement_suggestions(
        self, min_improvement: Optional[float] = None
    ) -> List[PlacementSuggestion]:
        """
        Generate placement suggestions based on heuristic analysis.

        Args:
            min_improvement: Minimum improvement threshold for suggestions

        Returns:
            List of placement suggestions
        """
        if min_improvement is None:
            min_improvement = (
                self.config_manager.get_performance_config()
                .get("thresholds", {})
                .get("min_improvement_threshold", 0.05)
            )

        suggestions = []

        # Get current heuristic scores
        current_scores = self.calculate_heuristic_scores()

        # For each product, evaluate alternative placements
        for score in current_scores:
            product_id = score.product_id
            current_location = self.layout_optimizer.current_layout[product_id]

            best_alternative = self._find_best_alternative_placement(
                product_id, current_location, score.final_score
            )

            if best_alternative:
                alt_section, alt_score, reasoning = best_alternative
                improvement = alt_score - score.final_score

                if improvement >= min_improvement:
                    suggestion = PlacementSuggestion(
                        product_id=product_id,
                        current_section=self._get_section_for_location(
                            current_location
                        ),
                        suggested_section=alt_section,
                        confidence_score=min(
                            improvement / 0.5, 1.0
                        ),  # Normalize to 0-1
                        expected_improvement=improvement,
                        heuristic_breakdown=score.components,
                        reasoning=reasoning,
                    )
                    suggestions.append(suggestion)

        return sorted(suggestions, key=lambda x: x.expected_improvement, reverse=True)

    def _find_best_alternative_placement(
        self, product_id: int, current_location: ProductLocation, current_score: float
    ) -> Optional[Tuple[str, float, str]]:
        """Find the best alternative placement for a product."""
        best_section = None
        best_score = current_score
        best_reasoning = ""

        # Check section optimizer if available
        if hasattr(self.layout_optimizer, "section_optimizer"):
            sections = self.layout_optimizer.section_optimizer.store_sections

            for section_id, section in sections.items():
                # Skip current section
                current_section_id = self._get_section_for_location(current_location)
                if section_id == current_section_id:
                    continue

                # Create hypothetical location in this section
                hypothetical_location = ProductLocation(
                    product_id=product_id,
                    product_name=current_location.product_name,
                    category=current_location.category,
                    zone=section.parent_zone,
                    x_coordinate=(section.x_range[0] + section.x_range[1]) / 2,
                    y_coordinate=(section.y_range[0] + section.y_range[1]) / 2,
                    shelf_priority=current_location.shelf_priority,
                )

                # Calculate score for hypothetical placement
                hyp_score = self._calculate_hypothetical_score(
                    product_id, hypothetical_location
                )

                if hyp_score > best_score:
                    best_score = hyp_score
                    best_section = section_id
                    best_reasoning = self._generate_placement_reasoning(
                        section, hyp_score - current_score
                    )

        if best_section:
            return (best_section, best_score, best_reasoning)
        return None

    def _calculate_hypothetical_score(
        self, product_id: int, location: ProductLocation
    ) -> float:
        """Calculate heuristic score for a hypothetical placement."""
        # Temporarily update layout for scoring
        original_location = self.layout_optimizer.current_layout.get(product_id)
        self.layout_optimizer.current_layout[product_id] = location

        try:
            scores = self.calculate_heuristic_scores([product_id])
            return scores[0].final_score if scores else 0.0
        finally:
            # Restore original location
            if original_location:
                self.layout_optimizer.current_layout[product_id] = original_location
            else:
                del self.layout_optimizer.current_layout[product_id]

    # Helper methods
    def _get_category_zone_fit(self, category: str, zone: str) -> float:
        """Calculate how well a category fits in a zone."""
        category_zone_mapping = {
            "Produce": {"entrance": 1.0, "center_aisles": 0.3},
            "Dairy": {"dairy_cooler": 1.0, "center_aisles": 0.4},
            "Meat": {"meat_deli": 1.0, "center_aisles": 0.3},
            "Beverages": {"center_aisles": 1.0, "dairy_cooler": 0.6, "checkout": 0.8},
            "Snacks": {"center_aisles": 1.0, "checkout": 0.9},
            "Bakery": {"bakery": 1.0, "entrance": 0.7},
        }

        return category_zone_mapping.get(category, {}).get(zone, 0.5)

    def _get_section_for_location(self, location: ProductLocation) -> str:
        """Get section ID for a given location."""
        # Simplified section mapping based on coordinates
        if hasattr(self.layout_optimizer, "section_optimizer"):
            for (
                section_id,
                section,
            ) in self.layout_optimizer.section_optimizer.store_sections.items():
                if (
                    section.parent_zone == location.zone
                    and section.x_range[0]
                    <= location.x_coordinate
                    <= section.x_range[1]
                    and section.y_range[0]
                    <= location.y_coordinate
                    <= section.y_range[1]
                ):
                    return str(section_id)

        return f"{location.zone}_main"  # Fallback

    def _get_section_utilization(self, section_id: str) -> float:
        """Get utilization rate for a section."""
        if section_id in self._section_utilization_cache:
            return self._section_utilization_cache[section_id]

        # Calculate utilization based on products in section
        if hasattr(self.layout_optimizer, "section_optimizer"):
            section = self.layout_optimizer.section_optimizer.store_sections.get(
                section_id
            )
            if section:
                utilization = self._get_products_in_section(section_id) / section.capacity
                self._section_utilization_cache[section_id] = float(utilization)
                return float(utilization)

        return 0.7  # Default assumption

    def _get_products_in_section(self, section_id: str) -> List[int]:
        """Get list of products in a specific section."""
        if hasattr(self.layout_optimizer, "section_optimizer"):
            section = self.layout_optimizer.section_optimizer.store_sections.get(
                section_id
            )
            if section:
                return cast(List[int], self.layout_optimizer.section_optimizer._get_products_in_section(
                    section
                ))
        return []

    def _get_product_popularity(self, product_id: int) -> float:
        """Get popularity score for a product."""
        if product_id in self._product_popularity_cache:
            return self._product_popularity_cache[product_id]

        popularity = 0.0

        if (
            self.layout_optimizer.association_engine
            and self.layout_optimizer.association_engine.is_trained
        ):

            # Calculate popularity from frequent itemsets
            for (
                itemset_size,
                itemsets,
            ) in self.layout_optimizer.association_engine.frequent_itemsets.items():
                if itemset_size == 1:
                    for itemset, count in itemsets.items():
                        if list(itemset)[0] == product_id:
                            popularity = count / len(
                                self.layout_optimizer.association_engine.transactions
                            )
                            break

        self._product_popularity_cache[product_id] = float(popularity)
        return float(popularity)

    def _calculate_distance(
        self, loc1: ProductLocation, loc2: ProductLocation
    ) -> float:
        """Calculate normalized distance between two locations."""
        return float(np.sqrt(
            (loc1.x_coordinate - loc2.x_coordinate) ** 2
            + (loc1.y_coordinate - loc2.y_coordinate) ** 2
        ))

    def _generate_placement_reasoning(
        self, section: "StoreSection", improvement: float
    ) -> str:
        """Generate human-readable reasoning for placement suggestions."""
        reasons = []

        if improvement > 0.2:
            reasons.append("significant performance improvement expected")
        elif improvement > 0.1:
            reasons.append("moderate performance improvement expected")
        else:
            reasons.append("minor performance improvement expected")

        if section.section_type == "endcap":
            reasons.append("premium endcap visibility")
        elif section.foot_traffic_multiplier > 1.0:
            reasons.append("high-traffic section placement")

        if section.current_utilization < 0.8:
            reasons.append("section has available capacity")

        return f"Move to {section.section_name}: " + "; ".join(reasons)

    def clear_caches(self) -> None:
        """Clear all internal caches."""
        self._section_utilization_cache.clear()
        self._product_popularity_cache.clear()
        self._association_cache.clear()

    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get summary of current optimization state."""
        scores = self.calculate_heuristic_scores()
        suggestions = self.generate_placement_suggestions()

        return {
            "total_products_analyzed": len(scores),
            "average_heuristic_score": np.mean([s.final_score for s in scores]),
            "optimization_suggestions": len(suggestions),
            "high_impact_suggestions": len(
                [s for s in suggestions if s.expected_improvement > 0.2]
            ),
            "crowding_issues": len([s for s in scores if s.crowding_penalty > 0.1]),
            "well_placed_products": len([s for s in scores if s.final_score > 0.8]),
            "configuration_status": (
                "loaded" if self.config_manager.config else "default"
            ),
            "generated_at": pd.Timestamp.now().isoformat(),
        }
