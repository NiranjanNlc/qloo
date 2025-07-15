"""
Supermarket Layout Optimizer

This module provides a comprehensive layout optimization system that combines:
1. Association rules from transaction data (Apriori algorithm)
2. External recommendations from Qloo API
3. Category-based layout principles
4. Customer flow optimization
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from collections import defaultdict
import networkx as nx
import warnings

from association_engine import AprioriAssociationEngine
from qloo_client import QlooClient


@dataclass
class ProductLocation:
    """Represents a product's location in the store."""
    product_id: int
    product_name: str
    category: str
    zone: str
    x_coordinate: float
    y_coordinate: float
    shelf_priority: int  # 1 = eye level, 2 = reach level, 3 = stoop level


@dataclass
class LayoutRecommendation:
    """Represents a layout optimization recommendation."""
    product_id: int
    current_location: Optional[ProductLocation]
    recommended_location: ProductLocation
    reason: str
    confidence_score: float
    potential_lift: float


@dataclass
class StoreZone:
    """Represents a zone in the store layout."""
    zone_id: str
    zone_name: str
    zone_type: str  # entrance, main, checkout, etc.
    foot_traffic_score: float
    x_range: Tuple[float, float]
    y_range: Tuple[float, float]


class SupermarketLayoutOptimizer:
    """
    Main class for optimizing supermarket layouts using multiple data sources.
    """
    
    def __init__(self, qloo_client: Optional[QlooClient] = None):
        """
        Initialize the layout optimizer.
        
        Args:
            qloo_client: Optional Qloo API client for external recommendations
        """
        self.qloo_client = qloo_client
        self.association_engine: Optional[AprioriAssociationEngine] = None
        self.product_catalog: Optional[pd.DataFrame] = None
        self.current_layout: Dict[int, ProductLocation] = {}
        self.store_zones: Dict[str, StoreZone] = {}
        self.association_graph: Optional[nx.Graph] = None
        
        # Initialize default store zones
        self._initialize_default_zones()
    
    def _initialize_default_zones(self):
        """Initialize default store zones with typical supermarket layout."""
        self.store_zones = {
            'entrance': StoreZone(
                zone_id='entrance',
                zone_name='Entrance/Produce',
                zone_type='entrance',
                foot_traffic_score=0.95,
                x_range=(0, 0.3),
                y_range=(0, 1.0)
            ),
            'dairy_cooler': StoreZone(
                zone_id='dairy_cooler',
                zone_name='Dairy & Refrigerated',
                zone_type='perimeter',
                foot_traffic_score=0.85,
                x_range=(0.7, 1.0),
                y_range=(0, 0.5)
            ),
            'meat_deli': StoreZone(
                zone_id='meat_deli',
                zone_name='Meat & Deli',
                zone_type='perimeter',
                foot_traffic_score=0.75,
                x_range=(0.7, 1.0),
                y_range=(0.5, 1.0)
            ),
            'center_aisles': StoreZone(
                zone_id='center_aisles',
                zone_name='Center Aisles',
                zone_type='main',
                foot_traffic_score=0.60,
                x_range=(0.3, 0.7),
                y_range=(0, 1.0)
            ),
            'bakery': StoreZone(
                zone_id='bakery',
                zone_name='Bakery',
                zone_type='perimeter',
                foot_traffic_score=0.70,
                x_range=(0, 0.3),
                y_range=(0.7, 1.0)
            ),
            'checkout': StoreZone(
                zone_id='checkout',
                zone_name='Checkout Area',
                zone_type='checkout',
                foot_traffic_score=1.0,
                x_range=(0, 0.2),
                y_range=(0, 0.3)
            )
        }
    
    def load_data(self, product_catalog: pd.DataFrame, 
                  transactions_df: Optional[pd.DataFrame] = None,
                  current_layout_df: Optional[pd.DataFrame] = None):
        """
        Load product catalog and training data.
        
        Args:
            product_catalog: DataFrame with product information
            transactions_df: Optional transaction data for association mining
            current_layout_df: Optional current layout information
        """
        self.product_catalog = product_catalog.copy()
        
        # Train association engine if transaction data is provided
        if transactions_df is not None:
            self.association_engine = AprioriAssociationEngine(
                min_support=0.03,
                min_confidence=0.2,
                min_lift=1.1,
                max_itemset_size=3
            )
            print("Training association engine for layout optimization...")
            self.association_engine.train(transactions_df)
            
            # Build association graph
            self._build_association_graph()
        
        # Load current layout if provided
        if current_layout_df is not None:
            self._load_current_layout(current_layout_df)
        else:
            self._generate_initial_layout()
    
    def _build_association_graph(self):
        """Build a graph of product associations for layout optimization."""
        if not self.association_engine or not self.association_engine.is_trained:
            return
        
        self.association_graph = nx.Graph()
        
        # Add all products as nodes
        for _, product in self.product_catalog.iterrows():
            self.association_graph.add_node(
                product['product_id'],
                name=product['product_name'],
                category=product['category']
            )
        
        # Add edges based on association rules
        for rule in self.association_engine.association_rules:
            # Create edges between all products in antecedent and consequent
            antecedent_list = list(rule['antecedent'])
            consequent_list = list(rule['consequent'])
            
            for ant_product in antecedent_list:
                for cons_product in consequent_list:
                    if (ant_product in self.association_graph.nodes and 
                        cons_product in self.association_graph.nodes):
                        
                        # Add edge with weight based on lift
                        self.association_graph.add_edge(
                            ant_product, cons_product,
                            weight=rule['lift'],
                            confidence=rule['confidence'],
                            support=rule['support']
                        )
    
    def _load_current_layout(self, layout_df: pd.DataFrame):
        """Load current product layout from DataFrame."""
        for _, row in layout_df.iterrows():
            product_info = self.product_catalog[
                self.product_catalog['product_id'] == row['product_id']
            ].iloc[0]
            
            location = ProductLocation(
                product_id=row['product_id'],
                product_name=product_info['product_name'],
                category=product_info['category'],
                zone=row.get('zone', 'center_aisles'),
                x_coordinate=row.get('x_coordinate', 0.5),
                y_coordinate=row.get('y_coordinate', 0.5),
                shelf_priority=row.get('shelf_priority', 2)
            )
            self.current_layout[row['product_id']] = location
    
    def _generate_initial_layout(self):
        """Generate an initial layout based on product categories."""
        category_zones = {
            'Produce': 'entrance',
            'Dairy': 'dairy_cooler',
            'Meat': 'meat_deli',
            'Bakery': 'bakery',
            'Beverages': 'center_aisles',
            'Snacks': 'center_aisles'
        }
        
        for _, product in self.product_catalog.iterrows():
            category = product['category']
            zone = category_zones.get(category, 'center_aisles')
            zone_info = self.store_zones[zone]
            
            # Random placement within zone
            x_coord = np.random.uniform(zone_info.x_range[0], zone_info.x_range[1])
            y_coord = np.random.uniform(zone_info.y_range[0], zone_info.y_range[1])
            
            location = ProductLocation(
                product_id=product['product_id'],
                product_name=product['product_name'],
                category=category,
                zone=zone,
                x_coordinate=x_coord,
                y_coordinate=y_coord,
                shelf_priority=2  # Default to reach level
            )
            self.current_layout[product['product_id']] = location
    
    def optimize_layout(self, optimization_goals: List[str] = None) -> List[LayoutRecommendation]:
        """
        Generate layout optimization recommendations.
        
        Args:
            optimization_goals: List of optimization goals
                              ('maximize_associations', 'improve_flow', 'category_adjacency')
        
        Returns:
            List of layout recommendations
        """
        if optimization_goals is None:
            optimization_goals = ['maximize_associations', 'improve_flow', 'category_adjacency']
        
        recommendations = []
        
        for goal in optimization_goals:
            if goal == 'maximize_associations':
                recommendations.extend(self._optimize_for_associations())
            elif goal == 'improve_flow':
                recommendations.extend(self._optimize_for_flow())
            elif goal == 'category_adjacency':
                recommendations.extend(self._optimize_for_categories())
        
        # Remove duplicates and rank by confidence
        unique_recommendations = self._deduplicate_recommendations(recommendations)
        return sorted(unique_recommendations, key=lambda x: x.confidence_score, reverse=True)
    
    def _optimize_for_associations(self) -> List[LayoutRecommendation]:
        """Generate recommendations based on product associations."""
        recommendations = []
        
        if not self.association_graph:
            return recommendations
        
        # Find strongly connected products that are far apart
        for product_id in self.association_graph.nodes():
            neighbors = list(self.association_graph.neighbors(product_id))
            if not neighbors:
                continue
            
            current_loc = self.current_layout.get(product_id)
            if not current_loc:
                continue
            
            # Find the strongest associations
            best_neighbor = None
            best_weight = 0
            
            for neighbor_id in neighbors:
                edge_data = self.association_graph[product_id][neighbor_id]
                if edge_data['weight'] > best_weight:
                    best_weight = edge_data['weight']
                    best_neighbor = neighbor_id
            
            if best_neighbor and best_weight > 2.0:  # Strong association
                neighbor_loc = self.current_layout.get(best_neighbor)
                if neighbor_loc:
                    # Calculate current distance
                    current_distance = self._calculate_distance(current_loc, neighbor_loc)
                    
                    if current_distance > 0.3:  # Products are far apart
                        # Recommend moving closer
                        new_location = self._find_optimal_location_near(
                            neighbor_loc, current_loc.category
                        )
                        
                        if new_location:
                            recommendation = LayoutRecommendation(
                                product_id=product_id,
                                current_location=current_loc,
                                recommended_location=new_location,
                                reason=f"Move closer to {neighbor_loc.product_name} (Lift: {best_weight:.2f})",
                                confidence_score=min(best_weight / 5.0, 1.0),
                                potential_lift=best_weight
                            )
                            recommendations.append(recommendation)
        
        return recommendations
    
    def _optimize_for_flow(self) -> List[LayoutRecommendation]:
        """Generate recommendations to improve customer flow."""
        recommendations = []
        
        # High-frequency products should be in high-traffic zones
        if self.association_engine and self.association_engine.is_trained:
            item_popularity = {}
            
            # Calculate item popularity from support scores
            for itemset_size, itemsets in self.association_engine.frequent_itemsets.items():
                if itemset_size == 1:  # Individual items
                    for itemset, count in itemsets.items():
                        product_id = list(itemset)[0]
                        support = count / len(self.association_engine.transactions)
                        item_popularity[product_id] = support
            
            # Recommend high-popularity items for high-traffic zones
            sorted_popularity = sorted(item_popularity.items(), key=lambda x: x[1], reverse=True)
            
            for product_id, popularity in sorted_popularity[:5]:  # Top 5 popular items
                current_loc = self.current_layout.get(product_id)
                if not current_loc:
                    continue
                
                current_zone = self.store_zones[current_loc.zone]
                
                # If popular item is in low-traffic zone, recommend moving
                if current_zone.foot_traffic_score < 0.8 and popularity > 0.3:
                    # Find high-traffic zone that fits the category
                    best_zone = self._find_best_zone_for_category(
                        current_loc.category, min_traffic=0.8
                    )
                    
                    if best_zone:
                        new_location = ProductLocation(
                            product_id=product_id,
                            product_name=current_loc.product_name,
                            category=current_loc.category,
                            zone=best_zone.zone_id,
                            x_coordinate=(best_zone.x_range[0] + best_zone.x_range[1]) / 2,
                            y_coordinate=(best_zone.y_range[0] + best_zone.y_range[1]) / 2,
                            shelf_priority=1  # Eye level for popular items
                        )
                        
                        recommendation = LayoutRecommendation(
                            product_id=product_id,
                            current_location=current_loc,
                            recommended_location=new_location,
                            reason=f"Move popular item to high-traffic zone (Support: {popularity:.2f})",
                            confidence_score=popularity,
                            potential_lift=popularity * best_zone.foot_traffic_score
                        )
                        recommendations.append(recommendation)
        
        return recommendations
    
    def _optimize_for_categories(self) -> List[LayoutRecommendation]:
        """Generate recommendations for better category organization."""
        recommendations = []
        
        # Group products by category and find scattered items
        category_locations = defaultdict(list)
        
        for product_id, location in self.current_layout.items():
            category_locations[location.category].append(location)
        
        # For each category, find outliers
        for category, locations in category_locations.items():
            if len(locations) < 2:
                continue
            
            # Calculate category centroid
            center_x = np.mean([loc.x_coordinate for loc in locations])
            center_y = np.mean([loc.y_coordinate for loc in locations])
            
            # Find products far from centroid
            for location in locations:
                distance_from_center = np.sqrt(
                    (location.x_coordinate - center_x)**2 + 
                    (location.y_coordinate - center_y)**2
                )
                
                if distance_from_center > 0.4:  # Outlier threshold
                    # Recommend moving closer to category center
                    best_zone = self._find_best_zone_for_category(category)
                    
                    if best_zone:
                        new_location = ProductLocation(
                            product_id=location.product_id,
                            product_name=location.product_name,
                            category=location.category,
                            zone=best_zone.zone_id,
                            x_coordinate=center_x,
                            y_coordinate=center_y,
                            shelf_priority=location.shelf_priority
                        )
                        
                        recommendation = LayoutRecommendation(
                            product_id=location.product_id,
                            current_location=location,
                            recommended_location=new_location,
                            reason=f"Group with other {category} products",
                            confidence_score=0.7,
                            potential_lift=1.2
                        )
                        recommendations.append(recommendation)
        
        return recommendations
    
    def get_qloo_recommendations(self, product_id: int, limit: int = 5) -> List[Dict]:
        """Get recommendations from Qloo API for a specific product."""
        if not self.qloo_client:
            return []
        
        try:
            product_name = self._get_product_name(product_id)
            return self.qloo_client.get_product_recommendations(product_name, limit=limit)
        except Exception as e:
            warnings.warn(f"Failed to get Qloo recommendations: {e}")
            return []
    
    def _calculate_distance(self, loc1: ProductLocation, loc2: ProductLocation) -> float:
        """Calculate Euclidean distance between two locations."""
        return np.sqrt(
            (loc1.x_coordinate - loc2.x_coordinate)**2 + 
            (loc1.y_coordinate - loc2.y_coordinate)**2
        )
    
    def _find_optimal_location_near(self, target_location: ProductLocation, 
                                   category: str) -> Optional[ProductLocation]:
        """Find optimal location near a target location for a given category."""
        # Simple implementation: place within 0.2 units of target
        new_x = max(0, min(1, target_location.x_coordinate + np.random.uniform(-0.2, 0.2)))
        new_y = max(0, min(1, target_location.y_coordinate + np.random.uniform(-0.2, 0.2)))
        
        # Find appropriate zone
        best_zone = None
        for zone in self.store_zones.values():
            if (zone.x_range[0] <= new_x <= zone.x_range[1] and 
                zone.y_range[0] <= new_y <= zone.y_range[1]):
                best_zone = zone
                break
        
        if not best_zone:
            best_zone = self.store_zones['center_aisles']
        
        return ProductLocation(
            product_id=0,  # Will be set by caller
            product_name="",  # Will be set by caller
            category=category,
            zone=best_zone.zone_id,
            x_coordinate=new_x,
            y_coordinate=new_y,
            shelf_priority=2
        )
    
    def _find_best_zone_for_category(self, category: str, 
                                    min_traffic: float = 0.0) -> Optional[StoreZone]:
        """Find the best zone for a given category."""
        category_zone_preferences = {
            'Produce': ['entrance'],
            'Dairy': ['dairy_cooler'],
            'Meat': ['meat_deli'],
            'Bakery': ['bakery'],
            'Beverages': ['center_aisles', 'dairy_cooler'],
            'Snacks': ['center_aisles', 'checkout']
        }
        
        preferred_zones = category_zone_preferences.get(category, ['center_aisles'])
        
        for zone_id in preferred_zones:
            zone = self.store_zones[zone_id]
            if zone.foot_traffic_score >= min_traffic:
                return zone
        
        # Fallback to highest traffic zone that meets criteria
        best_zone = None
        best_traffic = 0
        
        for zone in self.store_zones.values():
            if zone.foot_traffic_score >= min_traffic and zone.foot_traffic_score > best_traffic:
                best_zone = zone
                best_traffic = zone.foot_traffic_score
        
        return best_zone
    
    def _deduplicate_recommendations(self, recommendations: List[LayoutRecommendation]) -> List[LayoutRecommendation]:
        """Remove duplicate recommendations for the same product."""
        seen_products = set()
        unique_recommendations = []
        
        for rec in recommendations:
            if rec.product_id not in seen_products:
                seen_products.add(rec.product_id)
                unique_recommendations.append(rec)
        
        return unique_recommendations
    
    def _get_product_name(self, product_id: int) -> str:
        """Get product name from catalog."""
        if self.product_catalog is not None:
            product_row = self.product_catalog[self.product_catalog['product_id'] == product_id]
            if not product_row.empty:
                return product_row.iloc[0]['product_name']
        return f"Product {product_id}"
    
    def generate_layout_report(self) -> Dict:
        """Generate a comprehensive layout analysis report."""
        report = {
            'total_products': len(self.current_layout),
            'zones_utilized': len(set(loc.zone for loc in self.current_layout.values())),
            'category_distribution': {},
            'zone_utilization': {},
            'association_score': 0.0,
            'optimization_potential': 0.0
        }
        
        # Category distribution
        for location in self.current_layout.values():
            category = location.category
            if category not in report['category_distribution']:
                report['category_distribution'][category] = 0
            report['category_distribution'][category] += 1
        
        # Zone utilization
        for location in self.current_layout.values():
            zone = location.zone
            if zone not in report['zone_utilization']:
                report['zone_utilization'][zone] = 0
            report['zone_utilization'][zone] += 1
        
        # Calculate association score if engine is available
        if self.association_graph:
            total_weight = 0
            total_edges = 0
            
            for product_id, location in self.current_layout.items():
                neighbors = list(self.association_graph.neighbors(product_id))
                for neighbor_id in neighbors:
                    neighbor_location = self.current_layout.get(neighbor_id)
                    if neighbor_location:
                        distance = self._calculate_distance(location, neighbor_location)
                        edge_weight = self.association_graph[product_id][neighbor_id]['weight']
                        
                        # Good score if strong associations are close together
                        association_contribution = edge_weight / (1 + distance)
                        total_weight += association_contribution
                        total_edges += 1
            
            if total_edges > 0:
                report['association_score'] = total_weight / total_edges
        
        return report 