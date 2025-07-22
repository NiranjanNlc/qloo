"""
Centralized Scoring Algorithms Module

This module consolidates all scoring algorithms used throughout the application
and provides comprehensive evaluation metrics including ROC analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import logging
from sklearn.metrics import roc_curve, auc, precision_recall_curve, classification_report
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import joblib
from pathlib import Path

logger = logging.getLogger(__name__)


class ScoreType(str, Enum):
    """Types of scoring algorithms available."""
    ASSOCIATION_CONFIDENCE = "association_confidence"
    LAYOUT_OPTIMIZATION = "layout_optimization"
    COMBO_EFFECTIVENESS = "combo_effectiveness"
    SECTION_PERFORMANCE = "section_performance"
    POPULARITY_PREDICTION = "popularity_prediction"
    CROSS_SELLING_POTENTIAL = "cross_selling_potential"


@dataclass
class ScoringResult:
    """Results from a scoring algorithm."""
    score_type: ScoreType
    entity_id: Union[int, str]
    score: float
    confidence: float
    metadata: Dict[str, Any]
    computed_at: datetime


@dataclass
class ROCMetrics:
    """ROC analysis results."""
    auc_score: float
    fpr: np.ndarray
    tpr: np.ndarray
    thresholds: np.ndarray
    optimal_threshold: float
    precision: np.ndarray
    recall: np.ndarray
    precision_recall_auc: float
    confusion_matrix: np.ndarray
    classification_report: Dict[str, Any]


class AssociationScorer:
    """Scoring algorithms for association rules and product relationships."""
    
    @staticmethod
    def score_association_confidence(antecedent: List[int], 
                                   consequent: List[int], 
                                   transactions: List[List[int]]) -> float:
        """
        Calculate association rule confidence score.
        
        Args:
            antecedent: List of product IDs in antecedent
            consequent: List of product IDs in consequent
            transactions: List of transaction data
            
        Returns:
            Confidence score (0.0 to 1.0)
        """
        if not transactions:
            return 0.0
        
        antecedent_set = set(antecedent)
        consequent_set = set(consequent)
        
        # Count transactions containing antecedent
        antecedent_count = sum(1 for transaction in transactions 
                             if antecedent_set.issubset(set(transaction)))
        
        if antecedent_count == 0:
            return 0.0
        
        # Count transactions containing both antecedent and consequent
        both_count = sum(1 for transaction in transactions 
                        if antecedent_set.issubset(set(transaction)) and 
                        consequent_set.issubset(set(transaction)))
        
        return both_count / antecedent_count
    
    @staticmethod
    def score_lift(antecedent: List[int], 
                   consequent: List[int], 
                   transactions: List[List[int]]) -> float:
        """Calculate lift score for association rule."""
        if not transactions:
            return 0.0
        
        antecedent_set = set(antecedent)
        consequent_set = set(consequent)
        total_transactions = len(transactions)
        
        # Calculate support for antecedent
        antecedent_support = sum(1 for transaction in transactions 
                               if antecedent_set.issubset(set(transaction))) / total_transactions
        
        # Calculate support for consequent
        consequent_support = sum(1 for transaction in transactions 
                               if consequent_set.issubset(set(transaction))) / total_transactions
        
        # Calculate confidence
        confidence = AssociationScorer.score_association_confidence(
            antecedent, consequent, transactions
        )
        
        if consequent_support == 0:
            return 0.0
        
        return confidence / consequent_support
    
    @staticmethod
    def score_conviction(antecedent: List[int], 
                        consequent: List[int], 
                        transactions: List[List[int]]) -> float:
        """Calculate conviction score for association rule."""
        confidence = AssociationScorer.score_association_confidence(
            antecedent, consequent, transactions
        )
        
        if confidence == 1.0:
            return float('inf')
        if confidence == 0.0:
            return 0.0
        
        consequent_set = set(consequent)
        total_transactions = len(transactions)
        
        # Calculate support for NOT consequent
        not_consequent_support = sum(1 for transaction in transactions 
                                   if not consequent_set.issubset(set(transaction))) / total_transactions
        
        if not_consequent_support == 0:
            return float('inf')
        
        return not_consequent_support / (1 - confidence)


class ComboScorer:
    """Scoring algorithms for product combination effectiveness."""
    
    @staticmethod
    def score_combo_effectiveness(combo_products: List[int], 
                                transaction_data: List[List[int]],
                                combo_metadata: Dict[str, Any]) -> float:
        """
        Score the effectiveness of a product combination.
        
        Args:
            combo_products: List of product IDs in the combo
            transaction_data: Historical transaction data
            combo_metadata: Additional metadata about the combo
            
        Returns:
            Effectiveness score (0.0 to 1.0)
        """
        if len(combo_products) < 2:
            return 0.0
        
        combo_set = set(combo_products)
        total_transactions = len(transaction_data)
        
        if total_transactions == 0:
            return 0.0
        
        # Calculate combo frequency
        combo_frequency = sum(1 for transaction in transaction_data 
                            if combo_set.issubset(set(transaction))) / total_transactions
        
        # Calculate individual product frequencies
        individual_frequencies = []
        for product in combo_products:
            freq = sum(1 for transaction in transaction_data 
                      if product in transaction) / total_transactions
            individual_frequencies.append(freq)
        
        # Calculate expected frequency if products were independent
        expected_frequency = np.prod(individual_frequencies)
        
        # Lift-based effectiveness
        if expected_frequency == 0:
            lift_score = 0.0
        else:
            lift_score = min(combo_frequency / expected_frequency / 10.0, 1.0)  # Normalize
        
        # Size penalty (larger combos are harder to achieve)
        size_penalty = 1.0 - (len(combo_products) - 2) * 0.1
        size_penalty = max(size_penalty, 0.5)
        
        # Metadata bonuses
        discount_bonus = 0.0
        if 'expected_discount_percent' in combo_metadata:
            # Moderate discount is optimal (not too high, not too low)
            discount = combo_metadata['expected_discount_percent']
            if 10 <= discount <= 20:
                discount_bonus = 0.1
            elif 5 <= discount <= 25:
                discount_bonus = 0.05
        
        final_score = (lift_score * size_penalty) + discount_bonus
        return min(final_score, 1.0)
    
    @staticmethod
    def score_cross_selling_potential(product_id: int, 
                                    candidate_products: List[int],
                                    transaction_data: List[List[int]]) -> Dict[int, float]:
        """
        Score cross-selling potential between a product and candidates.
        
        Returns:
            Dictionary mapping candidate product IDs to cross-selling scores
        """
        scores = {}
        total_transactions = len(transaction_data)
        
        if total_transactions == 0:
            return {pid: 0.0 for pid in candidate_products}
        
        # Count transactions containing the main product
        main_product_transactions = [t for t in transaction_data if product_id in t]
        main_product_count = len(main_product_transactions)
        
        if main_product_count == 0:
            return {pid: 0.0 for pid in candidate_products}
        
        for candidate_id in candidate_products:
            if candidate_id == product_id:
                scores[candidate_id] = 0.0
                continue
            
            # Count co-occurrences
            co_occurrence_count = sum(1 for transaction in main_product_transactions 
                                    if candidate_id in transaction)
            
            # Calculate conditional probability
            conditional_prob = co_occurrence_count / main_product_count
            
            # Calculate baseline probability
            baseline_prob = sum(1 for transaction in transaction_data 
                              if candidate_id in transaction) / total_transactions
            
            # Calculate lift
            if baseline_prob == 0:
                lift = 0.0
            else:
                lift = conditional_prob / baseline_prob
            
            # Normalize score
            score = min(lift / 5.0, 1.0)  # Normalize to 0-1 range
            scores[candidate_id] = score
        
        return scores


class LayoutScorer:
    """Scoring algorithms for layout optimization and section performance."""
    
    @staticmethod
    def score_section_performance(section_id: str,
                                products_in_section: List[int],
                                section_metadata: Dict[str, Any],
                                transaction_data: List[List[int]]) -> float:
        """
        Score the performance of a store section.
        
        Args:
            section_id: Identifier for the section
            products_in_section: List of product IDs in the section
            section_metadata: Metadata about the section (capacity, traffic, etc.)
            transaction_data: Historical transaction data
            
        Returns:
            Performance score (0.0 to 1.0)
        """
        if not products_in_section:
            return 0.0
        
        # Utilization score
        capacity = section_metadata.get('capacity', len(products_in_section))
        utilization = len(products_in_section) / capacity
        utilization_score = min(utilization / 0.85, 1.0)  # Optimal at 85% utilization
        
        # Traffic efficiency score
        traffic_multiplier = section_metadata.get('traffic_multiplier', 1.0)
        
        # Calculate product popularity in this section
        total_transactions = len(transaction_data)
        if total_transactions == 0:
            popularity_score = 0.5
        else:
            section_transaction_count = sum(
                1 for transaction in transaction_data 
                if any(product in transaction for product in products_in_section)
            )
            popularity_score = min(section_transaction_count / total_transactions / 0.8, 1.0)
        
        # Cross-selling effectiveness within section
        cross_sell_score = LayoutScorer._calculate_section_cross_selling(
            products_in_section, transaction_data
        )
        
        # Weighted final score
        weights = {
            'utilization': 0.3,
            'popularity': 0.4,
            'cross_selling': 0.2,
            'traffic': 0.1
        }
        
        final_score = (
            utilization_score * weights['utilization'] +
            popularity_score * weights['popularity'] +
            cross_sell_score * weights['cross_selling'] +
            traffic_multiplier * weights['traffic']
        )
        
        return min(final_score, 1.0)
    
    @staticmethod
    def _calculate_section_cross_selling(products: List[int], 
                                       transaction_data: List[List[int]]) -> float:
        """Calculate cross-selling effectiveness within a section."""
        if len(products) < 2:
            return 0.0
        
        total_pairs = len(products) * (len(products) - 1) / 2
        total_transactions = len(transaction_data)
        
        if total_transactions == 0:
            return 0.0
        
        cross_sell_strength = 0.0
        
        for i, product1 in enumerate(products):
            for product2 in products[i+1:]:
                # Count co-occurrences
                co_occurrences = sum(1 for transaction in transaction_data 
                                   if product1 in transaction and product2 in transaction)
                
                if co_occurrences > 0:
                    # Individual frequencies
                    freq1 = sum(1 for t in transaction_data if product1 in t) / total_transactions
                    freq2 = sum(1 for t in transaction_data if product2 in t) / total_transactions
                    
                    # Expected co-occurrence if independent
                    expected = freq1 * freq2 * total_transactions
                    
                    if expected > 0:
                        lift = co_occurrences / expected
                        cross_sell_strength += min(lift / 3.0, 1.0)  # Normalize
        
        return cross_sell_strength / total_pairs if total_pairs > 0 else 0.0


class PopularityScorer:
    """Scoring algorithms for product popularity prediction."""
    
    @staticmethod
    def score_popularity_trend(product_id: int,
                             transaction_history: List[Tuple[datetime, List[int]]],
                             time_window_days: int = 30) -> float:
        """
        Score the popularity trend for a product.
        
        Args:
            product_id: Product to analyze
            transaction_history: List of (timestamp, transaction) tuples
            time_window_days: Number of days to analyze for trend
            
        Returns:
            Trend score (-1.0 to 1.0, negative = declining, positive = growing)
        """
        if not transaction_history:
            return 0.0
        
        # Sort by timestamp
        sorted_history = sorted(transaction_history, key=lambda x: x[0])
        
        # Calculate daily frequencies
        daily_counts = {}
        for timestamp, transaction in sorted_history:
            date_key = timestamp.date()
            if date_key not in daily_counts:
                daily_counts[date_key] = 0
            if product_id in transaction:
                daily_counts[date_key] += 1
        
        if len(daily_counts) < 7:  # Need at least a week of data
            return 0.0
        
        # Get recent vs historical averages
        dates = sorted(daily_counts.keys())
        recent_dates = dates[-time_window_days//2:]
        historical_dates = dates[:-time_window_days//2]
        
        if not historical_dates:
            return 0.0
        
        recent_avg = np.mean([daily_counts.get(date, 0) for date in recent_dates])
        historical_avg = np.mean([daily_counts.get(date, 0) for date in historical_dates])
        
        if historical_avg == 0:
            return 1.0 if recent_avg > 0 else 0.0
        
        # Calculate relative change
        relative_change = (recent_avg - historical_avg) / historical_avg
        
        # Normalize to -1.0 to 1.0 range
        return np.tanh(relative_change)
    
    @staticmethod
    def score_seasonal_factor(product_id: int,
                            transaction_history: List[Tuple[datetime, List[int]]],
                            current_season: str) -> float:
        """Score how well a product performs in the current season."""
        season_mapping = {
            'spring': [3, 4, 5],
            'summer': [6, 7, 8], 
            'fall': [9, 10, 11],
            'winter': [12, 1, 2]
        }
        
        current_months = season_mapping.get(current_season.lower(), [])
        if not current_months:
            return 0.5  # Neutral if season unknown
        
        # Calculate average frequency by season
        seasonal_counts = {season: [] for season in season_mapping.keys()}
        
        for timestamp, transaction in transaction_history:
            month = timestamp.month
            if product_id in transaction:
                for season, months in season_mapping.items():
                    if month in months:
                        seasonal_counts[season].append(1)
                        break
                else:
                    for season in seasonal_counts:
                        seasonal_counts[season].append(0)
        
        # Calculate seasonal averages
        seasonal_averages = {}
        for season, counts in seasonal_counts.items():
            seasonal_averages[season] = np.mean(counts) if counts else 0.0
        
        overall_average = np.mean(list(seasonal_averages.values()))
        current_seasonal_avg = seasonal_averages.get(current_season.lower(), overall_average)
        
        if overall_average == 0:
            return 0.5
        
        # Return factor relative to overall average
        return min(current_seasonal_avg / overall_average, 2.0) / 2.0  # Normalize to 0-1


class ROCAnalyzer:
    """Provides ROC analysis and model evaluation capabilities."""
    
    @staticmethod
    def analyze_binary_predictions(y_true: np.ndarray, 
                                 y_scores: np.ndarray,
                                 pos_label: int = 1) -> ROCMetrics:
        """
        Perform comprehensive ROC analysis for binary predictions.
        
        Args:
            y_true: True binary labels
            y_scores: Prediction scores/probabilities
            pos_label: Label for positive class
            
        Returns:
            ROCMetrics object with analysis results
        """
        # Calculate ROC curve
        fpr, tpr, roc_thresholds = roc_curve(y_true, y_scores, pos_label=pos_label)
        roc_auc = auc(fpr, tpr)
        
        # Calculate precision-recall curve
        precision, recall, pr_thresholds = precision_recall_curve(y_true, y_scores, pos_label=pos_label)
        pr_auc = auc(recall, precision)
        
        # Find optimal threshold (Youden's J statistic)
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = roc_thresholds[optimal_idx]
        
        # Generate predictions with optimal threshold
        y_pred = (y_scores >= optimal_threshold).astype(int)
        
        # Calculate confusion matrix
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Generate classification report
        try:
            class_report = classification_report(y_true, y_pred, output_dict=True)
        except Exception:
            class_report = {}
        
        return ROCMetrics(
            auc_score=roc_auc,
            fpr=fpr,
            tpr=tpr,
            thresholds=roc_thresholds,
            optimal_threshold=optimal_threshold,
            precision=precision,
            recall=recall,
            precision_recall_auc=pr_auc,
            confusion_matrix=cm,
            classification_report=class_report
        )
    
    @staticmethod
    def plot_roc_analysis(metrics: ROCMetrics, 
                        title: str = "ROC Analysis",
                        save_path: Optional[str] = None) -> plt.Figure:
        """
        Create comprehensive ROC analysis plots.
        
        Args:
            metrics: ROC metrics to plot
            title: Title for the plot
            save_path: Optional path to save the plot
            
        Returns:
            Matplotlib figure object
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(title, fontsize=16)
        
        # ROC Curve
        ax1.plot(metrics.fpr, metrics.tpr, linewidth=2, 
                label=f'ROC curve (AUC = {metrics.auc_score:.3f})')
        ax1.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
        ax1.scatter(metrics.fpr[np.argmax(metrics.tpr - metrics.fpr)], 
                   metrics.tpr[np.argmax(metrics.tpr - metrics.fpr)], 
                   color='red', s=100, label=f'Optimal (threshold={metrics.optimal_threshold:.3f})')
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.set_title('ROC Curve')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Precision-Recall Curve
        ax2.plot(metrics.recall, metrics.precision, linewidth=2,
                label=f'PR curve (AUC = {metrics.precision_recall_auc:.3f})')
        ax2.set_xlabel('Recall')
        ax2.set_ylabel('Precision')
        ax2.set_title('Precision-Recall Curve')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Confusion Matrix
        sns.heatmap(metrics.confusion_matrix, annot=True, fmt='d', cmap='Blues', ax=ax3)
        ax3.set_title('Confusion Matrix')
        ax3.set_xlabel('Predicted')
        ax3.set_ylabel('Actual')
        
        # Threshold Analysis
        threshold_range = np.linspace(0, 1, 100)
        precision_scores = []
        recall_scores = []
        f1_scores = []
        
        for thresh in threshold_range:
            y_pred_thresh = (np.linspace(0, 1, len(metrics.precision)) >= thresh).astype(int)
            # Simplified calculation for demonstration
            if thresh < len(metrics.precision):
                idx = min(int(thresh * len(metrics.precision)), len(metrics.precision) - 1)
                precision_scores.append(metrics.precision[idx])
                recall_scores.append(metrics.recall[idx])
                f1_scores.append(2 * metrics.precision[idx] * metrics.recall[idx] / 
                               (metrics.precision[idx] + metrics.recall[idx] + 1e-8))
            else:
                precision_scores.append(0)
                recall_scores.append(0)
                f1_scores.append(0)
        
        ax4.plot(threshold_range, precision_scores, label='Precision', linewidth=2)
        ax4.plot(threshold_range, recall_scores, label='Recall', linewidth=2)
        ax4.plot(threshold_range, f1_scores, label='F1 Score', linewidth=2)
        ax4.axvline(metrics.optimal_threshold, color='red', linestyle='--', 
                   label=f'Optimal Threshold ({metrics.optimal_threshold:.3f})')
        ax4.set_xlabel('Threshold')
        ax4.set_ylabel('Score')
        ax4.set_title('Threshold Analysis')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig


class ScoringsEngine:
    """Main engine that coordinates all scoring algorithms."""
    
    def __init__(self):
        """Initialize the scoring engine."""
        self.association_scorer = AssociationScorer()
        self.combo_scorer = ComboScorer()
        self.layout_scorer = LayoutScorer()
        self.popularity_scorer = PopularityScorer()
        self.roc_analyzer = ROCAnalyzer()
        
        self.scoring_history: List[ScoringResult] = []
    
    def score_entity(self, 
                    score_type: ScoreType,
                    entity_id: Union[int, str],
                    **kwargs) -> ScoringResult:
        """
        Score an entity using the specified scoring algorithm.
        
        Args:
            score_type: Type of scoring to perform
            entity_id: ID of the entity to score
            **kwargs: Additional parameters for the scoring algorithm
            
        Returns:
            ScoringResult object
        """
        score = 0.0
        confidence = 0.0
        metadata = {}
        
        try:
            if score_type == ScoreType.ASSOCIATION_CONFIDENCE:
                score = self.association_scorer.score_association_confidence(**kwargs)
                confidence = 0.9 if kwargs.get('transactions') else 0.1
                
            elif score_type == ScoreType.COMBO_EFFECTIVENESS:
                score = self.combo_scorer.score_combo_effectiveness(**kwargs)
                confidence = 0.8
                
            elif score_type == ScoreType.SECTION_PERFORMANCE:
                score = self.layout_scorer.score_section_performance(str(entity_id), **kwargs)
                confidence = 0.85
                
            elif score_type == ScoreType.POPULARITY_PREDICTION:
                score = self.popularity_scorer.score_popularity_trend(int(entity_id), **kwargs)
                confidence = 0.7
                
            elif score_type == ScoreType.CROSS_SELLING_POTENTIAL:
                scores_dict = self.combo_scorer.score_cross_selling_potential(int(entity_id), **kwargs)
                score = max(scores_dict.values()) if scores_dict else 0.0
                metadata['cross_selling_scores'] = scores_dict
                confidence = 0.75
                
            else:
                logger.warning(f"Unknown score type: {score_type}")
                
        except Exception as e:
            logger.error(f"Error calculating score for {entity_id}: {e}")
            score = 0.0
            confidence = 0.0
            metadata['error'] = str(e)
        
        result = ScoringResult(
            score_type=score_type,
            entity_id=entity_id,
            score=score,
            confidence=confidence,
            metadata=metadata,
            computed_at=datetime.now()
        )
        
        self.scoring_history.append(result)
        return result
    
    def batch_score(self, 
                   score_requests: List[Tuple[ScoreType, Union[int, str], Dict[str, Any]]]) -> List[ScoringResult]:
        """
        Perform batch scoring for multiple entities.
        
        Args:
            score_requests: List of (score_type, entity_id, kwargs) tuples
            
        Returns:
            List of ScoringResult objects
        """
        results = []
        for score_type, entity_id, kwargs in score_requests:
            result = self.score_entity(score_type, entity_id, **kwargs)
            results.append(result)
        
        return results
    
    def evaluate_model_performance(self, 
                                 predictions: List[float],
                                 ground_truth: List[int],
                                 model_name: str = "Unknown") -> ROCMetrics:
        """
        Evaluate model performance using ROC analysis.
        
        Args:
            predictions: Model prediction scores
            ground_truth: True binary labels
            model_name: Name of the model being evaluated
            
        Returns:
            ROC metrics for the model
        """
        y_true = np.array(ground_truth)
        y_scores = np.array(predictions)
        
        metrics = self.roc_analyzer.analyze_binary_predictions(y_true, y_scores)
        
        logger.info(f"Model '{model_name}' Performance:")
        logger.info(f"  ROC AUC: {metrics.auc_score:.3f}")
        logger.info(f"  Precision-Recall AUC: {metrics.precision_recall_auc:.3f}")
        logger.info(f"  Optimal Threshold: {metrics.optimal_threshold:.3f}")
        
        return metrics
    
    def save_scoring_model(self, model_data: Dict[str, Any], filepath: str):
        """Save scoring model data to disk."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model_data, filepath)
        logger.info(f"Scoring model saved to {filepath}")
    
    def load_scoring_model(self, filepath: str) -> Dict[str, Any]:
        """Load scoring model data from disk."""
        return joblib.load(filepath)
    
    def get_scoring_summary(self) -> Dict[str, Any]:
        """Get summary of all scoring activities."""
        if not self.scoring_history:
            return {'message': 'No scoring history available'}
        
        by_type = {}
        for result in self.scoring_history:
            score_type = result.score_type.value
            if score_type not in by_type:
                by_type[score_type] = []
            by_type[score_type].append(result.score)
        
        summary = {
            'total_scores_computed': len(self.scoring_history),
            'unique_score_types': len(by_type),
            'average_scores_by_type': {
                st: np.mean(scores) for st, scores in by_type.items()
            },
            'last_scoring_time': max(r.computed_at for r in self.scoring_history).isoformat(),
            'score_distribution': {
                st: {
                    'count': len(scores),
                    'mean': np.mean(scores),
                    'std': np.std(scores),
                    'min': np.min(scores),
                    'max': np.max(scores)
                }
                for st, scores in by_type.items()
            }
        }
        
        return summary 