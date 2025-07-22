"""
SQL Aggregation Views for Association Analysis

This module provides SQL views and Pandas integration for analyzing
association rule changes week-over-week and generating insights.
"""

import sqlite3
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from pathlib import Path
import logging

from models import Combo, Product

logger = logging.getLogger(__name__)


class AssociationAggregator:
    """Provides SQL views and aggregation for association rule analysis."""
    
    def __init__(self, db_path: str = "data/association_analytics.db"):
        """
        Initialize the aggregator with database connection.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(exist_ok=True)
        
        # Initialize database and create tables
        self._init_database()
    
    def _init_database(self):
        """Initialize database schema with required tables and views."""
        with sqlite3.connect(self.db_path) as conn:
            # Create tables for storing association data
            conn.executescript("""
                -- Association rules table
                CREATE TABLE IF NOT EXISTS association_rules (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    week_start_date DATE NOT NULL,
                    rule_id TEXT NOT NULL,
                    antecedent TEXT NOT NULL,  -- JSON array of product IDs
                    consequent TEXT NOT NULL,  -- JSON array of product IDs
                    confidence REAL NOT NULL,
                    support REAL NOT NULL,
                    lift REAL NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(week_start_date, rule_id)
                );
                
                -- Combos table for tracking generated combinations
                CREATE TABLE IF NOT EXISTS combos (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    week_start_date DATE NOT NULL,
                    combo_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    products TEXT NOT NULL,  -- JSON array of product IDs
                    confidence_score REAL NOT NULL,
                    support REAL NOT NULL,
                    lift REAL NOT NULL,
                    expected_discount_percent REAL,
                    category_mix TEXT,  -- JSON array of categories
                    is_active INTEGER DEFAULT 1,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(week_start_date, combo_id)
                );
                
                -- Weekly metrics summary table
                CREATE TABLE IF NOT EXISTS weekly_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    week_start_date DATE NOT NULL UNIQUE,
                    total_rules INTEGER NOT NULL,
                    high_confidence_rules INTEGER NOT NULL,
                    avg_confidence REAL NOT NULL,
                    avg_support REAL NOT NULL,
                    avg_lift REAL NOT NULL,
                    total_combos INTEGER NOT NULL,
                    active_combos INTEGER NOT NULL,
                    avg_discount_percent REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                
                -- Indexes for performance
                CREATE INDEX IF NOT EXISTS idx_rules_week ON association_rules(week_start_date);
                CREATE INDEX IF NOT EXISTS idx_combos_week ON combos(week_start_date);
                CREATE INDEX IF NOT EXISTS idx_rules_confidence ON association_rules(confidence);
                CREATE INDEX IF NOT EXISTS idx_combos_confidence ON combos(confidence_score);
            """)
            
            # Create views for analysis
            self._create_analysis_views(conn)
    
    def _create_analysis_views(self, conn: sqlite3.Connection):
        """Create SQL views for common analysis patterns."""
        conn.executescript("""
            -- View: Week-over-week rule comparison
            DROP VIEW IF EXISTS week_over_week_rules;
            CREATE VIEW week_over_week_rules AS
            WITH week_pairs AS (
                SELECT DISTINCT 
                    w1.week_start_date as current_week,
                    w2.week_start_date as previous_week
                FROM weekly_metrics w1
                JOIN weekly_metrics w2 ON 
                    date(w1.week_start_date) = date(w2.week_start_date, '+7 days')
            )
            SELECT 
                wp.current_week,
                wp.previous_week,
                
                -- Current week metrics
                cur.total_rules as current_rules,
                cur.high_confidence_rules as current_high_conf,
                cur.avg_confidence as current_avg_confidence,
                cur.avg_lift as current_avg_lift,
                
                -- Previous week metrics
                prev.total_rules as previous_rules,
                prev.high_confidence_rules as previous_high_conf,
                prev.avg_confidence as previous_avg_confidence,
                prev.avg_lift as previous_avg_lift,
                
                -- Changes
                (cur.total_rules - prev.total_rules) as rules_change,
                (cur.high_confidence_rules - prev.high_confidence_rules) as high_conf_change,
                (cur.avg_confidence - prev.avg_confidence) as confidence_change,
                (cur.avg_lift - prev.avg_lift) as lift_change,
                
                -- Percentage changes
                ROUND(
                    CASE 
                        WHEN prev.total_rules > 0 
                        THEN ((cur.total_rules - prev.total_rules) * 100.0 / prev.total_rules)
                        ELSE 0 
                    END, 2
                ) as rules_change_pct,
                
                ROUND(
                    CASE 
                        WHEN prev.avg_confidence > 0 
                        THEN ((cur.avg_confidence - prev.avg_confidence) * 100.0 / prev.avg_confidence)
                        ELSE 0 
                    END, 2
                ) as confidence_change_pct
                
            FROM week_pairs wp
            JOIN weekly_metrics cur ON cur.week_start_date = wp.current_week
            JOIN weekly_metrics prev ON prev.week_start_date = wp.previous_week;
            
            -- View: New vs returning associations
            DROP VIEW IF EXISTS new_vs_returning_associations;
            CREATE VIEW new_vs_returning_associations AS
            WITH rule_weeks AS (
                SELECT 
                    rule_id,
                    week_start_date,
                    confidence,
                    support,
                    lift,
                    LAG(week_start_date) OVER (PARTITION BY rule_id ORDER BY week_start_date) as prev_week
                FROM association_rules
            )
            SELECT 
                week_start_date,
                COUNT(*) as total_rules,
                COUNT(CASE WHEN prev_week IS NULL THEN 1 END) as new_rules,
                COUNT(CASE WHEN prev_week IS NOT NULL THEN 1 END) as returning_rules,
                ROUND(
                    COUNT(CASE WHEN prev_week IS NULL THEN 1 END) * 100.0 / COUNT(*), 2
                ) as new_rules_pct,
                AVG(CASE WHEN prev_week IS NULL THEN confidence END) as avg_new_confidence,
                AVG(CASE WHEN prev_week IS NOT NULL THEN confidence END) as avg_returning_confidence
            FROM rule_weeks
            GROUP BY week_start_date
            ORDER BY week_start_date;
            
            -- View: Top performing rule trends
            DROP VIEW IF EXISTS top_rule_trends;
            CREATE VIEW top_rule_trends AS
            SELECT 
                ar.rule_id,
                ar.antecedent,
                ar.consequent,
                COUNT(*) as weeks_active,
                AVG(ar.confidence) as avg_confidence,
                AVG(ar.support) as avg_support,
                AVG(ar.lift) as avg_lift,
                MIN(ar.confidence) as min_confidence,
                MAX(ar.confidence) as max_confidence,
                CASE 
                    WHEN COUNT(*) >= 4 THEN 'Stable'
                    WHEN COUNT(*) >= 2 THEN 'Emerging'
                    ELSE 'New'
                END as trend_category
            FROM association_rules ar
            GROUP BY ar.rule_id, ar.antecedent, ar.consequent
            HAVING COUNT(*) >= 1
            ORDER BY avg_confidence DESC, avg_lift DESC;
            
            -- View: Category-based association insights
            DROP VIEW IF EXISTS category_association_insights;
            CREATE VIEW category_association_insights AS
            SELECT 
                c.week_start_date,
                c.category_mix,
                COUNT(*) as combo_count,
                AVG(c.confidence_score) as avg_confidence,
                AVG(c.expected_discount_percent) as avg_discount,
                AVG(c.lift) as avg_lift,
                COUNT(CASE WHEN c.confidence_score >= 0.9 THEN 1 END) as high_conf_count
            FROM combos c
            WHERE c.category_mix IS NOT NULL
            GROUP BY c.week_start_date, c.category_mix
            ORDER BY c.week_start_date DESC, avg_confidence DESC;
        """)
    
    def store_association_rules(self, rules: List[Dict], week_start: datetime):
        """
        Store association rules for a given week.
        
        Args:
            rules: List of association rule dictionaries
            week_start: Start date of the week
        """
        with sqlite3.connect(self.db_path) as conn:
            for rule in rules:
                conn.execute("""
                    INSERT OR REPLACE INTO association_rules 
                    (week_start_date, rule_id, antecedent, consequent, 
                     confidence, support, lift)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    week_start.date(),
                    rule.get('rule_id', f"rule_{hash(str(rule))}"),
                    str(rule.get('antecedent', [])),
                    str(rule.get('consequent', [])),
                    rule.get('confidence', 0.0),
                    rule.get('support', 0.0),
                    rule.get('lift', 0.0)
                ))
    
    def store_combos(self, combos: List[Combo], week_start: datetime):
        """
        Store combo data for a given week.
        
        Args:
            combos: List of Combo objects
            week_start: Start date of the week
        """
        with sqlite3.connect(self.db_path) as conn:
            for combo in combos:
                conn.execute("""
                    INSERT OR REPLACE INTO combos 
                    (week_start_date, combo_id, name, products, confidence_score,
                     support, lift, expected_discount_percent, category_mix, is_active)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    week_start.date(),
                    combo.combo_id,
                    combo.name,
                    str(combo.products),
                    combo.confidence_score,
                    combo.support,
                    combo.lift,
                    combo.expected_discount_percent,
                    str(combo.category_mix) if combo.category_mix else None,
                    1 if combo.is_active else 0
                ))
    
    def update_weekly_metrics(self, week_start: datetime):
        """
        Calculate and store weekly metrics summary.
        
        Args:
            week_start: Start date of the week
        """
        with sqlite3.connect(self.db_path) as conn:
            # Calculate metrics from stored rules and combos
            metrics = conn.execute("""
                SELECT 
                    COUNT(r.id) as total_rules,
                    COUNT(CASE WHEN r.confidence >= 0.9 THEN 1 END) as high_confidence_rules,
                    AVG(r.confidence) as avg_confidence,
                    AVG(r.support) as avg_support,
                    AVG(r.lift) as avg_lift,
                    COUNT(c.id) as total_combos,
                    COUNT(CASE WHEN c.is_active = 1 THEN 1 END) as active_combos,
                    AVG(c.expected_discount_percent) as avg_discount_percent
                FROM association_rules r
                LEFT JOIN combos c ON r.week_start_date = c.week_start_date
                WHERE r.week_start_date = ?
                GROUP BY r.week_start_date
            """, (week_start.date(),)).fetchone()
            
            if metrics:
                conn.execute("""
                    INSERT OR REPLACE INTO weekly_metrics 
                    (week_start_date, total_rules, high_confidence_rules, 
                     avg_confidence, avg_support, avg_lift, total_combos, 
                     active_combos, avg_discount_percent)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, metrics + (week_start.date(),))
    
    def get_week_over_week_analysis(self) -> pd.DataFrame:
        """Get week-over-week analysis using the SQL view."""
        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql_query("""
                SELECT * FROM week_over_week_rules 
                ORDER BY current_week DESC
            """, conn)
    
    def get_new_vs_returning_analysis(self) -> pd.DataFrame:
        """Get new vs returning associations analysis."""
        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql_query("""
                SELECT * FROM new_vs_returning_associations 
                ORDER BY week_start_date DESC
            """, conn)
    
    def get_top_rule_trends(self, min_weeks: int = 2) -> pd.DataFrame:
        """Get trending association rules analysis."""
        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql_query("""
                SELECT * FROM top_rule_trends 
                WHERE weeks_active >= ?
                ORDER BY avg_confidence DESC, weeks_active DESC
            """, conn, params=(min_weeks,))
    
    def get_category_insights(self, weeks_back: int = 4) -> pd.DataFrame:
        """Get category-based association insights."""
        cutoff_date = datetime.now() - timedelta(weeks=weeks_back)
        
        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql_query("""
                SELECT * FROM category_association_insights 
                WHERE week_start_date >= ?
                ORDER BY week_start_date DESC, avg_confidence DESC
            """, conn, params=(cutoff_date.date(),))
    
    def get_association_performance_metrics(self, weeks_back: int = 8) -> Dict[str, Any]:
        """
        Get comprehensive performance metrics for associations.
        
        Args:
            weeks_back: Number of weeks to analyze
            
        Returns:
            Dictionary with performance metrics and trends
        """
        cutoff_date = datetime.now() - timedelta(weeks=weeks_back)
        
        with sqlite3.connect(self.db_path) as conn:
            # Get basic metrics
            basic_metrics = conn.execute("""
                SELECT 
                    COUNT(DISTINCT week_start_date) as weeks_analyzed,
                    AVG(total_rules) as avg_rules_per_week,
                    AVG(high_confidence_rules) as avg_high_conf_per_week,
                    AVG(avg_confidence) as overall_avg_confidence,
                    AVG(avg_lift) as overall_avg_lift,
                    AVG(total_combos) as avg_combos_per_week,
                    AVG(avg_discount_percent) as avg_discount
                FROM weekly_metrics
                WHERE week_start_date >= ?
            """, (cutoff_date.date(),)).fetchone()
            
            # Get trend analysis
            trend_analysis = conn.execute("""
                SELECT 
                    COUNT(CASE WHEN rules_change > 0 THEN 1 END) as weeks_rules_increased,
                    COUNT(CASE WHEN confidence_change > 0 THEN 1 END) as weeks_confidence_increased,
                    AVG(rules_change_pct) as avg_rules_change_pct,
                    AVG(confidence_change_pct) as avg_confidence_change_pct,
                    COUNT(*) as total_week_comparisons
                FROM week_over_week_rules
                WHERE current_week >= ?
            """, (cutoff_date.date(),)).fetchone()
            
            # Get rule stability metrics
            stability_metrics = conn.execute("""
                SELECT 
                    COUNT(CASE WHEN trend_category = 'Stable' THEN 1 END) as stable_rules,
                    COUNT(CASE WHEN trend_category = 'Emerging' THEN 1 END) as emerging_rules,
                    COUNT(CASE WHEN trend_category = 'New' THEN 1 END) as new_rules,
                    COUNT(*) as total_unique_rules,
                    AVG(CASE WHEN trend_category = 'Stable' THEN avg_confidence END) as stable_avg_confidence
                FROM top_rule_trends
            """).fetchone()
            
            return {
                'analysis_period': {
                    'weeks_analyzed': basic_metrics[0] if basic_metrics else 0,
                    'start_date': cutoff_date.date().isoformat(),
                    'end_date': datetime.now().date().isoformat()
                },
                'weekly_averages': {
                    'rules_per_week': round(basic_metrics[1], 1) if basic_metrics and basic_metrics[1] else 0,
                    'high_conf_rules_per_week': round(basic_metrics[2], 1) if basic_metrics and basic_metrics[2] else 0,
                    'avg_confidence': round(basic_metrics[3], 3) if basic_metrics and basic_metrics[3] else 0,
                    'avg_lift': round(basic_metrics[4], 3) if basic_metrics and basic_metrics[4] else 0,
                    'combos_per_week': round(basic_metrics[5], 1) if basic_metrics and basic_metrics[5] else 0,
                    'avg_discount_percent': round(basic_metrics[6], 1) if basic_metrics and basic_metrics[6] else 0
                },
                'trends': {
                    'weeks_with_rule_growth': trend_analysis[0] if trend_analysis else 0,
                    'weeks_with_confidence_growth': trend_analysis[1] if trend_analysis else 0,
                    'avg_weekly_rules_change_pct': round(trend_analysis[2], 2) if trend_analysis and trend_analysis[2] else 0,
                    'avg_weekly_confidence_change_pct': round(trend_analysis[3], 2) if trend_analysis and trend_analysis[3] else 0,
                    'total_comparisons': trend_analysis[4] if trend_analysis else 0
                },
                'rule_stability': {
                    'stable_rules': stability_metrics[0] if stability_metrics else 0,
                    'emerging_rules': stability_metrics[1] if stability_metrics else 0,
                    'new_rules': stability_metrics[2] if stability_metrics else 0,
                    'total_unique_rules': stability_metrics[3] if stability_metrics else 0,
                    'stable_rule_avg_confidence': round(stability_metrics[4], 3) if stability_metrics and stability_metrics[4] else 0
                }
            }
    
    def export_analysis_to_pandas(self) -> Dict[str, pd.DataFrame]:
        """
        Export all analysis views to Pandas DataFrames.
        
        Returns:
            Dictionary mapping view names to DataFrames
        """
        return {
            'week_over_week': self.get_week_over_week_analysis(),
            'new_vs_returning': self.get_new_vs_returning_analysis(),
            'top_trends': self.get_top_rule_trends(),
            'category_insights': self.get_category_insights()
        }
    
    def generate_insights_report(self) -> Dict[str, Any]:
        """Generate comprehensive insights report from all data."""
        insights = {
            'performance_metrics': self.get_association_performance_metrics(),
            'top_trends': self.get_top_rule_trends(min_weeks=2).to_dict('records')[:10],
            'category_performance': self.get_category_insights().to_dict('records')[:15],
            'recent_changes': self.get_week_over_week_analysis().head(4).to_dict('records')
        }
        
        # Add summary insights
        perf = insights['performance_metrics']
        insights['summary'] = {
            'total_weeks_analyzed': perf['analysis_period']['weeks_analyzed'],
            'average_rules_per_week': perf['weekly_averages']['rules_per_week'],
            'trend_direction': 'improving' if perf['trends']['avg_weekly_confidence_change_pct'] > 0 else 'declining',
            'stability_score': round(
                (perf['rule_stability']['stable_rules'] / max(1, perf['rule_stability']['total_unique_rules'])) * 100, 1
            ),
            'generated_at': datetime.now().isoformat()
        }
        
        return insights 