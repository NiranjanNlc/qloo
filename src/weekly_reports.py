"""
Weekly KPI Reporting System

This module provides functionality for generating weekly KPI reports
with customizable Jinja2 templates and data aggregation.
"""

import os
import yaml
import json
import pandas as pd
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from pathlib import Path
import logging
from jinja2 import Environment, FileSystemLoader, Template

from models import Combo, Product

logger = logging.getLogger(__name__)


@dataclass
class WeeklyKPI:
    """Represents a weekly KPI metric."""

    metric_name: str
    current_value: float
    previous_value: Optional[float] = None
    target_value: Optional[float] = None
    unit: str = ""
    change_percent: Optional[float] = None
    is_improvement: Optional[bool] = None

    def __post_init__(self) -> None:
        """Calculate derived metrics after initialization."""
        if self.previous_value is not None and self.previous_value != 0:
            self.change_percent = (
                (self.current_value - self.previous_value) / abs(self.previous_value)
            ) * 100

            # Determine if change is improvement (depends on metric type)
            improvement_metrics = [
                "combos_generated",
                "high_confidence_rules",
                "avg_confidence",
                "avg_lift",
                "total_revenue_potential",
                "discount_efficiency",
            ]
            decline_metrics = ["avg_discount_percent", "processing_time"]

            if any(
                metric in self.metric_name.lower() for metric in improvement_metrics
            ):
                self.is_improvement = self.change_percent > 0
            elif any(metric in self.metric_name.lower() for metric in decline_metrics):
                self.is_improvement = self.change_percent < 0
            else:
                self.is_improvement = None


@dataclass
class WeeklyKPITable:
    """Represents the complete weekly KPI table structure."""

    week_start_date: datetime
    week_end_date: datetime
    report_generated_at: datetime
    kpis: List[WeeklyKPI]
    summary_stats: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "week_start_date": self.week_start_date.isoformat(),
            "week_end_date": self.week_end_date.isoformat(),
            "report_generated_at": self.report_generated_at.isoformat(),
            "kpis": [asdict(kpi) for kpi in self.kpis],
            "summary_stats": self.summary_stats,
        }

    def to_dataframe(self) -> pd.DataFrame:
        """Convert KPIs to a pandas DataFrame."""
        kpi_data = []
        for kpi in self.kpis:
            kpi_data.append(
                {
                    "metric_name": kpi.metric_name,
                    "current_value": kpi.current_value,
                    "previous_value": kpi.previous_value,
                    "target_value": kpi.target_value,
                    "unit": kpi.unit,
                    "change_percent": kpi.change_percent,
                    "is_improvement": kpi.is_improvement,
                }
            )
        return pd.DataFrame(kpi_data)


class WeeklyKPICalculator:
    """Calculates weekly KPIs from combo and association data."""

    def __init__(self) -> None:
        self.default_targets = {
            "combos_generated": 50,
            "high_confidence_rules": 20,
            "avg_confidence": 0.85,
            "avg_lift": 1.5,
            "avg_discount_percent": 15.0,
            "total_revenue_potential": 10000.0,
        }

    def calculate_weekly_kpis(
        self,
        combos: List[Combo],
        previous_combos: Optional[List[Combo]] = None,
        week_start: Optional[datetime] = None,
    ) -> WeeklyKPITable:
        """
        Calculate weekly KPIs from combo data.

        Args:
            combos: Current week's combo data
            previous_combos: Previous week's combo data for comparison
            week_start: Start date of the week

        Returns:
            WeeklyKPITable with calculated metrics
        """
        if week_start is None:
            week_start = datetime.now() - timedelta(days=7)

        week_end = week_start + timedelta(days=6)

        # Calculate current week KPIs
        current_kpis = self._calculate_combo_kpis(combos)

        # Calculate previous week KPIs if available
        previous_kpis = {}
        if previous_combos:
            previous_kpis = self._calculate_combo_kpis(previous_combos)

        # Build KPI objects with comparisons
        kpi_objects = []
        for metric_name, current_value in current_kpis.items():
            previous_value = previous_kpis.get(metric_name)
            target_value = self.default_targets.get(metric_name)

            # Determine unit based on metric
            unit = self._get_metric_unit(metric_name)

            kpi = WeeklyKPI(
                metric_name=metric_name,
                current_value=current_value,
                previous_value=previous_value,
                target_value=target_value,
                unit=unit,
            )
            kpi_objects.append(kpi)

        # Calculate summary statistics
        summary_stats = self._calculate_summary_stats(combos, kpi_objects)

        return WeeklyKPITable(
            week_start_date=week_start,
            week_end_date=week_end,
            report_generated_at=datetime.now(),
            kpis=kpi_objects,
            summary_stats=summary_stats,
        )

    def _calculate_combo_kpis(self, combos: List[Combo]) -> Dict[str, float]:
        """Calculate KPI metrics from combo list."""
        if not combos:
            return {
                "combos_generated": 0,
                "high_confidence_rules": 0,
                "avg_confidence": 0,
                "avg_lift": 0,
                "avg_support": 0,
                "avg_discount_percent": 0,
                "total_revenue_potential": 0,
            }

        # Basic counts
        total_combos = len(combos)
        high_conf_combos = len([c for c in combos if c.confidence_score >= 0.9])

        # Average metrics
        avg_confidence = sum(c.confidence_score for c in combos) / total_combos
        avg_lift = sum(c.lift for c in combos) / total_combos
        avg_support = sum(c.support for c in combos) / total_combos

        # Discount metrics
        discount_combos = [c for c in combos if c.expected_discount_percent is not None]
        avg_discount = 0.0
        if discount_combos:
            avg_discount = sum(
                c.expected_discount_percent for c in discount_combos if c.expected_discount_percent is not None
            ) / len(discount_combos)

        # Revenue potential (simplified calculation)
        revenue_potential = (
            total_combos * avg_confidence * 500
        )  # $500 per combo estimated

        return {
            "combos_generated": float(total_combos),
            "high_confidence_rules": float(high_conf_combos),
            "avg_confidence": round(avg_confidence, 3),
            "avg_lift": round(avg_lift, 3),
            "avg_support": round(avg_support, 4),
            "avg_discount_percent": round(avg_discount, 1),
            "total_revenue_potential": round(revenue_potential, 2),
        }

    def _get_metric_unit(self, metric_name: str) -> str:
        """Get appropriate unit for metric."""
        unit_mapping = {
            "combos_generated": "count",
            "high_confidence_rules": "count",
            "avg_confidence": "ratio",
            "avg_lift": "ratio",
            "avg_support": "ratio",
            "avg_discount_percent": "%",
            "total_revenue_potential": "$",
        }
        return unit_mapping.get(metric_name, "")

    def _calculate_summary_stats(
        self, combos: List[Combo], kpis: List[WeeklyKPI]
    ) -> Dict[str, Any]:
        """Calculate additional summary statistics."""
        improvements = [kpi for kpi in kpis if kpi.is_improvement is True]
        declines = [kpi for kpi in kpis if kpi.is_improvement is False]

        # Category distribution
        category_dist: dict[str, int] = {}
        for combo in combos:
            if combo.category_mix:
                for category in combo.category_mix:
                    category_dist[category] = category_dist.get(category, 0) + 1

        return {
            "total_metrics": len(kpis),
            "improving_metrics": len(improvements),
            "declining_metrics": len(declines),
            "stable_metrics": len(kpis) - len(improvements) - len(declines),
            "category_distribution": category_dist,
            "top_performing_combos": len(
                [c for c in combos if c.confidence_score >= 0.95]
            ),
            "active_combos": len([c for c in combos if c.is_active]),
        }


class WeeklyReportGenerator:
    """Generates formatted weekly reports using Jinja2 templates."""

    def __init__(self, templates_dir: str = "templates"):
        """
        Initialize report generator.

        Args:
            templates_dir: Directory containing Jinja2 templates
        """
        self.templates_dir = Path(templates_dir)
        self.templates_dir.mkdir(exist_ok=True)

        # Initialize Jinja2 environment
        self.jinja_env = Environment(
            loader=FileSystemLoader(str(self.templates_dir)), autoescape=True
        )

        # Create default templates if they don't exist
        self._create_default_templates()

    def generate_html_report(self, kpi_table: WeeklyKPITable) -> str:
        """Generate HTML report from KPI table."""
        template = self.jinja_env.get_template("weekly_report.html")
        return template.render(
            kpi_table=kpi_table,
            report_title=f"Weekly KPI Report - {kpi_table.week_start_date.strftime('%Y-%m-%d')}",
            current_date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )

    def generate_markdown_report(self, kpi_table: WeeklyKPITable) -> str:
        """Generate Markdown report from KPI table."""
        template = self.jinja_env.get_template("weekly_report.md")
        return template.render(
            kpi_table=kpi_table,
            report_title=f"Weekly KPI Report - {kpi_table.week_start_date.strftime('%Y-%m-%d')}",
            current_date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )

    def generate_json_report(self, kpi_table: WeeklyKPITable) -> str:
        """Generate JSON report from KPI table."""
        return json.dumps(kpi_table.to_dict(), indent=2)

    def _create_default_templates(self) -> None:
        """Create default Jinja2 templates."""
        # HTML template
        html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>{{ report_title }}</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .header { background: #f4f4f4; padding: 15px; border-radius: 5px; }
        .kpi-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin: 20px 0; }
        .kpi-card { border: 1px solid #ddd; padding: 15px; border-radius: 5px; }
        .improvement { color: green; }
        .decline { color: red; }
        .stable { color: #666; }
        .metric-value { font-size: 2em; font-weight: bold; }
        .metric-change { font-size: 0.9em; }
        table { width: 100%; border-collapse: collapse; margin: 20px 0; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
    </style>
</head>
<body>
    <div class="header">
        <h1>{{ report_title }}</h1>
        <p>Report Period: {{ kpi_table.week_start_date.strftime('%Y-%m-%d') }} to {{ kpi_table.week_end_date.strftime('%Y-%m-%d') }}</p>
        <p>Generated: {{ current_date }}</p>
    </div>

    <h2>Key Performance Indicators</h2>
    <div class="kpi-grid">
        {% for kpi in kpi_table.kpis %}
        <div class="kpi-card">
            <h3>{{ kpi.metric_name.replace('_', ' ').title() }}</h3>
            <div class="metric-value">{{ kpi.current_value }}{{ kpi.unit }}</div>
            {% if kpi.change_percent is not none %}
            <div class="metric-change {% if kpi.is_improvement %}improvement{% elif kpi.is_improvement == false %}decline{% else %}stable{% endif %}">
                {{ "↑" if kpi.is_improvement else "↓" if kpi.is_improvement == false else "→" }} 
                {{ "%.1f"|format(kpi.change_percent) }}% vs last week
            </div>
            {% endif %}
            {% if kpi.target_value %}
            <div class="target">Target: {{ kpi.target_value }}{{ kpi.unit }}</div>
            {% endif %}
        </div>
        {% endfor %}
    </div>

    <h2>Summary Statistics</h2>
    <table>
        <tr><th>Metric</th><th>Value</th></tr>
        {% for key, value in kpi_table.summary_stats.items() %}
        {% if key != 'category_distribution' %}
        <tr><td>{{ key.replace('_', ' ').title() }}</td><td>{{ value }}</td></tr>
        {% endif %}
        {% endfor %}
    </table>

    {% if kpi_table.summary_stats.category_distribution %}
    <h2>Category Distribution</h2>
    <table>
        <tr><th>Category</th><th>Combo Count</th></tr>
        {% for category, count in kpi_table.summary_stats.category_distribution.items() %}
        <tr><td>{{ category }}</td><td>{{ count }}</td></tr>
        {% endfor %}
    </table>
    {% endif %}
</body>
</html>
        """

        # Markdown template
        md_template = """
# {{ report_title }}

**Report Period:** {{ kpi_table.week_start_date.strftime('%Y-%m-%d') }} to {{ kpi_table.week_end_date.strftime('%Y-%m-%d') }}  
**Generated:** {{ current_date }}

## Key Performance Indicators

{% for kpi in kpi_table.kpis %}
### {{ kpi.metric_name.replace('_', ' ').title() }}
- **Current Value:** {{ kpi.current_value }}{{ kpi.unit }}
{% if kpi.change_percent is not none -%}
- **Change:** {{ "↑" if kpi.is_improvement else "↓" if kpi.is_improvement == false else "→" }} {{ "%.1f"|format(kpi.change_percent) }}% vs last week
{% endif -%}
{% if kpi.target_value -%}
- **Target:** {{ kpi.target_value }}{{ kpi.unit }}
{% endif %}

{% endfor %}

## Summary Statistics

{% for key, value in kpi_table.summary_stats.items() %}
{% if key != 'category_distribution' -%}
- **{{ key.replace('_', ' ').title() }}:** {{ value }}
{% endif -%}
{% endfor %}

{% if kpi_table.summary_stats.category_distribution %}
## Category Distribution

{% for category, count in kpi_table.summary_stats.category_distribution.items() -%}
- **{{ category }}:** {{ count }} combos
{% endfor -%}
{% endif %}
        """

        # Write templates to files
        (self.templates_dir / "weekly_report.html").write_text(html_template.strip())
        (self.templates_dir / "weekly_report.md").write_text(md_template.strip())


@dataclass
class WeeklyReport:
    """Main weekly report model with YAML configuration support."""

    config: Dict[str, Any]
    kpi_table: WeeklyKPITable
    metadata: Dict[str, Any]

    @classmethod
    def from_yaml_config(
        cls,
        config_path: str,
        combos: List[Combo],
        previous_combos: Optional[List[Combo]] = None,
    ) -> "WeeklyReport":
        """
        Create WeeklyReport from YAML configuration file.

        Args:
            config_path: Path to YAML configuration file
            combos: Current week's combo data
            previous_combos: Previous week's combo data for comparison

        Returns:
            WeeklyReport instance
        """
        # Load YAML configuration
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        # Extract week configuration
        week_config = config.get("week", {})
        week_start = None
        if "start_date" in week_config:
            week_start = datetime.fromisoformat(week_config["start_date"])

        # Initialize KPI calculator with custom targets if provided
        calculator = WeeklyKPICalculator()
        if "targets" in config:
            calculator.default_targets.update(config["targets"])

        # Calculate KPIs
        kpi_table = calculator.calculate_weekly_kpis(
            combos=combos, previous_combos=previous_combos, week_start=week_start
        )

        # Create metadata
        metadata = {
            "config_path": config_path,
            "created_at": datetime.now().isoformat(),
            "version": config.get("version", "1.0"),
            "author": config.get("author", "System"),
            "data_sources": config.get("data_sources", []),
            "total_combos_analyzed": len(combos),
            "has_comparison_data": previous_combos is not None,
        }

        return cls(config=config, kpi_table=kpi_table, metadata=metadata)

    def to_yaml(self, output_path: str) -> None:
        """Export report configuration and results to YAML."""
        output_data = {
            "report_metadata": self.metadata,
            "configuration": self.config,
            "kpi_results": self.kpi_table.to_dict(),
            "exported_at": datetime.now().isoformat(),
        }

        with open(output_path, "w") as f:
            yaml.dump(output_data, f, default_flow_style=False, indent=2)

    def generate_report(
        self, format_type: str = "html", output_dir: str = "reports"
    ) -> str:
        """
        Generate formatted report.

        Args:
            format_type: Report format ('html', 'markdown', 'json', 'yaml')
            output_dir: Output directory for the report

        Returns:
            Path to generated report file
        """
        output_path_dir = Path(output_dir)
        output_path_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename with timestamp
        week_str = self.kpi_table.week_start_date.strftime("%Y-%m-%d")
        timestamp = datetime.now().strftime("%H%M%S")
        filename = f"weekly_report_{week_str}_{timestamp}.{format_type}"
        output_path = output_path_dir / filename

        # Generate report content
        generator = WeeklyReportGenerator()

        if format_type == "html":
            content = generator.generate_html_report(self.kpi_table)
        elif format_type == "markdown":
            content = generator.generate_markdown_report(self.kpi_table)
        elif format_type == "json":
            content = generator.generate_json_report(self.kpi_table)
        elif format_type == "yaml":
            self.to_yaml(str(output_path))
            return str(output_path)
        else:
            raise ValueError(f"Unsupported format type: {format_type}")

        # Write content to file
        output_path.write_text(content, encoding="utf-8")
        return str(output_path)


def create_default_config(output_path: str = "weekly_report_config.yaml") -> str:
    """Create a default YAML configuration file for weekly reports."""
    default_config = {
        "version": "1.0",
        "author": "Data Engineering Team",
        "description": "Weekly KPI report configuration for supermarket optimization",
        "week": {
            "start_date": None,  # Will be calculated if None
            "include_previous_comparison": True,
            "business_days_only": False,
        },
        "targets": {
            "combos_generated": 50,
            "high_confidence_rules": 20,
            "avg_confidence": 0.85,
            "avg_lift": 1.5,
            "avg_discount_percent": 15.0,
            "total_revenue_potential": 10000.0,
        },
        "data_sources": ["association_rules", "combo_generator", "price_api"],
        "output": {
            "formats": ["html", "json"],
            "directory": "reports",
            "include_charts": True,
            "include_raw_data": False,
        },
        "notifications": {
            "enabled": False,
            "email_recipients": [],
            "slack_webhook": None,
            "alert_thresholds": {
                "declining_metrics_threshold": 3,
                "target_miss_threshold": 0.8,
            },
        },
    }

    with open(output_path, "w") as f:
        yaml.dump(default_config, f, default_flow_style=False, indent=2)

    print(f"Default configuration created at: {output_path}")
    return output_path


# CLI Command functionality
def parse_week_date(week_str: str) -> datetime:
    """Parse week date string in YYYY-MM-DD format to datetime."""
    try:
        return datetime.strptime(week_str, "%Y-%m-%d")
    except ValueError:
        raise ValueError(f"Invalid week date format: {week_str}. Expected YYYY-MM-DD")


def generate_weekly_report_cli(
    week: str, config_path: Optional[str] = None, output_format: str = "html"
) -> str:
    """
    CLI command to generate weekly report.

    Args:
        week: Week start date in YYYY-MM-DD format
        config_path: Path to YAML configuration file
        output_format: Output format (html, markdown, json, yaml)
    """
    import sys
    import os

    # Add src to path for imports
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, current_dir)

    try:
        from models import ComboGenerator, Product
        from association_engine import AprioriAssociationEngine

        # Parse week date
        week_start = parse_week_date(week)
        print(
            f"Generating weekly report for week starting: {week_start.strftime('%Y-%m-%d')}"
        )

        # Create default config if none provided
        if config_path is None:
            config_path = create_default_config()
        
        # Ensure config_path is not None before passing to from_yaml_config
        assert config_path is not None, "config_path should not be None after default creation"

        # TODO: In a real implementation, load actual data from database
        # For now, generate mock data for demonstration
        print("Loading combo data...")

        # Mock combo data (in real implementation, this would come from database)
        mock_combos = []
        for i in range(25):
            from models import Combo

            combo = Combo(
                combo_id=f"weekly_combo_{i}",
                name=f"Weekly Combo {i+1}",
                products=[i * 2 + 1, i * 2 + 2, i * 2 + 3],
                confidence_score=0.8 + (i % 3) * 0.05,
                support=0.01 + (i % 5) * 0.005,
                lift=1.2 + (i % 4) * 0.2,
                expected_discount_percent=10.0 + (i % 3) * 5.0,
                category_mix=["Dairy", "Produce", "Meat"][i % 3 : i % 3 + 2],
            )
            mock_combos.append(combo)

        # Generate previous week data for comparison
        previous_combos = []
        for i in range(20):  # Fewer combos in previous week
            combo = Combo(
                combo_id=f"prev_combo_{i}",
                name=f"Previous Combo {i+1}",
                products=[i * 2 + 1, i * 2 + 2],
                confidence_score=0.75 + (i % 3) * 0.05,
                support=0.008 + (i % 5) * 0.004,
                lift=1.1 + (i % 4) * 0.15,
                expected_discount_percent=12.0 + (i % 3) * 4.0,
                category_mix=["Dairy", "Produce"][i % 2 : i % 2 + 1],
            )
            previous_combos.append(combo)

        print(
            f"Loaded {len(mock_combos)} current combos and {len(previous_combos)} previous combos"
        )

        # Create weekly report
        report = WeeklyReport.from_yaml_config(
            config_path=config_path, combos=mock_combos, previous_combos=previous_combos
        )

        # Update week start date in config
        report.config["week"]["start_date"] = week_start.isoformat()

        # Generate report
        print(f"Generating {output_format} report...")
        output_path = report.generate_report(format_type=output_format)

        print(f"Report generated successfully: {output_path}")

        # Print summary
        kpi_table = report.kpi_table
        print("\nReport Summary:")
        print(
            f"  Week: {kpi_table.week_start_date.strftime('%Y-%m-%d')} to {kpi_table.week_end_date.strftime('%Y-%m-%d')}"
        )
        print(f"  Total KPIs: {len(kpi_table.kpis)}")
        print(f"  Improving metrics: {kpi_table.summary_stats['improving_metrics']}")
        print(f"  Declining metrics: {kpi_table.summary_stats['declining_metrics']}")

        return output_path

    except Exception as e:
        print(f"Error generating weekly report: {e}")
        raise


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate weekly KPI report")
    parser.add_argument("--week", required=True, help="Week start date (YYYY-MM-DD)")
    parser.add_argument("--config", help="Path to YAML configuration file")
    parser.add_argument(
        "--format",
        default="html",
        choices=["html", "markdown", "json", "yaml"],
        help="Output format (default: html)",
    )

    args = parser.parse_args()

    generate_weekly_report_cli(
        week=args.week, config_path=args.config, output_format=args.format
    )
