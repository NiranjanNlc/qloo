"""
Weekly Report Generation DAG

This DAG generates weekly KPI reports every Friday at 02:00 AM.
It includes data processing, report generation, and notification capabilities.
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.operators.bash_operator import BashOperator
from airflow.operators.dummy_operator import DummyOperator
from airflow.operators.email_operator import EmailOperator
from airflow.hooks.http_hook import HttpHook
from airflow.models import Variable
from airflow.utils.dates import days_ago
import pandas as pd
import logging
import sys
import os
import json
from pathlib import Path

# Add src to path for imports
sys.path.append("/opt/airflow/dags/src")

try:
    from weekly_reports import WeeklyReport, WeeklyKPICalculator, WeeklyReportGenerator
    from models import Combo, ComboGenerator
    from association_engine import AprioriAssociationEngine
    from aggregation_views import AssociationAggregator
except ImportError as e:
    logging.error(f"Import error in DAG: {e}")

# DAG configuration
DAG_ID = "weekly_report_generation"
DEFAULT_ARGS = {
    "owner": "data-engineering-team",
    "depends_on_past": False,
    "start_date": datetime(2024, 1, 1),
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "catchup": False,
}

# DAG schedule: Every Friday at 02:00 AM
SCHEDULE_INTERVAL = "0 2 * * 5"  # Cron: minute hour day_of_month month day_of_week

# Create DAG
dag = DAG(
    DAG_ID,
    default_args=DEFAULT_ARGS,
    description="Generate weekly KPI reports and load to reports directory",
    schedule_interval=SCHEDULE_INTERVAL,
    max_active_runs=1,  # Prevent overlapping runs
    tags=["reporting", "weekly", "kpis"],
)


def extract_transaction_data(**context):
    """
    Extract transaction data for the past week.
    This function is idempotent and can be safely re-run.
    """
    execution_date = context["execution_date"]
    week_start = execution_date - timedelta(days=execution_date.weekday())
    week_end = week_start + timedelta(days=6)

    logging.info(
        f"Extracting transaction data for week: {week_start.date()} to {week_end.date()}"
    )

    try:
        # In a real implementation, this would query the actual database
        # For now, generate mock data that's deterministic based on the week

        data_dir = Path("/opt/airflow/dags/data")
        data_dir.mkdir(exist_ok=True)

        # Create mock transaction data for this week
        import random

        random.seed(int(week_start.timestamp()))  # Deterministic seed

        transactions = []
        for day in range(7):
            current_day = week_start + timedelta(days=day)
            daily_transactions = random.randint(100, 200)

            for transaction_id in range(daily_transactions):
                transaction = {
                    "transaction_id": f"{current_day.strftime('%Y%m%d')}_{transaction_id}",
                    "date": current_day.date().isoformat(),
                    "products": random.sample(range(1, 60), random.randint(2, 8)),
                    "total_amount": round(random.uniform(10.0, 150.0), 2),
                }
                transactions.append(transaction)

        # Save extracted data
        output_file = (
            data_dir / f"transactions_week_{week_start.strftime('%Y%m%d')}.json"
        )
        with open(output_file, "w") as f:
            json.dump(transactions, f, indent=2)

        logging.info(f"Extracted {len(transactions)} transactions to {output_file}")

        # Store metadata for downstream tasks
        context["task_instance"].xcom_push(
            key="extraction_metadata",
            value={
                "week_start": week_start.isoformat(),
                "week_end": week_end.isoformat(),
                "transaction_count": len(transactions),
                "output_file": str(output_file),
            },
        )

        return str(output_file)

    except Exception as e:
        logging.error(f"Failed to extract transaction data: {e}")
        raise


def generate_association_rules(**context):
    """
    Generate association rules from transaction data.
    This function is idempotent and deterministic.
    """
    # Get metadata from previous task
    extraction_metadata = context["task_instance"].xcom_pull(
        task_ids="extract_transaction_data", key="extraction_metadata"
    )

    logging.info(
        f"Generating association rules for {extraction_metadata['transaction_count']} transactions"
    )

    try:
        # Load transaction data
        with open(extraction_metadata["output_file"], "r") as f:
            transactions = json.load(f)

        # Convert to format expected by association engine
        transaction_lists = [t["products"] for t in transactions]

        # Initialize and train association engine
        engine = AprioriAssociationEngine()
        # Convert to DataFrame format expected by the engine
        transaction_df = pd.DataFrame(
            {
                f"product_{i}": [1 if i in t else 0 for t in transaction_lists]
                for i in range(1, 60)
            }
        )

        engine.train(transaction_df)

        # Generate association rules
        rules = engine.association_rules

        # Save rules
        week_start = datetime.fromisoformat(extraction_metadata["week_start"])
        rules_file = (
            Path("/opt/airflow/dags/data")
            / f"association_rules_week_{week_start.strftime('%Y%m%d')}.json"
        )

        # Convert rules to serializable format
        serializable_rules = []
        for rule in rules:
            serializable_rules.append(
                {
                    "antecedent": rule["antecedent"],
                    "consequent": rule["consequent"],
                    "confidence": float(rule["confidence"]),
                    "support": float(rule["support"]),
                    "lift": float(rule["lift"]),
                }
            )

        with open(rules_file, "w") as f:
            json.dump(serializable_rules, f, indent=2)

        logging.info(f"Generated {len(serializable_rules)} association rules")

        # Store metadata
        context["task_instance"].xcom_push(
            key="rules_metadata",
            value={
                "rules_count": len(serializable_rules),
                "rules_file": str(rules_file),
                "high_confidence_rules": len(
                    [r for r in serializable_rules if r["confidence"] > 0.8]
                ),
            },
        )

        return str(rules_file)

    except Exception as e:
        logging.error(f"Failed to generate association rules: {e}")
        raise


def generate_combo_offers(**context):
    """
    Generate product combo offers based on association rules.
    This function is idempotent.
    """
    # Get metadata from previous tasks
    rules_metadata = context["task_instance"].xcom_pull(
        task_ids="generate_association_rules", key="rules_metadata"
    )

    extraction_metadata = context["task_instance"].xcom_pull(
        task_ids="extract_transaction_data", key="extraction_metadata"
    )

    logging.info(f"Generating combo offers from {rules_metadata['rules_count']} rules")

    try:
        # Load association rules
        with open(rules_metadata["rules_file"], "r") as f:
            rules = json.load(f)

        # Generate combos using ComboGenerator
        generator = ComboGenerator(min_confidence=0.6, min_support=0.01)

        # Create mock products for combo generation
        mock_products = [
            type(
                "Product",
                (),
                {"id": i, "name": f"Product {i}", "category": f"Category {i//10}"},
            )()
            for i in range(1, 60)
        ]

        combos = generator.generate_weekly_combos(rules, mock_products)

        # Save combos
        week_start = datetime.fromisoformat(extraction_metadata["week_start"])
        combos_file = (
            Path("/opt/airflow/dags/data")
            / f"combos_week_{week_start.strftime('%Y%m%d')}.json"
        )

        # Convert combos to serializable format
        serializable_combos = []
        for combo in combos:
            serializable_combos.append(
                {
                    "combo_id": combo.combo_id,
                    "name": combo.name,
                    "products": combo.products,
                    "confidence_score": combo.confidence_score,
                    "support": combo.support,
                    "lift": combo.lift,
                    "expected_discount_percent": combo.expected_discount_percent,
                    "category_mix": combo.category_mix,
                    "is_active": combo.is_active,
                    "created_at": combo.created_at.isoformat(),
                }
            )

        with open(combos_file, "w") as f:
            json.dump(serializable_combos, f, indent=2)

        logging.info(f"Generated {len(serializable_combos)} combo offers")

        # Store metadata
        context["task_instance"].xcom_push(
            key="combos_metadata",
            value={
                "combos_count": len(serializable_combos),
                "combos_file": str(combos_file),
                "high_confidence_combos": len(
                    [c for c in serializable_combos if c["confidence_score"] > 0.8]
                ),
            },
        )

        return str(combos_file)

    except Exception as e:
        logging.error(f"Failed to generate combo offers: {e}")
        raise


def generate_weekly_report(**context):
    """
    Generate the weekly KPI report in multiple formats.
    This is the main report generation task.
    """
    # Get metadata from all previous tasks
    extraction_metadata = context["task_instance"].xcom_pull(
        task_ids="extract_transaction_data", key="extraction_metadata"
    )

    rules_metadata = context["task_instance"].xcom_pull(
        task_ids="generate_association_rules", key="rules_metadata"
    )

    combos_metadata = context["task_instance"].xcom_pull(
        task_ids="generate_combo_offers", key="combos_metadata"
    )

    execution_date = context["execution_date"]
    week_start = execution_date - timedelta(days=execution_date.weekday())

    logging.info(f"Generating weekly report for week starting {week_start.date()}")

    try:
        # Load combo data
        with open(combos_metadata["combos_file"], "r") as f:
            combo_data = json.load(f)

        # Convert back to Combo objects
        combos = []
        for combo_dict in combo_data:
            combo = type("MockCombo", (), combo_dict)()
            combo.created_at = datetime.fromisoformat(combo_dict["created_at"])
            combos.append(combo)

        # Calculate KPIs
        calculator = WeeklyKPICalculator()
        kpi_table = calculator.calculate_weekly_kpis(
            combos=combos,
            previous_combos=None,  # Could load from previous week
            week_start=week_start,
        )

        # Generate reports in multiple formats
        generator = WeeklyReportGenerator()
        reports_dir = Path("/opt/airflow/dags/reports")
        reports_dir.mkdir(exist_ok=True)

        # Generate HTML report with inline heatmaps
        html_content = generator.generate_html_report(kpi_table)

        # Add base64 encoded heatmaps (mock implementation)
        import base64
        import io
        import matplotlib.pyplot as plt
        import seaborn as sns

        # Create a sample heatmap
        fig, ax = plt.subplots(figsize=(8, 6))
        data = [
            [combo.confidence_score for combo in combos[:10]],
            [combo.lift for combo in combos[:10]],
        ]
        sns.heatmap(data, annot=True, cmap="viridis", ax=ax)
        ax.set_title("Combo Performance Heatmap")

        # Convert to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format="png", bbox_inches="tight", dpi=100)
        buffer.seek(0)
        heatmap_base64 = base64.b64encode(buffer.read()).decode()
        plt.close()

        # Insert heatmap into HTML
        heatmap_html = f'<img src="data:image/png;base64,{heatmap_base64}" alt="Performance Heatmap" style="max-width: 100%; height: auto;">'
        html_content = html_content.replace("</body>", f"{heatmap_html}</body>")

        # Save reports
        timestamp = execution_date.strftime("%Y%m%d_%H%M%S")
        week_str = week_start.strftime("%Y%m%d")

        html_file = reports_dir / f"weekly_report_{week_str}_{timestamp}.html"
        json_file = reports_dir / f"weekly_report_{week_str}_{timestamp}.json"

        # Save HTML report
        with open(html_file, "w", encoding="utf-8") as f:
            f.write(html_content)

        # Save JSON report
        json_content = generator.generate_json_report(kpi_table)
        with open(json_file, "w") as f:
            f.write(json_content)

        logging.info(f"Generated reports: {html_file}, {json_file}")

        # Store report metadata
        report_metadata = {
            "html_file": str(html_file),
            "json_file": str(json_file),
            "week_start": week_start.isoformat(),
            "total_combos": len(combos),
            "total_transactions": extraction_metadata["transaction_count"],
            "total_rules": rules_metadata["rules_count"],
            "generation_time": datetime.now().isoformat(),
        }

        context["task_instance"].xcom_push(key="report_metadata", value=report_metadata)

        return report_metadata

    except Exception as e:
        logging.error(f"Failed to generate weekly report: {e}")
        raise


def send_slack_notification(**context):
    """
    Send Slack notification about report generation success or failure.
    """
    # Get Slack webhook URL from Airflow Variables
    slack_webhook_url = Variable.get("slack_webhook_url", default_var=None)

    if not slack_webhook_url:
        logging.warning("Slack webhook URL not configured, skipping notification")
        return

    task_instance = context["task_instance"]
    execution_date = context["execution_date"]

    # Determine if this is a success or failure notification
    is_success = task_instance.current_state() != "failed"

    try:
        if is_success:
            # Get report metadata
            report_metadata = context["task_instance"].xcom_pull(
                task_ids="generate_weekly_report", key="report_metadata"
            )

            message = {
                "channel": "#data-engineering",
                "username": "Airflow Bot",
                "icon_emoji": ":bar_chart:",
                "attachments": [
                    {
                        "color": "good",
                        "title": "Weekly Report Generated Successfully",
                        "fields": [
                            {
                                "title": "Week",
                                "value": datetime.fromisoformat(
                                    report_metadata["week_start"]
                                ).strftime("%Y-%m-%d"),
                                "short": True,
                            },
                            {
                                "title": "Combos Generated",
                                "value": str(report_metadata["total_combos"]),
                                "short": True,
                            },
                            {
                                "title": "Transactions Processed",
                                "value": str(report_metadata["total_transactions"]),
                                "short": True,
                            },
                            {
                                "title": "Association Rules",
                                "value": str(report_metadata["total_rules"]),
                                "short": True,
                            },
                        ],
                        "footer": "Qloo Supermarket Optimizer",
                        "ts": int(execution_date.timestamp()),
                    }
                ],
            }
        else:
            message = {
                "channel": "#data-engineering",
                "username": "Airflow Bot",
                "icon_emoji": ":rotating_light:",
                "attachments": [
                    {
                        "color": "danger",
                        "title": "Weekly Report Generation Failed",
                        "text": f"DAG {DAG_ID} failed at {execution_date}",
                        "footer": "Qloo Supermarket Optimizer",
                        "ts": int(execution_date.timestamp()),
                    }
                ],
            }

        # Send to Slack
        http_hook = HttpHook(method="POST", http_conn_id="slack_webhook")
        response = http_hook.run(
            endpoint="", json=message, headers={"Content-Type": "application/json"}
        )

        logging.info(f"Slack notification sent successfully: {response.status_code}")

    except Exception as e:
        logging.error(f"Failed to send Slack notification: {e}")
        # Don't fail the DAG if Slack notification fails


# Define tasks
start_task = DummyOperator(task_id="start", dag=dag)

extract_data_task = PythonOperator(
    task_id="extract_transaction_data",
    python_callable=extract_transaction_data,
    dag=dag,
    doc_md="""
    ## Extract Transaction Data
    
    This task extracts transaction data for the past week from the database.
    The task is idempotent and can be safely re-run.
    
    **Outputs:**
    - Transaction data in JSON format
    - Metadata about extraction process
    """,
)

generate_rules_task = PythonOperator(
    task_id="generate_association_rules",
    python_callable=generate_association_rules,
    dag=dag,
    doc_md="""
    ## Generate Association Rules
    
    This task uses the Apriori algorithm to generate association rules
    from the extracted transaction data.
    
    **Inputs:**
    - Transaction data from previous task
    
    **Outputs:**
    - Association rules in JSON format
    - Rule statistics and metadata
    """,
)

generate_combos_task = PythonOperator(
    task_id="generate_combo_offers",
    python_callable=generate_combo_offers,
    dag=dag,
    doc_md="""
    ## Generate Combo Offers
    
    This task generates product combination offers based on the
    association rules with confidence and support thresholds.
    
    **Inputs:**
    - Association rules from previous task
    
    **Outputs:**
    - Product combo offers in JSON format
    - Combo statistics and metadata
    """,
)

generate_report_task = PythonOperator(
    task_id="generate_weekly_report",
    python_callable=generate_weekly_report,
    dag=dag,
    doc_md="""
    ## Generate Weekly Report
    
    This task generates the final weekly KPI report in HTML and JSON formats.
    The HTML report includes inline base64-encoded heatmaps for visualization.
    
    **Inputs:**
    - Transaction data, rules, and combos from previous tasks
    
    **Outputs:**
    - HTML report with embedded visualizations
    - JSON report for API consumption
    """,
)

notify_success_task = PythonOperator(
    task_id="notify_success",
    python_callable=send_slack_notification,
    dag=dag,
    trigger_rule="all_success",
)

notify_failure_task = PythonOperator(
    task_id="notify_failure",
    python_callable=send_slack_notification,
    dag=dag,
    trigger_rule="one_failed",
)

end_task = DummyOperator(task_id="end", dag=dag, trigger_rule="none_failed_or_skipped")

# Set task dependencies
(
    start_task
    >> extract_data_task
    >> generate_rules_task
    >> generate_combos_task
    >> generate_report_task
)

# Notification tasks
generate_report_task >> notify_success_task >> end_task
[
    extract_data_task,
    generate_rules_task,
    generate_combos_task,
    generate_report_task,
] >> notify_failure_task
