"""
Unit Tests for Weekly Report DAG

This module contains tests to ensure the DAG is idempotent and functions correctly.
"""

import pytest
import json
import tempfile
from unittest.mock import patch, MagicMock, mock_open
from datetime import datetime, timedelta
from pathlib import Path
import sys
import os

# Add DAG directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "dags"))

try:
    from weekly_report_dag import (
        extract_transaction_data,
        generate_association_rules,
        generate_combo_offers,
        generate_weekly_report,
        send_slack_notification,
    )
except ImportError as e:
    pytest.skip(f"DAG imports not available: {e}", allow_module_level=True)


class TestWeeklyReportDAG:
    """Test suite for the weekly report DAG."""

    @pytest.fixture
    def mock_context(self):
        """Create a mock Airflow context."""
        execution_date = datetime(2024, 1, 5)  # A Friday
        task_instance = MagicMock()

        context = {
            "execution_date": execution_date,
            "task_instance": task_instance,
            "dag": MagicMock(),
            "ds": execution_date.strftime("%Y-%m-%d"),
            "ts": execution_date.isoformat(),
        }

        # Set up xcom_pull/push methods
        task_instance.xcom_pull.return_value = None
        task_instance.xcom_push.return_value = None
        task_instance.current_state.return_value = "success"

        return context

    @pytest.fixture
    def temp_data_dir(self):
        """Create a temporary directory for test data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    def test_extract_transaction_data_idempotency(self, mock_context, temp_data_dir):
        """Test that extract_transaction_data is idempotent."""

        with patch("weekly_report_dag.Path") as mock_path:
            mock_path.return_value = temp_data_dir
            mock_path.side_effect = lambda x: (
                temp_data_dir if "/opt/airflow/dags/data" in str(x) else Path(x)
            )

            # First run
            result1 = extract_transaction_data(**mock_context)

            # Verify file was created
            assert temp_data_dir.exists()

            # Second run with same context
            result2 = extract_transaction_data(**mock_context)

            # Results should be the same (idempotent)
            assert result1 == result2

            # Verify xcom_push was called
            assert mock_context["task_instance"].xcom_push.called

    def test_extract_transaction_data_deterministic(self, mock_context, temp_data_dir):
        """Test that extract_transaction_data produces deterministic results."""

        with patch("weekly_report_dag.Path") as mock_path:
            mock_path.return_value = temp_data_dir
            mock_path.side_effect = lambda x: (
                temp_data_dir if "/opt/airflow/dags/data" in str(x) else Path(x)
            )

            # First run
            result1 = extract_transaction_data(**mock_context)

            # Load generated data
            with open(result1, "r") as f:
                data1 = json.load(f)

            # Clear directory and run again
            for file in temp_data_dir.glob("*"):
                file.unlink()

            # Second run with same execution date
            result2 = extract_transaction_data(**mock_context)

            with open(result2, "r") as f:
                data2 = json.load(f)

            # Data should be identical (deterministic)
            assert data1 == data2

    def test_generate_association_rules_with_valid_data(
        self, mock_context, temp_data_dir
    ):
        """Test association rules generation with valid transaction data."""

        # Mock previous task output
        sample_transactions = [
            {
                "transaction_id": "20240105_1",
                "date": "2024-01-05",
                "products": [1, 2, 3],
                "total_amount": 25.50,
            },
            {
                "transaction_id": "20240105_2",
                "date": "2024-01-05",
                "products": [1, 4, 5],
                "total_amount": 35.75,
            },
            {
                "transaction_id": "20240105_3",
                "date": "2024-01-05",
                "products": [2, 3, 6],
                "total_amount": 45.25,
            },
        ]

        # Create mock transaction file
        transaction_file = temp_data_dir / "transactions_test.json"
        with open(transaction_file, "w") as f:
            json.dump(sample_transactions, f)

        # Mock xcom_pull to return our test data
        mock_context["task_instance"].xcom_pull.return_value = {
            "week_start": "2024-01-01T00:00:00",
            "week_end": "2024-01-07T23:59:59",
            "transaction_count": len(sample_transactions),
            "output_file": str(transaction_file),
        }

        with patch("weekly_report_dag.Path") as mock_path:
            mock_path.return_value = temp_data_dir
            mock_path.side_effect = lambda x: (
                temp_data_dir if "/opt/airflow/dags/data" in str(x) else Path(x)
            )

            # Test the function
            result = generate_association_rules(**mock_context)

            # Verify rules file was created
            assert Path(result).exists()

            # Load and validate rules
            with open(result, "r") as f:
                rules = json.load(f)

            assert isinstance(rules, list)

            # Verify xcom_push was called with metadata
            mock_context["task_instance"].xcom_push.assert_called()

    def test_generate_combo_offers_idempotency(self, mock_context, temp_data_dir):
        """Test that combo generation is idempotent."""

        # Mock inputs from previous tasks
        sample_rules = [
            {
                "antecedent": [1],
                "consequent": [2],
                "confidence": 0.8,
                "support": 0.1,
                "lift": 1.5,
            },
            {
                "antecedent": [2],
                "consequent": [3],
                "confidence": 0.7,
                "support": 0.15,
                "lift": 1.2,
            },
        ]

        rules_file = temp_data_dir / "rules_test.json"
        with open(rules_file, "w") as f:
            json.dump(sample_rules, f)

        # Mock xcom pulls
        def mock_xcom_pull(task_ids, key):
            if task_ids == "generate_association_rules":
                return {
                    "rules_count": len(sample_rules),
                    "rules_file": str(rules_file),
                    "high_confidence_rules": 1,
                }
            elif task_ids == "extract_transaction_data":
                return {
                    "week_start": "2024-01-01T00:00:00",
                    "week_end": "2024-01-07T23:59:59",
                    "transaction_count": 100,
                }

        mock_context["task_instance"].xcom_pull.side_effect = mock_xcom_pull

        with patch("weekly_report_dag.Path") as mock_path:
            mock_path.return_value = temp_data_dir
            mock_path.side_effect = lambda x: (
                temp_data_dir if "/opt/airflow/dags/data" in str(x) else Path(x)
            )

            # First run
            result1 = generate_combo_offers(**mock_context)

            # Load first result
            with open(result1, "r") as f:
                combos1 = json.load(f)

            # Second run
            result2 = generate_combo_offers(**mock_context)

            # Load second result
            with open(result2, "r") as f:
                combos2 = json.load(f)

            # Results should be identical (idempotent)
            assert combos1 == combos2

    def test_generate_weekly_report_with_valid_inputs(
        self, mock_context, temp_data_dir
    ):
        """Test weekly report generation with valid inputs."""

        # Mock combo data
        sample_combos = [
            {
                "combo_id": "combo_001",
                "name": "Test Combo 1",
                "products": [1, 2, 3],
                "confidence_score": 0.8,
                "support": 0.1,
                "lift": 1.5,
                "expected_discount_percent": 10.0,
                "category_mix": ["Category A", "Category B"],
                "is_active": True,
                "created_at": "2024-01-01T00:00:00",
            }
        ]

        combos_file = temp_data_dir / "combos_test.json"
        with open(combos_file, "w") as f:
            json.dump(sample_combos, f)

        # Mock xcom pulls
        def mock_xcom_pull(task_ids, key):
            if task_ids == "generate_combo_offers":
                return {
                    "combos_count": len(sample_combos),
                    "combos_file": str(combos_file),
                    "high_confidence_combos": 1,
                }
            elif task_ids == "extract_transaction_data":
                return {"week_start": "2024-01-01T00:00:00", "transaction_count": 100}
            elif task_ids == "generate_association_rules":
                return {"rules_count": 10}

        mock_context["task_instance"].xcom_pull.side_effect = mock_xcom_pull

        with patch("weekly_report_dag.Path") as mock_path:
            mock_path.return_value = temp_data_dir
            mock_path.side_effect = lambda x: (
                temp_data_dir if "/opt/airflow/dags/data" in str(x) else Path(x)
            )

            with patch("weekly_report_dag.plt") as mock_plt:
                mock_plt.subplots.return_value = (MagicMock(), MagicMock())
                mock_plt.savefig.return_value = None
                mock_plt.close.return_value = None

                # Test the function
                result = generate_weekly_report(**mock_context)

                # Verify result structure
                assert isinstance(result, dict)
                assert "html_file" in result
                assert "json_file" in result
                assert "week_start" in result

                # Verify files were created
                assert Path(result["html_file"]).exists()
                assert Path(result["json_file"]).exists()

    def test_send_slack_notification_success(self, mock_context):
        """Test Slack notification for successful runs."""

        with patch("weekly_report_dag.Variable") as mock_variable:
            mock_variable.get.return_value = "https://hooks.slack.com/test"

            with patch("weekly_report_dag.HttpHook") as mock_http:
                mock_hook = MagicMock()
                mock_response = MagicMock()
                mock_response.status_code = 200
                mock_hook.run.return_value = mock_response
                mock_http.return_value = mock_hook

                # Mock successful report metadata
                mock_context["task_instance"].xcom_pull.return_value = {
                    "week_start": "2024-01-01T00:00:00",
                    "total_combos": 25,
                    "total_transactions": 100,
                    "total_rules": 15,
                }

                # Test the function
                send_slack_notification(**mock_context)

                # Verify HTTP hook was called
                mock_hook.run.assert_called_once()

                # Verify message structure
                call_args = mock_hook.run.call_args
                message = call_args[1]["json"]
                assert "attachments" in message
                assert message["attachments"][0]["color"] == "good"

    def test_send_slack_notification_failure(self, mock_context):
        """Test Slack notification for failed runs."""

        # Mock failed task state
        mock_context["task_instance"].current_state.return_value = "failed"

        with patch("weekly_report_dag.Variable") as mock_variable:
            mock_variable.get.return_value = "https://hooks.slack.com/test"

            with patch("weekly_report_dag.HttpHook") as mock_http:
                mock_hook = MagicMock()
                mock_response = MagicMock()
                mock_response.status_code = 200
                mock_hook.run.return_value = mock_response
                mock_http.return_value = mock_hook

                # Test the function
                send_slack_notification(**mock_context)

                # Verify HTTP hook was called
                mock_hook.run.assert_called_once()

                # Verify failure message structure
                call_args = mock_hook.run.call_args
                message = call_args[1]["json"]
                assert "attachments" in message
                assert message["attachments"][0]["color"] == "danger"

    def test_dag_structure_and_dependencies(self):
        """Test that the DAG has correct structure and dependencies."""

        try:
            from weekly_report_dag import dag

            # Verify DAG properties
            assert dag.dag_id == "weekly_report_generation"
            assert dag.schedule_interval == "0 2 * * 5"  # Fridays at 02:00
            assert dag.max_active_runs == 1

            # Verify tasks exist
            task_ids = [task.task_id for task in dag.tasks]
            expected_tasks = [
                "start",
                "extract_transaction_data",
                "generate_association_rules",
                "generate_combo_offers",
                "generate_weekly_report",
                "notify_success",
                "notify_failure",
                "end",
            ]

            for expected_task in expected_tasks:
                assert expected_task in task_ids, f"Missing task: {expected_task}"

            # Verify key task dependencies
            extract_task = dag.get_task("extract_transaction_data")
            rules_task = dag.get_task("generate_association_rules")
            combos_task = dag.get_task("generate_combo_offers")
            report_task = dag.get_task("generate_weekly_report")

            # Check upstream dependencies
            assert extract_task in rules_task.upstream_list
            assert rules_task in combos_task.upstream_list
            assert combos_task in report_task.upstream_list

        except ImportError:
            pytest.skip("DAG module not available for structure testing")

    def test_error_handling_in_functions(self, mock_context, temp_data_dir):
        """Test error handling in DAG functions."""

        # Test extract_transaction_data with permission error
        with patch("weekly_report_dag.Path") as mock_path:
            mock_path.return_value = temp_data_dir
            mock_path.side_effect = lambda x: (
                temp_data_dir if "/opt/airflow/dags/data" in str(x) else Path(x)
            )

            with patch("builtins.open", mock_open()) as mock_file:
                mock_file.side_effect = PermissionError("Permission denied")

                with pytest.raises(PermissionError):
                    extract_transaction_data(**mock_context)

        # Test generate_association_rules with invalid file
        mock_context["task_instance"].xcom_pull.return_value = {
            "week_start": "2024-01-01T00:00:00",
            "output_file": "/nonexistent/file.json",
        }

        with pytest.raises(FileNotFoundError):
            generate_association_rules(**mock_context)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
