"""
DAG Status Monitor Page

This page provides real-time monitoring of Airflow DAG status with auto-refresh
functionality and comprehensive status tracking for the weekly report generation.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import json
import sys
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import time

# Import auto-refresh component
try:
    from streamlit_autorefresh import st_autorefresh
except ImportError:
    st.error(
        "streamlit-autorefresh not installed. Run: pip install streamlit-autorefresh"
    )
    st.stop()

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

try:
    from weekly_reports import WeeklyKPICalculator
    from aggregation_views import AssociationAggregator
except ImportError as e:
    st.warning(f"Import warning: {e}")

# Page configuration
st.set_page_config(page_title="DAG Status Monitor", page_icon="📊", layout="wide")

st.title("📊 DAG Status Monitor")
st.markdown(
    "Real-time monitoring of Airflow DAG executions and weekly report generation"
)

# Auto-refresh configuration
if "auto_refresh_enabled" not in st.session_state:
    st.session_state.auto_refresh_enabled = True

if "refresh_interval" not in st.session_state:
    st.session_state.refresh_interval = 30  # 30 seconds default

# Initialize session state for DAG data
if "dag_data" not in st.session_state:
    st.session_state.dag_data = None
if "last_refresh" not in st.session_state:
    st.session_state.last_refresh = datetime.now()


class DAGStatusMonitor:
    """Class for monitoring Airflow DAG status."""

    def __init__(self, airflow_base_url: str = "http://localhost:8080"):
        """
        Initialize DAG status monitor.

        Args:
            airflow_base_url: Base URL for Airflow webserver
        """
        self.airflow_base_url = airflow_base_url
        self.dag_id = "weekly_report_generation"

    def get_dag_status(self) -> Optional[Dict[str, Any]]:
        """
        Get current DAG status from Airflow API.

        Returns:
            Dictionary with DAG status information
        """
        try:
            # In a real implementation, this would call the Airflow REST API
            # For demo purposes, generate mock status data
            return self._generate_mock_dag_status()

        except Exception as e:
            st.error(f"Failed to fetch DAG status: {e}")
            return None

    def get_dag_runs(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent DAG runs.

        Args:
            limit: Number of recent runs to fetch

        Returns:
            List of DAG run information
        """
        try:
            # Mock DAG runs for demonstration
            return self._generate_mock_dag_runs(limit)

        except Exception as e:
            st.error(f"Failed to fetch DAG runs: {e}")
            return []

    def get_task_status(self, dag_run_id: str) -> List[Dict[str, Any]]:
        """
        Get task status for a specific DAG run.

        Args:
            dag_run_id: ID of the DAG run

        Returns:
            List of task status information
        """
        try:
            return self._generate_mock_task_status(dag_run_id)

        except Exception as e:
            st.error(f"Failed to fetch task status: {e}")
            return []

    def _generate_mock_dag_status(self) -> Dict[str, Any]:
        """Generate mock DAG status for demonstration."""
        import random

        # Simulate different DAG states
        states = ["running", "success", "failed", "queued"]
        weights = [0.2, 0.6, 0.1, 0.1]
        current_state = np.random.choice(states, p=weights)

        # Calculate next scheduled run
        now = datetime.now()
        # Weekly report runs every Friday at 02:00
        days_until_friday = (4 - now.weekday()) % 7  # 4 = Friday
        if days_until_friday == 0 and now.hour >= 2:
            days_until_friday = 7  # Next Friday

        next_run = now.replace(hour=2, minute=0, second=0, microsecond=0) + timedelta(
            days=days_until_friday
        )

        return {
            "dag_id": self.dag_id,
            "is_paused": False,
            "current_state": current_state,
            "last_run_time": now - timedelta(days=random.randint(1, 7)),
            "next_scheduled_run": next_run,
            "success_rate": random.uniform(0.85, 0.98),
            "average_duration": random.uniform(120, 300),  # seconds
            "total_runs": random.randint(50, 200),
            "successful_runs": random.randint(45, 190),
            "failed_runs": random.randint(2, 10),
        }

    def _generate_mock_dag_runs(self, limit: int) -> List[Dict[str, Any]]:
        """Generate mock DAG runs for demonstration."""
        runs = []

        for i in range(limit):
            run_date = datetime.now() - timedelta(days=i * 7)  # Weekly runs
            states = ["success", "running", "failed", "queued"]
            weights = [0.7, 0.1, 0.1, 0.1]

            if i == 0:  # Current run
                state = "running"
                end_date = None
                duration = None
            else:
                state = np.random.choice(states, p=weights)
                duration = np.random.uniform(120, 400)  # seconds
                end_date = run_date + timedelta(seconds=duration)

            runs.append(
                {
                    "dag_run_id": f"weekly_report_{run_date.strftime('%Y%m%d')}",
                    "execution_date": run_date,
                    "start_date": run_date,
                    "end_date": end_date,
                    "state": state,
                    "duration": duration,
                    "week_start": run_date - timedelta(days=run_date.weekday()),
                }
            )

        return runs

    def _generate_mock_task_status(self, dag_run_id: str) -> List[Dict[str, Any]]:
        """Generate mock task status for demonstration."""
        tasks = [
            "start",
            "extract_transaction_data",
            "generate_association_rules",
            "generate_combo_offers",
            "generate_weekly_report",
            "notify_success",
            "end",
        ]

        task_status = []
        current_time = datetime.now()

        for i, task_id in enumerate(tasks):
            # Simulate task progression
            if "running" in dag_run_id or i < 3:
                state = "success" if i < 3 else ("running" if i == 3 else "queued")
            else:
                state = np.random.choice(["success", "failed"], p=[0.9, 0.1])

            start_time = (
                current_time - timedelta(minutes=len(tasks) - i)
                if state != "queued"
                else None
            )
            end_time = (
                current_time - timedelta(minutes=len(tasks) - i - 1)
                if state == "success"
                else None
            )
            duration = (
                (end_time - start_time).total_seconds()
                if start_time and end_time
                else None
            )

            task_status.append(
                {
                    "task_id": task_id,
                    "state": state,
                    "start_date": start_time,
                    "end_date": end_time,
                    "duration": duration,
                    "try_number": 1,
                    "max_tries": 3,
                }
            )

        return task_status


@st.cache_data(ttl=60)  # Cache for 1 minute
def load_dag_status():
    """Load DAG status with caching."""
    monitor = DAGStatusMonitor()

    dag_status = monitor.get_dag_status()
    dag_runs = monitor.get_dag_runs(limit=10)

    # Get task status for the most recent run
    task_status = []
    if dag_runs:
        task_status = monitor.get_task_status(dag_runs[0]["dag_run_id"])

    return dag_status, dag_runs, task_status


def render_dag_overview(dag_status: Dict[str, Any]):
    """Render DAG overview section."""
    st.subheader("📋 DAG Overview")

    # Status indicators
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        state_color = {"running": "🟡", "success": "🟢", "failed": "🔴", "queued": "🔵"}
        st.metric(
            "Current State",
            f"{state_color.get(dag_status['current_state'], '⚪')} {dag_status['current_state'].title()}",
        )

    with col2:
        st.metric(
            "Success Rate",
            f"{dag_status['success_rate']:.1%}",
            delta=f"{dag_status['successful_runs']}/{dag_status['total_runs']} runs",
        )

    with col3:
        avg_duration = dag_status["average_duration"]
        st.metric(
            "Avg Duration", f"{avg_duration/60:.1f} min", delta=f"{avg_duration:.0f}s"
        )

    with col4:
        next_run = dag_status["next_scheduled_run"]
        time_until = next_run - datetime.now()
        days = time_until.days
        hours = time_until.seconds // 3600

        st.metric(
            "Next Run", f"{days}d {hours}h", delta=next_run.strftime("%Y-%m-%d %H:%M")
        )


def render_current_run_status(task_status: List[Dict[str, Any]]):
    """Render current run task status."""
    st.subheader("⚡ Current Run Status")

    if not task_status:
        st.info("No active DAG run")
        return

    # Create task timeline visualization
    fig = go.Figure()

    # Define colors for task states
    colors = {
        "success": "#28a745",
        "running": "#ffc107",
        "failed": "#dc3545",
        "queued": "#6c757d",
        "skipped": "#17a2b8",
    }

    # Calculate positions
    y_positions = list(range(len(task_status)))
    task_names = [task["task_id"] for task in task_status]

    # Add bars for each task
    for i, task in enumerate(task_status):
        color = colors.get(task["state"], "#6c757d")

        # Calculate bar width based on duration or default
        if task["duration"]:
            width = task["duration"] / 60  # Convert to minutes
        else:
            width = 2  # Default width for queued/running tasks

        fig.add_trace(
            go.Bar(
                x=[width],
                y=[task_names[i]],
                orientation="h",
                marker_color=color,
                name=task["state"],
                text=(
                    f"{task['state']} ({width:.1f}m)"
                    if task["duration"]
                    else task["state"]
                ),
                textposition="inside",
                showlegend=i == 0,  # Only show legend for first occurrence
            )
        )

    fig.update_layout(
        title="Task Execution Timeline",
        xaxis_title="Duration (minutes)",
        yaxis_title="Tasks",
        height=400,
        barmode="overlay",
    )

    st.plotly_chart(fig, use_container_width=True)

    # Task details table
    st.subheader("📝 Task Details")

    task_df = pd.DataFrame(task_status)

    # Format the dataframe for display
    display_df = task_df[
        ["task_id", "state", "start_date", "end_date", "duration"]
    ].copy()
    display_df["start_date"] = (
        display_df["start_date"].dt.strftime("%H:%M:%S")
        if display_df["start_date"].notna().any()
        else None
    )
    display_df["end_date"] = (
        display_df["end_date"].dt.strftime("%H:%M:%S")
        if display_df["end_date"].notna().any()
        else None
    )
    display_df["duration"] = display_df["duration"].apply(
        lambda x: f"{x:.1f}s" if pd.notna(x) else "-"
    )

    # Color code the state column
    def color_state(val):
        color_map = {
            "success": "background-color: #d4edda",
            "running": "background-color: #fff3cd",
            "failed": "background-color: #f8d7da",
            "queued": "background-color: #e2e3e5",
        }
        return color_map.get(val, "")

    styled_df = display_df.style.applymap(color_state, subset=["state"])
    st.dataframe(styled_df, use_container_width=True)


def render_recent_runs(dag_runs: List[Dict[str, Any]]):
    """Render recent DAG runs section."""
    st.subheader("📚 Recent Runs")

    if not dag_runs:
        st.info("No recent runs found")
        return

    # Convert to DataFrame
    runs_df = pd.DataFrame(dag_runs)

    # Create success rate visualization
    success_counts = runs_df["state"].value_counts()

    col1, col2 = st.columns([2, 1])

    with col1:
        # Timeline chart
        fig = px.timeline(
            runs_df,
            x_start="start_date",
            x_end="end_date",
            y="dag_run_id",
            color="state",
            title="DAG Run Timeline",
            color_discrete_map={
                "success": "#28a745",
                "running": "#ffc107",
                "failed": "#dc3545",
                "queued": "#6c757d",
            },
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Success rate pie chart
        fig = px.pie(
            values=success_counts.values,
            names=success_counts.index,
            title="Run Status Distribution",
            color_discrete_map={
                "success": "#28a745",
                "running": "#ffc107",
                "failed": "#dc3545",
                "queued": "#6c757d",
            },
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    # Recent runs table
    st.subheader("📋 Run Details")

    display_runs = runs_df[["dag_run_id", "execution_date", "state", "duration"]].copy()
    display_runs["execution_date"] = display_runs["execution_date"].dt.strftime(
        "%Y-%m-%d %H:%M"
    )
    display_runs["duration"] = display_runs["duration"].apply(
        lambda x: f"{x/60:.1f} min" if pd.notna(x) else "In progress"
    )

    # Color code the state column
    def color_state(val):
        color_map = {
            "success": "background-color: #d4edda",
            "running": "background-color: #fff3cd",
            "failed": "background-color: #f8d7da",
            "queued": "background-color: #e2e3e5",
        }
        return color_map.get(val, "")

    styled_runs = display_runs.style.applymap(color_state, subset=["state"])
    st.dataframe(styled_runs, use_container_width=True)


def render_performance_metrics(dag_runs: List[Dict[str, Any]]):
    """Render performance metrics section."""
    st.subheader("📈 Performance Metrics")

    if not dag_runs:
        st.info("No performance data available")
        return

    # Calculate metrics
    successful_runs = [
        run for run in dag_runs if run["state"] == "success" and run["duration"]
    ]

    if not successful_runs:
        st.info("No successful runs with duration data")
        return

    durations = [run["duration"] for run in successful_runs]
    execution_dates = [run["execution_date"] for run in successful_runs]

    col1, col2 = st.columns(2)

    with col1:
        # Duration trend
        fig = px.line(
            x=execution_dates,
            y=[d / 60 for d in durations],  # Convert to minutes
            title="Execution Duration Trend",
            labels={"x": "Execution Date", "y": "Duration (minutes)"},
        )
        fig.add_hline(
            y=3, line_dash="dash", line_color="red", annotation_text="Target: 3 min"
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Duration distribution
        fig = px.histogram(
            x=[d / 60 for d in durations],
            title="Duration Distribution",
            labels={"x": "Duration (minutes)", "y": "Count"},
            nbins=10,
        )
        st.plotly_chart(fig, use_container_width=True)

    # Performance summary
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Avg Duration", f"{np.mean(durations)/60:.1f} min")

    with col2:
        st.metric("Min Duration", f"{np.min(durations)/60:.1f} min")

    with col3:
        st.metric("Max Duration", f"{np.max(durations)/60:.1f} min")

    with col4:
        st.metric("Std Dev", f"{np.std(durations)/60:.1f} min")


def render_alerts_section(dag_status: Dict[str, Any], dag_runs: List[Dict[str, Any]]):
    """Render alerts and notifications section."""
    st.subheader("🚨 Alerts & Notifications")

    alerts = []

    # Check for failed runs
    recent_failures = [run for run in dag_runs[:5] if run["state"] == "failed"]
    if recent_failures:
        alerts.append(
            {
                "type": "error",
                "message": f"{len(recent_failures)} failed runs in last 5 executions",
                "details": [run["dag_run_id"] for run in recent_failures],
            }
        )

    # Check for long-running tasks
    if dag_status["current_state"] == "running":
        last_run = dag_runs[0] if dag_runs else None
        if last_run and last_run["start_date"]:
            running_time = (datetime.now() - last_run["start_date"]).total_seconds()
            if running_time > 600:  # 10 minutes
                alerts.append(
                    {
                        "type": "warning",
                        "message": f"Current run has been running for {running_time/60:.1f} minutes",
                        "details": ["Consider checking task logs for potential issues"],
                    }
                )

    # Check success rate
    if dag_status["success_rate"] < 0.9:
        alerts.append(
            {
                "type": "warning",
                "message": f"Success rate is {dag_status['success_rate']:.1%} (below 90%)",
                "details": ["Review recent failures and consider parameter tuning"],
            }
        )

    if not alerts:
        st.success("✅ No active alerts - all systems operational")
    else:
        for alert in alerts:
            if alert["type"] == "error":
                st.error(f"🔴 {alert['message']}")
            else:
                st.warning(f"🟡 {alert['message']}")

            if alert["details"]:
                with st.expander("Details"):
                    for detail in alert["details"]:
                        st.write(f"• {detail}")


def main():
    """Main function for the DAG Status Monitor page."""

    # Sidebar controls
    st.sidebar.header("🎛️ Monitor Settings")

    # Auto-refresh controls
    st.sidebar.subheader("🔄 Auto-Refresh")

    auto_refresh_enabled = st.sidebar.checkbox(
        "Enable Auto-Refresh",
        value=st.session_state.auto_refresh_enabled,
        help="Automatically refresh the page every few seconds",
    )

    if auto_refresh_enabled:
        refresh_interval = st.sidebar.selectbox(
            "Refresh Interval",
            options=[10, 30, 60, 120, 300],
            index=1,  # Default to 30 seconds
            format_func=lambda x: f"{x} seconds",
        )
        st.session_state.refresh_interval = refresh_interval

        # Auto-refresh component
        count = st_autorefresh(interval=refresh_interval * 1000, key="dag_monitor")

        if count > 0:
            st.sidebar.success(f"🔄 Refreshed {count} times")

    st.session_state.auto_refresh_enabled = auto_refresh_enabled

    # Manual refresh button
    if st.sidebar.button("🔄 Manual Refresh"):
        st.cache_data.clear()
        st.experimental_rerun()

    # Connection settings
    st.sidebar.subheader("🔗 Connection")
    airflow_url = st.sidebar.text_input(
        "Airflow URL",
        value="http://localhost:8080",
        help="URL to your Airflow webserver",
    )

    # Load status data
    try:
        with st.spinner("Loading DAG status..."):
            dag_status, dag_runs, task_status = load_dag_status()

        st.session_state.last_refresh = datetime.now()

    except Exception as e:
        st.error(f"Failed to load DAG status: {e}")
        return

    # Main content
    if dag_status:
        # Header with last refresh time
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown("### 📊 Weekly Report DAG Status")
        with col2:
            st.markdown(
                f"**Last Updated:** {st.session_state.last_refresh.strftime('%H:%M:%S')}"
            )

        # Render sections
        render_dag_overview(dag_status)

        st.markdown("---")

        # Current run status
        if task_status:
            render_current_run_status(task_status)
            st.markdown("---")

        # Recent runs
        if dag_runs:
            render_recent_runs(dag_runs)
            st.markdown("---")

            # Performance metrics
            render_performance_metrics(dag_runs)
            st.markdown("---")

        # Alerts section
        render_alerts_section(dag_status, dag_runs)

        # Export functionality
        st.subheader("📁 Export Data")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("📊 Export Status Report"):
                report_data = {
                    "dag_status": dag_status,
                    "recent_runs": dag_runs,
                    "task_status": task_status,
                    "export_time": datetime.now().isoformat(),
                }

                st.download_button(
                    label="Download JSON Report",
                    data=json.dumps(report_data, indent=2, default=str),
                    file_name=f"dag_status_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                )

        with col2:
            if st.button("📋 Export Run History"):
                if dag_runs:
                    runs_df = pd.DataFrame(dag_runs)
                    csv = runs_df.to_csv(index=False)

                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name=f"dag_runs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                    )

    else:
        st.error("Unable to load DAG status. Please check your Airflow connection.")

    # Footer
    st.markdown("---")
    st.markdown(
        "💡 **Tips:** \n"
        "- Enable auto-refresh for real-time monitoring\n"
        "- Check alerts section for any issues requiring attention\n"
        "- Export reports for historical analysis\n"
        "- Monitor performance trends to optimize execution times"
    )


if __name__ == "__main__":
    main()
