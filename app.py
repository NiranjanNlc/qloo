import streamlit as st
import requests
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from datetime import datetime, timedelta
import time

st.set_page_config(
    page_title="Qloo Supermarket Optimizer",
    page_icon="🛒",
    layout="wide"
)

# Custom CSS for enhanced KPI cards
st.markdown("""
<style>
.metric-card {
    background-color: #f8f9fa;
    border: 1px solid #dee2e6;
    border-radius: 0.5rem;
    padding: 1rem;
    margin: 0.5rem 0;
    box-shadow: 0 0.125rem 0.25rem rgba(0, 0, 0, 0.075);
}

.metric-value {
    font-size: 2rem;
    font-weight: bold;
    color: #198754;
}

.metric-label {
    font-size: 0.875rem;
    color: #6c757d;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.status-good {
    color: #198754;
}

.status-warning {
    color: #fd7e14;
}

.status-critical {
    color: #dc3545;
}

.trend-up {
    color: #198754;
}

.trend-down {
    color: #dc3545;
}

.trend-stable {
    color: #6c757d;
}

@media (max-width: 768px) {
    .metric-value {
        font-size: 1.5rem;
    }
}
</style>
""", unsafe_allow_html=True)

# API Configuration
API_BASE_URL = "http://localhost:8000"  # Will be configurable via environment

def fetch_metrics():
    """Fetch metrics from the API."""
    try:
        response = requests.get(f"{API_BASE_URL}/metrics", timeout=5)
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except Exception as e:
        st.error(f"Failed to fetch metrics: {e}")
        return None

def create_kpi_card(title, current, target, previous, unit, status, trend, change_percent=None):
    """Create a KPI card with metrics."""
    status_class = f"status-{status}"
    trend_class = f"trend-{trend}"
    
    # Format values based on unit
    if unit == "currency":
        current_formatted = f"${current:,.0f}"
        target_formatted = f"${target:,.0f}"
    elif unit == "ms":
        current_formatted = f"{current:.0f}ms"
        target_formatted = f"{target:.0f}ms"
    elif unit == "score":
        current_formatted = f"{current:.3f}"
        target_formatted = f"{target:.3f}"
    else:
        current_formatted = f"{current:,.0f}"
        target_formatted = f"{target:,.0f}"
    
    # Trend arrow
    trend_arrow = "↗️" if trend == "up" else "↘️" if trend == "down" else "➡️"
    
    # Change text
    change_text = ""
    if change_percent is not None:
        change_sign = "+" if change_percent > 0 else ""
        change_text = f" ({change_sign}{change_percent:.1f}%)"
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{title}</div>
            <div class="metric-value {status_class}">{current_formatted}</div>
            <div style="font-size: 0.75rem; color: #6c757d;">
                Target: {target_formatted} | {trend_arrow} <span class="{trend_class}">{trend}</span>{change_text}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Create a mini gauge chart
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = (current / target * 100) if target > 0 else 0,
            domain = {'x': [0, 1], 'y': [0, 1]},
            gauge = {
                'axis': {'range': [None, 150]},
                'bar': {'color': "#198754" if status == "good" else "#fd7e14" if status == "warning" else "#dc3545"},
                'steps': [
                    {'range': [0, 80], 'color': "#f8d7da"},
                    {'range': [80, 100], 'color': "#fff3cd"},
                    {'range': [100, 150], 'color': "#d1e7dd"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 100
                }
            }
        ))
        fig.update_layout(height=120, margin=dict(l=5, r=5, t=5, b=5))
        st.plotly_chart(fig, use_container_width=True)

def create_performance_chart(metrics_data):
    """Create performance overview chart."""
    if not metrics_data:
        return
    
    performance = metrics_data.get('performance', {})
    
    # Create performance metrics chart
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('API Response Time', 'Success Rate', 'Cache Hit Rate', 'Memory Usage'),
        specs=[[{"type": "indicator"}, {"type": "indicator"}],
               [{"type": "indicator"}, {"type": "indicator"}]]
    )
    
    # API Response Time
    fig.add_trace(go.Indicator(
        mode = "gauge+number",
        value = performance.get('api_response_time_ms', 0),
        domain = {'row': 0, 'column': 0},
        title = {'text': "Response Time (ms)"},
        gauge = {'axis': {'range': [None, 500]},
                'bar': {'color': "darkblue"},
                'steps': [{'range': [0, 200], 'color': "lightgray"},
                         {'range': [200, 500], 'color': "yellow"}],
                'threshold': {'line': {'color': "red", 'width': 4},
                            'thickness': 0.75, 'value': 300}}),
        row=1, col=1)
    
    # Success Rate
    fig.add_trace(go.Indicator(
        mode = "gauge+number",
        value = performance.get('success_rate', 0),
        domain = {'row': 0, 'column': 1},
        title = {'text': "Success Rate (%)"},
        gauge = {'axis': {'range': [None, 100]},
                'bar': {'color': "green"},
                'steps': [{'range': [0, 95], 'color': "lightgray"},
                         {'range': [95, 100], 'color': "lightgreen"}],
                'threshold': {'line': {'color': "red", 'width': 4},
                            'thickness': 0.75, 'value': 99}}),
        row=1, col=2)
    
    # Cache Hit Rate
    fig.add_trace(go.Indicator(
        mode = "gauge+number",
        value = performance.get('cache_hit_rate', 0),
        domain = {'row': 1, 'column': 0},
        title = {'text': "Cache Hit Rate (%)"},
        gauge = {'axis': {'range': [None, 100]},
                'bar': {'color': "orange"},
                'steps': [{'range': [0, 70], 'color': "lightgray"},
                         {'range': [70, 100], 'color': "lightyellow"}],
                'threshold': {'line': {'color': "red", 'width': 4},
                            'thickness': 0.75, 'value': 80}}),
        row=2, col=1)
    
    # Memory Usage
    fig.add_trace(go.Indicator(
        mode = "gauge+number",
        value = performance.get('memory_usage_mb', 0),
        domain = {'row': 1, 'column': 1},
        title = {'text': "Memory Usage (MB)"},
        gauge = {'axis': {'range': [None, 1000]},
                'bar': {'color': "purple"},
                'steps': [{'range': [0, 500], 'color': "lightgray"},
                         {'range': [500, 1000], 'color': "lightcoral"}],
                'threshold': {'line': {'color': "red", 'width': 4},
                            'thickness': 0.75, 'value': 800}}),
        row=2, col=2)
    
    fig.update_layout(height=400, margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig, use_container_width=True)

# Main Dashboard
st.title("🛒 Qloo Supermarket Optimizer - Dashboard")

# Auto-refresh every 30 seconds
if 'last_refresh' not in st.session_state:
    st.session_state.last_refresh = time.time()

if time.time() - st.session_state.last_refresh > 30:
    st.rerun()

# Fetch metrics
with st.spinner('Loading metrics...'):
    metrics_data = fetch_metrics()

if metrics_data:
    kpis = metrics_data.get('kpis', {})
    business = metrics_data.get('business', {})
    performance = metrics_data.get('performance', {})
    
    # Header with refresh info
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.markdown(f"**Last Updated:** {datetime.fromisoformat(metrics_data['timestamp']).strftime('%Y-%m-%d %H:%M:%S')}")
    with col2:
        st.markdown(f"**Period:** {metrics_data['period']}")
    with col3:
        if st.button("🔄 Refresh", key="refresh_metrics"):
            st.rerun()
    
    st.markdown("---")
    
    # KPI Cards Section
    st.header("📊 Key Performance Indicators")
    
    # Row 1: Business KPIs
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if 'combos_generated' in kpis:
            kpi = kpis['combos_generated']
            create_kpi_card(
                "Combos Generated",
                kpi['current'],
                kpi['target'], 
                kpi['previous'],
                kpi['unit'],
                kpi['status'],
                kpi['trend'],
                kpi.get('change_percent')
            )
    
    with col2:
        if 'avg_confidence' in kpis:
            kpi = kpis['avg_confidence']
            create_kpi_card(
                "Avg Confidence Score",
                kpi['current'],
                kpi['target'],
                kpi['previous'],
                kpi['unit'],
                kpi['status'],
                kpi['trend'],
                kpi.get('change_percent')
            )
    
    with col3:
        if 'revenue_potential' in kpis:
            kpi = kpis['revenue_potential']
            create_kpi_card(
                "Revenue Potential",
                kpi['current'],
                kpi['target'],
                kpi['previous'],
                kpi['unit'],
                kpi['status'],
                kpi['trend'],
                kpi.get('change_percent')
            )
    
    with col4:
        if 'api_performance' in kpis:
            kpi = kpis['api_performance']
            create_kpi_card(
                "API Response Time",
                kpi['current'],
                kpi['target'],
                kpi['previous'],
                kpi['unit'],
                kpi['status'],
                kpi['trend'],
                kpi.get('change_percent')
            )
    
    # Business Metrics Summary
    st.header("📈 Business Metrics Summary")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("High Confidence Rules", business.get('high_confidence_rules', 0))
        st.metric("Active Categories", business.get('active_categories', 0))
    
    with col2:
        st.metric("Avg Lift Score", f"{business.get('avg_lift_score', 0):.3f}")
        st.metric("Avg Discount %", f"{business.get('avg_discount_percent', 0):.1f}%")
    
    with col3:
        st.metric("Success Rate", f"{business.get('combo_success_rate', 0):.1f}%")
        st.metric("Total Combos", business.get('total_combos_generated', 0))
    
    # Performance Overview
    st.header("⚡ System Performance Overview")
    create_performance_chart(metrics_data)
    
    # System Health Status
    st.header("🔧 System Health")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        requests_per_min = performance.get('requests_per_minute', 0)
        st.metric("Requests/min", requests_per_min, delta=f"+{requests_per_min-20}" if requests_per_min > 20 else None)
    
    with col2:
        cpu_usage = performance.get('cpu_usage_percent', 0)
        st.metric("CPU Usage", f"{cpu_usage:.1f}%", delta=f"+{cpu_usage-10:.1f}%" if cpu_usage > 10 else None)
    
    with col3:
        db_query_time = performance.get('database_query_time_ms', 0)
        st.metric("DB Query Time", f"{db_query_time}ms", delta=f"-{30-db_query_time}ms" if db_query_time < 30 else None)
    
    with col4:
        cache_hit_rate = performance.get('cache_hit_rate', 0)
        st.metric("Cache Hit Rate", f"{cache_hit_rate:.1f}%", delta=f"+{cache_hit_rate-70:.1f}%" if cache_hit_rate > 70 else None)

else:
    # Fallback UI when API is not available
    st.error("🚨 Unable to connect to metrics API. Please ensure the API service is running.")
    
    st.markdown("""
    ### System Status: Offline
    
    The metrics dashboard requires the API service to be running. Please:
    
    1. Start the API service: `docker-compose up -d`
    2. Check API health: `curl http://localhost:8000/health`
    3. Refresh this page
    """)

# Sidebar Navigation
st.sidebar.success("Select a page above to explore detailed features.")

st.sidebar.markdown("""
### Quick Actions
- 🔄 Auto-refresh: Every 30 seconds
- 📊 View detailed metrics in other pages
- ⚙️ Configure alerts and thresholds
""")

st.sidebar.markdown("""
### System Info
- **Environment**: Development
- **Version**: 1.0.0
- **Last Deploy**: Today
""")

# Footer
st.markdown("---")
st.markdown("""
**🛒 Qloo-Powered Supermarket Layout Optimizer**  
*Optimizing supermarket layouts through AI-powered product association analysis*

**👈 Select a page from the sidebar** to explore detailed analytics and configurations.
""") 