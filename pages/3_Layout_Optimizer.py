import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os

# Add src directory to path so we can import our modules
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))

from layout_optimizer import SupermarketLayoutOptimizer, ProductLocation, StoreZone
from association_engine import AprioriAssociationEngine
from qloo_client import create_qloo_client

st.set_page_config(page_title="Layout Optimizer", page_icon="🏪", layout="wide")

st.markdown("# 🏪 Supermarket Layout Optimizer")
st.sidebar.header("Layout Optimizer")


# Cache data loading
@st.cache_data
def load_data():
    """Load all necessary data."""
    try:
        catalog_df = pd.read_csv("data/grocery_catalog.csv")
        transactions_df = pd.read_csv("data/sample_transactions.csv")
        return catalog_df, transactions_df
    except FileNotFoundError as e:
        st.error(f"Data file not found: {e}")
        return pd.DataFrame(), pd.DataFrame()


@st.cache_resource
def initialize_optimizer():
    """Initialize the layout optimizer with all components."""
    catalog_df, transactions_df = load_data()

    if catalog_df.empty or transactions_df.empty:
        return None, catalog_df, transactions_df

    # Initialize Qloo client
    qloo_client = None
    try:
        qloo_client = create_qloo_client()
    except Exception as e:
        st.warning(f"Qloo API not available: {e}")

    # Initialize optimizer
    optimizer = SupermarketLayoutOptimizer(qloo_client=qloo_client)

    with st.spinner("Loading data and training models..."):
        optimizer.load_data(catalog_df, transactions_df)

    return optimizer, catalog_df, transactions_df


def create_store_layout_visualization(optimizer, recommendations=None):
    """Create an interactive store layout visualization."""

    # Create figure
    fig = go.Figure()

    # Add store zones as background rectangles
    for zone in optimizer.store_zones.values():
        fig.add_shape(
            type="rect",
            x0=zone.x_range[0],
            y0=zone.y_range[0],
            x1=zone.x_range[1],
            y1=zone.y_range[1],
            fillcolor=get_zone_color(zone.zone_type),
            opacity=0.3,
            line=dict(color="gray", width=1),
            layer="below",
        )

        # Add zone labels
        fig.add_annotation(
            x=(zone.x_range[0] + zone.x_range[1]) / 2,
            y=(zone.y_range[0] + zone.y_range[1]) / 2,
            text=zone.zone_name,
            showarrow=False,
            font=dict(size=10),
            bgcolor="white",
            opacity=0.8,
        )

    # Add current product locations
    for product_id, location in optimizer.current_layout.items():
        # Determine color based on category
        color = get_category_color(location.category)

        fig.add_trace(
            go.Scatter(
                x=[location.x_coordinate],
                y=[location.y_coordinate],
                mode="markers",
                marker=dict(
                    size=12,
                    color=color,
                    symbol="circle",
                    line=dict(width=1, color="black"),
                ),
                text=f"{location.product_name}<br>Category: {location.category}<br>Zone: {location.zone}",
                hovertemplate="%{text}<extra></extra>",
                name=location.category,
                showlegend=(
                    True
                    if product_id == min(optimizer.current_layout.keys())
                    else False
                ),
                legendgroup=location.category,
            )
        )

    # Add recommendations if provided
    if recommendations:
        for rec in recommendations[:10]:  # Show top 10 recommendations
            current = rec.current_location
            recommended = rec.recommended_location

            if current and recommended:
                # Add arrow from current to recommended location
                fig.add_annotation(
                    x=recommended.x_coordinate,
                    y=recommended.y_coordinate,
                    ax=current.x_coordinate,
                    ay=current.y_coordinate,
                    xref="x",
                    yref="y",
                    axref="x",
                    ayref="y",
                    text="",
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=2,
                    arrowcolor="red",
                )

                # Add recommended location marker
                fig.add_trace(
                    go.Scatter(
                        x=[recommended.x_coordinate],
                        y=[recommended.y_coordinate],
                        mode="markers",
                        marker=dict(
                            size=15,
                            color="red",
                            symbol="star",
                            line=dict(width=2, color="darkred"),
                        ),
                        text=f"RECOMMENDED: {recommended.product_name}<br>{rec.reason}<br>Confidence: {rec.confidence_score:.2f}",
                        hovertemplate="%{text}<extra></extra>",
                        name="Recommendations",
                        showlegend=True if rec == recommendations[0] else False,
                    )
                )

    fig.update_layout(
        title="Store Layout Visualization",
        xaxis_title="Store Width",
        yaxis_title="Store Depth",
        width=800,
        height=600,
        showlegend=True,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=1.01),
    )

    return fig


def get_zone_color(zone_type):
    """Get color for different zone types."""
    colors = {
        "entrance": "lightgreen",
        "perimeter": "lightblue",
        "main": "lightyellow",
        "checkout": "lightcoral",
    }
    return colors.get(zone_type, "lightgray")


def get_category_color(category):
    """Get color for different product categories."""
    colors = {
        "Produce": "green",
        "Dairy": "blue",
        "Meat": "red",
        "Bakery": "orange",
        "Beverages": "purple",
        "Snacks": "brown",
    }
    return colors.get(category, "gray")


def display_recommendations_table(recommendations, catalog_df):
    """Display recommendations in a formatted table."""
    if not recommendations:
        st.info(
            "No recommendations generated. Try adjusting the optimization parameters."
        )
        return

    # Prepare data for display
    rec_data = []
    for rec in recommendations:
        rec_data.append(
            {
                "Product": (
                    rec.current_location.product_name
                    if rec.current_location
                    else "Unknown"
                ),
                "Category": (
                    rec.current_location.category if rec.current_location else "Unknown"
                ),
                "Current Zone": (
                    rec.current_location.zone if rec.current_location else "Unknown"
                ),
                "Recommended Zone": rec.recommended_location.zone,
                "Reason": rec.reason,
                "Confidence": f"{rec.confidence_score:.2f}",
                "Potential Lift": f"{rec.potential_lift:.2f}",
                "Priority": get_priority_level(rec.confidence_score),
            }
        )

    rec_df = pd.DataFrame(rec_data)

    # Style the dataframe
    def highlight_priority(val):
        if val == "High":
            return "background-color: #d4edda"
        elif val == "Medium":
            return "background-color: #fff3cd"
        elif val == "Low":
            return "background-color: #f8d7da"
        return ""

    styled_df = rec_df.style.applymap(highlight_priority, subset=["Priority"])
    st.dataframe(styled_df, use_container_width=True)


def get_priority_level(confidence_score):
    """Get priority level based on confidence score."""
    if confidence_score >= 0.7:
        return "High"
    elif confidence_score >= 0.4:
        return "Medium"
    else:
        return "Low"


def display_layout_metrics(optimizer):
    """Display layout performance metrics."""
    report = optimizer.generate_layout_report()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Products", report["total_products"])

    with col2:
        st.metric("Zones Used", report["zones_utilized"])

    with col3:
        st.metric("Association Score", f"{report['association_score']:.2f}")

    with col4:
        categories = len(report["category_distribution"])
        st.metric("Categories", categories)

    # Category distribution chart
    if report["category_distribution"]:
        st.subheader("📊 Category Distribution")
        cat_df = pd.DataFrame(
            list(report["category_distribution"].items()), columns=["Category", "Count"]
        )

        fig_cat = px.bar(
            cat_df,
            x="Category",
            y="Count",
            title="Products per Category",
            color="Category",
        )
        st.plotly_chart(fig_cat, use_container_width=True)

    # Zone utilization chart
    if report["zone_utilization"]:
        st.subheader("🏬 Zone Utilization")
        zone_df = pd.DataFrame(
            list(report["zone_utilization"].items()), columns=["Zone", "Products"]
        )

        fig_zone = px.pie(
            zone_df, values="Products", names="Zone", title="Products per Zone"
        )
        st.plotly_chart(fig_zone, use_container_width=True)


def display_qloo_integration(optimizer, selected_product_id):
    """Display Qloo API integration results."""
    if not optimizer.qloo_client:
        st.warning(
            "Qloo API not available. Layout optimization is based only on transaction data."
        )
        return

    st.subheader("🔮 Qloo AI Recommendations")

    try:
        qloo_recommendations = optimizer.get_qloo_recommendations(
            selected_product_id, limit=5
        )

        if qloo_recommendations:
            st.success(f"✅ Found {len(qloo_recommendations)} Qloo recommendations")

            for i, rec in enumerate(qloo_recommendations):
                with st.expander(f"Recommendation {i+1}: {rec.get('name', 'Unknown')}"):
                    st.write(f"**ID:** {rec.get('id', 'N/A')}")
                    st.write(f"**Name:** {rec.get('name', 'N/A')}")
                    if "confidence_score" in rec:
                        st.write(f"**Confidence:** {rec['confidence_score']:.2f}")
        else:
            st.info("No Qloo recommendations available for this product.")

    except Exception as e:
        st.error(f"Error getting Qloo recommendations: {e}")


def main():
    """Main function for the Layout Optimizer page."""

    # Initialize optimizer
    optimizer, catalog_df, transactions_df = initialize_optimizer()

    if optimizer is None:
        st.error("Failed to initialize layout optimizer. Please check your data files.")
        return

    # Main interface
    st.markdown(
        """
    This tool optimizes supermarket layouts using:
    - **Association Rules**: From transaction data analysis
    - **Qloo API**: External AI-powered recommendations  
    - **Category Logic**: Industry best practices
    - **Customer Flow**: Traffic pattern optimization
    """
    )

    # Sidebar controls
    st.sidebar.subheader("Optimization Settings")

    optimization_goals = st.sidebar.multiselect(
        "Select Optimization Goals:",
        options=["maximize_associations", "improve_flow", "category_adjacency"],
        default=["maximize_associations", "improve_flow"],
        help="Choose which factors to optimize for",
    )

    show_recommendations = st.sidebar.checkbox(
        "Show Layout Recommendations", value=True
    )
    show_qloo_integration = st.sidebar.checkbox("Enable Qloo Integration", value=True)

    # Product selection for detailed analysis
    if not catalog_df.empty:
        product_options = {
            f"{row['product_name']} (ID: {row['product_id']})": row["product_id"]
            for _, row in catalog_df.iterrows()
        }

        selected_product_display = st.sidebar.selectbox(
            "Select Product for Analysis:",
            options=list(product_options.keys()),
            index=0,
        )
        selected_product_id = product_options[selected_product_display]

    # Generate recommendations
    recommendations = []
    if show_recommendations and optimization_goals:
        with st.spinner("Generating layout recommendations..."):
            recommendations = optimizer.optimize_layout(optimization_goals)

    # Main content area
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("🗺️ Store Layout")

        # Create and display layout visualization
        layout_fig = create_store_layout_visualization(
            optimizer, recommendations if show_recommendations else None
        )
        st.plotly_chart(layout_fig, use_container_width=True)

        # Layout metrics
        display_layout_metrics(optimizer)

    with col2:
        st.subheader("📈 Performance Metrics")

        # Current layout analysis
        report = optimizer.generate_layout_report()

        st.metric("Layout Efficiency", f"{report['association_score']:.1%}")

        if report["association_score"] < 0.5:
            st.warning(
                "⚠️ Layout efficiency is low. Consider implementing recommendations."
            )
        elif report["association_score"] < 0.7:
            st.info("ℹ️ Layout efficiency is moderate. Some improvements possible.")
        else:
            st.success("✅ Layout efficiency is good!")

        # Qloo integration
        if show_qloo_integration and "selected_product_id" in locals():
            display_qloo_integration(optimizer, selected_product_id)

    # Recommendations section
    if show_recommendations and recommendations:
        st.header("🎯 Layout Optimization Recommendations")

        # Summary metrics
        high_priority = sum(1 for rec in recommendations if rec.confidence_score >= 0.7)
        medium_priority = sum(
            1 for rec in recommendations if 0.4 <= rec.confidence_score < 0.7
        )
        low_priority = sum(1 for rec in recommendations if rec.confidence_score < 0.4)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Recommendations", len(recommendations))
        with col2:
            st.metric("High Priority", high_priority)
        with col3:
            st.metric("Medium Priority", medium_priority)
        with col4:
            st.metric("Low Priority", low_priority)

        # Recommendations table
        st.subheader("📋 Detailed Recommendations")
        display_recommendations_table(recommendations, catalog_df)

        # Implementation guide
        with st.expander("📖 Implementation Guide"):
            st.markdown(
                """
            **How to Implement These Recommendations:**
            
            1. **High Priority (0.7+ confidence)**: Implement immediately for maximum impact
            2. **Medium Priority (0.4-0.7 confidence)**: Consider for next layout refresh
            3. **Low Priority (<0.4 confidence)**: Monitor and re-evaluate with more data
            
            **Best Practices:**
            - Implement changes gradually to avoid customer confusion
            - Monitor sales data before and after changes
            - Consider seasonal variations in product demand
            - Maintain clear sight lines and accessibility
            - Test changes in a pilot section first
            """
            )

    # Export functionality
    if st.button("📊 Export Layout Report"):
        # Create comprehensive report
        report_data = {
            "layout_metrics": optimizer.generate_layout_report(),
            "recommendations": [
                {
                    "product_id": rec.product_id,
                    "product_name": (
                        rec.current_location.product_name
                        if rec.current_location
                        else "Unknown"
                    ),
                    "current_zone": (
                        rec.current_location.zone if rec.current_location else "Unknown"
                    ),
                    "recommended_zone": rec.recommended_location.zone,
                    "reason": rec.reason,
                    "confidence_score": rec.confidence_score,
                    "potential_lift": rec.potential_lift,
                }
                for rec in recommendations
            ],
        }

        # Convert to DataFrame for download
        if recommendations:
            rec_df = pd.DataFrame(report_data["recommendations"])
            csv = rec_df.to_csv(index=False)

            st.download_button(
                label="📥 Download Recommendations CSV",
                data=csv,
                file_name=f"layout_recommendations_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
            )


if __name__ == "__main__":
    main()
