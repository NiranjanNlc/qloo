"""
Product Flow Analysis Page

This page provides Sankey diagram visualizations for analyzing customer flow patterns,
product associations, and section-to-section movement analytics.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import networkx as nx
import sys
import os
from datetime import datetime, timedelta
from collections import defaultdict

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

try:
    from layout_optimizer import SupermarketLayoutOptimizer, StoreSection
    from association_engine import AprioriAssociationEngine
    from models import Product
except ImportError as e:
    st.error(f"Import error: {e}")
    st.stop()

# Page configuration
st.set_page_config(page_title="Product Flow Analysis", page_icon="🌊", layout="wide")

st.title("🌊 Product Flow Analysis")
st.markdown(
    "Interactive Sankey diagrams showing customer flow patterns and product associations"
)

# Initialize session state
if "flow_data" not in st.session_state:
    st.session_state.flow_data = None
if "section_data" not in st.session_state:
    st.session_state.section_data = None


@st.cache_data
def generate_mock_flow_data():
    """Generate mock data for flow analysis."""
    np.random.seed(42)

    # Define store sections with realistic flow patterns
    sections = {
        "entrance": {"name": "Entrance", "traffic_weight": 1.0, "category": "entry"},
        "produce_fresh": {
            "name": "Fresh Produce",
            "traffic_weight": 0.85,
            "category": "fresh",
        },
        "produce_packaged": {
            "name": "Packaged Produce",
            "traffic_weight": 0.65,
            "category": "fresh",
        },
        "dairy_milk_eggs": {
            "name": "Milk & Eggs",
            "traffic_weight": 0.90,
            "category": "dairy",
        },
        "dairy_cheese_yogurt": {
            "name": "Cheese & Yogurt",
            "traffic_weight": 0.70,
            "category": "dairy",
        },
        "meat_fresh": {
            "name": "Fresh Meat",
            "traffic_weight": 0.75,
            "category": "meat",
        },
        "meat_deli": {
            "name": "Deli Counter",
            "traffic_weight": 0.60,
            "category": "meat",
        },
        "center_aisle_1": {
            "name": "Beverages & Snacks",
            "traffic_weight": 0.80,
            "category": "packaged",
        },
        "center_aisle_2": {
            "name": "Pantry Staples",
            "traffic_weight": 0.70,
            "category": "packaged",
        },
        "center_aisle_3": {
            "name": "Household Items",
            "traffic_weight": 0.50,
            "category": "household",
        },
        "bakery": {"name": "Bakery", "traffic_weight": 0.55, "category": "fresh"},
        "frozen": {
            "name": "Frozen Foods",
            "traffic_weight": 0.65,
            "category": "frozen",
        },
        "checkout": {"name": "Checkout", "traffic_weight": 1.0, "category": "exit"},
    }

    # Define typical flow patterns (from -> to -> weight)
    flow_patterns = [
        # Entry flows
        ("entrance", "produce_fresh", 0.7),
        ("entrance", "center_aisle_1", 0.4),
        ("entrance", "dairy_milk_eggs", 0.3),
        # Fresh produce flows
        ("produce_fresh", "produce_packaged", 0.5),
        ("produce_fresh", "dairy_milk_eggs", 0.6),
        ("produce_fresh", "meat_fresh", 0.4),
        # Dairy flows
        ("dairy_milk_eggs", "dairy_cheese_yogurt", 0.4),
        ("dairy_milk_eggs", "meat_fresh", 0.3),
        ("dairy_milk_eggs", "checkout", 0.8),
        # Meat section flows
        ("meat_fresh", "meat_deli", 0.3),
        ("meat_fresh", "frozen", 0.2),
        ("meat_fresh", "checkout", 0.6),
        # Center aisle flows
        ("center_aisle_1", "center_aisle_2", 0.5),
        ("center_aisle_1", "checkout", 0.7),
        ("center_aisle_2", "center_aisle_3", 0.3),
        ("center_aisle_2", "checkout", 0.8),
        # Other flows
        ("bakery", "checkout", 0.9),
        ("frozen", "checkout", 0.9),
        ("center_aisle_3", "checkout", 0.9),
        # Cross-category flows
        ("produce_packaged", "center_aisle_1", 0.3),
        ("dairy_cheese_yogurt", "bakery", 0.2),
        ("meat_deli", "center_aisle_2", 0.2),
    ]

    # Generate customer journey data
    n_customers = 5000
    customer_journeys = []

    for customer_id in range(n_customers):
        # Start at entrance
        current_section = "entrance"
        journey = [current_section]
        visited = {current_section}

        # Simulate customer path through store
        max_sections = np.random.poisson(4) + 2  # 2-8 sections typically

        for _ in range(max_sections):
            # Find possible next sections
            possible_next = [
                (target, weight)
                for source, target, weight in flow_patterns
                if source == current_section and target not in visited
            ]

            if not possible_next:
                break

            # Choose next section based on weights
            targets, weights = zip(*possible_next)
            weights = np.array(weights)
            weights = weights / weights.sum()  # Normalize

            next_section = np.random.choice(targets, p=weights)
            journey.append(next_section)
            visited.add(next_section)
            current_section = next_section

            # High probability of ending at checkout
            if current_section == "checkout":
                break

        # Ensure journey ends at checkout if not already
        if journey[-1] != "checkout":
            journey.append("checkout")

        customer_journeys.append(
            {
                "customer_id": customer_id,
                "journey": journey,
                "journey_length": len(journey),
                "sections_visited": len(set(journey)),
                "total_time_minutes": np.random.normal(25, 8),  # Mock shopping time
            }
        )

    return sections, flow_patterns, customer_journeys


@st.cache_data
def generate_product_association_flows():
    """Generate product-to-product association flows."""
    np.random.seed(42)

    # Define product categories and representative products
    product_categories = {
        "Dairy": ["Milk", "Cheese", "Yogurt", "Butter", "Eggs"],
        "Produce": ["Bananas", "Apples", "Lettuce", "Tomatoes", "Onions"],
        "Meat": ["Ground Beef", "Chicken Breast", "Salmon", "Bacon", "Turkey"],
        "Beverages": ["Orange Juice", "Soda", "Water", "Coffee", "Tea"],
        "Snacks": ["Chips", "Crackers", "Nuts", "Cookies", "Candy"],
        "Bakery": ["Bread", "Bagels", "Muffins", "Cake", "Donuts"],
        "Pantry": ["Rice", "Pasta", "Beans", "Cereal", "Oil"],
    }

    # Generate strong product associations
    product_flows = []

    # Within-category associations
    for category, products in product_categories.items():
        for i, product1 in enumerate(products):
            for product2 in products[i + 1 :]:
                if np.random.random() < 0.3:  # 30% chance of association
                    strength = np.random.uniform(0.2, 0.8)
                    product_flows.append((product1, product2, strength))

    # Cross-category associations (common shopping patterns)
    cross_category_pairs = [
        ("Milk", "Cookies", 0.6),
        ("Bread", "Butter", 0.7),
        ("Pasta", "Ground Beef", 0.5),
        ("Bananas", "Yogurt", 0.4),
        ("Chips", "Soda", 0.8),
        ("Coffee", "Milk", 0.5),
        ("Lettuce", "Tomatoes", 0.6),
        ("Chicken Breast", "Rice", 0.4),
        ("Apples", "Cheese", 0.3),
        ("Bacon", "Eggs", 0.9),
    ]

    product_flows.extend(cross_category_pairs)

    return product_categories, product_flows


def create_customer_flow_sankey(sections, flow_patterns, min_flow_strength=0.1):
    """Create Sankey diagram for customer flow between sections."""
    # Filter flows by minimum strength
    filtered_flows = [(s, t, w) for s, t, w in flow_patterns if w >= min_flow_strength]

    # Create node lists
    all_sections = list(sections.keys())
    section_names = [sections[s]["name"] for s in all_sections]

    # Create section-to-index mapping
    section_to_idx = {section: i for i, section in enumerate(all_sections)}

    # Build flows for Sankey
    source_indices = []
    target_indices = []
    flow_values = []

    for source, target, weight in filtered_flows:
        if source in section_to_idx and target in section_to_idx:
            source_indices.append(section_to_idx[source])
            target_indices.append(section_to_idx[target])
            # Scale flow values for better visualization
            flow_values.append(weight * 1000)

    # Create node colors based on section category
    category_colors = {
        "entry": "#FF6B6B",  # Red
        "fresh": "#4ECDC4",  # Teal
        "dairy": "#45B7D1",  # Blue
        "meat": "#FFA07A",  # Light Orange
        "packaged": "#98D8C8",  # Light Green
        "household": "#F7DC6F",  # Yellow
        "frozen": "#BB8FCE",  # Purple
        "exit": "#85C1E9",  # Light Blue
    }

    node_colors = [
        category_colors.get(sections[section]["category"], "#CCCCCC")
        for section in all_sections
    ]

    # Create Sankey diagram
    fig = go.Figure(
        data=[
            go.Sankey(
                node=dict(
                    pad=15,
                    thickness=20,
                    line=dict(color="black", width=0.5),
                    label=section_names,
                    color=node_colors,
                    x=[
                        (
                            0.1
                            if "entrance" in sections[s]["name"].lower()
                            else (
                                0.9
                                if "checkout" in sections[s]["name"].lower()
                                else 0.3 + (i % 3) * 0.2
                            )
                        )
                        for i, s in enumerate(all_sections)
                    ],
                    y=[0.1 + (i % 10) * 0.08 for i in range(len(all_sections))],
                ),
                link=dict(
                    source=source_indices,
                    target=target_indices,
                    value=flow_values,
                    color=["rgba(255,182,193,0.4)" for _ in flow_values],
                ),
            )
        ]
    )

    fig.update_layout(
        title_text="Customer Flow Between Store Sections",
        title_x=0.5,
        font_size=12,
        height=600,
    )

    return fig


def create_product_association_sankey(
    product_categories, product_flows, min_strength=0.3
):
    """Create Sankey diagram for product associations."""
    # Filter by minimum strength
    filtered_flows = [(p1, p2, s) for p1, p2, s in product_flows if s >= min_strength]

    # Get all unique products
    all_products = set()
    for p1, p2, _ in filtered_flows:
        all_products.add(p1)
        all_products.add(p2)

    all_products = sorted(list(all_products))
    product_to_idx = {product: i for i, product in enumerate(all_products)}

    # Create flows
    source_indices = []
    target_indices = []
    flow_values = []

    for product1, product2, strength in filtered_flows:
        source_indices.append(product_to_idx[product1])
        target_indices.append(product_to_idx[product2])
        flow_values.append(strength * 100)  # Scale for visualization

    # Assign colors based on product categories
    product_colors = {}
    category_color_map = {
        "Dairy": "#3498db",
        "Produce": "#2ecc71",
        "Meat": "#e74c3c",
        "Beverages": "#9b59b6",
        "Snacks": "#f39c12",
        "Bakery": "#e67e22",
        "Pantry": "#95a5a6",
    }

    for category, products in product_categories.items():
        for product in products:
            product_colors[product] = category_color_map.get(category, "#34495e")

    node_colors = [product_colors.get(product, "#34495e") for product in all_products]

    # Create Sankey diagram
    fig = go.Figure(
        data=[
            go.Sankey(
                node=dict(
                    pad=15,
                    thickness=20,
                    line=dict(color="black", width=0.5),
                    label=all_products,
                    color=node_colors,
                ),
                link=dict(
                    source=source_indices,
                    target=target_indices,
                    value=flow_values,
                    color=["rgba(135,206,235,0.4)" for _ in flow_values],
                ),
            )
        ]
    )

    fig.update_layout(
        title_text="Product Association Flows", title_x=0.5, font_size=10, height=700
    )

    return fig


def create_category_to_section_flow(sections, product_categories):
    """Create flow from product categories to store sections."""
    # Mock assignment of categories to sections
    category_section_mapping = {
        "Dairy": ["dairy_milk_eggs", "dairy_cheese_yogurt"],
        "Produce": ["produce_fresh", "produce_packaged"],
        "Meat": ["meat_fresh", "meat_deli"],
        "Beverages": ["center_aisle_1", "dairy_milk_eggs"],
        "Snacks": ["center_aisle_1", "checkout"],
        "Bakery": ["bakery"],
        "Pantry": ["center_aisle_2"],
        "Household": ["center_aisle_3"],
        "Frozen": ["frozen"],
    }

    # Build flows
    flows = []
    for category, section_list in category_section_mapping.items():
        for section in section_list:
            if section in sections:
                # Mock strength based on category fit
                strength = np.random.uniform(0.6, 1.0)
                flows.append((category, sections[section]["name"], strength))

    # Create nodes
    categories = list(category_section_mapping.keys())
    section_names = list(set(sections[s]["name"] for s in sections.keys()))
    all_nodes = categories + section_names

    # Create node mapping
    node_to_idx = {node: i for i, node in enumerate(all_nodes)}

    # Build Sankey data
    source_indices = []
    target_indices = []
    flow_values = []

    for category, section_name, strength in flows:
        source_indices.append(node_to_idx[category])
        target_indices.append(node_to_idx[section_name])
        flow_values.append(strength * 100)

    # Create colors
    category_colors = [
        "#FF6B6B",
        "#4ECDC4",
        "#45B7D1",
        "#FFA07A",
        "#98D8C8",
        "#F7DC6F",
        "#BB8FCE",
        "#85C1E9",
        "#DDA0DD",
    ]
    section_colors = ["#E8E8E8"] * len(section_names)
    node_colors = category_colors[: len(categories)] + section_colors

    # Create Sankey
    fig = go.Figure(
        data=[
            go.Sankey(
                node=dict(
                    pad=15,
                    thickness=20,
                    line=dict(color="black", width=0.5),
                    label=all_nodes,
                    color=node_colors,
                    x=[
                        0.1 if i < len(categories) else 0.9
                        for i in range(len(all_nodes))
                    ],
                    y=[i * (1.0 / len(all_nodes)) for i in range(len(all_nodes))],
                ),
                link=dict(
                    source=source_indices,
                    target=target_indices,
                    value=flow_values,
                    color=["rgba(0,123,255,0.3)" for _ in flow_values],
                ),
            )
        ]
    )

    fig.update_layout(
        title_text="Product Category to Section Assignment",
        title_x=0.5,
        font_size=11,
        height=600,
    )

    return fig


def analyze_flow_metrics(customer_journeys, sections):
    """Analyze flow metrics and create summary statistics."""
    metrics = {}

    # Journey length analysis
    journey_lengths = [j["journey_length"] for j in customer_journeys]
    metrics["avg_journey_length"] = np.mean(journey_lengths)
    metrics["median_journey_length"] = np.median(journey_lengths)

    # Section visit frequency
    section_visits = defaultdict(int)
    for journey in customer_journeys:
        for section in journey["journey"]:
            section_visits[section] += 1

    metrics["most_visited_section"] = max(section_visits, key=section_visits.get)
    metrics["least_visited_section"] = min(section_visits, key=section_visits.get)

    # Flow efficiency (sections visited / journey length)
    unique_sections = [j["sections_visited"] for j in customer_journeys]
    total_sections = [j["journey_length"] for j in customer_journeys]
    efficiency_scores = [
        u / t if t > 0 else 0 for u, t in zip(unique_sections, total_sections)
    ]
    metrics["avg_flow_efficiency"] = np.mean(efficiency_scores)

    # Time analysis
    shopping_times = [j["total_time_minutes"] for j in customer_journeys]
    metrics["avg_shopping_time"] = np.mean(shopping_times)

    return metrics, section_visits


def main():
    # Sidebar controls
    st.sidebar.header("🎛️ Flow Analysis Controls")

    # Load data
    if st.session_state.flow_data is None:
        with st.spinner("Generating flow analysis data..."):
            sections, flow_patterns, customer_journeys = generate_mock_flow_data()
            product_categories, product_flows = generate_product_association_flows()
            st.session_state.flow_data = {
                "sections": sections,
                "flow_patterns": flow_patterns,
                "customer_journeys": customer_journeys,
                "product_categories": product_categories,
                "product_flows": product_flows,
            }

    data = st.session_state.flow_data

    # Analysis controls
    flow_type = st.sidebar.selectbox(
        "Analysis Type",
        options=[
            "Customer Section Flow",
            "Product Associations",
            "Category-Section Mapping",
            "Flow Metrics Dashboard",
        ],
    )

    min_flow_strength = st.sidebar.slider(
        "Minimum Flow Strength",
        min_value=0.1,
        max_value=1.0,
        value=0.3,
        step=0.1,
        help="Filter flows below this strength threshold",
    )

    # Main content based on selected analysis
    if flow_type == "Customer Section Flow":
        st.header("🚶 Customer Flow Between Sections")
        st.markdown(
            "This Sankey diagram shows how customers move between different store sections"
        )

        # Create and display Sankey
        fig = create_customer_flow_sankey(
            data["sections"], data["flow_patterns"], min_flow_strength
        )
        st.plotly_chart(fig, use_container_width=True)

        # Flow insights
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🎯 Key Insights")
            insights = [
                "💫 **Entrance → Fresh Produce**: Strongest initial flow (70%)",
                "🥛 **Dairy Section**: High checkout conversion (80%)",
                "🛒 **Center Aisles**: Strong sequential movement pattern",
                "✅ **Checkout**: Ultimate destination for most journeys",
            ]
            for insight in insights:
                st.markdown(insight)

        with col2:
            st.subheader("📊 Flow Statistics")
            # Calculate flow statistics
            total_flows = len(data["flow_patterns"])
            strong_flows = len([f for f in data["flow_patterns"] if f[2] >= 0.5])
            avg_flow_strength = np.mean([f[2] for f in data["flow_patterns"]])

            st.metric("Total Flow Connections", total_flows)
            st.metric("Strong Flows (>50%)", strong_flows)
            st.metric("Average Flow Strength", f"{avg_flow_strength:.3f}")

    elif flow_type == "Product Associations":
        st.header("🔗 Product Association Network")
        st.markdown(
            "Sankey diagram showing which products are frequently bought together"
        )

        # Filter controls
        col1, col2 = st.columns(2)
        with col1:
            selected_categories = st.multiselect(
                "Filter by Categories",
                options=list(data["product_categories"].keys()),
                default=list(data["product_categories"].keys())[:4],
            )

        with col2:
            min_association_strength = st.slider(
                "Minimum Association Strength",
                min_value=0.2,
                max_value=0.9,
                value=0.4,
                step=0.1,
            )

        # Filter product flows by selected categories
        if selected_categories:
            filtered_products = set()
            for category in selected_categories:
                filtered_products.update(data["product_categories"][category])

            filtered_flows = [
                (p1, p2, s)
                for p1, p2, s in data["product_flows"]
                if p1 in filtered_products and p2 in filtered_products
            ]

            filtered_categories = {
                k: v
                for k, v in data["product_categories"].items()
                if k in selected_categories
            }
        else:
            filtered_flows = data["product_flows"]
            filtered_categories = data["product_categories"]

        # Create and display Sankey
        fig = create_product_association_sankey(
            filtered_categories, filtered_flows, min_association_strength
        )
        st.plotly_chart(fig, use_container_width=True)

        # Association insights
        st.subheader("🎯 Strong Associations")
        strong_associations = [
            (p1, p2, s) for p1, p2, s in filtered_flows if s >= min_association_strength
        ]
        strong_associations.sort(key=lambda x: x[2], reverse=True)

        if strong_associations:
            for i, (p1, p2, strength) in enumerate(strong_associations[:10]):
                st.write(f"{i+1}. **{p1}** ↔ **{p2}**: {strength:.3f}")
        else:
            st.info("No associations found above the selected threshold")

    elif flow_type == "Category-Section Mapping":
        st.header("📍 Category to Section Flow")
        st.markdown("How product categories are distributed across store sections")

        # Create and display Sankey
        fig = create_category_to_section_flow(
            data["sections"], data["product_categories"]
        )
        st.plotly_chart(fig, use_container_width=True)

        # Section utilization
        st.subheader("📊 Section Utilization")
        section_util_data = []
        for section_id, section_info in data["sections"].items():
            # Mock utilization calculation
            utilization = section_info["traffic_weight"] * np.random.uniform(0.7, 1.0)
            section_util_data.append(
                {
                    "Section": section_info["name"],
                    "Traffic Weight": section_info["traffic_weight"],
                    "Utilization": utilization,
                    "Category": section_info["category"],
                }
            )

        util_df = pd.DataFrame(section_util_data)

        # Utilization chart
        fig = px.bar(
            util_df,
            x="Section",
            y="Utilization",
            color="Category",
            title="Section Utilization by Category",
        )
        fig.update_xaxis(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)

    else:  # Flow Metrics Dashboard
        st.header("📈 Flow Metrics Dashboard")
        st.markdown("Comprehensive analytics of customer flow patterns")

        # Calculate metrics
        metrics, section_visits = analyze_flow_metrics(
            data["customer_journeys"], data["sections"]
        )

        # Key metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Avg Journey Length", f"{metrics['avg_journey_length']:.1f}")
            st.metric(
                "Median Journey Length", f"{metrics['median_journey_length']:.1f}"
            )

        with col2:
            st.metric(
                "Most Visited Section",
                data["sections"][metrics["most_visited_section"]]["name"],
            )
            st.metric(
                "Least Visited Section",
                data["sections"][metrics["least_visited_section"]]["name"],
            )

        with col3:
            st.metric("Flow Efficiency", f"{metrics['avg_flow_efficiency']:.3f}")
            st.metric("Avg Shopping Time", f"{metrics['avg_shopping_time']:.1f} min")

        with col4:
            st.metric("Total Customers", len(data["customer_journeys"]))
            st.metric("Total Sections", len(data["sections"]))

        # Charts
        col1, col2 = st.columns(2)

        with col1:
            # Journey length distribution
            journey_lengths = [j["journey_length"] for j in data["customer_journeys"]]
            fig = px.histogram(
                x=journey_lengths,
                title="Journey Length Distribution",
                labels={"x": "Journey Length (sections)", "y": "Number of Customers"},
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Section popularity
            section_names = [data["sections"][s]["name"] for s in section_visits.keys()]
            visit_counts = list(section_visits.values())

            fig = px.bar(
                x=section_names, y=visit_counts, title="Section Visit Frequency"
            )
            fig.update_xaxis(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)

        # Detailed journey analysis
        st.subheader("🔍 Journey Analysis")

        # Sample journeys
        sample_journeys = data["customer_journeys"][:5]
        for i, journey in enumerate(sample_journeys):
            with st.expander(f"Customer {journey['customer_id']} Journey"):
                journey_sections = [
                    data["sections"][s]["name"] for s in journey["journey"]
                ]
                st.write(f"**Path**: {' → '.join(journey_sections)}")
                st.write(f"**Sections Visited**: {journey['sections_visited']}")
                st.write(f"**Total Time**: {journey['total_time_minutes']:.1f} minutes")

    # Export functionality
    st.sidebar.markdown("---")
    st.sidebar.subheader("📥 Export Options")

    if st.sidebar.button("💾 Export Flow Data"):
        # Prepare export data
        export_data = {
            "sections": data["sections"],
            "flow_patterns": data["flow_patterns"],
            "analysis_timestamp": datetime.now().isoformat(),
            "parameters": {
                "min_flow_strength": min_flow_strength,
                "analysis_type": flow_type,
            },
        }

        st.sidebar.download_button(
            label="📊 Download JSON",
            data=pd.Series(export_data).to_json(),
            file_name=f"flow_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
        )

    # Tips and help
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 💡 Tips")
    st.sidebar.markdown(
        "- **Thicker flows** = Higher customer traffic\n"
        "- **Node colors** represent section categories\n"
        "- **Hover** over flows for detailed values\n"
        "- **Adjust filters** to focus on specific patterns"
    )


if __name__ == "__main__":
    main()
