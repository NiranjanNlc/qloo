"""
Shelf Placements Page

This page provides detailed analysis of product shelf placements with interactive
filtering, sorting, and recommendations for optimal product positioning.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os
from datetime import datetime, timedelta

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from layout_optimizer import SupermarketLayoutOptimizer, ProductLocation, SectionOptimizationResult
    from optimization_heuristics import HeuristicOptimizer, ConfigurationManager
    from models import Product
    from association_engine import AprioriAssociationEngine
except ImportError as e:
    st.error(f"Import error: {e}")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="Shelf Placements Analysis",
    page_icon="📋",
    layout="wide"
)

st.title("📋 Shelf Placements Analysis")
st.markdown("Interactive analysis of product shelf placements with optimization recommendations")

# Initialize session state
if 'placement_data' not in st.session_state:
    st.session_state.placement_data = None
if 'optimizer' not in st.session_state:
    st.session_state.optimizer = None


@st.cache_data
def generate_mock_placement_data():
    """Generate mock data for shelf placement analysis."""
    np.random.seed(42)
    
    # Categories and their typical zones
    categories = {
        'Dairy': ['dairy_cooler', 'center_aisles'],
        'Produce': ['entrance', 'produce_section'],
        'Meat': ['meat_deli', 'frozen_section'],
        'Beverages': ['center_aisles', 'dairy_cooler', 'checkout'],
        'Snacks': ['center_aisles', 'checkout'],
        'Bakery': ['bakery', 'entrance'],
        'Pantry': ['center_aisles'],
        'Frozen': ['frozen_section']
    }
    
    zones = ['entrance', 'dairy_cooler', 'meat_deli', 'center_aisles', 'checkout', 'bakery', 'produce_section', 'frozen_section']
    shelf_priorities = [1, 2, 3]  # 1=eye level, 2=reach level, 3=stoop level
    
    placement_data = []
    
    for product_id in range(1, 201):  # 200 products
        category = np.random.choice(list(categories.keys()))
        zone = np.random.choice(categories[category])
        shelf_priority = np.random.choice(shelf_priorities, p=[0.3, 0.5, 0.2])  # Bias toward eye/reach level
        
        # Generate realistic coordinates within zone
        zone_coords = {
            'entrance': (0.1, 0.3, 0.1, 0.9),
            'dairy_cooler': (0.7, 1.0, 0.0, 0.5),
            'meat_deli': (0.7, 1.0, 0.5, 1.0),
            'center_aisles': (0.3, 0.7, 0.0, 1.0),
            'checkout': (0.0, 0.2, 0.0, 0.3),
            'bakery': (0.0, 0.3, 0.7, 1.0),
            'produce_section': (0.0, 0.3, 0.1, 0.7),
            'frozen_section': (0.4, 0.7, 0.8, 1.0)
        }
        
        x_min, x_max, y_min, y_max = zone_coords[zone]
        x_coord = np.random.uniform(x_min, x_max)
        y_coord = np.random.uniform(y_min, y_max)
        
        # Calculate performance metrics
        base_score = np.random.uniform(0.3, 0.9)
        
        # Eye level gets boost
        shelf_boost = {1: 0.2, 2: 0.0, 3: -0.1}[shelf_priority]
        
        # High traffic zones get boost for popular products
        traffic_zones = ['entrance', 'checkout', 'dairy_cooler']
        traffic_boost = 0.1 if zone in traffic_zones else 0.0
        
        confidence_score = np.clip(base_score + shelf_boost + traffic_boost + np.random.normal(0, 0.05), 0, 1)
        
        # Association strength (mock)
        association_strength = np.random.uniform(0.1, 0.8)
        
        # Sales performance (mock)
        weekly_sales = np.random.randint(50, 500)
        
        # Cross-selling potential
        cross_sell_score = np.random.uniform(0.2, 0.9)
        
        placement_data.append({
            'product_id': product_id,
            'product_name': f'Product {product_id}',
            'category': category,
            'zone': zone,
            'x_coordinate': round(x_coord, 3),
            'y_coordinate': round(y_coord, 3),
            'shelf_priority': shelf_priority,
            'shelf_level': {1: 'Eye Level', 2: 'Reach Level', 3: 'Stoop Level'}[shelf_priority],
            'confidence_score': round(confidence_score, 3),
            'association_strength': round(association_strength, 3),
            'weekly_sales': weekly_sales,
            'cross_sell_score': round(cross_sell_score, 3),
            'last_updated': datetime.now() - timedelta(days=np.random.randint(0, 30))
        })
    
    return pd.DataFrame(placement_data)


def create_placement_summary(df):
    """Create summary statistics for placements."""
    summary = {
        'total_products': len(df),
        'categories': df['category'].nunique(),
        'zones': df['zone'].nunique(),
        'avg_confidence': df['confidence_score'].mean(),
        'high_confidence_products': len(df[df['confidence_score'] >= 0.8]),
        'eye_level_products': len(df[df['shelf_priority'] == 1]),
        'total_weekly_sales': df['weekly_sales'].sum(),
        'avg_association_strength': df['association_strength'].mean()
    }
    return summary


def create_interactive_datatable(df, filters):
    """Create interactive DataTable with filtering and sorting."""
    # Apply filters
    filtered_df = df.copy()
    
    if filters['categories']:
        filtered_df = filtered_df[filtered_df['category'].isin(filters['categories'])]
    
    if filters['zones']:
        filtered_df = filtered_df[filtered_df['zone'].isin(filters['zones'])]
    
    if filters['shelf_levels']:
        filtered_df = filtered_df[filtered_df['shelf_level'].isin(filters['shelf_levels'])]
    
    if filters['confidence_range']:
        min_conf, max_conf = filters['confidence_range']
        filtered_df = filtered_df[
            (filtered_df['confidence_score'] >= min_conf) & 
            (filtered_df['confidence_score'] <= max_conf)
        ]
    
    return filtered_df


def create_zone_heatmap(df):
    """Create heatmap showing product placement efficiency by zone."""
    zone_metrics = df.groupby('zone').agg({
        'confidence_score': 'mean',
        'weekly_sales': 'sum',
        'product_id': 'count',
        'association_strength': 'mean'
    }).round(3)
    
    zone_metrics.columns = ['Avg Confidence', 'Total Sales', 'Product Count', 'Avg Association']
    
    # Calculate zone efficiency score
    zone_metrics['Efficiency Score'] = (
        zone_metrics['Avg Confidence'] * 0.4 +
        (zone_metrics['Total Sales'] / zone_metrics['Total Sales'].max()) * 0.3 +
        (zone_metrics['Product Count'] / zone_metrics['Product Count'].max()) * 0.1 +
        zone_metrics['Avg Association'] * 0.2
    ).round(3)
    
    return zone_metrics


def create_shelf_level_analysis(df):
    """Analyze performance by shelf level."""
    shelf_analysis = df.groupby('shelf_level').agg({
        'confidence_score': ['mean', 'count'],
        'weekly_sales': 'mean',
        'association_strength': 'mean'
    }).round(3)
    
    shelf_analysis.columns = ['Avg Confidence', 'Product Count', 'Avg Weekly Sales', 'Avg Association']
    
    return shelf_analysis


def create_confidence_distribution_chart(df):
    """Create confidence score distribution chart."""
    fig = px.histogram(
        df, x='confidence_score', color='category',
        title='Confidence Score Distribution by Category',
        nbins=20,
        labels={'confidence_score': 'Confidence Score', 'count': 'Number of Products'}
    )
    
    fig.update_layout(
        showlegend=True,
        height=400,
        bargap=0.1
    )
    
    return fig


def create_placement_scatter(df):
    """Create scatter plot of product placements."""
    fig = px.scatter(
        df, x='x_coordinate', y='y_coordinate',
        color='confidence_score',
        size='weekly_sales',
        hover_data=['product_name', 'category', 'zone', 'shelf_level'],
        title='Product Placement Map (Confidence & Sales)',
        color_continuous_scale='viridis',
        size_max=15
    )
    
    fig.update_layout(
        xaxis_title='X Coordinate',
        yaxis_title='Y Coordinate',
        height=500
    )
    
    return fig


def create_optimization_recommendations(df):
    """Generate optimization recommendations."""
    recommendations = []
    
    # Low confidence products in high-traffic zones
    low_conf_high_traffic = df[
        (df['confidence_score'] < 0.6) & 
        (df['zone'].isin(['entrance', 'checkout', 'dairy_cooler']))
    ]
    
    if not low_conf_high_traffic.empty:
        recommendations.append({
            'type': 'Low Confidence in Prime Zones',
            'count': len(low_conf_high_traffic),
            'description': 'Products with low confidence scores in high-traffic zones',
            'action': 'Consider relocating or improving product positioning',
            'priority': 'High'
        })
    
    # High-performing products on lower shelves
    high_perf_low_shelf = df[
        (df['confidence_score'] > 0.8) & 
        (df['shelf_priority'] == 3)
    ]
    
    if not high_perf_low_shelf.empty:
        recommendations.append({
            'type': 'High Performers on Low Shelves',
            'count': len(high_perf_low_shelf),
            'description': 'High-confidence products placed on stoop level',
            'action': 'Move to eye or reach level for better visibility',
            'priority': 'Medium'
        })
    
    # Overcrowded zones
    zone_counts = df['zone'].value_counts()
    overcrowded_threshold = zone_counts.mean() + zone_counts.std()
    overcrowded_zones = zone_counts[zone_counts > overcrowded_threshold]
    
    if not overcrowded_zones.empty:
        recommendations.append({
            'type': 'Overcrowded Zones',
            'count': len(overcrowded_zones),
            'description': f'Zones with excessive product density: {", ".join(overcrowded_zones.index)}',
            'action': 'Redistribute products to less crowded zones',
            'priority': 'Medium'
        })
    
    # Under-performing categories
    category_performance = df.groupby('category')['confidence_score'].mean()
    low_perf_categories = category_performance[category_performance < 0.6]
    
    if not low_perf_categories.empty:
        recommendations.append({
            'type': 'Under-performing Categories',
            'count': len(low_perf_categories),
            'description': f'Categories with low average confidence: {", ".join(low_perf_categories.index)}',
            'action': 'Review category placement strategy and associations',
            'priority': 'Low'
        })
    
    return recommendations


# Main app
def main():
    # Sidebar filters
    st.sidebar.header("🔍 Filters")
    
    # Load data
    if st.session_state.placement_data is None:
        with st.spinner("Loading placement data..."):
            st.session_state.placement_data = generate_mock_placement_data()
    
    df = st.session_state.placement_data
    
    # Filter controls
    filters = {
        'categories': st.sidebar.multiselect(
            "Categories",
            options=df['category'].unique(),
            default=[]
        ),
        'zones': st.sidebar.multiselect(
            "Zones",
            options=df['zone'].unique(),
            default=[]
        ),
        'shelf_levels': st.sidebar.multiselect(
            "Shelf Levels",
            options=df['shelf_level'].unique(),
            default=[]
        ),
        'confidence_range': st.sidebar.slider(
            "Confidence Score Range",
            min_value=0.0,
            max_value=1.0,
            value=(0.0, 1.0),
            step=0.1
        )
    }
    
    # Apply filters
    filtered_df = create_interactive_datatable(df, filters)
    
    # Summary metrics
    st.header("📊 Placement Summary")
    summary = create_placement_summary(filtered_df)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Products", summary['total_products'])
        st.metric("Categories", summary['categories'])
    
    with col2:
        st.metric("Zones", summary['zones'])
        st.metric("Avg Confidence", f"{summary['avg_confidence']:.3f}")
    
    with col3:
        st.metric("High Confidence Products", summary['high_confidence_products'])
        st.metric("Eye Level Products", summary['eye_level_products'])
    
    with col4:
        st.metric("Total Weekly Sales", f"${summary['total_weekly_sales']:,}")
        st.metric("Avg Association Strength", f"{summary['avg_association_strength']:.3f}")
    
    # Interactive DataTable
    st.header("📋 Product Placement Details")
    
    # Sorting options
    col1, col2 = st.columns([3, 1])
    with col1:
        sort_column = st.selectbox(
            "Sort by",
            options=['confidence_score', 'weekly_sales', 'association_strength', 'product_name'],
            index=0
        )
    with col2:
        sort_ascending = st.checkbox("Ascending", value=False)
    
    # Sort and display data
    display_df = filtered_df.sort_values(sort_column, ascending=sort_ascending)
    
    # Format columns for display
    display_columns = [
        'product_id', 'product_name', 'category', 'zone', 'shelf_level',
        'confidence_score', 'association_strength', 'weekly_sales', 'cross_sell_score'
    ]
    
    # Color-code confidence scores
    def color_confidence(val):
        if val >= 0.8:
            return 'background-color: #d4edda'  # Green
        elif val >= 0.6:
            return 'background-color: #fff3cd'  # Yellow
        else:
            return 'background-color: #f8d7da'  # Red
    
    styled_df = display_df[display_columns].style.applymap(
        color_confidence, subset=['confidence_score']
    ).format({
        'confidence_score': '{:.3f}',
        'association_strength': '{:.3f}',
        'cross_sell_score': '{:.3f}',
        'weekly_sales': '${:,.0f}'
    })
    
    st.dataframe(styled_df, use_container_width=True, height=400)
    
    # Export options
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📊 Export to CSV"):
            csv = display_df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"shelf_placements_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    with col2:
        if st.button("📈 Generate Report"):
            # Generate detailed report
            report_data = {
                'summary': summary,
                'filtered_products': len(filtered_df),
                'timestamp': datetime.now().isoformat()
            }
            st.json(report_data)
    
    # Visualizations
    st.header("📈 Placement Analytics")
    
    tab1, tab2, tab3, tab4 = st.tabs(["Distribution", "Placement Map", "Zone Analysis", "Shelf Analysis"])
    
    with tab1:
        st.plotly_chart(create_confidence_distribution_chart(filtered_df), use_container_width=True)
        
        # Category performance table
        st.subheader("Category Performance")
        category_perf = filtered_df.groupby('category').agg({
            'confidence_score': 'mean',
            'weekly_sales': 'sum',
            'product_id': 'count'
        }).round(3)
        category_perf.columns = ['Avg Confidence', 'Total Sales', 'Product Count']
        st.dataframe(category_perf, use_container_width=True)
    
    with tab2:
        st.plotly_chart(create_placement_scatter(filtered_df), use_container_width=True)
        
        st.info("💡 Bubble size represents weekly sales, color represents confidence score")
    
    with tab3:
        st.subheader("Zone Efficiency Analysis")
        zone_metrics = create_zone_heatmap(filtered_df)
        
        # Zone efficiency heatmap
        fig = px.imshow(
            zone_metrics[['Efficiency Score']].T,
            labels=dict(x="Zone", y="Metric", color="Score"),
            title="Zone Efficiency Heatmap",
            color_continuous_scale="viridis"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.dataframe(zone_metrics, use_container_width=True)
    
    with tab4:
        st.subheader("Shelf Level Performance")
        shelf_analysis = create_shelf_level_analysis(filtered_df)
        st.dataframe(shelf_analysis, use_container_width=True)
        
        # Shelf level performance chart
        fig = px.bar(
            shelf_analysis.reset_index(),
            x='shelf_level',
            y='Avg Confidence',
            title='Average Confidence by Shelf Level',
            text='Avg Confidence'
        )
        fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
        st.plotly_chart(fig, use_container_width=True)
    
    # Optimization Recommendations
    st.header("🎯 Optimization Recommendations")
    
    recommendations = create_optimization_recommendations(filtered_df)
    
    if recommendations:
        for i, rec in enumerate(recommendations):
            priority_color = {
                'High': '🔴',
                'Medium': '🟡',
                'Low': '🟢'
            }
            
            with st.expander(f"{priority_color[rec['priority']]} {rec['type']} ({rec['count']} items)"):
                st.write(f"**Description:** {rec['description']}")
                st.write(f"**Recommended Action:** {rec['action']}")
                st.write(f"**Priority:** {rec['priority']}")
    else:
        st.success("🎉 No optimization issues found! Current placement strategy appears optimal.")
    
    # Real-time monitoring section
    st.header("⏱️ Real-time Monitoring")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Refresh Data"):
            st.session_state.placement_data = None
            st.experimental_rerun()
    
    with col2:
        auto_refresh = st.checkbox("Auto-refresh every 30 seconds")
        if auto_refresh:
            st.write("⏳ Auto-refresh enabled")
            # In a real app, this would set up automatic refresh
    
    # Footer
    st.markdown("---")
    st.markdown(
        "💡 **Tip:** Use filters in the sidebar to focus on specific categories, zones, or confidence ranges. "
        "Click on column headers in the data table to sort by different metrics."
    )


if __name__ == "__main__":
    main() 