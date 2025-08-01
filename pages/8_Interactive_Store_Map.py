"""
Interactive Store Map Page

This page provides an interactive 2D store map with product placement overlays,
traffic flow analysis, and optimization recommendations using Plotly.
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime, timedelta
import requests
from typing import Dict, List, Any

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from layout_optimizer import SupermarketLayoutOptimizer
    from models import Product
except ImportError as e:
    st.warning(f"Import warning: {e}")

# Page configuration
st.set_page_config(
    page_title="Interactive Store Map",
    page_icon="🗺️",
    layout="wide"
)

# Custom CSS for enhanced styling
st.markdown("""
<style>
.map-container {
    border: 2px solid #ddd;
    border-radius: 8px;
    padding: 10px;
    margin: 10px 0;
    background-color: #f9f9f9;
}

.section-legend {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    margin: 10px 0;
}

.legend-item {
    display: flex;
    align-items: center;
    gap: 5px;
    padding: 5px 10px;
    border-radius: 15px;
    background-color: white;
    border: 1px solid #ddd;
    font-size: 0.8rem;
}

.color-box {
    width: 15px;
    height: 15px;
    border-radius: 3px;
}

.metric-card {
    background-color: #ffffff;
    border: 1px solid #ddd;
    border-radius: 8px;
    padding: 15px;
    margin: 5px 0;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.optimization-suggestion {
    background-color: #f8f9fa;
    border-left: 4px solid #1f77b4;
    padding: 15px;
    margin: 10px 0;
    border-radius: 0 8px 8px 0;
    border: 1px solid #dee2e6;
    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    color: #212529 !important;
}

.optimization-suggestion h4 {
    color: inherit !important;
    margin-bottom: 10px !important;
}

.optimization-suggestion p {
    color: #495057 !important;
    margin-bottom: 8px !important;
    line-height: 1.5;
}

@media (max-width: 768px) {
    .map-container {
        padding: 5px;
    }
    
    .section-legend {
        flex-direction: column;
    }
}
</style>
""", unsafe_allow_html=True)

def generate_store_layout():
    """Generate a sample store layout with sections and coordinates."""
    
    # Define store sections with coordinates
    sections = {
        'entrance': {'x': 5, 'y': 1, 'width': 3, 'height': 1, 'color': '#2E8B57', 'category': 'Entry'},
        'produce': {'x': 1, 'y': 3, 'width': 6, 'height': 3, 'color': '#32CD32', 'category': 'Fresh'},
        'dairy': {'x': 8, 'y': 1, 'width': 4, 'height': 2, 'color': '#87CEEB', 'category': 'Refrigerated'},
        'meat': {'x': 8, 'y': 4, 'width': 4, 'height': 3, 'color': '#CD5C5C', 'category': 'Fresh'},
        'bakery': {'x': 1, 'y': 7, 'width': 3, 'height': 2, 'color': '#DEB887', 'category': 'Fresh'},
        'deli': {'x': 5, 'y': 7, 'width': 3, 'height': 2, 'color': '#F4A460', 'category': 'Fresh'},
        'frozen': {'x': 9, 'y': 8, 'width': 3, 'height': 3, 'color': '#B0E0E6', 'category': 'Frozen'},
        'beverages': {'x': 1, 'y': 10, 'width': 4, 'height': 2, 'color': '#4169E1', 'category': 'Beverages'},
        'snacks': {'x': 6, 'y': 10, 'width': 3, 'height': 2, 'color': '#FFD700', 'category': 'Packaged'},
        'household': {'x': 1, 'y': 13, 'width': 5, 'height': 2, 'color': '#9370DB', 'category': 'Non-Food'},
        'pharmacy': {'x': 7, 'y': 13, 'width': 3, 'height': 2, 'color': '#FF69B4', 'category': 'Health'},
        'checkout': {'x': 11, 'y': 13, 'width': 2, 'height': 3, 'color': '#696969', 'category': 'Checkout'}
    }
    
    return sections

def generate_product_placements():
    """Generate sample product placements within sections."""
    
    np.random.seed(42)
    
    placements = []
    products = [
        # Produce
        ('PROD_001', 'Bananas', 'produce', 2.5, 4.5, 0.85),
        ('PROD_002', 'Apples', 'produce', 4.0, 4.0, 0.92),
        ('PROD_003', 'Lettuce', 'produce', 3.0, 5.0, 0.78),
        ('PROD_004', 'Tomatoes', 'produce', 5.5, 4.5, 0.88),
        
        # Dairy
        ('PROD_005', 'Milk', 'dairy', 9.0, 1.5, 0.95),
        ('PROD_006', 'Cheese', 'dairy', 10.5, 1.5, 0.82),
        ('PROD_007', 'Yogurt', 'dairy', 11.0, 2.0, 0.79),
        
        # Meat
        ('PROD_008', 'Chicken Breast', 'meat', 9.5, 5.0, 0.91),
        ('PROD_009', 'Ground Beef', 'meat', 10.5, 5.5, 0.87),
        ('PROD_010', 'Salmon', 'meat', 11.0, 6.0, 0.84),
        
        # Beverages
        ('PROD_011', 'Orange Juice', 'beverages', 2.0, 11.0, 0.86),
        ('PROD_012', 'Soda', 'beverages', 3.5, 11.0, 0.72),
        ('PROD_013', 'Water', 'beverages', 4.5, 11.5, 0.90),
        
        # Snacks
        ('PROD_014', 'Chips', 'snacks', 7.0, 11.0, 0.75),
        ('PROD_015', 'Cookies', 'snacks', 8.0, 11.5, 0.68),
        
        # Household
        ('PROD_016', 'Detergent', 'household', 3.0, 14.0, 0.82),
        ('PROD_017', 'Paper Towels', 'household', 5.0, 14.0, 0.88),
    ]
    
    for product_id, name, section, x, y, performance in products:
        placements.append({
            'product_id': product_id,
            'product_name': name,
            'section': section,
            'x': x,
            'y': y,
            'performance_score': performance,
            'weekly_sales': np.random.randint(50, 500),
            'inventory_turns': np.random.uniform(2.0, 12.0),
            'profit_margin': np.random.uniform(0.15, 0.45)
        })
    
    return pd.DataFrame(placements)

def generate_traffic_flow_data():
    """Generate sample customer traffic flow data."""
    
    np.random.seed(42)
    
    # Generate traffic heatmap data
    x_coords = np.arange(0, 14, 0.5)
    y_coords = np.arange(0, 16, 0.5)
    
    traffic_data = []
    
    for x in x_coords:
        for y in y_coords:
            # Higher traffic near entrance, produce, dairy, and checkout
            base_traffic = 10
            
            # Entrance area
            if 4 <= x <= 8 and 0 <= y <= 2:
                base_traffic += 40
            
            # Main aisles (horizontal)
            if y in [6, 9, 12]:
                base_traffic += 20
            
            # Main aisles (vertical)
            if x in [7, 10]:
                base_traffic += 15
            
            # Popular sections
            if 1 <= x <= 6 and 3 <= y <= 5:  # Produce
                base_traffic += 25
            elif 8 <= x <= 11 and 1 <= y <= 2:  # Dairy
                base_traffic += 30
            elif 11 <= x <= 13 and 13 <= y <= 15:  # Checkout
                base_traffic += 35
            
            # Add some randomness
            traffic = max(0, base_traffic + np.random.normal(0, 10))
            
            traffic_data.append({
                'x': x,
                'y': y,
                'traffic_count': traffic
            })
    
    return pd.DataFrame(traffic_data)

def create_store_map(sections, placements_df, traffic_df, view_mode='layout'):
    """Create interactive store map with different view modes."""
    
    fig = go.Figure()
    
    if view_mode == 'traffic':
        # Traffic heatmap
        fig.add_trace(go.Heatmap(
            x=traffic_df['x'],
            y=traffic_df['y'],
            z=traffic_df['traffic_count'],
            colorscale='YlOrRd',
            name='Traffic Density',
            opacity=0.7,
            showscale=True,
            colorbar=dict(title="Traffic Count")
        ))
    
    # Add section rectangles
    for section_name, section_data in sections.items():
        x0, y0 = section_data['x'], section_data['y']
        x1, y1 = x0 + section_data['width'], y0 + section_data['height']
        
        # Section rectangle
        fig.add_shape(
            type="rect",
            x0=x0, y0=y0, x1=x1, y1=y1,
            fillcolor=section_data['color'],
            opacity=0.3 if view_mode == 'traffic' else 0.6,
            line=dict(color=section_data['color'], width=2),
        )
        
        # Section label
        fig.add_annotation(
            x=(x0 + x1) / 2,
            y=(y0 + y1) / 2,
            text=section_name.title(),
            showarrow=False,
            font=dict(size=10, color='black'),
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='gray',
            borderwidth=1
        )
    
    # Add product placements
    if view_mode in ['layout', 'performance']:
        colors = placements_df['performance_score'] if view_mode == 'performance' else ['blue'] * len(placements_df)
        
        fig.add_trace(go.Scatter(
            x=placements_df['x'],
            y=placements_df['y'],
            mode='markers',
            marker=dict(
                size=12,
                color=colors,
                colorscale='RdYlGn' if view_mode == 'performance' else None,
                showscale=view_mode == 'performance',
                colorbar=dict(title="Performance Score") if view_mode == 'performance' else None,
                line=dict(width=1, color='white')
            ),
            text=placements_df['product_name'],
            hovertemplate='<b>%{text}</b><br>' +
                         'Location: (%{x}, %{y})<br>' +
                         'Performance: %{marker.color:.2f}<br>' +
                         '<extra></extra>',
            name='Products'
        ))
    
    # Layout configuration
    fig.update_layout(
        title=f"Store Layout - {view_mode.title()} View",
        xaxis=dict(
            title="X Coordinate (feet)",
            range=[-1, 14],
            constrain="domain"
        ),
        yaxis=dict(
            title="Y Coordinate (feet)",
            range=[-1, 17],
            scaleanchor="x",
            scaleratio=1
        ),
        showlegend=True,
        height=600,
        margin=dict(l=0, r=0, t=40, b=0)
    )
    
    return fig

def calculate_optimization_metrics(sections, placements_df, traffic_df):
    """Calculate optimization metrics for the store layout."""
    
    metrics = {}
    
    # Overall performance
    metrics['avg_performance'] = placements_df['performance_score'].mean()
    metrics['total_weekly_sales'] = placements_df['weekly_sales'].sum()
    metrics['avg_profit_margin'] = placements_df['profit_margin'].mean()
    
    # Section-wise performance
    section_performance = placements_df.groupby('section').agg({
        'performance_score': 'mean',
        'weekly_sales': 'sum',
        'profit_margin': 'mean'
    }).round(3)
    
    # Traffic efficiency
    high_traffic_areas = traffic_df[traffic_df['traffic_count'] > traffic_df['traffic_count'].quantile(0.8)]
    metrics['high_traffic_coverage'] = len(high_traffic_areas) / len(traffic_df)
    
    # Optimization opportunities
    low_performing_products = placements_df[placements_df['performance_score'] < 0.8]
    metrics['optimization_opportunities'] = len(low_performing_products)
    
    return metrics, section_performance

def generate_optimization_suggestions(placements_df, traffic_df, sections):
    """Generate optimization suggestions based on current layout."""
    
    suggestions = []
    
    # Find underperforming products
    low_performers = placements_df[placements_df['performance_score'] < 0.8]
    
    for _, product in low_performers.iterrows():
        # Check if product is in low-traffic area
        nearby_traffic = traffic_df[
            (abs(traffic_df['x'] - product['x']) <= 1) & 
            (abs(traffic_df['y'] - product['y']) <= 1)
        ]['traffic_count'].mean()
        
        if nearby_traffic < traffic_df['traffic_count'].median():
            suggestions.append({
                'type': 'Relocation',
                'priority': 'High',
                'product': product['product_name'],
                'current_location': f"({product['x']:.1f}, {product['y']:.1f})",
                'issue': 'Low performance in low-traffic area',
                'suggestion': 'Move to high-traffic zone near entrance or main aisles',
                'expected_impact': '+15-25% sales increase'
            })
    
    # Cross-selling opportunities
    high_performers = placements_df[placements_df['performance_score'] > 0.9]
    if len(high_performers) > 0:
        suggestions.append({
            'type': 'Cross-selling',
            'priority': 'Medium',
            'product': 'Complementary items',
            'current_location': 'Various',
            'issue': 'Missing cross-selling opportunities',
            'suggestion': f'Place complementary products near {high_performers.iloc[0]["product_name"]}',
            'expected_impact': '+10-15% basket size increase'
        })
    
    # Traffic flow optimization
    suggestions.append({
        'type': 'Traffic Flow',
        'priority': 'Medium',
        'product': 'Aisle layout',
        'current_location': 'Store-wide',
        'issue': 'Uneven traffic distribution',
        'suggestion': 'Create more intuitive pathways and reduce bottlenecks',
        'expected_impact': '+5-10% customer satisfaction'
    })
    
    return suggestions

# Main app
def main():
    st.title("🗺️ Interactive Store Map & Layout Optimizer")
    st.markdown("Visualize store layout, analyze traffic patterns, and optimize product placements")
    
    # Sidebar controls
    st.sidebar.header("🎛️ Map Controls")
    
    view_mode = st.sidebar.selectbox(
        "Select View Mode",
        ['layout', 'performance', 'traffic'],
        format_func=lambda x: {
            'layout': '🏪 Store Layout',
            'performance': '📊 Product Performance',
            'traffic': '🚶 Traffic Flow'
        }[x]
    )
    
    show_analytics = st.sidebar.checkbox("Show Analytics Panel", value=True)
    show_suggestions = st.sidebar.checkbox("Show Optimization Suggestions", value=True)
    
    # Generate data
    sections = generate_store_layout()
    placements_df = generate_product_placements()
    traffic_df = generate_traffic_flow_data()
    
    # Main content area
    col1, col2 = st.columns([3, 1] if show_analytics else [1])
    
    with col1:
        st.markdown('<div class="map-container">', unsafe_allow_html=True)
        
        # Create and display map
        fig = create_store_map(sections, placements_df, traffic_df, view_mode)
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Section legend
        if view_mode == 'layout':
            st.markdown("### 📍 Section Legend")
            legend_html = '<div class="section-legend">'
            
            for section_name, section_data in sections.items():
                legend_html += f'''
                <div class="legend-item">
                    <div class="color-box" style="background-color: {section_data['color']};"></div>
                    <span>{section_name.title()} ({section_data['category']})</span>
                </div>
                '''
            
            legend_html += '</div>'
            st.markdown(legend_html, unsafe_allow_html=True)
    
    if show_analytics:
        with col2:
            st.markdown("### 📊 Analytics Panel")
            
            # Calculate metrics
            metrics, section_performance = calculate_optimization_metrics(sections, placements_df, traffic_df)
            
            # Overall metrics
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Avg Performance", f"{metrics['avg_performance']:.3f}", 
                     delta=f"{(metrics['avg_performance'] - 0.8):.3f}")
            st.metric("Weekly Sales", f"${metrics['total_weekly_sales']:,}", 
                     delta="12.5%")
            st.metric("Profit Margin", f"{metrics['avg_profit_margin']:.1%}", 
                     delta="2.3%")
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Section performance
            st.markdown("### 🏪 Section Performance")
            section_performance_display = section_performance.reset_index()
            st.dataframe(section_performance_display, use_container_width=True)
            
            # Top performers
            st.markdown("### 🌟 Top Performers")
            top_products = placements_df.nlargest(3, 'performance_score')[['product_name', 'performance_score', 'section']]
            for _, product in top_products.iterrows():
                st.markdown(f"• **{product['product_name']}** ({product['section']}) - {product['performance_score']:.3f}")
    
    # Optimization suggestions
    if show_suggestions:
        st.markdown("### 💡 Optimization Suggestions")
        
        suggestions = generate_optimization_suggestions(placements_df, traffic_df, sections)
        
        for suggestion in suggestions:
            priority_color = {
                'High': '#ff4444',
                'Medium': '#ffaa00', 
                'Low': '#44ff44'
            }[suggestion['priority']]
            
            st.markdown(f'''
            <div class="optimization-suggestion">
                <h4 style="color: {priority_color}; margin-top: 0;">
                    {suggestion['type']} - {suggestion['priority']} Priority
                </h4>
                <p><strong>Product:</strong> {suggestion['product']}</p>
                <p><strong>Issue:</strong> {suggestion['issue']}</p>
                <p><strong>Suggestion:</strong> {suggestion['suggestion']}</p>
                <p><strong>Expected Impact:</strong> {suggestion['expected_impact']}</p>
            </div>
            ''', unsafe_allow_html=True)
    
    # Advanced analytics
    st.markdown("### 📈 Advanced Analytics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Performance distribution
        fig_perf = px.histogram(
            placements_df,
            x='performance_score',
            title='Product Performance Distribution',
            nbins=10
        )
        fig_perf.update_layout(height=300)
        st.plotly_chart(fig_perf, use_container_width=True)
    
    with col2:
        # Sales by section
        section_sales = placements_df.groupby('section')['weekly_sales'].sum().reset_index()
        fig_sales = px.pie(
            section_sales,
            values='weekly_sales',
            names='section',
            title='Sales Distribution by Section'
        )
        fig_sales.update_layout(height=300)
        st.plotly_chart(fig_sales, use_container_width=True)
    
    # Footer with instructions
    st.markdown("---")
    st.markdown("""
    ### 📋 How to Use This Map:
    - **Layout View**: Shows store sections and product locations
    - **Performance View**: Color-codes products by performance score (green = high, red = low)
    - **Traffic View**: Displays customer traffic heatmap overlaid on store layout
    - Click and drag to pan, use mouse wheel to zoom
    - Hover over products for detailed information
    """)

if __name__ == "__main__":
    main() 