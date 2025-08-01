"""
Combo Offers Page

This page provides an interactive interface for viewing, managing, and exporting
product combination offers with beautiful card layouts and comprehensive analytics.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
import sys
import os
from datetime import datetime, timedelta
from typing import List, Dict, Any

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from models import Combo, ComboGenerator, Product
    from price_api import PriceAPIStub, DiscountStrategy
    from association_engine import AprioriAssociationEngine
except ImportError as e:
    st.error(f"Import error: {e}")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="Combo Offers",
    page_icon="🎁",
    layout="wide"
)

# Custom CSS for beautiful cards
st.markdown("""
<style>
    .combo-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
        color: white !important;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        transition: transform 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .combo-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }
    
    .high-confidence {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
    }
    
    .medium-confidence {
        background: linear-gradient(135deg, #fcb045 0%, #fd1d1d 100%);
    }
    
    .low-confidence {
        background: linear-gradient(135deg, #bc4e9c 0%, #f80759 100%);
    }
    
    .combo-title {
        font-size: 1.3em;
        font-weight: bold;
        margin-bottom: 10px;
        color: white !important;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
    }
    
    .combo-metric {
        display: inline-block;
        background: rgba(255,255,255,0.3);
        border-radius: 20px;
        padding: 5px 15px;
        margin: 5px 5px 5px 0;
        font-size: 0.9em;
        color: white !important;
        font-weight: 500;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.2);
    }
    
    .discount-badge {
        background: #ff6b6b;
        color: white;
        padding: 8px 15px;
        border-radius: 25px;
        font-weight: bold;
        font-size: 1.1em;
        display: inline-block;
        margin: 10px 0;
    }
    
    .download-btn {
        background: linear-gradient(45deg, #ff6b6b, #ffa500);
        border: none;
        color: white;
        padding: 12px 25px;
        border-radius: 25px;
        font-weight: bold;
        cursor: pointer;
        transition: all 0.3s ease;
        margin: 10px 5px;
    }
    
    .download-btn:hover {
        transform: scale(1.05);
        box-shadow: 0 5px 15px rgba(255,107,107,0.4);
    }
    
    .stats-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        padding: 20px;
        color: white;
        margin: 20px 0;
    }
    
    .category-tag {
        background: rgba(255,255,255,0.3);
        border-radius: 15px;
        padding: 3px 10px;
        margin: 2px;
        font-size: 0.8em;
        display: inline-block;
        color: white !important;
        font-weight: 500;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.2);
    }
    
    /* Ensure all text in combo cards is visible */
    .combo-card * {
        color: white !important;
    }
    
    .combo-card strong {
        font-weight: bold !important;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
    }
</style>
""", unsafe_allow_html=True)

st.title("🎁 Combo Offers Management")
st.markdown("Create, manage, and export product combination offers with AI-powered recommendations")

# Initialize session state
if 'combo_data' not in st.session_state:
    st.session_state.combo_data = None
if 'combo_generator' not in st.session_state:
    st.session_state.combo_generator = None


@st.cache_data
def generate_mock_combo_data():
    """Generate comprehensive mock combo data."""
    np.random.seed(42)
    
    # Product categories and names
    product_categories = {
        'Dairy': {
            1: 'Whole Milk', 2: 'Greek Yogurt', 3: 'Cheddar Cheese', 
            4: 'Butter', 5: 'Cream Cheese', 6: 'Mozzarella'
        },
        'Produce': {
            11: 'Bananas', 12: 'Strawberries', 13: 'Spinach', 
            14: 'Tomatoes', 15: 'Avocados', 16: 'Bell Peppers'
        },
        'Meat': {
            21: 'Ground Beef', 22: 'Chicken Breast', 23: 'Salmon Fillet',
            24: 'Turkey Slices', 25: 'Bacon', 26: 'Pork Chops'
        },
        'Beverages': {
            31: 'Orange Juice', 32: 'Cola', 33: 'Sparkling Water',
            34: 'Coffee Beans', 35: 'Green Tea', 36: 'Energy Drink'
        },
        'Snacks': {
            41: 'Potato Chips', 42: 'Almonds', 43: 'Granola Bars',
            44: 'Crackers', 45: 'Popcorn', 46: 'Trail Mix'
        },
        'Bakery': {
            51: 'Whole Wheat Bread', 52: 'Croissants', 53: 'Bagels',
            54: 'Muffins', 55: 'Cookies', 56: 'Pizza Dough'
        }
    }
    
    # Create product lookup
    all_products = {}
    for category, products in product_categories.items():
        for pid, name in products.items():
            all_products[pid] = {'name': name, 'category': category}
    
    # Generate realistic combos with strong associations
    combo_templates = [
        # Breakfast combos
        {'products': [51, 4, 31], 'name': 'Morning Essentials', 'theme': 'breakfast'},
        {'products': [53, 5, 34], 'name': 'Bagel Coffee Combo', 'theme': 'breakfast'},
        {'products': [2, 12, 43], 'name': 'Healthy Breakfast', 'theme': 'breakfast'},
        
        # Dinner combos
        {'products': [21, 14, 3], 'name': 'Burger Night Special', 'theme': 'dinner'},
        {'products': [22, 13, 16], 'name': 'Healthy Dinner Kit', 'theme': 'dinner'},
        {'products': [23, 15, 33], 'name': 'Gourmet Salmon Meal', 'theme': 'dinner'},
        
        # Snack combos
        {'products': [41, 32, 55], 'name': 'Movie Night Pack', 'theme': 'snacks'},
        {'products': [42, 46, 35], 'name': 'Energy Boost Bundle', 'theme': 'snacks'},
        {'products': [44, 3, 36], 'name': 'Quick Bite Combo', 'theme': 'snacks'},
        
        # Family combos
        {'products': [1, 51, 25, 6], 'name': 'Family Breakfast', 'theme': 'family'},
        {'products': [21, 26, 52, 1], 'name': 'BBQ Family Pack', 'theme': 'family'},
        
        # Health combos
        {'products': [13, 15, 42], 'name': 'Super Greens Pack', 'theme': 'health'},
        {'products': [2, 12, 42], 'name': 'Protein Power Bowl', 'theme': 'health'},
        
        # Random combinations for variety
        {'products': [24, 53, 31], 'name': 'Lunch Special', 'theme': 'lunch'},
        {'products': [56, 6, 14], 'name': 'Pizza Night Kit', 'theme': 'dinner'},
    ]
    
    combos = []
    for i, template in enumerate(combo_templates):
        # Generate realistic metrics
        base_confidence = np.random.uniform(0.75, 0.95)
        theme_boost = {'breakfast': 0.05, 'dinner': 0.03, 'health': 0.07, 'family': 0.02}.get(template['theme'], 0)
        confidence = min(0.98, base_confidence + theme_boost + np.random.normal(0, 0.02))
        
        support = np.random.uniform(0.02, 0.08)
        lift = np.random.uniform(1.2, 2.5)
        
        # Calculate discount based on combo strength and theme
        base_discount = 15.0
        if template['theme'] == 'health':
            base_discount = 12.0  # Health combos get lower discounts
        elif template['theme'] == 'family':
            base_discount = 18.0  # Family packs get higher discounts
        
        discount_variation = np.random.uniform(-3, 3)
        expected_discount = max(5.0, min(25.0, base_discount + discount_variation))
        
        # Get category mix
        categories = []
        for pid in template['products']:
            if pid in all_products:
                categories.append(all_products[pid]['category'])
        
        combo = Combo(
            combo_id=f"combo_{i+1:03d}",
            name=template['name'],
            products=template['products'],
            confidence_score=round(confidence, 3),
            support=round(support, 4),
            lift=round(lift, 2),
            expected_discount_percent=round(expected_discount, 1),
            category_mix=list(set(categories)),
            created_at=datetime.now() - timedelta(days=np.random.randint(0, 30)),
            is_active=np.random.choice([True, False], p=[0.8, 0.2])
        )
        
        # Add metadata
        combo_dict = {
            'combo': combo,
            'theme': template['theme'],
            'product_names': [all_products[pid]['name'] for pid in template['products'] if pid in all_products],
            'estimated_revenue': np.random.uniform(25, 150),
            'popularity_score': confidence * lift * 0.5,
            'last_performance': np.random.uniform(0.6, 1.2)
        }
        
        combos.append(combo_dict)
    
    return combos, all_products


def get_confidence_class(confidence_score):
    """Get CSS class based on confidence score."""
    if confidence_score >= 0.8:
        return "high-confidence"
    elif confidence_score >= 0.6:
        return "medium-confidence"
    else:
        return "low-confidence"


def render_combo_card(combo_data, all_products):
    """Render a beautiful combo card."""
    combo = combo_data['combo']
    confidence_class = get_confidence_class(combo.confidence_score)
    
    # Product names string
    product_names = ", ".join(combo_data['product_names'])
    
    # Category tags
    category_tags = "".join([f'<span class="category-tag">{cat}</span>' for cat in combo.category_mix])
    
    # Status indicator
    status_icon = "🟢" if combo.is_active else "🔴"
    status_text = "Active" if combo.is_active else "Inactive"
    
    card_html = f"""
    <div class="combo-card {confidence_class}">
        <div class="combo-title">{status_icon} {combo.name}</div>
        
        <div style="margin: 10px 0;">
            <strong>Products:</strong> {product_names}
        </div>
        
        <div style="margin: 10px 0;">
            {category_tags}
        </div>
        
        <div style="margin: 15px 0;">
            <span class="combo-metric">Confidence: {combo.confidence_score:.3f}</span>
            <span class="combo-metric">Lift: {combo.lift:.2f}x</span>
            <span class="combo-metric">Support: {combo.support:.4f}</span>
        </div>
        
        <div class="discount-badge">
            {combo.expected_discount_percent:.1f}% OFF
        </div>
        
        <div style="margin-top: 15px; font-size: 0.9em;">
            <div>💰 Est. Revenue: ${combo_data['estimated_revenue']:.0f}</div>
            <div>⭐ Popularity: {combo_data['popularity_score']:.3f}</div>
            <div>📈 Performance: {combo_data['last_performance']:.1f}x</div>
            <div>📅 Created: {combo.created_at.strftime('%Y-%m-%d')}</div>
            <div>🆔 ID: {combo.combo_id}</div>
        </div>
    </div>
    """
    
    return card_html


def create_combo_analytics(combo_list):
    """Create analytics dashboard for combos."""
    if not combo_list:
        return None
    
    # Prepare data
    df_data = []
    for combo_data in combo_list:
        combo = combo_data['combo']
        df_data.append({
            'combo_id': combo.combo_id,
            'name': combo.name,
            'confidence': combo.confidence_score,
            'lift': combo.lift,
            'support': combo.support,
            'discount': combo.expected_discount_percent,
            'theme': combo_data['theme'],
            'revenue': combo_data['estimated_revenue'],
            'popularity': combo_data['popularity_score'],
            'active': combo.is_active,
            'category_count': len(combo.category_mix),
            'product_count': len(combo.products)
        })
    
    df = pd.DataFrame(df_data)
    return df


def export_combos_json(combo_list, format_type="detailed"):
    """Export combos to JSON format."""
    export_data = {
        'export_metadata': {
            'timestamp': datetime.now().isoformat(),
            'total_combos': len(combo_list),
            'format': format_type,
            'generated_by': 'Qloo Supermarket Optimizer'
        },
        'combos': []
    }
    
    for combo_data in combo_list:
        combo = combo_data['combo']
        
        if format_type == "detailed":
            combo_export = {
                'combo_id': combo.combo_id,
                'name': combo.name,
                'products': combo.products,
                'product_names': combo_data['product_names'],
                'confidence_score': combo.confidence_score,
                'support': combo.support,
                'lift': combo.lift,
                'expected_discount_percent': combo.expected_discount_percent,
                'category_mix': combo.category_mix,
                'theme': combo_data['theme'],
                'estimated_revenue': combo_data['estimated_revenue'],
                'popularity_score': combo_data['popularity_score'],
                'is_active': combo.is_active,
                'created_at': combo.created_at.isoformat(),
                'metadata': {
                    'last_performance': combo_data['last_performance']
                }
            }
        else:  # simplified
            combo_export = {
                'combo_id': combo.combo_id,
                'name': combo.name,
                'products': combo.products,
                'discount_percent': combo.expected_discount_percent,
                'confidence': combo.confidence_score,
                'active': combo.is_active
            }
        
        export_data['combos'].append(combo_export)
    
    return json.dumps(export_data, indent=2)


def main():
    # Sidebar filters and controls
    st.sidebar.header("🎛️ Combo Management")
    
    # Load data
    if st.session_state.combo_data is None:
        with st.spinner("Generating combo offers..."):
            combo_list, all_products = generate_mock_combo_data()
            st.session_state.combo_data = combo_list
            st.session_state.all_products = all_products
    
    combo_list = st.session_state.combo_data
    all_products = st.session_state.all_products
    
    # Filters
    st.sidebar.subheader("🔍 Filters")
    
    # Theme filter
    themes = list(set([combo_data['theme'] for combo_data in combo_list]))
    selected_themes = st.sidebar.multiselect(
        "Themes",
        options=themes,
        default=themes
    )
    
    # Confidence filter
    confidence_range = st.sidebar.slider(
        "Confidence Range",
        min_value=0.0,
        max_value=1.0,
        value=(0.0, 1.0),
        step=0.1
    )
    
    # Status filter
    status_filter = st.sidebar.selectbox(
        "Status",
        options=["All", "Active Only", "Inactive Only"]
    )
    
    # Discount range filter
    discount_range = st.sidebar.slider(
        "Discount Range (%)",
        min_value=0.0,
        max_value=30.0,
        value=(0.0, 30.0),
        step=1.0
    )
    
    # Apply filters
    filtered_combos = []
    for combo_data in combo_list:
        combo = combo_data['combo']
        
        # Theme filter
        if combo_data['theme'] not in selected_themes:
            continue
        
        # Confidence filter
        if not (confidence_range[0] <= combo.confidence_score <= confidence_range[1]):
            continue
        
        # Status filter
        if status_filter == "Active Only" and not combo.is_active:
            continue
        elif status_filter == "Inactive Only" and combo.is_active:
            continue
        
        # Discount filter
        if not (discount_range[0] <= combo.expected_discount_percent <= discount_range[1]):
            continue
        
        filtered_combos.append(combo_data)
    
    # Main dashboard
    st.header("📊 Combo Overview")
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    total_combos = len(filtered_combos)
    active_combos = len([c for c in filtered_combos if c['combo'].is_active])
    avg_confidence = np.mean([c['combo'].confidence_score for c in filtered_combos]) if filtered_combos else 0
    total_revenue = sum([c['estimated_revenue'] for c in filtered_combos])
    
    with col1:
        st.metric("Total Combos", total_combos)
        st.metric("Active Combos", active_combos)
    
    with col2:
        st.metric("Average Confidence", f"{avg_confidence:.3f}")
        st.metric("Est. Total Revenue", f"${total_revenue:.0f}")
    
    with col3:
        if filtered_combos:
            avg_discount = np.mean([c['combo'].expected_discount_percent for c in filtered_combos])
            avg_lift = np.mean([c['combo'].lift for c in filtered_combos])
        else:
            avg_discount = avg_lift = 0
        st.metric("Average Discount", f"{avg_discount:.1f}%")
        st.metric("Average Lift", f"{avg_lift:.2f}x")
    
    with col4:
        theme_count = len(set([c['theme'] for c in filtered_combos]))
        category_count = len(set([cat for c in filtered_combos for cat in c['combo'].category_mix]))
        st.metric("Themes", theme_count)
        st.metric("Categories", category_count)
    
    # Export buttons
    st.header("📥 Export Options")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Export Detailed JSON", key="detailed"):
            json_data = export_combos_json(filtered_combos, "detailed")
            st.download_button(
                label="⬇️ Download Detailed JSON",
                data=json_data,
                file_name=f"combo_offers_detailed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                key="download_detailed"
            )
    
    with col2:
        if st.button("📋 Export Simple JSON", key="simple"):
            json_data = export_combos_json(filtered_combos, "simple")
            st.download_button(
                label="⬇️ Download Simple JSON",
                data=json_data,
                file_name=f"combo_offers_simple_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                key="download_simple"
            )
    
    with col3:
        if st.button("📈 Export Analytics CSV", key="csv"):
            df = create_combo_analytics(filtered_combos)
            if df is not None:
                csv_data = df.to_csv(index=False)
                st.download_button(
                    label="⬇️ Download CSV",
                    data=csv_data,
                    file_name=f"combo_analytics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    key="download_csv"
                )
    
    # Analytics section
    if filtered_combos:
        st.header("📈 Analytics Dashboard")
        
        df = create_combo_analytics(filtered_combos)
        
        tab1, tab2, tab3, tab4 = st.tabs(["Performance", "Distribution", "Themes", "Revenue"])
        
        with tab1:
            # Confidence vs Lift scatter
            fig = px.scatter(
                df, x='confidence', y='lift', 
                size='revenue', color='theme',
                hover_data=['name', 'discount'],
                title='Combo Performance: Confidence vs Lift'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Top performers
            st.subheader("🏆 Top Performing Combos")
            top_combos = df.nlargest(5, 'popularity')[['name', 'confidence', 'lift', 'revenue', 'theme']]
            st.dataframe(top_combos, use_container_width=True)
        
        with tab2:
            col1, col2 = st.columns(2)
            
            with col1:
                # Confidence distribution
                fig = px.histogram(
                    df, x='confidence', 
                    title='Confidence Score Distribution',
                    nbins=15
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Discount distribution
                fig = px.histogram(
                    df, x='discount',
                    title='Discount Distribution',
                    nbins=15
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            # Theme analysis
            theme_stats = df.groupby('theme').agg({
                'confidence': 'mean',
                'lift': 'mean',
                'revenue': 'sum',
                'name': 'count'
            }).round(3)
            theme_stats.columns = ['Avg Confidence', 'Avg Lift', 'Total Revenue', 'Count']
            
            st.subheader("📊 Performance by Theme")
            st.dataframe(theme_stats, use_container_width=True)
            
            # Theme performance chart
            fig = px.bar(
                theme_stats.reset_index(), 
                x='theme', y='Total Revenue',
                title='Revenue by Theme'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with tab4:
            # Revenue analysis
            fig = px.box(
                df, x='theme', y='revenue',
                title='Revenue Distribution by Theme'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Revenue metrics
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Estimated Revenue", f"${df['revenue'].sum():.0f}")
                st.metric("Average Revenue per Combo", f"${df['revenue'].mean():.0f}")
            
            with col2:
                st.metric("Highest Revenue Combo", f"${df['revenue'].max():.0f}")
                st.metric("Revenue Standard Deviation", f"${df['revenue'].std():.0f}")
    
    # Combo cards display
    st.header("🎁 Combo Collection")
    
    # Sort options
    col1, col2 = st.columns([3, 1])
    with col1:
        sort_by = st.selectbox(
            "Sort by",
            options=["Confidence", "Lift", "Revenue", "Discount", "Created Date"],
            index=0
        )
    with col2:
        sort_desc = st.checkbox("Descending", value=True)
    
    # Sort combos
    sort_key_map = {
        "Confidence": lambda x: x['combo'].confidence_score,
        "Lift": lambda x: x['combo'].lift,
        "Revenue": lambda x: x['estimated_revenue'],
        "Discount": lambda x: x['combo'].expected_discount_percent,
        "Created Date": lambda x: x['combo'].created_at
    }
    
    sorted_combos = sorted(
        filtered_combos, 
        key=sort_key_map[sort_by], 
        reverse=sort_desc
    )
    
    # Display cards
    if sorted_combos:
        # Pagination
        items_per_page = st.selectbox("Items per page", [6, 12, 18, 24], index=1)
        total_pages = (len(sorted_combos) - 1) // items_per_page + 1
        
        if total_pages > 1:
            page = st.number_input(
                "Page", 
                min_value=1, 
                max_value=total_pages, 
                value=1
            ) - 1
        else:
            page = 0
        
        start_idx = page * items_per_page
        end_idx = min(start_idx + items_per_page, len(sorted_combos))
        page_combos = sorted_combos[start_idx:end_idx]
        
        # Display cards in grid
        cols_per_row = 2
        for i in range(0, len(page_combos), cols_per_row):
            cols = st.columns(cols_per_row)
            for j in range(cols_per_row):
                if i + j < len(page_combos):
                    with cols[j]:
                        card_html = render_combo_card(page_combos[i + j], all_products)
                        st.markdown(card_html, unsafe_allow_html=True)
        
        # Page info
        if total_pages > 1:
            st.markdown(f"*Showing page {page + 1} of {total_pages} ({start_idx + 1}-{end_idx} of {len(sorted_combos)} combos)*")
    
    else:
        st.info("🔍 No combos match the current filters. Try adjusting your filter criteria.")
    
    # Management actions
    st.header("⚙️ Management Actions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔄 Refresh Combos"):
            st.session_state.combo_data = None
            st.rerun()
    
    with col2:
        if st.button("✨ Generate New Combos"):
            st.info("🚧 New combo generation feature coming soon!")
    
    with col3:
        if st.button("📊 Detailed Analytics"):
            st.info("🚧 Advanced analytics dashboard coming soon!")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "💡 **Pro Tips:** \n"
        "- Use filters to focus on specific combo types or performance ranges\n"
        "- Export detailed JSON for integration with POS systems\n"
        "- Monitor combo performance regularly to optimize discount strategies\n"
        "- Green cards = High confidence, Orange = Medium, Purple = Lower confidence"
    )


if __name__ == "__main__":
    main() 