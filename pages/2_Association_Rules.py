import streamlit as st
import pandas as pd
import sys
import os

# Add src directory to path so we can import our modules
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

from association_engine import AprioriAssociationEngine

st.set_page_config(page_title="Association Rules", page_icon="🔗")

st.markdown("# 🔗 Association Rules Viewer")
st.sidebar.header("Association Rules")

# Cache the data loading and training
@st.cache_data
def load_product_catalog():
    """Load product catalog data."""
    try:
        return pd.read_csv('data/grocery_catalog.csv')
    except FileNotFoundError:
        st.error("Product catalog not found. Please ensure data/grocery_catalog.csv exists.")
        return pd.DataFrame()

@st.cache_data
def load_transaction_data():
    """Load transaction data."""
    try:
        return pd.read_csv('data/sample_transactions.csv')
    except FileNotFoundError:
        st.error("Transaction data not found. Running transaction generator...")
        # Try to generate the data
        try:
            os.system('python scripts/generate_sample_transactions.py')
            return pd.read_csv('data/sample_transactions.csv')
        except:
            st.error("Could not generate transaction data. Please run: python scripts/generate_sample_transactions.py")
            return pd.DataFrame()

@st.cache_resource
def train_association_engine(min_support=0.05, min_confidence=0.3, min_lift=1.0):
    """Train the association engine with the transaction data."""
    transactions_df = load_transaction_data()
    if transactions_df.empty:
        return None
    
    # Initialize and train the engine
    engine = AprioriAssociationEngine(
        min_support=min_support,
        min_confidence=min_confidence,
        min_lift=min_lift,
        max_itemset_size=3
    )
    
    with st.spinner("Training association engine..."):
        engine.train(transactions_df)
    
    return engine

def format_product_name(product_id, catalog_df):
    """Get product name from catalog."""
    if catalog_df.empty:
        return f"Product {product_id}"
    
    product_row = catalog_df[catalog_df['product_id'] == product_id]
    if not product_row.empty:
        return product_row.iloc[0]['product_name']
    return f"Product {product_id}"

def display_association_rules(engine, catalog_df):
    """Display association rules analysis."""
    
    st.header("🎯 Product Association Analysis")
    
    # Get engine statistics
    stats = engine.get_stats()
    
    # Display training statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Transactions", stats['total_transactions'])
    with col2:
        st.metric("Association Rules", stats['total_rules'])
    with col3:
        st.metric("Unique Products", stats['unique_items'])
    with col4:
        frequent_itemsets_total = sum(stats['frequent_itemsets_by_size'].values())
        st.metric("Frequent Itemsets", frequent_itemsets_total)
    
    # Product selection
    st.subheader("🛍️ Select a Product to Explore Associations")
    
    if not catalog_df.empty:
        # Create product options
        product_options = {}
        for _, row in catalog_df.iterrows():
            product_options[f"{row['product_name']} (ID: {row['product_id']})"] = row['product_id']
        
        selected_product_display = st.selectbox(
            "Choose a product:", 
            options=list(product_options.keys()),
            index=0
        )
        
        selected_product_id = product_options[selected_product_display]
        
        # Get associations for selected product
        associations = engine.get_associations(selected_product_id, top_k=10)
        
        if associations:
            st.subheader(f"📊 Products Associated with {format_product_name(selected_product_id, catalog_df)}")
            
            # Create association display
            association_data = []
            for assoc_product_id, lift_score in associations:
                association_data.append({
                    'Associated Product': format_product_name(assoc_product_id, catalog_df),
                    'Product ID': assoc_product_id,
                    'Association Strength (Lift)': f"{lift_score:.2f}",
                    'Strength Category': categorize_strength(lift_score)
                })
            
            assoc_df = pd.DataFrame(association_data)
            
            # Color code by strength
            def color_strength(val):
                if 'Very Strong' in val:
                    return 'background-color: #d4edda'
                elif 'Strong' in val:
                    return 'background-color: #fff3cd'
                elif 'Moderate' in val:
                    return 'background-color: #f8d7da'
                return ''
            
            styled_df = assoc_df.style.applymap(color_strength, subset=['Strength Category'])
            st.dataframe(styled_df, use_container_width=True)
            
            # Display detailed rules
            with st.expander("🔍 Detailed Association Rules"):
                rule_details = engine.get_rule_details(selected_product_id)
                if rule_details:
                    for i, rule in enumerate(rule_details[:10]):  # Show top 10 rules
                        antecedent_names = [format_product_name(pid, catalog_df) for pid in rule['antecedent']]
                        consequent_names = [format_product_name(pid, catalog_df) for pid in rule['consequent']]
                        
                        st.write(f"**Rule {i+1}:** {', '.join(antecedent_names)} → {', '.join(consequent_names)}")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Support", f"{rule['support']:.3f}")
                        with col2:
                            st.metric("Confidence", f"{rule['confidence']:.3f}")
                        with col3:
                            st.metric("Lift", f"{rule['lift']:.2f}")
                        with col4:
                            st.metric("Conviction", f"{rule['conviction']:.2f}" if rule['conviction'] != float('inf') else "∞")
                        
                        st.divider()
                else:
                    st.info("No detailed rules found for this product.")
        else:
            st.info(f"No strong associations found for {format_product_name(selected_product_id, catalog_df)}. Try adjusting the parameters below.")
    
    else:
        st.error("Product catalog not available.")

def categorize_strength(lift_score):
    """Categorize association strength based on lift score."""
    if lift_score >= 3.0:
        return "Very Strong (3.0+)"
    elif lift_score >= 2.0:
        return "Strong (2.0+)"
    elif lift_score >= 1.5:
        return "Moderate (1.5+)"
    else:
        return "Weak (< 1.5)"

def display_parameter_tuning():
    """Display parameter tuning interface."""
    st.header("⚙️ Algorithm Parameters")
    
    with st.expander("Adjust Association Mining Parameters"):
        st.write("""
        **Parameter Explanations:**
        - **Minimum Support**: How often an itemset appears in transactions (higher = more frequent)
        - **Minimum Confidence**: How likely the consequent is when antecedent is present (higher = more reliable)
        - **Minimum Lift**: How much more likely items appear together vs. independently (higher = stronger association)
        """)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            min_support = st.slider("Minimum Support", 0.01, 0.3, 0.05, 0.01, 
                                   help="Minimum frequency for itemsets to be considered")
        
        with col2:
            min_confidence = st.slider("Minimum Confidence", 0.1, 0.9, 0.3, 0.05,
                                     help="Minimum confidence for association rules")
        
        with col3:
            min_lift = st.slider("Minimum Lift", 1.0, 5.0, 1.0, 0.1,
                               help="Minimum lift score for association rules")
        
        if st.button("🔄 Retrain with New Parameters"):
            # Clear cache and retrain
            st.cache_resource.clear()
            engine = train_association_engine(min_support, min_confidence, min_lift)
            if engine:
                st.success("✅ Engine retrained successfully!")
                st.rerun()
            else:
                st.error("❌ Failed to retrain engine.")

def main():
    """Main function for the Association Rules page."""
    
    # Load data
    catalog_df = load_product_catalog()
    transactions_df = load_transaction_data()
    
    if transactions_df.empty:
        st.error("No transaction data available. Please generate transaction data first.")
        if st.button("🔄 Generate Sample Data"):
            with st.spinner("Generating sample transaction data..."):
                os.system('python scripts/generate_sample_transactions.py')
            st.success("Sample data generated! Please refresh the page.")
        return
    
    # Display basic transaction statistics
    st.info(f"""
    **Data Overview:**
    - {transactions_df['transaction_id'].nunique():,} transactions
    - {transactions_df['product_id'].nunique()} unique products
    - {len(transactions_df):,} total items purchased
    - Average {len(transactions_df) / transactions_df['transaction_id'].nunique():.1f} items per transaction
    """)
    
    # Train association engine
    engine = train_association_engine()
    
    if engine is None:
        st.error("Failed to train association engine.")
        return
    
    if not engine.is_trained:
        st.error("Association engine training failed.")
        return
    
    # Display association rules
    display_association_rules(engine, catalog_df)
    
    # Parameter tuning section
    display_parameter_tuning()
    
    # Additional insights
    with st.expander("📈 Market Basket Analysis Insights"):
        st.write("""
        **How to Interpret the Results:**
        
        1. **Association Strength (Lift):**
           - **> 3.0**: Very strong association - products are bought together much more than expected
           - **2.0 - 3.0**: Strong association - good candidates for bundling or cross-promotion
           - **1.5 - 2.0**: Moderate association - some correlation in purchases
           - **< 1.5**: Weak association - little to no correlation
        
        2. **Business Applications:**
           - **Product Placement**: Place strongly associated products near each other
           - **Cross-Selling**: Recommend associated products to customers
           - **Inventory Management**: Stock related items in similar quantities
           - **Promotional Strategies**: Create bundles or discounts for associated products
        
        3. **Layout Optimization:**
           - Products with high associations should be placed in proximity
           - Create "convenience zones" with frequently bought-together items
           - Design customer flow paths that naturally lead past associated products
        """)

if __name__ == "__main__":
    main() 