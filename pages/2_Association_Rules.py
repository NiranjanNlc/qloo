import streamlit as st

st.set_page_config(page_title="Association Rules", page_icon="🔗")

st.markdown("# Association Rules Viewer")
st.sidebar.header("Association Rules")

st.info(
    """
    This page will display the discovered association rules. Once the `AssociationEngine`
    is implemented and trained, you will be able to select a product and see which
    other products are most frequently purchased with it.

    *Implementation is pending the completion of the core algorithm.*
    """,
    icon="ℹ️"
)

# Placeholder for future functionality
st.selectbox("Select a product to see its associations:", options=["(algorithm not yet trained)"])

st.write("### Top Associated Products:")
st.warning("No data to display.") 