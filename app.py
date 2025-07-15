import streamlit as st

st.set_page_config(
    page_title="Qloo Supermarket Optimizer",
    page_icon="🛒",
    layout="wide"
)

st.title("🛒 Welcome to the Qloo-Powered Supermarket Optimizer!")

st.sidebar.success("Select a page above to begin.")

st.markdown(
    """
    This application is a proof-of-concept for optimizing supermarket layouts
    based on product association rules discovered from transaction data.

    **👈 Select a page from the sidebar** to explore the different functionalities.

    ### What can you do here?
    - **Product Catalog Explorer**: Browse the list of products available in the store.
    - **Association Rules Viewer**: See which products are most frequently purchased together.

    This interface is built to demonstrate the power of the underlying
    `AssociationEngine` and data processing pipeline.
    """
) 