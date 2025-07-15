import streamlit as st
import pandas as pd

st.set_page_config(page_title="Product Catalog", page_icon="🧾")

st.markdown("# Product Catalog Explorer")
st.sidebar.header("Product Catalog")
st.write(
    """
    This page displays the catalog of products available in the supermarket.
    Below is the data from `data/grocery_catalog.csv`.
    """
)

try:
    df = pd.read_csv('data/grocery_catalog.csv')
    st.dataframe(df)
except FileNotFoundError:
    st.error("The `data/grocery_catalog.csv` file was not found. Please ensure it exists.") 