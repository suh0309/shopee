import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from mlxtend.frequent_patterns import apriori, association_rules
import plotly.express as px
from scipy.stats import norm

# --- Synthetic Data Generation / Loading ---
@st.cache(allow_output_mutation=True)
def load_data():
    np.random.seed(42)
    # Customers
    n_customers = 500
    customers = pd.DataFrame({
        'customer_id': range(1, n_customers + 1),
        'total_orders': np.random.poisson(5, n_customers),
        'avg_order_value': np.round(np.random.uniform(20, 200, n_customers), 2),
        'days_since_last_order': np.random.randint(1, 60, n_customers),
        'app_opens': np.random.randint(1, 100, n_customers),
        'session_duration': np.round(np.random.uniform(1, 30, n_customers), 1),
        'cart_abandons': np.random.randint(0, 10, n_customers)
    })
    customers['churn'] = (customers['days_since_last_order'] > 30).astype(int)

    # Transactions for association rules
    n_trans = 1000
    items = ['Electronics','Clothing','Home','Beauty','Toys']
    transactions = pd.DataFrame([
        {'transaction_id': i, **{item: np.random.choice([0,1], p=[0.7,0.3])
         for item in items}}
        for i in range(1, n_trans+1)
    ])

    return customers, transactions

customers, transactions = load_data()

# --- Streamlit Layout ---
st.set_page_config(layout="wide", page_title="Shopee Analytics Dashboard")
st.sidebar.title("Navigation")
section = st.sidebar.radio("", [
    "Home", "Know-Your-Metrics", "Segmentation", "Churn",
    "Next-Purchase", "Sales Forecast", "Market Response",
    "Uplift", "Association Rules", "A/B Testing"
])

# --- 1. Home ---
if section == "Home":
    st.title("📊 Shopee Analytics Dashboard")
    st.markdown("""
    Explore these modules:
    1. Know-Your-Metri
