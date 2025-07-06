import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from lifetimes import BetaGeoFitter, GammaGammaFitter
from mlxtend.frequent_patterns import apriori, association_rules
import plotly.express as px
from scipy.stats import norm

@st.cache(allow_output_mutation=True)
def load_data():
    # Synthetic customer data
    np.random.seed(42)
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
    # Synthetic transaction data
    n_trans = 1000
    items = ['Electronics','Clothing','Home','Beauty','Toys']
    transactions = pd.DataFrame([
        {'transaction_id': i, **{item: np.random.choice([0,1], p=[0.7,0.3])
         for item in items}}
        for i in range(1, n_trans+1)
    ])
    return customers, transactions

customers, transactions = load_data()

st.set_page_config(layout="wide", page_title="Shopee Analytics Dashboard")
st.sidebar.title("Navigation")
section = st.sidebar.radio("", ["Home", "CLV", "Association Rules"])

if section == "Home":
    st.title("📊 Shopee Dashboard Demo")
    st.write("Sections: CLV Prediction & Association Rule Mining")

elif section == "CLV":
    st.header("Customer Lifetime Value (CLV) Prediction")
    # Prepare for lifetimes: drop zero-order customers
    df = customers[customers['total_orders'] > 0].copy()
    df['frequency'] = df['total_orders'] - 1
    df['recency'] = df['days_since_last_order']
    df['monetary'] = df['avg_order_value']
    st.write(f"Using {len(df)} customers with ≥1 order")

    bgf = BetaGeoFitter(penalizer_coef=0.0)
    bgf.fit(df['frequency'], df['recency'], T=60)
    ggf = GammaGammaFitter(penalizer_coef=0.0)
    ggf.fit(df['frequency'], df['monetary'])

    horizon = st.slider("Forecast horizon (days):", 30, 180, 90, step=30)
    clv = bgf.customer_lifetime_value(
        ggf, df['frequency'], df['recency'], df['monetary'], time=horizon
    )
    st.subheader("Top 10 Predicted CLVs")
    st.write(clv.sort_values(ascending=False).head(10))

elif section == "Association Rules":
    st.header("Association Rule Mining")

    # Elbow curve: # of frequent itemsets vs support threshold
    supports = np.linspace(0.01, 0.1, 10)
    counts = []
    basket = transactions.set_index('transaction_id')
    for s in supports:
        counts.append(len(apriori(basket, min_support=s, use_colnames=True)))
    elbow_df = pd.DataFrame({'min_support': supports, 'count': counts})
    fig_elbow = px.line(
        elbow_df, x='min_support', y='count',
        labels={'min_support':'Min Support','count':'Itemset Count'},
        title="Elbow Curve: Support vs Frequent Itemsets"
    )
    st.plotly_chart(fig_elbow, use_container_width=True)

    # Filtered association rules
    min_sup = st.slider("Min Support:", 0.01, 0.1, 0.05)
    min_conf = st.slider("Min Confidence:", 0.1, 1.0, 0.3)
    freq_sets = apriori(basket, min_support=min_sup, use_colnames=True)
    rules = association_rules(freq_sets, metric="confidence", min_threshold=min_conf)

    st.subheader("Top 10 Rules by Lift (Hypersona Table)")
    top10 = rules.sort_values('lift', ascending=False).head(10)
    st.dataframe(top10[['antecedents','consequents','support','confidence','lift']])
