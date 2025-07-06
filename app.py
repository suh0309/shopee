# app.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from lifetimes import BetaGeoFitter, GammaGammaFitter
from mlxtend.frequent_patterns import apriori, association_rules
import plotly.express as px
from scipy.stats import norm

# --- Synthetic Data Generation ---
@st.cache(allow_output_mutation=True)
def generate_data():
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
    # labels
    customers['churn'] = (customers['days_since_last_order'] > 30).astype(int)
    customers['next_30d'] = (customers['days_since_last_order'] <= 30).astype(int)

    n_trans = 1000
    items = ['Electronics','Clothing','Home','Beauty','Toys']
    transactions = pd.DataFrame([
        {'transaction_id': i, **{item: np.random.choice([0,1], p=[0.7,0.3]) 
                               for item in items}}
        for i in range(1, n_trans+1)
    ])
    return customers, transactions

customers, transactions = generate_data()

# --- Streamlit Layout ---
st.set_page_config(layout="wide", page_title="Shopee Analytics Dashboard")
st.sidebar.title("Navigation")
section = st.sidebar.radio("", [
    "Home", "Know-Your-Metrics", "Segmentation", "CLV", "Churn",
    "Next-Purchase", "Sales Forecast", "Market Response",
    "Uplift", "Association Rules", "A/B Testing"
])

# 1. Home
if section == "Home":
    st.title("📊 Shopee Analytics Dashboard")
    st.markdown("""
    Explore:
    - Know-Your-Metrics  
    - Segmentation (RFM + Clustering)  
    - CLV (Pareto/NBD + Gamma-Gamma)  
    - Churn Prediction  
    - Next-Purchase Regression  
    - Sales Forecast (Linear)  
    - Market Response  
    - Uplift Modeling  
    - Association Rule Mining  
    - A/B Testing Analysis  
    """)

# 2. Know-Your-Metrics
elif section == "Know-Your-Metrics":
    st.header("Know-Your-Metrics")
    window = st.selectbox("Window:", ["7 days","14 days","30 days"])
    # dummy MAU chart
    mau = np.random.randint(1000,5000,10)
    fig = px.line(x=list(range(1,11)), y=mau, labels={'x':'Period','y':'MAU'})
    st.plotly_chart(fig, use_container_width=True)

# 3. Segmentation
elif section == "Segmentation":
    st.header("Customer Segmentation (RFM + K-Means)")
    rfm = customers[['customer_id','total_orders','avg_order_value','days_since_last_order']].copy()
    rfm.columns = ['id','frequency','monetary','recency']
    st.subheader("RFM Sample")
    st.dataframe(rfm.head())
    k = st.slider("Clusters (k):", 2, 6, 3)
    km = KMeans(n_clusters=k, random_state=42).fit(rfm[['recency','frequency','monetary']])
    rfm['cluster'] = km.labels_
    fig = px.scatter(rfm, x='frequency', y='monetary', color='cluster',
                     labels={'frequency':'Frequency','monetary':'Monetary'})
    st.plotly_chart(fig, use_container_width=True)

# 4. CLV
elif section == "CLV":
    st.header("Customer Lifetime Value (CLV)")
    bgf = BetaGeoFitter(penalizer_coef=0.0)
    bgf.fit(customers['total_orders'], customers['days_since_last_order'], T=60)
    ggf = GammaGammaFitter(penalizer_coef=0.0)
    ggf.fit(customers['total_orders'], customers['avg_order_value'])
    t = st.slider("Forecast horizon (days):", 30, 180, 90, step=30)
    clv = bgf.customer_lifetime_value(ggf, 
                                       customers['total_orders'],
                                       customers['days_since_last_order'],
                                       customers['avg_order_value'], time=t)
    st.write(clv.head())

# 5. Churn
elif section == "Churn":
    st.header("Churn Prediction")
    feats = ['total_orders','avg_order_value','app_opens','session_duration','cart_abandons']
    X, y = customers[feats], customers['churn']
    Xtr, Xte, ytr, yte = train_test_split(X, y, random_state=42)
    clf = GradientBoostingClassifier().fit(Xtr, ytr)
    customers['churn_prob'] = clf.predict_proba(X)[:,1]
    fig = px.histogram(customers, x='churn_prob', nbins=20, labels={'churn_prob':'Churn Probability'})
    st.plotly_chart(fig, use_container_width=True)

# 6. Next-Purchase
elif section == "Next-Purchase":
    st.header("Next-Purchase Day Regression")
    x = st.selectbox("Feature (X):", ['total_orders','avg_order_value','app_opens'])
    y = customers['days_since_last_order']
    model = RandomForestRegressor(n_estimators=50, random_state=42).fit(customers[[x]], y)
    customers['pred_next'] = model.predict(customers[[x]])
    fig = px.scatter(customers, x=x, y='pred_next',
                     labels={x:x.replace('_',' ').title(), 'pred_next':'Predicted Days'})
    st.plotly_chart(fig, use_container_width=True)

# 7. Sales Forecast
elif section == "Sales Forecast":
    st.header("Sales Forecast (Linear - Simulated)")
    days = np.arange(1,31)
    sales = 100 + 5*days + norm.rvs(0,20,30)
    df = pd.DataFrame({'Day':days,'Sales':sales})
    fig = px.line(df, x='Day', y='Sales')
    st.plotly_chart(fig, use_container_width=True)

# 8. Market Response
elif section == "Market Response":
    st.header("Market Response Curve")
    spend = np.linspace(100,1000,10)
    resp = 200 * np.log1p(spend)
    df = pd.DataFrame({'Ad Spend':spend,'Response':resp})
    fig = px.scatter(df, x='Ad Spend', y='Response')
    st.plotly_chart(fig, use_container_width=True)

# 9. Uplift
elif section == "Uplift":
    st.header("Uplift Modeling (Simulated)")
    df = pd.DataFrame({'treatment':np.random.choice([0,1],200),
                       'outcome':np.random.binomial(1,0.3,200)})
    fig = px.histogram(df, x='outcome', color='treatment',
                       barmode='group', labels={'treatment':'T/C','outcome':'Purchase'})
    st.plotly_chart(fig, use_container_width=True)

# 10. Association Rules
elif section == "Association Rules":
    st.header("Association Rule Mining")
    sup = st.slider("Min Support:", 0.01, 0.1, 0.05)
    conf = st.slider("Min Confidence:", 0.1, 1.0, 0.3)
    basket = transactions.set_index('transaction_id')
    freq = apriori(basket, min_support=sup, use_colnames=True)
    rules = association_rules(freq, metric="confidence", min_threshold=conf)
    st.write(rules[['antecedents','consequents','support','confidence','lift']])

# 11. A/B Testing
else:
    st.header("A/B Testing Analysis")
    cA = st.number_input("Conversions A:", 0, 1000, 50)
    tA = st.number_input("Total A:", 1, 5000, 500)
    cB = st.number_input("Conversions B:", 0, 1000, 65)
    tB = st.number_input("Total B:", 1, 5000, 500)
    p1, p2 = cA/tA, cB/tB
    lift = (p2-p1)/p1 if p1>0 else 0
    st.write(f"Lift: {lift:.2%}")
    p_pool = (cA+cB)/(tA+tB)
    se = np.sqrt(p_pool*(1-p_pool)*(1/tA+1/tB))
    z = (p2-p1)/se if se>0 else 0
    pval = 2*(1-norm.cdf(abs(z)))
    st.write(f"p-value: {pval:.3f}")
