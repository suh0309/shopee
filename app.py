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
    customers = pd.read_csv('synthetic_customers.csv')
    transactions = pd.read_csv('synthetic_transactions.csv')
    customers['churn'] = (customers['days_since_last_order'] > 30).astype(int)
    return customers, transactions

customers, transactions = load_data()

st.set_page_config(layout="wide", page_title="Shopee Analytics Dashboard")
st.sidebar.title("Navigation")
section = st.sidebar.radio("", [
    "Home", "Metrics", "Segmentation", "CLV", "Churn",
    "Next-Purchase", "Sales Forecast", "Market Response",
    "Uplift", "Association Rules", "A/B Testing"
])

if section == "Home":
    st.title("📊 Shopee Analytics Dashboard")
    st.markdown("Explore multiple analytics techniques on synthetic Shopee data.")

elif section == "Metrics":
    st.header("Know-Your-Metrics")
    window = st.selectbox("Window:", ["7 days","14 days","30 days"])
    mau = np.random.randint(1000,5000,10)
    fig = px.line(x=list(range(1,11)), y=mau, labels={'x':'Period','y':'MAU'})
    st.plotly_chart(fig, use_container_width=True)

elif section == "Segmentation":
    st.header("Customer Segmentation (RFM + K-Means)")
    rfm = customers[['customer_id','total_orders','avg_order_value','days_since_last_order']].copy()
    rfm.columns = ['id','frequency','monetary','recency']
    k = st.slider("Clusters (k):", 2, 6, 3)
    km = KMeans(n_clusters=k, random_state=42).fit(rfm[['recency','frequency','monetary']])
    rfm['cluster'] = km.labels_
    fig = px.scatter(rfm, x='frequency', y='monetary', color='cluster',
                     labels={'frequency':'Frequency','monetary':'Monetary'})
    st.plotly_chart(fig, use_container_width=True)

elif section == "CLV":
    st.header("Customer Lifetime Value (CLV)")
    bgf = BetaGeoFitter(penalizer_coef=0.0)
    bgf.fit(customers['total_orders'], customers['days_since_last_order'], T=60)
    ggf = GammaGammaFitter(penalizer_coef=0.0)
    ggf.fit(customers['total_orders'], customers['avg_order_value'])
    t = st.slider("Horizon (days):", 30, 180, 90, step=30)
    clv = bgf.customer_lifetime_value(
        ggf,
        customers['total_orders'],
        customers['days_since_last_order'],
        customers['avg_order_value'],
        time=t
    )
    st.write(clv.head())

elif section == "Churn":
    st.header("Churn Prediction")
    feats = ['total_orders','avg_order_value','app_opens','session_duration','cart_abandons']
    X, y = customers[feats], (customers['days_since_last_order'] > 30).astype(int)
    Xtr, Xte, ytr, yte = train_test_split(X, y, random_state=42)
    clf = GradientBoostingClassifier().fit(Xtr, ytr)
    customers['churn_prob'] = clf.predict_proba(X)[:,1]
    fig = px.histogram(customers, x='churn_prob', nbins=20, labels={'churn_prob':'Churn Probability'})
    st.plotly_chart(fig, use_container_width=True)

elif section == "Next-Purchase":
    st.header("Next-Purchase Regression")
    x = st.selectbox("Feature (X):", ['total_orders','avg_order_value','app_opens'])
    y = customers['days_since_last_order']
    model = RandomForestRegressor(n_estimators=50, random_state=42).fit(customers[[x]], y)
    customers['pred_next'] = model.predict(customers[[x]])
    fig = px.scatter(customers, x=x, y='pred_next',
                     labels={x:x.replace('_',' ').title(), 'pred_next':'Predicted Days'})
    st.plotly_chart(fig, use_container_width=True)

elif section == "Sales Forecast":
    st.header("Sales Forecast (Simulated)")
    days = np.arange(1,31)
    sales = 100 + 5*days + norm.rvs(0,20,30)
    df = pd.DataFrame({'Day':days,'Sales':sales})
    fig = px.line(df, x='Day', y='Sales')
    st.plotly_chart(fig, use_container_width=True)

elif section == "Market Response":
    st.header("Market Response Curve")
    spend = np.linspace(100,1000,10)
    resp = 200 * np.log1p(spend)
    df = pd.DataFrame({'Ad Spend':spend,'Response':resp})
    fig = px.scatter(df, x='Ad Spend', y='Response')
    st.plotly_chart(fig, use_container_width=True)

elif section == "Uplift":
    st.header("Uplift Modeling")
    df = pd.DataFrame({
        'treatment':np.random.choice([0,1],200),
        'outcome':np.random.binomial(1,0.3,200)
    })
    fig = px.histogram(df, x='outcome', color='treatment',
                       barmode='group', labels={'treatment':'T/C','outcome':'Purchase'})
    st.plotly_chart(fig, use_container_width=True)

elif section == "Association Rules":
    st.header("Association Rule Mining")
    sup = st.slider("Min Support:", 0.01, 0.1, 0.05)
    conf = st.slider("Min Confidence:", 0.1, 1.0, 0.3)
    basket = transactions.set_index('transaction_id')
    freq = apriori(basket, min_support=sup, use_colnames=True)
    rules = association_rules(freq, metric="confidence", min_threshold=conf)
    st.write(rules[['antecedents','consequents','support','confidence','lift']])

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
