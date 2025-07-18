import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from lifetimes import BetaGeoFitter, GammaGammaFitter
from lifetimes.utils import ConvergenceError
from mlxtend.frequent_patterns import apriori, association_rules
import plotly.express as px
from scipy.stats import norm

# --- Synthetic Data Generation / Loading ---
@st.cache(allow_output_mutation=True)
def load_data():
    np.random.seed(42)
    # Observation period for CLV (days)
    T = 90
    n_customers = 500

    # 1) Simulate true repeat frequency
    frequency = np.random.poisson(lam=3, size=n_customers)

    # 2) Generate recency: if freq==0 -> 0, else uniform[1, T]
    recency = np.array([
        np.random.randint(1, T+1) if f > 0 else 0
        for f in frequency
    ])

    # 3) Monetary value per customer (Gamma distribution)
    monetary = np.round(np.random.gamma(shape=2, scale=50, size=n_customers), 2)

    # Build DataFrame for modeling
    df = pd.DataFrame({
        'customer_id': range(1, n_customers+1),
        'frequency': frequency,
        'recency': recency,
        'monetary': monetary
    })

    # Derive features for other modules
    df['total_orders']        = df['frequency'] + 1               # at least one purchase
    df['avg_order_value']     = df['monetary']
    df['days_since_last_order'] = df['recency']                  # proxy
    df['app_opens']           = np.random.randint(1, 100, n_customers)
    df['session_duration']    = np.round(np.random.uniform(1, 30, n_customers), 1)
    df['cart_abandons']       = np.random.poisson(lam=2, size=n_customers)
    # Churn label: no purchase in last 30 days
    df['churn']               = (df['days_since_last_order'] > 30).astype(int)

    # Synthetic transactions for association rules
    n_trans = 1000
    items = ['Electronics','Clothing','Home','Beauty','Toys']
    transactions = pd.DataFrame([
        { 'transaction_id': i, 
          **{item: np.random.choice([0,1], p=[0.7,0.3]) for item in items} }
        for i in range(1, n_trans+1)
    ])

    return df, transactions

customers, transactions = load_data()

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
    Explore these modules:
    1. Know-Your-Metrics  
    2. Segmentation (RFM + Clustering)  
    3. CLV Prediction  
    4. Churn Prediction  
    5. Next-Purchase Regression  
    6. Sales Forecast  
    7. Market Response Modeling  
    8. Uplift Modeling  
    9. Association Rule Mining  
    10. A/B Testing Analysis  
    """)


# 2. Know-Your-Metrics
elif section == "Know-Your-Metrics":
    st.header("Know-Your-Metrics")
    window = st.selectbox("Window:", ["7 days","14 days","30 days"])
    mau = np.random.randint(1000,5000,10)
    fig = px.line(x=list(range(1,11)), y=mau,
                  labels={'x':'Period','y':'MAU'},
                  title=f"MAU over {window}")
    st.plotly_chart(fig, use_container_width=True)

# 3. Segmentation
elif section == "Segmentation":
    st.header("Customer Segmentation (RFM + K-Means)")
    rfm = customers[['customer_id','frequency','monetary','recency']].copy()
    rfm.columns = ['id','frequency','monetary','recency']
    k = st.slider("Clusters (k):", 2, 6, 3)
    rfm['cluster'] = KMeans(n_clusters=k, random_state=42) \
                       .fit_predict(rfm[['recency','frequency','monetary']])
    fig = px.scatter(rfm, x='frequency', y='monetary', color='cluster',
                     labels={'frequency':'Frequency','monetary':'Monetary'},
                     title="Clusters: Frequency vs. Monetary")
    st.plotly_chart(fig, use_container_width=True)

# 4. CLV Prediction (Fixed & Regularized)

elif section == "CLV":
    st.header("Customer Lifetime Value (CLV) Prediction")
    df = customers[customers['frequency'] > 0].copy()
    st.write(f"Modeling on {len(df)} customers with >0 repeat purchases")

    # 1) BG/NBD
    bgf = BetaGeoFitter(penalizer_coef=0.01)
    T_series = pd.Series(90, index=df.index)
    try:
        bgf.fit(df['frequency'], df['recency'], T=T_series)
    except ConvergenceError:
        st.error("⚠️ BG/NBD failed to converge.")
        st.stop()

    # 2) Gamma-Gamma
    ggf = GammaGammaFitter(penalizer_coef=0.01)
    ggf.fit(df['frequency'], df['monetary'])

    # Forecast
    horizon = st.slider("Forecast horizon (days):", 30, 180, 90, step=30)
    clv = ggf.customer_lifetime_value(
        bgf,                         # pass the BG/NBD model here
        df['frequency'],
        df['recency'],
        df['monetary'],
        horizon
    )

   # 5) Prepare a nicely headed DataFrame
    top10 = clv.sort_values(ascending=False).head(10)
    clv_df = top10.reset_index()
    clv_df.columns = ["Customer ID", "Predicted CLV"]

    # 6) Display
    st.subheader("Top 10 Customers by Predicted CLV")
    st.dataframe(clv_df, use_container_width=True)

# 5. Churn Prediction
elif section == "Churn":
    st.header("Churn Prediction")
    feats = ['total_orders','avg_order_value','app_opens','session_duration','cart_abandons']
    X = customers[feats]
    y = customers['churn']

    # Correct train/test split unpacking
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    clf = GradientBoostingClassifier().fit(X_train, y_train)

    # Predict on the full feature set
    customers['churn_prob'] = clf.predict_proba(X)[:,1]

    fig = px.histogram(
        customers, x='churn_prob', nbins=20,
        labels={'churn_prob':'Churn Probability'},
        title="Predicted Churn Probability Distribution"
    )
    st.plotly_chart(fig, use_container_width=True)


# 6. Next-Purchase Regression
elif section == "Next-Purchase":
    st.header("Next-Purchase Day Regression")
    x = st.selectbox("Feature:", ['total_orders','avg_order_value','app_opens'])
    y = customers['days_since_last_order']
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(customers[[x]], y)
    customers['pred_next'] = model.predict(customers[[x]])
    fig = px.scatter(customers, x=x, y='pred_next',
                     labels={x:x.replace('_',' ').title(),'pred_next':'Predicted Days'},
                     title=f"Predicted Days to Next Order vs {x.replace('_',' ').title()}")
    st.plotly_chart(fig, use_container_width=True)

# 7. Sales Forecast
elif section == "Sales Forecast":
    st.header("Sales Forecast (Simulated)")
    days = np.arange(1,31)
    sales = 100 + 5*days + norm.rvs(0,20,30)
    df_sales = pd.DataFrame({'Day':days,'Sales':sales})
    fig = px.line(df_sales, x='Day', y='Sales', title="Simulated Daily Sales")
    st.plotly_chart(fig, use_container_width=True)

# 8. Market Response
elif section == "Market Response":
    st.header("Market Response Curve")
    spend = np.linspace(100,1000,10)
    response = 200 * np.log1p(spend)
    df_mr = pd.DataFrame({'Ad Spend':spend,'Response':response})
    fig = px.scatter(df_mr, x='Ad Spend', y='Response', title="Market Response")
    st.plotly_chart(fig, use_container_width=True)

# 9. Uplift Modeling
elif section == "Uplift":
    st.header("Uplift Modeling (Simulated)")
    df_up = pd.DataFrame({
        'treatment': np.random.choice([0,1],200),
        'outcome':  np.random.binomial(1,0.3,200)
    })
    fig = px.histogram(df_up, x='outcome', color='treatment',
                       barmode='group',
                       labels={'treatment':'Group','outcome':'Outcome'},
                       title="Simulated Uplift: Purchase by Group")
    st.plotly_chart(fig, use_container_width=True)

# 10. Association Rule Mining
elif section == "Association Rules":
    st.header("Association Rule Mining")
    basket = transactions.set_index('transaction_id')

    # Elbow curve
    supports = np.linspace(0.01,0.1,10)
    counts = [
        len(apriori(basket, min_support=s, use_colnames=True))
        for s in supports
    ]
    elbow_df = pd.DataFrame({'Support':supports,'Itemsets':counts})
    fig_elbow = px.line(elbow_df, x='Support', y='Itemsets',
                        title="Elbow: Support vs # Itemsets")
    st.plotly_chart(fig_elbow, use_container_width=True)

    # Rule sliders
    min_sup  = st.slider("Min Support:", 0.01,0.1,0.05)
    min_conf = st.slider("Min Confidence:", 0.1,1.0,0.3)
    freq_sets = apriori(basket, min_support=min_sup, use_colnames=True)
    rules = association_rules(freq_sets, metric="confidence", min_threshold=min_conf)

    st.subheader("Top 10 Rules by Lift")
    st.dataframe(
        rules.sort_values('lift', ascending=False)
             .head(10)[['antecedents','consequents','support','confidence','lift']]
    )

# 11. A/B Testing
else:
    st.header("A/B Testing Analysis")
    cA = st.number_input("Conversions A:", 0, 1000, 50)
    tA = st.number_input("Total A:",       1, 5000, 500)
    cB = st.number_input("Conversions B:", 0, 1000, 65)
    tB = st.number_input("Total B:",       1, 5000, 500)

    pA, pB = cA/tA, cB/tB
    lift   = (pB-pA)/pA if pA>0 else 0
    p_pool = (cA+cB)/(tA+tB)
    se     = np.sqrt(p_pool*(1-p_pool)*(1/tA+1/tB))
    z      = (pB-pA)/se if se>0 else 0
    pval   = 2 * (1 - norm.cdf(abs(z)))

    st.write(f"**Lift:** {lift:.2%}")
    st.write(f"**p-value:** {pval:.3f}")
