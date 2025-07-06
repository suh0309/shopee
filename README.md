# Shopee Analytics Dashboard

This Streamlit application demonstrates key data-science techniques on synthetic Shopee data:

1. **Know-Your-Metrics**: Visualize MAU and basic KPIs  
2. **Segmentation**: RFM analysis with K-Means clustering  
3. **CLV**: Pareto/NBD + Gamma-Gamma lifetime value prediction  
4. **Churn**: Gradient Boosting classifier for churn risk  
5. **Next-Purchase**: Random Forest regression for reorder timing  
6. **Sales Forecast**: Simulated linear trend forecasting  
7. **Market Response**: Logarithmic spend-response curve  
8. **Uplift Modeling**: Treatment vs. control impact analysis  
9. **Association Rules**: Apriori-based product co-purchase mining  
10. **A/B Testing**: Lift calculation and z-test p-value

## Setup

1. **Clone** this repo.  
2. Ensure `synthetic_customers.csv` and `synthetic_transactions.csv` are in the root.  
3. Install dependencies:

   ```bash
   pip install -r requirements.txt
