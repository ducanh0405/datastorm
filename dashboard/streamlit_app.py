#!/usr/bin/env python3
"""
SmartGrocy Interactive Dashboard
================================
Streamlit dashboard for real-time forecasting insights.

Run: streamlit run dashboard/streamlit_app.py

Author: SmartGrocy Team
Date: 2025-11-18
"""

try:
    import streamlit as st
    import pandas as pd
    import numpy as np
    import plotly.express as px
    import plotly.graph_objects as go
    from datetime import datetime, timedelta
except ImportError:
    print("ERROR: Streamlit not installed")
    print("Install: pip install streamlit plotly")
    exit(1)

# Page config
st.set_page_config(
    page_title="SmartGrocy Forecast Dashboard",
    page_icon="📊",
    layout="wide"
)

# Title
st.title("📊 SmartGrocy Demand Forecast Dashboard")
st.markdown("Real-time inventory and pricing optimization")

# Sidebar filters
st.sidebar.header("Filters")

date_range = st.sidebar.date_input(
    "Date Range",
    value=(datetime.now() - timedelta(days=30), datetime.now())
)

product_filter = st.sidebar.multiselect(
    "Product Category",
    options=["Fresh Produce", "Dairy", "Packaged", "Frozen"],
    default=["Fresh Produce", "Dairy"]
)

store_filter = st.sidebar.multiselect(
    "Store/Region",
    options=["Store 1", "Store 2", "Store 3", "All"],
    default=["All"]
)

# Load data (placeholder)
@st.cache_data
def load_data():
    """Load forecast data."""
    # In production, load from database or CSV
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    
    return pd.DataFrame({
        'date': dates,
        'product': np.random.choice(['P001', 'P002', 'P003'], 100),
        'category': np.random.choice(product_filter if product_filter else ['Fresh Produce'], 100),
        'forecast_q50': np.random.randint(50, 200, 100),
        'forecast_q05': np.random.randint(30, 150, 100),
        'forecast_q95': np.random.randint(70, 250, 100),
        'actual': np.random.randint(45, 205, 100)
    })

df = load_data()

# Main metrics
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Avg Daily Demand",
        f"{df['forecast_q50'].mean():.0f}",
        f"{(df['forecast_q50'].mean() - 100):+.0f}"
    )

with col2:
    accuracy = 1 - np.abs(df['forecast_q50'] - df['actual']).mean() / df['actual'].mean()
    st.metric(
        "Forecast Accuracy",
        f"{accuracy:.1%}",
        "+2.5%"
    )

with col3:
    st.metric(
        "Products Monitored",
        df['product'].nunique(),
        "+3"
    )

with col4:
    high_risk = (df['forecast_q50'] < 50).sum()
    st.metric(
        "High Risk Items",
        high_risk,
        f"{high_risk - 5:+d}"
    )

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Forecasts",
    "📦 Inventory",
    "💰 Pricing",
    "📊 Analytics"
])

with tab1:
    st.header("Demand Forecasts")
    
    # Time series plot
    fig = go.Figure()
    
    for product in df['product'].unique():
        df_prod = df[df['product'] == product]
        
        fig.add_trace(go.Scatter(
            x=df_prod['date'],
            y=df_prod['forecast_q50'],
            mode='lines',
            name=f"{product} Forecast"
        ))
        
        fig.add_trace(go.Scatter(
            x=df_prod['date'],
            y=df_prod['actual'],
            mode='markers',
            name=f"{product} Actual",
            marker=dict(size=4)
        ))
    
    fig.update_layout(
        title="Forecast vs Actual Demand",
        xaxis_title="Date",
        yaxis_title="Units",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Forecast table
    st.subheader("Latest Forecasts")
    st.dataframe(
        df.sort_values('date', ascending=False).head(20),
        use_container_width=True
    )

with tab2:
    st.header("Inventory Status")
    
    # Inventory metrics by product
    inv_data = pd.DataFrame({
        'product': ['P001', 'P002', 'P003'],
        'current_stock': [120, 85, 150],
        'reorder_point': [100, 90, 130],
        'safety_stock': [30, 25, 40],
        'stockout_risk': [15, 45, 10]
    })
    
    # Bar chart
    fig = px.bar(
        inv_data,
        x='product',
        y=['current_stock', 'reorder_point', 'safety_stock'],
        title="Inventory Levels vs Thresholds",
        barmode='group'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Risk table
    st.subheader("Stockout Risk")
    st.dataframe(
        inv_data.style.background_gradient(
            subset=['stockout_risk'],
            cmap='RdYlGn_r'
        ),
        use_container_width=True
    )

with tab3:
    st.header("Pricing Recommendations")
    
    # Pricing data
    price_data = pd.DataFrame({
        'product': ['P001', 'P002', 'P003'],
        'current_price': [50000, 35000, 45000],
        'recommended_price': [45000, 35000, 40000],
        'discount_pct': [10, 0, 11],
        'expected_revenue_impact': [125000, 0, 98000]
    })
    
    # Scatter plot
    fig = px.scatter(
        price_data,
        x='discount_pct',
        y='expected_revenue_impact',
        size='current_price',
        text='product',
        title="Discount vs Revenue Impact"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Recommendations table
    st.subheader("Price Recommendations")
    st.dataframe(price_data, use_container_width=True)

with tab4:
    st.header("Performance Analytics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Accuracy by category
        acc_data = df.groupby('category').apply(
            lambda x: 1 - np.abs(x['forecast_q50'] - x['actual']).mean() / x['actual'].mean()
        ).reset_index(name='accuracy')
        
        fig = px.bar(
            acc_data,
            x='category',
            y='accuracy',
            title="Forecast Accuracy by Category"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Error distribution
        errors = df['forecast_q50'] - df['actual']
        
        fig = px.histogram(
            x=errors,
            title="Forecast Error Distribution",
            labels={'x': 'Error (units)'}
        )
        st.plotly_chart(fig, use_container_width=True)

# Footer
st.sidebar.markdown("---")
st.sidebar.info(
    """
    **SmartGrocy v4.0**
    
    Last updated: 2025-11-18
    
    [Documentation](docs/) | [GitHub](https://github.com/ducanh0405/datastorm)
    """
)
