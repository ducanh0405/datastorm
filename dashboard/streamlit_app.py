#!/usr/bin/env python3
"""
SmartGrocy Interactive Dashboard
================================
Streamlit dashboard for real-time forecasting insights with LLM integration.

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
    import os
    import json
    import base64
    from pathlib import Path
    from math import erf
    import google.generativeai as genai
    from dotenv import load_dotenv
except ImportError as e:
    print(f"ERROR: Missing required package: {e}")
    print("Install: pip install streamlit plotly python-dotenv google-generativeai")
    exit(1)

# Load environment variables
load_dotenv(dotenv_path="../.env")

# Page config
st.set_page_config(
    page_title="SmartGrocy Forecast Dashboard",
    page_icon="📊",
    layout="wide"
)

# Title
st.title("📊 SmartGrocy Demand Forecast Dashboard")
st.markdown("Real-time inventory and pricing optimization with AI insights")

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

# Load real data from reports
@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_forecast_data(sample_size=10000):
    """Load forecast data from reports with sampling for performance."""
    try:
        # Try to load from parquet first, fallback to CSV
        data_path = Path("../reports/predictions_test_set.parquet")
        if data_path.exists():
            # Load only a sample for performance
            df_full = pd.read_parquet(data_path)
            if len(df_full) > sample_size:
                df = df_full.sample(n=sample_size, random_state=42)
            else:
                df = df_full.copy()
        else:
            csv_path = Path("../reports/predictions_test_set.csv")
            if csv_path.exists():
                df = pd.read_csv(csv_path, nrows=sample_size)
            else:
                raise FileNotFoundError("Neither parquet nor CSV file found")

        # Rename columns for consistency
        column_mapping = {
            'hour_timestamp': 'date',
            'product_id': 'product',
            'sales_quantity': 'actual',
            'forecast_q50': 'forecast_q50',
            'forecast_q05': 'forecast_q05',
            'forecast_q95': 'forecast_q95'
        }
        df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})

        # Add category column if it doesn't exist
        if 'category' not in df.columns:
            # Group products into categories based on product_id
            df['category'] = pd.cut(df['product'], bins=3, labels=['Fresh Produce', 'Dairy', 'Packaged'])

        # Convert date column and sort
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')

        return df
    except Exception as e:
        st.warning(f"Could not load forecast data: {e}. Using sample data.")
        # Fallback to sample data
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        return pd.DataFrame({
            'date': dates,
            'product': np.random.choice(['P001', 'P002', 'P003'], 100),
            'category': np.random.choice(['Fresh Produce', 'Dairy', 'Packaged'], 100),
            'forecast_q50': np.random.randint(50, 200, 100),
            'forecast_q05': np.random.randint(30, 150, 100),
            'forecast_q95': np.random.randint(70, 250, 100),
            'actual': np.random.randint(45, 205, 100)
        })

@st.cache_data(ttl=3600)
def load_business_metrics():
    """Load business metrics from reports."""
    try:
        return pd.read_csv("../reports/business_report_summary.csv")
    except Exception as e:
        st.warning(f"Could not load business metrics: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def load_llm_insights():
    """Load LLM insights from reports."""
    try:
        return pd.read_csv("../reports/llm_insights.csv")
    except Exception as e:
        st.warning(f"Could not load LLM insights: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def load_pricing_recommendations():
    """Load pricing recommendations from reports."""
    try:
        df = pd.read_csv("../reports/pricing_recommendations.csv")
        # Limit to top recommendations for performance
        return df.head(100) if len(df) > 100 else df
    except Exception as e:
        st.warning(f"Could not load pricing recommendations: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def load_inventory_recommendations():
    """Load inventory recommendations and recompute risk metrics for dashboard accuracy."""
    try:
        df = pd.read_csv("../reports/inventory_recommendations.csv")

        if df.empty:
            return df

        df = df.copy()

        # Estimate lead-time days (fallback to 7 when demand is zero/unknown)
        avg_daily = df['avg_daily_demand'].replace(0, np.nan)
        lead_time_days = (df['lead_time_demand'] / avg_daily).replace([np.inf, -np.inf], np.nan)
        lead_time_days = lead_time_days.fillna(7.0).clip(lower=1.0, upper=30.0)

        # Stockout risk: probability that current inventory is insufficient during lead time
        demand_std_horizon = df['demand_std'] * np.sqrt(lead_time_days)
        safe_denominator = demand_std_horizon.replace(0, np.nan)
        z_scores = (df['current_inventory'] - df['lead_time_demand']) / safe_denominator

        # Use math.erf with vectorization to avoid scipy dependency at runtime
        erf_vectorized = np.vectorize(lambda val: erf(val) if np.isfinite(val) else np.nan)
        cdf_values = 0.5 * (1 + erf_vectorized(z_scores / np.sqrt(2)))
        stockout_prob = (1 - cdf_values)  # Return as fraction (0-1), not percentage

        df['stockout_risk_pct'] = stockout_prob * 100  # Convert fraction to percentage
        zero_std_mask = safe_denominator.isna()
        df.loc[zero_std_mask, 'stockout_risk_pct'] = np.where(
            df.loc[zero_std_mask, 'current_inventory'] >= df.loc[zero_std_mask, 'lead_time_demand'],
            0.0,
            5.0  # 5% stockout risk for 95% service level
        )
        df['stockout_risk_pct'] = df['stockout_risk_pct'].fillna(0.0).clip(0, 7).round(2)

        # Overstock risk: compare days of inventory vs target coverage (lead time + safety stock buffer)
        inventory_days = (df['current_inventory'] / avg_daily).replace([np.inf, -np.inf], np.nan)
        target_days = ((df['lead_time_demand'] + df['safety_stock']) / avg_daily).replace([np.inf, -np.inf], np.nan)
        target_days = target_days.fillna(lead_time_days).clip(lower=1.0, upper=60.0)
        inventory_days = inventory_days.fillna(target_days)

        overstock_ratio = (inventory_days - target_days) / target_days.replace(0, 1)
        df['overstock_risk_pct'] = np.clip(overstock_ratio * 100, 0, 95).round(2)

        # Drop raw probability columns to avoid confusion
        df = df.drop(columns=['stockout_risk', 'overstock_risk'], errors='ignore')

        return df
    except Exception as e:
        st.warning(f"Could not load inventory recommendations: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def load_model_metrics():
    """Load model metrics from validation report (more accurate metrics)."""
    try:
        # Try validation_report.json first (better metrics from final validation)
        with open("../reports/validation_report.json", 'r') as f:
            data = json.load(f)
            if 'model_metrics' in data and 'metrics' in data['model_metrics']:
                # Transform to expected structure for dashboard compatibility
                return {
                    'lightgbm': data['model_metrics']['metrics']
                }

        # Fallback to model_metrics.json if validation_report doesn't exist
        with open("../reports/metrics/model_metrics.json", 'r') as f:
            return json.load(f)
    except Exception as e:
        st.warning(f"Could not load model metrics: {e}")
        return {}

@st.cache_data(ttl=3600)
def load_shap_values():
    """Load SHAP values for feature importance."""
    try:
        df = pd.read_csv("../reports/shap_values/feature_importance.csv")
        # Return top 20 for performance
        return df.head(20)
    except Exception as e:
        st.warning(f"Could not load SHAP values: {e}")
        return pd.DataFrame()

# Performance optimization: Load data with progress indicator
with st.spinner("Loading dashboard data..."):
    df = load_forecast_data(sample_size=5000)  # Reduced sample size for performance
    business_metrics = load_business_metrics()
    llm_insights = load_llm_insights()
    pricing_recs = load_pricing_recommendations()
    inventory_recs = load_inventory_recommendations()
    model_metrics = load_model_metrics()
    shap_values = load_shap_values()

# Performance Status
col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    st.success("✅ Dashboard Optimized for Performance")
with col2:
    if len(df) > 0:
        st.info(f"📊 {len(df):,} records loaded")
    else:
        st.warning("⚠️ No data loaded")
with col3:
    try:
        # Check if AI model is available
        api_key = os.getenv('GOOGLE_API_KEY')
        if api_key:
            st.success("🤖 AI Ready")
        else:
            st.warning("⚠️ AI Offline")
    except:
        st.warning("⚠️ AI Status Unknown")

# Main metrics
col1, col2, col3, col4 = st.columns(4)

# Extract metrics from business report
avg_demand = df['forecast_q50'].mean() if not df.empty else 0
accuracy_value = 0
products_count = df['product'].nunique() if not df.empty else 0
high_risk_count = len(llm_insights[llm_insights['stockout_risk_pct'] > 50]) if not llm_insights.empty else 0

# Get metrics from business report
if not business_metrics.empty:
    accuracy_row = business_metrics[business_metrics['Metric'] == 'R² Score']
    if not accuracy_row.empty:
        accuracy_value = float(accuracy_row['Value'].iloc[0])

with col1:
    st.metric(
        "Avg Daily Demand",
        f"{avg_demand:.1f}",
        "units"
    )

with col2:
    st.metric(
        "Forecast Accuracy (R²)",
        f"{accuracy_value:.3f}",
        "score"
    )

with col3:
    st.metric(
        "Products Monitored",
        products_count,
        "items"
    )

with col4:
    st.metric(
        "High Risk Items",
        high_risk_count,
        "stockout > 50%"
    )

# Tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📈 Forecasts",
    "📦 Inventory",
    "💰 Pricing",
    "📊 Analytics",
    "🤖 LLM Insights",
    "🔍 Model Performance"
])

with tab1:
    st.header("Demand Forecasts")

    # Performance optimization: Sample data for charts
    chart_data = df.sample(min(2000, len(df)), random_state=42) if len(df) > 2000 else df

    # Time series plot - show only top 3 products for performance
    top_products = chart_data['product'].value_counts().head(3).index.tolist()

    fig = go.Figure()

    for product in top_products:
        df_prod = chart_data[chart_data['product'] == product].sort_values('date')

        fig.add_trace(go.Scatter(
            x=df_prod['date'],
            y=df_prod['forecast_q50'],
            mode='lines',
            name=f"{product} Forecast",
            line=dict(width=2)
        ))

        # Sample actual values to reduce plot points
        actual_sample = df_prod.sample(min(100, len(df_prod)), random_state=42)
        fig.add_trace(go.Scatter(
            x=actual_sample['date'],
            y=actual_sample['actual'],
            mode='markers',
            name=f"{product} Actual",
            marker=dict(size=3, opacity=0.7)
        ))

    fig.update_layout(
        title="Forecast vs Actual Demand (Top 3 Products)",
        xaxis_title="Date",
        yaxis_title="Units",
        height=400,
        showlegend=True
    )

    st.plotly_chart(fig, width='stretch')

    # Forecast table - show recent data
    st.subheader("Latest Forecasts")
    recent_forecasts = df.sort_values('date', ascending=False).head(10)
    st.dataframe(recent_forecasts, width='stretch')

    # Performance info
    st.info(f"📊 Showing data sample of {len(chart_data)} records from {len(df)} total records for optimal performance")

with tab2:
    st.header("Inventory Status")

    if not inventory_recs.empty:
        # Display inventory recommendations with formatted risk values
        st.subheader("Inventory Recommendations")
        
        # Format display dataframe with percentage formatting
        display_df = inventory_recs.head(20).copy()
        
        # Format risk columns as percentages if they exist
        if 'stockout_risk_pct' in display_df.columns:
            display_df['stockout_risk_pct'] = display_df['stockout_risk_pct'].apply(lambda x: f"{x:.2f}%")
        if 'overstock_risk_pct' in display_df.columns:
            display_df['overstock_risk_pct'] = display_df['overstock_risk_pct'].apply(lambda x: f"{x:.2f}%")
        
        st.dataframe(display_df, width='stretch')

        # Stockout risk visualization using inventory_recs
        if 'stockout_risk_pct' in inventory_recs.columns:
            st.subheader("Stockout Risk Analysis")

            # Group by risk levels
            risk_levels = pd.cut(inventory_recs['stockout_risk_pct'],
                               bins=[0, 25, 50, 75, 100],
                               labels=['Low', 'Medium', 'High', 'Critical'])

            risk_counts = risk_levels.value_counts().sort_index()

            fig = px.bar(
                x=risk_counts.index,
                y=risk_counts.values,
                title="Products by Stockout Risk Level",
                labels={'x': 'Risk Level', 'y': 'Number of Products'},
                color=risk_counts.index,
                color_discrete_map={
                    'Low': 'green',
                    'Medium': 'yellow',
                    'High': 'orange',
                    'Critical': 'red'
                }
            )
            st.plotly_chart(fig, width='stretch')

            # High risk products table
            high_risk_cols = ['product_id', 'store_id', 'stockout_risk_pct', 'overstock_risk_pct']
            available_cols = [col for col in high_risk_cols if col in inventory_recs.columns]
            
            high_risk_products = inventory_recs[inventory_recs['stockout_risk_pct'] > 50][available_cols].head(10)
            
            # Format risk percentages for display
            if not high_risk_products.empty:
                display_high_risk = high_risk_products.copy()
                if 'stockout_risk_pct' in display_high_risk.columns:
                    display_high_risk['stockout_risk_pct'] = display_high_risk['stockout_risk_pct'].apply(lambda x: f"{x:.2f}%")
                if 'overstock_risk_pct' in display_high_risk.columns:
                    display_high_risk['overstock_risk_pct'] = display_high_risk['overstock_risk_pct'].apply(lambda x: f"{x:.2f}%")
                
                st.subheader("High Risk Products (>50% stockout risk)")
                st.dataframe(display_high_risk, width='stretch')
            
            # Overstock risk analysis
            if 'overstock_risk_pct' in inventory_recs.columns:
                st.subheader("Overstock Risk Analysis")
                
                overstock_levels = pd.cut(inventory_recs['overstock_risk_pct'],
                                        bins=[0, 25, 50, 75, 100],
                                        labels=['Low', 'Medium', 'High', 'Critical'])
                
                overstock_counts = overstock_levels.value_counts().sort_index()
                
                fig2 = px.bar(
                    x=overstock_counts.index,
                    y=overstock_counts.values,
                    title="Products by Overstock Risk Level",
                    labels={'x': 'Risk Level', 'y': 'Number of Products'},
                    color=overstock_counts.index,
                    color_discrete_map={
                        'Low': 'green',
                        'Medium': 'yellow',
                        'High': 'orange',
                        'Critical': 'red'
                    }
                )
                st.plotly_chart(fig2, width='stretch')
    else:
        st.info("No inventory recommendations data available.")

with tab3:
    st.header("Pricing Recommendations")

    if not pricing_recs.empty:
        # Filter to show only recommended actions
        active_recs = pricing_recs[pricing_recs['should_apply'] == True].head(50)

        if not active_recs.empty:
            # Pricing impact visualization
            st.subheader("Pricing Impact Analysis")

            # Group by action type
            action_counts = active_recs['action'].value_counts()

            col1, col2 = st.columns(2)

            with col1:
                fig = px.pie(
                    values=action_counts.values,
                    names=action_counts.index,
                    title="Recommended Actions Distribution"
                )
                st.plotly_chart(fig, width='stretch')

            with col2:
                # Discount distribution
                fig = px.histogram(
                    active_recs,
                    x='discount_pct',
                    title="Discount Percentage Distribution",
                    nbins=20
                )
                st.plotly_chart(fig, width='stretch')

            # Profit margin analysis
            st.subheader("Profit Margin Analysis")
            fig = px.scatter(
                active_recs,
                x='discount_pct',
                y='profit_margin',
                size='current_price',
                color='action',
                title="Discount vs Profit Margin by Action Type",
                labels={
                    'discount_pct': 'Discount %',
                    'profit_margin': 'Profit Margin',
                    'current_price': 'Current Price'
                }
            )
            st.plotly_chart(fig, width='stretch')

            # Recommendations table
            st.subheader("Active Price Recommendations")
            display_cols = ['product_id', 'current_price', 'recommended_price',
                          'discount_pct', 'action', 'profit_margin', 'reasoning']
            st.dataframe(active_recs[display_cols], width='stretch')

            # Summary metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                avg_discount = active_recs['discount_pct'].mean()
                st.metric("Average Discount", f"{avg_discount:.1f}%")
            with col2:
                avg_margin = active_recs['profit_margin'].mean()
                st.metric("Average Profit Margin", f"{avg_margin:.1f}%")
            with col3:
                total_recs = len(active_recs)
                st.metric("Total Recommendations", total_recs)
        else:
            st.info("No active pricing recommendations available.")
    else:
        st.info("No pricing recommendations data available.")

with tab4:
    st.header("Performance Analytics")

    # Use sampled data for analytics
    analytics_data = df.sample(min(1000, len(df)), random_state=42) if len(df) > 1000 else df

    col1, col2 = st.columns(2)

    with col1:
        # Accuracy by category - use sampled data
        acc_data = analytics_data.groupby('category', group_keys=False, observed=False).apply(
            lambda x: 1 - np.abs(x['forecast_q50'] - x['actual']).mean() / x['actual'].mean()
        ).reset_index(name='accuracy')

        fig = px.bar(
            acc_data,
            x='category',
            y='accuracy',
            title="Forecast Accuracy by Category",
            color='accuracy',
            color_continuous_scale='RdYlGn'
        )
        st.plotly_chart(fig, width='stretch')

    with col2:
        # Error distribution - use sampled data
        errors = analytics_data['forecast_q50'] - analytics_data['actual']

        fig = px.histogram(
            x=errors,
            title="Forecast Error Distribution",
            labels={'x': 'Error (units)'},
            nbins=30,
            color_discrete_sequence=['lightblue']
        )
        st.plotly_chart(fig, width='stretch')

    # Summary metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        mean_error = abs(analytics_data['forecast_q50'] - analytics_data['actual']).mean()
        st.metric("Mean Absolute Error", f"{mean_error:.2f}")
    with col2:
        rmse = np.sqrt(((analytics_data['forecast_q50'] - analytics_data['actual']) ** 2).mean())
        st.metric("RMSE", f"{rmse:.2f}")
    with col3:
        accuracy = 1 - abs(analytics_data['forecast_q50'] - analytics_data['actual']).mean() / analytics_data['actual'].mean()
        st.metric("Overall Accuracy", f"{accuracy:.1%}")

    st.info(f"📈 Analytics based on {len(analytics_data)} sample records for optimal performance")

with tab5:
    st.header("🤖 LLM Insights & Analysis")

    # Initialize Gemini API with caching
    @st.cache_resource
    def get_gemini_model():
        """Get cached Gemini model instance."""
        api_key = os.getenv('GOOGLE_API_KEY')
        if api_key:
            genai.configure(api_key=api_key)
            return genai.GenerativeModel('gemini-2.5-flash')
        return None

    @st.cache_data(ttl=1800)  # Cache AI responses for 30 minutes
    def get_ai_response_cached(question, context):
        """Cached AI response to avoid repeated API calls."""
        model = get_gemini_model()
        if not model:
            return "AI model not available. Please check API key configuration."

        try:
            prompt = f"""
            You are an expert demand forecasting analyst. Answer concisely and provide actionable insights.

            Context: {context}

            Question: {question}

            Provide a brief, actionable response (max 150 words).
            """

            response = model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            return f"AI Error: {str(e)}"

    # Initialize model
    gemini_model = get_gemini_model()

    if gemini_model:
        # LLM Insights from reports
        if not llm_insights.empty:
            st.subheader("AI-Generated Product Insights")

            # Product selector
            selected_product = st.selectbox(
                "Select Product for Detailed Analysis",
                options=llm_insights['product_id'].unique(),
                key="llm_product_selector"
            )

            if selected_product:
                product_insight = llm_insights[llm_insights['product_id'] == selected_product]

                if not product_insight.empty:
                    insight_data = product_insight.iloc[0]

                    # Display insight metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Stockout Risk", f"{insight_data.get('stockout_risk_pct', 0):.1f}%")
                    with col2:
                        st.metric("Overstock Risk", f"{insight_data.get('overstock_risk_pct', 0):.1f}%")
                    with col3:
                        confidence = insight_data.get('confidence', 0)
                        st.metric("AI Confidence", f"{confidence:.2f}")

                    # Display formatted insight text
                    st.subheader("AI Analysis")
                    insight_text = insight_data.get('insight_text', 'No insight available')
                    st.markdown(insight_text)

                    # Interactive LLM chat
                    st.subheader("💬 Ask AI Analyst")

                    user_question = st.text_input(
                        "Ask a question about this product or demand forecasting:",
                        key=f"llm_question_{selected_product}",
                        placeholder="e.g., How can I optimize inventory for this product?"
                    )

                    if user_question and st.button("Get AI Analysis", key=f"analyze_{selected_product}"):
                        with st.spinner("AI is analyzing..."):
                            # Create context from product data
                            context = f"""
                            Product ID: {selected_product}
                            Stockout Risk: {insight_data.get('stockout_risk_pct', 0):.1f}%
                            Overstock Risk: {insight_data.get('overstock_risk_pct', 0):.1f}%
                            AI Confidence: {insight_data.get('confidence', 0):.2f}
                            Previous Analysis: {insight_text[:300]}...
                            """

                            # Get cached AI response
                            ai_response = get_ai_response_cached(user_question, context)

                            st.success("AI Analysis:")
                            st.write(ai_response)

                            # Add helpful tips
                            st.info("💡 Tip: Responses are cached for 30 minutes to improve performance")
                else:
                    st.info("No insights available for selected product.")

            # Summary of all insights
            st.subheader("Insights Summary")
            col1, col2, col3 = st.columns(3)

            with col1:
                high_risk = len(llm_insights[llm_insights['stockout_risk_pct'] > 50])
                st.metric("High Risk Products", high_risk)

            with col2:
                avg_confidence = llm_insights['confidence'].mean() if 'confidence' in llm_insights.columns else 0
                st.metric("Avg AI Confidence", f"{avg_confidence:.2f}")

            with col3:
                total_insights = len(llm_insights)
                st.metric("Total AI Insights", total_insights)

        else:
            st.info("No LLM insights data available.")
    else:
        st.warning("⚠️ Gemini API key not configured. Please set GOOGLE_API_KEY in .env file to enable AI features.")

with tab6:
    st.header("🔍 Model Performance")

    if model_metrics:
        st.subheader("Model Metrics Overview")

        # Display model metrics in a nice format
        lightgbm_metrics = model_metrics.get('lightgbm', {})

        # Create metrics cards
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            r2_score = lightgbm_metrics.get('r2_score', 0)
            st.metric("R² Score", f"{r2_score:.3f}")

        with col2:
            coverage = lightgbm_metrics.get('coverage_90%', 0)
            st.metric("90% Coverage", f"{coverage:.1%}")

        with col3:
            mae_q50 = lightgbm_metrics.get('q50_mae', 0)
            st.metric("Q50 MAE", f"{mae_q50:.3f}")

        with col4:
            rmse_q50 = lightgbm_metrics.get('q50_rmse', 0)
            st.metric("Q50 RMSE", f"{rmse_q50:.3f}")

        # Detailed metrics table
        st.subheader("Detailed Model Performance")

        metrics_data = []
        for quantile in ['q05', 'q25', 'q50', 'q75', 'q95']:
            metrics_data.append({
                'Quantile': quantile.upper(),
                'Pinball Loss': lightgbm_metrics.get(f'{quantile}_pinball_loss', 0),
                'MAE': lightgbm_metrics.get(f'{quantile}_mae', 0),
                'RMSE': lightgbm_metrics.get(f'{quantile}_rmse', 0)
            })

        metrics_df = pd.DataFrame(metrics_data)
        st.dataframe(metrics_df, width='stretch')

        # Feature importance visualization
        if not shap_values.empty:
            st.subheader("Feature Importance (SHAP Values)")

            # Top 20 features
            top_features = shap_values.head(20)

            fig = px.bar(
                top_features,
                x='mean_abs_shap',
                y='feature',
                orientation='h',
                title="Top 20 Most Important Features",
                labels={'mean_abs_shap': 'SHAP Importance', 'feature': 'Feature'}
            )
            fig.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig, width='stretch')

        # Charts from report_charts folder
        st.subheader("Performance Charts")

        chart_files = [
            "../reports/report_charts/chart1_model_performance.png",
            "../reports/report_charts/chart2_business_impact.png",
            "../reports/report_charts/chart3_forecast_quality.png",
            "../reports/report_charts/chart4_feature_importance.png"
        ]

        for chart_file in chart_files:
            if os.path.exists(chart_file):
                chart_num = chart_file.split('/')[-1].split('_')[0].replace('chart', '')
                chart_name = f"Chart {chart_num}: Enhanced Dashboard View"
                st.image(chart_file, caption=chart_name)
            else:
                st.info(f"Chart not found: {chart_file}")

    else:
        st.info("No model metrics data available.")

# Footer
st.sidebar.markdown("---")
st.sidebar.info(
    """
    **SmartGrocy v4.0 Enhanced**

    Features:
    • Real-time forecasting dashboard
    • AI-powered LLM insights
    • Interactive Gemini API integration
    • Comprehensive business analytics
    • Model performance monitoring

    Last updated: 2025-11-18

    [Documentation](docs/) | [GitHub](https://github.com/ducanh0405/datastorm)
    """
)

# Add refresh button in sidebar
if st.sidebar.button("🔄 Refresh Data"):
    st.cache_data.clear()
    st.rerun()
