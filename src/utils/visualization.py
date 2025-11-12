"""
Visualization Utilities for Dashboard
======================================
Creates charts and plots for forecasting results.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
from plotly.subplots import make_subplots

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def plot_forecast_timeseries(
    df: pd.DataFrame,
    product_id: str,
    store_id: str,
    weeks: list[int] | None = None,
    save_path: Path | None = None
) -> go.Figure:
    """
    Plot time series forecast with prediction intervals.

    Args:
        df: DataFrame with actual and forecast columns
        product_id: Product ID to plot
        store_id: Store ID to plot
        weeks: List of weeks to plot (if None, plots all)
        save_path: Path to save figure

    Returns:
        Plotly figure
    """
    # Filter data
    plot_df = df[(df['PRODUCT_ID'] == product_id) & (df['STORE_ID'] == store_id)].copy()

    if weeks is not None:
        plot_df = plot_df[plot_df['WEEK_NO'].isin(weeks)]

    plot_df = plot_df.sort_values('WEEK_NO')

    # Create figure
    fig = go.Figure()

    # Add actual values
    if 'actual' in plot_df.columns:
        fig.add_trace(        go.Scatter(
            x=plot_df['WEEK_NO'],
            y=plot_df['actual'],
            name='Actual',
            mode='lines+markers',
            line={'color': 'blue', 'width': 2},
            marker={'size': 8}
        ))

    # Add forecast median
    if 'forecast_q50' in plot_df.columns:
        fig.add_trace(        go.Scatter(
            x=plot_df['WEEK_NO'],
            y=plot_df['forecast_q50'],
            name='Forecast (Q50)',
            mode='lines+markers',
            line={'color': 'green', 'width': 2, 'dash': 'dash'},
            marker={'size': 6}
        ))

    # Add prediction interval
    if 'forecast_q05' in plot_df.columns and 'forecast_q95' in plot_df.columns:
        fig.add_trace(go.Scatter(
            x=plot_df['WEEK_NO'],
            y=plot_df['forecast_q95'],
            name='Upper Bound (Q95)',
            mode='lines',
            line={'width': 0},
            showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=plot_df['WEEK_NO'],
            y=plot_df['forecast_q05'],
            name='Lower Bound (Q05)',
            mode='lines',
            line={'width': 0},
            fillcolor='rgba(0,100,80,0.2)',
            fill='tonexty',
            showlegend=True
        ))

    # Update layout
    fig.update_layout(
        title=f'Forecast for Product {product_id} at Store {store_id}',
        xaxis_title='Week Number',
        yaxis_title='Sales Value',
        hovermode='x unified',
        template='plotly_white',
        height=500
    )

    if save_path:
        fig.write_html(str(save_path))

    return fig


def plot_prediction_accuracy(
    predictions: pd.DataFrame,
    save_path: Path | None = None
) -> go.Figure:
    """
    Plot prediction accuracy metrics.

    Args:
        predictions: DataFrame with actual and forecast columns
        save_path: Path to save figure

    Returns:
        Plotly figure
    """
    if 'actual' not in predictions.columns or 'forecast_q50' not in predictions.columns:
        raise ValueError("Missing required columns: 'actual' and 'forecast_q50'")

    # Calculate errors
    predictions = predictions.copy()
    predictions['error'] = predictions['actual'] - predictions['forecast_q50']
    predictions['abs_error'] = predictions['error'].abs()
    predictions['pct_error'] = (predictions['error'] / (predictions['actual'] + 1e-6)) * 100

    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Error Distribution', 'Absolute Error by Week',
                       'Percentage Error', 'Predicted vs Actual'),
        specs=[[{"type": "histogram"}, {"type": "scatter"}],
               [{"type": "histogram"}, {"type": "scatter"}]]
    )

    # Error distribution
    fig.add_trace(
        go.Histogram(x=predictions['error'], name='Error', nbinsx=50),
        row=1, col=1
    )

    # Absolute error by week
    error_by_week = predictions.groupby('WEEK_NO')['abs_error'].mean().reset_index()
    fig.add_trace(
        go.Scatter(x=error_by_week['WEEK_NO'], y=error_by_week['abs_error'],
                  name='Mean Abs Error', mode='lines+markers'),
        row=1, col=2
    )

    # Percentage error
    fig.add_trace(
        go.Histogram(x=predictions['pct_error'], name='Pct Error', nbinsx=50),
        row=2, col=1
    )

    # Predicted vs Actual
    fig.add_trace(
        go.Scatter(x=predictions['actual'], y=predictions['forecast_q50'],
                  mode='markers', name='Predictions',
                  marker={'size': 4, 'opacity': 0.6}),
        row=2, col=2
    )
    # Add diagonal line
    max_val = max(predictions['actual'].max(), predictions['forecast_q50'].max())
    fig.add_trace(
        go.Scatter(x=[0, max_val], y=[0, max_val], mode='lines',
                  name='Perfect', line={'dash': 'dash', 'color': 'red'}),
        row=2, col=2
    )

    fig.update_layout(
        title='Prediction Accuracy Metrics',
        height=800,
        template='plotly_white',
        showlegend=True
    )

    if save_path:
        fig.write_html(str(save_path))

    return fig


def plot_feature_importance(
    model,
    feature_names: list[str],
    top_n: int = 20,
    save_path: Path | None = None
) -> go.Figure:
    """
    Plot feature importance from trained model.

    Args:
        model: Trained LightGBM model
        feature_names: List of feature names
        top_n: Number of top features to show
        save_path: Path to save figure

    Returns:
        Plotly figure
    """
    importance = model.feature_importances_
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False).head(top_n)

    fig = go.Figure(go.Bar(
        x=importance_df['importance'],
        y=importance_df['feature'],
        orientation='h',
        marker={'color': importance_df['importance'], 'colorscale': 'Viridis'}
    ))

    fig.update_layout(
        title=f'Top {top_n} Feature Importance',
        xaxis_title='Importance',
        yaxis_title='Feature',
        template='plotly_white',
        height=600
    )

    if save_path:
        fig.write_html(str(save_path))

    return fig


def plot_quantile_comparison(
    predictions: pd.DataFrame,
    save_path: Path | None = None
) -> go.Figure:
    """
    Plot comparison between different quantiles.

    Args:
        predictions: DataFrame with quantile predictions
        save_path: Path to save figure

    Returns:
        Plotly figure
    """
    fig = go.Figure()

    if 'actual' in predictions.columns:
        fig.add_trace(go.Scatter(
            x=predictions['WEEK_NO'],
            y=predictions['actual'],
            name='Actual',
            mode='markers',
            marker={'size': 4, 'color': 'blue', 'opacity': 0.7}
        ))

    if 'forecast_q05' in predictions.columns:
        fig.add_trace(go.Scatter(
            x=predictions['WEEK_NO'],
            y=predictions['forecast_q05'],
            name='Q05 (Lower Bound)',
            mode='lines',
            line={'color': 'red', 'dash': 'dot'}
        ))

    if 'forecast_q50' in predictions.columns:
        fig.add_trace(go.Scatter(
            x=predictions['WEEK_NO'],
            y=predictions['forecast_q50'],
            name='Q50 (Median)',
            mode='lines',
            line={'color': 'green', 'width': 3}
        ))

    if 'forecast_q95' in predictions.columns:
        fig.add_trace(go.Scatter(
            x=predictions['WEEK_NO'],
            y=predictions['forecast_q95'],
            name='Q95 (Upper Bound)',
            mode='lines',
            line={'color': 'orange', 'dash': 'dot'}
        ))

    fig.update_layout(
        title='Quantile Forecast Comparison',
        xaxis_title='Week Number',
        yaxis_title='Sales Value',
        hovermode='x unified',
        template='plotly_white',
        height=500
    )

    if save_path:
        fig.write_html(str(save_path))

    return fig


def create_dashboard_summary(
    predictions: pd.DataFrame,
    metrics: dict,
    output_dir: Path
):
    """
    Create complete dashboard with multiple visualizations.

    Args:
        predictions: DataFrame with predictions
        metrics: Dictionary of evaluation metrics
        output_dir: Directory to save dashboard files
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Prediction accuracy plot
    fig1 = plot_prediction_accuracy(predictions)
    fig1.write_html(str(output_dir / 'prediction_accuracy.html'))

    # 2. Quantile comparison plot
    fig2 = plot_quantile_comparison(predictions)
    fig2.write_html(str(output_dir / 'quantile_comparison.html'))

    # 3. Sample time series plots
    sample_products = predictions[['PRODUCT_ID', 'STORE_ID']].drop_duplicates().head(5)
    for _idx, row in sample_products.iterrows():
        fig = plot_forecast_timeseries(
            predictions,
            row['PRODUCT_ID'],
            row['STORE_ID']
        )
        fig.write_html(
            str(output_dir / f"forecast_{row['PRODUCT_ID']}_{row['STORE_ID']}.html")
        )

    # 4. Feature importance plot (if we have a model)
    try:
        from src.pipelines._05_prediction import QuantileForecaster
        forecaster = QuantileForecaster()
        feature_names = forecaster.feature_config['all_features']

        fig3 = plot_feature_importance(
            forecaster.models[0.50],
            feature_names
        )
        fig3.write_html(str(output_dir / 'feature_importance.html'))
    except Exception as e:
        print(f"Could not create feature importance plot: {e}")

    # 5. Metrics summary
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(output_dir / 'metrics_summary.csv', index=False)

    # 6. Create index HTML
    create_dashboard_index(output_dir, metrics, predictions)

    print(f"Dashboard files saved to: {output_dir}")


def create_dashboard_index(
    output_dir: Path,
    metrics: dict,
    predictions: pd.DataFrame
):
    """Create an index HTML file for the dashboard."""

    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>SmartGrocy Dashboard</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        .header {{
            background-color: #2c3e50;
            color: white;
            padding: 20px;
            border-radius: 5px;
        }}
        .metric-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }}
        .metric-card {{
            background: white;
            padding: 15px;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .metric-value {{
            font-size: 24px;
            font-weight: bold;
            color: #2c3e50;
        }}
        .metric-label {{
            color: #7f8c8d;
            font-size: 12px;
            text-transform: uppercase;
        }}
        .charts-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
        }}
        .chart-card {{
            background: white;
            padding: 15px;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        iframe {{
            width: 100%;
            height: 400px;
            border: none;
            border-radius: 5px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>📊 SmartGrocy Dashboard</h1>
        <p>Probabilistic forecasting for retail optimization</p>
    </div>

    <h2>📈 Key Metrics</h2>
    <div class="metric-grid">
        <div class="metric-card">
            <div class="metric-value">{len(predictions):,}</div>
            <div class="metric-label">Total Predictions</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{metrics.get('prediction_interval_coverage', 0)*100:.1f}%</div>
            <div class="metric-label">Coverage (90% CI)</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{metrics.get('q50_pinball_loss', 0):.4f}</div>
            <div class="metric-label">Q50 Pinball Loss</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{metrics.get('q50_rmse', 0):.4f}</div>
            <div class="metric-label">Q50 RMSE</div>
        </div>
    </div>

    <h2>📊 Charts</h2>
    <div class="charts-grid">
        <div class="chart-card">
            <h3>Prediction Accuracy</h3>
            <iframe src="prediction_accuracy.html"></iframe>
        </div>
        <div class="chart-card">
            <h3>Quantile Comparison</h3>
            <iframe src="quantile_comparison.html"></iframe>
        </div>
    </div>

    <h2>🔍 Individual Product Forecasts</h2>
    <div class="charts-grid">
"""

    # Add sample product charts
    sample_products = predictions[['PRODUCT_ID', 'STORE_ID']].drop_duplicates().head(3)
    for _idx, row in sample_products.iterrows():
        html_content += f"""
        <div class="chart-card">
            <h3>Product {row['PRODUCT_ID']} - Store {row['STORE_ID']}</h3>
            <iframe src="forecast_{row['PRODUCT_ID']}_{row['STORE_ID']}.html"></iframe>
        </div>
"""

    html_content += """
    </div>

    <h2>📋 Additional Resources</h2>
    <ul>
        <li><a href="feature_importance.html">Feature Importance Analysis</a></li>
        <li><a href="metrics_summary.csv">Detailed Metrics (CSV)</a></li>
        <li><a href="predictions_test_set.csv">Full Predictions (CSV)</a></li>
    </ul>

    <p><small>Generated by SmartGrocy Pipeline</small></p>
</body>
</html>
"""

    with open(output_dir / 'index.html', 'w', encoding='utf-8') as f:
        f.write(html_content)
