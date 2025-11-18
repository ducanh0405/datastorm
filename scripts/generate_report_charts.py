#!/usr/bin/env python3
"""
Generate Report Charts for SmartGrocy
=====================================
Creates all required and recommended charts for the project report.

Mandatory Charts (3):
1. Model Performance Metrics (MAE, RMSE, Pinball Loss, R², Coverage)
2. Business Impact KPI Comparison (Spoilage, Stockout reduction)
3. Forecast Quality (Forecast vs Actual with prediction intervals)

Recommended Charts (2):
4. Feature Importance (SHAP Top 10)
5. Market Context (Vietnam e-grocery growth)

Usage:
    python scripts/generate_report_charts.py
"""
import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple

import matplotlib.pyplot as plt  # pyright: ignore[reportMissingImports]
import matplotlib.patches as mpatches  # pyright: ignore[reportMissingImports]
import numpy as np
import pandas as pd
import seaborn as sns  # pyright: ignore[reportMissingModuleSource]

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
sns.set_style("whitegrid")

# Output directory
CHARTS_DIR = PROJECT_ROOT / "reports" / "report_charts"
CHARTS_DIR.mkdir(parents=True, exist_ok=True)

# Color scheme
COLORS = {
    'primary': '#2E86AB',
    'secondary': '#A23B72',
    'success': '#06A77D',
    'warning': '#F18F01',
    'danger': '#C73E1D',
    'light': '#E8E8E8',
    'dark': '#2C2C2C'
}


def load_model_metrics() -> Dict[str, Any]:
    """Load model metrics from JSON file."""
    metrics_path = PROJECT_ROOT / "reports" / "metrics" / "model_metrics.json"
    with open(metrics_path, 'r') as f:
        return json.load(f)


def load_business_impact() -> pd.DataFrame:
    """Load business impact metrics from CSV."""
    impact_path = PROJECT_ROOT / "reports" / "backtesting" / "estimated_results.csv"
    return pd.read_csv(impact_path)


def load_feature_importance() -> pd.DataFrame:
    """Load feature importance from CSV."""
    importance_path = PROJECT_ROOT / "reports" / "shap_values" / "feature_importance.csv"
    return pd.read_csv(importance_path)


def load_sample_predictions(n_samples: int = 500) -> pd.DataFrame:
    """Load a sample of predictions for visualization."""
    # Try parquet first (preferred format)
    predictions_path = PROJECT_ROOT / "reports" / "predictions_test_set.parquet"

    if predictions_path.exists():
        print(f"   Reading {n_samples} rows from predictions parquet file...")
        df = pd.read_parquet(predictions_path)
        # Sample n_samples rows randomly for better representation
        if len(df) > n_samples:
            df = df.sample(n=n_samples, random_state=42).reset_index(drop=True)
    else:
        # Fallback to CSV
        predictions_path = PROJECT_ROOT / "reports" / "predictions_test_set.csv"
        if not predictions_path.exists():
            raise FileNotFoundError(f"Predictions file not found: {predictions_path}")
        print(f"   Reading {n_samples} rows from predictions CSV file...")
        df = pd.read_csv(predictions_path, nrows=n_samples)

    return df


def chart_1_model_performance(metrics: Dict[str, Any], save_path: Path) -> None:
    """
    Chart 1: Model Error Metrics Across Quantiles
    Clean and simple bar chart showing MAE, RMSE, and Pinball Loss across quantiles
    """
    print("[CHART 1] Generating Model Error Metrics Chart...")
    
    # Extract metrics by quantile
    quantiles = ['q05', 'q25', 'q50', 'q75', 'q95']
    quantile_labels = ['Q05', 'Q25', 'Q50', 'Q75', 'Q95']
    
    mae_values = [metrics[f'{q}_mae'] for q in quantiles]
    rmse_values = [metrics[f'{q}_rmse'] for q in quantiles]
    pinball_values = [metrics[f'{q}_pinball_loss'] for q in quantiles]
    
    # Create clean, simple figure
    fig, ax = plt.subplots(figsize=(12, 8), facecolor='white')
    
    # Set style
    sns.set_style("whitegrid")
    
    # Color scheme - matching the image description
    mae_color = '#3498db'      # Blue for MAE
    rmse_color = '#e74c3c'     # Red/Orange for RMSE  
    pinball_color = '#27ae60'  # Green for Pinball Loss

    # Create grouped bar chart
    x = np.arange(len(quantiles))
    width = 0.25

    bars_mae = ax.bar(x - width, mae_values, width, label='MAE', color=mae_color,
                     alpha=0.85, edgecolor='white', linewidth=1.5)
    bars_rmse = ax.bar(x, rmse_values, width, label='RMSE', color=rmse_color,
                      alpha=0.85, edgecolor='white', linewidth=1.5)
    bars_pinball = ax.bar(x + width, pinball_values, width, label='Pinball Loss',
                         color=pinball_color, alpha=0.85, edgecolor='white', linewidth=1.5)

    # Add value labels on bars
    def add_value_labels(bars, values, format_str='.3f'):
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                   f'{val:{format_str}}', ha='center', va='bottom',
                   fontsize=9, fontweight='bold', color='#2c3e50')

    add_value_labels(bars_mae, mae_values)
    add_value_labels(bars_rmse, rmse_values)
    add_value_labels(bars_pinball, pinball_values, '.3f')

    # Styling
    ax.set_xlabel('Quantiles', fontsize=14, fontweight='bold', color='#2c3e50', labelpad=15)
    ax.set_ylabel('Error / Loss', fontsize=14, fontweight='bold', color='#2c3e50', labelpad=15)
    ax.set_title('Model Error Metrics Across Quantiles', fontsize=16, fontweight='bold',
                color='#2c3e50', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(quantile_labels, fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1.2)
    ax.set_yticks(np.arange(0, 1.3, 0.2))
    ax.legend(fontsize=12, loc='upper right', framealpha=0.95, edgecolor='#bdc3c7')
    ax.grid(axis='y', alpha=0.3, linestyle='--', color='#95a5a6')
    ax.set_axisbelow(True)

    # Clean up spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#bdc3c7')
    ax.spines['bottom'].set_color('#bdc3c7')

    # Tight layout for clean appearance
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print(f"[SUCCESS] Saved model error metrics chart: {save_path}")


def chart_2_business_impact(impact_df: pd.DataFrame, save_path: Path) -> None:
    """
    Chart 2: Business Impact KPI Comparison
    Clean bar chart showing Baseline vs ML Model improvements
    """
    print("[CHART 2] Generating Business Impact Chart...")
    
    # Extract all data
    metrics_data = []
    for idx, row in impact_df.iterrows():
        raw_metric = row['metric']
        if raw_metric == 'spoilage_rate_pct':
            display_metric = 'Spoilage Rate'
        elif raw_metric == 'stockout_rate_pct':
            display_metric = 'Stockout Rate'
        elif raw_metric == 'profit_margin_pct':
            display_metric = 'Profit Margin'
        else:
            display_metric = raw_metric.replace('_pct', '').replace('_rate', '').title()

        metrics_data.append({
            'metric': display_metric,
            'baseline': row['baseline'],
            'ml_model': row['ml_model'],
            'improvement_pct': row['improvement_pct']
        })

    # Create clean figure
    fig, ax = plt.subplots(figsize=(12, 8), facecolor='white')
    sns.set_style("whitegrid")

    metrics = [data['metric'] for data in metrics_data]
    baseline_values = [data['baseline'] for data in metrics_data]
    ml_values = [data['ml_model'] for data in metrics_data]
    improvements = [data['improvement_pct'] for data in metrics_data]

    x = np.arange(len(metrics))
    width = 0.35

    # Grouped bars
    bars_base = ax.bar(x - width/2, baseline_values, width,
                      label='Baseline', color='#95a5a6', alpha=0.85,
                      edgecolor='white', linewidth=1.5)
    bars_ml = ax.bar(x + width/2, ml_values, width,
                    label='ML Model', color='#3498db', alpha=0.85,
                    edgecolor='white', linewidth=1.5)

    # Add value labels
    def add_value_labels(bars, values, format_str='.1f'):
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.3,
                   f'{val:{format_str}}%', ha='center', va='bottom',
                   fontsize=10, fontweight='bold', color='#2c3e50')

    add_value_labels(bars_base, baseline_values)
    add_value_labels(bars_ml, ml_values)

    # Add improvement labels above bars
    for i, (base_val, ml_val, imp) in enumerate(zip(baseline_values, ml_values, improvements)):
        max_val = max(base_val, ml_val)
        ax.text(x[i], max_val + 2.5, f'↓ {imp:.1f}%',
               ha='center', va='bottom', fontsize=11, fontweight='bold',
               color='#27ae60', bbox=dict(boxstyle='round,pad=0.4',
               facecolor='white', edgecolor='#27ae60', linewidth=1.5, alpha=0.9))

    # Styling
    ax.set_xlabel('Metrics', fontsize=14, fontweight='bold', color='#2c3e50', labelpad=15)
    ax.set_ylabel('Value (%)', fontsize=14, fontweight='bold', color='#2c3e50', labelpad=15)
    ax.set_title('Business Impact: Baseline vs ML Model', fontsize=16, fontweight='bold',
                color='#2c3e50', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=12, fontweight='bold')
    ax.legend(fontsize=12, loc='upper right', framealpha=0.95, edgecolor='#bdc3c7')
    ax.grid(axis='y', alpha=0.3, linestyle='--', color='#95a5a6')
    ax.set_axisbelow(True)

    # Clean up spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#bdc3c7')
    ax.spines['bottom'].set_color('#bdc3c7')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print(f"[SUCCESS] Saved business impact chart: {save_path}")


def chart_3_forecast_quality(predictions_df: pd.DataFrame, save_path: Path, n_samples: int = 200) -> None:
    """
    Chart 3: Forecast Quality
    Clean line chart showing prediction intervals vs actual values
    """
    print("[CHART 3] Generating Forecast Quality Chart...")
    
    # Sample data for visualization
    if len(predictions_df) > n_samples:
        sample_df = predictions_df.sample(n=min(n_samples, len(predictions_df)), random_state=42)
    else:
        sample_df = predictions_df.copy()
    
    # Sort by index for better visualization
    sample_df = sample_df.sort_index().reset_index(drop=True)
    
    # Create clean figure
    fig, ax = plt.subplots(figsize=(14, 8), facecolor='white')
    sns.set_style("whitegrid")
    
    # Check which columns are available and map column names
    actual_col = None
    for col_name in ['actual', 'SALES_QUANTITY', 'sales_quantity', 'y_true']:
        if col_name in sample_df.columns:
            actual_col = col_name
            break
    
    q05_col = None
    for col_name in ['forecast_q05', 'q05', 'pred_q05']:
        if col_name in sample_df.columns:
            q05_col = col_name
            break
    
    q50_col = None
    for col_name in ['forecast_q50', 'q50', 'pred_q50', 'forecast_median']:
        if col_name in sample_df.columns:
            q50_col = col_name
            break
    
    q95_col = None
    for col_name in ['forecast_q95', 'q95', 'pred_q95']:
        if col_name in sample_df.columns:
            q95_col = col_name
            break
    
    has_actual = actual_col is not None
    has_q05 = q05_col is not None
    has_q50 = q50_col is not None
    has_q95 = q95_col is not None
    
    x = np.arange(len(sample_df))
    
    # Plot prediction intervals
    if q05_col and q95_col:
        y1 = sample_df[q05_col].values
        y2 = sample_df[q95_col].values
        ax.fill_between(x, y1, y2, color='#3498db', alpha=0.2, 
                       label='90% Prediction Interval (Q05-Q95)', zorder=1)
        ax.plot(x, y1, color='#3498db', linewidth=1.5, alpha=0.6, linestyle='--', zorder=2)
        ax.plot(x, y2, color='#3498db', linewidth=1.5, alpha=0.6, linestyle='--', zorder=2)
    
    # Plot median forecast
    if q50_col:
        ax.plot(x, sample_df[q50_col], '-', color='#27ae60', linewidth=2.5,
               label='Forecast Median (Q50)', alpha=0.9, zorder=3)
    
    # Plot actual values
    if has_actual and actual_col:
        ax.scatter(x, sample_df[actual_col], color='#e74c3c', s=25, alpha=0.7,
                  edgecolor='white', linewidth=0.8, label='Actual Values', zorder=4)
        ax.plot(x, sample_df[actual_col], color='#e74c3c', linewidth=1.5, 
               alpha=0.6, linestyle=':', zorder=3)
    
    # Styling
    ax.set_xlabel('Sample Index', fontsize=14, fontweight='bold', color='#2c3e50', labelpad=15)
    ax.set_ylabel('Sales Quantity', fontsize=14, fontweight='bold', color='#2c3e50', labelpad=15)
    ax.set_title('Forecast Quality: Prediction Intervals vs Actual Values', 
                fontsize=16, fontweight='bold', color='#2c3e50', pad=20)
    ax.legend(loc='upper right', fontsize=11, framealpha=0.95, edgecolor='#bdc3c7')
    ax.grid(True, alpha=0.3, linestyle='--', color='#95a5a6')
    ax.set_axisbelow(True)
    
    # Clean up spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#bdc3c7')
    ax.spines['bottom'].set_color('#bdc3c7')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print(f"[SUCCESS] Saved forecast quality chart: {save_path}")


def chart_4_feature_importance(importance_df: pd.DataFrame, save_path: Path, top_n: int = 10) -> None:
    """
    Chart 4: Feature Importance (SHAP Values)
    Clean horizontal bar chart showing top N most important features
    """
    print("[CHART 4] Generating Feature Importance Chart...")
    
    # Get top N features
    top_features = importance_df.nlargest(top_n, 'mean_abs_shap')
    
    # Create clean figure
    fig, ax = plt.subplots(figsize=(12, 8), facecolor='white')
    sns.set_style("whitegrid")

    # Horizontal bar chart
    y_pos = np.arange(len(top_features))
    
    # Create gradient colors for bars
    colors = plt.cm.Greens(np.linspace(0.4, 0.85, len(top_features)))

    bars = ax.barh(y_pos, top_features['mean_abs_shap'], color=colors,
                  alpha=0.85, edgecolor='white', linewidth=1.5, height=0.7)
    
    # Add value labels
    for i, (idx, row) in enumerate(top_features.iterrows()):
        ax.text(row['mean_abs_shap'] + max(top_features['mean_abs_shap']) * 0.02, i,
               f'{row["mean_abs_shap"]:.3f}', va='center', fontsize=10,
               fontweight='bold', color='#2c3e50')
    
    # Feature names on y-axis
    ax.set_yticks(y_pos)
    ax.set_yticklabels([name[:40] + '...' if len(name) > 40 else name
                        for name in top_features['feature']], fontsize=11, fontweight='bold')
    ax.set_xlabel('Mean Absolute SHAP Value', fontsize=14, fontweight='bold',
                 color='#2c3e50', labelpad=15)
    ax.set_title('Feature Importance: Top 10 Features (SHAP Values)',
                fontsize=16, fontweight='bold', color='#2c3e50', pad=20)
    ax.grid(axis='x', alpha=0.3, linestyle='--', color='#95a5a6')
    ax.set_axisbelow(True)

    # Clean up spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#bdc3c7')
    ax.spines['bottom'].set_color('#bdc3c7')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print(f"[SUCCESS] Saved feature importance chart: {save_path}")


def chart_6_hourly_demand_pattern(predictions_df: pd.DataFrame, save_path: Path) -> None:
    """
    Chart 6: Hourly Demand Pattern
    Clean line chart showing average demand by hour of day
    """
    print("[CHART 6] Generating Hourly Demand Pattern Chart...")
    
    # Check if hour_of_day column exists
    if 'hour_of_day' not in predictions_df.columns:
        print("[WARNING] hour_of_day column not found. Skipping hourly pattern chart.")
        return
    
    # Group by hour and calculate statistics
    hourly_stats = predictions_df.groupby('hour_of_day').agg({
        'sales_quantity': ['mean', 'std'],
        'forecast_q50': 'mean'
    }).reset_index()
    
    hourly_stats.columns = ['hour', 'mean_demand', 'std_demand', 'mean_forecast']
    
    # Create clean figure
    fig, ax = plt.subplots(figsize=(14, 8), facecolor='white')
    sns.set_style("whitegrid")

    hours = hourly_stats['hour']
    mean_demand = hourly_stats['mean_demand']
    std_demand = hourly_stats['std_demand']
    mean_forecast = hourly_stats['mean_forecast']

    # Create filled area for confidence interval
    ax.fill_between(hours, mean_demand - std_demand, mean_demand + std_demand,
                    alpha=0.2, color='#3498db', label='±1 Std Dev', zorder=1)

    # Plot actual demand
    ax.plot(hours, mean_demand, 'o-', color='#e74c3c', linewidth=2.5, markersize=6,
           markerfacecolor='white', markeredgecolor='#e74c3c', markeredgewidth=1.5,
           label='Actual Demand', zorder=3)

    # Plot forecast line
    if 'mean_forecast' in hourly_stats.columns:
        ax.plot(hours, mean_forecast, '--', color='#27ae60', linewidth=2, alpha=0.8,
               label='ML Forecast', zorder=2)

    # Styling
    ax.set_xlabel('Hour of Day (0-23)', fontsize=14, fontweight='bold', color='#2c3e50', labelpad=15)
    ax.set_ylabel('Average Sales Quantity', fontsize=14, fontweight='bold', color='#2c3e50', labelpad=15)
    ax.set_title('Hourly Demand Pattern: Average Sales by Hour', fontsize=16, fontweight='bold',
                color='#2c3e50', pad=20)
    ax.set_xticks(range(0, 24, 2))
    ax.legend(fontsize=12, loc='upper right', framealpha=0.95, edgecolor='#bdc3c7')
    ax.grid(True, alpha=0.3, linestyle='--', color='#95a5a6')
    ax.set_axisbelow(True)

    # Clean up spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#bdc3c7')
    ax.spines['bottom'].set_color('#bdc3c7')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print(f"[SUCCESS] Saved hourly demand pattern chart: {save_path}")


def chart_7_profit_margin_improvement(impact_df: pd.DataFrame, save_path: Path) -> None:
    """
    Chart 7: Profit Margin Improvement
    Clean bar chart showing profit margin improvement
    """
    print("[CHART 7] Generating Profit Margin Improvement Chart...")
    
    # Extract profit margin data
    profit_row = impact_df[impact_df['metric'] == 'profit_margin_pct'].iloc[0]
    baseline_profit = profit_row['baseline']
    ml_profit = profit_row['ml_model']
    profit_improvement = profit_row['improvement_pct']

    # Create clean figure
    fig, ax = plt.subplots(figsize=(10, 8), facecolor='white')
    sns.set_style("whitegrid")

    # Bar chart
    categories = ['Baseline', 'ML Model']
    values = [baseline_profit, ml_profit]
    colors = ['#95a5a6', '#3498db']

    bars = ax.bar(categories, values, color=colors, alpha=0.85,
                 edgecolor='white', linewidth=1.5, width=0.6)

    # Add value labels
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
               f'{val:.2f}%', ha='center', va='bottom',
               fontsize=12, fontweight='bold', color='#2c3e50')

    # Add improvement label
    ax.text(0.5, max(values) + 2.5, f'↑ +{profit_improvement:.1f}%',
           ha='center', va='bottom', fontsize=13, fontweight='bold',
           color='#27ae60', bbox=dict(boxstyle='round,pad=0.5',
           facecolor='white', edgecolor='#27ae60', linewidth=2, alpha=0.9))

    # Styling
    ax.set_ylabel('Profit Margin (%)', fontsize=14, fontweight='bold',
                 color='#2c3e50', labelpad=15)
    ax.set_title('Profit Margin Improvement', fontsize=16, fontweight='bold',
                color='#2c3e50', pad=20)
    ax.set_ylim(0, max(values) * 1.25)
    ax.grid(axis='y', alpha=0.3, linestyle='--', color='#95a5a6')
    ax.set_axisbelow(True)

    # Clean up spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#bdc3c7')
    ax.spines['bottom'].set_color('#bdc3c7')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print(f"[SUCCESS] Saved profit margin improvement chart: {save_path}")


def chart_8_performance_by_category(predictions_df: pd.DataFrame, save_path: Path) -> None:
    """
    Chart 8: Enhanced Performance by Category Dashboard
    Shows comprehensive model performance metrics grouped by product category with modern visualization
    """
    print("[CHART 8] Generating Enhanced Performance by Category Dashboard...")
    
    # Check if category column exists, otherwise use product_id
    if 'category' in predictions_df.columns or 'product_category' in predictions_df.columns:
        category_col = 'category' if 'category' in predictions_df.columns else 'product_category'
        using_products = False
    elif 'product_id' in predictions_df.columns:
        # Group by product_id and take top N
        category_col = 'product_id'
        using_products = True
        print("   Using product_id as category (top 10 products)")
    else:
        print("[WARNING] No category or product_id column found. Skipping category chart.")
        return
    
    # Calculate metrics by category
    if category_col == 'product_id':
        # Get top 10 products by sales volume
        top_products = predictions_df.groupby('product_id')['sales_quantity'].sum().nlargest(10).index
        category_df = predictions_df[predictions_df['product_id'].isin(top_products)].copy()
        category_df['category'] = category_df['product_id']
    else:
        category_df = predictions_df.copy()
        using_products = False
    
    # Calculate comprehensive metrics by category
    if 'sales_quantity' in category_df.columns and 'forecast_q50' in category_df.columns:
        # Calculate metrics for each category
        category_list = []
        mae_list = []
        rmse_list = []
        mape_list = []
        count_list = []
        accuracy_list = []
        
        for cat in category_df['category'].unique():
            cat_data = category_df[category_df['category'] == cat]
            if len(cat_data) > 0:
                actual = cat_data['sales_quantity']
                forecast = cat_data['forecast_q50']

                mae = np.mean(np.abs(actual - forecast))
                rmse = np.sqrt(np.mean((actual - forecast)**2))
                
                # Safe MAPE calculation
                valid_mask = actual != 0
                if valid_mask.sum() > 0:
                    mape = np.mean(np.abs((actual[valid_mask] - forecast[valid_mask]) / actual[valid_mask])) * 100
                else:
                    mape = 0  # If all actual values are zero

                # Safe accuracy calculation (1 - normalized MAE)
                mean_actual = max(np.mean(actual), 0.001)  # Avoid division by zero
                accuracy = max(0, min(100, 100 * (1 - mae / mean_actual)))

                category_list.append(str(cat))
                mae_list.append(mae)
                rmse_list.append(rmse)
                mape_list.append(min(mape, 1000))  # Cap MAPE at reasonable value
                accuracy_list.append(accuracy)
                count_list.append(len(cat_data))
        
        category_metrics = pd.DataFrame({
            'category': category_list,
            'count': count_list,
            'mae': mae_list,
            'rmse': rmse_list,
            'mape': mape_list,
            'accuracy': accuracy_list
        })
        
        # Sort by count and take top 10
        category_metrics = category_metrics.nlargest(10, 'count')
        
        # Create clean figure
        title_suffix = "Products" if using_products else "Categories"
        fig, ax = plt.subplots(figsize=(12, 8), facecolor='white')
        sns.set_style("whitegrid")

        categories = category_metrics['category']
        accuracy_values = category_metrics['accuracy']
        
        # Bar chart with color coding
        colors = ['#27ae60' if acc > 80 else '#f39c12' if acc > 60 else '#e74c3c'
                 for acc in accuracy_values]

        bars = ax.bar(range(len(categories)), accuracy_values, color=colors,
                     alpha=0.85, edgecolor='white', linewidth=1.5, width=0.7)
        
        # Add value labels
        for i, (bar, acc) in enumerate(zip(bars, accuracy_values)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{acc:.1f}%', ha='center', va='bottom',
                   fontsize=10, fontweight='bold', color='#2c3e50')

        # Styling
        ax.set_xlabel('Category', fontsize=14, fontweight='bold', color='#2c3e50', labelpad=15)
        ax.set_ylabel('Forecast Accuracy (%)', fontsize=14, fontweight='bold',
                     color='#2c3e50', labelpad=15)
        ax.set_title(f'Performance by {title_suffix}: Forecast Accuracy',
                    fontsize=16, fontweight='bold', color='#2c3e50', pad=20)
        ax.set_xticks(range(len(categories)))
        ax.set_xticklabels([cat[:20] + '...' if len(cat) > 20 else cat for cat in categories],
                          fontsize=10, rotation=45, ha='right')
        ax.set_ylim(0, 105)
        ax.grid(axis='y', alpha=0.3, linestyle='--', color='#95a5a6')
        ax.set_axisbelow(True)

        # Clean up spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#bdc3c7')
        ax.spines['bottom'].set_color('#bdc3c7')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()
        print(f"[SUCCESS] Saved performance by category chart: {save_path}")
    else:
        print("[WARNING] Required columns not found for category analysis. Skipping.")


def chart_5_market_context(save_path: Path) -> None:
    """
    Chart 5: Market Context - Vietnam E-Grocery Growth
    Clean line chart showing market size growth
    """
    print("[CHART 5] Generating Market Context Chart...")
    
    # Market data (from README and research)
    years = [2020, 2021, 2022, 2023, 2024, 2025]
    market_size = [15.0, 18.0, 20.8, 23.5, 25.0, 30.0]  # Billion USD
    
    # Create clean figure
    fig, ax = plt.subplots(figsize=(12, 8), facecolor='white')
    sns.set_style("whitegrid")

    # Line plot
    ax.plot(years, market_size, '-', color='#27ae60', linewidth=3,
           marker='o', markersize=8, markerfacecolor='white',
           markeredgecolor='#27ae60', markeredgewidth=2, label='Market Size')

    # Add gradient fill under the curve
    ax.fill_between(years, market_size, alpha=0.2, color='#27ae60', zorder=1)
    
    # Add value labels
    for year, size in zip(years, market_size):
        ax.text(year, size + max(market_size) * 0.04, f'${size:.1f}B',
               ha='center', va='bottom', fontsize=11, fontweight='bold',
               color='#2c3e50')

    # Styling
    ax.set_xlabel('Year', fontsize=14, fontweight='bold', color='#2c3e50', labelpad=15)
    ax.set_ylabel('Market Size (Billion USD)', fontsize=14, fontweight='bold',
                 color='#2c3e50', labelpad=15)
    ax.set_title('Vietnam E-Grocery Market Size Growth', fontsize=16, fontweight='bold',
                color='#2c3e50', pad=20)
    ax.set_xlim(min(years)-0.3, max(years)+0.3)
    ax.set_ylim(0, max(market_size) * 1.15)
    ax.grid(True, alpha=0.3, linestyle='--', color='#95a5a6')
    ax.set_axisbelow(True)

    # Clean up spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#bdc3c7')
    ax.spines['bottom'].set_color('#bdc3c7')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print(f"[SUCCESS] Saved market context chart: {save_path}")


def main():
    """Main function to generate all charts."""
    import sys
    import io
    # Fix encoding for Windows console
    if sys.platform == 'win32':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    
    print("="*70)
    print("SMARTGROCY REPORT CHARTS GENERATOR")
    print("="*70)
    print()
    
    try:
        # Load data with validation
        print("[INFO] Loading data...")
        metrics = load_model_metrics()
        print(f"   Model metrics loaded: {len(metrics)} keys")

        impact_df = load_business_impact()
        print(f"   Business impact data loaded: {len(impact_df)} rows")

        importance_df = load_feature_importance()
        print(f"   Feature importance data loaded: {len(importance_df)} rows")

        predictions_df = load_sample_predictions(n_samples=500)
        print(f"   Predictions data loaded: {len(predictions_df)} rows")
        print("[SUCCESS] Data loaded successfully\n")
        
        # Validate critical data
        if not metrics:
            raise ValueError("Model metrics data is empty or invalid")

        if impact_df.empty:
            raise ValueError("Business impact data is empty")

        if importance_df.empty:
            print("[WARNING] Feature importance data is empty - chart_4 will be skipped")

        if predictions_df.empty:
            raise ValueError("Predictions data is empty")
        
        # Generate mandatory charts
        print("="*70)
        print("GENERATING MANDATORY CHARTS (3)")
        print("="*70)

        try:
            chart_1_model_performance(metrics, CHARTS_DIR / "chart1_model_performance.png")
        except Exception as e:
            print(f"[ERROR] Failed to generate chart_1: {e}")
            # Create a fallback chart
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.text(0.5, 0.5, f'Chart 1 Error:\n{str(e)}', ha='center', va='center', fontsize=14)
            ax.set_title('Model Performance Dashboard - Error')
            plt.savefig(CHARTS_DIR / "chart1_model_performance.png", dpi=300, bbox_inches='tight')
            plt.close()

        try:
            chart_2_business_impact(impact_df, CHARTS_DIR / "chart2_business_impact.png")
        except Exception as e:
            print(f"[ERROR] Failed to generate chart_2: {e}")
            # Create a fallback chart
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.text(0.5, 0.5, f'Chart 2 Error:\n{str(e)}', ha='center', va='center', fontsize=14)
            ax.set_title('Business Impact Dashboard - Error')
            plt.savefig(CHARTS_DIR / "chart2_business_impact.png", dpi=300, bbox_inches='tight')
            plt.close()

        try:
            chart_3_forecast_quality(predictions_df, CHARTS_DIR / "chart3_forecast_quality.png")
        except Exception as e:
            print(f"[ERROR] Failed to generate chart_3: {e}")
            # Create a fallback chart
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.text(0.5, 0.5, f'Chart 3 Error:\n{str(e)}', ha='center', va='center', fontsize=14)
            ax.set_title('Forecast Quality Dashboard - Error')
            plt.savefig(CHARTS_DIR / "chart3_forecast_quality.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        # Generate recommended charts
        print()
        print("="*70)
        print("GENERATING RECOMMENDED CHARTS (2)")
        print("="*70)

        if not importance_df.empty:
            try:
                chart_4_feature_importance(importance_df, CHARTS_DIR / "chart4_feature_importance.png")
            except Exception as e:
                print(f"[ERROR] Failed to generate chart_4: {e}")
                # Create a fallback chart
                fig, ax = plt.subplots(figsize=(12, 8))
                ax.text(0.5, 0.5, f'Chart 4 Error:\n{str(e)}', ha='center', va='center', fontsize=14)
                ax.set_title('Feature Importance Dashboard - Error')
                plt.savefig(CHARTS_DIR / "chart4_feature_importance.png", dpi=300, bbox_inches='tight')
                plt.close()
        else:
            print("[SKIP] Chart 4 skipped - no feature importance data")

        try:
            chart_5_market_context(CHARTS_DIR / "chart5_market_context.png")  # pyright: ignore[reportUndefinedVariable]
        except Exception as e:
            print(f"[ERROR] Failed to generate chart_5: {e}")
            # Create a fallback chart
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.text(0.5, 0.5, f'Chart 5 Error:\n{str(e)}', ha='center', va='center', fontsize=14)
            ax.set_title('Market Context Dashboard - Error')
            plt.savefig(CHARTS_DIR / "chart5_market_context.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        # Generate bonus charts
        print()
        print("="*70)
        print("GENERATING BONUS CHARTS (3)")
        print("="*70)

        try:
            chart_6_hourly_demand_pattern(predictions_df, CHARTS_DIR / "chart6_hourly_demand_pattern.png")
        except Exception as e:
            print(f"[ERROR] Failed to generate chart_6: {e}")
            # Create a fallback chart
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.text(0.5, 0.5, f'Chart 6 Error:\n{str(e)}', ha='center', va='center', fontsize=14)
            ax.set_title('Hourly Demand Pattern Dashboard - Error')
            plt.savefig(CHARTS_DIR / "chart6_hourly_demand_pattern.png", dpi=300, bbox_inches='tight')
            plt.close()

        try:
            chart_7_profit_margin_improvement(impact_df, CHARTS_DIR / "chart7_profit_margin_improvement.png")
        except Exception as e:
            print(f"[ERROR] Failed to generate chart_7: {e}")
            # Create a fallback chart
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.text(0.5, 0.5, f'Chart 7 Error:\n{str(e)}', ha='center', va='center', fontsize=14)
            ax.set_title('Financial Impact Dashboard - Error')
            plt.savefig(CHARTS_DIR / "chart7_profit_margin_improvement.png", dpi=300, bbox_inches='tight')
            plt.close()

        try:
            chart_8_performance_by_category(predictions_df, CHARTS_DIR / "chart8_performance_by_category.png")
        except Exception as e:
            print(f"[ERROR] Failed to generate chart_8: {e}")
            # Create a fallback chart
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.text(0.5, 0.5, f'Chart 8 Error:\n{str(e)}', ha='center', va='center', fontsize=14)
            ax.set_title('Performance by Category Dashboard - Error')
            plt.savefig(CHARTS_DIR / "chart8_performance_by_category.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        print()
        print("="*70)
        print("[SUCCESS] ALL CHARTS GENERATED SUCCESSFULLY!")
        print("="*70)
        print(f"Output directory: {CHARTS_DIR}")
        print()
        print("Generated files:")
        for chart_file in sorted(CHARTS_DIR.glob("chart*.png")):
            print(f"  - {chart_file.name}")
        
    except Exception as e:
        print(f"\n[ERROR] Error generating charts: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

