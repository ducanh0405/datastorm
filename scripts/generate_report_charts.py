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

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns

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
    predictions_path = PROJECT_ROOT / "reports" / "predictions_test_set.csv"
    if not predictions_path.exists():
        raise FileNotFoundError(f"Predictions file not found: {predictions_path}")
    
    # For large files, read only first n_samples rows to avoid memory issues
    # In production, you might want to use random sampling, but for visualization
    # reading consecutive rows is sufficient and faster
    print(f"   Reading {n_samples} rows from predictions file...")
    df = pd.read_csv(predictions_path, nrows=n_samples)
    return df


def chart_1_model_performance(metrics: Dict[str, Any], save_path: Path) -> None:
    """
    Chart 1: Model Performance Metrics
    Shows MAE, RMSE, Pinball Loss, R², Coverage across quantiles
    """
    print("[CHART 1] Generating Model Performance Metrics...")
    
    # Extract metrics by quantile
    quantiles = ['q05', 'q25', 'q50', 'q75', 'q95']
    quantile_labels = ['Q05', 'Q25', 'Q50', 'Q75', 'Q95']
    
    mae_values = [metrics[f'{q}_mae'] for q in quantiles]
    rmse_values = [metrics[f'{q}_rmse'] for q in quantiles]
    pinball_values = [metrics[f'{q}_pinball_loss'] for q in quantiles]
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Model Performance Metrics Across Quantiles', fontsize=16, fontweight='bold', y=0.995)
    
    # 1. MAE
    ax1 = axes[0, 0]
    bars1 = ax1.bar(quantile_labels, mae_values, color=COLORS['primary'], alpha=0.8, edgecolor='white', linewidth=2)
    ax1.set_title('Mean Absolute Error (MAE)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('MAE', fontsize=10)
    ax1.set_xlabel('Quantile', fontsize=10)
    ax1.grid(axis='y', alpha=0.3)
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    # 2. RMSE
    ax2 = axes[0, 1]
    bars2 = ax2.bar(quantile_labels, rmse_values, color=COLORS['secondary'], alpha=0.8, edgecolor='white', linewidth=2)
    ax2.set_title('Root Mean Squared Error (RMSE)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('RMSE', fontsize=10)
    ax2.set_xlabel('Quantile', fontsize=10)
    ax2.grid(axis='y', alpha=0.3)
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    # 3. Pinball Loss
    ax3 = axes[1, 0]
    bars3 = ax3.bar(quantile_labels, pinball_values, color=COLORS['success'], alpha=0.8, edgecolor='white', linewidth=2)
    ax3.set_title('Pinball Loss', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Pinball Loss', fontsize=10)
    ax3.set_xlabel('Quantile', fontsize=10)
    ax3.grid(axis='y', alpha=0.3)
    for bar in bars3:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    # 4. Overall Metrics (R² and Coverage)
    ax4 = axes[1, 1]
    overall_metrics = ['R² Score', 'Coverage (90%)']
    overall_values = [metrics['r2_score'], metrics['coverage_90%']]
    colors_overall = [COLORS['primary'], COLORS['warning']]
    bars4 = ax4.bar(overall_metrics, overall_values, color=colors_overall, alpha=0.8, edgecolor='white', linewidth=2)
    ax4.set_title('Overall Model Quality', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Score', fontsize=10)
    ax4.set_ylim(0, 1)
    ax4.grid(axis='y', alpha=0.3)
    for bar in bars4:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[SUCCESS] Saved: {save_path}")


def chart_2_business_impact(impact_df: pd.DataFrame, save_path: Path) -> None:
    """
    Chart 2: Business Impact KPI Comparison
    Shows Spoilage and Stockout reduction (Baseline vs ML Model)
    """
    print("[CHART 2] Generating Business Impact KPI Comparison...")
    
    # Extract data
    spoilage_baseline = impact_df.loc[0, 'baseline']
    spoilage_ml = impact_df.loc[0, 'ml_model']
    stockout_baseline = impact_df.loc[1, 'baseline']
    stockout_ml = impact_df.loc[1, 'ml_model']
    
    spoilage_improvement = impact_df.loc[0, 'improvement_pct']
    stockout_improvement = impact_df.loc[1, 'improvement_pct']
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Business Impact: KPI Comparison (Baseline vs ML Model)', 
                 fontsize=16, fontweight='bold', y=1.02)
    
    # Chart 1: Spoilage Rate
    categories = ['Baseline', 'ML Model']
    spoilage_values = [spoilage_baseline, spoilage_ml]
    colors_spoilage = [COLORS['danger'], COLORS['success']]
    
    bars1 = ax1.bar(categories, spoilage_values, color=colors_spoilage, alpha=0.8, 
                    edgecolor='white', linewidth=2, width=0.6)
    ax1.set_title(f'Spoilage Rate Reduction\n{spoilage_improvement:.1f}% Improvement', 
                  fontsize=12, fontweight='bold')
    ax1.set_ylabel('Spoilage Rate (%)', fontsize=10)
    ax1.set_ylim(0, max(spoilage_values) * 1.2)
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Add improvement arrow
    ax1.annotate('', xy=(1, spoilage_ml), xytext=(0, spoilage_baseline),
                arrowprops=dict(arrowstyle='->', lw=2, color=COLORS['success']))
    
    # Chart 2: Stockout Rate
    stockout_values = [stockout_baseline, stockout_ml]
    
    bars2 = ax2.bar(categories, stockout_values, color=colors_spoilage, alpha=0.8,
                    edgecolor='white', linewidth=2, width=0.6)
    ax2.set_title(f'Stockout Rate Reduction\n{stockout_improvement:.1f}% Improvement',
                  fontsize=12, fontweight='bold')
    ax2.set_ylabel('Stockout Rate (%)', fontsize=10)
    ax2.set_ylim(0, max(stockout_values) * 1.2)
    ax2.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Add improvement arrow
    ax2.annotate('', xy=(1, stockout_ml), xytext=(0, stockout_baseline),
                arrowprops=dict(arrowstyle='->', lw=2, color=COLORS['success']))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[SUCCESS] Saved: {save_path}")


def chart_3_forecast_quality(predictions_df: pd.DataFrame, save_path: Path, n_samples: int = 200) -> None:
    """
    Chart 3: Forecast Quality (Forecast vs Actual)
    Shows prediction intervals (Q05-Q95) with actual values overlay
    """
    print("[CHART 3] Generating Forecast Quality (Forecast vs Actual)...")
    
    # Sample data for visualization
    if len(predictions_df) > n_samples:
        sample_df = predictions_df.sample(n=min(n_samples, len(predictions_df)), random_state=42)
    else:
        sample_df = predictions_df.copy()
    
    # Sort by index for better visualization
    sample_df = sample_df.sort_index().reset_index(drop=True)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Check which columns are available and map column names
    # Try different possible column names
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
        ax.fill_between(x, sample_df[q05_col], sample_df[q95_col], 
                        alpha=0.3, color=COLORS['primary'], label='90% Prediction Interval (Q05-Q95)')
    
    # Plot median forecast
    if q50_col:
        ax.plot(x, sample_df[q50_col], '--', color=COLORS['success'], 
               linewidth=2, label='Forecast Median (Q50)', alpha=0.8)
    
    # Plot actual values if available
    if has_actual and actual_col:
        ax.plot(x, sample_df[actual_col], 'o-', color=COLORS['danger'], 
               linewidth=2, markersize=4, label='Actual Values', alpha=0.7)
    
    ax.set_title('Forecast Quality: Prediction Intervals vs Actual Values', 
                fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Sample Index', fontsize=11)
    ax.set_ylabel('Sales Quantity', fontsize=11)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Add statistics text box
    if has_actual and actual_col and q50_col:
        mae = np.mean(np.abs(sample_df[actual_col] - sample_df[q50_col]))
        rmse = np.sqrt(np.mean((sample_df[actual_col] - sample_df[q50_col])**2))
        stats_text = f'MAE: {mae:.2f}\nRMSE: {rmse:.2f}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
               fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[SUCCESS] Saved: {save_path}")


def chart_4_feature_importance(importance_df: pd.DataFrame, save_path: Path, top_n: int = 10) -> None:
    """
    Chart 4: Feature Importance (SHAP Top 10)
    Shows top N most important features
    """
    print("[CHART 4] Generating Feature Importance (SHAP)...")
    
    # Get top N features
    top_features = importance_df.nlargest(top_n, 'mean_abs_shap')
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Horizontal bar chart
    y_pos = np.arange(len(top_features))
    bars = ax.barh(y_pos, top_features['mean_abs_shap'], 
                   color=COLORS['primary'], alpha=0.8, edgecolor='white', linewidth=1.5)
    
    # Customize
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_features['feature'], fontsize=10)
    ax.set_xlabel('Mean Absolute SHAP Value', fontsize=11, fontweight='bold')
    ax.set_title(f'Top {top_n} Feature Importance (SHAP Values)', 
                fontsize=14, fontweight='bold', pad=20)
    ax.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, (idx, row) in enumerate(top_features.iterrows()):
        ax.text(row['mean_abs_shap'], i, f' {row["mean_abs_shap"]:.3f}',
               va='center', fontsize=9, fontweight='bold')
    
    # Invert y-axis to show highest at top
    ax.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[SUCCESS] Saved: {save_path}")


def chart_6_hourly_demand_pattern(predictions_df: pd.DataFrame, save_path: Path) -> None:
    """
    Chart 6 (Bonus): Hourly Demand Pattern
    Shows demand distribution across hours of the day
    """
    print("[CHART 6] Generating Hourly Demand Pattern...")
    
    # Check if hour_of_day column exists
    if 'hour_of_day' not in predictions_df.columns:
        print("[WARNING] hour_of_day column not found. Skipping hourly pattern chart.")
        return
    
    # Group by hour and calculate statistics
    hourly_stats = predictions_df.groupby('hour_of_day').agg({
        'sales_quantity': ['mean', 'std', 'count'],
        'forecast_q50': 'mean'
    }).reset_index()
    
    hourly_stats.columns = ['hour', 'mean_demand', 'std_demand', 'count', 'mean_forecast']
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle('Hourly Demand Pattern Analysis', fontsize=16, fontweight='bold', y=0.995)
    
    # Chart 1: Mean demand by hour with error bars
    hours = hourly_stats['hour']
    mean_demand = hourly_stats['mean_demand']
    std_demand = hourly_stats['std_demand']
    
    ax1.bar(hours, mean_demand, color=COLORS['primary'], alpha=0.7, 
            edgecolor='white', linewidth=1.5, yerr=std_demand, 
            capsize=5, error_kw={'elinewidth': 2, 'alpha': 0.5})
    ax1.set_title('Average Sales Quantity by Hour of Day', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Hour of Day (0-23)', fontsize=10)
    ax1.set_ylabel('Average Sales Quantity', fontsize=10)
    ax1.set_xticks(range(0, 24, 2))
    ax1.grid(axis='y', alpha=0.3)
    
    # Highlight peak hours
    peak_hours = [7, 8, 9, 17, 18, 19, 20]  # Morning and evening peaks
    for h in peak_hours:
        if h in hours.values:
            idx = hours[hours == h].index[0]
            ax1.bar(h, mean_demand.iloc[idx], color=COLORS['warning'], alpha=0.8)
    
    # Chart 2: Demand distribution (box plot style)
    hourly_data = []
    hourly_labels = []
    for hour in sorted(predictions_df['hour_of_day'].unique()):
        hour_data = predictions_df[predictions_df['hour_of_day'] == hour]['sales_quantity']
        if len(hour_data) > 0:
            hourly_data.append(hour_data.values)
            hourly_labels.append(f'{hour:02d}:00')
    
    if hourly_data:
        bp = ax2.boxplot(hourly_data, tick_labels=hourly_labels, patch_artist=True,
                        showmeans=True, meanline=True)
        for patch in bp['boxes']:
            patch.set_facecolor(COLORS['primary'])
            patch.set_alpha(0.7)
        
        ax2.set_title('Sales Quantity Distribution by Hour', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Hour of Day', fontsize=10)
        ax2.set_ylabel('Sales Quantity', fontsize=10)
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[SUCCESS] Saved: {save_path}")


def chart_7_profit_margin_improvement(impact_df: pd.DataFrame, save_path: Path) -> None:
    """
    Chart 7 (Bonus): Profit Margin Improvement
    Shows profit margin improvement from baseline to ML model
    """
    print("[CHART 7] Generating Profit Margin Improvement...")
    
    # Extract profit margin data
    profit_baseline = impact_df.loc[2, 'baseline']  # Row 2 is profit_margin_pct
    profit_ml = impact_df.loc[2, 'ml_model']
    profit_improvement = impact_df.loc[2, 'improvement_pct']
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Profit Margin Improvement: Baseline vs ML Model', 
                 fontsize=16, fontweight='bold', y=1.02)
    
    # Chart 1: Profit margin comparison
    categories = ['Baseline', 'ML Model']
    profit_values = [profit_baseline, profit_ml]
    colors_profit = [COLORS['danger'], COLORS['success']]
    
    bars1 = ax1.bar(categories, profit_values, color=colors_profit, alpha=0.8,
                    edgecolor='white', linewidth=2, width=0.6)
    ax1.set_title(f'Profit Margin Comparison\n{profit_improvement:.1f}% Improvement',
                  fontsize=12, fontweight='bold')
    ax1.set_ylabel('Profit Margin (%)', fontsize=10)
    ax1.set_ylim(0, max(profit_values) * 1.3)
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Add improvement arrow
    ax1.annotate('', xy=(1, profit_ml), xytext=(0, profit_baseline),
                arrowprops=dict(arrowstyle='->', lw=2, color=COLORS['success']))
    
    # Chart 2: All KPIs improvement comparison
    metrics = ['Spoilage\nRate', 'Stockout\nRate', 'Profit\nMargin']
    baseline_values = [impact_df.loc[0, 'baseline'], 
                       impact_df.loc[1, 'baseline'],
                       impact_df.loc[2, 'baseline']]
    ml_values = [impact_df.loc[0, 'ml_model'],
                 impact_df.loc[1, 'ml_model'],
                 impact_df.loc[2, 'ml_model']]
    improvements = [impact_df.loc[0, 'improvement_pct'],
                    impact_df.loc[1, 'improvement_pct'],
                    impact_df.loc[2, 'improvement_pct']]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    bars2 = ax2.bar(x - width/2, baseline_values, width, label='Baseline',
                   color=COLORS['danger'], alpha=0.7, edgecolor='white', linewidth=1.5)
    bars3 = ax2.bar(x + width/2, ml_values, width, label='ML Model',
                   color=COLORS['success'], alpha=0.7, edgecolor='white', linewidth=1.5)
    
    ax2.set_title('All KPIs: Baseline vs ML Model', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Value (%)', fontsize=10)
    ax2.set_xticks(x)
    ax2.set_xticklabels(metrics, fontsize=9)
    ax2.legend(fontsize=9)
    ax2.grid(axis='y', alpha=0.3)
    
    # Add improvement percentages
    for i, (baseline, ml, improvement) in enumerate(zip(baseline_values, ml_values, improvements)):
        ax2.text(i, max(baseline, ml) + 1, f'+{improvement:.1f}%',
                ha='center', va='bottom', fontsize=9, fontweight='bold',
                color=COLORS['success'])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[SUCCESS] Saved: {save_path}")


def chart_8_performance_by_category(predictions_df: pd.DataFrame, save_path: Path) -> None:
    """
    Chart 8 (Bonus): Performance by Category
    Shows model performance metrics grouped by product category (or product_id if category not available)
    """
    print("[CHART 8] Generating Performance by Category...")
    
    # Check if category column exists, otherwise use product_id
    if 'category' in predictions_df.columns or 'product_category' in predictions_df.columns:
        category_col = 'category' if 'category' in predictions_df.columns else 'product_category'
    elif 'product_id' in predictions_df.columns:
        # Group by product_id and take top N
        category_col = 'product_id'
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
    
    # Calculate MAE and RMSE by category
    if 'sales_quantity' in category_df.columns and 'forecast_q50' in category_df.columns:
        # Calculate metrics for each category
        category_list = []
        mae_list = []
        rmse_list = []
        count_list = []
        
        for cat in category_df['category'].unique():
            cat_data = category_df[category_df['category'] == cat]
            if len(cat_data) > 0:
                actual = cat_data['sales_quantity']
                forecast = cat_data['forecast_q50']
                mae = np.mean(np.abs(actual - forecast))
                rmse = np.sqrt(np.mean((actual - forecast)**2))
                
                category_list.append(cat)
                mae_list.append(mae)
                rmse_list.append(rmse)
                count_list.append(len(cat_data))
        
        category_metrics = pd.DataFrame({
            'category': category_list,
            'count': count_list,
            'mae': mae_list,
            'rmse': rmse_list
        })
        
        # Sort by count and take top 10
        category_metrics = category_metrics.nlargest(10, 'count')
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('Model Performance by Category (Top 10)', 
                     fontsize=16, fontweight='bold', y=1.02)
        
        # Chart 1: MAE by category
        categories = category_metrics['category'].astype(str)
        mae_values = category_metrics['mae']
        
        bars1 = ax1.barh(range(len(categories)), mae_values, 
                        color=COLORS['primary'], alpha=0.8, edgecolor='white', linewidth=1.5)
        ax1.set_yticks(range(len(categories)))
        ax1.set_yticklabels(categories, fontsize=9)
        ax1.set_xlabel('Mean Absolute Error (MAE)', fontsize=10, fontweight='bold')
        ax1.set_title('MAE by Category', fontsize=12, fontweight='bold')
        ax1.invert_yaxis()
        ax1.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for i, (bar, mae) in enumerate(zip(bars1, mae_values)):
            ax1.text(mae, i, f' {mae:.2f}', va='center', fontsize=9, fontweight='bold')
        
        # Chart 2: RMSE by category
        rmse_values = category_metrics['rmse']
        
        bars2 = ax2.barh(range(len(categories)), rmse_values,
                        color=COLORS['secondary'], alpha=0.8, edgecolor='white', linewidth=1.5)
        ax2.set_yticks(range(len(categories)))
        ax2.set_yticklabels(categories, fontsize=9)
        ax2.set_xlabel('Root Mean Squared Error (RMSE)', fontsize=10, fontweight='bold')
        ax2.set_title('RMSE by Category', fontsize=12, fontweight='bold')
        ax2.invert_yaxis()
        ax2.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for i, (bar, rmse) in enumerate(zip(bars2, rmse_values)):
            ax2.text(rmse, i, f' {rmse:.2f}', va='center', fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"[SUCCESS] Saved: {save_path}")
    else:
        print("[WARNING] Required columns not found for category analysis. Skipping.")


def chart_5_market_context(save_path: Path) -> None:
    """
    Chart 5: Market Context (Vietnam E-Grocery Growth)
    Shows market growth trends to justify problem importance
    """
    print("[CHART 5] Generating Market Context (Vietnam E-Grocery Growth)...")
    
    # Market data (from README and research)
    years = [2020, 2021, 2022, 2023, 2024, 2025]
    market_size = [15.0, 18.0, 20.8, 23.5, 25.0, 30.0]  # Billion USD
    growth_rate = [None, 20.0, 15.6, 13.0, 6.4, 20.0]  # YoY %
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Vietnam E-Grocery Market Growth Context', 
                 fontsize=16, fontweight='bold', y=1.02)
    
    # Chart 1: Market Size Trend
    ax1.plot(years, market_size, 'o-', color=COLORS['primary'], 
            linewidth=3, markersize=10, markerfacecolor='white', 
            markeredgewidth=2, markeredgecolor=COLORS['primary'])
    ax1.fill_between(years, market_size, alpha=0.2, color=COLORS['primary'])
    ax1.set_title('Market Size Growth', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Year', fontsize=10)
    ax1.set_ylabel('Market Size (Billion USD)', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, max(market_size) * 1.2)
    
    # Add value labels
    for year, size in zip(years, market_size):
        ax1.text(year, size, f'${size:.1f}B', ha='center', va='bottom', 
                fontsize=9, fontweight='bold')
    
    # Chart 2: Growth Rate
    growth_years = years[1:]
    growth_values = [g for g in growth_rate if g is not None]
    colors_growth = [COLORS['success'] if g >= 15 else COLORS['warning'] for g in growth_values]
    
    bars = ax2.bar(growth_years, growth_values, color=colors_growth, 
                   alpha=0.8, edgecolor='white', linewidth=2)
    ax2.set_title('Year-over-Year Growth Rate', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Year', fontsize=10)
    ax2.set_ylabel('Growth Rate (%)', fontsize=10)
    ax2.grid(axis='y', alpha=0.3)
    ax2.axhline(y=15, color=COLORS['success'], linestyle='--', 
               linewidth=2, alpha=0.5, label='Strong Growth Threshold')
    ax2.legend(fontsize=9)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Add annotation
    fig.text(0.5, 0.02, 'Source: Vietnam E-Commerce Report 2024, Ministry of Industry and Trade', 
            ha='center', fontsize=9, style='italic', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[SUCCESS] Saved: {save_path}")


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
        # Load data
        print("[INFO] Loading data...")
        metrics = load_model_metrics()
        impact_df = load_business_impact()
        importance_df = load_feature_importance()
        predictions_df = load_sample_predictions(n_samples=500)
        print("[SUCCESS] Data loaded successfully\n")
        
        # Generate mandatory charts
        print("="*70)
        print("GENERATING MANDATORY CHARTS (3)")
        print("="*70)
        chart_1_model_performance(metrics, CHARTS_DIR / "chart1_model_performance.png")
        chart_2_business_impact(impact_df, CHARTS_DIR / "chart2_business_impact.png")
        chart_3_forecast_quality(predictions_df, CHARTS_DIR / "chart3_forecast_quality.png")
        
        # Generate recommended charts
        print()
        print("="*70)
        print("GENERATING RECOMMENDED CHARTS (2)")
        print("="*70)
        chart_4_feature_importance(importance_df, CHARTS_DIR / "chart4_feature_importance.png")
        chart_5_market_context(CHARTS_DIR / "chart5_market_context.png")
        
        # Generate bonus charts
        print()
        print("="*70)
        print("GENERATING BONUS CHARTS (3)")
        print("="*70)
        chart_6_hourly_demand_pattern(predictions_df, CHARTS_DIR / "chart6_hourly_demand_pattern.png")
        chart_7_profit_margin_improvement(impact_df, CHARTS_DIR / "chart7_profit_margin_improvement.png")
        chart_8_performance_by_category(predictions_df, CHARTS_DIR / "chart8_performance_by_category.png")
        
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

