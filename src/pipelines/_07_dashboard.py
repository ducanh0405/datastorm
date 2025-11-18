"""
Forecast Dashboard Generator
============================
Tạo dashboard HTML tối ưu UX/UI để hiển thị kết quả prediction và SHAP values.
"""
import logging
import sys
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

# Setup project path FIRST before any other imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Now import config
try:
    from src.config import (
        OUTPUT_FILES,
        SHAP_CONFIG,
        TRAINING_CONFIG,
        ensure_directories,
        get_dataset_config,
        setup_logging,
        setup_project_path,
    )
    setup_project_path()
    setup_logging()
    ensure_directories()
    logger = logging.getLogger(__name__)
except ImportError as e:
    print("Error: Cannot import config. Please ensure src/config.py exists.")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Import error: {e}")
    print(f"Python path: {sys.path[:3]}")
    sys.exit(1)

# Import other dependencies after config is loaded
import json
from typing import Any

import pandas as pd


def load_predictions() -> pd.DataFrame:
    """Load predictions từ file."""
    # Always load from gzip CSV file
    predictions_path = OUTPUT_FILES['predictions_test'].with_suffix('.csv.gz')

    if not predictions_path.exists():
        raise FileNotFoundError(f"Predictions file not found: {predictions_path}")

    df = pd.read_csv(predictions_path, compression='gzip')
    logger.info(f"Loaded predictions: {df.shape}")
    return df


def load_metrics() -> dict[str, Any]:
    """Load metrics từ file."""
    metrics_path = OUTPUT_FILES['model_metrics']
    if not metrics_path.exists():
        logger.warning(f"Metrics file not found: {metrics_path}")
        return {}

    with open(metrics_path) as f:
        metrics = json.load(f)

    # Check if nested or flat structure
    if metrics and isinstance(metrics, dict) and len(metrics) > 0:
        first_value = next(iter(metrics.values()))
        if isinstance(first_value, dict):
            logger.info(f"Loaded metrics for {len(metrics)} model types")
        else:
            logger.info(f"Loaded {len(metrics)} metrics (flat structure)")
    else:
        logger.info("Loaded empty metrics")

    return metrics


def load_shap_values() -> dict[str, Any] | None:
    """Load SHAP values từ file."""
    shap_dir = OUTPUT_FILES['shap_values_dir']
    shap_df_path = shap_dir / 'shap_values.csv'
    shap_summary_path = shap_dir / 'shap_summary.json'

    if not shap_df_path.exists():
        logger.warning(f"SHAP values file not found: {shap_df_path}")
        return None

    try:
        shap_df = pd.read_csv(shap_df_path, index_col=0)
        shap_summary = {}
        if shap_summary_path.exists():
            with open(shap_summary_path) as f:
                shap_summary = json.load(f)

        return {
            'shap_values': shap_df,
            'summary': shap_summary
        }
    except Exception as e:
        logger.warning(f"Error loading SHAP values: {e}")
        return None


def create_dashboard_html(
    predictions: pd.DataFrame,
    metrics: dict[str, Any],
    shap_values: dict[str, Any] | None = None
) -> str:
    """
    Tạo dashboard HTML với visualizations.

    Args:
        predictions: DataFrame với predictions
        metrics: Dict với metrics
        shap_values: Optional SHAP values dict

    Returns:
        HTML string
    """
    config = get_dataset_config()
    target_col = config['target_column']
    quantiles = TRAINING_CONFIG['quantiles']

    # Tính toán các statistics
    pred_cols = [f'forecast_q{int(q*100):02d}' for q in quantiles]
    available_pred_cols = [col for col in pred_cols if col in predictions.columns]

    # Tạo data cho charts
    predictions_json = predictions[available_pred_cols + [target_col]].to_dict('records') if target_col in predictions.columns else predictions[available_pred_cols].to_dict('records')

    # Top features từ SHAP values
    top_features = []
    if shap_values and 'shap_values' in shap_values:
        shap_df = shap_values['shap_values']
        feature_importance = shap_df.abs().mean().sort_values(ascending=False).head(20)
        top_features = [
            {'feature': feat, 'importance': float(imp)}
            for feat, imp in feature_importance.items()
        ]

    # Metrics summary
    # Handle both nested (by model_type) and flat metrics structure
    metrics_summary = []
    num_models = 0

    # Check if metrics is nested (has model_type keys) or flat
    if metrics and isinstance(metrics, dict) and len(metrics) > 0:
        # Check if first value is a dict (nested structure)
        first_key = next(iter(metrics.keys()))
        first_value = metrics[first_key]

        if isinstance(first_value, dict):
            # Nested structure: {model_type: {metric: value}}
            num_models = len(metrics)
            # Chỉ hiển thị metrics cho 5 quantiles: Q05, Q25, Q50, Q75, Q95
            quantile_keys = ['q05', 'q25', 'q50', 'q75', 'q95']
            for model_type, model_metrics in metrics.items():
                if isinstance(model_metrics, dict):
                    for metric_name, metric_value in model_metrics.items():
                        if isinstance(metric_value, int | float):
                            # Chỉ hiển thị metrics cho 5 quantiles được train
                            if any(q in metric_name.lower() for q in quantile_keys):
                                if 'pinball' in metric_name.lower() or 'mae' in metric_name.lower() or 'rmse' in metric_name.lower():
                                    metrics_summary.append({
                                        'model': model_type,
                                        'metric': metric_name,
                                        'value': float(metric_value)
                                    })
        else:
            # Flat structure: {metric: value} - assume single model (lightgbm or default)
            num_models = 1
            model_type = 'lightgbm'  # Default model type
            # Chỉ hiển thị metrics cho 5 quantiles: Q05, Q25, Q50, Q75, Q95
            quantile_keys = ['q05', 'q25', 'q50', 'q75', 'q95']
            for metric_name, metric_value in metrics.items():
                if isinstance(metric_value, int | float):
                    # Chỉ hiển thị metrics cho 5 quantiles được train
                    if any(q in metric_name.lower() for q in quantile_keys):
                        if 'pinball' in metric_name.lower() or 'mae' in metric_name.lower() or 'rmse' in metric_name.lower():
                            metrics_summary.append({
                                'model': model_type,
                                'metric': metric_name,
                                'value': float(metric_value)
                            })

    # Tính toán thêm statistics cho dashboard
    total_predictions = len(predictions)
    num_quantiles = len(available_pred_cols)

    # HTML template với design cải tiến
    html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Forecast Dashboard - E-Grocery Forecaster</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');

        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: 'Poppins', 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
            background-size: 400% 400%;
            animation: gradientShift 15s ease infinite;
            color: #333;
            padding: 20px;
            min-height: 100vh;
        }}

        @keyframes gradientShift {{
            0% {{ background-position: 0% 50%; }}
            50% {{ background-position: 100% 50%; }}
            100% {{ background-position: 0% 50%; }}
        }}

        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: rgba(255, 255, 255, 0.98);
            border-radius: 25px;
            box-shadow: 0 25px 80px rgba(0,0,0,0.3);
            padding: 40px;
            animation: fadeInUp 0.8s ease-out;
            backdrop-filter: blur(10px);
        }}

        @keyframes fadeInUp {{
            from {{
                opacity: 0;
                transform: translateY(30px);
            }}
            to {{
                opacity: 1;
                transform: translateY(0);
            }}
        }}

        .header {{
            text-align: center;
            margin-bottom: 50px;
            padding-bottom: 25px;
            border-bottom: 4px solid transparent;
            border-image: linear-gradient(90deg, #667eea, #764ba2, #f093fb) 1;
            position: relative;
        }}

        .header::after {{
            content: '';
            position: absolute;
            bottom: -4px;
            left: 50%;
            transform: translateX(-50%);
            width: 100px;
            height: 4px;
            background: linear-gradient(90deg, #667eea, #764ba2);
            border-radius: 2px;
        }}

        .header h1 {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            font-size: 3em;
            margin-bottom: 15px;
            font-weight: 700;
            animation: pulse 2s ease-in-out infinite;
        }}

        @keyframes pulse {{
            0%, 100% {{ transform: scale(1); }}
            50% {{ transform: scale(1.02); }}
        }}

        .header p {{
            color: #666;
            font-size: 1.2em;
            font-weight: 300;
        }}

        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 25px;
            margin-bottom: 50px;
        }}

        .stat-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 20px;
            text-align: center;
            box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
            transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
            position: relative;
            overflow: hidden;
            cursor: pointer;
        }}

        .stat-card::before {{
            content: '';
            position: absolute;
            top: -50%;
            left: -50%;
            width: 200%;
            height: 200%;
            background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
            opacity: 0;
            transition: opacity 0.4s;
        }}

        .stat-card:hover {{
            transform: translateY(-10px) scale(1.05);
            box-shadow: 0 20px 40px rgba(102, 126, 234, 0.4);
        }}

        .stat-card:hover::before {{
            opacity: 1;
        }}

        .stat-card:nth-child(1) {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }}

        .stat-card:nth-child(2) {{
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        }}

        .stat-card:nth-child(3) {{
            background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        }}

        .stat-card:nth-child(4) {{
            background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
        }}

        .stat-card .icon {{
            font-size: 2.5em;
            margin-bottom: 15px;
            opacity: 0.9;
        }}

        .stat-card h3 {{
            font-size: 1em;
            margin-bottom: 15px;
            opacity: 0.95;
            font-weight: 400;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}

        .stat-card .value {{
            font-size: 2.5em;
            font-weight: 700;
            line-height: 1;
        }}

        .chart-container {{
            margin-bottom: 50px;
            background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%);
            padding: 30px;
            border-radius: 20px;
            box-shadow: 0 5px 20px rgba(0,0,0,0.08);
            transition: all 0.3s ease;
            border: 1px solid rgba(102, 126, 234, 0.1);
        }}

        .chart-container:hover {{
            box-shadow: 0 10px 30px rgba(102, 126, 234, 0.15);
            transform: translateY(-2px);
        }}

        .chart-container h2 {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 25px;
            font-size: 1.8em;
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 15px;
        }}

        .chart-container h2::before {{
            content: '📈';
            font-size: 1.2em;
        }}

        .chart {{
            width: 100%;
            height: 550px;
            border-radius: 10px;
        }}

        .metrics-table {{
            width: 100%;
            border-collapse: separate;
            border-spacing: 0;
            margin-top: 20px;
            border-radius: 15px;
            overflow: hidden;
            box-shadow: 0 5px 15px rgba(0,0,0,0.08);
        }}

        .metrics-table th,
        .metrics-table td {{
            padding: 18px;
            text-align: left;
            border-bottom: 1px solid rgba(0,0,0,0.05);
        }}

        .metrics-table th {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            font-weight: 600;
            text-transform: uppercase;
            font-size: 0.9em;
            letter-spacing: 0.5px;
        }}

        .metrics-table tr {{
            transition: all 0.3s ease;
        }}

        .metrics-table tr:hover {{
            background: linear-gradient(90deg, rgba(102, 126, 234, 0.05) 0%, rgba(118, 75, 162, 0.05) 100%);
            transform: scale(1.01);
        }}

        .metrics-table td:last-child {{
            font-weight: 600;
            color: #667eea;
        }}

        .feature-list {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
            gap: 20px;
            margin-top: 30px;
        }}

        .feature-item {{
            background: white;
            padding: 20px;
            border-radius: 15px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.08);
            border-left: 5px solid #667eea;
            transition: all 0.3s ease;
            position: relative;
            overflow: hidden;
        }}

        .feature-item::before {{
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(102, 126, 234, 0.1), transparent);
            transition: left 0.5s;
        }}

        .feature-item:hover {{
            transform: translateX(5px);
            box-shadow: 0 8px 25px rgba(102, 126, 234, 0.2);
        }}

        .feature-item:hover::before {{
            left: 100%;
        }}

        .feature-item .feature-name {{
            font-weight: 600;
            color: #333;
            margin-bottom: 10px;
            font-size: 0.95em;
        }}

        .feature-item .feature-importance {{
            color: #667eea;
            font-size: 1.5em;
            font-weight: 700;
        }}

        .tabs {{
            display: flex;
            gap: 15px;
            margin-bottom: 30px;
            border-bottom: 3px solid rgba(102, 126, 234, 0.1);
            padding-bottom: 0;
        }}

        .tab {{
            padding: 15px 30px;
            cursor: pointer;
            background: transparent;
            border: none;
            border-radius: 10px 10px 0 0;
            font-size: 1.1em;
            font-weight: 500;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            color: #666;
            position: relative;
        }}

        .tab::after {{
            content: '';
            position: absolute;
            bottom: -3px;
            left: 0;
            width: 0;
            height: 3px;
            background: linear-gradient(90deg, #667eea, #764ba2);
            transition: width 0.3s ease;
        }}

        .tab:hover {{
            color: #667eea;
            background: rgba(102, 126, 234, 0.05);
        }}

        .tab.active {{
            color: #667eea;
            background: rgba(102, 126, 234, 0.1);
        }}

        .tab.active::after {{
            width: 100%;
        }}

        .tab-content {{
            display: none;
            animation: fadeIn 0.5s ease-in;
        }}

        .tab-content.active {{
            display: block;
        }}

        @keyframes fadeIn {{
            from {{
                opacity: 0;
                transform: translateY(10px);
            }}
            to {{
                opacity: 1;
                transform: translateY(0);
            }}
        }}

        @media (max-width: 768px) {{
            .container {{
                padding: 20px;
            }}

            .header h1 {{
                font-size: 2em;
            }}

            .stats-grid {{
                grid-template-columns: 1fr;
            }}

            .chart {{
                height: 400px;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 Forecast Dashboard</h1>
            <p>E-Grocery Forecasting System - Prediction Results & Model Analysis</p>
        </div>

        <div class="stats-grid">
            <div class="stat-card">
                <div class="icon">📊</div>
                <h3>Total Predictions</h3>
                <div class="value" data-target=""" + str(total_predictions) + """>0</div>
            </div>
            <div class="stat-card">
                <div class="icon">📈</div>
                <h3>Quantiles</h3>
                <div class="value" data-target=""" + str(num_quantiles) + """>0</div>
            </div>
            <div class="stat-card">
                <div class="icon">🤖</div>
                <h3>Models Trained</h3>
                <div class="value" data-target=""" + str(num_models) + """>0</div>
            </div>
            <div class="stat-card">
                <div class="icon">⭐</div>
                <h3>Top Features</h3>
                <div class="value" data-target=""" + str(len(top_features)) + """>0</div>
            </div>
        </div>

        <div class="tabs">
            <button class="tab active" onclick="showTab('predictions', this)">📊 Predictions</button>
            <button class="tab" onclick="showTab('metrics', this)">📉 Metrics</button>
            <button class="tab" onclick="showTab('features', this)">⭐ Feature Importance</button>
        </div>

        <div id="predictions" class="tab-content active">
            <div class="chart-container">
                <h2>Prediction Distribution</h2>
                <div id="prediction-distribution" class="chart"></div>
            </div>

            <div class="chart-container">
                <h2>Prediction Intervals</h2>
                <div id="prediction-intervals" class="chart"></div>
            </div>
        </div>

        <div id="metrics" class="tab-content">
            <div class="chart-container">
                <h2>Model Metrics</h2>
                <table class="metrics-table">
                    <thead>
                        <tr>
                            <th>Model</th>
                            <th>Metric</th>
                            <th>Value</th>
                        </tr>
                    </thead>
                    <tbody>
                        """ + ''.join([
                            f'<tr><td>{m["model"]}</td><td>{m["metric"]}</td><td>{m["value"]:.4f}</td></tr>'
                            for m in metrics_summary
                        ]) + """
                    </tbody>
                </table>
            </div>
        </div>

        <div id="features" class="tab-content">
            <div class="chart-container">
                <h2>Top Features by Importance (SHAP Values)</h2>
                <div id="feature-importance" class="chart"></div>
                <div class="feature-list">
                    """ + ''.join([
                        f'<div class="feature-item"><div class="feature-name">{f["feature"]}</div><div class="feature-importance">{f["importance"]:.4f}</div></div>'
                        for f in top_features[:14]
                    ]) + """
                </div>
            </div>
        </div>
    </div>

    <script>
        // Counter animation for stat cards
        function animateCounter(element) {{
            const target = parseInt(element.getAttribute('data-target'));
            const duration = 2000;
            const increment = target / (duration / 16);
            let current = 0;

            const updateCounter = () => {{
                current += increment;
                if (current < target) {{
                    element.textContent = Math.floor(current).toLocaleString();
                    requestAnimationFrame(updateCounter);
                }} else {{
                    element.textContent = target.toLocaleString();
                }}
            }};

            updateCounter();
        }}

        // Initialize counters when page loads
        window.addEventListener('load', () => {{
            document.querySelectorAll('.stat-card .value').forEach(counter => {{
                animateCounter(counter);
            }});
        }});

        // Tab switching with smooth transition
        function showTab(tabName, button) {{
            // Hide all tabs
            document.querySelectorAll('.tab-content').forEach(tab => {{
                tab.classList.remove('active');
            }});
            document.querySelectorAll('.tab').forEach(tab => {{
                tab.classList.remove('active');
            }});

            // Show selected tab
            document.getElementById(tabName).classList.add('active');
            button.classList.add('active');
        }}

        // Prediction Distribution Chart
        const predData = """ + json.dumps(predictions_json[:200]) + """;
        const predValues = predData.map(d => d['""" + (available_pred_cols[0] if available_pred_cols else "forecast_q50") + """'] || 0);

        Plotly.newPlot('prediction-distribution', [{{
            x: predValues,
            type: 'histogram',
            nbinsx: 40,
            marker: {{
                color: '#667eea',
                line: {{color: '#764ba2', width: 2}},
                opacity: 0.8
            }},
            name: 'Predictions'
        }}], {{
            title: {{
                text: 'Distribution of Predictions',
                font: {{size: 18, color: '#333'}}
            }},
            xaxis: {{
                title: {{text: 'Predicted Value', font: {{size: 14}}}},
                gridcolor: 'rgba(102, 126, 234, 0.1)'
            }},
            yaxis: {{
                title: {{text: 'Frequency', font: {{size: 14}}}},
                gridcolor: 'rgba(102, 126, 234, 0.1)'
            }},
            plot_bgcolor: 'white',
            paper_bgcolor: 'white',
            font: {{family: 'Poppins, sans-serif'}}
        }}, {{responsive: true}});

        // Prediction Intervals Chart
        const sampleSize = Math.min(200, predData.length);
        const sampleData = predData.slice(0, sampleSize);
        const xAxis = Array.from({{length: sampleSize}}, (_, i) => i);

        const traces = [];

        """ + (f"""
        if (predData.length > 0 && '{available_pred_cols[0] if len(available_pred_cols) > 0 else ""}' !== '') {{
            traces.push({{
                x: xAxis,
                y: sampleData.map(d => d['{available_pred_cols[0] if len(available_pred_cols) > 0 else "forecast_q05"}'] || 0),
                name: 'Q05',
                type: 'scatter',
                mode: 'lines',
                line: {{color: 'rgba(102, 126, 234, 0.4)', width: 2}},
                fill: 'none'
            }});
        }}

        if (predData.length > 0 && '{available_pred_cols[-1] if len(available_pred_cols) > 0 else ""}' !== '') {{
            traces.push({{
                x: xAxis,
                y: sampleData.map(d => d['{available_pred_cols[-1] if len(available_pred_cols) > 0 else "forecast_q95"}'] || 0),
                name: 'Q95',
                type: 'scatter',
                mode: 'lines',
                fill: 'tonexty',
                fillcolor: 'rgba(102, 126, 234, 0.15)',
                line: {{color: 'rgba(102, 126, 234, 0.4)', width: 2}}
            }});
        }}

        if (predData.length > 0 && '{available_pred_cols[len(available_pred_cols)//2] if len(available_pred_cols) > 0 else ""}' !== '') {{
            traces.push({{
                x: xAxis,
                y: sampleData.map(d => d['{available_pred_cols[len(available_pred_cols)//2] if len(available_pred_cols) > 0 else "forecast_q50"}'] || 0),
                name: 'Q50 (Median)',
                type: 'scatter',
                mode: 'lines',
                line: {{color: '#667eea', width: 3}}
            }});
        }}
        """ if available_pred_cols else """
        traces.push({
            x: xAxis,
            y: sampleData.map(d => d['forecast_q05'] || 0),
            name: 'Q05',
            type: 'scatter',
            mode: 'lines',
            line: {color: 'rgba(102, 126, 234, 0.4)', width: 2}
        });

        traces.push({
            x: xAxis,
            y: sampleData.map(d => d['forecast_q95'] || 0),
            name: 'Q95',
            type: 'scatter',
            mode: 'lines',
            fill: 'tonexty',
            fillcolor: 'rgba(102, 126, 234, 0.15)',
            line: {color: 'rgba(102, 126, 234, 0.4)', width: 2}
        });

        traces.push({
            x: xAxis,
            y: sampleData.map(d => d['forecast_q50'] || 0),
            name: 'Q50 (Median)',
            type: 'scatter',
            mode: 'lines',
            line: {color: '#667eea', width: 3}
        });
        """) + """

        Plotly.newPlot('prediction-intervals', traces, {{
            title: {{
                text: 'Prediction Intervals (Sample)',
                font: {{size: 18, color: '#333'}}
            }},
            xaxis: {{
                title: {{text: 'Sample Index', font: {{size: 14}}}},
                gridcolor: 'rgba(102, 126, 234, 0.1)'
            }},
            yaxis: {{
                title: {{text: 'Predicted Value', font: {{size: 14}}}},
                gridcolor: 'rgba(102, 126, 234, 0.1)'
            }},
            plot_bgcolor: 'white',
            paper_bgcolor: 'white',
            hovermode: 'x unified',
            font: {{family: 'Poppins, sans-serif'}},
            legend: {{
                x: 0.02,
                y: 0.98,
                bgcolor: 'rgba(255, 255, 255, 0.8)',
                bordercolor: 'rgba(102, 126, 234, 0.3)',
                borderwidth: 1
            }}
        }}, {{responsive: true}});

        // Feature Importance Chart
        const features = """ + json.dumps(top_features[:20]) + """;
        const featureNames = features.map(f => f.feature);
        const featureImportance = features.map(f => f.importance);

        // Create gradient colors for bars
        const maxImportance = Math.max(...featureImportance);
        const colors = featureImportance.map(imp => {{
            const ratio = maxImportance > 0 ? imp / maxImportance : 0;
            return `rgba(102, 126, 234, ${{0.3 + ratio * 0.7}})`;
        }});

        Plotly.newPlot('feature-importance', [{{
            x: featureImportance,
            y: featureNames,
            type: 'bar',
            orientation: 'h',
            marker: {{
                color: colors,
                line: {{color: '#764ba2', width: 1.5}},
                opacity: 0.9
            }},
            text: featureImportance.map(v => v.toFixed(4)),
            textposition: 'outside',
            textfont: {{size: 11, color: '#667eea'}}
        }}], {{
            title: {{
                text: 'Top Features by SHAP Importance',
                font: {{size: 18, color: '#333'}}
            }},
            xaxis: {{
                title: {{text: 'SHAP Importance', font: {{size: 14}}}},
                gridcolor: 'rgba(102, 126, 234, 0.1)'
            }},
            yaxis: {{
                title: {{text: 'Feature', font: {{size: 14}}}},
                gridcolor: 'rgba(102, 126, 234, 0.1)'
            }},
            plot_bgcolor: 'white',
            paper_bgcolor: 'white',
            margin: {{l: 220, r: 50, t: 50, b: 50}},
            font: {{family: 'Poppins, sans-serif'}}
        }}, {{responsive: true}});
    </script>
</body>
</html>
    """

    return html


def main():
    """Main function để tạo dashboard."""
    logger.info("=" * 70)
    logger.info("GENERATING FORECAST DASHBOARD")
    logger.info("=" * 70)

    try:
        # Load data
        logger.info("Loading predictions...")
        predictions = load_predictions()

        logger.info("Loading metrics...")
        metrics = load_metrics()

        logger.info("Loading SHAP values...")
        shap_values = load_shap_values()

        # Create dashboard
        logger.info("Creating dashboard HTML...")
        html = create_dashboard_html(predictions, metrics, shap_values)

        # Save dashboard
        dashboard_path = OUTPUT_FILES['dashboard_html']
        dashboard_path.parent.mkdir(parents=True, exist_ok=True)

        with open(dashboard_path, 'w', encoding='utf-8') as f:
            f.write(html)

        logger.info(f"✓ Dashboard saved to: {dashboard_path}")
        logger.info("=" * 70)
        logger.info("DASHBOARD GENERATION COMPLETE")
        logger.info("=" * 70)

    except Exception as e:
        logger.error(f"Error generating dashboard: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

