"""
Quantile Prediction Pipeline với SHAP Values
=============================================
Prediction pipeline với hỗ trợ nhiều models và SHAP explanation.
"""
import logging
import sys
import warnings
from pathlib import Path

# Setup project path FIRST before any other imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Now import config
try:
    from src.config import (
        MODEL_CONFIGS,
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

import joblib
import numpy as np
import pandas as pd

# SHAP imports
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    warnings.warn("SHAP not available. Install with: pip install shap", stacklevel=2)

warnings.filterwarnings('ignore')


class QuantileForecaster:
    """
    Quantile Forecaster class với hỗ trợ nhiều models và SHAP values.
    """

    def __init__(self, model_types: list[str] | None = None):
        """
        Initialize QuantileForecaster.

        Args:
            model_types: List các model types để load. Nếu None, load tất cả models có sẵn.
        """
        self.model_types = model_types or TRAINING_CONFIG.get('model_types', ['lightgbm'])
        self.quantiles = TRAINING_CONFIG['quantiles']
        self.models = {}  # {model_type: {quantile: model}}
        self.feature_names = None
        self.categorical_features = None
        self.config = get_dataset_config()

        logger.info(f"Initializing QuantileForecaster with models: {self.model_types}")

    def load_models(self, models_dir: Path | None = None) -> None:
        """
        Load trained models từ models directory.

        Args:
            models_dir: Directory chứa models. Nếu None, dùng OUTPUT_FILES['models_dir'].
        """
        if models_dir is None:
            models_dir = OUTPUT_FILES['models_dir']

        logger.info(f"Loading models from: {models_dir}")

        # Load model_features.json để lấy feature names
        features_path = OUTPUT_FILES['model_features']
        if features_path.exists():
            with open(features_path) as f:
                features_config = json.load(f)
                self.feature_names = features_config.get('all_features', [])
                self.categorical_features = features_config.get('categorical_features', [])
                logger.info(f"Loaded {len(self.feature_names)} features")
                logger.info(f"Loaded {len(self.categorical_features)} categorical features")
        else:
            logger.warning(f"Model features config not found: {features_path}")

        # Load models cho từng model type và quantile
        for model_type in self.model_types:
            self.models[model_type] = {}

            for quantile in self.quantiles:
                # Tạo tên file model
                model_filename = f"{model_type}_q{int(quantile*100):02d}_forecaster.joblib"
                model_path = models_dir / model_filename

                if model_path.exists():
                    try:
                        model = joblib.load(model_path)
                        self.models[model_type][quantile] = model
                        logger.info(f"  Loaded {model_type} Q{int(quantile*100):02d} model")
                    except Exception as e:
                        logger.error(f"  Error loading {model_type} Q{int(quantile*100):02d}: {e}")
                else:
                    logger.warning(f"  Model not found: {model_path}")

        # Kiểm tra xem có models nào được load không
        total_models = sum(len(models) for models in self.models.values())
        if total_models == 0:
            raise FileNotFoundError(f"No models found in {models_dir}. Please train models first.")

        logger.info(f"Loaded {total_models} models across {len(self.models)} model types")

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Chuẩn bị features cho prediction.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame với features đã được chuẩn bị
        """
        if self.feature_names is None:
            raise ValueError("Feature names not loaded. Call load_models() first.")

        # Lọc features có trong df
        [f for f in self.feature_names if f in df.columns]
        missing_features = [f for f in self.feature_names if f not in df.columns]

        if missing_features:
            logger.warning(f"Missing {len(missing_features)} features: {missing_features[:10]}...")
            # Fill missing features với 0
            for feature in missing_features:
                df[feature] = 0

        # Chọn features
        X = df[self.feature_names].copy()

        # Fill NaN
        numeric_features = X.select_dtypes(include=[np.number]).columns
        X[numeric_features] = X[numeric_features].fillna(0)

        # Chuẩn bị categorical features
        for col in self.categorical_features:
            if col in X.columns:
                if X[col].dtype.name not in ['category', 'object']:
                    X[col] = X[col].astype('category')
                # Fill missing categories với 'Unknown'
                if X[col].isnull().any():
                    if hasattr(X[col], 'cat'):
                        X[col] = X[col].cat.add_categories(['Unknown']).fillna('Unknown')
                    else:
                        X[col] = X[col].fillna('Unknown')

        return X

    def predict(self, df: pd.DataFrame, model_type: str | None = None) -> pd.DataFrame:
        """
        Generate predictions cho input data.

        Args:
            df: Input DataFrame
            model_type: Model type để predict. Nếu None, dùng model đầu tiên trong self.model_types.

        Returns:
            DataFrame với predictions cho từng quantile
        """
        if not self.models:
            raise ValueError("No models loaded. Call load_models() first.")

        if model_type is None:
            model_type = self.model_types[0]

        if model_type not in self.models:
            raise ValueError(f"Model type '{model_type}' not loaded. Available: {list(self.models.keys())}")

        logger.info(f"Generating predictions using {model_type} models...")

        # Chuẩn bị features
        X = self.prepare_features(df)

        # Predict cho từng quantile
        predictions = df.copy()
        models = self.models[model_type]

        for quantile in self.quantiles:
            if quantile in models:
                model = models[quantile]
                y_pred = model.predict(X)
                y_pred = np.maximum(y_pred, 0)  # Clip negative predictions
                predictions[f'forecast_q{int(quantile*100):02d}'] = y_pred
            else:
                logger.warning(f"Model for Q{int(quantile*100):02d} not found. Skipping.")

        # Tính prediction interval
        if len(self.quantiles) >= 2:
            lower_q = min(self.quantiles)
            upper_q = max(self.quantiles)
            lower_col = f'forecast_q{int(lower_q*100):02d}'
            upper_col = f'forecast_q{int(upper_q*100):02d}'
            if lower_col in predictions.columns and upper_col in predictions.columns:
                predictions['prediction_interval'] = predictions[upper_col] - predictions[lower_col]
                predictions['prediction_center'] = (predictions[upper_col] + predictions[lower_col]) / 2

        logger.info(f"Generated predictions for {len(predictions)} samples")
        return predictions

    def predict_shap(self, df: pd.DataFrame, model_type: str | None = None,
                    sample_size: int | None = None, quantile: float = 0.50) -> tuple[pd.DataFrame, dict]:
        """
        Generate predictions với SHAP values.

        Args:
            df: Input DataFrame
            model_type: Model type để predict. Nếu None, dùng model đầu tiên.
            sample_size: Số lượng samples để tính SHAP. Nếu None, dùng SHAP_CONFIG['sample_size'].
            quantile: Quantile để tính SHAP values (thường dùng median 0.50).

        Returns:
            Tuple của (predictions DataFrame, SHAP values dict)
        """
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP not available. Install with: pip install shap")

        if not self.models:
            raise ValueError("No models loaded. Call load_models() first.")

        if model_type is None:
            model_type = self.model_types[0]

        if model_type not in self.models:
            raise ValueError(f"Model type '{model_type}' not loaded. Available: {list(self.models.keys())}")

        if quantile not in self.models[model_type]:
            raise ValueError(f"Model for quantile {quantile} not found.")

        logger.info(f"Generating SHAP values using {model_type} Q{int(quantile*100):02d} model...")

        # Chuẩn bị features
        X = self.prepare_features(df)

        # Sample data nếu cần
        sample_size = sample_size or SHAP_CONFIG.get('sample_size', 1000)
        if len(X) > sample_size:
            sample_idx = np.random.choice(len(X), sample_size, replace=False)
            X_sample = X.iloc[sample_idx].copy()
            logger.info(f"Sampling {sample_size} rows for SHAP calculation")
        else:
            X_sample = X.copy()
            sample_idx = np.arange(len(X))

        # Get model
        model = self.models[model_type][quantile]

        # Tính SHAP values
        try:
            # TreeExplainer cho tree-based models
            if model_type in ['lightgbm', 'catboost', 'random_forest']:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_sample)

                # SHAP values có thể là array hoặc list
                if isinstance(shap_values, list):
                    shap_values = shap_values[0]  # Lấy giá trị đầu tiên nếu là list

                # Tạo DataFrame SHAP values
                shap_df = pd.DataFrame(
                    shap_values,
                    columns=X_sample.columns,
                    index=X_sample.index
                )

                # Tính base value
                base_value = explainer.expected_value
                if isinstance(base_value, np.ndarray):
                    base_value = base_value[0] if len(base_value) > 0 else 0.0

                logger.info(f"Calculated SHAP values for {len(shap_df)} samples")
                logger.info(f"Base value: {base_value:.4f}")

                # Tạo SHAP summary
                shap_summary = {
                    'shap_values': shap_df,
                    'base_value': base_value,
                    'feature_names': X_sample.columns.tolist(),
                    'sample_indices': sample_idx.tolist(),
                    'model_type': model_type,
                    'quantile': quantile,
                }

                return X_sample, shap_summary
            else:
                raise ValueError(f"SHAP not supported for model type: {model_type}")

        except Exception as e:
            logger.error(f"Error calculating SHAP values: {e}")
            raise

    def predict_with_shap(self, df: pd.DataFrame, model_type: str | None = None,
                         sample_size: int | None = None) -> dict[str, Any]:
        """
        Generate predictions và SHAP values cùng lúc.

        Args:
            df: Input DataFrame
            model_type: Model type để predict
            sample_size: Số lượng samples để tính SHAP

        Returns:
            Dict chứa predictions và SHAP values
        """
        # Generate predictions
        predictions = self.predict(df, model_type=model_type)

        # Generate SHAP values (sử dụng median quantile)
        median_quantile = 0.50
        try:
            X_sample, shap_summary = self.predict_shap(
                df, model_type=model_type, sample_size=sample_size, quantile=median_quantile
            )

            return {
                'predictions': predictions,
                'shap_values': shap_summary,
            }
        except Exception as e:
            logger.warning(f"Could not generate SHAP values: {e}")
            return {
                'predictions': predictions,
                'shap_values': None,
            }


def evaluate_predictions(predictions: pd.DataFrame, y_true: pd.Series,
                        quantiles: list[float]) -> dict[str, float]:
    """
    Evaluate predictions với nhiều metrics.

    Args:
        predictions: DataFrame với predictions
        y_true: True values
        quantiles: List quantiles

    Returns:
        Dict với metrics
    """
    from sklearn.metrics import (
        mean_absolute_error,
        mean_absolute_percentage_error,
        mean_pinball_loss,
        mean_squared_error,
        r2_score,
    )

    metrics = {}

    # Metrics cho từng quantile
    for quantile in quantiles:
        col = f'forecast_q{int(quantile*100):02d}'
        if col in predictions.columns:
            y_pred = predictions[col]

            # Pinball loss
            pinball = mean_pinball_loss(y_true, y_pred, alpha=quantile)
            metrics[f'q{int(quantile*100):02d}_pinball_loss'] = pinball

            # MAE
            mae = mean_absolute_error(y_true, y_pred)
            metrics[f'q{int(quantile*100):02d}_mae'] = mae

            # RMSE
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            metrics[f'q{int(quantile*100):02d}_rmse'] = rmse

            # MAPE với threshold để tránh chia cho 0 hoặc giá trị quá nhỏ
            # Chỉ tính MAPE cho các giá trị > threshold (default: 0.1)
            threshold = 0.1
            valid_mask = y_true > threshold
            if valid_mask.sum() > 0:
                mape = mean_absolute_percentage_error(
                    y_true[valid_mask], y_pred[valid_mask]
                )
                metrics[f'q{int(quantile*100):02d}_mape'] = mape
                metrics[f'q{int(quantile*100):02d}_mape_valid_samples'] = int(valid_mask.sum())
                metrics[f'q{int(quantile*100):02d}_mape_total_samples'] = len(y_true)
            else:
                # Nếu không có giá trị hợp lệ, set MAPE = None và cảnh báo
                metrics[f'q{int(quantile*100):02d}_mape'] = None
                metrics[f'q{int(quantile*100):02d}_mape_warning'] = 'No valid samples for MAPE calculation (all values <= threshold)'
                logger.warning(f"MAPE không thể tính cho quantile {quantile}: không có giá trị > {threshold}")

    # Coverage
    if len(quantiles) >= 2:
        lower_q = min(quantiles)
        upper_q = max(quantiles)
        lower_col = f'forecast_q{int(lower_q*100):02d}'
        upper_col = f'forecast_q{int(upper_q*100):02d}'
        if lower_col in predictions.columns and upper_col in predictions.columns:
            coverage = ((y_true >= predictions[lower_col]) &
                       (y_true <= predictions[upper_col])).mean()
            metrics[f'coverage_{(upper_q-lower_q)*100:.0f}%'] = coverage

    # R2 score (sử dụng median quantile)
    median_col = f'forecast_q{int(0.50*100):02d}'
    if median_col in predictions.columns:
        r2 = r2_score(y_true, predictions[median_col])
        metrics['r2_score'] = r2

    return metrics


def main():
    """Main function để chạy prediction pipeline."""
    import argparse

    parser = argparse.ArgumentParser(description='Generate predictions với SHAP values')
    parser.add_argument('--model-type', type=str, default=None,
                       help='Model type để predict (lightgbm, catboost, random_forest)')
    parser.add_argument('--shap', action='store_true', help='Generate SHAP values')
    parser.add_argument('--sample-size', type=int, default=None,
                       help='Số lượng samples để tính SHAP')
    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("STARTING PREDICTION PIPELINE")
    logger.info("=" * 70)

    # Load data
    logger.info("Loading data...")
    df = pd.read_parquet(OUTPUT_FILES['master_feature_table'])
    config = get_dataset_config()
    target_col = config['target_column']
    time_col = config['time_column']

    # Split test set
    time_col_data = pd.to_datetime(df[time_col])
    cutoff_percentile = TRAINING_CONFIG['train_test_split']['cutoff_percentile']
    cutoff_time = time_col_data.quantile(cutoff_percentile)

    test_mask = time_col_data >= cutoff_time
    df_test = df[test_mask].copy()
    y_test = df_test[target_col]

    logger.info(f"Test set size: {len(df_test)}")

    # Initialize forecaster
    logger.info("Initializing QuantileForecaster...")
    forecaster = QuantileForecaster()
    forecaster.load_models()

    # Generate predictions
    logger.info("Generating predictions...")
    if args.shap:
        results = forecaster.predict_with_shap(
            df_test, model_type=args.model_type, sample_size=args.sample_size
        )
        predictions = results['predictions']
        shap_summary = results['shap_values']
    else:
        predictions = forecaster.predict(df_test, model_type=args.model_type)
        shap_summary = None

    # Evaluate
    logger.info("Evaluating predictions...")
    metrics = evaluate_predictions(predictions, y_test, TRAINING_CONFIG['quantiles'])

    # Save predictions - tối ưu: lưu cả Parquet (nhỏ hơn) và CSV compressed
    predictions_path_csv = OUTPUT_FILES['predictions_test']
    predictions_path_parquet = predictions_path_csv.with_suffix('.parquet')

    # Lưu Parquet (format tối ưu, nhỏ hơn nhiều)
    predictions.to_parquet(predictions_path_parquet, index=False, compression='snappy')
    logger.info(f"Predictions saved to Parquet: {predictions_path_parquet} ({predictions_path_parquet.stat().st_size / 1024 / 1024:.2f} MB)")

    # Lưu CSV với compression (cho compatibility)
    csv_gz_path = predictions_path_csv.with_suffix(predictions_path_csv.suffix + '.gz')
    predictions.to_csv(csv_gz_path, index=False, compression='gzip')
    csv_size = csv_gz_path.stat().st_size / 1024 / 1024
    logger.info(f"Predictions saved to CSV (gzip): {csv_gz_path} ({csv_size:.2f} MB)")

    # Lưu CSV không nén (nếu cần cho compatibility)
    # predictions.to_csv(predictions_path_csv, index=False)

    # Save metrics
    metrics_path = OUTPUT_FILES['model_metrics']
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    logger.info(f"Metrics saved to: {metrics_path}")

    # Save SHAP values nếu có
    if shap_summary is not None:
        shap_dir = OUTPUT_FILES['shap_values_dir']
        shap_dir.mkdir(parents=True, exist_ok=True)

        # Save SHAP values DataFrame
        shap_df_path = shap_dir / 'shap_values.csv'
        shap_summary['shap_values'].to_csv(shap_df_path)
        logger.info(f"SHAP values saved to: {shap_df_path}")

        # Save SHAP summary
        shap_summary_path = shap_dir / 'shap_summary.json'
        shap_summary_to_save = {
            'base_value': float(shap_summary['base_value']),
            'feature_names': shap_summary['feature_names'],
            'model_type': shap_summary['model_type'],
            'quantile': shap_summary['quantile'],
            'sample_size': len(shap_summary['shap_values']),
        }
        with open(shap_summary_path, 'w') as f:
            json.dump(shap_summary_to_save, f, indent=4)
        logger.info(f"SHAP summary saved to: {shap_summary_path}")

    # Print metrics
    logger.info("=" * 70)
    logger.info("PREDICTION METRICS")
    logger.info("=" * 70)
    for key, value in metrics.items():
        logger.info(f"  {key}: {value:.4f}")

    logger.info("=" * 70)
    logger.info("PREDICTION PIPELINE COMPLETE")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
