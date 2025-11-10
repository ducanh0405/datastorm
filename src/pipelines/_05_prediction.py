"""
Prediction/Inference Pipeline
==============================
Loads trained models and generates forecasts for new data.
"""
import json
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# FIX: Add PROJECT_ROOT to sys.path BEFORE importing from src
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import CATEGORICAL_FEATURES, NUMERIC_FEATURES, OUTPUT_FILES, QUANTILES

logger = logging.getLogger(__name__)


class QuantileForecaster:
    """Wrapper class for quantile regression models."""

    def __init__(self, models_dir: Path = None):
        """
        Initialize forecaster with trained models.

        Args:
            models_dir: Directory containing model files. Defaults to config.
        """
        if models_dir is None:
            models_dir = OUTPUT_FILES['models_dir']

        self.models_dir = models_dir
        self.models = {}
        self.feature_config = None
        self.quantiles = QUANTILES

        self._load_models()
        self._load_feature_config()

    def _load_models(self):
        """Load all quantile models."""
        logger.info("Loading quantile models...")
        for alpha in self.quantiles:
            model_path = self.models_dir / f'q{int(alpha*100):02d}_forecaster.joblib'
            if model_path.exists():
                self.models[alpha] = joblib.load(model_path)
                logger.info(f"  Loaded Q{int(alpha*100):02d} model")
            else:
                logger.error(f"  Model not found: {model_path}")
                raise FileNotFoundError(f"Model not found: {model_path}")

    def _load_feature_config(self):
        """Load feature configuration."""
        config_path = OUTPUT_FILES['model_features']
        if config_path.exists():
            with open(config_path) as f:
                self.feature_config = json.load(f)
            logger.info("  Loaded feature config")
        else:
            logger.warning("  Feature config not found, using default")
            self.feature_config = {
                'all_features': NUMERIC_FEATURES + CATEGORICAL_FEATURES,
                'categorical_features': CATEGORICAL_FEATURES
            }

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features for prediction.

        Args:
            df: Input dataframe with raw features

        Returns:
            DataFrame with prepared features
        """
        # Get required features
        required_features = self.feature_config['all_features']
        categorical_features = self.feature_config.get('categorical_features', [])

        # Check missing features
        missing_features = set(required_features) - set(df.columns)
        if missing_features:
            logger.warning(f"Missing features (will be filled with 0/NaN): {missing_features}")
            for feat in missing_features:
                if feat in categorical_features:
                    df[feat] = 'unknown'
                else:
                    df[feat] = 0

        # Select only required features
        x = df[required_features].copy()

        # Convert categorical to category dtype and ensure 'unknown' is in categories
        for col in categorical_features:
            if col in x.columns:
                # Convert to category and add 'unknown' if not present
                x[col] = x[col].astype('category')
                if 'unknown' not in x[col].cat.categories:
                    x[col] = x[col].cat.add_categories(['unknown'])

        # Fill NaN - handle categorical columns properly
        fill_values = {}
        for col in x.columns:
            if x[col].dtype.name == 'category':
                # For categorical, use 'unknown' as fill value (should be in categories)
                fill_values[col] = 'unknown'
            else:
                # For numeric columns, use 0
                fill_values[col] = 0

        x = x.fillna(fill_values)

        return x

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate predictions for input data.

        Args:
            df: Input dataframe with features

        Returns:
            DataFrame with predictions for all quantiles
        """
        logger.info("Generating predictions...")

        # Prepare features
        x = self.prepare_features(df)

        # Generate predictions for each quantile
        predictions = {}
        for alpha in self.quantiles:
            y_pred = self.models[alpha].predict(x)
            y_pred = np.maximum(y_pred, 0)  # Clip negative predictions
            predictions[f'forecast_q{int(alpha*100):02d}'] = y_pred

        # Create results dataframe
        results = df[['PRODUCT_ID', 'STORE_ID', 'WEEK_NO']].copy()
        for key, values in predictions.items():
            results[key] = values

        # Calculate prediction interval dynamically (not hardcoded)
        quantiles_sorted = sorted(self.quantiles)
        min_quantile = quantiles_sorted[0]
        max_quantile = quantiles_sorted[-1]
        median_quantile = 0.50 if 0.50 in quantiles_sorted else quantiles_sorted[len(quantiles_sorted) // 2]

        results['forecast_lower'] = results[f'forecast_q{int(min_quantile*100):02d}']
        results['forecast_median'] = results[f'forecast_q{int(median_quantile*100):02d}']
        results['forecast_upper'] = results[f'forecast_q{int(max_quantile*100):02d}']
        results['prediction_interval'] = results['forecast_upper'] - results['forecast_lower']

        logger.info(f"Generated predictions for {len(results)} records")

        return results

    def predict_single(self, product_id: str, store_id: str, week_no: int,
                      features: dict) -> dict:
        """
        Predict for a single product-store-week combination.

        Args:
            product_id: Product ID
            store_id: Store ID
            week_no: Week number
            features: Dictionary of feature values

        Returns:
            Dictionary with predictions
        """
        # Create single row dataframe
        df = pd.DataFrame({
            'PRODUCT_ID': [product_id],
            'STORE_ID': [store_id],
            'WEEK_NO': [week_no],
            **features
        })

        # Predict
        results = self.predict(df)

        # Get quantiles dynamically
        quantiles_sorted = sorted(self.quantiles)
        min_quantile = quantiles_sorted[0]
        max_quantile = quantiles_sorted[-1]
        median_quantile = 0.50 if 0.50 in quantiles_sorted else quantiles_sorted[len(quantiles_sorted) // 2]

        result_dict = {
            'product_id': product_id,
            'store_id': store_id,
            'week_no': week_no,
            'forecast_q05': float(results.get('forecast_q05', pd.Series([np.nan])).iloc[0])
                if f'forecast_q05' in results.columns else float(results[f'forecast_q{int(min_quantile*100):02d}'].iloc[0]),
            'forecast_q50': float(results.get('forecast_q50', pd.Series([np.nan])).iloc[0])
                if f'forecast_q50' in results.columns else float(results[f'forecast_q{int(median_quantile*100):02d}'].iloc[0]),
            'forecast_q95': float(results.get('forecast_q95', pd.Series([np.nan])).iloc[0])
                if f'forecast_q95' in results.columns else float(results[f'forecast_q{int(max_quantile*100):02d}'].iloc[0]),
            'prediction_interval': float(results['prediction_interval'].iloc[0])
        }

        # Also include all quantile predictions (dynamic)
        for alpha in quantiles_sorted:
            key = f'forecast_q{int(alpha*100):02d}'
            if key in results.columns:
                result_dict[key] = float(results[key].iloc[0])

        return result_dict


def predict_on_test_set():
    """Generate predictions on test set for evaluation."""
    logger.info("=" * 70)
    logger.info("GENERATING PREDICTIONS ON TEST SET")
    logger.info("=" * 70)

    # Load test data
    from src.pipelines._03_model_training import load_data, prepare_data

    df = load_data(OUTPUT_FILES['master_feature_table'])
    x_train, x_test, y_train, y_test, features, cat_features = prepare_data(df)

    # Add back identifiers
    test_indices = df[df['WEEK_NO'] >= df['WEEK_NO'].quantile(0.8)].index
    x_test_with_ids = df.loc[test_indices, ['PRODUCT_ID', 'STORE_ID', 'WEEK_NO']].copy()
    x_test_with_ids = pd.concat([x_test_with_ids, x_test], axis=1)

    # Initialize forecaster
    forecaster = QuantileForecaster()

    # Generate predictions
    predictions = forecaster.predict(x_test_with_ids)
    predictions['actual'] = y_test.values

    # Calculate metrics
    from src.pipelines._03_model_training import evaluate_quantile_models
    metrics = evaluate_quantile_models(
        forecaster.models,
        x_test,
        y_test
    )

    # Save predictions
    output_path = OUTPUT_FILES['reports_dir'] / 'predictions_test_set.csv'
    predictions.to_csv(output_path, index=False)
    logger.info(f"Predictions saved to: {output_path}")

    return predictions, metrics


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Test prediction on test set
    predictions, metrics = predict_on_test_set()
    print("\nPredictions preview:")
    print(predictions.head(10))
