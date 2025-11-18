"""
Ensemble Models Pipeline
========================
Meta-models that combine quantile predictions for improved forecasting.
"""
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class EnsembleForecaster:
    """
    Meta-model forecaster using ensemble methods on quantile predictions.

    This class provides two ensemble approaches:
    1. Ensemble Forecast: Weighted average across multiple quantiles
    2. Uncertainty-Weighted Forecast: Weight by prediction interval sharpness
    """

    def __init__(self, quantile_forecaster):
        """
        Initialize ensemble forecaster with base quantile forecaster.

        Args:
            quantile_forecaster: Trained QuantileForecaster instance
        """
        self.quantile_forecaster = quantile_forecaster
        self.available_quantiles = None

        logger.info("EnsembleForecaster initialized")

    def ensemble_forecast(self, predictions_df: pd.DataFrame) -> pd.Series:
        """
        Create ensemble forecast using weighted average across quantiles.

        Weights: Q05=10%, Q25=20%, Q50=40%, Q75=20%, Q95=10%
        Emphasizes median (Q50) while incorporating conservative/optimistic views.

        Args:
            predictions_df: DataFrame with quantile predictions

        Returns:
            Series with ensemble forecasts
        """
        # Check available quantiles
        available_cols = [col for col in ['forecast_q05', 'forecast_q10', 'forecast_q25',
                                        'forecast_q50', 'forecast_q75', 'forecast_q90', 'forecast_q95']
                         if col in predictions_df.columns]

        if len(available_cols) < 3:
            logger.warning(f"Only {len(available_cols)} quantiles available, using simple average")
            # Fallback to simple average of available quantiles
            quantile_cols = [col for col in predictions_df.columns if col.startswith('forecast_q')]
            return predictions_df[quantile_cols].mean(axis=1)

        # Define weights for ensemble (emphasize Q50, balance others)
        weights = {
            'forecast_q05': 0.1,   # 10% - conservative
            'forecast_q10': 0.1,   # 10% - slightly conservative
            'forecast_q25': 0.2,   # 20% - moderately conservative
            'forecast_q50': 0.4,   # 40% - main estimate
            'forecast_q75': 0.15,  # 15% - moderately optimistic
            'forecast_q90': 0.025, # 2.5% - slightly optimistic
            'forecast_q95': 0.025  # 2.5% - very optimistic
        }

        # Only use available quantiles
        available_weights = {k: v for k, v in weights.items() if k in available_cols}

        # Normalize weights
        total_weight = sum(available_weights.values())
        normalized_weights = {k: v/total_weight for k, v in available_weights.items()}

        logger.info(f"Ensemble weights: {normalized_weights}")

        # Calculate weighted average
        ensemble_pred = sum(predictions_df[col] * weight
                           for col, weight in normalized_weights.items())

        return ensemble_pred

    def uncertainty_weighted_forecast(self, predictions_df: pd.DataFrame) -> pd.Series:
        """
        Create uncertainty-weighted forecast based on prediction interval sharpness.

        Sharper intervals (narrower) get higher weights as they indicate more confidence.

        Args:
            predictions_df: DataFrame with quantile predictions

        Returns:
            Series with uncertainty-weighted forecasts
        """
        # Calculate sharpness (inverse of interval width)
        if 'prediction_interval' in predictions_df.columns:
            sharpness = 1 / (predictions_df['prediction_interval'] + 1e-6)
        else:
            # Calculate from available quantiles
            upper_cols = [col for col in predictions_df.columns if col.startswith('forecast_q') and
                         float(col.split('_q')[1]) > 50]  # Upper quantiles
            lower_cols = [col for col in predictions_df.columns if col.startswith('forecast_q') and
                         float(col.split('_q')[1]) < 50]  # Lower quantiles

            if upper_cols and lower_cols:
                # Approximate interval width
                upper_avg = predictions_df[upper_cols].mean(axis=1)
                lower_avg = predictions_df[lower_cols].mean(axis=1)
                interval_width = upper_avg - lower_avg
                sharpness = 1 / (interval_width + 1e-6)
            else:
                logger.warning("Cannot calculate intervals, using uniform weights")
                sharpness = np.ones(len(predictions_df))

        # Use Q50 as base prediction, or average if not available
        if 'forecast_q50' in predictions_df.columns:
            base_prediction = predictions_df['forecast_q50']
        else:
            quantile_cols = [col for col in predictions_df.columns if col.startswith('forecast_q')]
            base_prediction = predictions_df[quantile_cols].mean(axis=1)

        # Weight by sharpness (normalized)
        sharpness_normalized = sharpness / sharpness.sum()

        # Apply weighting
        weighted_prediction = base_prediction * sharpness_normalized

        logger.info(f"Uncertainty weighting applied. Mean sharpness: {sharpness.mean():.4f}")

        return weighted_prediction

    def predict_ensemble(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate ensemble predictions for input data.

        Args:
            df: Input data for prediction

        Returns:
            DataFrame with original quantile predictions + ensemble forecast
        """
        logger.info("Generating ensemble predictions...")

        # Get base quantile predictions
        quantile_preds = self.quantile_forecaster.predict(df)

        # Apply ensemble method
        ensemble_pred = self.ensemble_forecast(quantile_preds)

        # Add to results
        quantile_preds['ensemble_forecast'] = ensemble_pred

        logger.info(f"Ensemble predictions generated. Shape: {quantile_preds.shape}")

        return quantile_preds

    def predict_uncertainty_weighted(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate uncertainty-weighted predictions for input data.

        Args:
            df: Input data for prediction

        Returns:
            DataFrame with original quantile predictions + uncertainty-weighted forecast
        """
        logger.info("Generating uncertainty-weighted predictions...")

        # Get base quantile predictions
        quantile_preds = self.quantile_forecaster.predict(df)

        # Apply uncertainty weighting
        uncertainty_pred = self.uncertainty_weighted_forecast(quantile_preds)

        # Add to results
        quantile_preds['uncertainty_weighted_forecast'] = uncertainty_pred

        logger.info(f"Uncertainty-weighted predictions generated. Shape: {quantile_preds.shape}")

        return quantile_preds


def main() -> None:
    """
    Main function for ensemble predictions pipeline.
    Compatible with orchestrator calling pattern.
    """
    logger.info("=" * 70)
    logger.info("ENSEMBLE PREDICTIONS PIPELINE")
    logger.info("=" * 70)

    # Setup project path and logging
    from src.config import OUTPUT_FILES, setup_logging, setup_project_path
    setup_project_path()
    setup_logging()

    import pandas as pd

    from src.pipelines._05_prediction import QuantileForecaster

    # Load data
    logger.info("Loading master feature table...")
    df = pd.read_parquet(OUTPUT_FILES['master_feature_table'])

    # Use test set (last 20% of time period - same as prediction pipeline)
    test_cutoff = df['hour_timestamp'].quantile(0.8)
    df_test = df[df['hour_timestamp'] >= test_cutoff].copy()
    logger.info(f"Predicting for {len(df_test)} test records")

    # Initialize forecasters
    logger.info("Initializing forecasters...")
    quantile_forecaster = QuantileForecaster()
    quantile_forecaster.load_models()  # Load trained models
    ensemble_forecaster = EnsembleForecaster(quantile_forecaster)

    # Generate ensemble predictions
    logger.info("Generating ensemble predictions...")
    ensemble_predictions = ensemble_forecaster.predict_ensemble(df_test)

    # Save results
    output_path = OUTPUT_FILES['reports_dir'] / 'ensemble_predictions.csv'
    ensemble_predictions.to_csv(output_path, index=False)
    logger.info(f"✅ Ensemble predictions saved to: {output_path}")

    # Print summary
    logger.info("ENSEMBLE PREDICTIONS SUMMARY")
    logger.info(f"Total predictions: {len(ensemble_predictions)}")
    logger.info("Forecast statistics:")
    logger.info(f"{ensemble_predictions[['forecast_q50', 'ensemble_forecast']].describe()}")

    logger.info("Ensemble pipeline completed successfully")


if __name__ == "__main__":
    # Setup project path and logging
    from src.config import OUTPUT_FILES, setup_logging, setup_project_path
    setup_project_path()
    setup_logging()

    import pandas as pd

    from src.pipelines._05_prediction import QuantileForecaster

    logger.info("=" * 70)
    logger.info("RUNNING ENSEMBLE PREDICTIONS")
    logger.info("=" * 70)

    # Load data
    logger.info("Loading master feature table...")
    df = pd.read_parquet(OUTPUT_FILES['master_feature_table'])

    # Use test set (last 20% of time period - same as prediction pipeline)
    test_cutoff = df['hour_timestamp'].quantile(0.8)
    df_test = df[df['hour_timestamp'] >= test_cutoff].copy()
    logger.info(f"Predicting for {len(df_test)} test records")

    # Initialize forecasters
    logger.info("Initializing forecasters...")
    quantile_forecaster = QuantileForecaster()
    quantile_forecaster.load_models()  # Load trained models
    ensemble_forecaster = EnsembleForecaster(quantile_forecaster)

    # Generate ensemble predictions
    logger.info("Generating ensemble predictions...")
    ensemble_predictions = ensemble_forecaster.predict_ensemble(df_test)

    # Save results
    output_path = OUTPUT_FILES['reports_dir'] / 'ensemble_predictions.csv'
    ensemble_predictions.to_csv(output_path, index=False)
    logger.info(f"✅ Ensemble predictions saved to: {output_path}")

    # Print summary
    print("\n" + "=" * 70)
    print("ENSEMBLE PREDICTIONS SUMMARY")
    print("=" * 70)
    print(f"\nTotal predictions: {len(ensemble_predictions)}")
    print("\nForecast statistics:")
    print(ensemble_predictions[['forecast_q50', 'ensemble_forecast']].describe())
    print("\nSample predictions:")
    print(ensemble_predictions[['product_id', 'store_id', 'hour_timestamp',
                                'forecast_q50', 'ensemble_forecast']].head(10))
