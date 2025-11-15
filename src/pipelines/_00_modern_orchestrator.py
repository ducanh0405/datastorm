"""
Modern Pipeline Orchestrator (Prefect-based)
============================================
Replaces subprocess-based orchestration with modern DAG-based workflow management.
Provides caching, error handling, monitoring, and data quality checks.
"""
import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional, List
import pandas as pd
from datetime import datetime, timedelta

try:
    from prefect import flow, task  # pyright: ignore[reportMissingImports]
    from prefect.logging import get_run_logger  # pyright: ignore[reportMissingImports]
    from prefect.states import Failed, Completed  # pyright: ignore[reportMissingImports]
    PREFECT_AVAILABLE = True
    # Try to import cache-related features (may vary by Prefect version)
    try:
        from prefect.tasks import task_input_hash  # pyright: ignore[reportMissingImports]
        from prefect.cache_policies import INPUTS  # pyright: ignore[reportMissingImports]
    except ImportError:
        # Fallback for newer Prefect versions
        task_input_hash = None
        INPUTS = None
except ImportError:
    PREFECT_AVAILABLE = False
    import logging
    _temp_logger = logging.getLogger(__name__)
    _temp_logger.warning("Prefect not installed. Pipeline will run without orchestration features.")
    _temp_logger.warning("Install with: pip install prefect")
    # Create dummy decorators for graceful degradation
    def flow(name=None):
        def decorator(func):
            return func
        return decorator
    def task(retries=0, retry_delay_seconds=0):
        def decorator(func):
            return func
        return decorator
    def get_run_logger():
        return logging.getLogger(__name__)

from src.config import (
    setup_project_path, setup_logging, ensure_directories,
    OUTPUT_FILES, get_dataset_config, PROJECT_ROOT
)
from src.pipelines._01_load_data import load_data
from src.features import ws0_aggregation as ws0
from src.utils.validation import comprehensive_validation
from src.utils.data_quality import DataQualityMonitor
from src.utils.alerting import alert_manager
from src.utils.caching import pipeline_cache
# Import monitoring & lineage tracking for pipeline observability
from src.utils.performance_monitor import performance_monitor
from src.utils.data_lineage import lineage_tracker, DataArtifact, PipelineStep

setup_project_path()
setup_logging()
ensure_directories()

logger = logging.getLogger(__name__)

# Initialize data quality monitor
quality_monitor = DataQualityMonitor()


@task(retries=3, retry_delay_seconds=60)
def load_and_validate_data(dataset_config: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
    """
    Load data with caching and validation.

    Args:
        dataset_config: Dataset configuration

    Returns:
        Dictionary of loaded dataframes
    """
    logger = get_run_logger()
    logger.info("🔄 Loading and validating data...")

    try:
        # Load data
        dataframes, config = load_data()

        # Validate each dataframe
        for name, df in dataframes.items():
            logger.info(f"Validating {name}: {df.shape}")

            # Run comprehensive validation
            validation_results = comprehensive_validation(df, verbose=False)

            # Log quality score
            quality_score = validation_results.get('quality_score', 0)
            if quality_score >= 90:
                logger.info(f"✅ {name}: EXCELLENT ({quality_score}/100)")
            elif quality_score >= 75:
                logger.info(f"⚠️ {name}: GOOD ({quality_score}/100)")
            else:
                logger.warning(f"❌ {name}: POOR ({quality_score}/100)")
                if validation_results.get('issues'):
                    for issue in validation_results['issues']:
                        logger.warning(f"   - {issue}")

            # Store validation results for monitoring
            quality_monitor.store_validation_results(name, validation_results)

        logger.info(f"✅ Data loading complete: {len(dataframes)} datasets")
        return dataframes

    except Exception as e:
        logger.error(f"❌ Data loading failed: {e}")
        raise


@task(retries=2)
def create_master_dataframe(dataframes: Dict[str, pd.DataFrame],
                          dataset_config: Dict[str, Any]) -> pd.DataFrame:
    """
    Create master feature table with caching.

    Args:
        dataframes: Loaded dataframes
        dataset_config: Dataset configuration

    Returns:
        Master dataframe
    """
    logger = get_run_logger()
    logger.info("🔄 Creating master dataframe...")

    start_time = time.time()

    try:
        if 'freshretail_train' in dataframes:
            df = dataframes['freshretail_train']
            logger.info(f"Processing {len(df)} rows...")

            # Create master dataframe
            master_df = ws0.prepare_master_dataframe(df)

            # Validate master dataframe
            validation_results = comprehensive_validation(master_df, verbose=False)
            quality_score = validation_results.get('quality_score', 0)

            # Store quality metrics
            quality_monitor.store_validation_results('master_dataframe', validation_results)

            processing_time = time.time() - start_time
            logger.info(f"✅ Master dataframe created: {master_df.shape} in {processing_time:.1f}s")
            logger.info(f"Quality Score: {quality_score}/100")

            return master_df
        else:
            raise ValueError("No suitable training data found")

    except Exception as e:
        logger.error(f"❌ Master dataframe creation failed: {e}")
        raise


@task(retries=2, retry_delay_seconds=30)
def enrich_features(master_df: pd.DataFrame,
                   dataset_config: Dict[str, Any]) -> pd.DataFrame:
    """
    Enrich features from all workstreams with caching and monitoring.

    Args:
        master_df: Master dataframe
        dataset_config: Dataset configuration

    Returns:
        Feature-enriched dataframe
    """
    logger = get_run_logger()
    logger.info("🔄 Enriching features from all workstreams...")

    start_time = time.time()
    enriched_df = master_df.copy()

    try:
        # Import feature modules dynamically
        feature_modules = []

        # WS1: Relational features
        if dataset_config.get('has_relational', True):
            from src.features import ws1_relational_features as ws1
            feature_modules.append(('WS1', ws1))

        # WS2: Time-series features
        from src.features import ws2_timeseries_features as ws2
        feature_modules.append(('WS2', ws2))

        # WS3: Behavior features
        if dataset_config.get('has_behavior', False):
            from src.features import ws3_behavior_features as ws3
            feature_modules.append(('WS3', ws3))

        # WS4: Price features
        if dataset_config.get('has_price_promo', False):
            from src.features import ws4_price_features as ws4
            feature_modules.append(('WS4', ws4))

        # WS5: Stockout features
        if dataset_config.get('has_stockout', True):
            from src.features import ws5_stockout_recovery as ws5
            feature_modules.append(('WS5', ws5))

        # WS6: Weather features
        if dataset_config.get('has_weather', True):
            from src.features import ws6_weather_features as ws6
            feature_modules.append(('WS6', ws6))

        # Apply each feature module
        failed_modules = []
        for ws_name, module in feature_modules:
            logger.info(f"Processing {ws_name} features...")
            try:
                enriched_df = module.main(enriched_df, dataset_config)
                logger.info(f"✅ {ws_name} features added")
            except Exception as e:
                logger.warning(f"⚠️ {ws_name} failed: {e}")
                failed_modules.append(ws_name)
                # Continue with other modules

        # Alert for failed feature modules
        if failed_modules and alert_manager:
            alert_manager.alert_pipeline_failure(
                pipeline_stage=f"feature_enrichment_{'_'.join(failed_modules)}",
                error_message=f"Feature modules failed: {', '.join(failed_modules)}"
            )

        # Final validation
        validation_results = comprehensive_validation(enriched_df, verbose=False)
        quality_score = validation_results.get('quality_score', 0)

        # Store quality metrics
        quality_monitor.store_validation_results('enriched_features', validation_results)

        processing_time = time.time() - start_time
        logger.info(f"✅ Feature enrichment complete: {enriched_df.shape} in {processing_time:.1f}s")
        logger.info(f"Quality Score: {quality_score}/100")

        # Performance monitoring
        if alert_manager and processing_time > 3600:  # More than 1 hour
            alert_manager.alert_performance_issue(
                component="feature_enrichment",
                metric_name="processing_time_seconds",
                current_value=processing_time,
                threshold=3600
            )

        # Save enriched dataframe
        output_path = OUTPUT_FILES['master_feature_table']
        enriched_df.to_parquet(output_path)
        logger.info(f"Saved to: {output_path}")

        return enriched_df

    except Exception as e:
        logger.error(f"❌ Feature enrichment failed: {e}")
        if alert_manager:
            alert_manager.alert_pipeline_failure(
                pipeline_stage="feature_enrichment",
                error_message=str(e)
            )
        raise


@task(retries=1)
def train_models(enriched_df: pd.DataFrame,
                dataset_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Train models with monitoring.

    Args:
        enriched_df: Feature-enriched dataframe
        dataset_config: Dataset configuration

    Returns:
        Training results and metrics
    """
    logger = get_run_logger()
    logger.info("🔄 Training models...")

    try:
        from src.pipelines._03_model_training import main as train_main

        # Train models
        results = train_main(enriched_df, dataset_config)

        logger.info("✅ Model training complete")
        logger.info(f"Models saved to: {OUTPUT_FILES['models_dir']}")
        logger.info(f"Metrics saved to: {OUTPUT_FILES['model_metrics']}")

        return results

    except Exception as e:
        logger.error(f"❌ Model training failed: {e}")
        raise


@task
def generate_quality_report():
    """
    Generate comprehensive data quality report.
    """
    logger = get_run_logger()
    logger.info("🔄 Generating quality report...")

    try:
        # Generate quality dashboard
        quality_monitor.generate_quality_dashboard()

        # Check for data drift
        drift_report = quality_monitor.check_data_drift()

        if drift_report:
            logger.warning("⚠️ Data drift detected:")
            for issue in drift_report:
                logger.warning(f"   - {issue}")
        else:
            logger.info("✅ No significant data drift detected")

        logger.info("✅ Quality report generated")

    except Exception as e:
        logger.error(f"❌ Quality report generation failed: {e}")
        raise


@flow(name="SmartGrocy Pipeline")
def modern_pipeline_flow(full_data: bool = False):
    """
    Modern pipeline orchestration using Prefect.

    Args:
        full_data: Whether to use full dataset
    """
    logger = get_run_logger()

    # Start performance monitoring at pipeline start
    performance_monitor.start_monitoring()
    logger.info("✅ Performance monitoring started")

    # Get configuration
    dataset_config = get_dataset_config()
    logger.info(f"🚀 Starting pipeline for dataset: {dataset_config['name']}")
    logger.info(f"Full data mode: {full_data}")

    # Step 1: Load and validate data
    dataframes = load_and_validate_data(dataset_config)
    
    # Track data loading in lineage for observability
    for name, df in dataframes.items():
        artifact = DataArtifact(
            name=name,
            artifact_type='raw_data',
            shape=df.shape,
            created_at=datetime.now().isoformat()
        )
        lineage_tracker.register_artifact(artifact)

    # Step 2: Create master dataframe
    master_df = create_master_dataframe(dataframes, dataset_config)

    # Step 3: Enrich features
    enriched_df = enrich_features(master_df, dataset_config)

    # Step 4: Train models
    training_results = train_models(enriched_df, dataset_config)

    # Step 5: Generate quality report
    generate_quality_report()

    # Stop performance monitoring and save results
    monitoring_summary = performance_monitor.stop_monitoring()
    logger.info("✅ Performance monitoring stopped")
    logger.info(f"📊 Session summary: {monitoring_summary.get('session_duration', 0):.2f}s total")
    
    # Save lineage data
    lineage_tracker.save_lineage()
    logger.info("✅ Data lineage saved")

    logger.info("🎉 Pipeline execution complete!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Run modern pipeline with Prefect')
    parser.add_argument('--full-data', action='store_true',
                       help='Use full dataset')
    args = parser.parse_args()

    # Run the flow
    modern_pipeline_flow(full_data=args.full_data)
