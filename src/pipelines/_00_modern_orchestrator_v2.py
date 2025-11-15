#!/usr/bin/env python3
"""
Modern Pipeline Orchestrator v2 (Enhanced with GX)
===================================================

Prefect-based orchestration with full data quality monitoring.

Enhancements:
- Great Expectations validation at each stage
- Automatic quality report generation
- Pipeline blocking on critical failures
- Enhanced alerting and monitoring
- Performance tracking per stage

Usage:
    python -m src.pipelines._00_modern_orchestrator_v2 --full-data
    
Author: SmartGrocy Team
Date: 2025-11-15
"""

import logging
import time
import sys
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd
from datetime import datetime, timedelta

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from prefect import flow, task
    from prefect.logging import get_run_logger
    PREFECT_AVAILABLE = True
except ImportError:
    PREFECT_AVAILABLE = False
    def flow(name=None, **kwargs):
        def decorator(func):
            return func
        return decorator
    def task(**kwargs):
        def decorator(func):
            return func
        return decorator
    def get_run_logger():
        return logging.getLogger(__name__)

from src.config import (
    setup_project_path, setup_logging, ensure_directories,
    OUTPUT_FILES, get_dataset_config, DATA_QUALITY_CONFIG
)

# Import pipeline modules
from src.pipelines._01_load_data import load_data
from src.features import ws0_aggregation as ws0

# Import utilities
from src.utils.data_quality_gx import DataQualityValidator, validate_dataframe
from src.utils.alerting import alert_manager

try:
    from src.utils.performance_monitor import performance_monitor
    from src.utils.data_lineage import lineage_tracker, DataArtifact
except ImportError:
    performance_monitor = None
    lineage_tracker = None
    DataArtifact = None

setup_project_path()
setup_logging()
ensure_directories()

logger = logging.getLogger(__name__)

# Initialize GX validator
gx_validator = DataQualityValidator()


@task(retries=3, retry_delay_seconds=60, name="load_and_validate_data")
def load_and_validate_data_task(dataset_config: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
    """
    Load data with automatic GX validation.
    
    Args:
        dataset_config: Dataset configuration
    
    Returns:
        Dictionary of loaded dataframes
    
    Raises:
        ValueError: If data quality check fails critically
    """
    logger = get_run_logger()
    logger.info("⚙️ Stage 1: Load & Validate Data")
    logger.info("=" * 70)
    
    start_time = time.time()
    
    try:
        # Load data
        dataframes, config = load_data()
        
        logger.info(f"✓ Loaded {len(dataframes)} dataset(s)")
        
        # Validate each dataframe
        for name, df in dataframes.items():
            # Skip None dataframes (optional data that wasn't found)
            if df is None:
                logger.info(f"\nSkipping {name}: not available (optional data)")
                continue
                
            logger.info(f"\nValidating {name}: {df.shape}")
            
            # Run GX validation
            validation_result = validate_dataframe(
                df,
                asset_name=name,
                fail_on_error=False  # Don't fail on raw data issues
            )
            
            if validation_result['success']:
                logger.info(f"  ✅ {name} validation passed")
            else:
                stats = validation_result.get('statistics', {})
                success_rate = stats.get('success_percent', 0)
                logger.warning(f"  ⚠️ {name} validation: {success_rate:.1f}% passed")
                
                # Alert if quality is very poor
                if success_rate < DATA_QUALITY_CONFIG['quality_thresholds']['poor']:
                    alert_manager.alert_data_quality_issue(
                        data_asset=name,
                        quality_score=success_rate,
                        threshold=DATA_QUALITY_CONFIG['quality_thresholds']['poor']
                    )
        
        elapsed = time.time() - start_time
        logger.info(f"\n✓ Stage 1 complete in {elapsed:.1f}s")
        
        return dataframes
        
    except Exception as e:
        logger.error(f"❌ Stage 1 failed: {e}")
        if alert_manager:
            alert_manager.alert_pipeline_failure("load_data", str(e))
        raise


@task(retries=2, name="create_master_dataframe")
def create_master_dataframe_task(
    dataframes: Dict[str, pd.DataFrame],
    dataset_config: Dict[str, Any]
) -> pd.DataFrame:
    """
    Create master dataframe with validation.
    
    Args:
        dataframes: Loaded dataframes
        dataset_config: Dataset configuration
    
    Returns:
        Master dataframe
    """
    logger = get_run_logger()
    logger.info("\n⚙️ Stage 2: Create Master Dataframe")
    logger.info("=" * 70)
    
    start_time = time.time()
    
    try:
        # Get training data - use 'sales' key (standardized across datasets)
        if 'sales' in dataframes and dataframes['sales'] is not None:
            df = dataframes['sales']
        elif 'freshretail_train' in dataframes:
            df = dataframes['freshretail_train']
        else:
            raise ValueError("No training data found. Expected 'sales' or 'freshretail_train' key.")
        
        logger.info(f"Processing {len(df):,} rows...")
        
        # Create master dataframe
        master_df = ws0.prepare_master_dataframe(df)
        
        logger.info(f"✓ Master dataframe: {master_df.shape}")
        
        elapsed = time.time() - start_time
        logger.info(f"✓ Stage 2 complete in {elapsed:.1f}s")
        
        return master_df
        
    except Exception as e:
        logger.error(f"❌ Stage 2 failed: {e}")
        raise


@task(retries=2, retry_delay_seconds=30, name="enrich_features")
def enrich_features_task(
    master_df: pd.DataFrame,
    dataset_config: Dict[str, Any]
) -> pd.DataFrame:
    """
    Enrich features with GX validation.
    
    Args:
        master_df: Master dataframe
        dataset_config: Dataset configuration
    
    Returns:
        Feature-enriched dataframe
    """
    logger = get_run_logger()
    logger.info("\n⚙️ Stage 3: Feature Enrichment")
    logger.info("=" * 70)
    
    start_time = time.time()
    enriched_df = master_df.copy()
    
    # Get dataframes for feature enrichment (needed for some workstreams)
    from src.pipelines._01_load_data import load_data
    dataframes, _ = load_data()
    
    try:
        # WS1: Relational Features
        if dataset_config.get('has_relational', True):
            ws_start = time.time()
            logger.info("  Processing WS1-Relational...")
            try:
                from src.features import ws1_relational_features as ws1
                enriched_df = ws1.enrich_relational_features(enriched_df, dataframes)
                ws_elapsed = time.time() - ws_start
                logger.info(f"    ✅ WS1-Relational complete in {ws_elapsed:.1f}s")
            except Exception as e:
                logger.warning(f"    ⚠️ WS1-Relational failed: {e}")
        
        # WS5: Stockout Recovery (run before WS2)
        if dataset_config.get('has_stockout', True):
            ws_start = time.time()
            logger.info("  Processing WS5-Stockout...")
            try:
                from src.features import ws5_stockout_recovery as ws5
                enriched_df = ws5.recover_latent_demand(enriched_df, dataset_config)
                enriched_df = ws5.add_stockout_features(enriched_df, dataset_config)
                ws_elapsed = time.time() - ws_start
                logger.info(f"    ✅ WS5-Stockout complete in {ws_elapsed:.1f}s")
            except Exception as e:
                logger.warning(f"    ⚠️ WS5-Stockout failed: {e}")
        
        # WS6: Weather Features
        if dataset_config.get('has_weather', True) and 'weather' in dataframes and dataframes['weather'] is not None:
            ws_start = time.time()
            logger.info("  Processing WS6-Weather...")
            try:
                from src.features import ws6_weather_features as ws6
                enriched_df = ws6.merge_weather_data(enriched_df, dataframes['weather'])
                enriched_df = ws6.create_weather_features(enriched_df)
                ws_elapsed = time.time() - ws_start
                logger.info(f"    ✅ WS6-Weather complete in {ws_elapsed:.1f}s")
            except Exception as e:
                logger.warning(f"    ⚠️ WS6-Weather failed: {e}")
        
        # WS2: Time-Series Features (always run)
        ws_start = time.time()
        logger.info("  Processing WS2-TimeSeries...")
        try:
            from src.features import ws2_timeseries_features as ws2
            enriched_df = ws2.add_timeseries_features_config(enriched_df, dataset_config)
            ws_elapsed = time.time() - ws_start
            logger.info(f"    ✅ WS2-TimeSeries complete in {ws_elapsed:.1f}s")
        except Exception as e:
            logger.error(f"    ❌ WS2-TimeSeries failed: {e}")
            raise  # WS2 is critical
        
        # WS3: Behavior Features
        if dataset_config.get('has_behavior', False):
            ws_start = time.time()
            logger.info("  Processing WS3-Behavior...")
            try:
                from src.features import ws3_behavior_features as ws3
                enriched_df = ws3.add_behavioral_features(enriched_df, dataframes)
                ws_elapsed = time.time() - ws_start
                logger.info(f"    ✅ WS3-Behavior complete in {ws_elapsed:.1f}s")
            except Exception as e:
                logger.warning(f"    ⚠️ WS3-Behavior failed: {e}")
        
        # WS4: Price/Promo Features
        if dataset_config.get('has_price_promo', False):
            ws_start = time.time()
            logger.info("  Processing WS4-Price...")
            try:
                from src.features import ws4_price_features as ws4
                enriched_df = ws4.add_price_promotion_features(enriched_df, dataframes)
                ws_elapsed = time.time() - ws_start
                logger.info(f"    ✅ WS4-Price complete in {ws_elapsed:.1f}s")
            except Exception as e:
                logger.warning(f"    ⚠️ WS4-Price failed: {e}")
        
        # CRITICAL: Validate enriched features with GX
        logger.info("\n" + "=" * 70)
        logger.info("🔍 CRITICAL VALIDATION: Enriched Features")
        logger.info("=" * 70)
        
        validation_result = gx_validator.validate(
            enriched_df,
            suite_name="master_feature_table_suite",
            asset_name="enriched_features",
            fail_on_error=DATA_QUALITY_CONFIG.get('fail_pipeline_on_quality_issues', False),
            return_detailed=True
        )
        
        # Check quality score
        quality_score = validation_result['statistics'].get('success_percent', 0)
        logger.info(f"\nQuality Score: {quality_score:.1f}/100")
        
        # Determine quality level
        thresholds = DATA_QUALITY_CONFIG['quality_thresholds']
        if quality_score >= thresholds['excellent']:
            logger.info("✅ Quality: EXCELLENT")
        elif quality_score >= thresholds['good']:
            logger.info("✅ Quality: GOOD")
        elif quality_score >= thresholds['fair']:
            logger.warning("⚠️ Quality: FAIR")
        else:
            logger.error("❌ Quality: POOR")
            # Alert
            if alert_manager and DATA_QUALITY_CONFIG['alerting']['alert_on_quality_below'] > quality_score:
                failed_expectations = validation_result.get('failed_expectations', [])
                issues = [f.get('expectation', 'Unknown issue') for f in failed_expectations[:5]]
                alert_manager.alert_data_quality_issue(
                    dataset_name="enriched_features",
                    quality_score=quality_score,
                    issues=issues
                )
        
        # Show failed expectations (if any)
        failed = validation_result.get('failed_expectations', [])
        if failed:
            logger.warning(f"\n⚠️ {len(failed)} expectations failed:")
            for i, fail in enumerate(failed[:5], 1):  # Show first 5
                logger.warning(f"  {i}. {fail['expectation']} on {fail.get('column', 'N/A')}")
            if len(failed) > 5:
                logger.warning(f"  ... and {len(failed) - 5} more")
        
        # Save enriched dataframe
        output_path = OUTPUT_FILES['master_feature_table']
        enriched_df.to_parquet(output_path)
        logger.info(f"\n✓ Saved to: {output_path}")
        
        elapsed = time.time() - start_time
        logger.info(f"✓ Stage 3 complete in {elapsed:.1f}s")
        
        return enriched_df
        
    except Exception as e:
        logger.error(f"❌ Stage 3 failed: {e}")
        if alert_manager:
            alert_manager.alert_pipeline_failure("feature_enrichment", str(e))
        raise


@task(retries=1, name="train_models")
def train_models_task(
    enriched_df: pd.DataFrame,
    dataset_config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Train models with validation.
    
    Args:
        enriched_df: Enriched dataframe
        dataset_config: Dataset configuration
    
    Returns:
        Training results
    """
    logger = get_run_logger()
    logger.info("\n⚙️ Stage 4: Model Training")
    logger.info("=" * 70)
    
    start_time = time.time()
    
    try:
        # Import training module
        import subprocess
        import json
        
        # Run training script
        logger.info("Starting model training...")
        result = subprocess.run(
            [sys.executable, "src/pipelines/_03_model_training.py"],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT
        )
        
        if result.returncode != 0:
            logger.error(f"Training failed with code {result.returncode}")
            logger.error(f"stderr: {result.stderr}")
            raise RuntimeError(f"Model training failed: {result.stderr}")
        
        logger.info("✓ Training completed")
        
        # Load metrics
        metrics_path = OUTPUT_FILES['model_metrics']
        if metrics_path.exists():
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
            
            # Log metrics
            logger.info("\nModel Metrics:")
            for model_type, model_metrics in metrics.items():
                logger.info(f"  {model_type.upper()}:")
                for metric_name, value in list(model_metrics.items())[:5]:  # First 5 metrics
                    if isinstance(value, (int, float)):
                        logger.info(f"    {metric_name}: {value:.4f}")
        
        elapsed = time.time() - start_time
        logger.info(f"\n✓ Stage 4 complete in {elapsed:.1f}s")
        
        return {'success': True, 'metrics': metrics if metrics_path.exists() else {}}
        
    except Exception as e:
        logger.error(f"❌ Stage 4 failed: {e}")
        if alert_manager:
            alert_manager.alert_pipeline_failure("model_training", str(e))
        raise


@task(name="generate_quality_report")
def generate_quality_report_task():
    """
    Generate comprehensive data quality report.
    """
    logger = get_run_logger()
    logger.info("\n⚙️ Stage 5: Quality Report Generation")
    logger.info("=" * 70)
    
    try:
        # Check GX data docs
        gx_root = PROJECT_ROOT / "great_expectations"
        data_docs_path = gx_root / "uncommitted" / "data_docs" / "local_site" / "index.html"
        
        if data_docs_path.exists():
            logger.info(f"✓ GX Data Docs available: {data_docs_path}")
            logger.info("  Open in browser to view validation results")
        else:
            logger.warning("⚠️ GX Data Docs not generated")
            logger.warning("  Run: python scripts/setup_great_expectations.py")
        
        # Generate summary report
        report_path = OUTPUT_FILES['reports_dir'] / 'quality_summary.txt'
        with open(report_path, 'w') as f:
            f.write(f"SmartGrocy Data Quality Report\n")
            f.write(f"{'='*70}\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"GX Data Docs: {data_docs_path}\n")
            f.write(f"\nValidation Status: See GX reports for details\n")
        
        logger.info(f"✓ Summary report: {report_path}")
        logger.info("✓ Stage 5 complete")
        
    except Exception as e:
        logger.warning(f"⚠️ Quality report generation failed: {e}")
        # Don't fail pipeline for reporting issues


@flow(name="SmartGrocy-Pipeline-v2", log_prints=True)
def modern_pipeline_flow_v2(full_data: bool = False):
    """
    Modern pipeline with full GX integration.
    
    Args:
        full_data: Use full dataset (vs. sample)
    
    Returns:
        Pipeline execution summary
    """
    logger = get_run_logger() if PREFECT_AVAILABLE else logging.getLogger(__name__)
    
    logger.info("="*70)
    logger.info("🚀 SMARTGROCY MODERN PIPELINE V2")
    logger.info("="*70)
    logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Full Data Mode: {full_data}")
    logger.info(f"GX Validation: {gx_validator.is_available()}")
    logger.info("="*70)
    
    pipeline_start = time.time()
    
    # Start performance monitoring
    if performance_monitor:
        performance_monitor.start_monitoring()
    
    try:
        # Get config
        dataset_config = get_dataset_config()
        logger.info(f"\nDataset: {dataset_config['name']}")
        
        # Stage 1: Load & validate
        dataframes = load_and_validate_data_task(dataset_config)
        
        # Stage 2: Create master dataframe
        master_df = create_master_dataframe_task(dataframes, dataset_config)
        
        # Stage 3: Enrich features (with GX validation)
        enriched_df = enrich_features_task(master_df, dataset_config)
        
        # Stage 4: Train models
        training_results = train_models_task(enriched_df, dataset_config)
        
        # Stage 5: Generate quality report
        generate_quality_report_task()
        
        # Pipeline summary
        pipeline_elapsed = time.time() - pipeline_start
        
        logger.info("\n" + "="*70)
        logger.info("✅ PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("="*70)
        logger.info(f"Total Duration: {pipeline_elapsed:.1f}s ({pipeline_elapsed/60:.1f} min)")
        logger.info(f"Final Data Shape: {enriched_df.shape}")
        logger.info(f"Models Saved: {OUTPUT_FILES['models_dir']}")
        logger.info(f"Reports: {OUTPUT_FILES['reports_dir']}")
        logger.info("="*70)
        
        # Stop performance monitoring
        if performance_monitor:
            summary = performance_monitor.stop_monitoring()
            logger.info(f"\n📊 Performance Summary:")
            logger.info(f"  Peak Memory: {summary.get('peak_memory_mb', 0):.1f} MB")
            logger.info(f"  Avg CPU: {summary.get('avg_cpu_percent', 0):.1f}%")
        
        return {
            'success': True,
            'duration_seconds': pipeline_elapsed,
            'final_shape': enriched_df.shape,
            'metrics': training_results.get('metrics', {})
        }
        
    except Exception as e:
        logger.error("\n" + "="*70)
        logger.error("❌ PIPELINE FAILED")
        logger.error("="*70)
        logger.error(f"Error: {e}")
        logger.error(f"Duration before failure: {time.time() - pipeline_start:.1f}s")
        logger.error("="*70)
        
        # Alert
        if alert_manager:
            alert_manager.alert_pipeline_failure("pipeline", str(e))
        
        raise


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run modern pipeline v2 with GX')
    parser.add_argument('--full-data', action='store_true',
                       help='Use full dataset')
    args = parser.parse_args()
    
    # Run pipeline
    result = modern_pipeline_flow_v2(full_data=args.full_data)
    
    # Exit with appropriate code
    sys.exit(0 if result['success'] else 1)
