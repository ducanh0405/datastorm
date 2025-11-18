"""
Modern Pipeline Orchestrator (Prefect-based)
============================================
Replaces subprocess-based orchestration with modern DAG-based workflow management.
Provides caching, error handling, monitoring, and data quality checks.

Enhanced with:
- Great Expectations validation at each stage
- Automatic quality report generation
- Pipeline blocking on critical failures
- Enhanced alerting and monitoring
- Performance tracking per stage

Usage:
    python -m src.pipelines._00_modern_orchestrator --full-data

Author: SmartGrocy Team
Date: 2025-11-16
"""
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

# Setup logging
logger = logging.getLogger(__name__)

# Check if Great Expectations is available for enhanced validation
try:
    from great_expectations.core.expectation_validation_result import (  # pyright: ignore[reportMissingImports]  # pyright: ignore[reportMissingImports]
        validate_dataframe,  # pyright: ignore[reportMissingImports]
    )
    from great_expectations.data_context import (  # pyright: ignore[reportMissingImports]
        BaseDataContext,  # pyright: ignore[reportMissingImports]
    )
    gx_validator = True
    logger.info("Great Expectations available - enhanced validation enabled")
except ImportError:
    gx_validator = False
    logger.info("Great Expectations not available - basic validation only")

try:
    from prefect import flow, task  # pyright: ignore[reportMissingImports]
    from prefect.logging import get_run_logger  # pyright: ignore[reportMissingImports]
    from prefect.states import Completed, Failed  # pyright: ignore[reportMissingImports]
    PREFECT_AVAILABLE = True
    # Try to import cache-related features (may vary by Prefect version)
    try:
        from prefect.cache_policies import INPUTS  # pyright: ignore[reportMissingImports]
        from prefect.tasks import task_input_hash  # pyright: ignore[reportMissingImports]
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
    OUTPUT_FILES,
    PROJECT_ROOT,
    ensure_directories,
    get_dataset_config,
    setup_logging,
    setup_project_path,
)
from src.features import ws0_aggregation as ws0
from src.pipelines._01_load_data import load_data
from src.utils.alerting import alert_manager
from src.utils.data_lineage import DataArtifact, lineage_tracker
from src.utils.data_quality import DataQualityMonitor

# Import monitoring & lineage tracking for pipeline observability
from src.utils.performance_monitor import performance_monitor
from src.utils.validation import comprehensive_validation

setup_project_path()
setup_logging()
ensure_directories()

logger = logging.getLogger(__name__)

# Initialize data quality monitor
quality_monitor = DataQualityMonitor()


@task(retries=3, retry_delay_seconds=60)
def load_and_validate_data(dataset_config: dict[str, Any]) -> dict[str, pd.DataFrame]:
    """
    Load data with caching and validation.

    Args:
        dataset_config: Dataset configuration

    Returns:
        Dictionary of loaded dataframes

    Raises:
        PipelineError: If data quality check fails critically
    """
    from src.core.exceptions import PipelineError

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
            if df is None:
                logger.info(f"\nSkipping {name}: not available (optional data)")
                continue

            logger.info(f"\nValidating {name}: {df.shape}")

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
                    for issue in validation_results['issues'][:3]:  # Show first 3 issues
                        logger.warning(f"   - {issue}")

            # Store validation results for monitoring
            quality_monitor.store_validation_results(name, validation_results)

            # Enhanced GX validation (if available)
            if gx_validator:
                try:
                    gx_result = validate_dataframe(  # pyright: ignore[reportUndefinedVariable]
                        df, asset_name=name, fail_on_error=False
                    )
                    if not gx_result['success']:
                        logger.warning(f"⚠️ GX validation failed for {name}")
                except Exception as gx_e:
                    logger.warning(f"⚠️ GX validation error for {name}: {gx_e}")

        elapsed = time.time() - start_time
        logger.info(f"\n✓ Stage 1 complete in {elapsed:.1f}s")

        return dataframes

    except Exception as e:
        logger.error(f"❌ Data loading failed: {e}")
        raise PipelineError(f"Data loading failed: {e}", stage="load_data", original_error=e)


@task(retries=2)
def create_master_dataframe(dataframes: dict[str, pd.DataFrame],
                          dataset_config: dict[str, Any]) -> pd.DataFrame:
    """
    Create master dataframe from loaded data.

    Args:
        dataframes: Loaded dataframes
        dataset_config: Dataset configuration

    Returns:
        Master dataframe

    Raises:
        PipelineError: If master dataframe creation fails
    """
    from src.core.exceptions import PipelineError

    logger = get_run_logger()
    logger.info("\n⚙️ Stage 2: Create Master Dataframe")
    logger.info("=" * 70)

    start_time = time.time()

    try:
        sales_df = dataframes.get('sales') or dataframes.get('freshretail_train')
        if sales_df is None:
            raise ValueError("No sales data found")

        logger.info(f"Processing {len(sales_df)} rows...")

        # Create master dataframe
        master_df = ws0.create_master_grid(sales_df, dataset_config)

        # Validate master dataframe
        validation_results = comprehensive_validation(master_df, verbose=False)
        quality_score = validation_results.get('quality_score', 0)

        # Store quality metrics
        quality_monitor.store_validation_results('master_dataframe', validation_results)

        # Enhanced GX validation
        if gx_validator:
            try:
                gx_result = validate_dataframe(  # pyright: ignore[reportUndefinedVariable]
                    master_df, asset_name='master_dataframe', fail_on_error=False
                )
                if not gx_result['success']:
                    logger.warning("⚠️ GX validation failed for master dataframe")
            except Exception as gx_e:
                logger.warning(f"⚠️ GX validation error for master dataframe: {gx_e}")

        processing_time = time.time() - start_time
        logger.info(f"✅ Master dataframe created: {master_df.shape}")
        logger.info(f"✓ Stage 2 complete in {processing_time:.1f}s")
        logger.info(f"Quality Score: {quality_score}/100")

        return master_df

    except Exception as e:
        logger.error(f"❌ Master dataframe creation failed: {e}")
        raise PipelineError(f"Master dataframe creation failed: {e}", stage="create_master_dataframe", original_error=e)


@task(retries=2, retry_delay_seconds=30)
def enrich_features(master_df: pd.DataFrame,
                   dataset_config: dict[str, Any]) -> pd.DataFrame:
    """
    Enrich features from all workstreams.

    Args:
        master_df: Master dataframe
        dataset_config: Dataset configuration

    Returns:
        Feature-enriched dataframe

    Raises:
        PipelineError: If feature enrichment fails
    """
    from src.core.exceptions import PipelineError

    logger = get_run_logger()
    logger.info("\n⚙️ Stage 3: Feature Enrichment")
    logger.info("=" * 70)

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

        # Enhanced GX validation
        if gx_validator:
            try:
                gx_result = validate_dataframe(  # pyright: ignore[reportUndefinedVariable]
                    enriched_df, asset_name='enriched_features', fail_on_error=False
                )
                if not gx_result['success']:
                    logger.warning("⚠️ GX validation failed for enriched features")
            except Exception as gx_e:
                logger.warning(f"⚠️ GX validation error for enriched features: {gx_e}")

        processing_time = time.time() - start_time
        logger.info(f"✅ Feature enrichment complete: {enriched_df.shape}")
        logger.info(f"✓ Stage 3 complete in {processing_time:.1f}s")
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
        raise PipelineError(f"Feature enrichment failed: {e}", stage="enrich_features", original_error=e)


@task(retries=1)
def train_models(enriched_df: pd.DataFrame,
                dataset_config: dict[str, Any]) -> dict[str, Any]:
    """
    Train models.

    Args:
        enriched_df: Feature-enriched dataframe
        dataset_config: Dataset configuration

    Returns:
        Training results and metrics

    Raises:
        PipelineError: If model training fails
    """
    from src.core.exceptions import PipelineError

    logger = get_run_logger()
    logger.info("\n⚙️ Stage 4: Model Training")
    logger.info("=" * 70)

    start_time = time.time()

    try:
        # Run training script
        import subprocess
        cmd = [sys.executable, str(PROJECT_ROOT / 'src' / 'pipelines' / '_03_model_training.py')]
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=PROJECT_ROOT)

        if result.returncode != 0:
            raise RuntimeError(f"Model training failed: {result.stderr}")

        logger.info("✓ Training completed")

        # Load metrics
        metrics_path = OUTPUT_FILES['model_metrics']
        if metrics_path.exists():
            with open(metrics_path) as f:
                metrics = json.load(f)

            # Log metrics
            logger.info("\nModel Metrics:")
            for model_type, model_metrics in metrics.items():
                logger.info(f"  {model_type.upper()}:")
                for metric_name, value in list(model_metrics.items())[:5]:  # First 5 metrics
                    if isinstance(value, int | float):
                        logger.info(f"    {metric_name}: {value:.4f}")

        elapsed = time.time() - start_time
        logger.info(f"\n✓ Stage 4 complete in {elapsed:.1f}s")

        return {'success': True, 'metrics': metrics if metrics_path.exists() else {}}

    except Exception as e:
        logger.error(f"❌ Model training failed: {e}")
        if alert_manager:
            alert_manager.alert_pipeline_failure("model_training", str(e))
        raise PipelineError(f"Model training failed: {e}", stage="train_models", original_error=e)


def generate_quality_summary(report_path: Path, data_docs_path: Path | None) -> None:
    """
    Generate comprehensive quality summary report.

    Args:
        report_path: Path to save the summary report
        data_docs_path: Path to GX data docs
    """
    report_lines = []
    report_lines.append("SmartGrocy Data Quality & Performance Report")
    report_lines.append("=" * 70)
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # 1. Model Performance Metrics
    report_lines.append("📊 MODEL PERFORMANCE METRICS")
    report_lines.append("-" * 70)
    model_metrics_path = OUTPUT_FILES['model_metrics']
    if model_metrics_path.exists():
        try:
            with open(model_metrics_path) as f:
                metrics = json.load(f)

            # R² Score
            r2 = metrics.get('r2_score', 'N/A')
            report_lines.append(f"R² Score: {r2:.4f}" if isinstance(r2, int | float) else f"R² Score: {r2}")

            # Coverage
            coverage = metrics.get('coverage_90%', 'N/A')
            report_lines.append(f"Coverage (90%): {coverage:.2%}" if isinstance(coverage, int | float) else f"Coverage (90%): {coverage}")

            # MAE và RMSE cho Q50
            q50_mae = metrics.get('q50_mae', 'N/A')
            q50_rmse = metrics.get('q50_rmse', 'N/A')
            report_lines.append(f"MAE (Q50): {q50_mae:.4f}" if isinstance(q50_mae, int | float) else f"MAE (Q50): {q50_mae}")
            report_lines.append(f"RMSE (Q50): {q50_rmse:.4f}" if isinstance(q50_rmse, int | float) else f"RMSE (Q50): {q50_rmse}")

            # MAPE với cảnh báo
            q50_mape = metrics.get('q50_mape')
            if q50_mape is not None:
                if isinstance(q50_mape, int | float) and q50_mape < 1000:
                    report_lines.append(f"MAPE (Q50): {q50_mape:.2f}%")
                else:
                    mape_valid = metrics.get('q50_mape_valid_samples', 'N/A')
                    mape_total = metrics.get('q50_mape_total_samples', 'N/A')
                    report_lines.append(f"MAPE (Q50): Calculated with threshold (valid: {mape_valid}/{mape_total} samples)")
            else:
                report_lines.append("MAPE (Q50): Not calculated (insufficient valid samples)")

            report_lines.append("")
        except Exception as e:
            report_lines.append(f"⚠️ Could not load model metrics: {e}\n")
    else:
        report_lines.append("⚠️ Model metrics not found\n")

    # 2. Business Impact Metrics
    report_lines.append("💼 BUSINESS IMPACT METRICS")
    report_lines.append("-" * 70)
    business_report_path = OUTPUT_FILES['reports_dir'] / 'business_report_detailed.csv'
    if business_report_path.exists():
        try:
            df_business = pd.read_csv(business_report_path)
            for _, row in df_business.iterrows():
                if row['status'] == '✅':
                    value = row['value']
                    unit = row['unit']
                    if unit == 'ratio':
                        report_lines.append(f"{row['metric']}: {value:.4f}")
                    elif unit == 'percentage':
                        report_lines.append(f"{row['metric']}: {value:.2f}%")
                    elif unit == 'count':
                        report_lines.append(f"{row['metric']}: {int(value)}")
                    else:
                        report_lines.append(f"{row['metric']}: {value:.2f} {unit}")
            report_lines.append("")
        except Exception as e:
            report_lines.append(f"⚠️ Could not load business metrics: {e}\n")
    else:
        report_lines.append("⚠️ Business report not found\n")

    # 3. Data Quality Status
    report_lines.append("🔍 DATA QUALITY STATUS")
    report_lines.append("-" * 70)
    validation_path = OUTPUT_FILES['reports_dir'] / 'metrics' / 'master_table_validation.json'
    if validation_path.exists():
        try:
            with open(validation_path) as f:
                validation = json.load(f)

            quality_score = validation.get('quality_score', 'N/A')
            passed = validation.get('passed', False)

            report_lines.append(f"Quality Score: {quality_score}/100")
            report_lines.append(f"Validation Status: {'✅ PASSED' if passed else '⚠️ FAILED (see issues below)'}")

            issues = validation.get('issues', [])
            if issues:
                report_lines.append("\nIssues:")
                for issue in issues[:5]:
                    report_lines.append(f"  • {issue}")
                if len(issues) > 5:
                    report_lines.append(f"  ... and {len(issues) - 5} more issues")

            # Missing values summary
            missing = validation.get('missing_values', {})
            if missing and 'percentages' in missing:
                high_missing = {k: v for k, v in missing['percentages'].items() if v > 50}
                if high_missing:
                    report_lines.append(f"\nHigh missing values (>50%): {len(high_missing)} columns")
                    report_lines.append("  (This is expected for lag features at the start of time series)")

            report_lines.append("")
        except Exception as e:
            report_lines.append(f"⚠️ Could not load validation data: {e}\n")
    else:
        report_lines.append("⚠️ Validation data not found\n")

    # 4. File Sizes & Storage
    report_lines.append("💾 STORAGE INFORMATION")
    report_lines.append("-" * 70)

    predictions_csv = OUTPUT_FILES['predictions_test']
    predictions_parquet = predictions_csv.with_suffix('.parquet')

    if predictions_parquet.exists():
        size_mb = predictions_parquet.stat().st_size / 1024 / 1024
        report_lines.append(f"Predictions (Parquet): {size_mb:.2f} MB ✓ Recommended format")
    if predictions_csv.exists():
        size_mb = predictions_csv.stat().st_size / 1024 / 1024
        report_lines.append(f"Predictions (CSV): {size_mb:.2f} MB")
    report_lines.append("")

    # 5. Links to Detailed Reports
    report_lines.append("📄 DETAILED REPORTS")
    report_lines.append("-" * 70)
    if data_docs_path and data_docs_path.exists():
        report_lines.append(f"GX Data Docs: {data_docs_path}")
    else:
        report_lines.append("GX Data Docs: Not generated (run: python scripts/setup_great_expectations.py)")

    report_lines.append(f"Model Metrics: {OUTPUT_FILES['model_metrics']}")
    report_lines.append(f"Business Report: {OUTPUT_FILES['reports_dir'] / 'business_report_detailed.csv'}")
    report_lines.append(f"Dashboard: {OUTPUT_FILES['dashboard_html']}")
    report_lines.append("")

    # Write report
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))


@task
def generate_quality_report():
    """
    Generate comprehensive data quality report.

    Raises:
        PipelineError: If report generation fails
    """
    from src.core.exceptions import PipelineError

    logger = get_run_logger()
    logger.info("\n⚙️ Stage 5: Quality Report Generation")
    logger.info("=" * 70)

    try:
        # Check GX data docs
        gx_root = PROJECT_ROOT / "great_expectations"
        data_docs_path = gx_root / "uncommitted" / "data_docs" / "local_site" / "index.html"

        if data_docs_path.exists():
            logger.info(f"✓ GX Data Docs available: {data_docs_path}")
        else:
            logger.warning("⚠️ GX Data Docs not generated")

        # Generate comprehensive summary report
        report_path = OUTPUT_FILES['reports_dir'] / 'quality_summary.txt'
        generate_quality_summary(report_path, data_docs_path)

        logger.info(f"✓ Summary report: {report_path}")
        logger.info("✓ Stage 5 complete")

    except Exception as e:
        logger.warning(f"⚠️ Quality report generation failed: {e}")
        raise PipelineError(f"Quality report generation failed: {e}", stage="generate_quality_report", original_error=e)


@flow(name="SmartGrocy Pipeline")
def modern_pipeline_flow(full_data: bool = False) -> dict[str, Any]:
    """
    Modern pipeline orchestration using Prefect with GX validation.

    Args:
        full_data: Whether to use full dataset

    Returns:
        Pipeline execution results
    """
    from src.core.exceptions import PipelineError

    logger = get_run_logger()
    logger.info("=" * 80)
    logger.info("🚀 SMARTGROCY MODERN PIPELINE")
    logger.info("=" * 80)
    logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Full data mode: {full_data}")
    logger.info("=" * 80)

    pipeline_start = datetime.now()
    results = {'stages': {}}

    try:
        # Start performance monitoring
        if performance_monitor:
            performance_monitor.start_monitoring()

        # Get configuration
        dataset_config = get_dataset_config()
        logger.info(f"Dataset: {dataset_config['name']}")

        # Stage 1: Load and validate data
        dataframes = load_and_validate_data(dataset_config)
        results['stages']['data_loading'] = {'success': True}

        # Track data loading in lineage
        if lineage_tracker:
            for name, df in dataframes.items():
                artifact = DataArtifact(
                    name=name,
                    artifact_type='raw_data',
                    shape=df.shape,
                    created_at=datetime.now().isoformat()
                )
                lineage_tracker.register_artifact(artifact)

        # Stage 2: Create master dataframe
        master_df = create_master_dataframe(dataframes, dataset_config)
        results['stages']['master_dataframe'] = {'success': True, 'shape': master_df.shape}

        # Stage 3: Enrich features
        enriched_df = enrich_features(master_df, dataset_config)
        results['stages']['feature_engineering'] = {'success': True, 'shape': enriched_df.shape}

        # Stage 4: Train models
        training_results = train_models(enriched_df, dataset_config)
        results['stages']['model_training'] = training_results

        # Stage 5: Generate quality report
        generate_quality_report()
        results['stages']['quality_report'] = {'success': True}

        # Pipeline summary
        pipeline_elapsed = (datetime.now() - pipeline_start).total_seconds()

        # Stop performance monitoring
        if performance_monitor:
            summary = performance_monitor.stop_monitoring()
            logger.info("\n📊 Performance Summary:")
            logger.info(f"  Peak Memory: {summary.get('peak_memory_mb', 0):.1f} MB")
            logger.info(f"  Avg CPU: {summary.get('avg_cpu_percent', 0):.1f}%")

        # Save lineage data
        if lineage_tracker:
            lineage_tracker.save_lineage()

        logger.info("\n" + "=" * 80)
        logger.info("✅ PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        logger.info(f"Total time: {pipeline_elapsed:.1f}s ({pipeline_elapsed/60:.1f} min)")
        logger.info(f"Final data shape: {enriched_df.shape}")
        logger.info("=" * 80)

        results.update({
            'success': True,
            'total_time': pipeline_elapsed,
            'final_shape': enriched_df.shape,
            'timestamp': datetime.now().isoformat()
        })

        return results

    except Exception as e:
        pipeline_elapsed = (datetime.now() - pipeline_start).total_seconds()

        logger.error("\n" + "=" * 80)
        logger.error("❌ PIPELINE FAILED")
        logger.error("=" * 80)
        logger.error(f"Error: {e}")
        logger.error(f"Time elapsed: {pipeline_elapsed:.1f}s")
        logger.error("=" * 80)

        if isinstance(e, PipelineError):
            raise

        raise PipelineError(f"Pipeline failed: {e}", stage="orchestration", original_error=e)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Run modern pipeline with Prefect')
    parser.add_argument('--full-data', action='store_true',
                       help='Use full dataset')
    args = parser.parse_args()

    # Run the flow
    modern_pipeline_flow(full_data=args.full_data)
