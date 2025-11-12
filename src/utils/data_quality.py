"""
Data Quality Monitoring Module
==============================
Comprehensive data quality monitoring using Great Expectations and custom validations.
Provides statistical profiling, drift detection, and quality dashboards.
"""
import logging
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

try:
    import great_expectations as ge
    from great_expectations.core import ExpectationSuite
    from great_expectations.core.expectation_configuration import ExpectationConfiguration
    from great_expectations.validator.validator import Validator
    from evidently import ColumnMapping
    from evidently.report import Report
    from evidently.metrics import DataDriftTable, DatasetDriftMetric
except ImportError:
    print("Data quality libraries not installed. Install with: pip install great-expectations evidently")
    raise

from src.config import OUTPUT_FILES, PROJECT_ROOT

logger = logging.getLogger(__name__)


class DataQualityMonitor:
    """
    Comprehensive data quality monitoring system.
    """

    def __init__(self):
        self.quality_history = {}
        self.baseline_profiles = {}
        self.gx_context = None
        self._setup_great_expectations()

    def _setup_great_expectations(self):
        """Setup Great Expectations context."""
        try:
            gx_dir = PROJECT_ROOT / 'great_expectations'
            gx_dir.mkdir(exist_ok=True)

            # Initialize GX context if not exists
            if not (gx_dir / 'great_expectations.yml').exists():
                import subprocess
                subprocess.run([
                    'great_expectations', 'init', '--no-view'
                ], cwd=PROJECT_ROOT, capture_output=True)

            import great_expectations as ge
            self.gx_context = ge.get_context()

        except Exception as e:
            logger.warning(f"Great Expectations setup failed: {e}")
            self.gx_context = None

    def create_expectation_suite(self, df: pd.DataFrame, dataset_name: str) -> ExpectationSuite:
        """
        Create comprehensive expectation suite for a dataset.

        Args:
            df: DataFrame to create expectations for
            dataset_name: Name of the dataset

        Returns:
            ExpectationSuite with comprehensive validations
        """
        suite = ExpectationSuite(f"{dataset_name}_suite")

        # Basic expectations
        suite.add_expectation(
            ExpectationConfiguration(
                expectation_type="expect_table_row_count_to_be_between",
                kwargs={"min_value": 1000, "max_value": 10000000}
            )
        )

        # Column existence expectations
        for col in df.columns:
            suite.add_expectation(
                ExpectationConfiguration(
                    expectation_type="expect_column_to_exist",
                    kwargs={"column": col}
                )
            )

        # Data type expectations
        for col in df.select_dtypes(include=[np.number]).columns:
            suite.add_expectation(
                ExpectationConfiguration(
                    expectation_type="expect_column_values_to_be_of_type",
                    kwargs={"column": col, "type_": "numeric"}
                )
            )

        # Value range expectations for numeric columns
        for col in df.select_dtypes(include=[np.number]).columns:
            if col.lower() in ['sales', 'quantity', 'price', 'value']:
                suite.add_expectation(
                    ExpectationConfiguration(
                        expectation_type="expect_column_values_to_be_between",
                        kwargs={
                            "column": col,
                            "min_value": 0,
                            "max_value": df[col].max() * 2  # Allow some headroom
                        }
                    )
                )

        # Missing value expectations
        for col in df.columns:
            missing_pct = df[col].isnull().mean()
            if missing_pct < 0.1:  # Less than 10% missing
                suite.add_expectation(
                    ExpectationConfiguration(
                        expectation_type="expect_column_proportion_of_unique_values_to_be_between",
                        kwargs={
                            "column": col,
                            "min_value": 0.8,
                            "max_value": 1.0
                        }
                    )
                )

        # Uniqueness expectations for ID columns
        id_columns = [col for col in df.columns if 'id' in col.lower() or '_id' in col.lower()]
        for col in id_columns:
            suite.add_expectation(
                ExpectationConfiguration(
                    expectation_type="expect_column_proportion_of_unique_values_to_be_between",
                    kwargs={
                        "column": col,
                        "min_value": 0.95,
                        "max_value": 1.0
                    }
                )
            )

        return suite

    def validate_dataframe(self, df: pd.DataFrame, dataset_name: str) -> Dict[str, Any]:
        """
        Validate dataframe using Great Expectations.

        Args:
            df: DataFrame to validate
            dataset_name: Name of the dataset

        Returns:
            Validation results
        """
        if not self.gx_context:
            return {"gx_validation": "skipped", "reason": "GX not initialized"}

        try:
            # Create expectation suite
            suite = self.create_expectation_suite(df, dataset_name)

            # Create validator
            batch = ge.Batch(data=df, name=f"{dataset_name}_batch")
            validator = self.gx_context.get_validator(
                batch=batch,
                expectation_suite=suite
            )

            # Run validation
            results = validator.validate()

            # Extract summary
            validation_summary = {
                "successful_expectations": results.statistics["successful_expectations"],
                "evaluated_expectations": results.statistics["evaluated_expectations"],
                "success_percent": results.statistics["success_percent"],
                "failed_expectations": []
            }

            # Extract failed expectations
            for result in results.results:
                if not result.success:
                    validation_summary["failed_expectations"].append({
                        "expectation_type": result.expectation_config.expectation_type,
                        "kwargs": result.expectation_config.kwargs,
                        "result": result.result
                    })

            return {"gx_validation": validation_summary}

        except Exception as e:
            logger.error(f"GX validation failed: {e}")
            return {"gx_validation": "error", "error": str(e)}

    def create_baseline_profile(self, df: pd.DataFrame, dataset_name: str):
        """
        Create baseline statistical profile for drift detection.

        Args:
            df: DataFrame to profile
            dataset_name: Name of the dataset
        """
        profile = {}

        # Basic statistics for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            profile[col] = {
                "mean": df[col].mean(),
                "std": df[col].std(),
                "min": df[col].min(),
                "max": df[col].max(),
                "quantiles": df[col].quantile([0.25, 0.5, 0.75]).to_dict(),
                "null_count": df[col].isnull().sum(),
                "null_percentage": df[col].isnull().mean()
            }

        # Category distributions for categorical columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            value_counts = df[col].value_counts(normalize=True)
            profile[col] = {
                "unique_count": df[col].nunique(),
                "most_common": value_counts.head(10).to_dict(),
                "null_count": df[col].isnull().sum(),
                "null_percentage": df[col].isnull().mean()
            }

        self.baseline_profiles[dataset_name] = profile

        # Save baseline
        baseline_path = PROJECT_ROOT / 'reports' / 'quality' / f'{dataset_name}_baseline.json'
        baseline_path.parent.mkdir(exist_ok=True)

        with open(baseline_path, 'w') as f:
            json.dump(profile, f, indent=2, default=str)

        logger.info(f"Baseline profile created for {dataset_name}")

    def detect_drift(self, df: pd.DataFrame, dataset_name: str) -> List[str]:
        """
        Detect data drift compared to baseline.

        Args:
            df: Current DataFrame
            dataset_name: Name of the dataset

        Returns:
            List of drift alerts
        """
        if dataset_name not in self.baseline_profiles:
            return []

        baseline = self.baseline_profiles[dataset_name]
        alerts = []

        # Check numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col in baseline:
                current_stats = {
                    "mean": df[col].mean(),
                    "std": df[col].std(),
                    "null_percentage": df[col].isnull().mean()
                }

                baseline_stats = baseline[col]

                # Check for significant changes
                mean_change = abs(current_stats["mean"] - baseline_stats["mean"])
                mean_threshold = baseline_stats["std"] * 2  # 2 standard deviations

                if mean_change > mean_threshold:
                    alerts.append(
                        f"Mean drift in {col}: {baseline_stats['mean']:.2f} → {current_stats['mean']:.2f}"
                    )

                # Check null percentage change
                null_change = abs(current_stats["null_percentage"] - baseline_stats["null_percentage"])
                if null_change > 0.1:  # 10% change
                    alerts.append(
                        f"Null percentage drift in {col}: {baseline_stats['null_percentage']:.1%} → {current_stats['null_percentage']:.1%}"
                    )

        return alerts

    def store_validation_results(self, dataset_name: str, validation_results: Dict[str, Any]):
        """
        Store validation results for historical tracking.

        Args:
            dataset_name: Name of the dataset
            validation_results: Validation results from comprehensive_validation
        """
        timestamp = datetime.now().isoformat()

        if dataset_name not in self.quality_history:
            self.quality_history[dataset_name] = []

        result_entry = {
            "timestamp": timestamp,
            "results": validation_results
        }

        self.quality_history[dataset_name].append(result_entry)

        # Keep only last 100 entries
        if len(self.quality_history[dataset_name]) > 100:
            self.quality_history[dataset_name] = self.quality_history[dataset_name][-100:]

    def check_data_drift(self) -> List[str]:
        """
        Check for data drift across all monitored datasets.

        Returns:
            List of drift alerts
        """
        all_alerts = []

        for dataset_name, history in self.quality_history.items():
            if not history:
                continue

            # Get latest data (assuming it's stored in the master dataframe)
            try:
                master_path = OUTPUT_FILES['master_feature_table']
                if master_path.exists():
                    df = pd.read_parquet(master_path)
                    alerts = self.detect_drift(df, dataset_name)
                    all_alerts.extend(alerts)
            except Exception as e:
                logger.warning(f"Drift check failed for {dataset_name}: {e}")

        return all_alerts

    def generate_quality_dashboard(self):
        """
        Generate comprehensive quality dashboard.
        """
        try:
            dashboard_dir = OUTPUT_FILES['dashboard_dir']
            dashboard_dir.mkdir(exist_ok=True)

            # Generate quality report
            report = {
                "generated_at": datetime.now().isoformat(),
                "datasets": {},
                "overall_quality": {}
            }

            for dataset_name, history in self.quality_history.items():
                if not history:
                    continue

                latest = history[-1]
                report["datasets"][dataset_name] = {
                    "latest_quality_score": latest["results"].get("quality_score", 0),
                    "latest_timestamp": latest["timestamp"],
                    "issues_count": len(latest["results"].get("issues", [])),
                    "validation_history": len(history)
                }

            # Calculate overall quality
            if report["datasets"]:
                scores = [info["latest_quality_score"] for info in report["datasets"].values()]
                report["overall_quality"] = {
                    "average_score": sum(scores) / len(scores),
                    "min_score": min(scores),
                    "max_score": max(scores),
                    "datasets_count": len(scores)
                }

            # Save dashboard
            dashboard_path = dashboard_dir / 'data_quality_dashboard.json'
            with open(dashboard_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)

            logger.info(f"Quality dashboard generated: {dashboard_path}")

        except Exception as e:
            logger.error(f"Dashboard generation failed: {e}")

    def get_quality_metrics(self, dataset_name: str) -> Dict[str, Any]:
        """
        Get quality metrics for a dataset.

        Args:
            dataset_name: Name of the dataset

        Returns:
            Quality metrics dictionary
        """
        if dataset_name not in self.quality_history:
            return {}

        history = self.quality_history[dataset_name]
        if not history:
            return {}

        latest = history[-1]

        return {
            "current_score": latest["results"].get("quality_score", 0),
            "issues": latest["results"].get("issues", []),
            "history_length": len(history),
            "last_updated": latest["timestamp"]
        }


# Global instance
data_quality_monitor = DataQualityMonitor()
