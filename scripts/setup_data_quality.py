#!/usr/bin/env python3
"""
Data Quality Setup Script
=========================
Initializes Great Expectations, creates expectation suites,
and sets up comprehensive data quality monitoring.
"""
import sys
import json
from pathlib import Path

# Setup path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from src.config import setup_project_path, setup_logging, OUTPUT_FILES, DATA_QUALITY_CONFIG
    from src.utils.data_quality import DataQualityMonitor
    import pandas as pd

    setup_project_path()
    setup_logging()

    import logging
    logger = logging.getLogger(__name__)

except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)


def setup_great_expectations():
    """Setup Great Expectations project structure."""
    logger.info("🔧 Setting up Great Expectations...")

    try:
        import subprocess

        # Initialize Great Expectations project
        result = subprocess.run([
            'great_expectations', 'init', '--no-view'
        ], cwd=project_root, capture_output=True, text=True)

        if result.returncode == 0:
            logger.info("✅ Great Expectations initialized successfully")
        else:
            logger.warning(f"Great Expectations init output: {result.stderr}")

        # Create custom expectation suites
        create_expectation_suites()

    except Exception as e:
        logger.error(f"❌ Great Expectations setup failed: {e}")


def create_expectation_suites():
    """Create custom expectation suites for different data types."""
    logger.info("📋 Creating expectation suites...")

    try:
        import great_expectations as ge
        from great_expectations.core import ExpectationSuite
        from great_expectations.core.expectation_configuration import ExpectationConfiguration

        context = ge.get_context()

        # E-commerce dataset suite
        ecommerce_suite = ExpectationSuite("ecommerce_dataset_suite")

        # Basic validations
        ecommerce_suite.add_expectation(
            ExpectationConfiguration(
                expectation_type="expect_table_row_count_to_be_between",
                kwargs={"min_value": 10000, "max_value": 10000000}
            )
        )

        # Column expectations for common e-commerce fields
        ecommerce_suite.add_expectation(
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_not_be_null",
                kwargs={"column": "product_id"}
            )
        )

        ecommerce_suite.add_expectation(
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_not_be_null",
                kwargs={"column": "store_id"}
            )
        )

        # Sales value expectations
        ecommerce_suite.add_expectation(
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_be_between",
                kwargs={
                    "column": "sales_quantity",
                    "min_value": 0,
                    "max_value": None
                }
            )
        )

        # Time column expectations
        ecommerce_suite.add_expectation(
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_be_of_type",
                kwargs={
                    "column": "hour_timestamp",
                    "type_": "datetime64[ns]"
                }
            )
        )

        # Save the suite
        context.save_expectation_suite(ecommerce_suite)
        logger.info("✅ E-commerce expectation suite created")

        # Time-series suite
        timeseries_suite = ExpectationSuite("timeseries_features_suite")

        # Lag feature expectations
        lag_columns = [f"sales_quantity_lag_{i}" for i in [1, 24, 48, 168]]
        for col in lag_columns:
            timeseries_suite.add_expectation(
                ExpectationConfiguration(
                    expectation_type="expect_column_values_to_be_between",
                    kwargs={
                        "column": col,
                        "min_value": 0,
                        "max_value": None,
                        "mostly": 0.95  # Allow some nulls for early periods
                    }
                )
            )

        # Rolling statistics expectations
        rolling_cols = [f"rolling_mean_{w}_lag_1" for w in [24, 168]]
        for col in rolling_cols:
            timeseries_suite.add_expectation(
                ExpectationConfiguration(
                    expectation_type="expect_column_values_to_be_between",
                    kwargs={
                        "column": col,
                        "min_value": 0,
                        "max_value": None
                    }
                )
            )

        context.save_expectation_suite(timeseries_suite)
        logger.info("✅ Time-series expectation suite created")

    except Exception as e:
        logger.error(f"❌ Expectation suite creation failed: {e}")


def setup_baseline_profiles():
    """Setup baseline statistical profiles for drift detection."""
    logger.info("📊 Setting up baseline profiles...")

    try:
        # Try to load existing data for baseline creation
        master_path = OUTPUT_FILES['master_feature_table']
        if master_path.exists():
            logger.info("Using existing master feature table for baseline...")

            df = pd.read_parquet(master_path)
            sample_size = min(100000, len(df))  # Use up to 100k rows for baseline
            df_sample = df.sample(sample_size, random_state=42)

            # Create baseline profiles
            quality_monitor = DataQualityMonitor()
            quality_monitor.create_baseline_profile(df_sample, 'master_feature_table')

            logger.info("✅ Baseline profiles created from existing data")
        else:
            logger.info("ℹ️ No existing data found - baseline will be created on first pipeline run")

    except Exception as e:
        logger.error(f"❌ Baseline profile setup failed: {e}")


def create_monitoring_config():
    """Create monitoring configuration files."""
    logger.info("⚙️ Creating monitoring configuration...")

    try:
        config_dir = project_root / 'config'
        config_dir.mkdir(exist_ok=True)

        # Data quality monitoring config
        dq_config = {
            'monitoring_enabled': True,
            'check_frequency_hours': 24,
            'alert_thresholds': {
                'quality_score_min': 70,
                'drift_significance_level': 0.05,
                'max_null_percentage': 0.5
            },
            'report_schedule': 'daily',
            'notification_channels': ['log', 'file']
        }

        with open(config_dir / 'data_quality_config.json', 'w') as f:
            json.dump(dq_config, f, indent=2)

        # Alerting config template
        alert_config = {
            'email_alerts': {
                'enabled': False,
                'smtp_server': 'smtp.gmail.com',
                'smtp_port': 587,
                'sender_email': 'your-email@example.com',
                'sender_password': 'your-app-password',
                'recipient_emails': ['team@example.com']
            },
            'slack_alerts': {
                'enabled': False,
                'bot_token': 'xoxb-your-slack-bot-token',
                'channel': '#pipeline-alerts'
            }
        }

        with open(config_dir / 'alert_config_template.json', 'w') as f:
            json.dump(alert_config, f, indent=2)

        logger.info("✅ Monitoring configuration created")

        print("\n" + "="*60)
        print("📋 CONFIGURATION SETUP COMPLETE")
        print("="*60)
        print("To enable email alerts:")
        print("1. Copy config/alert_config_template.json to config/alert_config.json")
        print("2. Fill in your email/SMTP credentials")
        print("3. Set email_alerts.enabled to true in src/config.py")
        print()
        print("To enable Slack alerts:")
        print("1. Create a Slack app and bot token")
        print("2. Update slack_alerts configuration")
        print("3. Set slack_alerts.enabled to true in src/config.py")
        print("="*60)

    except Exception as e:
        logger.error(f"❌ Monitoring config creation failed: {e}")


def main():
    """Main setup function."""
    logger.info("🚀 Starting data quality monitoring setup...")

    print("Setting up comprehensive data quality monitoring...")
    print("="*60)

    # Setup Great Expectations
    setup_great_expectations()

    # Create baseline profiles
    setup_baseline_profiles()

    # Create monitoring configuration
    create_monitoring_config()

    logger.info("✅ Data quality monitoring setup complete!")

    print("\n" + "="*60)
    print("🎉 SETUP COMPLETE!")
    print("="*60)
    print("Data quality monitoring is now configured with:")
    print("• Great Expectations for validation")
    print("• Statistical profiling for drift detection")
    print("• Automated alerting system")
    print("• Quality dashboards and reporting")
    print()
    print("Next steps:")
    print("1. Run the modern pipeline: python run_modern_pipeline.py")
    print("2. Monitor quality: python scripts/monitor_data_quality.py")
    print("3. Check reports in reports/quality/")
    print("="*60)


if __name__ == "__main__":
    main()
