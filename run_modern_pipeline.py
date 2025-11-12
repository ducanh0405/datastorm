#!/usr/bin/env python3
"""
Modern Pipeline Runner
======================
Uses Prefect-based orchestration with modern practices including:
- Data quality monitoring
- Caching and performance optimization
- Alerting and error handling
- Comprehensive logging and monitoring
"""
import sys
import logging
from pathlib import Path

# Setup path
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from prefect import flow  # pyright: ignore[reportMissingImports]
    from src.pipelines._00_modern_orchestrator import modern_pipeline_flow
    from src.config import setup_project_path, setup_logging
    from src.utils.alerting import alert_manager

    setup_project_path()
    setup_logging()

    logger = logging.getLogger(__name__)

except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please install required dependencies:")
    print("pip install -r requirements.txt")
    sys.exit(1)


def main():
    """Run the modern pipeline with comprehensive monitoring."""
    import argparse

    parser = argparse.ArgumentParser(description='Run modern SmartGrocy pipeline')
    parser.add_argument('--full-data', action='store_true',
                       help='Use full dataset with memory optimizations')
    parser.add_argument('--prefect-server', action='store_true',
                       help='Use Prefect server for orchestration')
    parser.add_argument('--no-cache', action='store_true',
                       help='Disable caching for this run')

    args = parser.parse_args()

    logger.info("🚀 Starting Modern SmartGrocy Pipeline")
    logger.info("=" * 60)

    try:
        if args.prefect_server:
            logger.info("Using Prefect server for orchestration")
            # Additional Prefect server setup would go here
        else:
            logger.info("Using local Prefect execution")

        if args.no_cache:
            logger.info("⚠️ Caching disabled for this run")

        # Run the modern pipeline
        result = modern_pipeline_flow(full_data=args.full_data)

        logger.info("✅ Pipeline completed successfully!")
        logger.info("=" * 60)

        # Show final quality metrics
        quality_summary = alert_manager.get_alert_summary(hours=1)
        if quality_summary['total_alerts'] > 0:
            logger.info(f"📊 Quality Summary (last hour): {quality_summary}")
        else:
            logger.info("✅ No quality issues detected")

    except Exception as e:
        logger.error(f"❌ Pipeline failed: {e}")
        if alert_manager:
            alert_manager.alert_pipeline_failure(
                pipeline_stage="pipeline_execution",
                error_message=str(e)
            )
        sys.exit(1)


if __name__ == "__main__":
    main()
