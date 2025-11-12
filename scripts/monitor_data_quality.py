#!/usr/bin/env python3
"""
Data Quality Monitoring Dashboard
==================================
Generates comprehensive data quality reports and dashboards.
Can be run as a scheduled job or on-demand.
"""
import sys
import json
from pathlib import Path
from datetime import datetime, timedelta

# Setup path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from src.config import setup_project_path, setup_logging, OUTPUT_FILES
    from src.utils.data_quality import data_quality_monitor
    from src.utils.alerting import alert_manager
    from src.utils.validation import comprehensive_validation
    import pandas as pd

    setup_project_path()
    setup_logging()

    import logging
    logger = logging.getLogger(__name__)

except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)


def generate_quality_report():
    """Generate comprehensive data quality report."""
    logger.info("🔍 Generating data quality report...")

    report = {
        'generated_at': datetime.now().isoformat(),
        'datasets': {},
        'alerts_summary': {},
        'recommendations': []
    }

    try:
        # Check master feature table if it exists
        master_path = OUTPUT_FILES['master_feature_table']
        if master_path.exists():
            logger.info("Checking master feature table...")
            df = pd.read_parquet(master_path)

            validation_results = comprehensive_validation(df, verbose=False)
            quality_score = validation_results.get('quality_score', 0)

            report['datasets']['master_feature_table'] = {
                'shape': df.shape,
                'quality_score': quality_score,
                'issues': validation_results.get('issues', []),
                'last_modified': datetime.fromtimestamp(master_path.stat().st_mtime).isoformat()
            }

            # Add recommendations based on quality score
            if quality_score < 70:
                report['recommendations'].append(
                    "Critical: Master feature table quality is below acceptable threshold"
                )
            elif quality_score < 85:
                report['recommendations'].append(
                    "Warning: Consider reviewing data preprocessing steps"
                )

        # Check for data drift
        logger.info("Checking for data drift...")
        drift_alerts = data_quality_monitor.check_data_drift()
        if drift_alerts:
            report['drift_alerts'] = drift_alerts
            report['recommendations'].append(
                f"Data drift detected in {len(drift_alerts)} metrics"
            )

        # Get alert summary
        alert_summary = alert_manager.get_alert_summary(hours=24)
        report['alerts_summary'] = alert_summary

        if alert_summary['total_alerts'] > 10:
            report['recommendations'].append(
                "High alert volume detected - review pipeline stability"
            )

        # Generate quality dashboard
        data_quality_monitor.generate_quality_dashboard()

        # Save report
        reports_dir = OUTPUT_FILES['reports_dir']
        report_path = reports_dir / 'quality' / f'quality_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        report_path.parent.mkdir(exist_ok=True)

        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        logger.info(f"✅ Quality report saved to: {report_path}")

        # Print summary
        print("\n" + "="*60)
        print("📊 DATA QUALITY REPORT SUMMARY")
        print("="*60)
        print(f"Generated: {report['generated_at']}")
        print(f"Datasets checked: {len(report['datasets'])}")
        print(f"Quality issues: {sum(len(ds.get('issues', [])) for ds in report['datasets'].values())}")
        print(f"Alerts (24h): {alert_summary['total_alerts']}")
        print(f"Drift alerts: {len(drift_alerts) if 'drift_alerts' in report else 0}")

        if report['recommendations']:
            print("\n📋 RECOMMENDATIONS:")
            for rec in report['recommendations']:
                print(f"  • {rec}")

        print("="*60)

        return report

    except Exception as e:
        logger.error(f"❌ Quality report generation failed: {e}")
        raise


def main():
    """Main function for data quality monitoring."""
    import argparse

    parser = argparse.ArgumentParser(description='Monitor data quality and generate reports')
    parser.add_argument('--schedule', type=int,
                       help='Run continuously every N hours')
    parser.add_argument('--output-dir', type=str,
                       help='Custom output directory for reports')

    args = parser.parse_args()

    if args.schedule:
        logger.info(f"🔄 Running quality monitoring every {args.schedule} hours")

        while True:
            try:
                generate_quality_report()
                logger.info(f"⏰ Next report in {args.schedule} hours...")
            except Exception as e:
                logger.error(f"Quality monitoring failed: {e}")

            # Sleep for specified hours
            import time
            time.sleep(args.schedule * 3600)
    else:
        # Run once
        generate_quality_report()


if __name__ == "__main__":
    main()
