#!/usr/bin/env python3
"""
Modern Pipeline Demonstration
============================
Demonstrates all modern pipeline features working together.
"""
import sys
import json
from pathlib import Path
from datetime import datetime

# Setup path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from src.config import setup_project_path, setup_logging
    from src.utils.data_quality import data_quality_monitor
    from src.utils.alerting import alert_manager
    from src.utils.caching import pipeline_cache
    from src.utils.data_lineage import lineage_tracker, DataArtifact, PipelineStep
    from src.utils.performance_monitor import performance_monitor
    from src.utils.validation import comprehensive_validation
    import pandas as pd
    import numpy as np

    setup_project_path()
    setup_logging()

    import logging
    logger = logging.getLogger(__name__)

except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)


def create_sample_data() -> pd.DataFrame:
    """Create sample e-commerce data for demonstration."""
    logger.info("Creating sample e-commerce dataset...")

    np.random.seed(42)
    n_rows = 10000

    # Generate sample data
    data = {
        'product_id': np.random.randint(1000, 9999, n_rows),
        'store_id': np.random.randint(1, 50, n_rows),
        'hour_timestamp': pd.date_range('2024-01-01', periods=n_rows, freq='H'),
        'sales_quantity': np.random.exponential(2, n_rows).astype(int),
        'price': np.random.uniform(10, 500, n_rows),
        'category': np.random.choice(['electronics', 'clothing', 'food', 'books'], n_rows)
    }

    df = pd.DataFrame(data)

    # Add some quality issues for demonstration
    df.loc[np.random.choice(df.index, 50), 'sales_quantity'] = -1  # Negative sales
    df.loc[np.random.choice(df.index, 30), 'price'] = np.nan  # Missing prices

    return df


def demonstrate_data_quality_monitoring(df: pd.DataFrame):
    """Demonstrate data quality monitoring features."""
    print("\n" + "="*60)
    print("🔍 DATA QUALITY MONITORING DEMONSTRATION")
    print("="*60)

    # Register artifact in lineage tracker
    artifact = DataArtifact(
        name='raw_ecommerce_data',
        artifact_type='raw_data',
        shape=df.shape,
        schema={col: str(dtype) for col, dtype in df.dtypes.items()},
        metadata={'source': 'synthetic', 'rows': len(df)}
    )
    lineage_tracker.register_artifact(artifact)

    # Comprehensive validation
    print("Running comprehensive validation...")
    validation_results = comprehensive_validation(df, verbose=True)

    # Store validation results
    data_quality_monitor.store_validation_results('raw_ecommerce_data', validation_results)

    # Create baseline profile
    print("\nCreating statistical baseline...")
    data_quality_monitor.create_baseline_profile(df, 'raw_ecommerce_data')

    # Generate quality report
    print("\nGenerating quality dashboard...")
    data_quality_monitor.generate_quality_dashboard()

    print(f"✅ Quality score: {validation_results.get('quality_score', 0)}/100")
    print(f"📊 Issues found: {len(validation_results.get('issues', []))}")


def demonstrate_caching(df: pd.DataFrame):
    """Demonstrate intelligent caching."""
    print("\n" + "="*60)
    print("💾 INTELLIGENT CACHING DEMONSTRATION")
    print("="*60)

    # First run - should process everything
    print("First run (cold cache)...")
    start_time = datetime.now()

    # Simulate processing with caching
    processed_df = pipeline_cache.process_incremental(
        df, 'demo_dataset', chunk_size=1000, force_reprocess=True
    )

    first_run_time = (datetime.now() - start_time).total_seconds()
    print(f"First run completed in {first_run_time:.2f}s")

    # Second run - should use cache
    print("\nSecond run (warm cache)...")
    start_time = datetime.now()

    processed_df2 = pipeline_cache.process_incremental(
        df, 'demo_dataset', chunk_size=1000, force_reprocess=False
    )

    second_run_time = (datetime.now() - start_time).total_seconds()
    print(f"Second run completed in {second_run_time:.2f}s")

    # Show cache stats
    cache_stats = pipeline_cache.get_stats()
    incremental_stats = pipeline_cache.get_incremental_stats('demo_dataset')

    print("
📊 Cache Statistics:"    print(f"  Cache hits: {cache_stats['stats']['hits']}")
    print(f"  Cache misses: {cache_stats['stats']['misses']}")
    print(f"  Cache hit rate: {cache_stats.get('hit_rate', 0):.1%}")
    print(f"  Incremental cache hit rate: {incremental_stats.get('cache_hit_rate', 0):.1%}")
    print(f"  Performance improvement: {first_run_time/second_run_time:.1f}x faster")


def demonstrate_lineage_tracking():
    """Demonstrate data lineage tracking."""
    print("\n" + "="*60)
    print("🔗 DATA LINEAGE TRACKING DEMONSTRATION")
    print("="*60)

    # Record pipeline steps
    steps = [
        PipelineStep(
            step_name='data_loading',
            step_type='load',
            inputs=[],
            outputs=['raw_ecommerce_data'],
            parameters={'source': 'synthetic'},
            status='success',
            execution_time=1.2
        ),
        PipelineStep(
            step_name='data_validation',
            step_type='transform',
            inputs=['raw_ecommerce_data'],
            outputs=['validated_data'],
            parameters={'validation_rules': 'comprehensive'},
            status='success',
            execution_time=0.8
        ),
        PipelineStep(
            step_name='feature_engineering',
            step_type='feature_engineering',
            inputs=['validated_data'],
            outputs=['feature_table'],
            parameters={'features': ['time_features', 'categorical_encoding']},
            status='success',
            execution_time=2.1
        )
    ]

    for step in steps:
        lineage_tracker.record_step(step)

    # Show lineage information
    print("Pipeline flow:")
    for i, step in enumerate(steps, 1):
        print(f"  {i}. {step.step_name} → {', '.join(step.outputs)}")

    # Get lineage for final artifact
    lineage = lineage_tracker.get_artifact_lineage('feature_table')
    print(f"\nLineage for 'feature_table':")
    print(f"  Upstream steps: {len(lineage.get('upstream_steps', []))}")
    print(f"  Upstream artifacts: {len(lineage.get('upstream_artifacts', []))}")

    # Generate lineage report
    report = lineage_tracker.generate_lineage_report()
    print("
📊 Lineage Statistics:"    print(f"  Total artifacts: {report['total_artifacts']}")
    print(f"  Total steps: {report['total_steps']}")
    print(f"  Artifacts by type: {report['artifacts_by_type']}")

    # Export lineage graph
    graph_path = lineage_tracker.export_lineage_graph()
    if graph_path:
        print(f"  Lineage graph exported to: {graph_path}")

    # Save lineage
    lineage_tracker.save_lineage()


def demonstrate_performance_monitoring():
    """Demonstrate performance monitoring."""
    print("\n" + "="*60)
    print("⚡ PERFORMANCE MONITORING DEMONSTRATION")
    print("="*60)

    # Start monitoring
    print("Starting performance monitoring...")
    performance_monitor.start_monitoring()

    # Simulate some operations
    operations = [
        ('data_loading', lambda: time.sleep(0.5)),
        ('feature_engineering', lambda: time.sleep(1.0)),
        ('model_training', lambda: time.sleep(1.5)),
        ('validation', lambda: time.sleep(0.3))
    ]

    import time
    for op_name, op_func in operations:
        print(f"Running {op_name}...")
        with performance_monitor.time_operation(op_name):
            op_func()

    # Stop monitoring and get results
    session_summary = performance_monitor.stop_monitoring()

    print("
📊 Performance Session Summary:"    print(f"  Session duration: {session_summary.get('session_duration', 0):.2f}s")
    print(f"  Total measurements: {session_summary.get('total_measurements', 0)}")
    print(f"  Operations recorded: {session_summary.get('operations_count', 0)}")

    if 'cpu_avg_percent' in session_summary:
        print(f"  Avg CPU usage: {session_summary['cpu_avg_percent']:.1f}%")
        print(f"  Max CPU usage: {session_summary['cpu_max_percent']:.1f}%")
        print(f"  Avg memory usage: {session_summary['memory_avg_percent']:.1f}%")

    # Show operation stats
    op_stats = session_summary.get('operation_stats', {})
    if op_stats:
        print(f"\n  Operation Statistics:")
        for op_name, stats in op_stats.items():
            if not op_name.startswith('_'):
                print(f"    {op_name}: {stats['avg_time']:.2f}s avg, {stats['count']} runs")

    # Get recommendations
    recommendations = performance_monitor.get_performance_recommendations()
    if recommendations:
        print(f"\n  💡 Recommendations:")
        for rec in recommendations:
            print(f"    • {rec}")

    # System info
    system_info = performance_monitor.get_system_info()
    print(f"\n  🖥️  System Info:")
    print(f"    CPU cores: {system_info.get('cpu_count', 'N/A')}")
    print(f"    Memory: {system_info.get('memory_total_gb', 0):.1f} GB")


def demonstrate_alerting():
    """Demonstrate alerting system."""
    print("\n" + "="*60)
    print("🚨 ALERTING SYSTEM DEMONSTRATION")
    print("="*60)

    # Simulate different types of alerts
    alerts = [
        {
            'type': 'data_quality',
            'message': 'Sample data quality alert',
            'severity': 'warning'
        },
        {
            'type': 'pipeline_failure',
            'message': 'Simulated pipeline step failure',
            'severity': 'error'
        },
        {
            'type': 'performance',
            'message': 'High memory usage detected',
            'severity': 'warning'
        }
    ]

    for alert in alerts:
        alert_manager.send_alert(
            alert_type=alert['type'],
            message=alert['message'],
            severity=alert['severity'],
            metadata={'demo': True, 'timestamp': datetime.now().isoformat()}
        )
        print(f"Sent {alert['severity']} alert: {alert['message']}")

    # Show alert summary
    alert_summary = alert_manager.get_alert_summary(hours=1)
    print("
📊 Alert Summary (last hour):"    print(f"  Total alerts: {alert_summary['total_alerts']}")
    print(f"  By severity: {alert_summary['by_severity']}")
    print(f"  By type: {alert_summary['by_type']}")


def main():
    """Main demonstration function."""
    print("🎭 MODERN PIPELINE FEATURES DEMONSTRATION")
    print("="*60)
    print("This script demonstrates all modern pipeline features:")
    print("• Data Quality Monitoring with Great Expectations")
    print("• Intelligent Caching with incremental processing")
    print("• Data Lineage Tracking")
    print("• Performance Monitoring & Optimization")
    print("• Comprehensive Alerting System")
    print("="*60)

    try:
        # Create sample data
        df = create_sample_data()

        # Run demonstrations
        demonstrate_data_quality_monitoring(df)
        demonstrate_caching(df)
        demonstrate_lineage_tracking()
        demonstrate_performance_monitoring()
        demonstrate_alerting()

        print("\n" + "="*60)
        print("🎉 DEMONSTRATION COMPLETE!")
        print("="*60)
        print("All modern pipeline features have been successfully demonstrated.")
        print("\nNext steps:")
        print("1. Run: python scripts/setup_data_quality.py")
        print("2. Run: python run_modern_pipeline.py --full-data")
        print("3. Monitor: python scripts/monitor_data_quality.py")
        print("="*60)

    except Exception as e:
        logger.error(f"❌ Demonstration failed: {e}")
        raise


if __name__ == "__main__":
    main()
