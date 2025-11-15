#!/usr/bin/env python3
"""
SmartGrocy Pipeline Runner - Consolidated
==========================================
Unified pipeline runner combining v1 and v2 features.

Usage:
    # Full pipeline with full dataset
    python run_pipeline.py --full-data
    
    # Quick test with 10% sample
    python run_pipeline.py --full-data --sample 0.1
    
    # Use v2 orchestrator (with GX validation)
    python run_pipeline.py --full-data --use-v2
    
    # Disable caching
    python run_pipeline.py --full-data --no-cache
"""
import sys
import logging
import argparse
from pathlib import Path

# Setup path
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from src.config import setup_project_path, setup_logging, MEMORY_OPTIMIZATION
    from src.utils.alerting import alert_manager
    
    setup_project_path()
    setup_logging()
    
    logger = logging.getLogger(__name__)
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please install required dependencies:")
    print("  pip install -r requirements.txt")
    sys.exit(1)


def setup_memory_sampling(sample_fraction: float):
    """Configure memory sampling dynamically."""
    if sample_fraction < 1.0:
        from src import config
        config.MEMORY_OPTIMIZATION['enable_sampling'] = True
        config.MEMORY_OPTIMIZATION['sample_fraction'] = sample_fraction
        logger.info(f"📊 Memory Sampling: {sample_fraction*100:.0f}% of data")
        return True
    else:
        logger.info("📊 Memory Configuration: Full dataset (no sampling)")
        return False


def check_prerequisites():
    """Check if all prerequisites are met."""
    logger.info("\n✅ Checking prerequisites...")
    
    all_good = True
    
    # Check data directories
    from src.config import DATA_DIRS, ensure_directories
    ensure_directories()
    logger.info("✓ Data directories: Ready")
    
    # Check for training data
    raw_data_dir = DATA_DIRS['raw_data']
    if not raw_data_dir.exists() or not list(raw_data_dir.glob('*.parquet')):
        logger.warning("⚠️ No training data found in data/2_raw/")
        logger.warning("   Pipeline may fail at data loading stage")
        all_good = False
    else:
        data_files = list(raw_data_dir.glob('*.parquet'))
        logger.info(f"✓ Training data: {len(data_files)} file(s) found")
    
    return all_good


def main():
    """Main entry point for pipeline execution."""
    parser = argparse.ArgumentParser(
        description='SmartGrocy Pipeline Runner (Consolidated)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full data with GX validation
  python run_pipeline.py --full-data --use-v2
  
  # Quick test with 10% sample
  python run_pipeline.py --full-data --sample 0.1
  
  # Standard pipeline (v1)
  python run_pipeline.py --full-data
        """
    )
    
    parser.add_argument(
        '--full-data', action='store_true',
        help='Use full dataset (default: sample mode)'
    )
    
    parser.add_argument(
        '--sample', type=float, default=1.0,
        help='Sample fraction (0.1 = 10%% of data). Default: 1.0 (100%%)'
    )
    
    parser.add_argument(
        '--use-v2', action='store_true',
        help='Use v2 orchestrator with full GX integration (recommended)'
    )
    
    parser.add_argument(
        '--no-cache', action='store_true',
        help='Disable caching for fresh run'
    )
    
    parser.add_argument(
        '--prefect-server', action='store_true',
        help='Use Prefect server for orchestration'
    )
    
    args = parser.parse_args()
    
    # Header
    logger.info("="*70)
    logger.info("🚀 SMARTGROCY PIPELINE RUNNER")
    logger.info("="*70)
    
    # Check prerequisites
    prereq_ok = check_prerequisites()
    if not prereq_ok:
        logger.warning("\n⚠️ Some prerequisites not met. Pipeline may fail.")
        user_input = input("Continue anyway? (y/N): ")
        if user_input.lower() != 'y':
            logger.info("Pipeline cancelled by user.")
            sys.exit(0)
    
    # Setup memory sampling
    if args.sample < 1.0:
        setup_memory_sampling(args.sample)
    elif not args.full_data:
        logger.info("⚠️ Running in sample mode (default)")
        logger.info("   Use --full-data to process complete dataset")
        setup_memory_sampling(0.1)  # Default 10% sample
    
    # Cache configuration
    if args.no_cache:
        logger.info("⚠️ Caching disabled for this run")
        from src import config
        config.CACHE_CONFIG['enable_incremental_processing'] = False
    
    # Select orchestrator
    if args.use_v2:
        logger.info("⚙️ Using v2 orchestrator (with GX integration)")
        try:
            from src.pipelines._00_modern_orchestrator_v2 import modern_pipeline_flow_v2
            pipeline_func = modern_pipeline_flow_v2
        except ImportError as e:
            logger.error(f"❌ Failed to import v2 orchestrator: {e}")
            logger.info("Falling back to v1 orchestrator...")
            from src.pipelines._00_modern_orchestrator import modern_pipeline_flow
            pipeline_func = modern_pipeline_flow
    else:
        logger.info("⚙️ Using v1 orchestrator (standard)")
        from src.pipelines._00_modern_orchestrator import modern_pipeline_flow
        pipeline_func = modern_pipeline_flow
    
    logger.info("="*70)
    
    # Run pipeline
    try:
        result = pipeline_func(full_data=args.full_data)
        
        logger.info("\n" + "="*70)
        logger.info("✅ PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("="*70)
        
        # Show summary
        if isinstance(result, dict):
            logger.info(f"Duration: {result.get('duration_seconds', 0):.1f}s")
            logger.info(f"Final shape: {result.get('final_shape', 'N/A')}")
        
        # Quality summary
        if alert_manager:
            quality_summary = alert_manager.get_alert_summary(hours=1)
            if quality_summary['total_alerts'] > 0:
                logger.warning(f"\n⚠️ Quality Summary (last hour):")
                logger.warning(f"  Total alerts: {quality_summary['total_alerts']}")
            else:
                logger.info("\n✅ No quality issues detected")
        
        logger.info("="*70)
        
        sys.exit(0)
        
    except KeyboardInterrupt:
        logger.warning("\n⚠️ Pipeline interrupted by user (Ctrl+C)")
        sys.exit(130)
        
    except Exception as e:
        logger.error("\n" + "="*70)
        logger.error("❌ PIPELINE FAILED")
        logger.error("="*70)
        logger.error(f"Error: {e}")
        logger.error("\nFor debugging:")
        logger.error("  1. Check logs: logs/pipeline.log")
        logger.error("  2. Review GX reports: great_expectations/uncommitted/data_docs/")
        logger.error("  3. Check data quality: python scripts/run_data_quality_check.py")
        logger.error("="*70)
        
        # Alert
        if alert_manager:
            alert_manager.alert_pipeline_failure(
                pipeline_stage="pipeline_execution",
                error_message=str(e)
            )
        
        import traceback
        traceback.print_exc()
        
        sys.exit(1)


if __name__ == "__main__":
    main()

