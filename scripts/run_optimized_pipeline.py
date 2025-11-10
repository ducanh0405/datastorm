"""
OPTIMIZED PIPELINE RUNNER
=========================
Features:
1. Uses 10x faster WS2 feature engineering
2. Optional hyperparameter tuning with Optuna
3. Performance monitoring and comparison

Usage:
  # Quick run (no tuning)
  python scripts/run_optimized_pipeline.py
  
  # Full optimization (with tuning, ~30min)
  python scripts/run_optimized_pipeline.py --tune --trials 30
  
  # Fast test (small trials)
  python scripts/run_optimized_pipeline.py --tune --trials 10
"""
import sys
import time
import logging
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def check_dependencies():
    """Check if optimization dependencies are installed."""
    missing = []
    
    try:
        import optuna
    except ImportError:
        missing.append('optuna')
    
    try:
        import lightgbm
    except ImportError:
        missing.append('lightgbm')
    
    if missing:
        logging.error(f"Missing dependencies: {', '.join(missing)}")
        logging.error("Install with: pip install " + ' '.join(missing))
        return False
    
    return True


def run_feature_engineering():
    """Run optimized feature engineering pipeline."""
    from src.pipelines._02_feature_enrichment import main as build_features
    
    logging.info("\n" + "=" * 70)
    logging.info("STEP 1: FEATURE ENGINEERING (OPTIMIZED)")
    logging.info("=" * 70)
    
    start_time = time.time()
    
    build_features()
    
    elapsed = time.time() - start_time
    
    logging.info(f"\n[OK] Feature engineering completed in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    
    return elapsed


def run_model_training(tune: bool = False, n_trials: int = 30):
    """Run model training with optional tuning."""
    
    # Use tuned version if requested
    if tune:
        from src.pipelines._03_model_training_tuned import run_training_pipeline
        
        logging.info("\n" + "=" * 70)
        logging.info(f"STEP 2: MODEL TRAINING (TUNED - {n_trials} trials per model)")
        logging.info("=" * 70)
        
        start_time = time.time()
        
        run_training_pipeline(
            tune_hyperparameters=True,
            n_trials=n_trials
        )
        
        elapsed = time.time() - start_time
        
        logging.info(f"\n[OK] Tuned training completed in {elapsed:.1f}s ({elapsed/60:.1f} min)")
        
    else:
        # Use original fast training
        from src.pipelines._03_model_training import main as train_models
        
        logging.info("\n" + "=" * 70)
        logging.info("STEP 2: MODEL TRAINING (QUICK - No Tuning)")
        logging.info("=" * 70)
        
        start_time = time.time()
        
        train_models()
        
        elapsed = time.time() - start_time
        
        logging.info(f"\n[OK] Quick training completed in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    
    return elapsed


def compare_results():
    """Compare original vs optimized results."""
    import json
    from pathlib import Path
    
    logging.info("\n" + "=" * 70)
    logging.info("PERFORMANCE COMPARISON")
    logging.info("=" * 70)
    
    # Check if tuned results exist
    tuned_metrics = Path('models/tuned_model_metrics.json')
    original_metrics = Path('models/model_metrics_v1.json')
    
    if tuned_metrics.exists() and original_metrics.exists():
        with open(tuned_metrics) as f:
            tuned = json.load(f)
        
        with open(original_metrics) as f:
            original = json.load(f)
        
        logging.info("\nMETRIC COMPARISON:")
        logging.info("-" * 70)
        
        for metric in ['q50_pinball_loss', 'coverage_90pct']:
            if metric in tuned and metric in original:
                tuned_val = tuned[metric]
                orig_val = original[metric]
                
                if 'loss' in metric:
                    improvement = ((orig_val - tuned_val) / orig_val) * 100
                    status = "BETTER" if tuned_val < orig_val else "WORSE"
                else:
                    improvement = abs(tuned_val - orig_val) * 100
                    # Coverage: closer to 90% is better
                    target_dist_tuned = abs(tuned_val - 0.90)
                    target_dist_orig = abs(orig_val - 0.90)
                    status = "BETTER" if target_dist_tuned < target_dist_orig else "WORSE"
                
                logging.info(f"{metric:25s}: {orig_val:.6f} -> {tuned_val:.6f} ({status}, {improvement:+.1f}%)")
        
        logging.info("-" * 70)
    
    elif tuned_metrics.exists():
        with open(tuned_metrics) as f:
            tuned = json.load(f)
        
        logging.info("\nTUNED MODEL RESULTS:")
        logging.info("-" * 70)
        for key, val in tuned.items():
            if isinstance(val, float):
                logging.info(f"{key:25s}: {val:.6f}")
        logging.info("-" * 70)
    
    else:
        logging.warning("No metrics files found for comparison")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Run optimized DataStorm pipeline')
    parser.add_argument('--tune', action='store_true', help='Enable hyperparameter tuning (slow)')
    parser.add_argument('--trials', type=int, default=30, help='Number of Optuna trials per model')
    parser.add_argument('--features-only', action='store_true', help='Only run feature engineering')
    parser.add_argument('--models-only', action='store_true', help='Only run model training (requires features)')
    
    args = parser.parse_args()
    
    logging.info("=" * 70)
    logging.info("DATASTORM OPTIMIZED PIPELINE")
    logging.info("=" * 70)
    logging.info(f"Configuration:")
    logging.info(f"  - Hyperparameter Tuning: {args.tune}")
    logging.info(f"  - Optuna Trials: {args.trials}")
    logging.info(f"  - Features Only: {args.features_only}")
    logging.info(f"  - Models Only: {args.models_only}")
    logging.info("=" * 70)
    
    # Check dependencies
    if args.tune:
        if not check_dependencies():
            logging.error("Cannot run tuning without required dependencies")
            return
    
    total_start = time.time()
    timings = {}
    
    # Step 1: Feature Engineering
    if not args.models_only:
        timings['feature_engineering'] = run_feature_engineering()
    
    # Step 2: Model Training
    if not args.features_only:
        timings['model_training'] = run_model_training(tune=args.tune, n_trials=args.trials)
    
    total_elapsed = time.time() - total_start
    
    # Summary
    logging.info("\n" + "=" * 70)
    logging.info("PIPELINE SUMMARY")
    logging.info("=" * 70)
    
    for step, elapsed in timings.items():
        logging.info(f"{step:25s}: {elapsed:6.1f}s ({elapsed/60:5.1f} min)")
    
    logging.info(f"{'TOTAL':25s}: {total_elapsed:6.1f}s ({total_elapsed/60:5.1f} min)")
    logging.info("=" * 70)
    
    # Compare results if tuning was run
    if args.tune and not args.features_only:
        compare_results()
    
    logging.info("\n[COMPLETE] Pipeline finished successfully!")


if __name__ == '__main__':
    main()
