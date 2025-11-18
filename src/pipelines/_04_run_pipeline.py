import argparse
import logging
import os
import subprocess
import sys

# Setup project path and logging
from src.config import PROJECT_ROOT, setup_logging, setup_project_path

setup_project_path()
setup_logging()

# Determine Project Root and Pipelines Directory
PIPELINES_DIR = PROJECT_ROOT / 'src' / 'pipelines'


def run_script(script_name, extra_args=None):
    """Utility function to run a pipeline script and check for errors.

    Args:
        script_name: Name of the script to run
        extra_args: List of additional command-line arguments to pass
    """
    script_path = PIPELINES_DIR / script_name
    logging.info(f"\n--- STARTING SCRIPT: {script_name} ---")

    # Build command with extra arguments
    cmd = [sys.executable, str(script_path)]
    if extra_args:
        cmd.extend(extra_args)
        logging.info(f"With arguments: {' '.join(extra_args)}")

    # Set up environment with proper PYTHONPATH
    env = dict(os.environ)
    env['PYTHONPATH'] = str(PROJECT_ROOT)

    process = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding='utf-8',
        errors='replace',
        cwd=PROJECT_ROOT,
        env=env  # Pass environment variables with PYTHONPATH
    )

    if process.returncode != 0:
        logging.error(f"ERROR running {script_name}:")
        if process.stderr:
            logging.error(process.stderr)
        if process.stdout:
            logging.error(process.stdout)
        return False
    else:
        logging.info(f"--- OK. COMPLETED: {script_name} ---")
        logging.info("Output (last 1000 lines):\n" + process.stdout[-1000:])
        return True


def main():
    """
    Orchestrates the entire SmartGrocy project with optional memory optimizations.
    """
    parser = argparse.ArgumentParser(description='Run full SmartGrocy pipeline')
    parser.add_argument('--full-data', action='store_true',
                       help='Use full data from data/2_raw with memory optimizations (32GB RAM recommended)')
    args = parser.parse_args()

    logging.info("=" * 70)
    logging.info("STARTING ENTIRE PROJECT WORKFLOW")
    logging.info("=" * 70)

    if args.full_data:
        logging.info("🚀 MEMORY OPTIMIZATION MODE ENABLED")
        logging.info("   - Force pandas (no Polars)")
        logging.info("   - Optimized grid (active pairs only)")
        logging.info("   - Memory limit: 8GB")
        logging.info("   - Limited threads: 2 cores")
        logging.info("   - Data source: data/2_raw/")
        logging.info("=" * 70)

        # Set environment variable for downstream scripts
        os.environ['DATA_SOURCE'] = 'full'
        os.environ['USE_PANDAS_ONLY'] = '1'
        os.environ['FORCE_OPTIMIZED_GRID'] = '1'
    else:
        logging.info("📊 STANDARD MODE")
        logging.info("   - Auto-detect best data source (prioritizes data/2_raw)")
        logging.info("=" * 70)

    # Prepare extra args for feature enrichment
    extra_args = ['--full-data'] if args.full_data else None

    # Step 1: Data Processing
    if not run_script('_02_feature_enrichment.py', extra_args):
        logging.critical("Data processing pipeline failed. Halting workflow.")
        sys.exit(1)

    # Step 2: Model Training
    if not run_script('_03_model_training.py'):
        logging.critical("Model training pipeline failed. Halting workflow.")
        sys.exit(1)

    logging.info("\n" + "=" * 70)
    logging.info("✅ ENTIRE WORKFLOW COMPLETED SUCCESSFULLY!")
    logging.info("=" * 70)
    if args.full_data:
        logging.info("📊 Models trained on FULL DATASET with memory optimizations")
    logging.info("=" * 70)


if __name__ == "__main__":
    main()
