import subprocess
import sys
import logging
from pathlib import Path

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Determine Project Root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PIPELINES_DIR = PROJECT_ROOT / 'src' / 'pipelines'


def run_script(script_name):
    """Utility function to run a pipeline script and check for errors."""
    script_path = PIPELINES_DIR / script_name
    logging.info(f"\n--- STARTING SCRIPT: {script_name} ---")

    process = subprocess.run(
        [sys.executable, str(script_path)],
        capture_output=True,
        text=True,
        encoding='utf-8',
        errors='replace'
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
    Orchestrates the entire E-Grocery Forecaster project.
    """
    logging.info("========== STARTING ENTIRE PROJECT WORKFLOW ==========")

    # Step 1: Data Processing
    if not run_script('_02_feature_enrichment.py'):
        logging.critical("Data processing pipeline failed. Halting workflow.")
        sys.exit(1)

    # Step 2: Model Training
    if not run_script('_03_model_training.py'):
        logging.critical("Model training pipeline failed. Halting workflow.")
        sys.exit(1)

    logging.info("\n========== OK. ENTIRE WORKFLOW COMPLETED SUCCESSFULLY! ==========")


if __name__ == "__main__":
    main()