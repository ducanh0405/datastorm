#!/usr/bin/env python3
"""
Regenerate All Reports with Updated Backtesting Values
======================================================
Script để regenerate toàn bộ reports sau khi cập nhật estimated_results.csv

Usage:
    python regenerate_reports.py
"""
import sys
import subprocess
import logging
from pathlib import Path

# Setup path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_script(script_path: Path, description: str) -> bool:
    """Run a Python script and return success status"""
    logger.info(f"\n{'='*70}")
    logger.info(f"Running: {description}")
    logger.info(f"Script: {script_path}")
    logger.info(f"{'='*70}")
    
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=False,
            text=True
        )
        
        if result.returncode == 0:
            logger.info(f"✅ {description} completed successfully")
            return True
        else:
            logger.error(f"❌ {description} failed with return code {result.returncode}")
            return False
            
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ {description} failed: {e}")
        return False
    except FileNotFoundError:
        logger.error(f"❌ Script not found: {script_path}")
        return False
    except Exception as e:
        logger.error(f"❌ Error running {description}: {e}")
        return False


def main():
    """Main execution"""
    logger.info("\n" + "="*70)
    logger.info("REGENERATING ALL REPORTS WITH UPDATED BACKTESTING VALUES")
    logger.info("="*70)
    
    # Check if estimated_results.csv exists
    estimated_results = PROJECT_ROOT / "reports" / "backtesting" / "estimated_results.csv"
    if not estimated_results.exists():
        logger.error(f"❌ File not found: {estimated_results}")
        logger.error("Please ensure estimated_results.csv exists before regenerating reports")
        return False
    
    logger.info(f"✅ Found updated backtesting results: {estimated_results}")
    
    # Read the updated values to show what we're working with
    try:
        import pandas as pd
        df = pd.read_csv(estimated_results)
        logger.info("\n📊 Updated Backtesting Values:")
        for _, row in df.iterrows():
            logger.info(f"  {row['metric']}: {row['baseline']} → {row['ml_model']} "
                       f"(improvement: {row['improvement_pct']:.2f}%)")
    except Exception as e:
        logger.warning(f"⚠ Could not read estimated_results.csv: {e}")
    
    # List of scripts to run in order
    scripts_to_run = [
        {
            'path': PROJECT_ROOT / 'scripts' / 'run_backtesting_analysis.py',
            'description': 'Backtesting Analysis (regenerate backtesting reports)'
        },
        {
            'path': PROJECT_ROOT / 'run_business_modules.py',
            'description': 'Business Modules (Inventory + Pricing + LLM Insights)'
        },
        {
            'path': PROJECT_ROOT / 'scripts' / 'generate_report_charts.py',
            'description': 'Report Charts Generation (all visualizations)'
        },
        {
            'path': PROJECT_ROOT / 'scripts' / 'generate_technical_report.py',
            'description': 'Technical Report Generation'
        },
        {
            'path': PROJECT_ROOT / 'scripts' / 'generate_summary_statistics.py',
            'description': 'Summary Statistics Generation'
        }
    ]
    
    # Run each script
    results = []
    for script_info in scripts_to_run:
        script_path = script_info['path']
        description = script_info['description']
        
        if not script_path.exists():
            logger.warning(f"⚠ Skipping {description}: Script not found at {script_path}")
            results.append((description, False, "Script not found"))
            continue
        
        success = run_script(script_path, description)
        results.append((description, success, "Completed" if success else "Failed"))
    
    # Summary
    logger.info("\n" + "="*70)
    logger.info("REGENERATION SUMMARY")
    logger.info("="*70)
    
    success_count = sum(1 for _, success, _ in results if success)
    total_count = len(results)
    
    for description, success, status in results:
        status_icon = "✅" if success else "❌"
        logger.info(f"{status_icon} {description}: {status}")
    
    logger.info(f"\n📊 Results: {success_count}/{total_count} scripts completed successfully")
    
    if success_count == total_count:
        logger.info("\n🎉 All reports regenerated successfully!")
        logger.info("\n📁 Updated Reports:")
        logger.info("  - reports/backtesting/estimated_results.csv")
        logger.info("  - reports/backtesting/strategy_comparison.csv")
        logger.info("  - reports/business_report_summary.csv")
        logger.info("  - reports/business_report_detailed.csv")
        logger.info("  - reports/inventory_recommendations.csv")
        logger.info("  - reports/pricing_recommendations.csv")
        logger.info("  - reports/llm_insights.csv")
        logger.info("  - reports/report_charts/*.png")
        logger.info("  - reports/summary_statistics.json")
        return True
    else:
        logger.warning(f"\n⚠ Some scripts failed. {total_count - success_count} script(s) did not complete successfully.")
        return False


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.warning("\n⚠ Process interrupted by user (Ctrl+C)")
        sys.exit(130)
    except Exception as e:
        logger.error(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
