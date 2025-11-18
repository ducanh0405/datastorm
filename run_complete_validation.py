#!/usr/bin/env python3
"""
Complete Validation & Testing Suite
===================================
Run all validation and testing for SmartGrocy project.

Phases:
1. Module 4 validation tests
2. Report metrics validation
3. Generate summary statistics
4. Integration tests

Author: SmartGrocy Team
Date: 2025-11-18
"""

import sys
import logging
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import setup_logging
setup_logging()

logger = logging.getLogger(__name__)


def run_command(cmd: list, description: str) -> bool:
    """Run command and return success status."""
    logger.info(f"\n{'='*70}")
    logger.info(f"RUNNING: {description}")
    logger.info(f"{'='*70}")
    logger.info(f"Command: {' '.join(cmd)}\n")
    
    try:
        result = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            capture_output=False,
            text=True
        )
        
        if result.returncode == 0:
            logger.info(f"[PASS] {description} - PASSED")
            return True
        else:
            logger.error(f"[FAIL] {description} - FAILED (exit code: {result.returncode})")
            return False
            
    except Exception as e:
        logger.error(f"[ERROR] {description} - ERROR: {e}")
        return False


def main():
    """Run complete validation suite."""
    
    logger.info("\n" + "#"*70)
    logger.info("# SMARTGROCY COMPLETE VALIDATION SUITE")
    logger.info("#"*70 + "\n")
    
    results = []
    
    # Phase 1: Module 4 Validation Tests
    if Path('tests/test_module4_validation.py').exists():
        success = run_command(
            [sys.executable, '-m', 'pytest', 'tests/test_module4_validation.py', '-v'],
            "Module 4 Validation Tests"
        )
        results.append(('Module 4 Tests', success))
    else:
        logger.warning("Module 4 tests not found")
    
    # Phase 2: Report Metrics Validation
    if Path('scripts/validate_report_metrics.py').exists():
        success = run_command(
            [sys.executable, 'scripts/validate_report_metrics.py'],
            "Report Metrics Validation"
        )
        results.append(('Report Metrics', success))
    else:
        logger.warning("Report validator not found")
    
    # Phase 3: Summary Statistics
    if Path('scripts/generate_summary_statistics.py').exists():
        success = run_command(
            [sys.executable, 'scripts/generate_summary_statistics.py'],
            "Summary Statistics Generation"
        )
        results.append(('Summary Statistics', success))
    else:
        logger.warning("Summary generator not found")
    
    # Phase 4: MetricsValidator Tests
    success = run_command(
        [sys.executable, 'src/modules/metrics_validator.py'],
        "MetricsValidator Self-Tests"
    )
    results.append(('MetricsValidator', success))
    
    # Phase 5: Integration Test
    if Path('src/modules/integrated_insights.py').exists():
        success = run_command(
            [sys.executable, 'src/modules/integrated_insights.py'],
            "Integrated Insights Test"
        )
        results.append(('Integrated Insights', success))
    
    # Final Summary
    logger.info("\n" + "#"*70)
    logger.info("# VALIDATION SUMMARY")
    logger.info("#"*70 + "\n")
    
    for name, success in results:
        status = "[PASS]" if success else "[FAIL]"
        logger.info(f"  {name:30s}: {status}")
    
    total = len(results)
    passed = sum(1 for _, s in results if s)
    
    logger.info(f"\n{'='*70}")
    logger.info(f"TOTAL: {passed}/{total} passed ({passed/total*100:.0f}%)")
    logger.info(f"{'='*70}\n")
    
    # Exit code
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.warning("\n⚠️ Interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
