#!/usr/bin/env python3
"""
Great Expectations Integration Module
======================================

Utility functions for integrating GX validation into SmartGrocy pipeline.
Provides simple API for data quality checks at any pipeline stage.

Usage:
    from src.utils.data_quality_gx import validate_dataframe, DataQualityValidator

    # Quick validation
    result = validate_dataframe(df, "master_feature_table")
    if not result['success']:
        raise ValueError("Data quality check failed")

    # Advanced validation
    validator = DataQualityValidator()
    result = validator.validate(
        df,
        suite_name="master_feature_table_suite",
        fail_on_error=True
    )

Author: SmartGrocy Team
Date: 2025-11-15
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd
from datetime import datetime

try:
    import great_expectations as gx
    GX_AVAILABLE = True
except ImportError as e:
    GX_AVAILABLE = False
    gx = None
    _gx_import_error = e

logger = logging.getLogger(__name__)

# Log GX availability status when module is imported
if not GX_AVAILABLE:
    logger.warning(
        f"Great Expectations not available (ImportError: {_gx_import_error}). "
        "GX validation will be disabled. Install with: pip install great-expectations"
    )

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
GX_ROOT = PROJECT_ROOT / "great_expectations"


class DataQualityValidator:
    """
    Wrapper class for Great Expectations validation in pipeline.
    
    Provides simplified API for data quality checks with automatic
    context management and error handling.
    """
    
    def __init__(self, gx_root: Optional[Path] = None):
        """
        Initialize validator.
        
        Args:
            gx_root: Path to GX root directory (default: auto-detect)
        """
        self.gx_root = gx_root or GX_ROOT
        self.context = None
        self._initialize_context()
    
    def _initialize_context(self):
        """Initialize GX context if available"""
        if not GX_AVAILABLE or gx is None:
            logger.warning("Great Expectations not available. Validation disabled.")
            self.context = None
            return
        
        if not self.gx_root.exists():
            logger.warning(f"GX not setup at {self.gx_root}. Run setup_great_expectations.py")
            self.context = None
            return
        
        try:
            self.context = gx.get_context(context_root_dir=str(self.gx_root))
            logger.debug("GX context initialized")
        except Exception as e:
            logger.error(f"Failed to initialize GX context: {e}", exc_info=True)
            self.context = None
    
    def is_available(self) -> bool:
        """Check if GX validation is available"""
        return GX_AVAILABLE and gx is not None and self.context is not None
    
    def validate(
        self,
        df: pd.DataFrame,
        suite_name: str = "master_feature_table_suite",
        asset_name: str = "validation_data",
        fail_on_error: bool = False,
        return_detailed: bool = False
    ) -> Dict[str, Any]:
        """
        Validate dataframe against expectation suite.
        
        Args:
            df: Dataframe to validate
            suite_name: Name of expectation suite
            asset_name: Name for this validation run
            fail_on_error: Raise exception if validation fails
            return_detailed: Include detailed failure information
        
        Returns:
            Dictionary with validation results:
            {
                'success': bool,
                'statistics': dict,
                'failed_expectations': list (if return_detailed=True),
                'timestamp': str,
                'data_shape': tuple
            }
        """
        if not self.is_available():
            logger.warning("GX validation skipped (not available)")
            return {
                'success': True,  # Don't block pipeline
                'statistics': {},
                'message': 'GX not available - validation skipped',
                'timestamp': datetime.now().isoformat(),
                'data_shape': df.shape
            }
        
        logger.info(f"Running GX validation on {asset_name}: {df.shape}")
        
        try:
            if not GX_AVAILABLE or gx is None:
                raise ImportError("Great Expectations is not available")
            
            if self.context is None:
                raise RuntimeError("GX context is not initialized")
            
            # Create batch request
            batch_request = {
                "datasource_name": "master_feature_datasource",
                "data_connector_name": "default_runtime_data_connector",
                "data_asset_name": asset_name,
                "runtime_parameters": {"batch_data": df},
                "batch_identifiers": {
                    "default_identifier_name": f"{asset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                }
            }
            
            # Run checkpoint
            result = self.context.run_checkpoint(
                checkpoint_name="master_feature_checkpoint",
                batch_request=batch_request,
                expectation_suite_name=suite_name
            )
            
            # Parse results
            success = result["success"]
            run_results = result.get("run_results", {})
            
            # Extract statistics
            statistics = {}
            failed_expectations = []
            
            for run_id, run_result in run_results.items():
                validation_result = run_result.get("validation_result", {})
                statistics = validation_result.get("statistics", {})
                
                if return_detailed:
                    results = validation_result.get("results", [])
                    for exp_result in results:
                        if not exp_result.get("success", True):
                            failed_expectations.append({
                                "expectation": exp_result.get("expectation_config", {}).get("expectation_type"),
                                "column": exp_result.get("expectation_config", {}).get("kwargs", {}).get("column"),
                                "result": exp_result.get("result", {})
                            })
            
            # Log results
            success_rate = statistics.get('success_percent', 0)
            if success:
                logger.info(f"✅ Validation passed: {success_rate:.1f}% success rate")
            else:
                logger.warning(f"⚠️ Validation failed: {success_rate:.1f}% success rate")
                logger.warning(f"   {statistics.get('unsuccessful_expectations', 0)} expectations failed")
            
            # Prepare return dict
            validation_result = {
                'success': success,
                'statistics': statistics,
                'timestamp': datetime.now().isoformat(),
                'data_shape': df.shape
            }
            
            if return_detailed:
                validation_result['failed_expectations'] = failed_expectations
            
            # Raise exception if requested
            if fail_on_error and not success:
                error_msg = f"Data quality validation failed: {success_rate:.1f}% success rate"
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            return validation_result
            
        except ImportError as e:
            error_msg = f"Great Expectations not available: {e}"
            logger.error(error_msg)
            if fail_on_error:
                raise ImportError(error_msg) from e
            return {
                'success': False,
                'error': error_msg,
                'timestamp': datetime.now().isoformat(),
                'data_shape': df.shape
            }
        except Exception as e:
            error_msg = f"GX validation error: {e}"
            logger.error(error_msg, exc_info=True)
            if fail_on_error:
                raise RuntimeError(error_msg) from e
            return {
                'success': False,
                'error': error_msg,
                'timestamp': datetime.now().isoformat(),
                'data_shape': df.shape
            }
    
    def get_quality_score(self, validation_result: Dict[str, Any]) -> float:
        """
        Calculate quality score from validation result.
        
        Args:
            validation_result: Result from validate()
        
        Returns:
            Quality score (0-100)
        """
        if not validation_result.get('success'):
            stats = validation_result.get('statistics', {})
            return stats.get('success_percent', 0)
        return 100.0


# Convenience function for quick validation
def validate_dataframe(
    df: pd.DataFrame,
    asset_name: str = "data",
    fail_on_error: bool = False
) -> Dict[str, Any]:
    """
    Quick validation function for pipeline use.
    
    Args:
        df: Dataframe to validate
        asset_name: Name for logging
        fail_on_error: Raise exception on failure
    
    Returns:
        Validation result dictionary
    
    Example:
        >>> result = validate_dataframe(master_df, "master_table", fail_on_error=True)
        >>> if result['success']:
        >>>     print("Data quality check passed!")
    """
    validator = DataQualityValidator()
    return validator.validate(
        df,
        asset_name=asset_name,
        fail_on_error=fail_on_error
    )


# Module-level validator instance for reuse
_default_validator = None

def get_validator() -> DataQualityValidator:
    """Get or create default validator instance"""
    global _default_validator
    if _default_validator is None:
        _default_validator = DataQualityValidator()
    return _default_validator


if __name__ == "__main__":
    # Test validation
    print("Testing Data Quality Validator...")
    
    validator = DataQualityValidator()
    print(f"GX Available: {validator.is_available()}")
    
    if validator.is_available():
        # Create test dataframe
        test_df = pd.DataFrame({
            'sales_lag_1': [1, 2, 3, 4, 5],
            'week_of_year': [1, 2, 3, 4, 5]
        })
        
        print("\nRunning test validation...")
        result = validate_dataframe(test_df, "test_data", fail_on_error=False)
        print(f"Result: {result}")
    else:
        print("\nGX not available. Run: python scripts/setup_great_expectations.py")
