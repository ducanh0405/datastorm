"""
Validation Utilities
====================
Comprehensive data validation functions for pipeline quality assurance.
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Any

# Import centralized config
try:
    from src.config import VALIDATION_CONFIG, setup_logging
    setup_logging()
    logger = logging.getLogger(__name__)
except ImportError:
    # Fallback
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    VALIDATION_CONFIG = {
        'required_columns': ['PRODUCT_ID', 'STORE_ID', 'WEEK_NO', 'SALES_VALUE'],
        'data_ranges': {
            'SALES_VALUE': (0, None),
            'QUANTITY': (0, None),
            'WEEK_NO': (1, 104),
            'discount_pct': (0, 1),
        },
        'quality_thresholds': {
            'excellent': 90,
            'good': 75,
            'fair': 60,
        }
    }


def check_required_columns(df: pd.DataFrame, required_cols: List[str]) -> Dict[str, bool]:
    """
    Check if required columns exist in dataframe.
    
    Args:
        df: Dataframe to check
        required_cols: List of required column names
    
    Returns:
        Dictionary mapping column name to existence status
    """
    results = {}
    for col in required_cols:
        results[col] = col in df.columns
    return results


def validate_data_ranges(df: pd.DataFrame, column_ranges: Optional[Dict[str, tuple]] = None) -> Dict[str, Any]:
    """
    Validate that numeric columns are within expected ranges.

    Args:
        df: Dataframe to validate
        column_ranges: Optional dict mapping column name to (min, max) tuple

    Returns:
        Dictionary with validation results
    """
    results = {}

    if column_ranges is None:
        # Use centralized config or defaults
        column_ranges = VALIDATION_CONFIG.get('data_ranges', {
            'SALES_VALUE': (0, None),  # Non-negative
            'QUANTITY': (0, None),  # Non-negative
            'WEEK_NO': (1, 104),  # Typical week range
            'discount_pct': (0, 1),  # 0-100%
        })
    
    for col, (min_val, max_val) in column_ranges.items():
        if col not in df.columns:
            continue
        
        if df[col].dtype in [np.int64, np.float64, 'int64', 'float64']:
            if min_val is not None:
                below_min = (df[col] < min_val).sum()
                results[f'{col}_below_min'] = below_min
            
            if max_val is not None:
                above_max = (df[col] > max_val).sum()
                results[f'{col}_above_max'] = above_max
    
    return results


def check_feature_consistency(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Check consistency of derived features.
    
    Args:
        df: Dataframe to check
    
    Returns:
        Dictionary with consistency check results
    """
    results = {}
    
    # Check if base_price calculation is consistent
    if all(col in df.columns for col in ['SALES_VALUE', 'RETAIL_DISC', 'COUPON_DISC', 'base_price']):
        calculated_base = df['SALES_VALUE'] - (df['RETAIL_DISC'] + df['COUPON_DISC'])
        diff = (calculated_base - df['base_price']).abs()
        max_diff = diff.max()
        results['base_price_consistency'] = {
            'max_difference': float(max_diff) if not pd.isna(max_diff) else None,
            'is_consistent': (diff < 1e-6).all() if not diff.isna().all() else None
        }
    
    # Check if discount_pct is consistent
    if all(col in df.columns for col in ['total_discount', 'base_price', 'discount_pct']):
        calculated_pct = df['total_discount'] / (df['base_price'] + 1e-6)
        diff = (calculated_pct - df['discount_pct']).abs()
        max_diff = diff.max()
        results['discount_pct_consistency'] = {
            'max_difference': float(max_diff) if not pd.isna(max_diff) else None,
            'is_consistent': (diff < 1e-6).all() if not diff.isna().all() else None
        }
    
    return results


def comprehensive_validation(df: pd.DataFrame, verbose: bool = True) -> Dict[str, Any]:
    """
    Comprehensive data validation for pipeline quality assurance.
    
    Performs multiple validation checks:
    - Basic information (shape, dtypes)
    - Missing values
    - Duplicates
    - Required columns
    - Data ranges
    - Feature consistency
    
    Args:
        df: Dataframe to validate
        verbose: If True, log detailed information
    
    Returns:
        Dictionary with validation results and quality score
    """
    logging.info("[Validation] Starting comprehensive data validation...")
    validation_results = {}
    issues = []
    
    # 1. Basic information
    if verbose:
        logging.info("\n--- Basic Information ---")
    
    validation_results['shape'] = df.shape
    validation_results['memory_usage_mb'] = float(df.memory_usage(deep=True).sum() / 1024**2)
    validation_results['dtypes'] = df.dtypes.astype(str).to_dict()
    
    if verbose:
        logging.info(f"  Shape: {validation_results['shape']}")
        logging.info(f"  Memory usage: {validation_results['memory_usage_mb']:.2f} MB")
    
    # 2. Missing values
    if verbose:
        logging.info("\n--- Missing Values ---")
    
    missing_counts = df.isnull().sum()
    missing_pct = (missing_counts / len(df)) * 100
    validation_results['missing_values'] = {
        'counts': missing_counts[missing_counts > 0].to_dict(),
        'percentages': missing_pct[missing_pct > 0].to_dict()
    }
    
    high_missing = missing_pct[missing_pct > 50]
    if len(high_missing) > 0:
        issues.append(f"High missing values (>50%) in columns: {high_missing.index.tolist()}")
        if verbose:
            logging.warning(f"  WARNING: High missing values in: {high_missing.index.tolist()}")
    
    if verbose:
        total_missing = missing_counts.sum()
        logging.info(f"  Total missing values: {total_missing:,}")
        if total_missing > 0:
            logging.info(f"  Columns with missing values: {len(missing_counts[missing_counts > 0])}")
    
    # 3. Duplicates
    if verbose:
        logging.info("\n--- Duplicates ---")
    
    duplicate_count = df.duplicated().sum()
    validation_results['duplicates'] = int(duplicate_count)
    
    if duplicate_count > 0:
        issues.append(f"Found {duplicate_count} duplicate rows")
        if verbose:
            logging.warning(f"  WARNING: Found {duplicate_count} duplicate rows")
    else:
        if verbose:
            logging.info("  No duplicates found")
    
    # 4. Required columns (for forecasting pipeline)
    if verbose:
        logging.info("\n--- Required Columns ---")
    
    required_cols = ['PRODUCT_ID', 'STORE_ID', 'WEEK_NO', 'SALES_VALUE']
    col_check = check_required_columns(df, required_cols)
    validation_results['required_columns'] = col_check
    
    missing_cols = [col for col, exists in col_check.items() if not exists]
    if missing_cols:
        issues.append(f"Missing required columns: {missing_cols}")
        if verbose:
            logging.error(f"  ERROR: Missing required columns: {missing_cols}")
    else:
        if verbose:
            logging.info("  All required columns present")
    
    # 5. Data ranges
    if verbose:
        logging.info("\n--- Data Ranges ---")
    
    range_results = validate_data_ranges(df)
    validation_results['data_ranges'] = range_results
    
    if range_results:
        if verbose:
            for key, value in range_results.items():
                if isinstance(value, (int, float)) and value > 0:
                    logging.warning(f"  WARNING: {key} = {value}")
    
    # 6. Feature consistency
    if verbose:
        logging.info("\n--- Feature Consistency ---")
    
    consistency_results = check_feature_consistency(df)
    validation_results['feature_consistency'] = consistency_results
    
    if consistency_results:
        for feature, result in consistency_results.items():
            if isinstance(result, dict) and 'is_consistent' in result:
                if result['is_consistent'] is False:
                    issues.append(f"Inconsistent feature: {feature}")
                    if verbose:
                        logging.warning(f"  WARNING: Inconsistent {feature}")
    
    # 7. Calculate quality score
    score = 100
    score -= len(issues) * 10  # -10 points per issue
    score -= len(missing_cols) * 20  # -20 points per missing required column
    if duplicate_count > len(df) * 0.01:  # More than 1% duplicates
        score -= 15
    if high_missing is not None and len(high_missing) > 0:
        score -= 10
    
    validation_results['quality_score'] = max(0, score)
    validation_results['issues'] = issues
    validation_results['passed'] = len(issues) == 0 and len(missing_cols) == 0
    
    # 8. Summary
    if verbose:
        logging.info("\n--- Validation Summary ---")
        logging.info(f"  Quality Score: {validation_results['quality_score']}/100")
        if validation_results['quality_score'] >= 90:
            logging.info("  Status: EXCELLENT")
        elif validation_results['quality_score'] >= 75:
            logging.info("  Status: GOOD")
        elif validation_results['quality_score'] >= 60:
            logging.warning("  Status: FAIR")
        else:
            logging.error("  Status: POOR")
        
        if issues:
            logging.warning(f"  Issues found: {len(issues)}")
            for issue in issues:
                logging.warning(f"    - {issue}")
        else:
            logging.info("  No issues found")
    
    logging.info("[Validation] Comprehensive validation complete.")
    
    return validation_results