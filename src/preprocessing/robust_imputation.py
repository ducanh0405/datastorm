#!/usr/bin/env python3
"""
Robust Data Imputation Module
==============================
Advanced missing value handling for time series forecasting.

Strategies:
1. Forward/Backward fill for lag features
2. Rolling median for volatility features
3. Seasonal mean for periodic patterns
4. KNN imputation for complex missing patterns
5. Proxy features for high missing rate columns
6. Missing flags for model awareness

Author: SmartGrocy Team
Date: 2025-11-18
"""

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ImputationConfig:
    """Configuration for imputation strategies."""

    # Missing rate thresholds
    drop_threshold: float = 0.7  # Drop if >70% missing
    proxy_threshold: float = 0.3  # Create proxy if >30% missing

    # Imputation methods by feature type
    lag_method: str = 'ffill_bfill'  # Forward then backward fill
    volatility_method: str = 'rolling_median'  # Rolling window median
    change_method: str = 'zero'  # Fill with 0 (no change)

    # Rolling window sizes
    rolling_window: int = 7  # 7 days for rolling stats

    # Create missing flags
    add_missing_flags: bool = True

    # Validation after imputation
    validate_after: bool = True


class RobustImputer:
    """
    Robust imputation for time series features.

    Handles:
    - Lag features (sales_quantity_lag_N)
    - Volatility features (rolling_std_N)
    - Change features (wow_change, price_change)
    - Rolling means (rolling_mean_N)
    """

    def __init__(self, config: ImputationConfig | None = None):
        self.config = config or ImputationConfig()
        self.imputation_report = {}

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply robust imputation to dataframe.

        Args:
            df: Input dataframe with missing values

        Returns:
            Imputed dataframe with validation report
        """
        logger.info("\n" + "="*70)
        logger.info("ROBUST IMPUTATION PIPELINE")
        logger.info("="*70)

        df_clean = df.copy()

        # Step 1: Analyze missing patterns
        missing_report = self._analyze_missing(df_clean)
        logger.info("\nMissing value analysis:")
        for col, rate in missing_report.items():
            if rate > 0:
                logger.info(f"  {col}: {rate:.1%} missing")

        # Step 2: Drop columns with too much missing
        df_clean = self._drop_high_missing_cols(df_clean, missing_report)

        # Step 3: Create proxy features for moderate missing
        df_clean = self._create_proxy_features(df_clean, missing_report)

        # Step 4: Impute by feature type
        df_clean = self._impute_lag_features(df_clean)
        df_clean = self._impute_volatility_features(df_clean)
        df_clean = self._impute_change_features(df_clean)
        df_clean = self._impute_rolling_features(df_clean)

        # Step 5: Add missing flags
        if self.config.add_missing_flags:
            df_clean = self._add_missing_flags(df, df_clean)

        # Step 6: Final cleanup
        df_clean = self._final_cleanup(df_clean)

        # Step 7: Validation
        if self.config.validate_after:
            validation_report = self._validate_imputation(df, df_clean)
            self._log_validation_report(validation_report)

        logger.info("\n" + "="*70)
        logger.info("✅ IMPUTATION COMPLETE")
        logger.info("="*70)

        return df_clean

    def _analyze_missing(self, df: pd.DataFrame) -> dict[str, float]:
        """Analyze missing value patterns."""
        missing_rates = {}
        for col in df.columns:
            if df[col].dtype in ['float64', 'int64']:
                missing_rate = df[col].isna().mean()
                if missing_rate > 0:
                    missing_rates[col] = missing_rate
        return missing_rates

    def _drop_high_missing_cols(self, df: pd.DataFrame, missing_report: dict) -> pd.DataFrame:
        """Drop columns with excessive missing values."""
        cols_to_drop = [
            col for col, rate in missing_report.items()
            if rate > self.config.drop_threshold
        ]

        if cols_to_drop:
            logger.warning(f"\nDropping {len(cols_to_drop)} columns with >{self.config.drop_threshold:.0%} missing:")
            for col in cols_to_drop:
                logger.warning(f"  - {col} ({missing_report[col]:.1%})")
            df = df.drop(columns=cols_to_drop)

        return df

    def _create_proxy_features(self, df: pd.DataFrame, missing_report: dict) -> pd.DataFrame:
        """Create proxy features for columns with moderate missing."""
        for col, rate in missing_report.items():
            if self.config.proxy_threshold < rate <= self.config.drop_threshold:
                if col in df.columns:
                    logger.info(f"\nCreating proxy for {col} ({rate:.1%} missing)")

                    # Strategy: interpolate + rolling mean
                    proxy_col = f"{col}_proxy"
                    df[proxy_col] = df[col].interpolate(
                        method='linear',
                        limit_direction='both'
                    )

                    # If still has missing, use rolling mean
                    if df[proxy_col].isna().any():
                        df[proxy_col] = df[proxy_col].fillna(
                            df[proxy_col].rolling(
                                window=self.config.rolling_window,
                                min_periods=1
                            ).mean()
                        )

        return df

    def _impute_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Impute lag features using forward/backward fill."""
        lag_cols = [col for col in df.columns if 'lag_' in col.lower()]

        if lag_cols:
            logger.info(f"\nImputing {len(lag_cols)} lag features (ffill + bfill)...")
            for col in lag_cols:
                if df[col].isna().any():
                    # Forward fill first (most recent value)
                    df[col] = df[col].fillna(method='ffill')
                    # Then backward fill for leading NaNs
                    df[col] = df[col].fillna(method='bfill')
                    # If still missing, use 0
                    df[col] = df[col].fillna(0)

        return df

    def _impute_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Impute volatility features using rolling median."""
        vol_cols = [
            col for col in df.columns
            if 'std' in col.lower() or 'volatility' in col.lower()
        ]

        if vol_cols:
            logger.info(f"\nImputing {len(vol_cols)} volatility features (rolling median)...")
            for col in vol_cols:
                if df[col].isna().any():
                    # Use rolling median for volatility
                    df[col] = df[col].fillna(
                        df[col].rolling(
                            window=self.config.rolling_window,
                            min_periods=1
                        ).median()
                    )
                    # Fallback to column median
                    df[col] = df[col].fillna(df[col].median())

        return df

    def _impute_change_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Impute change features (wow_change, price_change) with 0."""
        change_cols = [
            col for col in df.columns
            if 'change' in col.lower() or 'diff' in col.lower()
        ]

        if change_cols:
            logger.info(f"\nImputing {len(change_cols)} change features (zero fill)...")
            for col in change_cols:
                if df[col].isna().any():
                    # Missing change = no change = 0
                    df[col] = df[col].fillna(0)

        return df

    def _impute_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Impute rolling mean features."""
        rolling_cols = [
            col for col in df.columns
            if 'rolling_mean' in col.lower()
        ]

        if rolling_cols:
            logger.info(f"\nImputing {len(rolling_cols)} rolling features (interpolate)...")
            for col in rolling_cols:
                if df[col].isna().any():
                    # Interpolate rolling means
                    df[col] = df[col].interpolate(
                        method='linear',
                        limit_direction='both'
                    )
                    # Fallback to column mean
                    df[col] = df[col].fillna(df[col].mean())

        return df

    def _add_missing_flags(self, df_original: pd.DataFrame, df_clean: pd.DataFrame) -> pd.DataFrame:
        """Add binary flags indicating which values were originally missing."""
        logger.info("\nAdding missing value flags...")

        numeric_cols = df_original.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df_original[col].isna().any():
                flag_col = f"{col}_was_missing"
                df_clean[flag_col] = df_original[col].isna().astype(int)

        return df_clean

    def _final_cleanup(self, df: pd.DataFrame) -> pd.DataFrame:
        """Final cleanup: remove any remaining NaNs."""
        # Check for remaining NaNs
        remaining_missing = df.isna().sum()
        remaining_missing = remaining_missing[remaining_missing > 0]

        if len(remaining_missing) > 0:
            logger.warning("\nRemaining missing values (filling with 0):")
            for col, count in remaining_missing.items():
                logger.warning(f"  {col}: {count} rows")
            df = df.fillna(0)

        return df

    def _validate_imputation(self, df_before: pd.DataFrame, df_after: pd.DataFrame) -> dict:
        """Validate imputation results."""
        report = {
            'rows_before': len(df_before),
            'rows_after': len(df_after),
            'cols_before': len(df_before.columns),
            'cols_after': len(df_after.columns),
            'missing_before': df_before.isna().sum().sum(),
            'missing_after': df_after.isna().sum().sum(),
            'missing_rate_before': df_before.isna().mean().mean(),
            'missing_rate_after': df_after.isna().mean().mean()
        }

        report['improvement'] = (
            report['missing_rate_before'] - report['missing_rate_after']
        ) / report['missing_rate_before'] if report['missing_rate_before'] > 0 else 0

        return report

    def _log_validation_report(self, report: dict):
        """Log validation report."""
        logger.info("\n" + "="*70)
        logger.info("IMPUTATION VALIDATION")
        logger.info("="*70)
        logger.info(f"Rows: {report['rows_before']} → {report['rows_after']}")
        logger.info(f"Columns: {report['cols_before']} → {report['cols_after']}")
        logger.info(f"Missing values: {report['missing_before']:,} → {report['missing_after']:,}")
        logger.info(f"Missing rate: {report['missing_rate_before']:.2%} → {report['missing_rate_after']:.2%}")
        logger.info(f"Improvement: {report['improvement']:.1%}")

        if report['missing_after'] == 0:
            logger.info("✅ All missing values resolved!")
        elif report['missing_after'] < report['missing_before'] * 0.1:
            logger.info("✅ Missing values reduced by >90%")
        else:
            logger.warning("⚠️  Some missing values remain")


def apply_robust_imputation(
    df: pd.DataFrame,
    config: ImputationConfig | None = None
) -> pd.DataFrame:
    """
    Convenience function to apply robust imputation.

    Usage:
        df_clean = apply_robust_imputation(df)
    """
    imputer = RobustImputer(config)
    return imputer.fit_transform(df)


if __name__ == "__main__":
    # Example usage
    import logging
    logging.basicConfig(level=logging.INFO)

    # Create sample data with missing values
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=100, freq='D')

    df = pd.DataFrame({
        'date': dates,
        'sales_quantity_lag_1': np.random.randint(50, 150, 100).astype(float),
        'sales_quantity_lag_7': np.random.randint(50, 150, 100).astype(float),
        'rolling_mean_7': np.random.uniform(80, 120, 100),
        'rolling_std_7': np.random.uniform(5, 20, 100),
        'wow_change': np.random.uniform(-0.2, 0.2, 100),
        'price': np.random.uniform(1000, 5000, 100)
    })

    # Introduce missing values
    for col in df.columns[1:]:
        mask = np.random.random(100) < 0.15  # 15% missing
        df.loc[mask, col] = np.nan

    # Add column with 80% missing (should be dropped)
    df['bad_feature'] = np.random.uniform(0, 1, 100)
    df.loc[np.random.random(100) < 0.8, 'bad_feature'] = np.nan

    # Add column with 40% missing (should get proxy)
    df['unstable_feature'] = np.random.uniform(0, 1, 100)
    df.loc[np.random.random(100) < 0.4, 'unstable_feature'] = np.nan

    print("\n" + "="*70)
    print("BEFORE IMPUTATION")
    print("="*70)
    print(df.isna().sum())
    print(f"\nTotal missing: {df.isna().sum().sum()}")

    # Apply imputation
    df_clean = apply_robust_imputation(df)

    print("\n" + "="*70)
    print("AFTER IMPUTATION")
    print("="*70)
    print(df_clean.isna().sum())
    print(f"\nTotal missing: {df_clean.isna().sum().sum()}")
    print(f"\nNew columns: {len(df_clean.columns)} (vs {len(df.columns)} before)")
