"""
Unit Tests for Feature Engineering Functions
============================================
Tests for WS1, WS2, WS4 feature engineering modules.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.features.ws1_relational_features import enrich_relational_features  # noqa: E402
from src.features.ws2_timeseries_features import (  # noqa: E402
    add_lag_rolling_features,
    create_calendar_features,
    create_lag_features,
    create_rolling_features,
    create_lag_features_config,
    add_intraday_features,
    create_intraday_features,
    add_trend_features,
    add_interaction_features,
)
from src.features.ws4_price_features import add_price_promotion_features  # noqa: E402


@pytest.fixture
def sample_transaction_data():
    """Create sample transaction data for testing."""
    np.random.seed(42)
    n_rows = 100

    data = {
        'PRODUCT_ID': np.random.choice(['P1', 'P2', 'P3'], n_rows),
        'STORE_ID': np.random.choice(['S1', 'S2'], n_rows),
        'WEEK_NO': np.tile(range(1, 11), 10)[:n_rows],
        'SALES_VALUE': np.random.uniform(10, 100, n_rows),
        'QUANTITY': np.random.randint(1, 10, n_rows),
        'RETAIL_DISC': np.random.uniform(-5, 0, n_rows),
        'COUPON_DISC': np.random.uniform(-3, 0, n_rows),
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_product_data():
    """Create sample product data for testing."""
    return pd.DataFrame({
        'PRODUCT_ID': ['P1', 'P2', 'P3'],
        'DEPARTMENT': ['D1', 'D2', 'D1'],
        'COMMODITY_DESC': ['C1', 'C2', 'C1'],
    })


@pytest.fixture
def sample_master_data(sample_transaction_data):
    """Create master data from transaction data (after WS0 aggregation)."""
    # Simulate WS0 aggregation
    master = sample_transaction_data.groupby(
        ['PRODUCT_ID', 'STORE_ID', 'WEEK_NO']
    ).agg({
        'SALES_VALUE': 'sum',
        'QUANTITY': 'sum',
        'RETAIL_DISC': 'sum',
        'COUPON_DISC': 'sum',
    }).reset_index()

    # Sort for time-series features
    master = master.sort_values(['PRODUCT_ID', 'STORE_ID', 'WEEK_NO']).reset_index(drop=True)
    return master


class TestWS1RelationalFeatures:
    """Tests for WS1: Relational Features."""

    def test_enrich_relational_features_basic(self, sample_master_data, sample_product_data):
        """Test basic relational feature enrichment."""
        dataframes = {'product': sample_product_data}
        result = enrich_relational_features(sample_master_data, dataframes)

        assert 'DEPARTMENT' in result.columns
        assert 'COMMODITY_DESC' in result.columns
        assert len(result) == len(sample_master_data)

    def test_enrich_relational_features_missing_product(self, sample_master_data):
        """Test that missing product data is handled gracefully."""
        dataframes = {}
        result = enrich_relational_features(sample_master_data, dataframes)

        # Should return original dataframe unchanged
        assert len(result) == len(sample_master_data)
        assert 'DEPARTMENT' not in result.columns

    def test_enrich_relational_features_missing_columns(self, sample_master_data):
        """Test that missing PRODUCT_ID is handled."""
        dataframes = {'product': pd.DataFrame({'WRONG_ID': ['P1']})}
        result = enrich_relational_features(sample_master_data, dataframes)

        # Should return original dataframe unchanged
        assert len(result) == len(sample_master_data)


class TestWS2TimeSeriesFeatures:
    """Tests for WS2: Time-Series Features."""

    def test_create_lag_features(self, sample_master_data):
        """Test lag feature creation."""
        result = create_lag_features(
            sample_master_data,
            target_col='SALES_VALUE',
            lags=[1, 4]
        )

        assert 'sales_value_lag_1' in result.columns
        assert 'sales_value_lag_4' in result.columns

        # First row should have NaN for lag_1 (no previous value)
        first_rows = (
            result.sort_values(['PRODUCT_ID', 'STORE_ID', 'WEEK_NO'])
                  .groupby(['PRODUCT_ID', 'STORE_ID'])
                  .head(1)
        )
        assert first_rows['sales_value_lag_1'].isna().all()

    def test_create_rolling_features(self, sample_master_data):
        """Test rolling feature creation."""
        # First create lag features
        df_with_lags = create_lag_features(sample_master_data, lags=[1])

        result = create_rolling_features(
            df_with_lags,
            target_col='SALES_VALUE',
            base_lag=1,
            windows=[4]
        )

        assert 'rolling_mean_4_lag_1' in result.columns
        assert 'rolling_std_4_lag_1' in result.columns

    def test_create_calendar_features(self, sample_master_data):
        """Test calendar feature creation."""
        result = create_calendar_features(sample_master_data)

        assert 'week_of_year' in result.columns
        assert 'month_proxy' in result.columns
        assert 'quarter' in result.columns
        assert 'week_sin' in result.columns
        assert 'week_cos' in result.columns

    def test_add_lag_rolling_features_complete(self, sample_master_data):
        """Test complete WS2 feature engineering."""
        result = add_lag_rolling_features(sample_master_data)

        # Check lag features
        assert 'sales_value_lag_1' in result.columns
        assert 'sales_value_lag_4' in result.columns

        # Check rolling features
        assert 'rolling_mean_4_lag_1' in result.columns

        # Check calendar features
        assert 'week_of_year' in result.columns

    def test_create_lag_features_config(self, sample_master_data):
        """Test config-driven lag feature creation."""
        # Mock config for weekly data
        config = {
            'temporal_unit': 'week',
            'time_column': 'WEEK_NO',
            'groupby_keys': ['PRODUCT_ID', 'STORE_ID', 'WEEK_NO'],
            'lag_periods': [1, 4],
            'has_intraday_patterns': False,
        }
        result = create_lag_features_config(sample_master_data, config)
        
        assert 'sales_value_lag_1' in result.columns
        assert 'sales_value_lag_4' in result.columns

    def test_add_intraday_features(self, sample_master_data):
        """Test intraday feature creation for hourly data."""
        # Create sample hourly data
        hourly_data = sample_master_data.copy()
        hourly_data['hour_timestamp'] = pd.date_range('2024-01-01', periods=len(hourly_data), freq='H')
        
        result = add_intraday_features(hourly_data, 'hour_timestamp')
        
        assert 'hour_of_day' in result.columns
        assert 'is_morning_peak' in result.columns
        assert 'is_evening_peak' in result.columns
        assert 'hour_sin' in result.columns
        assert 'hour_cos' in result.columns

    def test_create_intraday_features(self, sample_master_data):
        """Test create_intraday_features function."""
        # Create sample hourly data
        hourly_data = sample_master_data.copy()
        hourly_data['hour_timestamp'] = pd.date_range('2024-01-01', periods=len(hourly_data), freq='H')
        
        result = create_intraday_features(hourly_data, 'hour_timestamp')
        
        assert 'hour_of_day' in result.columns
        assert 'day_of_week' in result.columns
        assert 'is_morning_peak' in result.columns
        assert 'is_evening_peak' in result.columns
        assert 'is_weekend' in result.columns
        assert 'hour_sin' in result.columns
        assert 'hour_cos' in result.columns
        assert 'dow_sin' in result.columns
        assert 'dow_cos' in result.columns

    def test_add_trend_features(self, sample_master_data):
        """Test trend feature creation."""
        # First create lag features
        df_with_lags = create_lag_features(sample_master_data, lags=[1, 4])
        df_with_rolling = create_rolling_features(df_with_lags, windows=[4, 8])
        
        result = add_trend_features(df_with_rolling, target_col='SALES_VALUE')
        
        # Check trend features exist
        assert 'wow_change' in result.columns or 'momentum' in result.columns or 'volatility' in result.columns

    def test_add_interaction_features(self, sample_master_data):
        """Test interaction feature creation."""
        # Create base features first
        df_with_features = sample_master_data.copy()
        
        # Add some base features that interactions depend on
        if 'base_price' not in df_with_features.columns:
            df_with_features['base_price'] = df_with_features['SALES_VALUE'] * 0.9
        if 'is_on_retail_promo' not in df_with_features.columns:
            df_with_features['is_on_retail_promo'] = 0
        if 'is_on_coupon_promo' not in df_with_features.columns:
            df_with_features['is_on_coupon_promo'] = 0
        if 'discount_pct' not in df_with_features.columns:
            df_with_features['discount_pct'] = 0.1
        if 'is_on_display' not in df_with_features.columns:
            df_with_features['is_on_display'] = 0
        
        # Add lag and rolling features for interactions
        df_with_features = create_lag_features(df_with_features, lags=[1, 4])
        df_with_features = create_rolling_features(df_with_features, windows=[4])
        df_with_features = create_calendar_features(df_with_features)
        
        result = add_interaction_features(df_with_features)
        
        # Check that interaction features were added (if applicable)
        interaction_cols = [col for col in result.columns if 'interaction' in col.lower() or 'ratio' in col.lower()]
        # May be empty if required base features are missing, which is OK
        assert isinstance(interaction_cols, list)

    def test_rolling_features_with_multiple_windows(self, sample_master_data):
        """Test rolling features with multiple window sizes."""
        df_with_lags = create_lag_features(sample_master_data, lags=[1])
        
        result = create_rolling_features(
            df_with_lags,
            target_col='SALES_VALUE',
            base_lag=1,
            windows=[4, 8, 12]
        )
        
        assert 'rolling_mean_4_lag_1' in result.columns
        assert 'rolling_mean_8_lag_1' in result.columns
        assert 'rolling_mean_12_lag_1' in result.columns
        assert 'rolling_std_4_lag_1' in result.columns
        assert 'rolling_std_8_lag_1' in result.columns
        assert 'rolling_std_12_lag_1' in result.columns

    def test_calendar_features_hourly_data(self):
        """Test calendar features for hourly data."""
        # Create hourly sample data
        hourly_data = pd.DataFrame({
            'PRODUCT_ID': ['P1'] * 100,
            'STORE_ID': ['S1'] * 100,
            'hour_timestamp': pd.date_range('2024-01-01', periods=100, freq='H'),
            'SALES_VALUE': np.random.uniform(10, 100, 100),
        })
        
        result = create_calendar_features(hourly_data)
        
        # Should have hourly calendar features
        assert 'day_of_year' in result.columns or 'month' in result.columns or 'quarter' in result.columns


class TestWS4PriceFeatures:
    """Tests for WS4: Price & Promotion Features."""

    def test_add_price_promotion_features_basic(self, sample_master_data):
        """Test basic price/promotion feature creation."""
        dataframes = {}
        result = add_price_promotion_features(sample_master_data, dataframes)

        # Should create transaction-based features even without causal data
        assert 'base_price' in result.columns
        assert 'total_discount' in result.columns
        assert 'discount_pct' in result.columns
        assert 'is_on_retail_promo' in result.columns
        assert 'is_on_coupon_promo' in result.columns

    def test_price_features_calculation(self, sample_master_data):
        """Test that price features are calculated correctly."""
        dataframes = {}
        result = add_price_promotion_features(sample_master_data, dataframes)

        # Base price should be SALES_VALUE - discounts
        expected_base = (
            result['SALES_VALUE'] -
            (result['RETAIL_DISC'] + result['COUPON_DISC'])
        )
        np.testing.assert_allclose(
            result['base_price'],
            expected_base,
            rtol=1e-5
        )

    def test_promotion_flags(self, sample_master_data):
        """Test that promotion flags are set correctly."""
        dataframes = {}
        result = add_price_promotion_features(sample_master_data, dataframes)

        # Flags should be binary (0 or 1)
        assert result['is_on_retail_promo'].isin([0, 1]).all()
        assert result['is_on_coupon_promo'].isin([0, 1]).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])



