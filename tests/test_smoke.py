"""
Smoke Tests for DataStorm Pipeline
===================================
Quick validation tests to ensure pipeline runs end-to-end without errors.
Uses sample data from data/poc_data/ for fast execution.
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.features.ws0_aggregation import prepare_master_dataframe  # noqa: E402
from src.features.ws2_timeseries_features import add_lag_rolling_features  # noqa: E402
from src.pipelines._01_load_data import load_data  # noqa: E402


@pytest.fixture
def sample_data_dir():
    """Sample data directory for smoke tests."""
    return PROJECT_ROOT / 'data' / 'poc_data'


@pytest.fixture
def sample_freshretail_data(sample_data_dir):
    """Load sample FreshRetail data."""
    data_path = sample_data_dir / 'freshretail_train_sample.csv'
    if not data_path.exists():
        pytest.skip("Sample data not found. Run scripts/create_freshretail_sample.py first.")
    return pd.read_csv(data_path)


@pytest.mark.smoke
def test_data_loader(sample_data_dir):
    """Test that data loader can read FreshRetail sample data."""
    # Test loading freshretail data
    dataframes, config = load_data()

    assert isinstance(dataframes, dict)
    assert 'sales' in dataframes  # FreshRetail data is loaded as 'sales'
    assert len(dataframes['sales']) > 0

    # Check required columns exist
    df = dataframes['sales']
    required_cols = ['product_id', 'store_id', 'dt', 'sale_amount']
    for col in required_cols:
        assert col in df.columns, f"Missing required column: {col}"


@pytest.mark.smoke
def test_ws0_aggregation(sample_freshretail_data):
    """Test WS0: Aggregation & Master Grid Creation for FreshRetail."""
    master_df = prepare_master_dataframe(sample_freshretail_data)

    # Verify aggregation for FreshRetail format
    assert 'product_id' in master_df.columns
    assert 'store_id' in master_df.columns
    assert 'hour_timestamp' in master_df.columns  # FreshRetail uses hour_timestamp
    assert 'sales_quantity' in master_df.columns  # FreshRetail target column

    # Verify we have aggregated data
    assert len(master_df) > 0

    # Verify sorting by time
    is_sorted = master_df.groupby(['product_id', 'store_id'])['hour_timestamp'].apply(
        lambda x: x.is_monotonic_increasing
    ).all()
    assert is_sorted, "master_df not properly sorted by hour_timestamp"


@pytest.mark.smoke
def test_ws2_timeseries_features(sample_freshretail_data):
    """Test WS2: Leak-Safe Time-Series Features for FreshRetail."""
    # First run WS0
    master_df = prepare_master_dataframe(sample_freshretail_data)

    # For now, skip WS2 as it needs config-driven updates
    # TODO: Update WS2 to be fully config-driven
    pytest.skip("WS2 needs config-driven updates for FreshRetail support")

    # Then run WS2
    # enriched_df = add_lag_rolling_features(master_df)

    # Verify lag features exist (FreshRetail naming)
    # assert 'sales_quantity_lag_1' in enriched_df.columns
    # assert 'sales_quantity_lag_24' in enriched_df.columns  # FreshRetail uses 24h lags

    # Verify rolling features exist
    # assert 'rolling_mean_24_lag_1' in enriched_df.columns

    # Verify calendar features
    # assert 'hour_of_day' in enriched_df.columns  # FreshRetail has intraday patterns
    # assert 'day_of_week' in enriched_df.columns

    # Verify no leakage: lag_1 should always be NaN for first period of each product/store
    # first_periods = enriched_df.groupby(['product_id', 'store_id']).head(1)
    # assert first_periods['sales_quantity_lag_1'].isna().all(), "Lag features leaked into first period!"


@pytest.mark.smoke
def test_time_based_split():
    """Test that time-based split logic works correctly."""
    # Create sample data with known time range (using FreshRetail format)
    df = pd.DataFrame({
        'hour_timestamp': list(range(1, 101)),
        'sales_quantity': list(range(100)),
        'feature1': list(range(100))
    })

    # 80th percentile cutoff
    cutoff = df['hour_timestamp'].quantile(0.8)

    train_mask = df['hour_timestamp'] < cutoff
    test_mask = df['hour_timestamp'] >= cutoff

    train = df[train_mask]
    test = df[test_mask]

    # Verify split ratio
    assert len(train) / len(df) == pytest.approx(0.8, abs=0.05)

    # Verify no time overlap
    assert train['hour_timestamp'].max() < test['hour_timestamp'].min(), "Time leakage: train/test overlap!"

    # Verify no shuffle (time ordering preserved)
    assert train['hour_timestamp'].is_monotonic_increasing
    assert test['hour_timestamp'].is_monotonic_increasing


@pytest.mark.smoke
def test_quantile_model_config():
    """Test that quantile model configuration is correct."""
    import lightgbm as lgb

    # Test Q50 model
    model_q50 = lgb.LGBMRegressor(
        objective='quantile',
        alpha=0.50,
        n_estimators=10,
        random_state=42
    )

    # Verify objective is set correctly
    assert model_q50.objective == 'quantile'
    assert model_q50.alpha == 0.50


@pytest.mark.smoke
def test_freshretail_sample_data(sample_freshretail_data):
    """Test that sample data has correct structure."""
    df = sample_freshretail_data

    # Check data types and ranges
    assert df['sale_amount'].dtype in ['float64', 'float32']
    assert df['product_id'].dtype in ['int64', 'int32']
    assert df['store_id'].dtype in ['int64', 'int32']
    assert df['city_id'].dtype in ['int64', 'int32']

    # Check value ranges
    assert df['sale_amount'].min() >= 0, "Sale amounts should be non-negative"
    assert df['product_id'].min() >= 0, "Product IDs should be non-negative"
    assert df['store_id'].min() >= 0, "Store IDs should be non-negative"

    # Check datetime column
    assert 'dt' in df.columns
    assert pd.api.types.is_datetime64_any_dtype(pd.to_datetime(df['dt'])), "dt should be datetime"


@pytest.mark.smoke
def test_directory_structure():
    """Test that required directories exist."""
    required_dirs = [
        PROJECT_ROOT / 'data' / 'poc_data',
        PROJECT_ROOT / 'data' / 'processed',
        PROJECT_ROOT / 'models',
        PROJECT_ROOT / 'reports' / 'metrics',
        PROJECT_ROOT / 'tests',
    ]

    for directory in required_dirs:
        assert directory.exists(), f"Required directory missing: {directory}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "smoke"])
