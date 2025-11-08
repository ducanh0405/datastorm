"""
Smoke Tests for DataStorm Pipeline
===================================
Quick validation tests to ensure pipeline runs end-to-end without errors.
Uses 1% sample data from data/poc_data/ for fast execution.
"""
import pytest
import pandas as pd
from pathlib import Path
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipelines._01_load_data import load_competition_data
from src.features.ws0_aggregation import prepare_master_dataframe
from src.features.ws2_timeseries_features import add_lag_rolling_features


@pytest.fixture
def poc_data_dir():
    """POC data directory for smoke tests."""
    return PROJECT_ROOT / 'data' / 'poc_data'


@pytest.fixture
def sample_transactions(poc_data_dir):
    """Load sample transaction data."""
    tx_path = poc_data_dir / 'transaction_data.csv'
    if not tx_path.exists():
        pytest.skip("POC data not found. Run scripts/create_sample_data.py first.")
    return pd.read_csv(tx_path)


@pytest.mark.smoke
def test_data_loader(poc_data_dir, monkeypatch):
    """Test that data loader can read POC data."""
    data = load_competition_data(poc_data_dir)
    
    assert isinstance(data, dict)
    assert 'transaction_data' in data
    assert len(data['transaction_data']) > 0


@pytest.mark.smoke
def test_ws0_aggregation(sample_transactions):
    """Test WS0: Aggregation & Master Grid Creation."""
    master_df = prepare_master_dataframe(sample_transactions)
    
    # Verify aggregation
    assert 'PRODUCT_ID' in master_df.columns
    assert 'STORE_ID' in master_df.columns
    assert 'WEEK_NO' in master_df.columns
    assert 'SALES_VALUE' in master_df.columns
    
    # Verify zero-filling (grid should have more rows than original transactions)
    assert len(master_df) >= len(sample_transactions.groupby(['PRODUCT_ID', 'STORE_ID', 'WEEK_NO']))
    
    # Verify sorting
    is_sorted = master_df.groupby(['PRODUCT_ID', 'STORE_ID'])['WEEK_NO'].apply(
        lambda x: (x.diff().dropna() >= 0).all()
    ).all()
    assert is_sorted, "master_df not properly sorted by WEEK_NO"


@pytest.mark.smoke
def test_ws2_timeseries_features(sample_transactions):
    """Test WS2: Leak-Safe Time-Series Features."""
    # First run WS0
    master_df = prepare_master_dataframe(sample_transactions)
    
    # Then run WS2
    enriched_df = add_lag_rolling_features(master_df)
    
    # Verify lag features exist
    assert 'sales_value_lag_1' in enriched_df.columns
    assert 'sales_value_lag_4' in enriched_df.columns
    
    # Verify rolling features exist
    assert 'rolling_mean_4_lag_1' in enriched_df.columns
    assert 'rolling_std_4_lag_1' in enriched_df.columns
    
    # Verify calendar features
    assert 'week_of_year' in enriched_df.columns
    assert 'month_proxy' in enriched_df.columns
    
    # Verify no leakage: lag_1 should always be NaN for first week of each product/store
    first_weeks = enriched_df.groupby(['PRODUCT_ID', 'STORE_ID']).head(1)
    assert first_weeks['sales_value_lag_1'].isna().all(), "Lag features leaked into first period!"


@pytest.mark.smoke
def test_time_based_split():
    """Test that time-based split logic works correctly."""
    # Create sample data with known time range
    df = pd.DataFrame({
        'WEEK_NO': list(range(1, 101)),
        'SALES_VALUE': list(range(100)),
        'feature1': list(range(100))
    })
    
    # 80th percentile cutoff
    cutoff = df['WEEK_NO'].quantile(0.8)
    
    train_mask = df['WEEK_NO'] < cutoff
    test_mask = df['WEEK_NO'] >= cutoff
    
    train = df[train_mask]
    test = df[test_mask]
    
    # Verify split ratio
    assert len(train) / len(df) == pytest.approx(0.8, abs=0.05)
    
    # Verify no time overlap
    assert train['WEEK_NO'].max() < test['WEEK_NO'].min(), "Time leakage: train/test overlap!"
    
    # Verify no shuffle (time ordering preserved)
    assert train['WEEK_NO'].is_monotonic_increasing
    assert test['WEEK_NO'].is_monotonic_increasing


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
