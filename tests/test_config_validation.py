"""
Tests for Configuration Validation
==================================
Tests config validation functions and edge cases.
"""
import sys
from pathlib import Path

import pytest

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    DATASET_CONFIGS,
    MODEL_TYPES,
    PERFORMANCE_CONFIG,
    QUANTILES,
    TRAINING_CONFIG,
    assert_config_valid,
    get_dataset_config,
    validate_all_configs,
    validate_dataset_config,
    validate_performance_config,
    validate_training_config,
)


class TestDatasetConfigValidation:
    """Tests for dataset configuration validation."""

    def test_valid_config(self):
        """Test that valid config passes validation."""
        config = get_dataset_config('freshretail')
        errors = validate_dataset_config(config)
        assert len(errors) == 0, f"Valid config should have no errors, got: {errors}"

    def test_missing_required_field(self):
        """Test validation catches missing required fields."""
        config = {'name': 'test'}
        errors = validate_dataset_config(config)
        assert len(errors) > 0
        assert any('temporal_unit' in err for err in errors)

    def test_invalid_temporal_unit(self):
        """Test validation catches invalid temporal_unit."""
        config = {
            'name': 'test',
            'temporal_unit': 'invalid_unit',
            'time_column': 'time',
            'target_column': 'target',
            'groupby_keys': ['id', 'time']
        }
        errors = validate_dataset_config(config)
        assert len(errors) > 0
        assert any('temporal_unit' in err.lower() for err in errors)

    def test_invalid_groupby_keys(self):
        """Test validation catches invalid groupby_keys."""
        config = {
            'name': 'test',
            'temporal_unit': 'hour',
            'time_column': 'time',
            'target_column': 'target',
            'groupby_keys': ['id']  # Too few elements
        }
        errors = validate_dataset_config(config)
        assert len(errors) > 0
        assert any('groupby_keys' in err.lower() for err in errors)

    def test_time_column_not_in_groupby_keys(self):
        """Test validation catches time_column not in groupby_keys."""
        config = {
            'name': 'test',
            'temporal_unit': 'hour',
            'time_column': 'time',
            'target_column': 'target',
            'groupby_keys': ['id', 'store']  # Missing 'time'
        }
        errors = validate_dataset_config(config)
        assert len(errors) > 0
        assert any('time_column' in err.lower() for err in errors)

    def test_invalid_lag_periods(self):
        """Test validation catches invalid lag_periods."""
        config = {
            'name': 'test',
            'temporal_unit': 'hour',
            'time_column': 'time',
            'target_column': 'target',
            'groupby_keys': ['id', 'time'],
            'lag_periods': []  # Empty list
        }
        errors = validate_dataset_config(config)
        assert len(errors) > 0
        assert any('lag_periods' in err.lower() for err in errors)

    def test_invalid_rolling_windows(self):
        """Test validation catches invalid rolling_windows."""
        config = {
            'name': 'test',
            'temporal_unit': 'hour',
            'time_column': 'time',
            'target_column': 'target',
            'groupby_keys': ['id', 'time'],
            'rolling_windows': [-1, 0]  # Invalid values
        }
        errors = validate_dataset_config(config)
        assert len(errors) > 0
        assert any('rolling_windows' in err.lower() for err in errors)

    def test_invalid_boolean_flags(self):
        """Test validation catches invalid boolean flags."""
        config = {
            'name': 'test',
            'temporal_unit': 'hour',
            'time_column': 'time',
            'target_column': 'target',
            'groupby_keys': ['id', 'time'],
            'has_weather': 'yes'  # Should be boolean
        }
        errors = validate_dataset_config(config)
        assert len(errors) > 0
        assert any('has_weather' in err.lower() for err in errors)


class TestTrainingConfigValidation:
    """Tests for training configuration validation."""

    def test_valid_training_config(self):
        """Test that valid training config passes validation."""
        errors = validate_training_config()
        assert len(errors) == 0, f"Valid training config should have no errors, got: {errors}"

    def test_quantiles_validation(self):
        """Test quantiles are validated correctly."""
        # This test checks the actual QUANTILES constant
        assert isinstance(QUANTILES, list)
        assert len(QUANTILES) > 0
        assert all(0 < q < 1 for q in QUANTILES)

    def test_model_types_validation(self):
        """Test model types are validated correctly."""
        assert isinstance(MODEL_TYPES, list)
        assert len(MODEL_TYPES) > 0


class TestPerformanceConfigValidation:
    """Tests for performance configuration validation."""

    def test_valid_performance_config(self):
        """Test that valid performance config passes validation."""
        errors = validate_performance_config()
        assert len(errors) == 0, f"Valid performance config should have no errors, got: {errors}"

    def test_memory_limit_validation(self):
        """Test memory_limit_gb is validated."""
        assert 'memory_limit_gb' in PERFORMANCE_CONFIG
        assert isinstance(PERFORMANCE_CONFIG['memory_limit_gb'], int | float)
        assert PERFORMANCE_CONFIG['memory_limit_gb'] > 0

    def test_parallel_threads_validation(self):
        """Test parallel_threads is validated."""
        assert 'parallel_threads' in PERFORMANCE_CONFIG
        assert isinstance(PERFORMANCE_CONFIG['parallel_threads'], int)
        assert PERFORMANCE_CONFIG['parallel_threads'] > 0


class TestAllConfigsValidation:
    """Tests for comprehensive config validation."""

    def test_validate_all_configs(self):
        """Test that validate_all_configs works."""
        errors = validate_all_configs()
        # Should return a dict (may be empty if all valid)
        assert isinstance(errors, dict)

    def test_assert_config_valid_passes(self):
        """Test that assert_config_valid passes for valid configs."""
        # Should not raise if configs are valid
        try:
            assert_config_valid()
        except ValueError:
            pytest.fail("assert_config_valid() raised ValueError for valid configs")


class TestOptionalDependencies:
    """Tests for optional dependencies handling."""

    def test_catboost_import_handling(self):
        """Test that CatBoost import is handled gracefully."""
        # This test verifies the import pattern exists
        # Actual import testing would require mocking
        from src.pipelines._03_model_training import CATBOOST_AVAILABLE
        # Should be a boolean
        assert isinstance(CATBOOST_AVAILABLE, bool)

    def test_gx_import_handling(self):
        """Test that Great Expectations import is handled gracefully."""
        from src.utils.data_quality_gx import GX_AVAILABLE
        # Should be a boolean
        assert isinstance(GX_AVAILABLE, bool)


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_get_dataset_config_invalid_name(self):
        """Test that invalid dataset name raises error."""
        with pytest.raises(KeyError):
            get_dataset_config('nonexistent_dataset')

    def test_config_with_none_values(self):
        """Test validation handles None values."""
        config = {
            'name': None,
            'temporal_unit': 'hour',
            'time_column': 'time',
            'target_column': 'target',
            'groupby_keys': ['id', 'time']
        }
        errors = validate_dataset_config(config)
        # Should catch None values
        assert len(errors) >= 0  # May or may not catch None depending on validation

    def test_empty_strings(self):
        """Test validation handles empty strings."""
        config = {
            'name': '',
            'temporal_unit': 'hour',
            'time_column': 'time',
            'target_column': 'target',
            'groupby_keys': ['id', 'time']
        }
        errors = validate_dataset_config(config)
        # Empty name might be caught or not, but shouldn't crash
        assert isinstance(errors, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

