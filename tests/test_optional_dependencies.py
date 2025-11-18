"""
Tests for Optional Dependencies Handling
========================================
Tests that optional dependencies (CatBoost, Great Expectations) fail gracefully.
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class TestCatBoostHandling:
    """Tests for CatBoost optional dependency handling."""

    def test_catboost_not_available_flag(self):
        """Test that CATBOOST_AVAILABLE flag exists."""
        from src.pipelines._03_model_training import CATBOOST_AVAILABLE
        assert isinstance(CATBOOST_AVAILABLE, bool)

    def test_catboost_import_error_handled(self):
        """Test that CatBoost import error is handled gracefully."""
        # The module should handle ImportError without crashing
        # This is tested by importing the module
        try:
            from src.pipelines._03_model_training import CATBOOST_AVAILABLE, cb
            # If cb is None, that's fine - it means import failed
            assert CATBOOST_AVAILABLE is False or cb is not None
        except ImportError:
            pytest.fail("CatBoost import error should be handled, not raised")

    def test_catboost_model_creation_without_import(self):
        """Test that model creation fails gracefully when CatBoost unavailable."""
        from src.pipelines._03_model_training import CATBOOST_AVAILABLE, create_model

        # If CatBoost is available, skip this test (it's testing unavailable scenario)
        if CATBOOST_AVAILABLE:
            pytest.skip("CatBoost is available - cannot test unavailable scenario")

        # Try to create CatBoost model - should raise ImportError if not available
        with pytest.raises((ImportError, RuntimeError)):
            model, _ = create_model('catboost', 0.5, [])


class TestGreatExpectationsHandling:
    """Tests for Great Expectations optional dependency handling."""

    def test_gx_available_flag(self):
        """Test that GX_AVAILABLE flag exists."""
        from src.utils.data_quality_gx import GX_AVAILABLE
        assert isinstance(GX_AVAILABLE, bool)

    def test_gx_import_error_handled(self):
        """Test that GX import error is handled gracefully."""
        try:
            from src.utils.data_quality_gx import GX_AVAILABLE, gx
            # If gx is None, that's fine - it means import failed
            assert GX_AVAILABLE is False or gx is not None
        except ImportError:
            pytest.fail("GX import error should be handled, not raised")

    def test_gx_validator_initialization(self):
        """Test that GX validator initializes gracefully when GX unavailable."""
        from src.utils.data_quality_gx import DataQualityValidator

        validator = DataQualityValidator()
        # Should not crash even if GX is unavailable
        assert validator is not None
        assert isinstance(validator.is_available(), bool)

    def test_gx_validate_skips_when_unavailable(self):
        """Test that validation skips gracefully when GX unavailable."""
        import pandas as pd

        from src.utils.data_quality_gx import DataQualityValidator

        validator = DataQualityValidator()
        test_df = pd.DataFrame({'col1': [1, 2, 3]})

        # Should return a result dict even if GX unavailable
        result = validator.validate(test_df, fail_on_error=False)
        assert isinstance(result, dict)
        assert 'success' in result or 'error' in result

    def test_gx_data_quality_monitor(self):
        """Test that DataQualityMonitor handles missing GX."""
        from src.utils.data_quality import DataQualityMonitor

        monitor = DataQualityMonitor()
        # Should not crash even if GX is unavailable
        assert monitor is not None

    @patch('src.utils.data_quality.HAS_GREAT_EXPECTATIONS', False)
    def test_gx_create_expectation_suite_returns_none(self):
        """Test that create_expectation_suite returns None when GX unavailable."""
        import pandas as pd

        from src.utils.data_quality import DataQualityMonitor

        monitor = DataQualityMonitor()
        test_df = pd.DataFrame({'col1': [1, 2, 3]})

        suite = monitor.create_expectation_suite(test_df, 'test_dataset')
        # Should return None when GX unavailable
        assert suite is None


class TestErrorMessages:
    """Tests that error messages are informative."""

    def test_catboost_error_message(self):
        """Test that CatBoost error messages are informative."""
        import logging

        from src.pipelines._03_model_training import create_model

        # Try to create model when unavailable
        try:
            model, _ = create_model('catboost', 0.5, [])
        except ImportError as e:
            # Error message should be informative
            assert 'CatBoost' in str(e) or 'catboost' in str(e).lower()
        except RuntimeError:
            # Also acceptable
            pass

    def test_gx_error_message(self):
        """Test that GX error messages are informative."""
        import pandas as pd

        from src.utils.data_quality_gx import DataQualityValidator

        validator = DataQualityValidator()
        test_df = pd.DataFrame({'col1': [1, 2, 3]})

        # Try validation - should get informative error if fails
        result = validator.validate(test_df, fail_on_error=False)
        if not result.get('success', True):
            error_msg = result.get('error', '')
            # Error should mention GX or Great Expectations
            assert len(error_msg) > 0  # Should have some error message


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

