"""
Feature Selection Utilities
===========================
Automatic feature selection based on importance and correlation analysis.

Functions:
- select_important_features: Select features based on model importance
- remove_correlated_features: Remove highly correlated features
- get_optimal_features: Combined feature selection pipeline
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Optional imports for ML models
try:
    import lightgbm as lgb
    from sklearn.ensemble import RandomForestRegressor
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False

try:
    from src.config import setup_logging
    setup_logging()
    logger = logging.getLogger(__name__)
except ImportError:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)


def select_important_features(
    df: pd.DataFrame,
    target_col: str = 'SALES_VALUE',
    features: list[str] | None = None,
    importance_threshold: float = 0.01,
    method: str = 'lightgbm',
    max_features: int | None = None
) -> list[str]:
    """
    Select important features based on model feature importance.

    Args:
        df: DataFrame with features and target
        target_col: Target column name
        features: List of feature columns to consider (if None, uses all numeric)
        importance_threshold: Minimum importance score (0-1)
        method: Importance method ('lightgbm' or 'random_forest')
        max_features: Maximum number of features to select

    Returns:
        List of selected feature names
    """
    if features is None:
        # Use all numeric columns except target
        features = df.select_dtypes(include=[np.number]).columns.tolist()
        if target_col in features:
            features.remove(target_col)

    logger.info(f"Selecting important features from {len(features)} candidates using {method}")

    # Prepare data
    X = df[features].fillna(0)  # Fill NaN with 0 for importance calculation
    y = df[target_col]

    # Remove constant features
    constant_features = [col for col in features if X[col].std() == 0]
    if constant_features:
        logger.warning(f"Removing {len(constant_features)} constant features: {constant_features}")
        features = [col for col in features if col not in constant_features]
        X = X[features]

    if len(features) == 0:
        logger.error("No valid features remaining after filtering")
        return []

    # Calculate feature importance
    if method == 'lightgbm' and LGB_AVAILABLE:
        importance_scores = _calculate_lgb_importance(X, y)
    elif method == 'random_forest':
        importance_scores = _calculate_rf_importance(X, y)
    else:
        logger.warning(f"Method {method} not available, using correlation-based selection")
        importance_scores = _calculate_correlation_importance(X, y)

    # Create importance dataframe
    importance_df = pd.DataFrame({
        'feature': features,
        'importance': [importance_scores.get(col, 0) for col in features]
    }).sort_values('importance', ascending=False)

    # Select features above threshold
    selected = importance_df[importance_df['importance'] >= importance_threshold]['feature'].tolist()

    # Limit to max_features if specified
    if max_features and len(selected) > max_features:
        selected = selected[:max_features]
        logger.info(f"Limited to top {max_features} features")

    logger.info(f"Selected {len(selected)}/{len(features)} features with importance >= {importance_threshold}")
    if selected:
        logger.info(f"Top 5 features: {selected[:5]}")

    return selected


def remove_correlated_features(
    df: pd.DataFrame,
    features: list[str],
    correlation_threshold: float = 0.95,
    method: str = 'spearman'
) -> list[str]:
    """
    Remove highly correlated features to reduce multicollinearity.

    Args:
        df: DataFrame with features
        features: List of feature names to check
        correlation_threshold: Correlation threshold for removal
        method: Correlation method ('pearson', 'spearman', 'kendall')

    Returns:
        List of features after removing highly correlated ones
    """
    logger.info(f"Removing correlated features (threshold={correlation_threshold}) from {len(features)} features")

    if len(features) <= 1:
        return features

    # Calculate correlation matrix
    corr_matrix = df[features].corr(method=method)

    # Find highly correlated pairs
    to_remove = set()
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) >= correlation_threshold:
                col_i = corr_matrix.columns[i]
                col_j = corr_matrix.columns[j]

                # Keep the feature with higher variance (more informative)
                var_i = df[col_i].var()
                var_j = df[col_j].var()

                if var_i >= var_j:
                    to_remove.add(col_j)
                else:
                    to_remove.add(col_i)

    selected = [col for col in features if col not in to_remove]

    logger.info(f"Removed {len(to_remove)} correlated features, kept {len(selected)}")
    if to_remove:
        logger.info(f"Removed features: {list(to_remove)}")

    return selected


def get_optimal_features(
    df: pd.DataFrame,
    target_col: str = 'SALES_VALUE',
    importance_threshold: float = 0.005,
    correlation_threshold: float = 0.95,
    max_features: int | None = None,
    save_report: bool = True,
    output_path: Path | None = None
) -> dict[str, Any]:
    """
    Complete feature selection pipeline: importance + correlation filtering.

    Args:
        df: DataFrame with features and target
        target_col: Target column name
        importance_threshold: Minimum importance for selection
        correlation_threshold: Correlation threshold for removal
        max_features: Maximum number of features to select
        save_report: Whether to save selection report
        output_path: Path to save report (if None, uses default)

    Returns:
        Dictionary with selected features and metadata
    """
    logger.info("Starting optimal feature selection pipeline")

    # Step 1: Initial feature importance selection
    initial_features = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in initial_features:
        initial_features.remove(target_col)

    logger.info(f"Initial features: {len(initial_features)}")

    important_features = select_important_features(
        df=df,
        target_col=target_col,
        features=initial_features,
        importance_threshold=importance_threshold,
        max_features=max_features
    )

    # Step 2: Remove correlated features
    final_features = remove_correlated_features(
        df=df,
        features=important_features,
        correlation_threshold=correlation_threshold
    )

    # Prepare result
    result = {
        'selected_features': final_features,
        'n_selected': len(final_features),
        'n_initial': len(initial_features),
        'importance_threshold': importance_threshold,
        'correlation_threshold': correlation_threshold,
        'selection_method': 'importance_correlation'
    }

    logger.info(f"Final selection: {len(final_features)} features from {len(initial_features)} initial")

    # Save report if requested
    if save_report:
        if output_path is None:
            try:
                from src.config import DATA_DIRS
                output_path = DATA_DIRS['reports'] / 'feature_selection_report.json'
            except ImportError:
                output_path = Path('feature_selection_report.json')

        import json
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, default=str)

        logger.info(f"Feature selection report saved to {output_path}")

    return result


def _calculate_lgb_importance(X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    """Calculate feature importance using LightGBM."""
    try:
        model = lgb.LGBMRegressor(
            n_estimators=100,
            learning_rate=0.1,
            random_state=42,
            verbosity=-1
        )
        model.fit(X, y)

        # Normalize importance scores to 0-1 range
        importance = model.feature_importances_
        max_importance = importance.max()
        if max_importance > 0:
            importance = importance / max_importance

        return dict(zip(X.columns, importance, strict=False))

    except Exception as e:
        logger.warning(f"LightGBM importance calculation failed: {e}")
        return {col: 0.0 for col in X.columns}


def _calculate_rf_importance(X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    """Calculate feature importance using Random Forest."""
    try:
        from sklearn.ensemble import RandomForestRegressor

        model = RandomForestRegressor(
            n_estimators=50,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X, y)

        # Normalize importance scores
        importance = model.feature_importances_
        max_importance = importance.max()
        if max_importance > 0:
            importance = importance / max_importance

        return dict(zip(X.columns, importance, strict=False))

    except Exception as e:
        logger.warning(f"Random Forest importance calculation failed: {e}")
        return {col: 0.0 for col in X.columns}


def _calculate_correlation_importance(X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    """Calculate feature importance using correlation with target."""
    importance_scores = {}

    for col in X.columns:
        try:
            corr = abs(X[col].corr(y))
            importance_scores[col] = corr if not np.isnan(corr) else 0.0
        except:
            importance_scores[col] = 0.0

    return importance_scores


# Backward compatibility
def select_features_by_importance(*args, **kwargs):
    """Deprecated: Use select_important_features instead."""
    return select_important_features(*args, **kwargs)
