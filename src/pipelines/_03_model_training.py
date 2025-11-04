import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split, KFold , RandomizedSearchCV
from sklearn.metrics import mean_squared_error
import warnings
import time
import sys
import joblib
import json
from pathlib import Path

# === DEFINE PROJECT ROOT ===
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
# ===============================

warnings.filterwarnings('ignore')

# -----------------------------------------------------------------
# 1. PROJECT CONFIGURATION
# -----------------------------------------------------------------
CONFIG = {
    "data_file": PROJECT_ROOT / 'data' / '3_processed' / 'master_feature_table.parquet',
    "model_output_path": PROJECT_ROOT / 'models' / 'final_forecaster.joblib',
    "features_output_path": PROJECT_ROOT / 'models' / 'model_features.json',
    "metrics_output_path": PROJECT_ROOT / 'reports' / 'metrics' / 'final_model_metrics.json',
    "tuning_iterations": 20,
    "cv_folds": 3
}


# -----------------------------------------------------------------
# 2. FUNCTIONAL DEFINITIONS (All print/logging in English)
# -----------------------------------------------------------------

def load_data(filepath):
    """Loads the clean feature table from the processing pipeline."""
    print(f"[load_data] Loading data from: {filepath}...")  # SỬA LỖI TV
    start_time = time.time()
    try:
        if str(filepath).endswith('.parquet'):
            df = pd.read_parquet(filepath)
        elif str(filepath).endswith('.csv'):
            df = pd.read_csv(filepath)
        else:
            raise ValueError(f"Unsupported file format: {filepath}")

        print(f"OK. Load complete. Shape: {df.shape}. (Took {time.time() - start_time:.2f}s)")  # SỬA LỖI EMOJI
        return df
    except FileNotFoundError:
        print(f"ERROR: File not found {filepath}.")  # SỬA LỖI EMOJI
        print("Please run the data processing pipeline (_02_feature_enrichment.py) first.")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR loading file: {e}")  # SỬA LỖI EMOJI
        sys.exit(1)


def prepare_data(df):
    """
    Filters, creates target variable (sales), selects features
    from all integrated Workstreams, and splits data.
    """
    print("[prepare_data] Preparing data for modeling...")  # SỬA LỖI TV

    # === DEFINE TARGET VARIABLE ===
    target_col = 'SALES_VALUE'  # Assuming Dunnhumby

    if target_col not in df.columns:
        if 'sales' in df.columns:  # Fallback for M5
            target_col = 'sales'
        else:
            print(f"ERROR: Target column '{target_col}' or 'sales' not found.")  # SỬA LỖI TV
            sys.exit(1)

    print(f"Target variable (Y) set to: {target_col}")  # SỬA LỖI TV

    df_model = df.dropna(subset=[target_col]).copy()

    if df_model.empty:
        print("ERROR: No data left to train after dropping NaN target values.")  # SỬA LỖI TV
        sys.exit(1)

    # === DEFINE FEATURES (ALL 4 WORKSTREAMS) ===
    numeric_features = [
        # --- WS1 (E-commerce) Features ---
        'price', 'freight_value', 'dist_cust_seller_km', 'product_weight_g',
        'payment_value_total', 'payment_installments_total',

        # --- WS2 (Time-Series) Features ---
        'sales_lag_7', 'sales_lag_28',
        'rolling_mean_7_lag_28', 'rolling_std_7_lag_28',
        'days_until_next_event', 'days_since_last_event',

        # --- WS3 (Behavior) Features ---
        'total_views', 'total_addtocart', 'rate_view_to_cart',
        'session_duration_days', 'days_since_last_action',

        # --- WS4 (Price/Promo) Features ---
        'base_price', 'total_discount', 'discount_pct',
    ]

    categorical_features = [
        # --- WS1 (E-commerce) Features ---
        'product_category_name_english', 'customer_state', 'seller_state', 'payment_type_primary',

        # --- WS2 (Time-Series) Features ---
        'month', 'dayofweek', 'is_weekend', 'event_name_1', 'is_event', 'snap',

        # --- WS4 (Price/Promo) Features ---
        'is_on_display', 'is_on_mailer', 'is_on_retail_promo', 'is_on_coupon_promo',
    ]
    # === END OF FEATURE LIST ===

    all_features = [col for col in (numeric_features + categorical_features) if col in df.columns]
    categorical_features = [col for col in categorical_features if col in all_features]

    missing_features = set(numeric_features + categorical_features) - set(df.columns)
    if missing_features:
        print(f"Warning: Missing expected features (WS may be toggled off): {missing_features}")  # SỬA LỖI TV

    if not all_features:
        print("ERROR: No valid features found in the input file.")  # SỬA LỖI TV
        sys.exit(1)

    print(f"Found {len(all_features)} valid features for training.")  # SỬA LỖI TV

    X = df_model[all_features]
    y = df_model[target_col]

    print(f"Converting {len(categorical_features)} columns to 'category' dtype...")  # SỬA LỖI TV
    for col in categorical_features:
        X[col] = X[col].astype('category')

    print("Splitting data (80/20)...")  # SỬA LỖI TV
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        shuffle=True
    )
    print("OK. Data preparation complete.")  # SỬA LỖI TV

    return X_train, X_test, y_train, y_test, all_features, categorical_features


def tune_model(X_train, y_train, categorical_features):
    """Hyperparameter tuning using RandomizedSearchCV (for REGRESSION)."""
    print("[tune_model] Starting hyperparameter tuning (Regression)...")  # SỬA LỖI TV
    start_train = time.time()

    param_grid = {
        'n_estimators': [200, 500, 1000],
        'learning_rate': [0.02, 0.05, 0.1],
        'num_leaves': [31, 50, 70],
        'max_depth': [-1, 10, 20],
        'colsample_bytree': [0.7, 0.8, 0.9],
        'subsample': [0.7, 0.8, 0.9],
        'reg_alpha': [0, 0.1, 0.5],
        'reg_lambda': [0, 0.1, 0.5]
    }

    kfold = KFold(n_splits=CONFIG['cv_folds'], shuffle=True, random_state=42)

    base_model = lgb.LGBMRegressor(
        random_state=42,
        n_jobs=-1,
        objective='regression_l1'
    )

    random_search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_grid,
        n_iter=CONFIG['tuning_iterations'],
        cv=kfold,
        scoring='neg_root_mean_squared_error',
        n_jobs=-1,
        random_state=42,
        verbose=1
    )

    random_search.fit(
        X_train,
        y_train,
        categorical_feature=categorical_features
    )

    print(f"\nOK. Tuning complete (Took {time.time() - start_train:.2f}s)")  # SỬA LỖI TV
    print("\n" + "=" * 50)
    print("           BEST MODEL FOUND")
    print("=" * 50)
    print(f"Best Score (RMSE): {-1 * random_search.best_score_:.4f}")
    print("Best Hyperparameters:")
    print(random_search.best_params_)
    print("=" * 50)

    return random_search.best_estimator_


def evaluate_model(model, X_test, y_test):
    """Evaluates the final REGRESSION model."""
    print("[evaluate_model] Evaluating model on Test set...")  # SỬA LỖI TV
    y_pred = model.predict(X_test)
    y_pred[y_pred < 0] = 0

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    print("\n" + "=" * 50)
    print("      FINAL MODEL EVALUATION (REGRESSION)")
    print("=" * 50)
    print(f"RMSE (Root Mean Squared Error): {rmse:.4f}")  # SỬA LỖI EMOJI
    print("=" * 50)

    metrics = {
        "rmse": rmse,
        "mse": mse
    }
    return metrics


def save_artifacts(model, features_config, metrics):
    """Saves model, features, and metrics to disk."""
    print("[save_artifacts] Saving model artifacts...")  # SỬA LỖI TV

    (PROJECT_ROOT / 'models').mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / 'reports' / 'metrics').mkdir(parents=True, exist_ok=True)

    try:
        joblib.dump(model, CONFIG['model_output_path'])
        print(f"OK. Model saved to: {CONFIG['model_output_path']}")  # SỬA LỖI EMOJI
    except Exception as e:
        print(f"ERROR saving model: {e}")  # SỬA LỖI EMOJI

    try:
        with open(CONFIG['features_output_path'], 'w') as f:
            json.dump(features_config, f, indent=4)
        print(f"OK. Feature config saved to: {CONFIG['features_output_path']}")  # SỬA LỖI EMOJI
    except Exception as e:
        print(f"ERROR saving feature config: {e}")  # SỬA LỖI EMOJI

    try:
        with open(CONFIG['metrics_output_path'], 'w') as f:
            json.dump(metrics, f, indent=4)
        print(f"OK. Metrics saved to: {CONFIG['metrics_output_path']}")  # SỬA LỖI EMOJI
    except Exception as e:
        print(f"ERROR saving metrics: {e}")  # SỬA LỖI EMOJI


# -----------------------------------------------------------------
# 3. MAIN ORCHESTRATOR (All English)
# -----------------------------------------------------------------

def main():
    """Orchestrates the entire training pipeline."""
    print("========== STARTING MODEL TRAINING PIPELINE (REGRESSION) ==========")
    total_start_time = time.time()

    print("\n--- STEP 1: LOAD DATA ---")
    df = load_data(CONFIG['data_file'])

    print("\n--- STEP 2: PREPARE DATA & SPLIT ---")
    X_train, X_test, y_train, y_test, features, cat_features = prepare_data(df)

    print("\n--- STEP 3: TUNE MODEL (HYPERPARAMETER TUNING) ---")
    best_model = tune_model(X_train, y_train, cat_features)

    print("\n--- STEP 4: EVALUATE FINAL MODEL ---")
    metrics = evaluate_model(best_model, X_test, y_test)

    print("\n--- STEP 5: SAVE ARTIFACTS (MODEL, FEATURES, METRICS) ---")
    features_config = {
        "all_features": features,
        "categorical_features": cat_features
    }
    save_artifacts(best_model, features_config, metrics)

    print("\n========================================================")
    print(f"COMPLETE! Total runtime: {time.time() - total_start_time:.2f} seconds.")  # SỬA LỖI EMOJI
    print(f"Artifacts saved to: {CONFIG['model_output_path']} and related .json files.")
    print("========================================================")


if __name__ == "__main__":
    main()