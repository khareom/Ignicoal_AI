"""
Regression Modeling Module for IgniCoal AI.
Trains, benchmarks, and compares various regression models to predict:
  1. Ignition Temperature (Tign)
  2. Fixed Carbon Content
  3. Ash Content
Utilizing feature subsets defined in FeatureList_1.pptx and research papers.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor
)
from sklearn.neural_network import MLPRegressor


def get_regression_models() -> dict:
    """
    Returns a dictionary of candidate regression models wrapped in pipelines (scaling where appropriate).
    """
    return {
        'Ridge': Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', Ridge(alpha=1.0, random_state=42))
        ]),
        'SVR (RBF)': Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', SVR(C=10.0, epsilon=0.1))
        ]),
        'Random Forest': RandomForestRegressor(
            n_estimators=150, max_depth=8, min_samples_split=3, random_state=42
        ),
        'Extra Trees': ExtraTreesRegressor(
            n_estimators=150, max_depth=8, min_samples_split=3, random_state=42
        ),
        'Gradient Boosting': GradientBoostingRegressor(
            n_estimators=120, learning_rate=0.05, max_depth=4, random_state=42
        ),
        'Hist Gradient Boosting': HistGradientBoostingRegressor(
            max_iter=120, learning_rate=0.05, max_depth=4, random_state=42
        ),
        'MLP Regressor': Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42))
        ])
    }


def evaluate_model_cv(model, X: np.ndarray, y: np.ndarray, n_splits: int = 5) -> dict:
    """
    Performs 5-fold cross-validation and computes R2, RMSE, and MAE.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    scoring = {
        'r2': 'r2',
        'neg_rmse': 'neg_root_mean_squared_error',
        'neg_mae': 'neg_mean_absolute_error'
    }
    cv_res = cross_validate(model, X, y, cv=kf, scoring=scoring, return_train_score=True)
    
    return {
        'cv_train_r2_mean': float(np.mean(cv_res['train_r2'])),
        'cv_train_r2_std': float(np.std(cv_res['train_r2'])),
        'cv_test_r2_mean': float(np.mean(cv_res['test_r2'])),
        'cv_test_r2_std': float(np.std(cv_res['test_r2'])),
        'cv_test_rmse_mean': float(-np.mean(cv_res['test_neg_rmse'])),
        'cv_test_rmse_std': float(np.std(cv_res['test_neg_rmse'])),
        'cv_test_mae_mean': float(-np.mean(cv_res['test_neg_mae'])),
        'cv_test_mae_std': float(np.std(cv_res['test_neg_mae']))
    }


def train_and_eval_holdout(
    model,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray
) -> dict:
    """
    Fits model on train set and evaluates on both train and test holdout.
    """
    model.fit(X_train, y_train)
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    return {
        'train_r2': float(r2_score(y_train, y_pred_train)),
        'train_rmse': float(np.sqrt(mean_squared_error(y_train, y_pred_train))),
        'train_mae': float(mean_absolute_error(y_train, y_pred_train)),
        'test_r2': float(r2_score(y_test, y_pred_test)),
        'test_rmse': float(np.sqrt(mean_squared_error(y_test, y_pred_test))),
        'test_mae': float(mean_absolute_error(y_test, y_pred_test)),
        'y_pred_test': y_pred_test
    }


def run_benchmark_for_target(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: list[str],
    test_size: float = 0.20
) -> tuple[pd.DataFrame, dict, tuple]:
    """
    Benchmarks all candidate regression models on the specified dataset and target.
    Handles NaN removal (e.g. C3 for Ash/Carbon).
    Returns:
      - results_df: comparison table sorted by test R2
      - best_models: fitted best model instances
      - holdout_data: (X_train, X_test, y_train, y_test)
    """
    # Filter out missing ground truth targets
    valid_mask = df[target_col].notna()
    sub_df = df[valid_mask].copy()

    X = sub_df[feature_cols].values
    y = sub_df[target_col].values

    # 80:20 Train-Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    models = get_regression_models()
    records = []
    best_model_obj = None
    best_r2 = -float('inf')

    for name, model in models.items():
        # 1. 5-Fold Cross Validation
        cv_metrics = evaluate_model_cv(model, X, y, n_splits=5)
        
        # 2. Holdout 80:20 evaluation
        holdout_metrics = train_and_eval_holdout(model, X_train, X_test, y_train, y_test)
        
        rec = {
            'Model': name,
            'CV Test R² (mean ± std)': f"{cv_metrics['cv_test_r2_mean']:.4f} ± {cv_metrics['cv_test_r2_std']:.4f}",
            'CV Test RMSE': f"{cv_metrics['cv_test_rmse_mean']:.3f} ± {cv_metrics['cv_test_rmse_std']:.3f}",
            'CV Test MAE': f"{cv_metrics['cv_test_mae_mean']:.3f} ± {cv_metrics['cv_test_mae_std']:.3f}",
            'Holdout Train R²': holdout_metrics['train_r2'],
            'Holdout Test R²': holdout_metrics['test_r2'],
            'Holdout Test RMSE': holdout_metrics['test_rmse'],
            'Holdout Test MAE': holdout_metrics['test_mae'],
            'raw_cv_test_r2': cv_metrics['cv_test_r2_mean']
        }
        records.append(rec)

        if holdout_metrics['test_r2'] > best_r2:
            best_r2 = holdout_metrics['test_r2']
            best_model_obj = model

    results_df = pd.DataFrame(records).sort_values(by='Holdout Test R²', ascending=False).reset_index(drop=True)
    return results_df, best_model_obj, (X_train, X_test, y_train, y_test)
