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
from sklearn.preprocessing import StandardScaler, PowerTransformer, QuantileTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    VotingRegressor
)
from sklearn.neural_network import MLPRegressor


def get_regression_models(target_col: str = None) -> dict:
    """
    Returns a dictionary of candidate regression models wrapped in pipelines.
    Includes target-specific optimized architectures (PowerTransformer, VotingRegressor, etc.).
    """
    models = {}

    target_lower = (target_col or '').lower()
    
    # 1. Target-Specific High Performance Architectures
    if 'ash' in target_lower:
        models['PowerTransformer + Extra Trees (Optimized)'] = Pipeline([
            ('scaler', PowerTransformer(method='yeo-johnson')),
            ('regressor', ExtraTreesRegressor(n_estimators=200, max_depth=8, max_features=0.7, random_state=42))
        ])
        models['PowerTransformer + Gradient Boosting'] = Pipeline([
            ('scaler', PowerTransformer(method='yeo-johnson')),
            ('regressor', GradientBoostingRegressor(n_estimators=160, learning_rate=0.03, max_depth=5, subsample=0.85, random_state=42))
        ])
    elif 'carbon' in target_lower:
        models['Tuned Extra Trees (Optimized)'] = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', ExtraTreesRegressor(n_estimators=150, max_depth=8, min_samples_split=3, random_state=42))
        ])
        models['Tuned Gradient Boosting'] = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', GradientBoostingRegressor(n_estimators=160, learning_rate=0.03, max_depth=5, subsample=0.85, random_state=42))
        ])
    elif 'ignition' in target_lower or 'temp' in target_lower or 'thermal' in target_lower:
        models['QuantileTransformer + Voting Ensemble (Optimized)'] = Pipeline([
            ('scaler', QuantileTransformer(n_quantiles=50, random_state=42)),
            ('regressor', VotingRegressor([
                ('et', ExtraTreesRegressor(n_estimators=200, max_features=1.0, random_state=42)),
                ('gb', GradientBoostingRegressor(n_estimators=160, learning_rate=0.03, max_depth=5, subsample=0.85, random_state=42))
            ], weights=[0.65, 0.35]))
        ])
        models['QuantileTransformer + Extra Trees'] = Pipeline([
            ('scaler', QuantileTransformer(n_quantiles=50, random_state=42)),
            ('regressor', ExtraTreesRegressor(n_estimators=200, max_depth=8, random_state=42))
        ])

    # 2. General Candidate Regressors
    models['Extra Trees'] = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', ExtraTreesRegressor(n_estimators=150, max_depth=8, min_samples_split=3, random_state=42))
    ])
    models['Gradient Boosting'] = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', GradientBoostingRegressor(n_estimators=120, learning_rate=0.05, max_depth=4, random_state=42))
    ])
    models['Random Forest'] = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', RandomForestRegressor(n_estimators=150, max_depth=8, min_samples_split=3, random_state=42))
    ])
    models['Hist Gradient Boosting'] = HistGradientBoostingRegressor(
        max_iter=120, learning_rate=0.05, max_depth=4, random_state=42
    )
    models['Ridge'] = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', Ridge(alpha=1.0, random_state=42))
    ])
    models['SVR (RBF)'] = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', SVR(C=10.0, epsilon=0.1))
    ])
    models['MLP Regressor'] = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42))
    ])

    return models


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
    
    test_mape = float(np.mean(np.abs((y_test - y_pred_test) / np.maximum(np.abs(y_test), 1e-6)))) * 100.0
    
    return {
        'train_r2': float(r2_score(y_train, y_pred_train)),
        'train_rmse': float(np.sqrt(mean_squared_error(y_train, y_pred_train))),
        'train_mae': float(mean_absolute_error(y_train, y_pred_train)),
        'test_r2': float(r2_score(y_test, y_pred_test)),
        'test_rmse': float(np.sqrt(mean_squared_error(y_test, y_pred_test))),
        'test_mae': float(mean_absolute_error(y_test, y_pred_test)),
        'test_mape': test_mape,
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
    Uses stratified 80:20 splitting based on coal sample to ensure balanced representation.
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

    # 80:20 Stratified Train-Test split based on coal sample
    stratify_col = sub_df['sample_code'].values if 'sample_code' in sub_df.columns else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=stratify_col, random_state=42
    )

    models = get_regression_models(target_col)
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
            'Holdout Test MAPE (%)': f"{holdout_metrics['test_mape']:.2f}%",
            'raw_cv_test_r2': cv_metrics['cv_test_r2_mean']
        }
        records.append(rec)

        if holdout_metrics['test_r2'] > best_r2:
            best_r2 = holdout_metrics['test_r2']
            best_model_obj = model

    results_df = pd.DataFrame(records).sort_values(by='Holdout Test R²', ascending=False).reset_index(drop=True)
    return results_df, best_model_obj, (X_train, X_test, y_train, y_test)
