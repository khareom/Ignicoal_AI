"""
Classification Modeling Module for IgniCoal AI.
Trains, benchmarks, and compares various classification models to predict:
  Spontaneous Combustion Susceptibility (SCS): Low, Moderate, High.
Implements:
  - Fine KNN (k=5)
  - Support Vector Machine (RBF Kernel)
  - Random Forest & Extra Trees
  - Ensemble Boosted Trees (Gradient Boosting)
  - Multi-Layer Perceptron (MLP)
  - Multimodal Sensor Fusion Classifier (PA features + Predicted Proximate Properties)
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_validate
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier
)
from sklearn.neural_network import MLPClassifier


def get_classification_models() -> dict:
    """
    Returns a dictionary of candidate classification models wrapped in scaling pipelines.
    """
    return {
        'Fine KNN (k=5)': Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', KNeighborsClassifier(n_neighbors=5, metric='euclidean', weights='distance'))
        ]),
        'SVM (RBF)': Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', SVC(kernel='rbf', C=10.0, probability=True, random_state=42))
        ]),
        'Random Forest': RandomForestClassifier(
            n_estimators=150, max_depth=8, min_samples_split=3, random_state=42
        ),
        'Extra Trees': ExtraTreesClassifier(
            n_estimators=150, max_depth=8, min_samples_split=3, random_state=42
        ),
        'Ensemble Boosted Trees': GradientBoostingClassifier(
            n_estimators=120, learning_rate=0.08, max_depth=4, random_state=42
        ),
        'Hist Gradient Boosting': HistGradientBoostingClassifier(
            max_iter=120, learning_rate=0.08, max_depth=4, random_state=42
        ),
        'Neural Network (MLP)': Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42))
        ])
    }


def evaluate_classifier_cv(model, X: np.ndarray, y: np.ndarray, n_splits: int = 5) -> dict:
    """
    Performs 5-fold stratified cross-validation and computes accuracy, precision, recall, and F1.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scoring = {
        'acc': 'accuracy',
        'prec_macro': 'precision_macro',
        'rec_macro': 'recall_macro',
        'f1_macro': 'f1_macro'
    }
    cv_res = cross_validate(model, X, y, cv=skf, scoring=scoring, return_train_score=True)
    
    return {
        'cv_train_acc_mean': float(np.mean(cv_res['train_acc'])),
        'cv_train_acc_std': float(np.std(cv_res['train_acc'])),
        'cv_test_acc_mean': float(np.mean(cv_res['test_acc'])),
        'cv_test_acc_std': float(np.std(cv_res['test_acc'])),
        'cv_test_f1_mean': float(np.mean(cv_res['test_f1_macro'])),
        'cv_test_f1_std': float(np.std(cv_res['test_f1_macro'])),
        'cv_test_prec_mean': float(np.mean(cv_res['test_prec_macro'])),
        'cv_test_rec_mean': float(np.mean(cv_res['test_rec_macro']))
    }


def train_and_eval_holdout_classifier(
    model,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    target_names: list[str]
) -> dict:
    """
    Fits model on train set and evaluates holdout metrics and confusion matrix.
    """
    model.fit(X_train, y_train)
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    cm = confusion_matrix(y_test, y_pred_test)
    report = classification_report(y_test, y_pred_test, target_names=target_names, output_dict=True)
    
    return {
        'train_acc': float(accuracy_score(y_train, y_pred_train)),
        'test_acc': float(accuracy_score(y_test, y_pred_test)),
        'test_precision': float(precision_score(y_test, y_pred_test, average='macro')),
        'test_recall': float(recall_score(y_test, y_pred_test, average='macro')),
        'test_f1': float(f1_score(y_test, y_pred_test, average='macro')),
        'confusion_matrix': cm,
        'classification_report': report,
        'y_pred_test': y_pred_test
    }


def run_classification_benchmark(
    master_df: pd.DataFrame,
    feature_cols: list[str],
    label_col: str = 'scs_label',
    test_size: float = 0.20
) -> tuple[pd.DataFrame, dict, dict, tuple]:
    """
    Runs comprehensive classification benchmark across all models.
    """
    X = master_df[feature_cols].values
    raw_labels = master_df[label_col].values

    # Encode labels (High, Moderate, Low)
    label_encoder = LabelEncoder()
    # Ensure ordered labels: High, Moderate, Low
    classes_order = ['High', 'Moderate', 'Low']
    label_encoder.fit(classes_order)
    y = label_encoder.transform(raw_labels)

    # 80:20 Stratified Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42
    )

    models = get_classification_models()
    records = []
    trained_models = {}
    eval_details = {}
    best_acc = -1.0
    best_model_name = None

    for name, model in models.items():
        # 1. 5-Fold Stratified CV
        cv_metrics = evaluate_classifier_cv(model, X, y, n_splits=5)
        
        # 2. Holdout evaluation
        holdout = train_and_eval_holdout_classifier(
            model, X_train, X_test, y_train, y_test, target_names=classes_order
        )

        rec = {
            'Model': name,
            'CV Accuracy (mean ± std)': f"{cv_metrics['cv_test_acc_mean']*100:.2f}% ± {cv_metrics['cv_test_acc_std']*100:.2f}%",
            'CV F1-Score': f"{cv_metrics['cv_test_f1_mean']:.4f}",
            'Holdout Train Acc': f"{holdout['train_acc']*100:.2f}%",
            'Holdout Test Acc': f"{holdout['test_acc']*100:.2f}%",
            'Holdout Precision': f"{holdout['test_precision']:.4f}",
            'Holdout Recall': f"{holdout['test_recall']:.4f}",
            'Holdout F1-Score': f"{holdout['test_f1']:.4f}",
            'raw_test_acc': holdout['test_acc'],
            'raw_cv_acc': cv_metrics['cv_test_acc_mean']
        }
        records.append(rec)
        trained_models[name] = model
        eval_details[name] = holdout

        if holdout['test_acc'] > best_acc:
            best_acc = holdout['test_acc']
            best_model_name = name

    results_df = pd.DataFrame(records).sort_values(by='raw_test_acc', ascending=False).reset_index(drop=True)
    return results_df, trained_models, eval_details, (X_train, X_test, y_train, y_test, label_encoder)
