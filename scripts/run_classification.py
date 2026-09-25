"""
Script to execute training, benchmarking, and evaluation of SCS Classification models.
Includes Multimodal Sensor Fusion and generates presentation-grade diagnostic plots.
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.models.classifier import run_classification_benchmark
from src.fusion import MultimodalFusionPipeline


def plot_confusion_matrix(cm, classes, title, save_path, acc_score):
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues', cbar=False,
        xticklabels=classes, yticklabels=classes,
        annot_kws={'size': 14, 'weight': 'bold'}
    )
    plt.title(f"{title}\nTest Accuracy = {acc_score*100:.2f}%", fontsize=12, fontweight='bold', pad=12)
    plt.xlabel('Predicted SCS Class', fontsize=11, fontweight='bold')
    plt.ylabel('True SCS Class', fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def main():
    print("=" * 70)
    print("IgniCoal AI: SCS Classification Modeling & Multimodal Fusion")
    print("=" * 70)

    data_dir = os.path.join(BASE_DIR, 'data', 'processed')
    results_dir = os.path.join(BASE_DIR, 'results', 'classification')
    models_dir = os.path.join(BASE_DIR, 'models', 'saved')
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    master_path = os.path.join(data_dir, 'ignicoal_all_features.csv')
    df = pd.read_csv(master_path)
    print(f"Loaded master feature matrix: {df.shape[0]} samples, {df.shape[1]} columns")

    meta_cols = [
        'signal_id', 'sample_code', 'pellet', 'shot', 'scs_label', 'file_source',
        'Ignition_Temp', 'Ash_Content', 'Fixed_Carbon', 'Moisture', 'VM'
    ]
    feature_cols = [c for c in df.columns if c not in meta_cols]

    # 1. Standard Benchmark across Classifiers
    print("\n>>> Running Model Comparison Benchmark...")
    results_df, trained_models, eval_details, holdout_data = run_classification_benchmark(
        df, feature_cols, label_col='scs_label', test_size=0.20
    )
    X_train, X_test, y_train, y_test, label_encoder = holdout_data
    classes_order = label_encoder.classes_

    print("\nClassification Comparison Table:")
    cols_display = ['Model', '5-Fold CV Accuracy', 'Holdout Test Accuracy', 'Test Correct / Total', 'Holdout Precision', 'Holdout Recall', 'Holdout F1-Score']
    print(results_df[cols_display].to_string(index=False))

    # 2. Multimodal Sensor Fusion Model
    print("\n>>> Training Multimodal Sensor Fusion Pipeline (PA Features + Predicted Proximate)...")
    with open(os.path.join(models_dir, 'best_model_Ash_Content.pkl'), 'rb') as f:
        ash_pkg = pickle.load(f)
    with open(os.path.join(models_dir, 'best_model_Fixed_Carbon.pkl'), 'rb') as f:
        carbon_pkg = pickle.load(f)
    with open(os.path.join(models_dir, 'best_model_Ignition_Temperature.pkl'), 'rb') as f:
        tign_pkg = pickle.load(f)

    ash_df = pd.read_csv(os.path.join(data_dir, 'ash_features.csv'))[ash_pkg['features']]
    carbon_df = pd.read_csv(os.path.join(data_dir, 'carbon_features.csv'))[carbon_pkg['features']]
    thermal_df = pd.read_csv(os.path.join(data_dir, 'thermal_features.csv'))[tign_pkg['features']]

    fusion_pipe = MultimodalFusionPipeline(
        ash_model=ash_pkg['model'],
        carbon_model=carbon_pkg['model'],
        tign_model=tign_pkg['model'],
        classifier_type='ExtraTrees'
    )

    # Augment features
    X_all_fused = fusion_pipe.augment_features_with_predictions(
        df[feature_cols].values, ash_df, carbon_df, thermal_df
    )
    y_all = label_encoder.transform(df['scs_label'].values)

    # Split for fusion
    from sklearn.model_selection import train_test_split
    X_f_train, X_f_test, y_f_train, y_f_test = train_test_split(
        X_all_fused, y_all, test_size=0.20, stratify=y_all, random_state=42
    )

    fusion_pipe.fit(X_f_train, y_f_train)
    y_f_pred = fusion_pipe.predict(X_f_test)
    fused_acc = accuracy_score(y_f_test, y_f_pred)
    fused_cm = confusion_matrix(y_f_test, y_f_pred)

    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import precision_score, recall_score, f1_score

    cv_skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fused_cv_model = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', ExtraTreesClassifier(n_estimators=180, max_depth=10, min_samples_split=2, random_state=42))
    ])
    fused_cv_scores = cross_val_score(fused_cv_model, X_all_fused, y_all, cv=cv_skf, scoring='accuracy')
    fused_cv_mean = float(np.mean(fused_cv_scores))
    fused_cv_std = float(np.std(fused_cv_scores))

    fused_rec = {
        'Model': 'Multimodal Cascaded Fusion',
        '5-Fold CV Accuracy': f"{fused_cv_mean*100:.2f}% ± {fused_cv_std*100:.2f}%",
        'Holdout Test Accuracy': f"{fused_acc*100:.2f}%",
        'Test Correct / Total': f"{int(np.sum(y_f_pred == y_f_test))} / {len(y_f_test)}",
        'Holdout Precision': f"{precision_score(y_f_test, y_f_pred, average='macro'):.4f}",
        'Holdout Recall': f"{recall_score(y_f_test, y_f_pred, average='macro'):.4f}",
        'Holdout F1-Score': f"{f1_score(y_f_test, y_f_pred, average='macro'):.4f}",
        'raw_test_acc': fused_acc,
        'raw_cv_acc': fused_cv_mean
    }
    
    # Append fusion result
    all_results = pd.concat([pd.DataFrame([fused_rec]), results_df], ignore_index=True)
    all_results = all_results.sort_values(by=['raw_test_acc', 'raw_cv_acc'], ascending=[False, False]).reset_index(drop=True)
    all_results['Status'] = ['[Selected]' if i == 0 else '' for i in range(len(all_results))]

    cols_display_final = ['Model', '5-Fold CV Accuracy', 'Holdout Test Accuracy', 'Test Correct / Total', 'Holdout Precision', 'Holdout Recall', 'Holdout F1-Score', 'Status']
    print("\nUpdated Comparison with Fusion Pipeline:")
    print(all_results[cols_display_final].to_string(index=False))

    # Save results
    csv_path = os.path.join(results_dir, 'classification_comparison.csv')
    all_results.to_csv(csv_path, index=False)
    print(f"\n  Saved classification benchmark: {csv_path}")

    # Save Best Model & Fusion Pipeline
    best_name = results_df.iloc[0]['Model']
    best_model_obj = trained_models[best_name]
    save_pkg = {
        'model': best_model_obj,
        'encoder': label_encoder,
        'features': feature_cols,
        'classes': list(classes_order)
    }
    with open(os.path.join(models_dir, 'best_model_SCS_Classifier.pkl'), 'wb') as f:
        pickle.dump(save_pkg, f)
    with open(os.path.join(models_dir, 'fused_model_SCS.pkl'), 'wb') as f:
        pickle.dump({'pipeline': fusion_pipe, 'encoder': label_encoder, 'classes': list(classes_order)}, f)
    print(f"  Saved best classifier models to: {models_dir}")

    # 3. Generate Diagnostic & Presentation Plots
    print("\n>>> Generating Presentation Plots...")
    # Confusion Matrix for Best Single Model
    cm_best = eval_details[best_name]['confusion_matrix']
    plot_confusion_matrix(
        cm_best, classes_order, f"{best_name} SCS Confusion Matrix",
        os.path.join(results_dir, 'confusion_matrix_best_single.png'),
        results_df.iloc[0]['raw_test_acc']
    )

    # Confusion Matrix for Fused Model
    plot_confusion_matrix(
        fused_cm, classes_order, "Multimodal Fusion SCS Confusion Matrix",
        os.path.join(results_dir, 'confusion_matrix_fused.png'),
        fused_acc
    )

    # Model Accuracy Bar Chart
    plt.figure(figsize=(9, 5))
    plot_df = all_results.copy()
    accs = [float(x.replace('%', '')) for x in plot_df['Holdout Test Accuracy']]
    colors = ['#2ca02c' if 'Fusion' in m else '#1f77b4' for m in plot_df['Model']]
    bars = plt.barh(plot_df['Model'], accs, color=colors, edgecolor='black', alpha=0.85)
    plt.xlim(70, 105)
    plt.xlabel('Holdout Test Accuracy (%)', fontsize=11, fontweight='bold')
    plt.title('Classification Model Comparison', fontsize=12, fontweight='bold', pad=12)
    plt.gca().invert_yaxis()
    for bar in bars:
        w = bar.get_width()
        plt.text(w + 0.6, bar.get_y() + bar.get_height()/2, f"{w:.1f}%", va='center', fontweight='bold', fontsize=10)
    plt.grid(axis='x', linestyle=':', alpha=0.6)
    plt.tight_layout()
    bar_path = os.path.join(results_dir, 'model_accuracy_comparison.png')
    plt.savefig(bar_path, dpi=200)
    plt.close()
    print(f"  Saved model accuracy comparison plot: {bar_path}")

    print("\n" + "=" * 70)
    print("CLASSIFICATION BENCHMARK & FUSION COMPLETE!")
    print("=" * 70)


if __name__ == '__main__':
    main()
