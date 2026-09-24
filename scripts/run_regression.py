"""
Script to execute training and benchmarking of regression models for:
  1. Ignition Temperature (Tign)
  2. Fixed Carbon Content
  3. Ash Content
Saves metrics, comparison tables, and diagnostic plots.
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.models.regression import run_benchmark_for_target


def main():
    print("=" * 70)
    print("IgniCoal AI: Regression Modeling & Comparison Benchmark")
    print("=" * 70)

    data_dir = os.path.join(BASE_DIR, 'data', 'processed')
    results_dir = os.path.join(BASE_DIR, 'results', 'regression')
    models_dir = os.path.join(BASE_DIR, 'models', 'saved')
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    tasks = [
        {
            'name': 'Ignition_Temperature',
            'file': 'thermal_features.csv',
            'target': 'Ignition_Temp',
            'unit': '°C',
            'exclude_cols': ['signal_id', 'sample_code', 'scs_label', 'Ignition_Temp']
        },
        {
            'name': 'Fixed_Carbon',
            'file': 'carbon_features.csv',
            'target': 'Fixed_Carbon',
            'unit': '%',
            'exclude_cols': ['signal_id', 'sample_code', 'scs_label', 'Fixed_Carbon']
        },
        {
            'name': 'Ash_Content',
            'file': 'ash_features.csv',
            'target': 'Ash_Content',
            'unit': '%',
            'exclude_cols': ['signal_id', 'sample_code', 'scs_label', 'Ash_Content']
        }
    ]

    all_summary = {}

    for t in tasks:
        print(f"\n>>> Running Benchmark for: {t['name']} ({t['target']} in {t['unit']})")
        df_path = os.path.join(data_dir, t['file'])
        df = pd.read_csv(df_path)
        feature_cols = [c for c in df.columns if c not in t['exclude_cols']]

        print(f"  Dataset size: {len(df)} rows, {len(feature_cols)} features")
        valid_count = df[t['target']].notna().sum()
        print(f"  Valid target instances: {valid_count} (NaN count: {len(df) - valid_count})")

        # Run benchmark
        results_df, best_model, holdout = run_benchmark_for_target(
            df, t['target'], feature_cols, test_size=0.20
        )

        print("\n  Comparison Table:")
        cols_to_show = ['Model', 'CV Test R² (mean ± std)', 'Holdout Train R²', 'Holdout Test R²', 'Holdout Test RMSE', 'Holdout Test MAE']
        print(results_df[cols_to_show].to_string(index=False))

        # Save table to CSV
        csv_out = os.path.join(results_dir, f"{t['name']}_comparison.csv")
        results_df.to_csv(csv_out, index=False)
        print(f"  Saved comparison table: {csv_out}")

        # Save best model
        model_out = os.path.join(models_dir, f"best_model_{t['name']}.pkl")
        with open(model_out, 'wb') as f:
            pickle.dump({'model': best_model, 'features': feature_cols, 'target': t['target']}, f)
        print(f"  Saved best model: {model_out}")

        # Diagnostic parity plot (Actual vs Predicted)
        X_train, X_test, y_train, y_test = holdout
        best_model.fit(X_train, y_train)
        y_pred = best_model.predict(X_test)

        plt.figure(figsize=(7, 6))
        plt.scatter(y_test, y_pred, color='#1f77b4', edgecolors='k', alpha=0.8, s=60, label='Test Instances')
        min_val = min(min(y_test), min(y_pred))
        max_val = max(max(y_test), max(y_pred))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Ideal 1:1 Parity')
        
        test_r2 = results_df.iloc[0]['Holdout Test R²']
        test_rmse = results_df.iloc[0]['Holdout Test RMSE']
        test_mae = results_df.iloc[0]['Holdout Test MAE']
        best_name = results_df.iloc[0]['Model']

        plt.title(f"{t['name'].replace('_', ' ')} Prediction ({best_name})\nTest R² = {test_r2:.3f}, RMSE = {test_rmse:.2f}{t['unit']}, MAE = {test_mae:.2f}{t['unit']}", fontsize=11)
        plt.xlabel(f"Actual {t['name'].replace('_', ' ')} ({t['unit']})", fontsize=11)
        plt.ylabel(f"Predicted {t['name'].replace('_', ' ')} ({t['unit']})", fontsize=11)
        plt.legend(frameon=True)
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout()
        plot_out = os.path.join(results_dir, f"{t['name']}_parity_plot.png")
        plt.savefig(plot_out, dpi=200)
        plt.close()
        print(f"  Saved parity plot: {plot_out}")

        all_summary[t['name']] = results_df.iloc[0].to_dict()

    print("\n" + "=" * 70)
    print("ALL REGRESSION BENCHMARKS COMPLETED!")
    print("=" * 70)


if __name__ == '__main__':
    main()
