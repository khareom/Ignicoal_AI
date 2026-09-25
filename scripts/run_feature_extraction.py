"""
Script to run end-to-end data loading, signal cleaning, and feature extraction.
Saves processed feature datasets to data/processed/.
"""

import os
import sys
import numpy as np
import pandas as pd

# Add project root to sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.data_loader import clean_ground_truth, load_and_clean_signals, merge_ground_truth
from src.feature_extractor import FeatureExtractor


def main():
    print("=" * 60)
    print("IgniCoal AI: Feature Extraction & Dataset Generation")
    print("=" * 60)

    processed_dir = os.path.join(BASE_DIR, 'data', 'processed')
    os.makedirs(processed_dir, exist_ok=True)

    # 1. Load Ground Truth
    gt_path = os.path.join(BASE_DIR, 'Ground_Truth.xlsx')
    print(f"\n[1/4] Loading ground truth from {gt_path}...")
    gt_df = clean_ground_truth(gt_path)
    print(f"Loaded {len(gt_df)} ground truth sample records.")
    print("Ground Truth Preview (relevant samples):")
    relevant = ['C3_1', 'C4', 'C5', 'C6', 'C7', 'C9', 'C10', 'C12', 'C13', 'C14', 'C15', 'C16']
    print(gt_df[gt_df['Sample_Code'].isin(relevant)][['Sample_Code', 'Moisture', 'VM', 'Ash_Content', 'Fixed_Carbon', 'Ignition_Temp']])

    # 2. Load & Clean Raw Signals
    print(f"\n[2/4] Loading signals from Low_SCS, Medium_SCS, High_SCS (ignoring C1)...")
    time_axis, signal_records = load_and_clean_signals(BASE_DIR, ignore_c1=True, target_length=1306)
    print(f"Total usable signals extracted: {len(signal_records)}")
    print(f"Time axis length: {len(time_axis)} points (dt = {time_axis[1]-time_axis[0]:.2e} s, max_t = {time_axis[-1]:.2e} s)")

    # Sample distribution
    sample_counts = {}
    label_counts = {}
    for r in signal_records:
        sc = r['sample_code']
        lbl = r['scs_label']
        sample_counts[sc] = sample_counts.get(sc, 0) + 1
        label_counts[lbl] = label_counts.get(lbl, 0) + 1
        
    print(f"Signals per Susceptibility Class: {label_counts}")
    print(f"Signals per Coal Sample: {sorted(sample_counts.items())}")

    # 3. Merge Ground Truth
    meta_df = merge_ground_truth(signal_records, gt_df)

    # 4. Feature Extraction
    print(f"\n[3/4] Extracting multi-domain features for {len(signal_records)} signals...")
    fe = FeatureExtractor(sampling_rate=50e6)
    
    all_features_list = []
    thermal_subsets = []
    carbon_subsets = []
    ash_subsets = []

    for i, rec in enumerate(signal_records):
        sig = rec['signal']
        all_feats = fe.extract_all_features(sig, time_axis)
        all_features_list.append(all_feats)

        # Exact subsets from FeatureList_1.pptx
        # (p2p_tdb3 included in active thermal subset matching FeatureList_1.pptx)
        thermal_feats = fe.get_feature_subset(all_feats, 'thermal', include_p2p_tdb3=True)
        carbon_feats = fe.get_feature_subset(all_feats, 'carbon')
        ash_feats = fe.get_feature_subset(all_feats, 'ash')

        thermal_subsets.append(thermal_feats)
        carbon_subsets.append(carbon_feats)
        ash_subsets.append(ash_feats)

        if (i + 1) % 50 == 0 or (i + 1) == len(signal_records):
            print(f"  Processed {i + 1}/{len(signal_records)} signals...")

    # 5. Build and Save DataFrames
    print("\n[4/4] Assembling and saving processed datasets...")
    
    # Master dataset
    df_all_feats = pd.DataFrame(all_features_list)
    master_df = pd.concat([meta_df, df_all_feats], axis=1)
    master_csv = os.path.join(processed_dir, 'ignicoal_all_features.csv')
    master_df.to_csv(master_csv, index=False)
    print(f"  Saved master dataset: {master_csv} (Shape: {master_df.shape})")

    # Thermal dataset (Ignition Temp regression)
    df_thermal = pd.concat([meta_df[['signal_id', 'sample_code', 'scs_label', 'Ignition_Temp']], pd.DataFrame(thermal_subsets)], axis=1)
    thermal_csv = os.path.join(processed_dir, 'thermal_features.csv')
    df_thermal.to_csv(thermal_csv, index=False)
    print(f"  Saved Thermal features: {thermal_csv} (Shape: {df_thermal.shape})")

    # Carbon dataset (Fixed Carbon regression)
    df_carbon = pd.concat([meta_df[['signal_id', 'sample_code', 'scs_label', 'Fixed_Carbon']], pd.DataFrame(carbon_subsets)], axis=1)
    carbon_csv = os.path.join(processed_dir, 'carbon_features.csv')
    df_carbon.to_csv(carbon_csv, index=False)
    print(f"  Saved Carbon features: {carbon_csv} (Shape: {df_carbon.shape})")

    # Ash dataset (Ash Content regression)
    df_ash = pd.concat([meta_df[['signal_id', 'sample_code', 'scs_label', 'Ash_Content']], pd.DataFrame(ash_subsets)], axis=1)
    ash_csv = os.path.join(processed_dir, 'ash_features.csv')
    df_ash.to_csv(ash_csv, index=False)
    print(f"  Saved Ash features: {ash_csv} (Shape: {df_ash.shape})")

    print("\nFeature extraction completed successfully!")
    print("=" * 60)


if __name__ == '__main__':
    main()
