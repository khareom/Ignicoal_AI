"""
Data Loader module for IgniCoal AI.
Loads raw photoacoustic signal data from Low_SCS, Medium_SCS, and High_SCS files,
performs robust cleaning, aligns time axes, extracts metadata (Sample, Pellet, Shot),
and merges ground-truth proximate and thermal properties.
"""

import os
import re
import numpy as np
import pandas as pd


def clean_ground_truth(gt_path: str) -> pd.DataFrame:
    """
    Loads and cleans Ground_Truth.xlsx.
    Normalizes numeric columns (Moisture, VM, Ash, Fixed Carbon, Ignition Temp).
    Maps sample codes (e.g. C3 -> C3_1).
    """
    df = pd.read_excel(gt_path, header=None)
    # Row 0: ['Sample Code', 'Proximate_Analysis (Ground Truth)', ...]
    # Row 1: [NaN, 'Moisture', 'VM', 'Ash Content', 'Fixed Carbon Content', 'Ignition Temp', 'Mass Gain']
    col_names = ['Sample_Code', 'Moisture', 'VM', 'Ash_Content', 'Fixed_Carbon', 'Ignition_Temp', 'Mass_Gain']
    df_data = df.iloc[2:].copy()
    df_data = df_data.iloc[:, :7]
    df_data.columns = col_names
    
    # Strip string values
    for col in col_names:
        df_data[col] = df_data[col].astype(str).str.strip()
        
    # Clean Ignition Temp: remove '°C'
    df_data['Ignition_Temp'] = df_data['Ignition_Temp'].str.replace('°C', '', regex=False).str.strip()
    
    # Convert 'NA', 'NIL', 'nan' to np.nan and parse as float
    numeric_cols = ['Moisture', 'VM', 'Ash_Content', 'Fixed_Carbon', 'Ignition_Temp']
    for col in numeric_cols:
        df_data[col] = pd.to_numeric(df_data[col].replace({'NA': np.nan, 'NIL': np.nan, 'nan': np.nan}), errors='coerce')
        
    return df_data.reset_index(drop=True)


def parse_signal_header(col_name: str, default_coal: str = None) -> dict:
    """
    Parses column header format: INTENSITY_C#_P#_## or Intensity_P#_##
    Returns dict: {'sample_code': 'C#', 'pellet': 'P#', 'shot': '##'}
    """
    clean_name = str(col_name).strip()
    # Match INTENSITY_C#_P#_##
    match_full = re.search(r'INTENSITY_(C\d+)_([Pp]\d+)_(\d+)', clean_name, re.IGNORECASE)
    if match_full:
        return {
            'sample_code': match_full.group(1).upper(),
            'pellet': match_full.group(2).upper(),
            'shot': match_full.group(3)
        }
    
    # Match Intensity_P#_## (missing coal code)
    match_short = re.search(r'Intensity_([Pp]\d+)_(\d+)', clean_name, re.IGNORECASE)
    if match_short:
        coal = default_coal if default_coal else 'C1'
        return {
            'sample_code': coal.upper(),
            'pellet': match_short.group(1).upper(),
            'shot': match_short.group(2)
        }
        
    return None


def load_and_clean_signals(
    base_dir: str,
    ignore_c1: bool = True,
    target_length: int = 1306
) -> tuple[np.ndarray, list[dict]]:
    """
    Loads Low_SCS, Medium_SCS, and High_SCS Excel workbooks.
    Truncates all signals to target_length (default: 1306 samples = 26.10 microseconds at 50 MHz).
    Returns:
      - time_axis: 1D array of time stamps (seconds)
      - signal_records: list of dicts containing:
          'signal_id', 'signal', 'sample_code', 'pellet', 'shot',
          'scs_label', 'file_source'
    """
    file_configs = [
        {
            'filename': 'Low_SCS.xlsx',
            'scs_label': 'Low',
            'default_coal': 'C1',
            'time_max_row': 1307 # 1306 data points
        },
        {
            'filename': 'Medium_SCS.xlsx',
            'scs_label': 'Moderate',
            'default_coal': None,
            'time_max_row': 1310
        },
        {
            'filename': 'High_SCS.xlsx',
            'scs_label': 'High',
            'default_coal': None,
            'time_max_row': 1310
        }
    ]

    all_signals = []
    common_time_axis = None

    for cfg in file_configs:
        file_path = os.path.join(base_dir, cfg['filename'])
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Missing file: {file_path}")
            
        df = pd.read_excel(file_path)
        
        # 1. Identify Time column
        time_col = None
        for col in df.columns:
            if str(col).strip().lower() == 'time':
                time_col = col
                break
                
        if time_col is None:
            raise ValueError(f"Could not find Time column in {cfg['filename']}")
            
        # Clean Time column and establish valid time range
        time_series = pd.to_numeric(df[time_col], errors='coerce').dropna().values
        valid_rows = len(time_series)
        
        # Determine actual signal length to slice
        slice_len = min(valid_rows, target_length)
        if common_time_axis is None:
            common_time_axis = time_series[:slice_len]
            
        # 2. Iterate through signal columns
        col_counts = {}
        for col in df.columns:
            if col == time_col:
                continue
                
            col_str = str(col).strip()
            # Ignore empty columns and spurious columns like '112' or unnamed
            if not col_str or col_str == '112' or 'unnamed' in col_str.lower():
                continue
                
            meta = parse_signal_header(col_str, default_coal=cfg['default_coal'])
            if meta is None:
                continue
                
            # Filter out C1 signals if specified
            if ignore_c1 and meta['sample_code'] == 'C1':
                continue
                
            # Disambiguate repeated column names (e.g. replicate shots in High_SCS)
            base_col_id = f"{cfg['scs_label']}_{meta['sample_code']}_{meta['pellet']}_{meta['shot']}"
            col_counts[base_col_id] = col_counts.get(base_col_id, 0) + 1
            rep_idx = col_counts[base_col_id]
            signal_id = f"{base_col_id}_rep{rep_idx}" if rep_idx > 1 else base_col_id

            # Extract signal values and truncate to target length
            sig_values = pd.to_numeric(df[col], errors='coerce').values
            # Slice strictly within valid time axis and target_length
            sig_sliced = sig_values[:slice_len]
            
            # Check for NaN and handle
            if np.isnan(sig_sliced).any():
                # Linear interpolation for any rare missing points
                nans = np.isnan(sig_sliced)
                x = np.arange(len(sig_sliced))
                sig_sliced[nans] = np.interp(x[nans], x[~nans], sig_sliced[~nans])
                
            record = {
                'signal_id': signal_id,
                'signal': sig_sliced.astype(np.float64),
                'sample_code': meta['sample_code'],
                'pellet': meta['pellet'],
                'shot': meta['shot'],
                'scs_label': cfg['scs_label'],
                'file_source': cfg['filename']
            }
            all_signals.append(record)

    return common_time_axis, all_signals


def merge_ground_truth(signal_records: list[dict], gt_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merges signal metadata and ground truth proximate/thermal properties into a structured DataFrame.
    """
    # Create lookup map from gt_df
    # C3 in signals corresponds to C3_1 in Ground Truth
    gt_map = {}
    for _, row in gt_df.iterrows():
        code = row['Sample_Code']
        gt_map[code] = row
        if code == 'C3_1':
            gt_map['C3'] = row

    rows = []
    for rec in signal_records:
        sample_code = rec['sample_code']
        gt_data = gt_map.get(sample_code, {})
        
        entry = {
            'signal_id': rec['signal_id'],
            'sample_code': sample_code,
            'pellet': rec['pellet'],
            'shot': rec['shot'],
            'scs_label': rec['scs_label'],
            'file_source': rec['file_source'],
            'Ignition_Temp': gt_data.get('Ignition_Temp', np.nan),
            'Ash_Content': gt_data.get('Ash_Content', np.nan),
            'Fixed_Carbon': gt_data.get('Fixed_Carbon', np.nan),
            'Moisture': gt_data.get('Moisture', np.nan),
            'VM': gt_data.get('VM', np.nan)
        }
        rows.append(entry)

    return pd.DataFrame(rows)
