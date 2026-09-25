"""
IgniCoal AI — On-Field Handheld Photoacoustic Coal Analyzer
Interactive UI / Visualization Dashboard for Real-Time Coal Property Prediction
and Spontaneous Combustion Susceptibility (SCS) Risk Classification.
"""

import os
import re
import sys
import pickle
import numpy as np
import pandas as pd
import streamlit as st

try:
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False

# Ensure root is in PATH
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.preprocessor import SignalPreprocessor
from src.feature_extractor import FeatureExtractor
from src.data_loader import load_and_clean_signals

st.set_page_config(
    page_title="Ignicoal AI",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Minimalist Black/White Coal Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200');

    /* Global Typography: Optimistic VF, Helvetica, Arial, sans-serif */
    html, body, .stApp, p, h1, h2, h3, h4, h5, h6, input, select, textarea {
        font-family: "Optimistic VF", Helvetica, Arial, sans-serif !important;
    }
    .stMarkdown, .stText, .stDataFrame, .metric-title, .metric-value, .metric-subtitle, .main-title, .main-subtitle, .hero-card {
        font-family: "Optimistic VF", Helvetica, Arial, sans-serif !important;
    }

    /* Preserve Material Symbols & Icons font (Prevents raw ligature text like keyboard_double_arrow_right) */
    .material-symbols-rounded,
    .material-symbols-outlined,
    .material-symbols-sharp,
    .material-icons,
    [data-testid="stIconMaterial"],
    [data-testid*="Icon"],
    [data-testid="stExpandSidebarButton"] span,
    [data-testid="stSidebarCollapsedControl"] span,
    [data-testid="stSidebarCollapseButton"] span,
    [class*="material-symbols"] {
        font-family: 'Material Symbols Rounded', 'Material Symbols Outlined', 'Material Icons' !important;
        font-weight: normal !important;
        font-style: normal !important;
        font-size: 22px !important;
        line-height: 1 !important;
        letter-spacing: normal !important;
        text-transform: none !important;
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
        white-space: nowrap !important;
        word-wrap: normal !important;
        direction: ltr !important;
        -webkit-font-feature-settings: 'liga' 1 !important;
        font-feature-settings: 'liga' 1 !important;
        -webkit-font-smoothing: antialiased !important;
        color: #E5E5E5 !important;
    }

    /* Non-blocking Transparent Header */
    header[data-testid="stHeader"] {
        background: transparent !important;
        height: 2.75rem !important;
        z-index: 999 !important;
        pointer-events: none !important;
    }
    header[data-testid="stHeader"] button,
    [data-testid="stExpandSidebarButton"],
    [data-testid="stSidebarCollapsedControl"],
    [data-testid="stSidebarCollapseButton"] {
        pointer-events: auto !important;
    }

    /* Floating Collapsible Sidebar Toggle Button (Mobile & Desktop) */
    [data-testid="stExpandSidebarButton"],
    [data-testid="stSidebarCollapsedControl"] {
        display: flex !important;
        visibility: visible !important;
        opacity: 1 !important;
        top: 0.55rem !important;
        left: 0.75rem !important;
        z-index: 1001 !important;
    }
    [data-testid="stExpandSidebarButton"] button,
    [data-testid="stSidebarCollapsedControl"] button {
        background-color: #171717 !important;
        border: 1px solid #383838 !important;
        border-radius: 8px !important;
        color: #FFFFFF !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.6) !important;
        padding: 5px 9px !important;
        transition: all 0.2s ease !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
    }
    [data-testid="stExpandSidebarButton"] button:hover,
    [data-testid="stSidebarCollapsedControl"] button:hover {
        background-color: #242424 !important;
        border-color: #555555 !important;
        color: #FFFFFF !important;
    }
    [data-testid="stExpandSidebarButton"] svg,
    [data-testid="stSidebarCollapsedControl"] svg,
    [data-testid="stSidebarCollapseButton"] svg {
        fill: #E5E5E5 !important;
        stroke: #E5E5E5 !important;
        color: #E5E5E5 !important;
    }

    .block-container {
        padding-top: 1.0rem !important;
        padding-bottom: 2rem !important;
        max-width: 96% !important;
    }
    
    /* Sleek Sidebar Header with Functional Collapse Button */
    [data-testid="stSidebarHeader"] {
        background: transparent !important;
        padding-top: 0.35rem !important;
        padding-bottom: 0.2rem !important;
        padding-left: 0.75rem !important;
        padding-right: 0.75rem !important;
        min-height: 2.2rem !important;
        display: flex !important;
        justify-content: flex-end !important;
        align-items: center !important;
    }
    [data-testid="stSidebarCollapseButton"] button {
        color: #A3A3A3 !important;
        background: #141414 !important;
        border: 1px solid #282828 !important;
        border-radius: 6px !important;
        padding: 4px 8px !important;
        transition: all 0.2s ease !important;
    }
    [data-testid="stSidebarCollapseButton"] button:hover {
        color: #FFFFFF !important;
        border-color: #4A4A4A !important;
        background: #202020 !important;
    }

    [data-testid="stSidebar"] {
        background-color: #0C0C0C !important;
        border-right: 1px solid #222222 !important;
    }
    [data-testid="stSidebarUserContent"] {
        padding-top: 0.3rem !important;
        padding-left: 1.2rem !important;
        padding-right: 1.2rem !important;
    }
    [data-testid="stSidebar"] .block-container {
        padding-top: 0.3rem !important;
    }

    /* Consistent Sidebar Typography & Clean Spacing */
    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] > div {
        margin-bottom: 0.85rem !important;
    }
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] label p,
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
        font-size: 0.84rem !important;
        font-weight: 600 !important;
        color: #A3A3A3 !important;
        letter-spacing: 0.01em !important;
        margin-bottom: 5px !important;
        text-transform: none !important;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > label {
        margin-bottom: 5px !important;
        padding: 2px 0 !important;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label p,
    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] p {
        font-size: 0.85rem !important;
        color: #D4D4D4 !important;
        font-weight: 400 !important;
    }
    [data-testid="stSidebar"] [data-baseweb="select"] > div {
        background-color: #141414 !important;
        border: 1px solid #282828 !important;
        border-radius: 8px !important;
        padding: 1px 4px !important;
        margin-top: 2px !important;
    }
    [data-testid="stSidebar"] [data-baseweb="select"] > div:hover {
        border-color: #404040 !important;
    }
    [data-testid="stSidebar"] [data-baseweb="select"] div,
    [data-testid="stSidebar"] [data-baseweb="select"] span {
        font-size: 0.85rem !important;
        color: #FFFFFF !important;
    }
    div[data-baseweb="popover"] ul li,
    div[data-baseweb="menu"] ul li {
        font-size: 0.85rem !important;
    }
    [data-testid="stSidebar"] [data-testid="stFileUploader"] {
        margin-top: 6px !important;
    }
    [data-testid="stSidebar"] [data-testid="stFileUploader"] span,
    [data-testid="stSidebar"] [data-testid="stFileUploader"] small,
    [data-testid="stSidebar"] [data-testid="stFileUploader"] button {
        font-size: 0.80rem !important;
    }
    [data-testid="stSidebar"] p {
        font-size: 0.85rem !important;
    }

    /* Deep Coal Monochrome Canvas */
    .stApp {
        background: linear-gradient(180deg, #121212 0%, #0A0A0A 50%, #020202 100%) !important;
        color: #E5E5E5 !important;
    }
    
    /* Title Styling (Minimalist White-to-Silver Gradient) */
    .main-title {
        text-align: center;
        margin-top: 0px !important;
        margin-bottom: 2px !important;
        padding-top: 0px !important;
        font-size: 2.65rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        background: linear-gradient(180deg, #FFFFFF 20%, #E2E8F0 65%, #94A3B8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .main-subtitle {
        text-align: center;
        color: #888888;
        font-size: 0.92rem;
        font-weight: 600;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin-top: 0px;
        margin-bottom: 18px;
    }

    /* Minimalist Coal Hero Card (NO Rainbow Border) */
    .hero-card {
        background: linear-gradient(180deg, #161616 0%, #0E0E0E 100%);
        border: 1px solid #262626;
        border-radius: 12px;
        padding: 20px 26px;
        margin-bottom: 22px;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.5);
    }
    .hero-headline {
        font-size: 1.5rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        background: linear-gradient(180deg, #FFFFFF 0%, #D4D4D4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 4px;
    }
    .hero-subhead {
        font-size: 1.02rem;
        font-weight: 600;
        color: #9E9E9E;
        margin-bottom: 12px;
    }
    .hero-body {
        font-size: 0.96rem;
        line-height: 1.65;
        color: #888888;
    }
            
    .hero-body b {
        color: #888888;
        font-weight: 600;
    }
    .hero-footer {
        display: flex;
        align-items: center;
        gap: 8px;
        margin-top: 14px;
        padding-top: 12px;
        border-top: 1px solid #222222;
        font-size: 0.92rem;
        font-weight: 600;
        color: #E5E5E5;
        letter-spacing: 0.02em;
    }
    .pulse-dot {
        width: 7px;
        height: 7px;
        background-color: #FFFFFF;
        border-radius: 50%;
        box-shadow: 0 0 8px rgba(255, 255, 255, 0.7);
        display: inline-block;
    }
    .pill {
        padding: 2px 9px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.85rem;
        display: inline-block;
        margin: 0 2px;
    }
    .pill-low {
        background: #0E2214;
        color: #4ADE80;
        border: 1px solid #166534;
    }
    .pill-med {
        background: #241604;
        color: #FBBF24;
        border: 1px solid #854D0E;
    }
    .pill-high {
        background: #260A0A;
        color: #F87171;
        border: 1px solid #991B1B;
    }

    /* Minimalist Dark Metric Cards */
    .metric-card {
        background: linear-gradient(180deg, #181818 0%, #101010 100%);
        border: 1px solid #282828;
        border-radius: 12px;
        padding: 18px 20px;
        margin-bottom: 12px;
        box-shadow: 0 6px 18px rgba(0, 0, 0, 0.4);
        text-align: center;
        transition: border-color 0.2s ease-in-out;
    }
    .metric-card:hover {
        border-color: #454545;
    }
    .metric-title {
        color: #A3A3A3;
        font-size: 0.85rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 8px;
    }
    .metric-value {
        color: #FFFFFF;
        font-size: 1.95rem;
        font-weight: 800;
        margin-top: 4px;
        margin-bottom: 4px;
    }
    .metric-subtitle {
        color: #737373;
        font-size: 0.80rem;
        font-weight: 500;
        margin-top: 4px;
    }
    
    /* Badges */
    .badge-low {
        background: #0E2214;
        color: #4ADE80;
        border: 1px solid #166534;
        padding: 5px 16px;
        border-radius: 9999px;
        font-weight: 800;
        display: inline-block;
        font-size: 1.05rem;
        letter-spacing: 0.05em;
    }
    .badge-moderate {
        background: #241604;
        color: #FBBF24;
        border: 1px solid #854D0E;
        padding: 5px 16px;
        border-radius: 9999px;
        font-weight: 800;
        display: inline-block;
        font-size: 1.05rem;
        letter-spacing: 0.05em;
    }
    .badge-high {
        background: #260A0A;
        color: #F87171;
        border: 1px solid #991B1B;
        padding: 5px 16px;
        border-radius: 9999px;
        font-weight: 800;
        display: inline-block;
        font-size: 1.05rem;
        letter-spacing: 0.05em;
    }

    /* Mobile Responsive Optimizations */
    @media (max-width: 768px) {
        .block-container {
            padding-top: 3.2rem !important;
            padding-left: 0.75rem !important;
            padding-right: 0.75rem !important;
        }
        .main-title {
            font-size: 1.85rem !important;
            margin-top: 0.2rem !important;
        }
        .main-subtitle {
            font-size: 0.78rem !important;
            letter-spacing: 0.08em !important;
            margin-bottom: 14px !important;
        }
        .hero-card {
            padding: 14px 16px !important;
            margin-bottom: 16px !important;
        }
        [data-testid="stExpandSidebarButton"],
        [data-testid="stSidebarCollapsedControl"] {
            top: 0.5rem !important;
            left: 0.5rem !important;
            box-shadow: 0 4px 14px rgba(0, 0, 0, 0.7) !important;
        }
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_all_models_and_data():
    models_dir = os.path.join(BASE_DIR, 'models', 'saved')
    data_dir = os.path.join(BASE_DIR, 'data', 'processed')
    
    with open(os.path.join(models_dir, 'best_model_Ash_Content.pkl'), 'rb') as f:
        ash_pkg = pickle.load(f)
    with open(os.path.join(models_dir, 'best_model_Fixed_Carbon.pkl'), 'rb') as f:
        carbon_pkg = pickle.load(f)
    with open(os.path.join(models_dir, 'best_model_Ignition_Temperature.pkl'), 'rb') as f:
        tign_pkg = pickle.load(f)
    with open(os.path.join(models_dir, 'best_model_SCS_Classifier.pkl'), 'rb') as f:
        clf_pkg = pickle.load(f)
        
    time_axis, signals = load_and_clean_signals(BASE_DIR, ignore_c1=True)
    master_df = pd.read_csv(os.path.join(data_dir, 'ignicoal_all_features.csv'))
    
    return {
        'ash_pkg': ash_pkg,
        'carbon_pkg': carbon_pkg,
        'tign_pkg': tign_pkg,
        'clf_pkg': clf_pkg,
        'time_axis': time_axis,
        'signals': signals,
        'master_df': master_df
    }


def format_feature_val(feat_name, val):
    if val is None or pd.isna(val):
        return "N/A"
    feat_lower = feat_name.lower()
    if 'time' in feat_lower or 'duration' in feat_lower:
        if abs(val) < 1e-3:
            return f"{val * 1e6:.2f} µs"
        return f"{val:.4f} s"
    elif 'velocity' in feat_lower:
        return f"{val:.0f} m/s"
    elif 'centroid' in feat_lower or 'bandwidth' in feat_lower or 'freq' in feat_lower:
        return f"{val:.4f} MHz"
    elif 'entropy' in feat_lower:
        return f"{val:.4f}"
    elif 'area' in feat_lower or 'fft' in feat_lower or 'energy' in feat_lower:
        if abs(val) < 1e-3:
            return f"{val * 1e6:.2f} µV·MHz"
        return f"{val:.4f} mV·MHz"
    elif 'rms' in feat_lower or 'p2p' in feat_lower or 'absorption' in feat_lower:
        if abs(val) < 1e-3:
            return f"{val * 1e6:.2f} µV"
        return f"{val * 1e3:.3f} mV"
    else:
        if abs(val) < 1e-3 and val != 0:
            return f"{val:.3e}"
        return f"{val:.4f}"


def get_model_feature_importances(model_obj, num_features: int):
    """
    Safely extracts feature importances from a raw estimator, Pipeline, or VotingRegressor ensemble.
    Falls back to uniform weights if not directly available.
    """
    if hasattr(model_obj, 'named_steps'):
        if 'regressor' in model_obj.named_steps:
            estimator = model_obj.named_steps['regressor']
        elif 'classifier' in model_obj.named_steps:
            estimator = model_obj.named_steps['classifier']
        else:
            estimator = model_obj.steps[-1][1]
    elif hasattr(model_obj, 'steps'):
        estimator = model_obj.steps[-1][1]
    else:
        estimator = model_obj

    if hasattr(estimator, 'feature_importances_'):
        return estimator.feature_importances_
    elif hasattr(estimator, 'estimators_'):
        imps = []
        for est in estimator.estimators_:
            if hasattr(est, 'feature_importances_'):
                imps.append(est.feature_importances_)
        if imps:
            return np.mean(imps, axis=0)
    elif hasattr(estimator, 'coef_'):
        coef = np.abs(estimator.coef_).ravel()
        total = np.sum(coef)
        return coef / (total + 1e-12)

    return np.ones(num_features) / max(num_features, 1)


def main():
    assets = load_all_models_and_data()
    signals = assets['signals']
    time_axis = assets['time_axis']
    master_df = assets['master_df']
    
    prep = SignalPreprocessor()
    fe = FeatureExtractor()

    # Title & Subcaption (Centered with modern tech styling)
    st.markdown("<h1 class='main-title'>Ignicoal AI</h1>", unsafe_allow_html=True)
    st.markdown("<p class='main-subtitle'>AI Assisted Photoacoustic Coal Analyzer</p>", unsafe_allow_html=True)
# <div class="hero-subhead">From a small sample to critical insights.</div>

    # Hero Card: "Know Your Coal. Before It Knows You."
    st.markdown("""
    <div class="hero-card">
        <div class="hero-headline">Know Your Coal. Before It Knows You.</div>
        <div class="hero-body">
            Our <b>photoacoustic sensing technology</b> rapidly analyzes your coal sample to determine its 
            <b>ash content</b>, <b>fixed carbon content</b>, and <b>ignition temperature</b>, 
            while also identifying its <b>self-combustion susceptibility</b>.
        </div>
        <div class="hero-footer">
            <span class="pulse-dot"></span> Simple sample. <span class="pulse-dot"></span>Powerful insights. <span class="pulse-dot"></span>Safer decisions.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar: Sample Selection
    st.sidebar.markdown("""
    <div style='color: #FFFFFF; font-size: 1.12rem; font-weight: 700; margin: 0 0 6px 0; padding: 0;'>
        Field Telemetry Ingestion
    </div>
    <div style='height: 1px; background: #262626; margin-bottom: 12px;'></div>
    """, unsafe_allow_html=True)
    
    mode = st.sidebar.radio("Data Mode", ["Stockyard Database Sample", "Upload Custom Sensor Data (.csv/.xlsx)"])
    st.sidebar.markdown("<div style='height: 1px; background: #262626; margin: 10px 0 12px 0;'></div>", unsafe_allow_html=True)
    
    selected_sig = None
    sample_id = None
    meta_info = {}

    if mode == "Stockyard Database Sample":
        def natural_sort_key(s):
            return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', str(s))]

        coal_samples = sorted(list(set(s['sample_code'] for s in signals)), key=natural_sort_key)
        selected_coal = st.sidebar.selectbox("Select Coal Sample ID", coal_samples, index=0)
        
        subset_sigs = sorted([s for s in signals if s['sample_code'] == selected_coal], key=lambda x: natural_sort_key(x['signal_id']))
        sig_ids = [s['signal_id'] for s in subset_sigs]
        sample_id = st.sidebar.selectbox("Select Shot Instance", sig_ids)
        
        target_sig_obj = [s for s in subset_sigs if s['signal_id'] == sample_id][0]
        selected_sig = target_sig_obj['signal']
        meta_info = {
            'coal_sample': target_sig_obj['sample_code'],
            'pellet': target_sig_obj['pellet'],
            'shot': target_sig_obj['shot'],
            'ground_truth_scs': target_sig_obj['scs_label'],
            'file_source': target_sig_obj['file_source']
        }
    else:
        uploaded_file = st.sidebar.file_uploader("Upload Sensor Trace", type=['csv', 'xlsx'])
        if uploaded_file is not None:
            if uploaded_file.name.endswith('.csv'):
                up_df = pd.read_csv(uploaded_file)
            else:
                up_df = pd.read_excel(uploaded_file)
            
            # Use second column if time column is present, else first column
            col_idx = 1 if up_df.shape[1] > 1 else 0
            raw_vals = pd.to_numeric(up_df.iloc[:, col_idx], errors='coerce').dropna().values.astype(np.float64)
            if len(raw_vals) < len(time_axis):
                selected_sig = np.pad(raw_vals, (0, len(time_axis) - len(raw_vals)), mode='edge')
            else:
                selected_sig = raw_vals[:len(time_axis)]
            sample_id = uploaded_file.name
            meta_info = {'coal_sample': 'Uploaded Field Trace', 'pellet': 'Custom', 'shot': '1', 'ground_truth_scs': 'Unknown'}
        else:
            st.info("Please select a database sample or upload a signal trace from the sidebar.")
            return

    # Process Signal & Extract Features
    proc_sig = prep.process_signal(selected_sig, remove_trigger=True)
    all_feats = fe.extract_all_features(selected_sig, time_axis)
    v_act = all_feats['acoustic_velocity']
    
    ash_sub = fe.get_feature_subset(all_feats, 'ash')
    carbon_sub = fe.get_feature_subset(all_feats, 'carbon')
    thermal_sub = fe.get_feature_subset(all_feats, 'thermal')

    # Run ML Model Inference using exact trained features
    ash_input = pd.DataFrame([ash_sub])[assets['ash_pkg']['features']].values
    carbon_input = pd.DataFrame([carbon_sub])[assets['carbon_pkg']['features']].values
    thermal_input = pd.DataFrame([thermal_sub])[assets['tign_pkg']['features']].values

    ash_pred = float(assets['ash_pkg']['model'].predict(ash_input)[0])
    carbon_pred = float(assets['carbon_pkg']['model'].predict(carbon_input)[0])
    tign_pred = float(assets['tign_pkg']['model'].predict(thermal_input)[0])

    # Run SCS Classifier Inference using exact trained features
    clf_input = pd.DataFrame([all_feats])[assets['clf_pkg']['features']].values
    scs_pred_idx = assets['clf_pkg']['model'].predict(clf_input)[0]
    scs_proba = assets['clf_pkg']['model'].predict_proba(clf_input)[0]
    scs_pred_class = assets['clf_pkg']['encoder'].inverse_transform([scs_pred_idx])[0]

    # Display Top Telemetry Cards (4 Metrics)
    col1, col2, col3, col4 = st.columns(4)
    badge_class = f"badge-{scs_pred_class.lower()}"
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Predicted Susceptibility</div>
            <div class="metric-value"><span class="{badge_class}">{scs_pred_class.upper()} RISK</span></div>
            <div class="metric-subtitle">Confidence: {np.max(scs_proba)*100:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Ignition Temperature</div>
            <div class="metric-value">{tign_pred:.1f} °C</div>
            <div class="metric-subtitle">Crossing Point / DTGA</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Fixed Carbon Content</div>
            <div class="metric-value">{carbon_pred:.1f}%</div>
            <div class="metric-subtitle">Combustible Matrix</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Ash Content</div>
            <div class="metric-value">{ash_pred:.1f}%</div>
            <div class="metric-subtitle">Thermal Inertia Suppressor</div>
        </div>
        """, unsafe_allow_html=True)

    # Detailed Tabs
    tab1, tab2, tab3 = st.tabs([
        "Risk Assessment",
        "Your Signal",
        "Our Models"
    ])

    with tab1:

        # st.subheader("Coal Stockyard Spontaneous Combustion Risk Assessment")
        
        prob_df = pd.DataFrame({
            'SCS Susceptibility Class': assets['clf_pkg']['classes'],
            'Inference Probability': scs_proba
        })
        
        c_left, c_right = st.columns([1, 1.2])
        with c_left:
            st.markdown("#### Classification Confidence")
            if HAS_MPL:
                fig_p, ax_p = plt.subplots(figsize=(5, 3.5))
                c_map = {'Low': '#22c55e', 'Moderate': '#f59e0b', 'High': '#ef4444'}
                bar_colors = [c_map.get(c, '#3a86ff') for c in prob_df['SCS Susceptibility Class']]
                bars = ax_p.bar(prob_df['SCS Susceptibility Class'], prob_df['Inference Probability'] * 100, color=bar_colors, edgecolor='black', alpha=0.85)
                ax_p.set_ylabel('Confidence (%)', fontweight='bold')
                ax_p.set_ylim(0, 105)
                for bar in bars:
                    h = bar.get_height()
                    ax_p.text(bar.get_x() + bar.get_width()/2, h + 2, f"{h:.1f}%", ha='center', fontweight='bold', fontsize=10)
                ax_p.grid(axis='y', linestyle=':', alpha=0.6)
                plt.tight_layout()
                st.pyplot(fig_p)
                plt.close()
            else:
                prob_plot = prob_df.copy()
                prob_plot['Confidence (%)'] = prob_plot['Inference Probability'] * 100
                st.bar_chart(prob_plot.set_index('SCS Susceptibility Class')['Confidence (%)'])

        with c_right:
            st.markdown("#### Actionable Mine Management Advisory")
            if scs_pred_class == 'High':
                st.error("""
                **HIGH SPONTANEOUS COMBUSTION SUSCEPTIBILITY DETECTED**
                - **Incubation Window:** Rapid self-heating liable within **15–30 days**.
                - **Stockyard Strategy:** High evacuation priority. Rotate and dispatch to boiler feed immediately.
                - **Fire Mitigation Protocol:** Apply water-mist surface compaction and spray anti-oxidant inhibitors (e.g. $\\text{MgCl}_2$ / chemical fire retardants). Prevent air ingress through pile compaction.
                """)
            elif scs_pred_class == 'Moderate':
                st.warning("""
                **MODERATE SPONTANEOUS COMBUSTION SUSCEPTIBILITY**
                - **Incubation Window:** Safe storage incubation span up to **60–90 days**.
                - **Stockyard Strategy:** Scheduled rotational turnover. Continuous infrared thermal imaging surveillance recommended.
                - **Fire Mitigation Protocol:** Avoid loose dumping; dress pile batters to reduce chimney oxidation effects.
                """)
            else:
                st.success("""
                **LOW SPONTANEOUS COMBUSTION SUSCEPTIBILITY (STABLE COAL)**
                - **Incubation Window:** High ignition temperature ($T_{\\text{ign}} > 410^\\circ\\text{C}$); prolonged stable stockpile endurance (> 180 days).
                - **Stockyard Strategy:** Suitable for strategic reserve buffering and long-distance bulk rail transit.
                - **Fire Mitigation Protocol:** Standard stockpile maintenance.
                """)
        

    with tab2:

        st.subheader("Time-Resolved Photoacoustic Waveform")
        t_us = time_axis * 1e6
        
        if HAS_MPL:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
            
            # Raw Signal
            ax1.plot(t_us, selected_sig * 1000, color='#e63946', lw=1.2, label='Raw Sensor Signal')
            ax1.axvspan(0, 1.2, color='gray', alpha=0.25, label='Trigger Artifact Region (0 - 1.2 µs)')
            ax1.axvline(0.06, color='black', linestyle='--', alpha=0.7, label='Trigger Pulse Spike (60 ns)')
            ax1.set_ylabel('Amplitude (mV)', fontweight='bold')
            ax1.set_title(f'Raw Signal', fontweight='bold', fontsize=11)
            ax1.legend(loc='upper right', frameon=True)
            ax1.grid(True, linestyle=':', alpha=0.5)

            # Processed Signal
            ax2.plot(t_us, proc_sig * 1000, color='#3a86ff', lw=1.5, label='Trigger-Blanked & Denoised PA Wave')
            tp_us = all_feats['peak_time_proc'] * 1e6
            ax2.plot(tp_us, proc_sig[int(tp_us*50)] * 1000, 'ro', markersize=8, label=f'Acoustic Peak Arrival (Tp = {tp_us:.2f} µs)')
            ax2.set_xlabel('Time (µs)', fontweight='bold')
            ax2.set_ylabel('Amplitude (mV)', fontweight='bold')
            ax2.set_title('Processed Photoacoustic Signal', fontweight='bold', fontsize=11)
            ax2.legend(loc='upper right', frameon=True)
            ax2.grid(True, linestyle=':', alpha=0.5)

            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        else:
            wave_df = pd.DataFrame({
                'Raw Signal (mV)': selected_sig * 1000,
                'Processed Signal (mV)': proc_sig * 1000
            }, index=np.round(t_us, 2))
            st.line_chart(wave_df)

        # Top 4 Ranked Features per Property
        st.markdown("#### Top Ranked Predictive Features per Target Property")
        st.caption("Key photoacoustic features ranked by the best model relevance along with their extracted values for this signal instance:")

        col_ash, col_carb, col_tign = st.columns(3)

        with col_ash:
            st.markdown("##### Ash Content Features")
            ash_imps = get_model_feature_importances(assets['ash_pkg']['model'], len(assets['ash_pkg']['features']))
            ash_ranked = sorted(zip(assets['ash_pkg']['features'], ash_imps), key=lambda x: x[1], reverse=True)[:4]
            ash_df = pd.DataFrame([
                {
                    'Rank': f"{i}",
                    'Feature': feat,
                    'Weight': f"{imp*100:.1f}%",
                    'Value': format_feature_val(feat, ash_sub[feat])
                }
                for i, (feat, imp) in enumerate(ash_ranked, 1)
            ])
            st.dataframe(ash_df, hide_index=True)

        with col_carb:
            st.markdown("##### Fixed Carbon Features")
            carb_imps = get_model_feature_importances(assets['carbon_pkg']['model'], len(assets['carbon_pkg']['features']))
            carb_ranked = sorted(zip(assets['carbon_pkg']['features'], carb_imps), key=lambda x: x[1], reverse=True)[:4]
            carb_df = pd.DataFrame([
                {
                    'Rank': f"{i}",
                    'Feature': feat,
                    'Weight': f"{imp*100:.1f}%",
                    'Value': format_feature_val(feat, carbon_sub[feat])
                }
                for i, (feat, imp) in enumerate(carb_ranked, 1)
            ])
            st.dataframe(carb_df, hide_index=True)

        with col_tign:
            st.markdown("##### Ignition Temp Features")
            tign_imps = get_model_feature_importances(assets['tign_pkg']['model'], len(assets['tign_pkg']['features']))
            tign_ranked = sorted(zip(assets['tign_pkg']['features'], tign_imps), key=lambda x: x[1], reverse=True)[:4]
            tign_df = pd.DataFrame([
                {
                    'Rank': f"{i}",
                    'Feature': feat,
                    'Weight': f"{imp*100:.1f}%",
                    'Value': format_feature_val(feat, thermal_sub[feat])
                }
                for i, (feat, imp) in enumerate(tign_ranked, 1)
            ])
            st.dataframe(tign_df, hide_index=True)
        

    with tab3:

        # st.subheader("Research Performance Benchmarks & Presentation Visuals")
        # st.markdown("Comprehensive performance metrics validating against TCS Research Papers (IEEE SENSORS 2025 & IEEE Sensors Letters 2026).")

        tab_m1, tab_m2 = st.tabs(["Regression Benchmarks", "Classification Benchmarks"])
        
        with tab_m1:
            st.markdown("#### Regression Model Comparisons")
            reg_dir = os.path.join(BASE_DIR, 'results', 'regression')
            for tname in ['Fixed_Carbon', 'Ash_Content', 'Ignition_Temperature']:
                csv_p = os.path.join(reg_dir, f"{tname}_comparison.csv")
                if os.path.exists(csv_p):
                    st.markdown(f"**Target: {tname.replace('_', ' ')}**")
                    st.dataframe(pd.read_csv(csv_p), hide_index=True)

            st.markdown("#### Regression Prediction Parity Plots")
            p_reg = os.path.join(BASE_DIR, 'results', 'presentation', 'regression_dashboard_3x1.png')
            if os.path.exists(p_reg):
                st.image(p_reg, caption="Regression Prediction Parity Plots", use_container_width=True)

        with tab_m2:
            st.markdown("#### SCS Classification Model Comparisons")
            clf_csv = os.path.join(BASE_DIR, 'results', 'classification', 'classification_comparison.csv')
            if os.path.exists(clf_csv):
                clf_df = pd.read_csv(clf_csv)
                clean_cols = [c for c in clf_df.columns if not c.startswith('raw_')]
                st.dataframe(clf_df[clean_cols], hide_index=True)

            st.markdown("#### Model Accuracy Comparison & Confusion Matrix")
            col_c1, col_c2 = st.columns([1.4, 1.0])
            acc_p = os.path.join(BASE_DIR, 'results', 'classification', 'model_accuracy_comparison.png')
            cm_p = os.path.join(BASE_DIR, 'results', 'classification', 'confusion_matrix_fused.png')
            
            with col_c1:
                if os.path.exists(acc_p):
                    st.image(acc_p, caption="Model Accuracy Comparison", use_container_width=True)
            with col_c2:
                if os.path.exists(cm_p):
                    st.image(cm_p, caption="Confusion Matrix", use_container_width=True)

        
if __name__ == '__main__':
    main()

