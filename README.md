# IgniCoal AI

> **AI-Assisted Photoacoustic Coal Analyzer**  
> Multimodal Sensor Fusion for Predicting Coal Spontaneous Combustion Susceptibility (SCS), Ash Content, Fixed Carbon Content, and Ignition Temperature.

---

## 📌 Overview

Coal spontaneous combustion in stockpiles and underground mines poses severe safety hazards and environmental risks. Traditional proximate analysis (ISO/ASTM proximate analysis and crossing point temperature methods) is destructive, labor-intensive, and takes days to yield results.

**IgniCoal AI** is an intelligent photoacoustic sensing platform that non-destructively characterizes coal samples in seconds. Utilizing pulsed Nd:YAG laser-induced ultrasound (photoacoustic signals) captured across wide bandwidths (0.2–1.3 MHz), the system extracts multimodal spectral and time-frequency representations to predict:

1. **Self-Combustion Susceptibility (SCS)**: Classified as **Low**, **Medium**, or **High**.
2. **Ash Content (%)**: Extra Trees Regressor ($R^2 = 82.20\%$).
3. **Fixed Carbon (%)**: Extra Trees Regressor ($R^2 = 72.84\%$).
4. **Ignition Temperature ($T_{\text{ign}}$ in $^\circ\text{C}$)**: Extra Trees Regressor ($R^2 = 93.10\%$).

---

## 🔬 Scientific Foundations

Based on research methodologies published in:
- *Photoacoustic Sensing in Sustainable Mining: Predicting Coal Susceptibility to Spontaneous Combustion* (IEEE SENSORS 2025)
- *Photoacoustic Time-Frequency Analysis for Quantification of Coal Thermal Properties and its Classification to Spontaneous Combustion Susceptibility* (IEEE Sensors Letters 2026)

### Signal Processing Pipeline
- **Artifact Blanking**: Initial $1.2\,\mu\text{s}$ excitation laser trigger artifact suppression.
- **Wavelet Denoising**: Symlet (`sym4`) multi-level wavelet decomposition and soft thresholding.
- **Filtering**: Savitzky-Golay polynomial smoothing filter.
- **Feature Extraction**: 98 engineered multimodal features spanning 5 sub-bands (0.2–1.3 MHz), acoustic velocity ($v = 0.006 / T_p$), peak arrival times, coda decay rates, and spectral centroids.

---

## 🚀 Quickstart & Local Installation

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
```

### 2. Create and activate a virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit Dashboard
```bash
streamlit run app/app.py
```

---

## 📂 Project Architecture

```text
├── app/
│   └── app.py                     # Streamlit web dashboard & visualization engine
├── data/
│   ├── processed/                 # Extracted feature CSVs
│   ├── Ground_Truth.xlsx          # Laboratory proximate analysis ground truth
│   ├── Low_SCS.xlsx               # Raw photoacoustic waveforms (Low SCS)
│   ├── Medium_SCS.xlsx            # Raw photoacoustic waveforms (Medium SCS)
│   └── High_SCS.xlsx              # Raw photoacoustic waveforms (High SCS)
├── models/
│   ├── saved/                     # Serialized production models (.pkl)
│   └── ...
├── src/
│   ├── preprocessor.py            # Artifact suppression, wavelet denoising, filtering
│   ├── feature_extractor.py       # Time-domain, frequency-domain & sub-band extraction
│   ├── data_loader.py             # Excel parsing and raw signal ingestion
│   ├── train_regressors.py        # Ash, Carbon, and T_ign model training & cross-validation
│   └── train_classifier.py        # Multimodal fusion classifier training & evaluation
├── requirements.txt               # Production dependencies
└── README.md
```

---

## 🌐 Deploy to Streamlit Community Cloud (Free 24/7 Hosting)

1. Push this repository to your GitHub account.
2. Visit [share.streamlit.io](https://share.streamlit.io) and log in with your GitHub account.
3. Click **"New app"**.
4. Select your repository, set the branch to `main`, and specify the main file path as:
   ```text
   app/app.py
   ```
5. Click **"Deploy!"** — your dashboard will be live on a permanent public URL (e.g. `https://<your-app-name>.streamlit.app`).
