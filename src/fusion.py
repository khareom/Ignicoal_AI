"""
Multimodal Sensor Fusion Module for IgniCoal AI.
Implements:
  1. Handcrafted Multi-Domain PA Features + Proximate Analysis Fusion
  2. Two-Stage Cascaded Fusion Pipeline:
     - Stage 1: Infer Proximate & Thermal Properties (Ash, Carbon, Tign) from PA Waveform
     - Stage 2: Concatenate Inferred Thermodynamics with Multi-Domain Spectral/Temporal Descriptors
     - Stage 3: High-Confidence SCS Risk Classification (Low, Moderate, High)
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


class MultimodalFusionPipeline:
    def __init__(
        self,
        ash_model,
        carbon_model,
        tign_model,
        classifier_type: str = 'ExtraTrees'
    ):
        self.ash_model = ash_model
        self.carbon_model = carbon_model
        self.tign_model = tign_model
        self.classifier_type = classifier_type
        self.scaler = StandardScaler()
        
        if classifier_type == 'GradientBoosting':
            self.classifier = GradientBoostingClassifier(
                n_estimators=150, learning_rate=0.08, max_depth=4, random_state=42
            )
        else:
            self.classifier = ExtraTreesClassifier(
                n_estimators=180, max_depth=10, min_samples_split=2, random_state=42
            )

    def augment_features_with_predictions(
        self,
        X_pa: np.ndarray,
        ash_features_df: pd.DataFrame,
        carbon_features_df: pd.DataFrame,
        thermal_features_df: pd.DataFrame
    ) -> np.ndarray:
        """
        Augments the raw/multi-domain PA feature matrix by appending
        predicted Ash Content, Fixed Carbon, and Ignition Temperature.
        """
        # Get predictions from stage 1 regression models
        pred_ash = self.ash_model.predict(ash_features_df.values).reshape(-1, 1)
        pred_carbon = self.carbon_model.predict(carbon_features_df.values).reshape(-1, 1)
        pred_tign = self.tign_model.predict(thermal_features_df.values).reshape(-1, 1)

        # Derived thermodynamic indices (Acoustic-Thermal Index, Volatility Proxy)
        fused_matrix = np.hstack([X_pa, pred_ash, pred_carbon, pred_tign])
        return fused_matrix

    def fit(self, X_fused: np.ndarray, y: np.ndarray):
        X_scaled = self.scaler.fit_transform(X_fused)
        self.classifier.fit(X_scaled, y)
        return self

    def predict(self, X_fused: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X_fused)
        return self.classifier.predict(X_scaled)

    def predict_proba(self, X_fused: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X_fused)
        return self.classifier.predict_proba(X_scaled)
