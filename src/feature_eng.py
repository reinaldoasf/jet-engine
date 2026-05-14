import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class TelemetryFeatureEngineer:
    """
    A scikit-learn style transformer for ROV telemetry data.
    Handles dimensionality reduction (dropping flat sensors) and 
    temporal feature extraction (rolling windows).
    """
    def __init__(self, window_sizes: list[int] = [5, 10]):
        self.window_sizes = window_sizes
        self.useless_cols = []
        
    def fit(self, df: pd.DataFrame):
        """
        Learns which columns have zero variance from the training set so they 
        can be dropped consistently during both training and inference.
        """
        logger.info("Fitting Feature Engineer: Identifying zero-variance sensors...")
        
        # Identify static operating settings
        op_cols = ['op_setting_1', 'op_setting_2', 'op_setting_3']
        
        # Identify flatlined sensors (< 0.005 standard deviation)
        sensor_cols = [col for col in df.columns if 'sensor' in col]
        sensor_std = df[sensor_cols].std()
        flat_sensors = sensor_std[sensor_std < 0.005].index.tolist()
        
        self.useless_cols = op_cols + flat_sensors
        logger.info(f"Identified {len(self.useless_cols)} useless columns to drop.")
        return self
    