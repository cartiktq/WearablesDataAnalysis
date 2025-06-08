# Extracts 5 different patient data variables from wearables data and performs feature engineering
# Performing moving window averages on the raw data which are added to the input data frame containg 
# the wearables data
# --- File: /WearablesDataAnalysis/Python/FeatureEngineeringAgent.py ---
import pandas as pd

class FeatureEngineeringAgent:
    def __init__(self):
        pass

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy().dropna()
        window = 5
        features = {}
        for col in ['heart_rate', 'blood_pressure', 'spo2', 'skin_temp', 'pulse_rate']:
            features[f'{col}_mean'] = df[col].rolling(window=window, min_periods=1).mean().iloc[-1]
            features[f'{col}_std'] = df[col].rolling(window=window, min_periods=1).std().fillna(0).iloc[-1]
            features[f'{col}_delta'] = df[col].diff().mean()
        return pd.DataFrame([features])
