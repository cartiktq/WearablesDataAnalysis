# --- File: /Project/Python/WearablesPredictionExportAgent.py ---
import pandas as pd

class WearablesPredictionExportAgent:
    def __init__(self, output_path='outputs/predictions.csv'):
        self.output_path = output_path

    def export(self, features_df, predictions, scores):
        features_df['prediction'] = predictions
        features_df['score'] = scores
        features_df.to_csv(self.output_path, index=False)
