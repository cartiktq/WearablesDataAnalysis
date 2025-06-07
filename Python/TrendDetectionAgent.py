#A Python class to implement agent using Agentic AI principles to analyze real-time streaming wearable health data. 
#It uses moving average filters and optionally other techniques (e.g., rolling standard deviation, thresholding) to 
#detect trends or anomalies in time-series signals such as heart rate, blood pressure, etc.

import pandas as pd
import numpy as np
from scipy.signal import savgol_filter

class TrendDetectionAgent:
    def __init__(self, window_size=5, std_dev_threshold=2):
        """
        :param window_size: Size of the moving window for trend calculation
        :param std_dev_threshold: Threshold for detecting anomalies based on standard deviation
        """
        self.window_size = window_size
        self.std_dev_threshold = std_dev_threshold
        self.tracked_signals = [
            "heart_rate", "blood_pressure", "skin_temp",
            "oxygen_saturation", "pulse_rate"
        ]

    def _apply_moving_average(self, series):
        return series.rolling(window=self.window_size, min_periods=1).mean()

    def _apply_std_deviation_filter(self, series):
        rolling_mean = series.rolling(window=self.window_size).mean()
        rolling_std = series.rolling(window=self.window_size).std()
        anomalies = abs(series - rolling_mean) > self.std_dev_threshold * rolling_std
        return anomalies.fillna(False)

    def _apply_savgol_filter(self, series):
        # Requires odd window size and at least 3
        win = max(3, self.window_size | 1)  # ensure odd
        return savgol_filter(series, window_length=win, polyorder=2, mode='nearest')

    def execute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Main method to perform trend detection on input streaming data.

        :param df: Input dataframe with time-series data, must include `timestamp` and `subject_id`
        :return: DataFrame with additional columns for smoothed trends and anomalies
        """
        df = df.sort_values(by=["subject_id", "timestamp"])
        result_df = df.copy()

        for signal in self.tracked_signals:
            if signal not in df.columns:
                continue

            smoothed_col = f"{signal}_trend"
            anomaly_col = f"{signal}_anomaly"

            result_df[smoothed_col] = (
                df.groupby("subject_id")[signal]
                .transform(self._apply_moving_average)
            )

            result_df[anomaly_col] = (
                df.groupby("subject_id")[signal]
                .transform(self._apply_std_deviation_filter)
            )

        return result_df
