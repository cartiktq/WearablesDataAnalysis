# uses isolation forest algorithm to detect outliers in averaged data 
# this class (agent) works with the output of FeatureExtractionAgent class

# --- File: /WearablesDataAnalysis/Python/WearablesAnomalyDetectionAgent.py ---
import os
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

class WearablesAnomalyDetectionAgent:
    def __init__(self, model_path='models/isolation_forest.joblib'):
        self.model_path = model_path
        if os.path.exists(model_path):
            self.model = joblib.load(model_path)
        else:
            self.model = Pipeline([
                ('scaler', StandardScaler()),
                ('clf', IsolationForest(contamination=0.05, random_state=42))
            ])

    def train(self, X):
        self.model.fit(X)
        joblib.dump(self.model, self.model_path)

    def predict(self, X):
        return self.model.predict(X), self.model.decision_function(X)
