from agentic import Agent
from sklearn.metrics import recall_score
import numpy as np

class DeploymentMonitorAgent(Agent):
    def __init__(self, model_a, model_b, data_stream, outcomes_fetcher):
        self.model_a = model_a  # Champion
        self.model_b = model_b  # Challenger
        self.data_stream = data_stream
        self.outcomes_fetcher = outcomes_fetcher

    def run(self):
        print("Monitoring A/B deployment...")
        while True:
            # 1. Get live batch of data
            X_live, meta = self.data_stream.next_batch()

            # 2. Split data
            split_idx = int(0.7 * len(X_live))
            XA, XB = X_live[:split_idx], X_live[split_idx:]

            # 3. Get predictions
            pred_A = self.model_a.predict(XA)
            pred_B = self.model_b.predict(XB)

            # 4. Get ground truth labels
            y_true_A, y_true_B = self.outcomes_fetcher.fetch_labels(split=(split_idx, len(X_live)))

            # 5. Compute metrics
            recall_A = recall_score(y_true_A, pred_A, zero_division=0)
            recall_B = recall_score(y_true_B, pred_B, zero_division=0)

            print(f"Recall Champion: {recall_A:.4f}, Challenger: {recall_B:.4f}")

            if recall_B > recall_A + 0.01:
                print("Challenger outperforms Champion. Promote challenger.")
                self.model_a, self.model_b = self.model_b, self.model_a  # Promote
