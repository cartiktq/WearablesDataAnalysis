from agentic import Agent, Task
from sklearn.metrics import recall_score, f1_score
import numpy as np

class TrainingSupervisorAgent(Agent):
    def __init__(self, model, data_loader, outcomes_fetcher, trainer, threshold=0.95):
        self.model = model
        self.data_loader = data_loader
        self.outcomes_fetcher = outcomes_fetcher
        self.trainer = trainer
        self.threshold = threshold
        self.current_epoch = 0

    def run(self):
        while True:
            # 1. Load training data
            X_train, _ = self.data_loader.load_unlabeled_data()
            y_true = self.outcomes_fetcher.fetch_labels()

            # 2. Predict
            y_pred = self.model.predict(X_train)

            # 3. Evaluate
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)

            print(f"[Epoch {self.current_epoch}] Recall: {recall:.4f}, F1: {f1:.4f}")

            if recall >= self.threshold:
                print("Training objective met. Exiting loop.")
                break

            # 4. Retrain
            self.model = self.trainer.train_next_epoch(X_train, y_true)
            self.current_epoch += 1
