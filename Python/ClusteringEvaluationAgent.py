#Agent to evaluate the performance of the clustering agent 
from sklearn.metrics import f1_score

class ClusteringEvaluationAgent:
    def __init__(self, model, X_val, y_val):
        self.model = model
        self.X_val = X_val
        self.y_val = y_val

    def execute(self):
        preds = self.model.predict(self.X_val)
        score = f1_score(self.y_val, preds, average='macro')
        print(f"[Evaluation] F1 Score: {score}")
        return score
