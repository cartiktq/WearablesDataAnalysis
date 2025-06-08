import torch
import requests
import json
from torch import nn

class PredictorModel(nn.Module):
    def __init__(self, input_dim=1536, hidden_dim=512):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)

class AdverseEventPredictor:
    def __init__(self):
        self.model = PredictorModel()
        self.model.load_state_dict(torch.load("model_weights.pth", map_location=torch.device('cpu')))
        self.model.eval()

    def fetch_embedding(self, subject_id):
        resp = requests.get(f"http://vector-db:8000/query?subject_id={subject_id}")
        embedding = torch.tensor(resp.json()["embedding"])
        return embedding

    def predict(self, subject_id):
        emb = self.fetch_embedding(subject_id)
        with torch.no_grad():
            prob = self.model(emb.unsqueeze(0))
        prediction = int(prob.item() > 0.5)
        return {"subject_id": subject_id, "adverse_event": prediction, "probability": prob.item()}

if __name__ == "__main__":
    predictor = AdverseEventPredictor()
    subject_ids = ["001", "002", "003"]  # Example subject IDs
    for sid in subject_ids:
        result = predictor.predict(sid)
        print(json.dumps(result, indent=2))
