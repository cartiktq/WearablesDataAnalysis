import pandas as pd
import boto3
import psycopg2
import requests
from transformers import BertTokenizer, BertModel
import torch
from sentence_transformers import util

class BERTEncoder:
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.model = BertModel.from_pretrained("bert-base-uncased")
        self.vector_db_url = "http://vector-db:8000/upsert"

    def fetch_wearables(self, path="wearables.csv"):
        return pd.read_csv(path)

    def fetch_clinical(self):
        conn = psycopg2.connect(host="redshift-host", dbname="db", user="user", password="pass", port=5439)
        query = "SELECT * FROM clinical_history"
        df = pd.read_sql(query, conn)
        conn.close()
        return df

    def fetch_demographics(self):
        dynamodb = boto3.resource('dynamodb', region_name='us-east-1')
        table = dynamodb.Table('DemographicsTable')
        response = table.scan()
        return pd.DataFrame(response['Items'])

    def fetch_weather_embeddings(self, subject_id):
        resp = requests.get(f"http://weather-service:8080/embed?subject_id={subject_id}")
        return torch.tensor(resp.json()["embedding"])

    def encode_subject(self, wearable_row, clinical_row, demo_row, weather_emb):
        text_input = f"{wearable_row.to_json()} {clinical_row.to_json()} {demo_row.to_json()}"
        inputs = self.tokenizer(text_input, return_tensors="pt", truncation=True, padding=True)
        outputs = self.model(**inputs)
        pooled = outputs.pooler_output.squeeze(0)
        final_embedding = torch.cat((pooled, weather_emb), dim=0)
        return final_embedding

    def upsert_embedding(self, subject_id, embedding):
        embedding_list = embedding.detach().numpy().tolist()
        payload = {
            "subject_id": subject_id,
            "embedding": embedding_list
        }
        requests.post(self.vector_db_url, json=payload)

    def run(self):
        wearables = self.fetch_wearables()
        clinical = self.fetch_clinical()
        demographics = self.fetch_demographics()

        for subject_id in wearables["subject_id"].unique():
            w_row = wearables[wearables["subject_id"] == subject_id].iloc[0]
            c_row = clinical[clinical["subject_id"] == subject_id].iloc[0]
            d_row = demographics[demographics["subject_id"] == subject_id].iloc[0]
            weather_emb = self.fetch_weather_embeddings(subject_id)
            emb = self.encode_subject(w_row, c_row, d_row, weather_emb)
            self.upsert_embedding(subject_id, emb)

if __name__ == "__main__":
    encoder = BERTEncoder()
    encoder.run()
