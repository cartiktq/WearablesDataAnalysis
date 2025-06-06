# utils.py
import openai
import pandas as pd

class AgenticClient:
    def __init__(self, api_key):
        self.api_key = api_key

    def query(self, source_type, source_config, query_params=None):
        # Simulated query to Agentic API
        print(f"[AgenticClient] Querying {source_type} with config {source_config}")
        return pd.DataFrame()  # return empty for now


class LLMClient:
    def __init__(self, model="gpt-4", api_key=None):
        self.model = model
        openai.api_key = api_key

    def summarize_notes(self, note_text):
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=[{"role": "user", "content": f"Summarize and extract codes: {note_text}"}]
        )
        return response['choices'][0]['message']['content']
