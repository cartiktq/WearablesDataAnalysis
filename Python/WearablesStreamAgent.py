#For extracting streaming data from wearables
# --- File: /WearablesDataAnalysis/Python/WearablesStreamAgent.py ---
from kafka import KafkaConsumer
import pandas as pd
import json
import time

class WearablesStreamAgent:
    def __init__(self, topic='wearables', bootstrap_servers='localhost:9092', group_id='stream-group'):
        self.consumer = KafkaConsumer(
            topic,
            bootstrap_servers=bootstrap_servers,
            group_id=group_id,
            value_deserializer=lambda m: json.loads(m.decode('utf-8')),
            auto_offset_reset='latest'
        )

    def collect_data(self, duration_seconds=60):
        end_time = time.time() + duration_seconds
        records = []
        for message in self.consumer:
            if time.time() > end_time:
                break
            records.append(message.value)
        return pd.DataFrame(records)
