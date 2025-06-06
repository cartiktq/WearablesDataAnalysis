#For extracting streaming data from wearables
class WearablesStreamAgent:
    def __init__(self, agentic_client):
        self.client = agentic_client

    def execute(self):
        config = {"stream": "kinesis_or_kafka", "topic": "wearables"}
        return self.client.query("stream", config)
