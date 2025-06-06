#Agent,. which extracts demographics data from a DynamoDB NoSQL database
class DemographicsDataAgent:
    def __init__(self, agentic_client):
        self.client = agentic_client

    def execute(self):
        config = {"table": "demographics", "database": "DynamoDB"}
        return self.client.query("static_db", config)
