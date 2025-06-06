#Agent which uses an LLM to interpret the generated clusters and output an explanatory summary
class ClusterInterpreterAgent:
    def __init__(self, llm_client):
        self.client = llm_client

    def execute(self, df):
        summaries = []
        for cluster_id, sub_df in df.groupby('cluster'):
            text_summary = self.client.summarize_notes(sub_df.to_string())
            summaries.append(f"Cluster {cluster_id}: {text_summary}")
        return summaries
