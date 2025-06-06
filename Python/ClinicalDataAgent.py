#Agent which queries clinical data from a clinical data warehouse implemented in Redshift
class ClinicalDataAgent:
    def __init__(self, agentic_client, llm_agent):
        self.client = agentic_client
        self.notes_agent = llm_agent

    def execute(self):
        clinical_df = self.client.query("warehouse", {"db": "redshift", "table": "clinical"})
        notes_df = clinical_df[['subject_id', 'notes']].copy()
        notes_df['summarized'] = notes_df['notes'].apply(self.notes_agent.execute)
        return clinical_df.drop(columns=['notes']).join(notes_df[['subject_id', 'summarized']], on='subject_id')
