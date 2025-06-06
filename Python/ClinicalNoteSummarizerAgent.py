#Agent implements an LLM to summarize clinical notes and extract codes from ICD-9/10, SNOMED CT, RxNORM, CPT.
class ClinicalNoteSummarizerAgent:
    def __init__(self, llm_client):
        self.client = llm_client

    def execute(self, note_text):
        return self.client.summarize_notes(note_text)
