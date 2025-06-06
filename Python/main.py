# main.py Main agent to orchestrate all the steps in the workflow
if __name__ == "__main__":
    agentic_client = AgenticClient(api_key="your-agentic-api-key")
    llm_client = LLMClient(api_key="your-openai-api-key")

    agents = {
        "wearables": WearablesStreamAgent(agentic_client),
        "weather": WeatherDataAgent(agentic_client),
        "demographics": DemographicsDataAgent(agentic_client),
        "notes": ClinicalNoteSummarizerAgent(llm_client),
        "clinical": ClinicalDataAgent(agentic_client, ClinicalNoteSummarizerAgent(llm_client)),
        "clustering": ClusteringAgent(),
        "interpreter": ClusterInterpreterAgent(llm_client),
        "tsne": ClusterTSNEVisualizer(),
        "output": ClusterOutputAgent()
    }

    # Execute pipeline
    wearables_df = agents["wearables"].execute()
    weather_df = agents["weather"].execute()
    demo_df = agents["demographics"].execute()
    clinical_df = agents["clinical"].execute()

    full_df = pd.merge(pd.merge(wearables_df, demo_df, on='subject_id'), clinical_df, on='subject_id')
    clustered_df, model = agents["clustering"].execute(full_df)
    
    agents["tsne"].execute(clustered_df)
    print(agents["output"].execute(clustered_df))
    print("\n".join(agents["interpreter"].execute(clustered_df)))
