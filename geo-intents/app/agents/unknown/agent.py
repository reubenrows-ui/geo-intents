from google.adk.agents import LlmAgent

unknown_intent_agent = LlmAgent(
    name="unknown_intent", 
    description="Handles all other requests that are not nearby intent related."
    )