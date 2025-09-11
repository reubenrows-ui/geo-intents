# Conceptual Code: Coordinator using LLM Transfer
from google.adk.agents import LlmAgent

from app.agents.nearby.agent import nearby_intent_agent
from app.agents.unknown.agent import unknown_intent_agent



root_agent = LlmAgent(
    name="intent_router",
    model="gemini-2.5-flash",
    instruction="""You are a router agent that directs user queries to the appropriate sub-agent based on intent.
    - If you get any questions related to finding nearby places, restaurants, or points of interest, transfer to the nearby_intent agent. See examples below:
        - “What cafés are within 1 mile of Times Square?”
        - “How many competitors are near my downtown Vancouver store?”
        - “List restaurants close to Union Station, Toronto.”
    - If the user is asking about anything else, transfer to the unknown_intent agent.
    """,
    description="internt_router.",
    # allow_transfer=True is often implicit with sub_agents in AutoFlow
    sub_agents=[
        nearby_intent_agent, 
        unknown_intent_agent]
)
