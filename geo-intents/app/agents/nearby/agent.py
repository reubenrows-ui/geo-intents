import json
from google.adk.agents import LlmAgent, SequentialAgent
from google.adk.tools import VertexAiSearchTool

from .schema import SlotCandidates, PoiCoords

entity_extractor_agent = LlmAgent(
    name="EntityExtractor", 
    model="gemini-2.5-flash",
    description="Calls Gemini to extract nearby intent *candidate* slots..",
    instruction=f"""You are an agent that extracts potential slippets for nearby intent queries.
    From the user's request, extract entity CANDIDATES for:
    - anchor_place: place name or POI (e.g., 'Union Station, Toronto')
    - radius_value: number if user specifies a distance (e.g., 2)
    - radius_unit: unit of distance (km, miles, mi, m)
    - entity: competitor or brand name (e.g., 'Starbucks')
    Respond ONLY with a JSON object matching this exact schema:
    {json.dumps(SlotCandidates.model_json_schema(), indent=2)}
    Use your knowledge to determine the candidate terms or snippets.
    """,
    output_key="slot_candidates",
    )

DATASTORE_PATH = "projects/kaggle-hackathon-471317/locations/global/collections/default_collection/dataStores/poi-locations-data"

anchor_place_validator_agent = LlmAgent(
    name="AnchorPlaceValidator",
    input_schema=SlotCandidates,
    instruction=f"""You are an agent that validates if the extracted anchor_place matches any known places in the datastore.
    Use the provided tool to search the datastore for places matching the extracted anchor_place.
        If you are confident there is a match, return the latitude and longitude of the best matching place.
        If no match is found, return document_name as an empty response for latitude and longitude.
    Respond ONLY with a JSON object matching this exact schema:
    {json.dumps(PoiCoords.model_json_schema(), indent=2)}

    """,
    tools = [VertexAiSearchTool(data_store_id=DATASTORE_PATH,max_results=1)],
    output_key="anchor_place_coords",

) # Custom non-LLM agent

nearby_intent_agent= SequentialAgent(
    name="nearby_intent_agent",
    sub_agents=[entity_extractor_agent, anchor_place_validator_agent])