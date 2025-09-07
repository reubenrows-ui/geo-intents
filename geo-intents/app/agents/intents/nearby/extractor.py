"""
Extractor for the 'nearby' intent.

This extractor is responsible for pure token extraction using Gemini structured outputs.
It returns only slot candidates, not the intent itself.

Slots:
- anchor_place: place name or POI (e.g., "Union Station, Toronto")
- radius_value: number if user specifies a distance (e.g., 2)
- radius_unit: unit of distance (km, miles, mi, m)
- entity: competitor or brand name (e.g., "Starbucks")
"""

import json
from typing import Dict, Any

# GenAI client (Vertex AI mode), same pattern used by your agent/router.
from google import genai
from google.genai.types import HttpOptions


# ----------------------------
# Structured Output Schema
# ----------------------------
# Constrains Gemini to return ONLY the expected JSON shape.
# Define the structured schema used for extraction
SLOT_CANDIDATES_SCHEMA = {
    "type": "object",
    "properties": {
        "anchor_place": {"type": "string"},
        "radius_value": {"type": "number"},
        "radius_unit": {"type": "string", "enum": ["km", "miles", "mi", "m"]},
        "entity": {"type": "string"},
    },
    "additionalProperties": False,
}

def extract_slot_candidates(user_query: str) -> dict:
    """
    Calls Gemini in Vertex AI mode to extract *candidate* slots.

    Parameters
    ----------
    user_query : str
        The raw user utterance.

    Returns
    -------
    dict
        A Python dict shaped by SLOT_CANDIDATES_SCHEMA.

    Notes
    -----
    - Uses `response_schema` to strictly enforce the JSON format.
    - Uses `response_mime_type='application/json'` so we can parse `resp.text`.
    """
    client = genai.Client(http_options=HttpOptions(api_version="v1"))

    # Minimal instruction; the schema dictates the final shape.
    prompt = (
        "From the user's request, extract slot CANDIDATES for:\n"
        "- anchor_place: place name or POI (e.g., 'Union Station, Toronto')\n"
        "- radius_value: number if user specifies a distance (e.g., 2)\n"
        "- radius_unit: unit of distance (km, miles, mi, m)\n"
        "- entity: competitor or brand name (e.g., 'Starbucks')\n\n"
        "Guidelines:\n"
        "- Return only what is implied or common-sense in the text.\n"
        "- Provide multiple candidates when ambiguous.\n"
        "- Include a confidence in [0,1].\n\n"
        "Return ONLY JSON per the provided response schema.\n\n"
        f"User query: {user_query}"
    )

    resp = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config={
            "response_mime_type": "application/json",
            "response_schema": SLOT_CANDIDATES_SCHEMA,
        },
    )

    return json.loads(resp.text)
