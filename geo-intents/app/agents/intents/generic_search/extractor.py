"""
Extractor for slot candidates used by the `generic_search` workflow.

Responsibilities
---------------
- Define a strict response schema for structured outputs.
- Call Gemini (Vertex AI mode) with `response_schema` and `response_mime_type='application/json'`.
- Return a Python `dict` with candidate slots for metrics, dimensions, and filters.

Output shape (dict)
-------------------
{
  "metrics":    [{"name": <str>, "confidence": <0..1>}...],
  "dimensions": [{"name": <str>, "confidence": <0..1>}...],
  "filters":    [{"field": <str>, "operator": <enum>, "value": <str>, "confidence": <0..1>}...]
}

Why this split?
---------------
- Keeps LLM I/O (prompt + schema + API call) isolated from business/workflow logic.
- Makes it easy to unit-test the workflow by monkey-patching this function.
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
SLOT_CANDIDATES_SCHEMA: Dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "metrics": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "name": {"type": "STRING"},
                    "confidence": {"type": "NUMBER", "minimum": 0.0, "maximum": 1.0},
                },
                "required": ["name", "confidence"],
            },
        },
        "dimensions": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "name": {"type": "STRING"},
                    "confidence": {"type": "NUMBER", "minimum": 0.0, "maximum": 1.0},
                },
                "required": ["name", "confidence"],
            },
        },
        "filters": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "field": {"type": "STRING"},
                    "operator": {
                        "type": "STRING",
                        "enum": ["=", "!=", ">", ">=", "<", "<=", "IN", "NOT IN", "LIKE"],
                    },
                    "value": {"type": "STRING"},
                    "confidence": {"type": "NUMBER", "minimum": 0.0, "maximum": 1.0},
                },
                "required": ["field", "operator", "value", "confidence"],
            },
        },
    },
    "required": ["metrics", "dimensions", "filters"],
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
        "- metrics: business measures (e.g., sales, revenue, traffic)\n"
        "- dimensions: entities/attributes to group or slice by (e.g., brand, city)\n"
        "- filters: field, operator, value triplets (e.g., city = Toronto)\n\n"
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
