"""
Root ADK agent that:
  1) Extracts an intent using Gemini with a strict JSON schema (structured outputs).
  2) Routes to an intent-specific workflow module (folder-per-intent).
  3) Returns the workflow's JSON result verbatim.
"""

import os
import json
import google.auth
from google.adk.agents import Agent  # ADK Agent base class

# GenAI client (Vertex AI mode)
from google import genai
from google.genai.types import HttpOptions

# Import intent workflows from folder-per-intent modules
from app.agents.intents.nearby.intent import handle_nearby
from app.agents.intents.unsupported_intent.intent import handle_unsupported_intent

# ----------------------------
# Environment: Vertex AI mode
# ----------------------------
# In starter-pack patterns, ADC (Application Default Credentials) provides the project.
_, _project_id = google.auth.default()
# Ensure the GenAI SDK uses Vertex AI (your GCP project/quota), not the public endpoint.
os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "True")

# --------------------------------------------
# Structured output schema for intent extract
# --------------------------------------------
# Constrains the model to return ONLY this JSON shape:
# {
#   "intent": "nearby" | "unsupported_intent",
#   "confidence": <0.0..1.0>
# }
_INTENT_RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "intent": {
            "type": "STRING",
            "enum": ["nearby", "unsupported_intent"],
        },
        "confidence": {
            "type": "NUMBER",
            "minimum": 0.0,
            "maximum": 1.0,
        },
    },
    "required": ["intent", "confidence"],
}

def _extract_intent(user_query: str) -> dict:
    """
    Call Gemini (Vertex AI mode) with a strict response schema to classify the query.

    Input:
      user_query (str)

    Returns:
      dict shaped as _INTENT_RESPONSE_SCHEMA, e.g.:
        {"intent":"nearby","confidence":0.82}
        or
        {"intent":"unsupported_intent","confidence":0.60}
    """
    client = genai.Client(http_options=HttpOptions(api_version="v1"))

    # Minimal instruction; schema enforces the output shape.
    prompt = (
        "Classify the user's request into exactly one intent.\n"
        "Return JSON only, following the provided response schema.\n"
        "Intents:\n"
        "- nearby: when the user asks a generic metric-related question.\n"
        "- unsupported_intent: for anything else.\n\n"
        f"User query: {user_query}"
    )

    resp = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config={
            "response_mime_type": "application/json",   # force JSON
            "response_schema": _INTENT_RESPONSE_SCHEMA, # enforce schema
        },
    )
    # Official examples read resp.text; parse it to a Python dict.
    return json.loads(resp.text)

def route_intent(user_query: str) -> str:
    """
    Tool exposed to the ADK Agent. Deterministic orchestrator:
      1) Extract intent (structured outputs).
      2) Dispatch to the specific workflow module.
      3) Return that workflow's JSON verbatim.

    Output examples:
      {"intent":"nearby","query":"What were sales last quarter?"}
      {"intent":"unsupported_intent","query":"Tell me a bedtime story"}
    """
    extraction = _extract_intent(user_query)
    intent = extraction.get("intent", "unsupported_intent")

    if intent == "nearby":
        return handle_nearby(user_query)
    else:
        return handle_unsupported_intent(user_query)

# Root Agent:
#  - Instruct the model to always call route_intent(user_query=...).
#  - Return ONLY the tool's JSON payload (no extra words).
root_agent = Agent(
    name="root_agent",
    model="gemini-2.5-flash",
    instruction=(
        "You are an intent router. For every user message:\n"
        "1) Call the tool `route_intent(user_query=...)`.\n"
        "2) Return ONLY the tool's JSON result verbatim.\n"
        "Do NOT add any extra words or explanations."
    ),
    tools=[route_intent],
)
