"""
`unsupported_intent` module (fallback intent).

Current scaffolding behavior:
- Returns a small JSON object with:
    - intent label ("unsupported_intent")
    - original user query
- You can expand this later with guidance to the user, logging, or analytics.
"""

def handle_unsupported_intent(user_query: str) -> str:
    """
    Fallback workflow when no supported intent applies.

    Example output:
      {"intent":"unsupported_intent","query":"Tell me a bedtime story"}
    """
    return {
        "intent": "unsupported_intent",
        "query": user_query,
    }
