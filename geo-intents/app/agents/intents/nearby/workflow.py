"""
Workflow orchestration for the `nearby` intent.

Responsibilities
---------------
- Invoke the extractor to get candidate slots.
- Construct a stable Python dict containing:
    - intent label
    - original query
    - extracted slots
- (Future) Add custom validation/normalization/mapping here before downstream calls.

Why this split?
---------------
- Keeps workflow logic independent of the LLM call details.
- Makes this easy to expand (e.g., validate metrics against a catalog, map synonyms).
"""

from typing import Dict, Any
from .extractor import extract_slot_candidates


def run_nearby_workflow(user_query: str) -> Dict[str, Any]:
    """
    Execute the core hierarchy search workflow.

    Returns
    -------
    dict
        {
          "intent": "nearby",
          "query":  "<original user query>",
          "slots":  { ... candidates dict from extractor ... }
        }
    """
    slots = extract_slot_candidates(user_query)

    result: Dict[str, Any] = {
        "intent": "nearby",
        "query": user_query,
        "slots": slots,
    }
    return result
