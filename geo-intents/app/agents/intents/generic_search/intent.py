"""
Public entrypoint for the `generic_search` intent.

What this file exports
----------------------
- `handle_generic_search(user_query: str) -> str`

Behavior
--------
- Calls the workflow to produce a Python dict result.
- Serializes the dict to a JSON string and returns it *verbatim*.
- This function is what your router imports and calls.

Why this layer?
---------------
- Keeps the router interface stable (returns a JSON string).
- Lets the workflow stay pure-Python (dict), improving testability.
"""

import json
from .workflow import run_generic_search_workflow

__all__ = ["handle_generic_search"]


def handle_generic_search(user_query: str) -> str:
    """
    Entrypoint used by the router for this intent.

    Parameters
    ----------
    user_query : str
        Raw user message.

    Returns
    -------
    str
        JSON string that the agent will return verbatim.
        Example:
        {
          "intent": "generic_search",
          "query":  "What were sales last quarter in Toronto?",
          "slots":  { "metrics": [...], "dimensions": [...], "filters": [...] }
        }
    """
    result_dict = run_generic_search_workflow(user_query)
    return json.dumps(result_dict)
