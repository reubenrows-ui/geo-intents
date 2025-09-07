"""
Public entrypoint for the `nearby` intent.

What this file exports
----------------------
- `handle_nearby(user_query: str) -> str`

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

from .workflow import run_nearby_workflow

__all__ = ["handle_nearby"]


def handle_nearby(user_query: str) -> str:
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
          "intent": "nearby",
          "query":  "What were sales last quarter in Toronto?",
          "slots":  {
            "anchor_place": "Toronto",
            "radius_value": 2,
            "radius_unit": "km",
            "entity": "Starbucks"
          }
        }
    """
    result_dict = run_nearby_workflow(user_query)
    return result_dict
