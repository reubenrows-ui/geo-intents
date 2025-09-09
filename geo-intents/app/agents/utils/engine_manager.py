"""
Engine Manager for Vertex AI Discovery Engine / Agent Builder

- create_engine(project_id, location, engine_id, display_name, data_store_id,
                solution_type=..., search_tier=..., search_add_ons=...)
- get_engine(...)
- list_engines(...)
- update_engine(..., display_name=..., disable_analytics=...)
- delete_engine(...)

Notes:
- Use `solution_type` (SEARCH / RECOMMENDATION / CHAT).
- For SEARCH, you may set Search tier/add-ons via `search_engine_config`.
- Always attach at least one DataStore ID.
"""

from __future__ import annotations
from typing import Iterable, Optional, Sequence

from google.api_core.client_options import ClientOptions
from google.protobuf.field_mask_pb2 import FieldMask
from google.cloud import discoveryengine_v1 as discoveryengine


# ---------------------------
# Helpers
# ---------------------------

def _client_options_for(location: str | None) -> Optional[ClientOptions]:
    return (
        ClientOptions(api_endpoint=f"{location}-discoveryengine.googleapis.com")
        if location and location != "global"
        else None
    )

def collection_path(project_id: str, location: str, collection_id: str = "default_collection") -> str:
    return discoveryengine.EngineServiceClient.collection_path(
        project=project_id, location=location, collection=collection_id
    )

def engine_name(project_id: str, location: str, engine_id: str, collection_id: str = "default_collection") -> str:
    return discoveryengine.EngineServiceClient.engine_path(
        project=project_id, location=location, collection=collection_id, engine=engine_id
    )


# ---------------------------
# Engine Lifecycle
# ---------------------------

def create_engine(
    project_id: str,
    location: str,
    engine_id: str,
    display_name: str,
    data_store_id: str,
    *,
    collection_id: str = "default_collection",
    solution_type: discoveryengine.SolutionType = discoveryengine.SolutionType.SOLUTION_TYPE_SEARCH,
    industry_vertical: discoveryengine.IndustryVertical = discoveryengine.IndustryVertical.GENERIC,
    # Search-only knobs (ignored for other solution types)
    search_tier: discoveryengine.SearchTier = discoveryengine.SearchTier.SEARCH_TIER_STANDARD,
    search_add_ons: Sequence[discoveryengine.SearchAddOn] = (),
) -> discoveryengine.Engine:
    """
    Create an Engine linked to a DataStore.

    For SEARCH engines, you can pass `search_tier` and `search_add_ons`
    (e.g., LLM add-on). See Google sample for reference.  # :contentReference[oaicite:1]{index=1}
    """
    client = discoveryengine.EngineServiceClient(client_options=_client_options_for(location))
    parent = collection_path(project_id, location, collection_id)

    # Base engine fields
    engine = discoveryengine.Engine(
        display_name=display_name,
        data_store_ids=[data_store_id],
        solution_type=solution_type,                     # <-- correct field
        industry_vertical=industry_vertical,
    )

    # Attach search config only if solution_type==SEARCH
    if solution_type == discoveryengine.SolutionType.SOLUTION_TYPE_SEARCH:
        engine.search_engine_config = discoveryengine.Engine.SearchEngineConfig(
            search_tier=search_tier,
            search_add_ons=list(search_add_ons) if search_add_ons else None,
        )

    op = client.create_engine(
        request=discoveryengine.CreateEngineRequest(
            parent=parent,
            engine=engine,
            engine_id=engine_id,
        )
    )
    return op.result()  # LRO wait; mirrors official sample.  # :contentReference[oaicite:2]{index=2}


def get_engine(project_id: str, location: str, engine_id: str, collection_id: str = "default_collection") -> discoveryengine.Engine:
    client = discoveryengine.EngineServiceClient(client_options=_client_options_for(location))
    name = engine_name(project_id, location, engine_id, collection_id)
    return client.get_engine(request=discoveryengine.GetEngineRequest(name=name))


def list_engines(project_id: str, location: str, collection_id: str = "default_collection") -> Iterable[discoveryengine.Engine]:
    client = discoveryengine.EngineServiceClient(client_options=_client_options_for(location))
    parent = collection_path(project_id, location, collection_id)
    return client.list_engines(request=discoveryengine.ListEnginesRequest(parent=parent))


def update_engine(
    project_id: str,
    location: str,
    engine_id: str,
    *,
    display_name: Optional[str] = None,
    disable_analytics: Optional[bool] = None,
    collection_id: str = "default_collection",
) -> discoveryengine.Engine:
    """
    Update mutable Engine fields. (display_name, disable_analytics are commonly mutable.)
    """
    client = discoveryengine.EngineServiceClient(client_options=_client_options_for(location))
    name = engine_name(project_id, location, engine_id, collection_id)

    engine = discoveryengine.Engine(name=name)
    paths = []
    if display_name is not None:
        engine.display_name = display_name
        paths.append("display_name")
    if disable_analytics is not None:
        engine.disable_analytics = disable_analytics
        paths.append("disable_analytics")

    return client.update_engine(
        request=discoveryengine.UpdateEngineRequest(
            engine=engine,
            update_mask=FieldMask(paths=paths),
        )
    )


def delete_engine(project_id: str, location: str, engine_id: str, collection_id: str = "default_collection") -> None:
    client = discoveryengine.EngineServiceClient(client_options=_client_options_for(location))
    name = engine_name(project_id, location, engine_id, collection_id)
    op = client.delete_engine(request=discoveryengine.DeleteEngineRequest(name=name))
    op.result()
