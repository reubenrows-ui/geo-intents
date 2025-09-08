"""
Datastore Manager for Vertex AI Search / Discovery Engine

Features
--------
- Create, list, get, update, delete DataStores
- Import documents into a DataStore branch from:
  - Google Cloud Storage (GCS) URIs (PDF/HTML/TXT/DOCX/PPTX, custom JSONL, discoveryengine.Document)
  - BigQuery tables (structured docs), FULL or INCREMENTAL reconciliation
- Safe polling of long-running operations (LROs)
- Regional endpoint selection (e.g., "global" or "us-central1")

Requirements
------------
pip install google-cloud-discoveryengine

Auth
----
Use Application Default Credentials (ADC) (e.g., `gcloud auth application-default login`)
and ensure the caller's principal has the necessary Discovery Engine IAM permissions.

Resource Naming
---------------
collection: typically "default_collection"
branch:     typically "default_branch"

References
----------
- DataStore create/list/get/update/delete (DataStoreServiceClient):
  https://cloud.google.com/python/docs/reference/discoveryengine/latest/google.cloud.discoveryengine_v1.services.data_store_service.DataStoreServiceClient  # :contentReference[oaicite:3]{index=3}
- Create Data Store sample (regional endpoint pattern):
  https://cloud.google.com/generative-ai-app-builder/docs/samples/genappbuilder-create-data-store  # :contentReference[oaicite:4]{index=4}
- Import Documents (GCS / BigQuery) samples and shapes:
  https://cloud.google.com/generative-ai-app-builder/docs/samples/genappbuilder-import-documents-gcs  # :contentReference[oaicite:5]{index=5}
  https://cloud.google.com/generative-ai-app-builder/docs/refresh-data  # :contentReference[oaicite:6]{index=6}
"""

from __future__ import annotations

import time
from typing import Iterable, Optional, Sequence

from google.api_core.client_options import ClientOptions
from google.api_core.operation import Operation
from google.protobuf.field_mask_pb2 import FieldMask

from google.cloud import discoveryengine_v1 as discoveryengine


# ---------------------------
# Helpers: endpoints & names
# ---------------------------

def _client_options_for(location: str | None) -> Optional[ClientOptions]:
    """
    Returns ClientOptions configured for a regional endpoint when location != "global".
    """
    if location and location != "global":
        return ClientOptions(api_endpoint=f"{location}-discoveryengine.googleapis.com")
    return None


def collection_path(project_id: str, location: str, collection_id: str = "default_collection") -> str:
    return discoveryengine.DataStoreServiceClient.collection_path(
        project=project_id, location=location, collection=collection_id
    )


def datastore_name(project_id: str, location: str, data_store_id: str) -> str:
    return discoveryengine.DataStoreServiceClient.data_store_path(
        project=project_id, location=location, data_store=data_store_id
    )


def branch_path(project_id: str, location: str, data_store_id: str, branch_id: str = "default_branch") -> str:
    return discoveryengine.DocumentServiceClient.branch_path(
        project=project_id,
        location=location,
        data_store=data_store_id,
        branch=branch_id
    )


# ---------------------------
# Core: DataStore lifecycle
# ---------------------------

def create_datastore(
    project_id: str,
    location: str,
    data_store_id: str,
    display_name: str,
    industry_vertical: discoveryengine.IndustryVertical = discoveryengine.IndustryVertical.GENERIC,
    solution_types: Sequence[discoveryengine.SolutionType] = (discoveryengine.SolutionType.SOLUTION_TYPE_SEARCH,),
    content_config: discoveryengine.DataStore.ContentConfig = discoveryengine.DataStore.ContentConfig.NO_CONTENT,
    collection_id: str = "default_collection",
) -> discoveryengine.DataStore:
    """
    Creates a DataStore under a collection.

    Notes:
    - You still need to create an Engine to serve search/recs over this DataStore.
    - `solution_types` controls SEARCH vs RECOMMENDATION flavors.

    Returns the created DataStore.
    """
    dss = discoveryengine.DataStoreServiceClient(client_options=_client_options_for(location))

    parent = collection_path(project_id, location, collection_id)

    ds = discoveryengine.DataStore(
        display_name=display_name,
        industry_vertical=industry_vertical,
        solution_types=list(solution_types),
        content_config=content_config,
    )

    op: Operation = dss.create_data_store(
        request=discoveryengine.CreateDataStoreRequest(
            parent=parent,
            data_store=ds,
            data_store_id=data_store_id,
            # Optionally: skip_default_schema_creation=True
        )
    )
    # LRO: creation is asynchronous
    return op.result()  # Waits until created. Pattern per official sample.  :contentReference[oaicite:7]{index=7}


def list_datastores(project_id: str, location: str, collection_id: str = "default_collection") -> Iterable[discoveryengine.DataStore]:
    dss = discoveryengine.DataStoreServiceClient(client_options=_client_options_for(location))
    parent = collection_path(project_id, location, collection_id)
    return dss.list_data_stores(request=discoveryengine.ListDataStoresRequest(parent=parent))


def get_datastore(project_id: str, location: str, data_store_id: str) -> discoveryengine.DataStore:
    dss = discoveryengine.DataStoreServiceClient(client_options=_client_options_for(location))
    name = datastore_name(project_id, location, data_store_id)
    return dss.get_data_store(request=discoveryengine.GetDataStoreRequest(name=name))


def update_datastore(
    project_id: str,
    location: str,
    data_store_id: str,
    *,
    display_name: Optional[str] = None,
    industry_vertical: Optional[discoveryengine.IndustryVertical] = None,
    solution_types: Optional[Sequence[discoveryengine.SolutionType]] = None,
    content_config: Optional[discoveryengine.DataStore.ContentConfig] = None,
    collection_id: str = "default_collection",
    update_mask_paths: Optional[Sequence[str]] = None,
) -> discoveryengine.DataStore:
    """
    Updates mutable fields on a DataStore using UpdateDataStore + FieldMask.

    If `update_mask_paths` is not provided, it will be inferred from which kwargs are not None.
    """
    dss = discoveryengine.DataStoreServiceClient(client_options=_client_options_for(location))
    name = datastore_name(project_id, location, data_store_id, collection_id)

    current = discoveryengine.DataStore(name=name)
    # Only set fields provided
    if display_name is not None:
        current.display_name = display_name
    if industry_vertical is not None:
        current.industry_vertical = industry_vertical
    if solution_types is not None:
        current.solution_types = list(solution_types)
    if content_config is not None:
        current.content_config = content_config

    if not update_mask_paths:
        paths = []
        if display_name is not None:
            paths.append("display_name")
        if industry_vertical is not None:
            paths.append("industry_vertical")
        if solution_types is not None:
            paths.append("solution_types")
        if content_config is not None:
            paths.append("content_config")
        update_mask_paths = paths

    resp = dss.update_data_store(
        request=discoveryengine.UpdateDataStoreRequest(
            data_store=current,
            update_mask=FieldMask(paths=list(update_mask_paths)),
        )
    )
    return resp  # v1 UpdateDataStore returns DataStore (non-LRO).  :contentReference[oaicite:8]{index=8}


def delete_datastore(project_id: str, location: str, data_store_id: str, collection_id: str = "default_collection") -> None:
    dss = discoveryengine.DataStoreServiceClient(client_options=_client_options_for(location))
    name = datastore_name(project_id, location, data_store_id, collection_id)
    op: Operation = dss.delete_data_store(request=discoveryengine.DeleteDataStoreRequest(name=name))
    op.result()  # Wait for deletion to complete.  :contentReference[oaicite:9]{index=9}


# ---------------------------
# Populate: import documents
# ---------------------------

def import_documents_from_gcs(
    project_id: str,
    location: str,
    data_store_id: str,
    gcs_uris: Sequence[str],
    *,
    data_schema: str = "content",
    id_field: Optional[str] = None,
    reconciliation_mode: discoveryengine.ImportDocumentsRequest.ReconciliationMode = discoveryengine.ImportDocumentsRequest.ReconciliationMode.INCREMENTAL,
    branch_id: str = "default_branch",
    collection_id: str = "default_collection",
    poll: bool = True,
    poll_interval_sec: float = 10.0,
) -> Operation:
    """
    Imports documents from one or more GCS URIs into the specified DataStore branch.

    data_schema options (per Google samples/docs):
      - "content": Unstructured content (PDF/HTML/TXT/DOCX/PPTX).
      - "custom":  Unstructured with custom JSONL metadata.
      - "document": Structured documents matching discoveryengine.Document.

    See: Import GCS sample + Refresh data docs.  :contentReference[oaicite:10]{index=10}
    """
    doc_client = discoveryengine.DocumentServiceClient(client_options=_client_options_for(location))
    parent = branch_path(project_id, location, data_store_id, branch_id)

    req = discoveryengine.ImportDocumentsRequest(
        parent=parent,
        gcs_source=discoveryengine.GcsSource(input_uris=list(gcs_uris), data_schema=data_schema),
        id_field=id_field or "",
        reconciliation_mode=reconciliation_mode,
    )

    op: Operation = doc_client.import_documents(request=req)
    if poll:
        return _poll_operation(op, poll_interval_sec)
    return op


def import_documents_from_bigquery(
    project_id: str,
    location: str,
    data_store_id: str,
    *,
    bq_project_id: str,
    bq_dataset: str,
    bq_table: str,
    data_schema: str = "custom",
    reconciliation_mode: discoveryengine.ImportDocumentsRequest.ReconciliationMode = discoveryengine.ImportDocumentsRequest.ReconciliationMode.INCREMENTAL,
    branch_id: str = "default_branch",
    collection_id: str = "default_collection",
    poll: bool = True,
    poll_interval_sec: float = 10.0,
) -> Operation:
    """
    Imports structured documents from a BigQuery table.

    Typical for custom JSON/structured doc pipelines. See refresh-data docs.  :contentReference[oaicite:11]{index=11}
    """
    doc_client = discoveryengine.DocumentServiceClient(client_options=_client_options_for(location))
    parent = branch_path(project_id, location, data_store_id, branch_id)

    req = discoveryengine.ImportDocumentsRequest(
        parent=parent,
        bigquery_source=discoveryengine.BigQuerySource(
            project_id=bq_project_id,
            dataset_id=bq_dataset,
            table_id=bq_table,
            data_schema=data_schema,
        ),
        reconciliation_mode=reconciliation_mode,
    )

    op: Operation = doc_client.import_documents(request=req)
    if poll:
        return _poll_operation(op, poll_interval_sec)
    return op


# ---------------------------
# LRO polling utility
# ---------------------------

def _poll_operation(op: Operation, poll_interval_sec: float = 10.0) -> Operation:
    """
    Polls an LRO until completion with a simple sleep loop, returning the completed Operation.
    """
    while not op.done():
        time.sleep(poll_interval_sec)
    # Raise if failed:
    _ = op.result()
    return op
