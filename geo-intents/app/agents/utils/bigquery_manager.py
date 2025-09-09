"""
BigQuery Manager (Python)

Features
--------
Datasets:
- create_dataset, get_dataset, list_datasets, update_dataset, delete_dataset

Tables:
- create_table, get_table, list_tables, update_table, delete_table
- load_table_from_gcs (CSV/JSON/AVRO/PARQUET/ORC)
- load_table_from_dataframe (pandas)
- insert_rows_json (streaming)
- copy_table
- extract_table_to_gcs
- run_query (to table or to rows)

Auth
----
Use ADC (e.g. `gcloud auth application-default login`) and ensure caller has BigQuery permissions.

Requirements
------------
pip install google-cloud-bigquery pandas  # pandas optional (for load_from_dataframe)

References
----------
- Client & core methods (python): https://cloud.google.com/python/docs/reference/bigquery/latest/google.cloud.bigquery.client.Client  # client+API  # [ref]
- Create dataset: https://cloud.google.com/bigquery/docs/samples/bigquery-create-dataset  # dataset create  # [ref]
- Manage datasets (delete/copy/labels): https://cloud.google.com/bigquery/docs/managing-datasets  # datasets mgmt  # [ref]
- Create/use tables (+ partition/cluster): https://cloud.google.com/bigquery/docs/tables  # tables  # [ref]
- Load CSV/JSON from GCS: https://cloud.google.com/bigquery/docs/loading-data-cloud-storage-csv  # load CSV/JSON  # [ref]
- Load Parquet from GCS: https://cloud.google.com/bigquery/docs/loading-data-cloud-storage-parquet  # load Parquet  # [ref]
- Streaming inserts (insert_rows_json): https://cloud.google.com/bigquery/docs/samples/bigquery-table-insert-rows  # streaming  # [ref]
- Copy table job: https://cloud.google.com/bigquery/docs/samples/bigquery-copy-table  # copy  # [ref]
- Extract (export) to GCS: https://cloud.google.com/bigquery/docs/exporting-data  # extract  # [ref]
"""

from __future__ import annotations

import time
from typing import Iterable, List, Optional, Sequence, Union

from google.cloud import bigquery
from google.api_core import retry as retries


# ---------------------------
# Helpers
# ---------------------------

_DEFAULT_RETRY = retries.Retry(deadline=300.0)  # generous default
WriteDisposition = bigquery.WriteDisposition
CreateDisposition = bigquery.CreateDisposition


def _client(project_id: Optional[str] = None, location: Optional[str] = None) -> bigquery.Client:
    """
    Returns a BigQuery client; if `location` is set, it becomes the default job location.
    """
    return bigquery.Client(project=project_id, location=location)


def _wait(job: bigquery.job.Job) -> bigquery.job.Job:
    """
    Waits for a BigQuery Job to complete and raises on error.
    """
    return job.result()  # raises on failure; preferred pattern in samples


# ---------------------------
# DATASETS
# ---------------------------

def create_dataset(
    project_id: str,
    dataset_id: str,
    *,
    location: str = "US",
    description: Optional[str] = None,
    labels: Optional[dict] = None,
    exists_ok: bool = False,
) -> bigquery.Dataset:
    """
    Create a dataset. Note: dataset location is immutable.  [ref: create_dataset]
    """
    client = _client(project_id, location)
    ds_ref = bigquery.Dataset(f"{project_id}.{dataset_id}")
    ds_ref.location = location
    if description:
        ds_ref.description = description
    if labels:
        ds_ref.labels = labels

    return client.create_dataset(ds_ref, exists_ok=exists_ok)  # [ref: create dataset sample]


def get_dataset(project_id: str, dataset_id: str) -> bigquery.Dataset:
    client = _client(project_id)
    return client.get_dataset(f"{project_id}.{dataset_id}")


def list_datasets(project_id: str) -> Iterable[bigquery.DatasetListItem]:
    client = _client(project_id)
    return client.list_datasets(project=project_id)


def update_dataset(
    project_id: str,
    dataset_id: str,
    *,
    description: Optional[str] = None,
    labels: Optional[dict] = None,
    default_table_expiration_ms: Optional[int] = None,
    default_partition_expiration_ms: Optional[int] = None,
) -> bigquery.Dataset:
    """
    Update mutable dataset fields (labels/description/default expirations).  [ref: manage datasets]
    """
    client = _client(project_id)
    ds = client.get_dataset(f"{project_id}.{dataset_id}")

    fields = []
    if description is not None:
        ds.description = description
        fields.append("description")
    if labels is not None:
        ds.labels = labels
        fields.append("labels")
    if default_table_expiration_ms is not None:
        ds.default_table_expiration_ms = default_table_expiration_ms
        fields.append("default_table_expiration_ms")
    if default_partition_expiration_ms is not None:
        ds.default_partition_expiration_ms = default_partition_expiration_ms
        fields.append("default_partition_expiration_ms")

    return client.update_dataset(ds, fields=fields)


def delete_dataset(project_id: str, dataset_id: str, *, delete_contents: bool = False) -> None:
    """
    Delete a dataset; set `delete_contents=True` to remove all tables/views.  [ref: manage datasets]
    """
    client = _client(project_id)
    client.delete_dataset(f"{project_id}.{dataset_id}", delete_contents=delete_contents, not_found_ok=True)


# ---------------------------
# TABLES
# ---------------------------

def create_table(
    project_id: str,
    dataset_id: str,
    table_id: str,
    *,
    schema: Optional[List[bigquery.SchemaField]] = None,
    description: Optional[str] = None,
    partition_field: Optional[str] = None,   # for time partitioning
    partition_type: Optional[str] = None,    # "DAY" | "HOUR" | "MONTH" | "YEAR"
    partition_expiration_ms: Optional[int] = None,
    clustering_fields: Optional[Sequence[str]] = None,
    external_config: Optional[bigquery.ExternalConfig] = None,
) -> bigquery.Table:
    """
    Create a table (standard or external). Supports clustering/partitioning.  [ref: tables]
    """
    client = _client(project_id)
    table_ref = bigquery.Table(f"{project_id}.{dataset_id}.{table_id}", schema=schema or [])
    if description:
        table_ref.description = description

    if partition_field:
        table_ref.time_partitioning = bigquery.TimePartitioning(
            type_=getattr(bigquery.TimePartitioningType, partition_type or "DAY"),
            field=partition_field,
            expiration_ms=partition_expiration_ms,
        )

    if clustering_fields:
        table_ref.clustering_fields = list(clustering_fields)

    if external_config:
        table_ref.external_data_configuration = external_config

    return client.create_table(table_ref, exists_ok=False)


def get_table(project_id: str, dataset_id: str, table_id: str) -> bigquery.Table:
    client = _client(project_id)
    return client.get_table(f"{project_id}.{dataset_id}.{table_id}")


def list_tables(project_id: str, dataset_id: str) -> Iterable[bigquery.TableListItem]:
    client = _client(project_id)
    return client.list_tables(f"{project_id}.{dataset_id}")


def update_table(
    project_id: str,
    dataset_id: str,
    table_id: str,
    *,
    description: Optional[str] = None,
    schema: Optional[List[bigquery.SchemaField]] = None,
    partition_field: Optional[str] = None,
    partition_type: Optional[str] = None,
    partition_expiration_ms: Optional[int] = None,
    clustering_fields: Optional[Sequence[str]] = None,
) -> bigquery.Table:
    """
    Update table metadata/schema, partitioning, clustering.  [ref: tables]
    """
    client = _client(project_id)
    table = client.get_table(f"{project_id}.{dataset_id}.{table_id}")

    fields = []
    if description is not None:
        table.description = description
        fields.append("description")

    if schema is not None:
        table.schema = schema
        fields.append("schema")

    if partition_field is not None:
        table.time_partitioning = bigquery.TimePartitioning(
            type_=getattr(bigquery.TimePartitioningType, partition_type or "DAY"),
            field=partition_field,
            expiration_ms=partition_expiration_ms,
        )
        fields.append("time_partitioning")

    if clustering_fields is not None:
        table.clustering_fields = list(clustering_fields) if clustering_fields else None
        fields.append("clustering_fields")

    return client.update_table(table, fields=fields)


def delete_table(project_id: str, dataset_id: str, table_id: str) -> None:
    client = _client(project_id)
    client.delete_table(f"{project_id}.{dataset_id}.{table_id}", not_found_ok=True)


# ---------------------------
# LOADS
# ---------------------------

def load_table_from_gcs(
    project_id: str,
    dataset_id: str,
    table_id: str,
    *,
    uris: Sequence[str],
    source_format: str = "CSV",   # CSV | NEWLINE_DELIMITED_JSON | AVRO | PARQUET | ORC
    schema: Optional[List[bigquery.SchemaField]] = None,
    autodetect: bool = False,
    write_disposition: WriteDisposition = WriteDisposition.WRITE_APPEND,
    field_delimiter: Optional[str] = None,
    skip_leading_rows: Optional[int] = None,
    quote_character: Optional[str] = None,
    allow_jagged_rows: Optional[bool] = None,
    allow_quoted_newlines: Optional[bool] = None,
    location: Optional[str] = None,
) -> bigquery.LoadJob:
    """
    Load files from GCS into a table.  CSV/JSON/AVRO/PARQUET/ORC supported.  [refs: load CSV/JSON, load Parquet]
    """
    client = _client(project_id, location)
    table_ref = f"{project_id}.{dataset_id}.{table_id}"

    job_config = bigquery.LoadJobConfig(
        source_format=getattr(bigquery.SourceFormat, source_format),
        write_disposition=write_disposition,
        autodetect=autodetect,
        schema=schema,
        field_delimiter=field_delimiter,
        skip_leading_rows=skip_leading_rows,
        quote_character=quote_character,
        allow_jagged_rows=allow_jagged_rows,
        allow_quoted_newlines=allow_quoted_newlines,
    )

    job = client.load_table_from_uri(uris, table_ref, job_config=job_config)
    return _wait(job)


def load_table_from_dataframe(
    project_id: str,
    dataset_id: str,
    table_id: str,
    *,
    df,
    schema: Optional[List[bigquery.SchemaField]] = None,
    write_disposition: WriteDisposition = WriteDisposition.WRITE_APPEND,
    location: Optional[str] = None,
) -> bigquery.LoadJob:
    """
    Load a pandas DataFrame to a table using the optimized Arrow path.  [ref: client docs]
    """
    client = _client(project_id, location)
    table_ref = f"{project_id}.{dataset_id}.{table_id}"
    job_config = bigquery.LoadJobConfig(write_disposition=write_disposition, schema=schema)
    job = client.load_table_from_dataframe(df, table_ref, job_config=job_config)
    return _wait(job)


# ---------------------------
# STREAMING INSERTS
# ---------------------------

def insert_rows_json(
    project_id: str, dataset_id: str, table_id: str, rows: List[dict]
) -> List[dict]:
    """
    Streaming inserts via insert_rows_json. Returns any errors.  [ref: streaming inserts]
    """
    client = _client(project_id)
    errors = client.insert_rows_json(f"{project_id}.{dataset_id}.{table_id}", rows)
    return errors or []


# ---------------------------
# COPY / EXTRACT
# ---------------------------

def copy_table(
    project_id: str,
    src_dataset: str,
    src_table: str,
    dst_dataset: str,
    dst_table: str,
    *,
    write_disposition: WriteDisposition = WriteDisposition.WRITE_TRUNCATE,
    location: Optional[str] = None,
) -> bigquery.CopyJob:
    """
    Copy one table to another (optionally overwrite).  [refs: copy table, write_disposition usage]
    """
    client = _client(project_id, location)
    src = f"{project_id}.{src_dataset}.{src_table}"
    dst = f"{project_id}.{dst_dataset}.{dst_table}"
    job_config = bigquery.CopyJobConfig(write_disposition=write_disposition)
    job = client.copy_table(sources=src, destination=dst, job_config=job_config)
    return _wait(job)


def extract_table_to_gcs(
    project_id: str,
    dataset_id: str,
    table_id: str,
    *,
    destination_uris: Sequence[str],
    destination_format: str = "PARQUET",   # CSV | JSON | AVRO | PARQUET
    location: Optional[str] = None,
) -> bigquery.ExtractJob:
    """
    Export a table to GCS (supports multi-file URIs like gs://bucket/prefix-*).  [ref: exporting-data]
    """
    client = _client(project_id, location)
    table_ref = f"{project_id}.{dataset_id}.{table_id}"
    job_config = bigquery.ExtractJobConfig(destination_format=getattr(bigquery.DestinationFormat, destination_format))
    job = client.extract_table(table_ref, destination_uris, job_config=job_config)
    return _wait(job)


# ---------------------------
# QUERIES
# ---------------------------

def run_query(
    project_id: str,
    sql: str,
    *,
    location: Optional[str] = None,
    destination_table: Optional[str] = None,  # "project.dataset.table"
    write_disposition: WriteDisposition = WriteDisposition.WRITE_TRUNCATE,
    use_legacy_sql: bool = False,
    maximum_bytes_billed: Optional[int] = None,
) -> bigquery.QueryJob:
    client = _client(project_id, location)

    # Build config without None fields
    job_config = bigquery.QueryJobConfig(
        use_legacy_sql=use_legacy_sql,
    )
    if maximum_bytes_billed is not None:
        job_config.maximum_bytes_billed = maximum_bytes_billed

    if destination_table:
        job_config.destination = bigquery.Table(destination_table)
        job_config.write_disposition = write_disposition

    job = client.query(sql, job_config=job_config)
    return _wait(job)