"""
discovery_search.py

Wrapper for calling Vertex AI Discovery Engine search via REST API.

This uses Application Default Credentials (ADC), the same as `gcloud auth print-access-token`.
"""

import json
import requests
import google.auth
import google.auth.transport.requests


class DiscoverySearchClient:
    def __init__(self, project_id: str, engine_id: str, location: str = "global"):
        self.project_id = project_id
        self.engine_id = engine_id
        self.location = location
        self._access_token = None

    def _get_access_token(self) -> str:
        """Fetch and cache an OAuth access token using ADC."""
        if self._access_token is None:
            creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
            auth_req = google.auth.transport.requests.Request()
            creds.refresh(auth_req)
            self._access_token = creds.token
        return self._access_token

    def search(
        self,
        query: str,
        page_size: int = 10,
        time_zone: str = "America/Toronto",
        **kwargs
    ) -> dict:
        """
        Run a search query against Discovery Engine.

        Args:
            query (str): The search query string.
            page_size (int): Number of results to return.
            time_zone (str): User time zone (e.g., "America/Toronto").
            **kwargs: Any additional Discovery Engine search params
                      (e.g., filter, boostSpec, facetSpecs, queryExpansionSpec, etc.)

        Returns:
            dict: Parsed JSON response from Discovery Engine.
        """
        url = (
            f"https://discoveryengine.googleapis.com/v1alpha/"
            f"projects/{self.project_id}/locations/{self.location}/"
            f"collections/default_collection/engines/{self.engine_id}/"
            f"servingConfigs/default_search:search"
        )

        # Base payload
        payload = {
            "query": query,
            "pageSize": page_size,
            "queryExpansionSpec": {"condition": "AUTO"},
            "spellCorrectionSpec": {"mode": "AUTO"},
            "languageCode": "en-US",
            "userInfo": {"timeZone": time_zone},
        }

        # Merge in any additional params
        payload.update(kwargs)

        headers = {
            "Authorization": f"Bearer {self._get_access_token()}",
            "Content-Type": "application/json",
        }

        response = requests.post(url, headers=headers, data=json.dumps(payload))
        response.raise_for_status()
        return response.json()
