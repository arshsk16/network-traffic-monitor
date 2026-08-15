"""
Tests for GET /health endpoint.

These are the simplest possible integration tests: spin up the app with a
TestClient, hit the endpoint, assert the contract holds.
"""

from fastapi.testclient import TestClient


class TestHealthEndpoint:
    """Group all /health tests in a class for readability."""

    def test_health_returns_200(self, client: TestClient) -> None:
        """The endpoint must return HTTP 200 OK."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_response_schema(self, client: TestClient) -> None:
        """Response body must contain the three required fields."""
        response = client.get("/health")
        body = response.json()

        assert "status" in body
        assert "app_name" in body
        assert "version" in body

    def test_health_status_is_ok(self, client: TestClient) -> None:
        """status field must be the string 'ok' while the service is running."""
        response = client.get("/health")
        assert response.json()["status"] == "ok"

    def test_health_content_type_is_json(self, client: TestClient) -> None:
        """Response must be application/json."""
        response = client.get("/health")
        assert "application/json" in response.headers["content-type"]
