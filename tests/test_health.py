"""
Tests del endpoint GET /health.
"""

import os

os.environ.setdefault("ANTHROPIC_API_KEY", "sk-ant-test-key")


class TestHealthEndpoint:
    """Tests del health check — no requiere servicios de ML."""

    def test_health_returns_200(self, client):
        """El health check debe retornar HTTP 200."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_status_is_ok(self, client):
        """El campo 'status' debe ser 'ok'."""
        assert client.get("/health").json()["status"] == "ok"

    def test_health_response_structure(self, client):
        """La respuesta debe tener los campos requeridos."""
        data = client.get("/health").json()
        for field in ["status", "service", "version", "timestamp"]:
            assert field in data, f"Campo '{field}' ausente"

    def test_health_version_semver(self, client):
        """La versión debe tener formato X.Y.Z."""
        version = client.get("/health").json()["version"]
        parts = version.split(".")
        assert len(parts) == 3
        assert all(p.isdigit() for p in parts)

    def test_health_content_type_json(self, client):
        """La respuesta debe ser JSON."""
        assert "application/json" in client.get("/health").headers["content-type"]
