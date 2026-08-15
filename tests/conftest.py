"""
Shared pytest fixtures available to all tests in the suite.

conftest.py is automatically discovered and loaded by pytest — no import
needed in individual test files. Put fixtures here that are shared across
multiple test modules (e.g. the test HTTP client).
"""

import pytest
from fastapi.testclient import TestClient

from app.main import create_app


@pytest.fixture(scope="session")
def app():
    """
    Create a fresh FastAPI application for the entire test session.

    Using scope="session" means the app is built once and reused across all
    tests, which is efficient. If a test needs to mutate app state, it should
    use scope="function" instead.
    """
    return create_app()


@pytest.fixture(scope="session")
def client(app):
    """
    Provide an httpx-backed synchronous test client for the FastAPI app.

    TestClient wraps the ASGI app and lets us make real HTTP requests
    without starting an actual server process.
    """
    with TestClient(app) as c:
        yield c
