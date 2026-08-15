"""
Health check router.

Provides a single GET /health endpoint that returns the application's
current status. This is the standard first endpoint for any API service —
it lets load balancers, orchestrators, and developers verify the service
is up without touching any business logic.
"""

from fastapi import APIRouter
from pydantic import BaseModel

from app.config import settings

router = APIRouter(tags=["health"])


class HealthResponse(BaseModel):
    """Schema for the health check response body."""

    status: str
    app_name: str
    version: str


@router.get("/health", response_model=HealthResponse, summary="Health check")
async def health() -> HealthResponse:
    """
    Return the current health status of the service.

    - **status**: always ``"ok"`` when the service is running
    - **app_name**: application name from config
    - **version**: current application version
    """
    return HealthResponse(
        status="ok",
        app_name=settings.app_name,
        version=settings.app_version,
    )
