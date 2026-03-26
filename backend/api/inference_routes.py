"""
inference_routes.py — /api/inference/* and /ws/inference/* endpoints.
Full implementation: Session 6 (real-time prediction WebSocket).
"""

from fastapi import APIRouter
from fastapi.responses import JSONResponse

router = APIRouter()


@router.get("/health")
async def health() -> JSONResponse:
    """Placeholder — confirms this router is registered and reachable."""
    return JSONResponse({"success": True, "data": {"status": "inference_routes stub"}, "error": None})
