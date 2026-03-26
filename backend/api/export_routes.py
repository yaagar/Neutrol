"""
export_routes.py — /api/export/* endpoints.
Full implementation: Session 7 (pipeline JSON + model weights + inference script export).
"""

from fastapi import APIRouter
from fastapi.responses import JSONResponse

router = APIRouter()


@router.get("/health")
async def health() -> JSONResponse:
    """Placeholder — confirms this router is registered and reachable."""
    return JSONResponse({"success": True, "data": {"status": "export_routes stub"}, "error": None})
