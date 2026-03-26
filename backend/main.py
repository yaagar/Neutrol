"""
main.py — FastAPI application entry point.

Responsibilities:
  1. Create the FastAPI app instance.
  2. Configure CORS so the frontend (http://localhost:5173) can call the API.
  3. Register all 6 routers with their URL prefixes.
  4. Expose a GET /health endpoint for quick sanity checks.

Start the server with:
  uvicorn main:app --reload --port 8000

The --reload flag watches for file changes and restarts automatically.
Remove it in production.
"""

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Import all routers — implemented and stub alike
from api.data_routes     import router as data_router
from api.pipeline_routes import router as pipeline_router
from api.analysis_routes import router as analysis_router
from api.label_routes    import router as label_router
from api.train_routes    import router as train_router
from api.inference_routes import router as inference_router
from api.export_routes   import router as export_router

# Configure Python's logging module.
# In production you'd point this at a file; here it goes to stdout so you can
# see request logs in the terminal while developing.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Create the FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Neutrol API",
    description=(
        "Backend API for the Neutrol EEG platform. "
        "Handles EEG file loading, preprocessing, signal analysis, labeling, training, and inference."
    ),
    version="0.1.0",
    # FastAPI generates interactive docs at /docs (Swagger UI) and /redoc automatically.
    # This is one of the main reasons we chose FastAPI over Flask.
)

# ---------------------------------------------------------------------------
# CORS — allow the Vite dev server to call this API
# ---------------------------------------------------------------------------

# CORS (Cross-Origin Resource Sharing) is a browser security feature.
# When the frontend at localhost:5173 makes a fetch() call to localhost:8000,
# the browser blocks it by default because the ports differ (different "origins").
# This middleware tells the browser "yes, localhost:5173 is allowed to call us".
#
# In production you would replace the origin with your actual domain name.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Vite's default dev server address
    allow_credentials=True,
    allow_methods=["*"],    # allow GET, POST, DELETE, PUT, OPTIONS, etc.
    allow_headers=["*"],    # allow all headers (Content-Type, Authorization, etc.)
)

# ---------------------------------------------------------------------------
# Register routers
# ---------------------------------------------------------------------------

# Each router handles a section of the API. The prefix is prepended to all routes
# defined in that router file. Example: data_router's "/upload" becomes "/api/data/upload".

app.include_router(data_router,     prefix="/api/data",      tags=["Data"])
app.include_router(pipeline_router, prefix="/api/pipeline",  tags=["Pipeline"])
app.include_router(analysis_router, prefix="/api/analysis",  tags=["Analysis"])
app.include_router(label_router,    prefix="/api/labels",    tags=["Labels"])
app.include_router(train_router,    prefix="/api/train",     tags=["Training"])
app.include_router(inference_router,prefix="/api/inference", tags=["Inference"])
app.include_router(export_router,   prefix="/api/export",    tags=["Export"])

logger.info("All routers registered.")

# ---------------------------------------------------------------------------
# Health check endpoint
# ---------------------------------------------------------------------------

@app.get("/health", tags=["Meta"])
async def health() -> dict:
    """
    Quick sanity check — confirms the server is running.
    Returns the standard APIResponse shape (success, data, error).
    """
    return {"success": True, "data": {"status": "running"}, "error": None}
