"""Main application entry point for CortexaAI - Advanced Multi-Agent Prompt Engineering System.

All API endpoints live in ``src/routes/``.  This module is responsible for:
  - FastAPI app creation & lifespan (startup/shutdown)
  - CORS middleware
  - Mounting route modules
  - Serving the React SPA (if built)
  - The ``main()`` entry-point for ``uvicorn``
"""

import asyncio
import sys
import os
import uvicorn
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# Ensure project root is importable (idempotent)
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from config.config import settings, setup_langsmith, get_logger

# Import shared state so backward-compat re-exports work for tests
from src.deps import (
    coordinator,
    classifier_instance,
    evaluator_instance,
    active_workflows,
    metrics,
    settings as _deps_settings,
    template_engine,
    plugin_manager,
    process_prompt_with_langgraph,
    PromptRequest,
    PromptResponse,
    SystemStats,
    WorkflowCancellationError,
)
from agents.coordinator import WorkflowCoordinator  # noqa: F401 re-export

# Expose at module level so existing @patch('src.main.coordinator') etc. still work
import psutil

logger = get_logger(__name__)

# Route imports
from src.routes import (
    prompts_router,
    workflows_router,
    system_router,
    features_router,
)


# Lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager: seed templates, load plugins, graceful shutdown."""
    # Startup
    template_engine._seed_defaults()
    plugins_dir = Path(__file__).parent.parent / "plugins"
    if plugins_dir.exists():
        plugin_manager.load_from_directory(str(plugins_dir))

    yield  # Server is running

    # Graceful shutdown: cancel active workflows
    running = [
        (wid, wf) for wid, wf in active_workflows.items()
        if wf.get("status") == "running"
    ]
    if running:
        logger.info(f"Shutting down: cancelling {len(running)} active workflow(s)...")
        for wid, wf in running:
            evt = wf.get("cancellation_event")
            if evt:
                evt.set()
            wf["status"] = "cancelled"
        await asyncio.sleep(0.5)
    active_workflows.clear()
    logger.info("Shutdown complete.")


# App creation
app = FastAPI(
    title="CortexaAI",
    description=(
        "Advanced Multi-Agent Prompt Engineering System with optimization, "
        "A/B testing, streaming, batch processing, marketplace, and multi-LLM support"
    ),
    version="3.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in settings.cors_origins.split(",") if o.strip()],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "X-API-Key", "X-Requested-With"],
)


# Mount routers
app.include_router(prompts_router)
app.include_router(workflows_router)
app.include_router(system_router)
app.include_router(features_router)


# Root & SPA serving
@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main web interface (prefer built assets over dev index)."""
    build_index = Path(__file__).parent.parent / "frontend-react" / "dist" / "index.html"
    if build_index.exists():
        return build_index.read_text(encoding="utf-8")
    dev_index = Path(__file__).parent.parent / "frontend-react" / "index.html"
    if dev_index.exists():
        return dev_index.read_text(encoding="utf-8")
    raise HTTPException(status_code=404, detail="Frontend not built. Run: cd frontend-react && npm run build")


_frontend_build = Path(__file__).parent.parent / "frontend-react" / "dist"
if _frontend_build.exists():
    app.mount("/assets", StaticFiles(directory=str(_frontend_build / "assets")), name="assets")

    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        """Serve the React SPA for any non-API route."""
        if full_path.startswith("api/") or full_path in (
            "docs", "redoc", "openapi.json", "health", "metrics",
        ):
            raise HTTPException(status_code=404)
        index_file = _frontend_build / "index.html"
        if index_file.exists():
            return HTMLResponse(index_file.read_text(encoding="utf-8"))
        raise HTTPException(status_code=404, detail="Frontend not built. Run: cd frontend-react && npm run build")


# Entry-point
def main():
    """Main entry point for running the application."""
    setup_langsmith()

    from src.deps import llm_provider

    logger.info("Starting CortexaAI - Advanced Multi-Agent Prompt Engineering System")
    logger.info(f"Server: http://{settings.host}:{settings.port}")
    logger.info(f"API Docs: http://{settings.host}:{settings.port}/docs")
    logger.info(f"Providers: {', '.join(llm_provider.get_available_providers()) or 'None (check .env)'}")

    uvicorn.run(
        "src.main:app",
        host=settings.host,
        port=settings.port,
        reload=os.getenv("CORTEXAAI_ENV", "development").lower() == "development",
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
