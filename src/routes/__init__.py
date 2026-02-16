"""Route sub-package for CortexaAI API endpoints."""

from src.routes.prompts import router as prompts_router
from src.routes.workflows import router as workflows_router
from src.routes.system import router as system_router
from src.routes.features import router as features_router

__all__ = [
    "prompts_router",
    "workflows_router",
    "system_router",
    "features_router",
]
