"""Shared dependencies, state, Pydantic models, and helper utilities used across route modules."""

import uuid
import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any

from pydantic import BaseModel, Field

from config.config import (
    settings,
    metrics,
    get_logger,
    cache_manager,
    generate_prompt_cache_key,
    generate_evaluation_cache_key,
    perf_config,
)
from config.llm_providers import llm_provider, PROVIDER_CONFIGS
from core.optimization import optimization_engine
from agents.coordinator import WorkflowCoordinator
from agents.classifier import DomainClassifier
from agents.evaluator import PromptEvaluator
from src.workflow import process_prompt_with_langgraph
import psutil

# ── New feature service imports ──────────────────────────────────────────
from core.database import db
from core.templates import template_engine
from core.error_recovery import error_analytics
from core.language import language_processor
from core.complexity import complexity_analyzer
from core.auth import auth_manager
from core.marketplace import marketplace
from core.finetuning import finetuning_manager
from core.prompt_builder import prompt_builder
from core.plugins import plugin_manager
from core.regression import regression_runner
from core.similarity import similarity_engine
from core.streaming import stream_workflow
from core.batch import batch_processor
from core.webhooks import webhook_manager

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Global instances (lazy – avoid calling get_llm() at import time)
# ---------------------------------------------------------------------------
classifier_instance = DomainClassifier()

_evaluator_instance = None
_coordinator_instance = None


def _get_evaluator():
    global _evaluator_instance
    if _evaluator_instance is None:
        _evaluator_instance = PromptEvaluator()
    return _evaluator_instance


def _get_coordinator():
    global _coordinator_instance
    if _coordinator_instance is None:
        _coordinator_instance = WorkflowCoordinator(classifier_instance, _get_evaluator())
    return _coordinator_instance


def __getattr__(name):
    """Lazy module-level access for ``evaluator_instance`` and ``coordinator``."""
    if name == "evaluator_instance":
        return _get_evaluator()
    if name == "coordinator":
        return _get_coordinator()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

# In-memory storage for active workflows with cancellation tokens
active_workflows: Dict[str, Dict[str, Any]] = {}

# Maximum prompt length (characters) to prevent abuse
MAX_PROMPT_LENGTH = 50_000

# Workflow cleanup: max completed/failed workflows to keep in memory
_MAX_FINISHED_WORKFLOWS = 200


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cleanup_finished_workflows() -> None:
    """Evict oldest completed/failed/cancelled workflows when the map grows too large."""
    finished = [
        (wid, wf) for wid, wf in active_workflows.items()
        if wf.get("status") in ("completed", "failed", "cancelled")
    ]
    if len(finished) > _MAX_FINISHED_WORKFLOWS:
        finished.sort(key=lambda x: x[1].get("start_time", datetime.min))
        to_remove = len(finished) - _MAX_FINISHED_WORKFLOWS
        for wid, _ in finished[:to_remove]:
            active_workflows.pop(wid, None)


class WorkflowCancellationError(Exception):
    """Exception raised when a workflow is cancelled."""
    pass


async def process_prompt_with_langgraph_cancellable(
    prompt: str, prompt_type: str, cancellation_event: asyncio.Event
):
    """Wrapper for LangGraph processing with cancellation support."""
    return await process_prompt_with_langgraph(
        prompt=prompt,
        prompt_type=prompt_type,
        cancellation_event=cancellation_event,
    )


async def check_cancellation(cancellation_event: asyncio.Event, workflow_id: str):
    """Check if workflow should be cancelled and raise exception if so."""
    if cancellation_event.is_set():
        logger.info(f"Workflow {workflow_id} cancellation detected")
        raise WorkflowCancellationError(f"Workflow {workflow_id} was cancelled")


def clear_workflow_caches(prompt: str, prompt_type: str = None):
    """Clear all cache entries related to a workflow prompt."""
    # Clear classification cache
    cache_key_classification = generate_prompt_cache_key(prompt, prefix="classification")
    cache_manager.delete(cache_key_classification)

    # Clear prompt type classification cache if applicable
    if prompt_type:
        cache_key_prompt_type = generate_prompt_cache_key(prompt, prefix="prompt_type_classification")
        cache_manager.delete(cache_key_prompt_type)

    # Clear caches for all known domains (dynamically obtained)
    try:
        common_domains = list(classifier_instance.get_available_domains().keys())
    except Exception:
        common_domains = ["software_engineering", "data_science", "report_writing",
                          "education", "business_strategy", "creative_writing", "general"]
    for domain in common_domains:
        for pt in ["raw", "structured", "auto"]:
            cache_key_improvement = generate_prompt_cache_key(prompt, domain, pt)
            cache_manager.delete(cache_key_improvement)
            cache_key_improvement_context = generate_prompt_cache_key(prompt, domain, f"{pt}_context")
            cache_manager.delete(cache_key_improvement_context)

    logger.info(f"Cleared all cache entries for prompt: {prompt[:50]}...")


# ---------------------------------------------------------------------------
# Pydantic request / response models
# ---------------------------------------------------------------------------

class PromptRequest(BaseModel):
    """Request model for prompt processing."""
    prompt: str
    prompt_type: str = "auto"
    return_comparison: bool = True
    use_langgraph: bool = False
    chat_history: Optional[List[Dict[str, str]]] = None
    user_id: Optional[str] = None
    workflow_id: Optional[str] = None
    advanced_mode: bool = False
    synchronous: bool = False
    callback_url: Optional[str] = None
    language: Optional[str] = None


class PromptResponse(BaseModel):
    """Response model for prompt processing."""
    workflow_id: str
    status: str
    message: Optional[str] = None
    timestamp: str
    processing_time_seconds: Optional[float]
    input: Dict[str, Any]
    output: Dict[str, Any]
    analysis: Optional[Dict[str, Any]]
    comparison: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any]


class SystemStats(BaseModel):
    """Model for system statistics."""
    total_workflows: int
    completed_workflows: int
    error_workflows: int
    success_rate: float
    average_quality_score: float
    average_processing_time: float
    domain_distribution: Dict[str, int]
