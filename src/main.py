"""Main application entry point for Multi-Agent Prompt Engineering System."""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
import uvicorn
import logging

import sys
import os
from pathlib import Path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import settings, setup_langsmith, metrics, get_logger
from agents.coordinator import WorkflowCoordinator
from agents.classifier import DomainClassifier
from agents.evaluator import PromptEvaluator
from src.workflow import process_prompt_with_langgraph
import psutil
import time

# Set up structured logging
logger = get_logger(__name__)

# Global instances for dependency injection
classifier_instance = DomainClassifier()
evaluator_instance = PromptEvaluator()
coordinator = WorkflowCoordinator(classifier_instance, evaluator_instance)

# Initialize FastAPI app
app = FastAPI(
    title="Multi-Agent Prompt Engineering System",
    description="A production-level system for improving and optimizing prompts using multiple AI agents",
    version="1.0.0"
)


# Pydantic models for API
class PromptRequest(BaseModel):
    """Request model for prompt processing."""
    prompt: str
    prompt_type: str = "auto"  # "auto", "raw", or "structured"
    return_comparison: bool = True
    use_langgraph: bool = False  # Whether to use LangGraph workflow


class PromptResponse(BaseModel):
    """Response model for prompt processing."""
    workflow_id: str
    status: str
    timestamp: str
    processing_time_seconds: Optional[float]
    input: Dict[str, Any]
    output: Dict[str, Any]
    analysis: Optional[Dict[str, Any]]
    comparison: Optional[Dict[str, Any]]
    metadata: Dict[str, Any]


class DomainInfo(BaseModel):
    """Model for domain information."""
    domain: str
    description: str
    keywords: List[str]
    has_expert_agent: bool
    agent_created: bool


class SystemStats(BaseModel):
    """Model for system statistics."""
    total_workflows: int
    completed_workflows: int
    error_workflows: int
    success_rate: float
    average_quality_score: float
    average_processing_time: float
    domain_distribution: Dict[str, int]


# API Routes
@app.post("/api/process-prompt", response_model=PromptResponse)
async def process_prompt(request: PromptRequest) -> PromptResponse:
    """Process a prompt through the multi-agent workflow."""
    try:
        logger.info(f"Processing prompt via API: {request.prompt[:100]}...")

        if request.use_langgraph:
            # Use LangGraph workflow
            result = await process_prompt_with_langgraph(
                prompt=request.prompt,
                prompt_type=request.prompt_type
            )
        else:
            # Use Coordinator workflow
            result = await coordinator.process_prompt(
                prompt=request.prompt,
                prompt_type=request.prompt_type,
                return_comparison=request.return_comparison
            )

        # Convert result to response format
        return PromptResponse(**result)

    except Exception as e:
        logger.error(f"Error processing prompt: {e}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


@app.get("/api/domains", response_model=List[DomainInfo])
async def get_available_domains():
    """Get information about all available domains."""
    try:
        domains = coordinator.get_available_domains()
        return [DomainInfo(**domain) for domain in domains]
    except Exception as e:
        logger.error(f"Error getting domains: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get domains: {str(e)}")


@app.get("/api/stats", response_model=SystemStats)
async def get_system_stats():
    """Get system statistics and workflow history."""
    try:
        stats = coordinator.get_workflow_stats()
        # Handle case where there's no workflow history yet
        if "error" in stats:
            return SystemStats(
                total_workflows=0,
                completed_workflows=0,
                error_workflows=0,
                success_rate=0.0,
                average_quality_score=0.0,
                average_processing_time=0.0,
                domain_distribution={}
            )
        return SystemStats(**stats)
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")


@app.get("/api/history", response_model=List[Dict[str, Any]])
async def get_workflow_history(limit: int = 10):
    """Get recent workflow history."""
    try:
        history = coordinator.get_workflow_history(limit=limit)
        return history
    except Exception as e:
        logger.error(f"Error getting history: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get history: {str(e)}")


# Web Interface Routes
@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main web interface."""
    template_path = Path(__file__).parent.parent / "frontend" / "index.html"
    return template_path.read_text(encoding="utf-8")


@app.get("/metrics")
async def get_metrics():
    """Prometheus-compatible metrics endpoint."""
    lines = []

    # Get current metrics
    current_metrics = metrics.get_metrics()

    # Add system metrics
    lines.append("# HELP system_info System information")
    lines.append("# TYPE system_info gauge")
    lines.append('system_info{version="1.0.0",langsmith_enabled="' + str(bool(settings.langsmith_api_key)).lower() + '"} 1')

    # Add LLM call metrics
    lines.append("# HELP llm_calls_total Total number of LLM calls")
    lines.append("# TYPE llm_calls_total counter")
    lines.append(f'llm_calls_total {current_metrics.get("llm_calls_total", 0)}')

    lines.append("# HELP llm_calls_success Successful LLM calls")
    lines.append("# TYPE llm_calls_success counter")
    lines.append(f'llm_calls_success {current_metrics.get("llm_calls_success", 0)}')

    lines.append("# HELP llm_calls_error Failed LLM calls")
    lines.append("# TYPE llm_calls_error counter")
    lines.append(f'llm_calls_error {current_metrics.get("llm_calls_error", 0)}')

    # Add workflow metrics
    lines.append("# HELP workflows_completed Completed workflows")
    lines.append("# TYPE workflows_completed counter")
    lines.append(f'workflows_completed {current_metrics.get("workflows_completed", 0)}')

    lines.append("# HELP workflows_failed Failed workflows")
    lines.append("# TYPE workflows_failed counter")
    lines.append(f'workflows_failed {current_metrics.get("workflows_failed", 0)}')

    # Add retry metrics
    lines.append("# HELP retry_attempts_total Total retry attempts")
    lines.append("# TYPE retry_attempts_total counter")
    lines.append(f'retry_attempts_total {current_metrics.get("retry_attempts", 0)}')

    # Add performance histograms
    durations = current_metrics.get("llm_call_duration_seconds", [])
    if durations:
        lines.append("# HELP llm_call_duration_seconds LLM call duration in seconds")
        lines.append("# TYPE llm_call_duration_seconds histogram")
        lines.append(f'llm_call_duration_seconds_count {len(durations)}')
        lines.append(f'llm_call_duration_seconds_sum {sum(durations)}')

        # Calculate buckets
        buckets = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, float('inf')]
        bucket_counts = [0] * len(buckets)
        for duration in durations:
            for i, bucket in enumerate(buckets):
                if duration <= bucket:
                    bucket_counts[i] += 1
                    break

        for i, bucket in enumerate(buckets):
            lines.append(f'llm_call_duration_seconds_bucket{{le="{bucket}"}} {bucket_counts[i]}')

    # Add domain distribution
    domains = current_metrics.get("domains_processed", {})
    for domain, count in domains.items():
        lines.append(f'# HELP domain_processed_total Total workflows processed for domain {domain}')
        lines.append("# TYPE domain_processed_total counter")
        lines.append(f'domain_processed_total{{domain="{domain}"}} {count}')

    # Add memory and system metrics
    try:
        process = psutil.Process()
        memory_info = process.memory_info()

        lines.append("# HELP process_memory_bytes Process memory usage in bytes")
        lines.append("# TYPE process_memory_bytes gauge")
        lines.append(f'process_memory_bytes {memory_info.rss}')

        lines.append("# HELP system_memory_percent System memory usage percentage")
        lines.append("# TYPE system_memory_percent gauge")
        lines.append(f'system_memory_percent {psutil.virtual_memory().percent}')

        lines.append("# HELP system_cpu_percent System CPU usage percentage")
        lines.append("# TYPE system_cpu_percent gauge")
        lines.append(f'system_cpu_percent {psutil.cpu_percent(interval=0.1)}')

    except ImportError:
        # psutil not available, skip system metrics
        pass

    return "\n".join(lines)


@app.get("/health")
async def health_check():
    """Enhanced health check endpoint with detailed system status."""
    health_status = {
        "status": "healthy",
        "timestamp": time.time(),
        "version": "1.0.0",
        "uptime_seconds": time.time() - getattr(health_check, 'start_time', time.time()),
        "components": {},
        "metrics": {}
    }

    # Store start time for uptime calculation
    if not hasattr(health_check, 'start_time'):
        health_check.start_time = time.time()

    # Check LLM provider connectivity
    health_status["components"]["llm_providers"] = {}
    for provider in ["openai", "anthropic", "google"]:
        api_key_attr = f"{provider}_api_key"
        has_key = getattr(settings, api_key_attr) is not None
        health_status["components"]["llm_providers"][provider] = {
            "configured": has_key,
            "status": "available" if has_key else "not_configured"
        }

    # Check LangSmith
    health_status["components"]["langsmith"] = {
        "enabled": bool(settings.langsmith_api_key),
        "status": "enabled" if settings.langsmith_api_key else "disabled"
    }

    # Check coordinator
    try:
        domains = coordinator.get_available_domains()
        health_status["components"]["coordinator"] = {
            "status": "healthy",
            "available_domains": len(domains)
        }
    except Exception as e:
        health_status["components"]["coordinator"] = {
            "status": "unhealthy",
            "error": str(e)
        }
        health_status["status"] = "unhealthy"

    # Add key metrics
    current_metrics = metrics.get_metrics()
    health_status["metrics"] = {
        "total_workflows": current_metrics.get("workflows_completed", 0) + current_metrics.get("workflows_failed", 0),
        "successful_workflows": current_metrics.get("workflows_completed", 0),
        "failed_workflows": current_metrics.get("workflows_failed", 0),
        "llm_calls_total": current_metrics.get("llm_calls_total", 0),
        "retry_attempts": current_metrics.get("retry_attempts", 0)
    }

    # Add system resource info
    try:
        health_status["system"] = {
            "memory_percent": psutil.virtual_memory().percent,
            "cpu_percent": psutil.cpu_percent(interval=0.1),
            "active_connections": len(psutil.net_connections()) if hasattr(psutil, 'net_connections') else 0
        }
    except ImportError:
        health_status["system"] = {
            "note": "System monitoring not available (psutil not installed)"
        }

    # Readiness and liveness probes
    health_status["readiness"] = health_status["status"] == "healthy"
    health_status["liveness"] = True  # Basic liveness check

    return health_status


def main():
    """Main entry point for running the application."""
    # Set up LangSmith if configured
    setup_langsmith()

    # Start the server
    logger.info("Starting Multi-Agent Prompt Engineering System...")
    logger.info(f"Server will run on {settings.host}:{settings.port}")

    uvicorn.run(
        "src.main:app",
        host=settings.host,
        port=settings.port,
        reload=True,
        log_level=settings.log_level.lower()
    )


if __name__ == "__main__":
    main()
