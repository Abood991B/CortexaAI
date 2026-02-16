"""System endpoints: health, metrics, dashboard, providers, cache, optimization, errors."""

import time
import asyncio
from typing import Dict, Any

from fastapi import APIRouter, HTTPException
import psutil

from src.deps import (
    logger,
    settings,
    metrics,
    coordinator,
    llm_provider,
    PROVIDER_CONFIGS,
    cache_manager,
    optimization_engine,
    error_analytics,
    db,
    plugin_manager,
    similarity_engine,
    marketplace,
)

router = APIRouter()

# Track server start time at module level for uptime calculation
_server_start_time = time.time()


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

@router.get("/health")
async def health_check():
    """Enhanced health check endpoint with detailed system status."""
    health_status: Dict[str, Any] = {
        "status": "healthy",
        "timestamp": time.time(),
        "version": "3.0.0",
        "uptime_seconds": time.time() - _server_start_time,
        "components": {},
        "metrics": {},
    }

    # Check LLM providers
    health_status["components"]["llm_providers"] = {}
    _placeholder_patterns = ("your_", "sk-xxx", "placeholder", "_here")

    async def _verify_single(provider: str):
        cached_key = f"_provider_verified_{provider}"
        cached = getattr(health_check, cached_key, None)
        cache_age = time.time() - getattr(health_check, f"_provider_ts_{provider}", 0)
        if cached and cache_age < 300:
            return provider, cached
        try:
            result = await llm_provider.verify_provider(provider)
            entry = {
                "configured": True,
                "status": "available" if result["available"] else "unavailable",
                "verified": result["available"],
                "latency_ms": result.get("latency_ms"),
                "model": result.get("model"),
                "error": result.get("error"),
            }
            setattr(health_check, cached_key, entry)
            setattr(health_check, f"_provider_ts_{provider}", time.time())
            return provider, entry
        except Exception:
            return provider, {"configured": True, "status": "unknown", "verified": False}

    verify_tasks = []
    for provider in ["openai", "anthropic", "google", "groq", "deepseek", "openrouter"]:
        api_key = getattr(settings, f"{provider}_api_key", None) or ""
        if not api_key or any(p in api_key.lower() for p in _placeholder_patterns):
            continue
        verify_tasks.append(_verify_single(provider))

    if verify_tasks:
        results = await asyncio.gather(*verify_tasks, return_exceptions=True)
        for item in results:
            if isinstance(item, Exception):
                continue
            prov_name, prov_entry = item
            health_status["components"]["llm_providers"][prov_name] = prov_entry

    # LangSmith
    health_status["components"]["langsmith"] = {
        "enabled": bool(settings.langsmith_api_key),
        "status": "enabled" if settings.langsmith_api_key else "disabled",
    }

    # Coordinator
    try:
        domains = coordinator.get_available_domains()
        health_status["components"]["coordinator"] = {
            "status": "healthy",
            "available_domains": len(domains),
        }
    except Exception as e:
        health_status["components"]["coordinator"] = {"status": "unhealthy", "error": str(e)}
        health_status["status"] = "unhealthy"

    # Key metrics
    current_metrics = metrics.get_metrics()
    health_status["metrics"] = {
        "total_workflows": current_metrics.get("workflows_completed", 0) + current_metrics.get("workflows_failed", 0),
        "successful_workflows": current_metrics.get("workflows_completed", 0),
        "failed_workflows": current_metrics.get("workflows_failed", 0),
        "llm_calls_total": current_metrics.get("llm_calls_total", 0),
        "retry_attempts": current_metrics.get("retry_attempts", 0),
    }

    # System resources
    try:
        health_status["system"] = {
            "memory_percent": psutil.virtual_memory().percent,
            "cpu_percent": psutil.cpu_percent(interval=0),
            "active_connections": len(psutil.net_connections()) if hasattr(psutil, "net_connections") else 0,
        }
    except ImportError:
        health_status["system"] = {"note": "System monitoring not available (psutil not installed)"}

    health_status["readiness"] = health_status["status"] == "healthy"
    health_status["liveness"] = True

    return health_status


# ---------------------------------------------------------------------------
# GET /metrics
# ---------------------------------------------------------------------------

@router.get("/metrics")
async def get_metrics_endpoint():
    """Prometheus-compatible metrics endpoint."""
    lines = []
    current_metrics = metrics.get_metrics()

    lines.append("# HELP system_info System information")
    lines.append("# TYPE system_info gauge")
    lines.append(
        'system_info{version="3.0.0",langsmith_enabled="'
        + str(bool(settings.langsmith_api_key)).lower()
        + '"} 1'
    )

    lines.append("# HELP llm_calls_total Total number of LLM calls")
    lines.append("# TYPE llm_calls_total counter")
    lines.append(f'llm_calls_total {current_metrics.get("llm_calls_total", 0)}')

    lines.append("# HELP llm_calls_success Successful LLM calls")
    lines.append("# TYPE llm_calls_success counter")
    lines.append(f'llm_calls_success {current_metrics.get("llm_calls_success", 0)}')

    lines.append("# HELP llm_calls_error Failed LLM calls")
    lines.append("# TYPE llm_calls_error counter")
    lines.append(f'llm_calls_error {current_metrics.get("llm_calls_error", 0)}')

    lines.append("# HELP workflows_completed Completed workflows")
    lines.append("# TYPE workflows_completed counter")
    lines.append(f'workflows_completed {current_metrics.get("workflows_completed", 0)}')

    lines.append("# HELP workflows_failed Failed workflows")
    lines.append("# TYPE workflows_failed counter")
    lines.append(f'workflows_failed {current_metrics.get("workflows_failed", 0)}')

    lines.append("# HELP retry_attempts_total Total retry attempts")
    lines.append("# TYPE retry_attempts_total counter")
    lines.append(f'retry_attempts_total {current_metrics.get("retry_attempts", 0)}')

    durations = current_metrics.get("llm_call_duration_seconds", [])
    if durations:
        lines.append("# HELP llm_call_duration_seconds LLM call duration in seconds")
        lines.append("# TYPE llm_call_duration_seconds histogram")
        lines.append(f"llm_call_duration_seconds_count {len(durations)}")
        lines.append(f"llm_call_duration_seconds_sum {sum(durations)}")
        buckets = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, float("inf")]
        bucket_counts = [0] * len(buckets)
        for duration in durations:
            for i, bucket in enumerate(buckets):
                if duration <= bucket:
                    bucket_counts[i] += 1
                    break
        for i, bucket in enumerate(buckets):
            lines.append(f'llm_call_duration_seconds_bucket{{le="{bucket}"}} {bucket_counts[i]}')

    domains = current_metrics.get("domains_processed", {})
    for domain, count in domains.items():
        lines.append(f"# HELP domain_processed_total Total workflows processed for domain {domain}")
        lines.append("# TYPE domain_processed_total counter")
        lines.append(f'domain_processed_total{{domain="{domain}"}} {count}')

    try:
        process = psutil.Process()
        memory_info = process.memory_info()
        lines.append("# HELP process_memory_bytes Process memory usage in bytes")
        lines.append("# TYPE process_memory_bytes gauge")
        lines.append(f"process_memory_bytes {memory_info.rss}")
        lines.append("# HELP system_memory_percent System memory usage percentage")
        lines.append("# TYPE system_memory_percent gauge")
        lines.append(f"system_memory_percent {psutil.virtual_memory().percent}")
        lines.append("# HELP system_cpu_percent System CPU usage percentage")
        lines.append("# TYPE system_cpu_percent gauge")
        lines.append(f"system_cpu_percent {psutil.cpu_percent(interval=0)}")
    except ImportError:
        pass

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Provider management
# ---------------------------------------------------------------------------

@router.get("/api/providers")
async def get_providers():
    """Get status of all configured LLM providers."""
    return {
        "providers": llm_provider.get_provider_status(),
        "default_provider": settings.default_model_provider,
        "default_model": settings.default_model_name,
        "available": llm_provider.get_available_providers(),
    }


@router.post("/api/providers/{provider_name}/reset")
async def reset_provider_health(provider_name: str):
    """Reset health status for a provider."""
    if provider_name not in PROVIDER_CONFIGS:
        raise HTTPException(status_code=404, detail=f"Unknown provider: {provider_name}")
    llm_provider.reset_health(provider_name)
    return {"status": "ok", "message": f"Health reset for provider: {provider_name}"}


# ---------------------------------------------------------------------------
# Cache management
# ---------------------------------------------------------------------------

@router.get("/api/cache/stats")
async def get_cache_stats():
    return cache_manager.get_stats()


@router.delete("/api/cache")
async def clear_cache():
    cache_manager.clear()
    return {"status": "ok", "message": "Cache cleared"}


# ---------------------------------------------------------------------------
# Optimization engine
# ---------------------------------------------------------------------------

@router.get("/api/optimization/dashboard")
async def get_optimization_dashboard():
    return optimization_engine.get_dashboard_data()


@router.get("/api/optimization/analytics")
async def get_optimization_analytics():
    return optimization_engine.analytics.get_summary()


@router.get("/api/optimization/ab-tests")
async def get_ab_tests():
    return {
        "stats": optimization_engine.ab_testing.get_stats(),
        "history": optimization_engine.ab_testing.get_test_history(),
    }


@router.get("/api/optimization/versions")
async def get_prompt_versions():
    return optimization_engine.version_control.get_stats()


# ---------------------------------------------------------------------------
# Error analytics
# ---------------------------------------------------------------------------

@router.get("/api/errors/analytics")
async def get_error_analytics():
    return error_analytics.get_summary()


@router.get("/api/errors/recent")
async def get_recent_errors(limit: int = 20):
    return error_analytics.get_recent(limit)


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------

@router.get("/api/dashboard")
async def get_dashboard():
    """Aggregated dashboard."""
    return {
        "database": db.get_dashboard_stats(),
        "cache": {"stats": metrics.get_metrics()},
        "optimization": optimization_engine.get_dashboard_data(),
        "plugins": plugin_manager.list_plugins(),
        "similarity": similarity_engine.stats(),
        "marketplace": marketplace.stats(),
        "version": "3.0.0",
    }
