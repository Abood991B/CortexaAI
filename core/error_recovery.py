"""
Error Recovery & Observability for CortexaAI.

Structured error taxonomy, workflow-level retry with exponential backoff,
error analytics, and webhook alerting on repeated failures.
"""

import time
import traceback
import asyncio
from enum import Enum
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
from dataclasses import dataclass, field

from config.config import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Error Taxonomy
# ---------------------------------------------------------------------------

class ErrorCategory(str, Enum):
    TRANSIENT = "transient"      # Retry automatically
    PERMANENT = "permanent"      # Fail fast
    QUOTA = "quota"              # Switch provider then retry
    TIMEOUT = "timeout"          # Retry with longer timeout
    VALIDATION = "validation"    # Bad input — fail fast
    UNKNOWN = "unknown"


@dataclass
class ClassifiedError:
    """A classified error with metadata for retry decisions."""
    category: ErrorCategory
    original_exception: Exception
    message: str
    provider: Optional[str] = None
    error_code: Optional[str] = None
    is_retryable: bool = False
    suggested_action: str = "fail"
    context: Dict[str, Any] = field(default_factory=dict)


def classify_error(exc: Exception, provider: str = None) -> ClassifiedError:
    """Classify an exception into a structured error category."""
    msg = str(exc).lower()

    # Quota / rate-limit errors
    if any(kw in msg for kw in ("rate limit", "quota", "429", "resource exhausted", "too many requests")):
        return ClassifiedError(
            category=ErrorCategory.QUOTA,
            original_exception=exc,
            message=str(exc),
            provider=provider,
            error_code="QUOTA_EXCEEDED",
            is_retryable=True,
            suggested_action="switch_provider",
        )

    # Timeout errors
    if any(kw in msg for kw in ("timeout", "timed out", "deadline exceeded", "asyncio.timeout")):
        return ClassifiedError(
            category=ErrorCategory.TIMEOUT,
            original_exception=exc,
            message=str(exc),
            provider=provider,
            error_code="TIMEOUT",
            is_retryable=True,
            suggested_action="retry_with_backoff",
        )

    # Transient / connection errors
    if any(kw in msg for kw in ("connection", "disconnect", "503", "502", "500", "service unavailable", "temporarily")):
        return ClassifiedError(
            category=ErrorCategory.TRANSIENT,
            original_exception=exc,
            message=str(exc),
            provider=provider,
            error_code="TRANSIENT_ERROR",
            is_retryable=True,
            suggested_action="retry_with_backoff",
        )

    # Validation errors
    if any(kw in msg for kw in ("validation", "invalid", "malformed", "400", "bad request", "schema")):
        return ClassifiedError(
            category=ErrorCategory.VALIDATION,
            original_exception=exc,
            message=str(exc),
            provider=provider,
            error_code="VALIDATION_ERROR",
            is_retryable=False,
            suggested_action="fail",
        )

    # Permanent errors
    if any(kw in msg for kw in ("401", "403", "not found", "404", "authentication", "forbidden")):
        return ClassifiedError(
            category=ErrorCategory.PERMANENT,
            original_exception=exc,
            message=str(exc),
            provider=provider,
            error_code="PERMANENT_ERROR",
            is_retryable=False,
            suggested_action="fail",
        )

    # Default: unknown — treat as retryable once
    return ClassifiedError(
        category=ErrorCategory.UNKNOWN,
        original_exception=exc,
        message=str(exc),
        provider=provider,
        error_code="UNKNOWN_ERROR",
        is_retryable=True,
        suggested_action="retry_once",
    )


# ---------------------------------------------------------------------------
# Workflow-level Retry with Exponential Backoff
# ---------------------------------------------------------------------------

class WorkflowRetryPolicy:
    """Configurable retry policy for workflow-level operations."""

    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
    ):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter

    def get_delay(self, attempt: int) -> float:
        """Calculate delay for a given attempt number (0-indexed)."""
        import random
        delay = self.base_delay * (self.exponential_base ** attempt)
        delay = min(delay, self.max_delay)
        if self.jitter:
            delay *= (0.5 + random.random())
        return delay


async def retry_with_policy(
    func: Callable,
    *args,
    policy: WorkflowRetryPolicy = None,
    on_retry: Callable = None,
    provider: str = None,
    **kwargs,
) -> Any:
    """
    Execute an async function with structured retry and error classification.

    Args:
        func: Async callable to execute
        policy: Retry policy (default: 3 retries, exponential backoff)
        on_retry: Optional callback(attempt, classified_error, delay)
        provider: LLM provider name for error classification
    """
    policy = policy or WorkflowRetryPolicy()
    last_error = None

    for attempt in range(policy.max_retries + 1):
        try:
            return await func(*args, **kwargs)
        except Exception as exc:
            classified = classify_error(exc, provider=provider)
            last_error = classified

            if not classified.is_retryable or attempt >= policy.max_retries:
                logger.error(
                    f"Operation failed permanently ({classified.category.value}): {classified.message}"
                )
                raise exc

            delay = policy.get_delay(attempt)
            logger.warning(
                f"Retry {attempt + 1}/{policy.max_retries} after {delay:.1f}s "
                f"({classified.category.value}): {classified.message}"
            )

            if on_retry:
                try:
                    on_retry(attempt, classified, delay)
                except Exception:
                    pass

            await asyncio.sleep(delay)

    # Should not reach here
    if last_error:
        raise last_error.original_exception


# ---------------------------------------------------------------------------
# Error Analytics Tracker (in-memory, flushed to DB periodically)
# ---------------------------------------------------------------------------

class ErrorAnalytics:
    """In-memory error analytics with periodic database flush."""

    def __init__(self):
        self._errors: List[Dict[str, Any]] = []
        self._counters: Dict[str, int] = {}

    def record(self, error: ClassifiedError, workflow_id: str = None):
        """Record a classified error."""
        entry = {
            "error_type": error.category.value,
            "error_code": error.error_code,
            "severity": "high" if error.category == ErrorCategory.PERMANENT else "medium",
            "message": error.message[:500],
            "provider": error.provider,
            "workflow_id": workflow_id,
            "stack_trace": traceback.format_exception(type(error.original_exception), error.original_exception, error.original_exception.__traceback__) if error.original_exception else None,
            "suggested_action": error.suggested_action,
            "timestamp": datetime.now().isoformat(),
        }
        self._errors.append(entry)

        # Update counters
        self._counters[error.category.value] = self._counters.get(error.category.value, 0) + 1
        if error.provider:
            key = f"provider:{error.provider}"
            self._counters[key] = self._counters.get(key, 0) + 1

        # Keep only last 1000 in memory
        if len(self._errors) > 1000:
            self._errors = self._errors[-1000:]

        # Persist to database
        try:
            from core.database import db
            db.log_error(entry)
        except Exception:
            pass

    def get_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get error summary for the specified period."""
        try:
            from core.database import db
            return db.get_error_analytics(hours=hours)
        except Exception:
            pass

        # Fallback to in-memory
        cutoff = time.time() - (hours * 3600)
        recent = [e for e in self._errors if datetime.fromisoformat(e["timestamp"]).timestamp() > cutoff]
        by_type: Dict[str, int] = {}
        by_provider: Dict[str, int] = {}
        for e in recent:
            by_type[e["error_type"]] = by_type.get(e["error_type"], 0) + 1
            if e.get("provider"):
                by_provider[e["provider"]] = by_provider.get(e["provider"], 0) + 1

        return {
            "total_errors": len(recent),
            "period_hours": hours,
            "by_type": by_type,
            "by_provider": by_provider,
            "recent_errors": recent[-20:],
        }

    def get_counters(self) -> Dict[str, int]:
        return dict(self._counters)

    def get_recent(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get the most recent errors."""
        try:
            from core.database import db
            rows = db.fetch_all(
                "SELECT error_type, message, context, timestamp FROM error_log ORDER BY timestamp DESC LIMIT ?",
                (limit,),
            )
            return [
                {"error_type": r[0], "message": r[1], "context": r[2], "timestamp": r[3]}
                for r in rows
            ]
        except Exception:
            return self._errors[-limit:]


# Global error analytics
error_analytics = ErrorAnalytics()
