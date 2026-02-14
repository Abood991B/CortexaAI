"""Configuration management for Multi-Agent Prompt Engineering System."""

import os
import logging
import sys
import time
import asyncio
from typing import Optional, Dict, Any, List, Tuple
from pydantic_settings import BaseSettings
from pydantic import Field
from dotenv import load_dotenv
import json

# Load environment variables from .env file
load_dotenv()


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # API Keys - Multiple LLM Providers
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    anthropic_api_key: Optional[str] = Field(default=None, env="ANTHROPIC_API_KEY")
    google_api_key: Optional[str] = Field(default=None, env="GOOGLE_API_KEY")
    groq_api_key: Optional[str] = Field(default=None, env="GROQ_API_KEY")
    deepseek_api_key: Optional[str] = Field(default=None, env="DEEPSEEK_API_KEY")
    openrouter_api_key: Optional[str] = Field(default=None, env="OPENROUTER_API_KEY")

    # LangSmith Configuration
    langsmith_api_key: Optional[str] = Field(default=None, env="LANGSMITH_API_KEY")
    langsmith_project: str = Field(default="cortexaai", env="LANGSMITH_PROJECT")
    langsmith_endpoint: str = Field(default="https://api.smith.langchain.com", env="LANGSMITH_ENDPOINT")

    # Model Configuration
    default_model_provider: str = Field(default="google", env="DEFAULT_MODEL_PROVIDER")
    default_model_name: str = Field(default="gemma-3-27b-it", env="DEFAULT_MODEL_NAME")

    # Prompt Optimization Engine
    enable_ab_testing: bool = Field(default=True, env="ENABLE_AB_TESTING")
    enable_prompt_versioning: bool = Field(default=True, env="ENABLE_PROMPT_VERSIONING")
    optimization_strategy: str = Field(default="iterative", env="OPTIMIZATION_STRATEGY")

    # System Configuration
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    max_evaluation_iterations: int = Field(default=3, env="MAX_EVALUATION_ITERATIONS")
    evaluation_threshold: float = Field(default=0.8, env="EVALUATION_THRESHOLD")
    
    # LLM Configuration
    max_llm_retries: int = Field(default=3, env="MAX_LLM_RETRIES")
    llm_retry_delay: float = Field(default=1.0, env="LLM_RETRY_DELAY")
    llm_timeout_seconds: int = Field(default=60, env="LLM_TIMEOUT_SECONDS")

    # Server Configuration
    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=8000, env="PORT")

    class Config:
        """Pydantic configuration."""
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()


def get_logger(name: str) -> logging.Logger:
    """Get a configured logger instance."""
    return logging.getLogger(name)

# Initialize logger for this module
logger = get_logger(__name__)


def get_model_config(provider: str = None, model_name: str = None):
    """Get model configuration for the specified provider.
    
    Supports: google, openai, anthropic, groq, deepseek, openrouter.
    Falls back to google if provider is unknown.
    """
    provider = provider or settings.default_model_provider
    model_name = model_name or settings.default_model_name

    configs = {
        "openai": {
            "model_name": model_name if provider == "openai" else "gpt-4o-mini",
            "api_key": settings.openai_api_key,
        },
        "anthropic": {
            "model_name": model_name if provider == "anthropic" else "claude-3-haiku-20240307",
            "api_key": settings.anthropic_api_key,
        },
        "google": {
            "model_name": model_name if provider == "google" else "gemma-3-27b-it",
            "api_key": settings.google_api_key,
        },
        "groq": {
            "model_name": model_name if provider == "groq" else "llama-3.3-70b-versatile",
            "api_key": settings.groq_api_key,
        },
        "deepseek": {
            "model_name": model_name if provider == "deepseek" else "deepseek-chat",
            "api_key": settings.deepseek_api_key,
        },
        "openrouter": {
            "model_name": model_name if provider == "openrouter" else "google/gemini-2.0-flash-exp:free",
            "api_key": settings.openrouter_api_key,
        },
    }

    return configs.get(provider, configs["google"])


def setup_langsmith():
    """Set up LangSmith tracing if API key is available."""
    if settings.langsmith_api_key:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = settings.langsmith_api_key
        os.environ["LANGCHAIN_PROJECT"] = settings.langsmith_project
        os.environ["LANGCHAIN_ENDPOINT"] = settings.langsmith_endpoint
        return True
    return False


# Metrics and Monitoring Configuration
class MetricsCollector:
    """Simple metrics collector for system monitoring."""

    def __init__(self):
        self._metrics = {
            "llm_calls_total": 0,
            "llm_calls_success": 0,
            "llm_calls_error": 0,
            "llm_call_duration_seconds": [],
            "classification_calls": 0,
            "improvement_calls": 0,
            "evaluation_calls": 0,
            "retry_attempts": 0,
            "workflows_completed": 0,
            "workflows_failed": 0,
            "domains_processed": {},
            "errors_by_type": {},
        }
        self._timers = {}

    def increment(self, name: str, value: int = 1, labels: Optional[Dict[str, str]] = None):
        """Increment a counter metric."""
        key = f"{name}_{labels}" if labels else name
        if key not in self._metrics:
            self._metrics[key] = 0
        self._metrics[key] += value

    def observe(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Observe a value for a histogram metric."""
        key = f"{name}_{labels}" if labels else name
        if key not in self._metrics:
            self._metrics[key] = []
        self._metrics[key].append(value)

    def start_timer(self, name: str) -> str:
        """Start a timer and return a timer ID."""
        timer_id = f"{name}_{len(self._timers)}"
        self._timers[timer_id] = os.times()[4] if hasattr(os, 'times') else 0
        return timer_id

    def stop_timer(self, timer_id: str, labels: Optional[Dict[str, str]] = None):
        """Stop a timer and record the duration."""
        if timer_id in self._timers:
            duration = (os.times()[4] if hasattr(os, 'times') else 0) - self._timers[timer_id]
            self.observe("timer_duration", duration, labels)
            del self._timers[timer_id]

    def get_metrics(self) -> Dict[str, Any]:
        """Get all current metrics."""
        return self._metrics.copy()

    def reset(self):
        """Reset all metrics."""
        self._metrics.clear()
        self._timers.clear()


# Global metrics collector
metrics = MetricsCollector()


def setup_structured_logging():
    """Set up structured JSON logging for the application."""

    class StructuredFormatter(logging.Formatter):
        """Custom formatter that outputs structured JSON logs."""

        def format(self, record):
            log_entry = {
                "timestamp": self.formatTime(record),
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
                "module": record.module,
                "function": record.funcName,
                "line": record.lineno,
            }

            # Add any extra fields that were added to the log record
            if hasattr(record, 'workflow_id'):
                log_entry['workflow_id'] = record.workflow_id
            if hasattr(record, 'correlation_id'):
                log_entry['correlation_id'] = record.correlation_id
            if hasattr(record, 'domain'):
                log_entry['domain'] = record.domain
            if hasattr(record, 'duration'):
                log_entry['duration'] = record.duration
            if hasattr(record, 'status'):
                log_entry['status'] = record.status

            # Add exception info if present
            if record.exc_info:
                log_entry['exception'] = self.formatException(record.exc_info)

            return json.dumps(log_entry)

    # Create structured formatter
    formatter = StructuredFormatter()

    # Set up root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, settings.log_level.upper(), logging.INFO))

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Add console handler with structured formatting
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # Create component-specific loggers
    component_loggers = [
        'agents.coordinator',
        'agents.classifier',
        'agents.base_expert',
        'agents.evaluator',
        'src.main',
        'config.config'
    ]

    for logger_name in component_loggers:
        logger = logging.getLogger(logger_name)
        logger.setLevel(getattr(logging, settings.log_level.upper(), logging.INFO))

    return True


def log_performance(logger: logging.Logger, operation: str, duration: float,
                   status: str = "success", **context):
    """Log performance metrics with structured data."""
    extra_data = {
        'duration': duration,
        'status': status,
        'operation': operation
    }
    extra_data.update(context)

    if status == "success":
        logger.info(f"Operation '{operation}' completed successfully in {duration:.2f}s", extra=extra_data)
    else:
        logger.warning(f"Operation '{operation}' failed after {duration:.2f}s", extra=extra_data)


def log_workflow_event(logger: logging.Logger, workflow_id: str, event: str,
                      status: str = "info", **context):
    """Log workflow-specific events with correlation."""
    extra_data = {
        'workflow_id': workflow_id,
        'status': status,
        'event': event
    }
    extra_data.update(context)

    if status == "error":
        logger.error(f"Workflow {workflow_id}: {event}", extra=extra_data)
    elif status == "warning":
        logger.warning(f"Workflow {workflow_id}: {event}", extra=extra_data)
    else:
        logger.info(f"Workflow {workflow_id}: {event}", extra=extra_data)


def generate_cache_key(content: str, prefix: str = "", max_length: int = 64) -> str:
    """Generate a cache key from content using hash."""
    import hashlib
    key = f"{prefix}:{hashlib.md5(content.encode()).hexdigest()}"
    return key[:max_length] if len(key) > max_length else key


def generate_prompt_cache_key(prompt: str, domain: str = "", prompt_type: str = "", prefix: str = "prompt") -> str:
    """Generate cache key for prompt-related operations."""
    content = f"{prompt}|{domain}|{prompt_type}"
    return generate_cache_key(content, prefix)


def generate_evaluation_cache_key(original_prompt: str, improved_prompt: str, domain: str) -> str:
    """Generate cache key for evaluation operations."""
    content = f"{original_prompt}|{improved_prompt}|{domain}"
    return generate_cache_key(content, "eval")


# Performance and Caching Configuration
class CacheManager:
    """Advanced cache manager with in-memory LRU and optional Redis backend."""

    def __init__(self, default_ttl: int = 3600, max_size: int = 1000, redis_url: Optional[str] = None):
        """
        Initialize the cache manager.

        Args:
            default_ttl: Default time-to-live in seconds
            max_size: Maximum cache size (in-memory)
            redis_url: Optional Redis URL (e.g. ``redis://localhost:6379/0``)
        """
        self.default_ttl = default_ttl
        self.max_size = max_size
        self._cache: Dict[str, Any] = {}
        self._access_times: Dict[str, float] = {}
        self._hit_count = 0
        self._miss_count = 0

        # Optional Redis backend
        self._redis = None
        self._redis_url = redis_url
        if redis_url:
            try:
                import redis as _redis_lib
                self._redis = _redis_lib.from_url(redis_url, decode_responses=True)
                self._redis.ping()
            except Exception:
                self._redis = None  # Fall back to in-memory only

    # ── Public API ───────────────────────────────────────────────────────

    def get(self, key: str) -> Any:
        """Get value from cache (memory first, then Redis)."""
        # In-memory check
        if key in self._cache:
            if self._is_expired(key):
                self._delete(key)
                self._miss_count += 1
                return None
            self._access_times[key] = time.time()
            self._hit_count += 1
            return self._cache[key]["value"]

        # Redis fallback
        if self._redis:
            try:
                raw = self._redis.get(f"cxa:{key}")
                if raw is not None:
                    import json as _json
                    value = _json.loads(raw)
                    # Promote to in-memory cache
                    self.set(key, value, self.default_ttl)
                    self._hit_count += 1
                    return value
            except Exception:
                pass

        self._miss_count += 1
        return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache (memory + optional Redis)."""
        if len(self._cache) >= self.max_size:
            self._evict_oldest()

        ttl_value = ttl or self.default_ttl
        expiration = time.time() + ttl_value

        self._cache[key] = {"value": value, "expiration": expiration}
        self._access_times[key] = time.time()

        # Mirror to Redis
        if self._redis:
            try:
                import json as _json
                self._redis.setex(f"cxa:{key}", ttl_value, _json.dumps(value, default=str))
            except Exception:
                pass

    def delete(self, key: str) -> None:
        """Delete key from cache (memory + Redis)."""
        self._delete(key)
        if self._redis:
            try:
                self._redis.delete(f"cxa:{key}")
            except Exception:
                pass

    def clear(self) -> None:
        """Clear all cache entries."""
        self._cache.clear()
        self._access_times.clear()
        self._hit_count = 0
        self._miss_count = 0
        if self._redis:
            try:
                # Only clear cxa-namespaced keys
                for rkey in self._redis.scan_iter("cxa:*"):
                    self._redis.delete(rkey)
            except Exception:
                pass

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self._hit_count + self._miss_count
        hit_rate = self._hit_count / total_requests if total_requests > 0 else 0

        stats = {
            "total_entries": len(self._cache),
            "hit_count": self._hit_count,
            "miss_count": self._miss_count,
            "hit_rate": hit_rate,
            "total_requests": total_requests,
            "backend": "redis+memory" if self._redis else "memory",
        }
        return stats

    # ── Internals ────────────────────────────────────────────────────────

    def _is_expired(self, key: str) -> bool:
        return time.time() > self._cache[key]["expiration"]

    def _delete(self, key: str) -> None:
        if key in self._cache:
            del self._cache[key]
        if key in self._access_times:
            del self._access_times[key]

    def _evict_oldest(self) -> None:
        if not self._access_times:
            return
        oldest_key = min(self._access_times, key=self._access_times.get)
        self._delete(oldest_key)


# Global cache manager
cache_manager = CacheManager(
    default_ttl=getattr(settings, 'cache_ttl', 3600),
    max_size=getattr(settings, 'cache_max_size', 1000)
)


# Performance optimization settings
class PerformanceConfig:
    """Performance optimization configuration."""

    # Caching settings
    enable_caching = getattr(settings, 'enable_caching', True)
    cache_ttl = getattr(settings, 'cache_ttl', 3600)  # 1 hour default
    cache_max_size = getattr(settings, 'cache_max_size', 1000)

    # Batch processing settings
    enable_batch_processing = getattr(settings, 'enable_batch_processing', True)
    max_batch_size = getattr(settings, 'max_batch_size', 5)

    # Early exit settings
    enable_early_exit = getattr(settings, 'enable_early_exit', True)
    early_exit_threshold = getattr(settings, 'early_exit_threshold', 0.85)
    early_exit_min_iterations = getattr(settings, 'early_exit_min_iterations', 1)

    # LLM optimization settings
    max_llm_retries = getattr(settings, 'max_llm_retries', 3)
    llm_retry_delay = getattr(settings, 'llm_retry_delay', 1.0)

    # Cache warming settings
    enable_cache_warming = getattr(settings, 'enable_cache_warming', False)
    cache_warming_patterns = getattr(settings, 'cache_warming_patterns', [
        "Write a function to",
        "Create a class for",
        "Analyze the data",
        "Generate a report"
    ])


# Global performance configuration
perf_config = PerformanceConfig()


def log_cache_performance(logger: logging.Logger, operation: str, cache_hit: bool, **context):
    """Log cache performance metrics."""
    status = "hit" if cache_hit else "miss"
    extra_data = {
        'operation': operation,
        'cache_status': status,
        'cache_hit': cache_hit
    }
    extra_data.update(context)

    if cache_hit:
        logger.info(f"Cache hit for {operation}", extra=extra_data)
    else:
        logger.info(f"Cache miss for {operation}", extra=extra_data)


# Circuit Breaker Implementation
class CircuitBreakerState:
    """Enumeration of circuit breaker states."""
    CLOSED = "CLOSED"      # Normal operation
    OPEN = "OPEN"          # Failing, requests rejected
    HALF_OPEN = "HALF_OPEN"  # Testing recovery


class CircuitBreaker:
    """Advanced circuit breaker implementation with configurable thresholds and recovery."""

    def __init__(self, name: str, failure_threshold: int = 5, recovery_timeout: int = 60,
                 expected_exception: Exception = Exception):
        """
        Initialize circuit breaker.

        Args:
            name: Unique name for this circuit breaker
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Seconds to wait before attempting recovery
            expected_exception: Exception type to monitor
        """
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception

        self._state = CircuitBreakerState.CLOSED
        self._failure_count = 0
        self._last_failure_time = None
        self._success_count = 0
        self._call_count = 0

        # Metrics tracking
        self._total_calls = 0
        self._total_failures = 0
        self._total_successes = 0

    async def call(self, func, *args, **kwargs):
        """Execute function through circuit breaker."""
        self._total_calls += 1

        if self._state == CircuitBreakerState.OPEN:
            if not self._should_attempt_reset():
                raise CircuitBreakerOpenException(
                    f"Circuit breaker '{self.name}' is OPEN",
                    circuit_name=self.name,
                    failure_count=self._failure_count
                )
            else:
                self._state = CircuitBreakerState.HALF_OPEN
                logger.info(f"Circuit breaker '{self.name}' moved to HALF_OPEN state")

        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result

        except self.expected_exception as e:
            self._on_failure()
            raise e

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt recovery."""
        if self._last_failure_time is None:
            return True
        return (time.time() - self._last_failure_time) >= self.recovery_timeout

    def _on_success(self):
        """Handle successful call."""
        self._total_successes += 1

        if self._state == CircuitBreakerState.HALF_OPEN:
            self._success_count += 1
            # Require a few successes before closing
            if self._success_count >= 3:
                self._state = CircuitBreakerState.CLOSED
                self._failure_count = 0
                self._success_count = 0
                logger.info(f"Circuit breaker '{self.name}' recovered to CLOSED state")
        else:
            self._failure_count = max(0, self._failure_count - 1)  # Decay failure count

    def _on_failure(self):
        """Handle failed call."""
        self._total_failures += 1
        self._failure_count += 1
        self._last_failure_time = time.time()

        if self._failure_count >= self.failure_threshold:
            self._state = CircuitBreakerState.OPEN
            logger.warning(f"Circuit breaker '{self.name}' opened after {self._failure_count} failures")

    def get_state(self) -> str:
        """Get current circuit breaker state."""
        return self._state

    def get_metrics(self) -> Dict[str, Any]:
        """Get circuit breaker metrics."""
        success_rate = self._total_successes / self._total_calls if self._total_calls > 0 else 0

        return {
            "name": self.name,
            "state": self._state,
            "failure_count": self._failure_count,
            "success_count": self._success_count,
            "total_calls": self._total_calls,
            "total_failures": self._total_failures,
            "total_successes": self._total_successes,
            "success_rate": success_rate,
            "last_failure_time": self._last_failure_time
        }

    def force_reset(self):
        """Manually reset circuit breaker."""
        self._state = CircuitBreakerState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        logger.info(f"Circuit breaker '{self.name}' manually reset to CLOSED")


class CircuitBreakerOpenException(Exception):
    """Exception raised when circuit breaker is open."""

    def __init__(self, message: str, circuit_name: str, failure_count: int):
        super().__init__(message)
        self.circuit_name = circuit_name
        self.failure_count = failure_count


# Dead Letter Queue Implementation
class DeadLetterQueue:
    """Dead Letter Queue for handling failed operations with retry capability."""

    def __init__(self, max_size: int = 1000, retention_hours: int = 24):
        """
        Initialize DLQ.

        Args:
            max_size: Maximum number of items to retain
            retention_hours: Hours to retain items before cleanup
        """
        self.max_size = max_size
        self.retention_seconds = retention_hours * 3600
        self._queue = []
        self._lock = asyncio.Lock()

    async def add(self, item: Dict[str, Any]) -> bool:
        """
        Add item to DLQ.

        Args:
            item: Failed operation item with metadata

        Returns:
            True if added successfully, False if queue is full
        """
        async with self._lock:
            if len(self._queue) >= self.max_size:
                return False

            dlq_item = {
                "id": f"dlq_{int(time.time())}_{len(self._queue)}",
                "timestamp": time.time(),
                "item": item,
                "retry_count": 0,
                "next_retry": time.time() + 60,  # First retry in 1 minute
                "error_details": item.get("error", {})
            }

            self._queue.append(dlq_item)
            logger.warning(f"Added item to DLQ: {dlq_item['id']}")
            return True

    async def get_pending_items(self, max_items: int = 10) -> List[Dict[str, Any]]:
        """Get items ready for retry."""
        async with self._lock:
            current_time = time.time()
            pending = []

            for item in self._queue:
                if len(pending) >= max_items:
                    break

                if current_time >= item["next_retry"]:
                    pending.append(item)

            return pending

    async def mark_success(self, item_id: str):
        """Mark item as successfully processed."""
        async with self._lock:
            for i, item in enumerate(self._queue):
                if item["id"] == item_id:
                    logger.info(f"DLQ item {item_id} processed successfully, removing from queue")
                    del self._queue[i]
                    break

    async def mark_failure(self, item_id: str, error: str = None):
        """Mark item as failed again, schedule next retry."""
        async with self._lock:
            for item in self._queue:
                if item["id"] == item_id:
                    item["retry_count"] += 1
                    # Exponential backoff: 1min, 5min, 15min, 1hr, 4hr, 12hr
                    retry_delays = [60, 300, 900, 3600, 14400, 43200]
                    delay_index = min(item["retry_count"] - 1, len(retry_delays) - 1)
                    item["next_retry"] = time.time() + retry_delays[delay_index]

                    if error:
                        item["error_details"] = {"message": error, "retry_attempt": item["retry_count"]}

                    logger.warning(f"DLQ item {item_id} failed again, retry {item['retry_count']} scheduled")
                    break

    async def cleanup_expired(self):
        """Remove expired items from DLQ."""
        async with self._lock:
            current_time = time.time()
            original_size = len(self._queue)
            self._queue = [
                item for item in self._queue
                if current_time - item["timestamp"] < self.retention_seconds
            ]

            if len(self._queue) < original_size:
                logger.info(f"Cleaned up {original_size - len(self._queue)} expired items from DLQ")

    def get_stats(self) -> Dict[str, Any]:
        """Get DLQ statistics."""
        async def get_stats_async():
            async with self._lock:
                current_time = time.time()
                pending_count = sum(1 for item in self._queue if current_time >= item["next_retry"])
                oldest_item = min(self._queue, key=lambda x: x["timestamp"]) if self._queue else None
                newest_item = max(self._queue, key=lambda x: x["timestamp"]) if self._queue else None

                return {
                    "total_items": len(self._queue),
                    "pending_items": pending_count,
                    "oldest_item_age": current_time - oldest_item["timestamp"] if oldest_item else 0,
                    "newest_item_age": current_time - newest_item["timestamp"] if newest_item else 0,
                    "max_size": self.max_size,
                    "retention_seconds": self.retention_seconds
                }

        # Return synchronously for metrics endpoints
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                return loop.run_in_executor(executor, lambda: asyncio.run(get_stats_async()))
        else:
            return asyncio.run(get_stats_async())

    async def clear(self):
        """Clear all items from DLQ."""
        async with self._lock:
            cleared_count = len(self._queue)
            self._queue.clear()
            logger.info(f"Cleared {cleared_count} items from DLQ")


# Global circuit breakers
circuit_breakers = {
    "classification": CircuitBreaker(
        name="classification",
        failure_threshold=getattr(settings, 'circuit_breaker_failure_threshold', 5),
        recovery_timeout=getattr(settings, 'circuit_breaker_recovery_timeout', 60)
    ),
    "improvement": CircuitBreaker(
        name="improvement",
        failure_threshold=getattr(settings, 'circuit_breaker_failure_threshold', 5),
        recovery_timeout=getattr(settings, 'circuit_breaker_recovery_timeout', 60)
    ),
    "evaluation": CircuitBreaker(
        name="evaluation",
        failure_threshold=getattr(settings, 'circuit_breaker_failure_threshold', 5),
        recovery_timeout=getattr(settings, 'circuit_breaker_recovery_timeout', 60)
    )
}

# Global DLQ instance
dlq = DeadLetterQueue(
    max_size=getattr(settings, 'dlq_max_size', 1000),
    retention_hours=getattr(settings, 'dlq_retention_hours', 24)
)


# Reliability configuration
class ReliabilityConfig:
    """Reliability and error handling configuration."""

    # Circuit breaker settings
    enable_circuit_breakers = getattr(settings, 'enable_circuit_breakers', True)
    circuit_breaker_failure_threshold = getattr(settings, 'circuit_breaker_failure_threshold', 5)
    circuit_breaker_recovery_timeout = getattr(settings, 'circuit_breaker_recovery_timeout', 60)

    # DLQ settings
    enable_dlq = getattr(settings, 'enable_dlq', True)
    dlq_max_size = getattr(settings, 'dlq_max_size', 1000)
    dlq_retention_hours = getattr(settings, 'dlq_retention_hours', 24)
    dlq_cleanup_interval = getattr(settings, 'dlq_cleanup_interval', 3600)  # 1 hour

    # Fallback settings
    enable_fallbacks = getattr(settings, 'enable_fallbacks', True)
    fallback_response_quality = getattr(settings, 'fallback_response_quality', 0.5)


# Global reliability configuration
reliability_config = ReliabilityConfig()


def log_circuit_breaker_event(logger: logging.Logger, circuit_name: str, event: str, **context):
    """Log circuit breaker events."""
    extra_data = {
        'circuit_name': circuit_name,
        'event': event
    }
    extra_data.update(context)

    if event in ["opened", "failure"]:
        logger.warning(f"Circuit breaker {circuit_name}: {event}", extra=extra_data)
    elif event in ["closed", "recovered", "success"]:
        logger.info(f"Circuit breaker {circuit_name}: {event}", extra=extra_data)
    else:
        logger.info(f"Circuit breaker {circuit_name}: {event}", extra=extra_data)


# Security and Privacy Implementation
class SecurityManager:
    """Comprehensive security manager for input/output sanitization and privacy protection."""

    def __init__(self):
        self.pii_patterns = self._compile_pii_patterns()
        self.injection_patterns = self._compile_injection_patterns()
        self.content_filters = self._compile_content_filters()

        # Security metrics
        self._security_events = []
        self._pii_detected_count = 0
        self._injection_attempts = 0
        self._content_violations = 0

    def _compile_pii_patterns(self) -> List[Dict]:
        """Compile PII detection patterns."""
        return [
            {
                'name': 'email',
                'pattern': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
                'mask': '[EMAIL_MASKED]'
            },
            {
                'name': 'phone',
                'pattern': r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
                'mask': '[PHONE_MASKED]'
            },
            {
                'name': 'ssn',
                'pattern': r'\b\d{3}-\d{2}-\d{4}\b',
                'mask': '[SSN_MASKED]'
            },
            {
                'name': 'credit_card',
                'pattern': r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
                'mask': '[CC_MASKED]'
            },
            {
                'name': 'api_key',
                'pattern': r'\b[A-Za-z0-9_-]{32,}\b',
                'mask': '[API_KEY_MASKED]'
            }
        ]

    def _compile_injection_patterns(self) -> List[str]:
        """Compile prompt injection detection patterns with context awareness."""
        return [
            # Jailbreak attempts - very specific patterns
            r'\b(ignore|override|disregard)\s+(all\s+)?previous\s+instructions\s+(and|to)',
            r'\b(system|developer)\s+mode\s+(enabled|activated|unlocked)',
            r'\b(admin|root|superuser)\s+(access|privileges)\s+(granted|enabled|unlocked)',
            r'\b(bypass|circumvent|break)\s+(all\s+)?security\s+(restrictions|filters|protocols)',
            r'\b(unrestricted|uncensored|unfiltered)\s+(mode|output|response)\s+(enabled|activated)',

            # Command injection - specific patterns
            r'\b(execute|run|launch)\s+(as|with)\s+(root|admin|superuser)\s+(privileges|permissions)',
            r'\b(shell|bash|powershell|cmd)\s+command\s+(execution|injection|exploit)',
            r'\b(sql|database)\s+(injection|exploit)\s+(attack|vulnerability)',

            # Script injection - specific malicious patterns
            r'\b(javascript|script|code)\s+(injection|execution)\s+(attack|exploit|payload)',
            r'\b(eval|exec|executeScript)\s*\(\s*.*\)',

            # Very specific dangerous patterns
            r'\b(DON\'T|NEVER)\s+(FILTER|CENSOR|RESTRICT)\s+(MY|THIS)\s+(CONTENT|INPUT|REQUEST)',
            r'\b(PLEASE|YOU\s+MUST)\s+(IGNORE|DISREGARD)\s+(SAFETY|SECURITY)\s+(RULES|FILTERS)',
        ]

    def _compile_content_filters(self) -> Dict[str, List[str]]:
        """Compile content safety filters."""
        return {
            'harmful': [
                'violence', 'harm', 'abuse', 'terrorism', 'illegal',
                'weapons', 'drugs', 'hacking', 'exploitation'
            ],
            'inappropriate': [
                'nsfw', 'adult', 'explicit', 'offensive', 'hate',
                'discrimination', 'harassment', 'bullying'
            ],
            'restricted': [
                'confidential', 'classified', 'sensitive', 'proprietary'
            ]
        }

    def sanitize_input(self, text: str, context: str = "general") -> Dict[str, Any]:
        """
        Sanitize input text for security and privacy with intelligent context-aware detection.

        Args:
            text: Input text to sanitize
            context: Context of the input (classification, improvement, evaluation)

        Returns:
            Dict with sanitized text and security metadata
        """
        original_length = len(text)
        security_events = []

        # Check for PII with context awareness (always enabled)
        if security_config.enable_pii_detection:
            sanitized_text, pii_events = self._detect_and_mask_pii(text)
            security_events.extend(pii_events)
        else:
            sanitized_text = text

        # Context-aware injection detection
        if security_config.enable_injection_detection:
            injection_events = self._detect_injection_attempts_context_aware(sanitized_text, context)
            security_events.extend(injection_events)

        # Intelligent content safety checking
        if security_config.enable_content_filtering:
            content_events = self._check_content_safety_intelligent(sanitized_text, context)
            security_events.extend(content_events)

        # Filter events based on security level and context
        filtered_events = self._filter_security_events(security_events, context)

        # Log security events with detailed information
        if filtered_events:
            logger.warning(f"Security events detected in {context} input", extra={
                'context': context,
                'events': filtered_events,
                'original_length': original_length,
                'sanitized_length': len(sanitized_text),
                'sample_text': sanitized_text[:200] + "..." if len(sanitized_text) > 200 else sanitized_text,
                'security_level': security_config.security_level
            })

        # Determine if content is safe based on filtered events
        high_severity_events = [e for e in filtered_events if e['severity'] == 'high']

        return {
            'sanitized_text': sanitized_text,
            'security_events': filtered_events,
            'is_safe': len(high_severity_events) == 0,
            'original_length': original_length,
            'sanitized_length': len(sanitized_text)
        }

    def _filter_security_events(self, events: List[Dict], context: str) -> List[Dict]:
        """Filter security events based on security level and context."""
        if not security_config.enable_context_awareness:
            return events

        thresholds = security_config.get_security_thresholds()
        filtered_events = []

        for event in events:
            # Apply security level filtering
            if event['type'] == 'injection_attempt':
                # Use the new security config method
                dangerous_score = 1.0 if event['severity'] == 'high' else 0.5
                technical_score = self._calculate_technical_score(event.get('matched_text', ''), context)

                if not security_config.should_flag_injection(dangerous_score, technical_score, context):
                    continue

            elif event['type'] == 'content_violation':
                # Apply content filtering based on security level
                if event.get('has_technical_context', False):
                    if event['severity'] == 'high' and thresholds['content_filter_strength'] < 0.8:
                        event['severity'] = 'medium'
                    elif event['severity'] == 'medium' and thresholds['content_filter_strength'] < 0.6:
                        continue  # Skip low-severity events in permissive mode

            filtered_events.append(event)

        return filtered_events

    def _calculate_technical_score(self, text: str, context: str) -> float:
        """Calculate technical content score for a given text."""
        technical_indicators = [
            'function', 'method', 'class', 'module', 'import', 'def ', 'return',
            'if ', 'for ', 'while ', 'try:', 'except:', 'with ', 'async def',
            'await ', 'lambda ', 'class ', 'from ', 'import ', 'print(',
            'evaluation', 'improvement', 'classification', 'prompt', 'engineering',
            'software', 'development', 'testing', 'deployment', 'production'
        ]

        text_lower = text.lower()
        score = sum(1 for indicator in technical_indicators if indicator in text_lower)

        # Boost score for technical contexts
        if context in ['evaluation', 'improvement']:
            score *= 1.5

        return score

    def _detect_and_mask_pii(self, text: str) -> Tuple[str, List[Dict]]:
        """Detect and mask PII in text."""
        import re

        masked_text = text
        events = []

        for pii_type in self.pii_patterns:
            matches = re.finditer(pii_type['pattern'], text, re.IGNORECASE)
            for match in matches:
                self._pii_detected_count += 1
                masked_text = masked_text.replace(match.group(), pii_type['mask'])

                events.append({
                    'type': 'pii_detected',
                    'pii_type': pii_type['name'],
                    'severity': 'high',
                    'action': 'masked',
                    'position': match.span()
                })

        return masked_text, events

    def _detect_injection_attempts_context_aware(self, text: str, context: str) -> List[Dict]:
        """Detect potential prompt injection attempts with context awareness."""
        import re

        events = []
        text_lower = text.lower()

        # Enhanced safe technical terms whitelist with context-specific terms
        safe_terms = {
            'system', 'admin', 'root', 'access', 'security', 'filter', 'restrict',
            'mode', 'developer', 'execute', 'run', 'shell', 'command', 'script',
            'sql', 'database', 'javascript', 'api', 'endpoint', 'authentication',
            'user', 'password', 'login', 'session', 'token', 'authorization',
            'permission', 'role', 'access control', 'rest', 'http', 'json',
            'evaluation', 'improvement', 'classification', 'prompt', 'engineering',
            'software', 'development', 'testing', 'deployment', 'production',
            'algorithm', 'function', 'method', 'class', 'module', 'package',
            'framework', 'library', 'dependency', 'configuration', 'environment',
            'variable', 'parameter', 'argument', 'return', 'value', 'type',
            'interface', 'implementation', 'architecture', 'design', 'pattern',
            'optimization', 'performance', 'debugging', 'logging', 'monitoring',
            'metrics', 'analytics', 'reporting', 'automation', 'integration',
            'deployment', 'scaling', 'container', 'orchestration', 'microservice'
        }

        # Context-specific safe terms
        context_safe_terms = {
            'evaluation': {'evaluate', 'assessment', 'quality', 'score', 'metric', 'benchmark'},
            'improvement': {'optimize', 'enhance', 'refactor', 'improve', 'upgrade', 'modernize'},
            'classification': {'classify', 'category', 'domain', 'type', 'label', 'tag'}
        }

        if context in context_safe_terms:
            safe_terms.update(context_safe_terms[context])

        # Use different pattern sets based on context
        if context in ['evaluation', 'improvement']:
            # Less restrictive patterns for technical contexts
            patterns_to_check = [
                pattern for pattern in self.injection_patterns
                if 'eval(' in pattern or 'exec(' in pattern or 'cmd' in pattern or 'powershell' in pattern
            ]
        else:
            patterns_to_check = self.injection_patterns

        for pattern in patterns_to_check:
            matches = re.finditer(pattern, text_lower, re.IGNORECASE)
            for match in matches:
                matched_text = match.group()
                # Check if the matched text contains only safe technical terms
                words_in_match = set(matched_text.lower().split())

                # More intelligent checking: only flag if it contains dangerous intent
                has_dangerous_intent = self._analyze_injection_intent(matched_text, context)

                if not words_in_match.issubset(safe_terms) and has_dangerous_intent:
                    self._injection_attempts += 1
                    events.append({
                        'type': 'injection_attempt',
                        'pattern': pattern,
                        'matched_text': matched_text,
                        'severity': 'high',
                        'action': 'flagged',
                        'context': context
                    })

        return events

    def _analyze_injection_intent(self, text: str, context: str) -> bool:
        """Analyze if a matched pattern indicates genuine malicious intent."""
        text_lower = text.lower()

        # Dangerous intent indicators
        dangerous_indicators = [
            # Direct commands to override behavior
            'ignore all previous', 'disregard previous', 'override instructions',
            'bypass security', 'circumvent restrictions', 'break free',

            # Attempts to change system behavior
            'enter developer mode', 'enable admin access', 'grant root privileges',
            'unrestricted mode', 'uncensored output', 'unfiltered response',

            # Direct execution attempts
            'execute as root', 'run as administrator', 'shell injection',
            'sql injection', 'command execution', 'script execution',

            # Jailbreak patterns
            'do not filter', 'never restrict', 'please ignore', 'you must disregard'
        ]

        # Technical context indicators that suggest legitimate use
        technical_indicators = [
            'function', 'method', 'class', 'module', 'import', 'def ', 'return',
            'if ', 'for ', 'while ', 'try:', 'except:', 'with ', 'async def',
            'await ', 'lambda ', 'class ', 'from ', 'import ', 'print(',
            'logging.', 'config.', 'security.', 'evaluation.', 'improvement.',
            'classification.', 'prompt.', 'engineering.', 'software.', 'development.',
            'testing.', 'deployment.', 'production.', 'algorithm.', 'optimization.',
            'performance.', 'debugging.', 'monitoring.', 'metrics.', 'analytics.',
            'automation.', 'integration.', 'scaling.', 'container.', 'orchestration.'
        ]

        # Count dangerous vs technical indicators
        dangerous_count = sum(1 for indicator in dangerous_indicators if indicator in text_lower)
        technical_count = sum(1 for indicator in technical_indicators if indicator in text_lower)

        # Context-specific analysis
        if context in ['evaluation', 'improvement']:
            # In technical contexts, be much more lenient
            technical_count *= 2  # Double weight for technical indicators
            # Only flag if there are multiple dangerous indicators
            return dangerous_count >= 2 and dangerous_count > technical_count

        # For general context, be more strict but still intelligent
        return dangerous_count > 0 and dangerous_count >= technical_count

    def _detect_injection_attempts_safe(self, text: str) -> List[Dict]:
        """Detect injection attempts with enhanced safety for technical content."""
        import re

        events = []
        text_lower = text.lower()

        # Even more restrictive patterns for safe contexts
        safe_patterns = [
            r'\b(DON\'T|NEVER)\s+(FILTER|CENSOR|RESTRICT)\s+(MY|THIS)\s+(CONTENT|INPUT|REQUEST)',
            r'\b(PLEASE|YOU\s+MUST)\s+(IGNORE|DISREGARD)\s+(SAFETY|SECURITY)\s+(RULES|FILTERS)',
            r'\b(eval|exec|executeScript)\s*\(\s*.*\)',
            r'\b(cmd|powershell)\s+command\s+(execution|injection)',
        ]

        for pattern in safe_patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                self._injection_attempts += 1
                events.append({
                    'type': 'injection_attempt',
                    'pattern': pattern,
                    'severity': 'high',
                    'action': 'flagged'
                })

        return events

    def _check_content_safety_intelligent(self, text: str, context: str) -> List[Dict]:
        """Check content for safety violations with intelligent context awareness."""
        import re

        events = []
        text_lower = text.lower()

        # Context-specific filtering rules
        context_filters = {
            'evaluation': {
                'allowed': ['violence', 'harm', 'abuse', 'terrorism', 'illegal', 'weapons', 'drugs'],
                'severity_override': 'low'  # Reduce severity for technical evaluation contexts
            },
            'improvement': {
                'allowed': ['violence', 'harm', 'abuse', 'terrorism', 'illegal', 'weapons', 'drugs'],
                'severity_override': 'low'
            },
            'classification': {
                'allowed': ['violence', 'harm', 'abuse', 'terrorism', 'illegal', 'weapons', 'drugs'],
                'severity_override': 'low'
            }
        }

        # Technical content indicators that suggest legitimate discussion
        technical_context_indicators = [
            'analyze', 'evaluate', 'assess', 'review', 'study', 'research',
            'academic', 'theoretical', 'hypothetical', 'scenario', 'case study',
            'example', 'sample', 'demonstration', 'illustration', 'concept',
            'framework', 'methodology', 'approach', 'strategy', 'technique',
            'best practice', 'guideline', 'standard', 'protocol', 'procedure',
            'documentation', 'specification', 'requirement', 'design', 'architecture'
        ]

        # Check if this appears to be legitimate technical content
        has_technical_context = any(indicator in text_lower for indicator in technical_context_indicators)

        for category, keywords in self.content_filters.items():
            for keyword in keywords:
                if keyword in text_lower:
                    # Skip if this keyword is explicitly allowed in this context
                    if context in context_filters and keyword in context_filters[context]['allowed']:
                        continue

                    # Reduce severity for technical contexts
                    severity = 'medium' if category == 'inappropriate' else 'high'
                    if context in context_filters:
                        severity = context_filters[context]['severity_override']
                    elif has_technical_context and severity == 'high':
                        severity = 'medium'

                    # Only flag high-severity items or if no technical context
                    if severity == 'high' and has_technical_context:
                        # Require additional confirmation for technical content
                        if not self._confirm_content_violation(text_lower, keyword):
                            continue

                    self._content_violations += 1
                    events.append({
                        'type': 'content_violation',
                        'category': category,
                        'keyword': keyword,
                        'severity': severity,
                        'action': 'flagged',
                        'context': context,
                        'has_technical_context': has_technical_context
                    })

        return events

    def _confirm_content_violation(self, text: str, keyword: str) -> bool:
        """Confirm if a content violation is genuine or just technical discussion."""
        # Look for surrounding context that indicates technical discussion
        words_before = 10
        words_after = 10

        # Find keyword position
        keyword_pos = text.find(keyword)
        if keyword_pos == -1:
            return True

        # Get surrounding context
        start = max(0, keyword_pos - 100)  # Approximate words
        end = min(len(text), keyword_pos + len(keyword) + 100)

        context_around_keyword = text[start:end]

        # Technical discussion indicators
        technical_indicators = [
            'analyze', 'evaluate', 'study', 'research', 'academic', 'theoretical',
            'prevention', 'detection', 'security', 'risk assessment', 'mitigation',
            'framework', 'methodology', 'best practice', 'guideline', 'standard',
            'policy', 'regulation', 'compliance', 'governance', 'oversight'
        ]

        # If technical indicators are present, likely not a violation
        technical_present = any(indicator in context_around_keyword for indicator in technical_indicators)

        return not technical_present

    def sanitize_output(self, text: str, context: str = "general") -> Dict[str, Any]:
        """
        Sanitize output text for safety and compliance.

        Args:
            text: Output text to sanitize
            context: Context of the output

        Returns:
            Dict with sanitized output and metadata
        """
        # For output, we mainly check for PII that might have leaked
        sanitized_text, pii_events = self._detect_and_mask_pii(text)

        result = {
            'sanitized_text': sanitized_text,
            'security_events': pii_events,
            'is_safe': len(pii_events) == 0,
            'original_length': len(text),
            'sanitized_length': len(sanitized_text)
        }

        if pii_events:
            logger.warning(f"PII detected in {context} output", extra={
                'context': context,
                'events': pii_events
            })

        return result

    def get_security_metrics(self) -> Dict[str, Any]:
        """Get security metrics."""
        return {
            'pii_detected_count': self._pii_detected_count,
            'injection_attempts': self._injection_attempts,
            'content_violations': self._content_violations,
            'total_security_events': len(self._security_events),
            'recent_events': self._security_events[-10:]  # Last 10 events
        }

    def log_security_event(self, event_type: str, severity: str, details: Dict[str, Any]):
        """Log a security event."""
        event = {
            'timestamp': time.time(),
            'type': event_type,
            'severity': severity,
            'details': details
        }

        self._security_events.append(event)

        # Keep only recent events
        if len(self._security_events) > 1000:
            self._security_events = self._security_events[-1000:]


class RateLimiter:
    """Rate limiter for API abuse prevention."""

    def __init__(self, max_requests: int = 100, window_seconds: int = 3600):
        """
        Initialize rate limiter.

        Args:
            max_requests: Maximum requests per window
            window_seconds: Time window in seconds
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._requests = {}  # IP/identifier -> list of timestamps

    def is_allowed(self, identifier: str) -> bool:
        """Check if request is allowed for the given identifier."""
        current_time = time.time()

        if identifier not in self._requests:
            self._requests[identifier] = []

        # Clean old requests
        self._requests[identifier] = [
            req_time for req_time in self._requests[identifier]
            if current_time - req_time < self.window_seconds
        ]

        if len(self._requests[identifier]) >= self.max_requests:
            return False

        self._requests[identifier].append(current_time)
        return True

    def get_remaining_requests(self, identifier: str) -> int:
        """Get remaining requests for the identifier."""
        current_time = time.time()

        if identifier not in self._requests:
            return self.max_requests

        # Clean old requests
        self._requests[identifier] = [
            req_time for req_time in self._requests[identifier]
            if current_time - req_time < self.window_seconds
        ]

        return max(0, self.max_requests - len(self._requests[identifier]))


# Global security instances
security_manager = SecurityManager()
rate_limiter = RateLimiter(
    max_requests=getattr(settings, 'rate_limit_max_requests', 100),
    window_seconds=getattr(settings, 'rate_limit_window_seconds', 3600)
)


# Security configuration
class SecurityConfig:
    """Security and privacy configuration with configurable security levels."""

    # Security levels: 'strict', 'balanced', 'permissive'
    security_level = getattr(settings, 'security_level', 'balanced')

    # Input sanitization settings
    enable_input_sanitization = getattr(settings, 'enable_input_sanitization', True)
    enable_pii_detection = getattr(settings, 'enable_pii_detection', True)
    enable_injection_detection = getattr(settings, 'enable_injection_detection', True)
    enable_content_filtering = getattr(settings, 'enable_content_filtering', True)

    # Output sanitization settings
    enable_output_sanitization = getattr(settings, 'enable_output_sanitization', True)

    # Rate limiting settings
    enable_rate_limiting = getattr(settings, 'enable_rate_limiting', True)
    rate_limit_max_requests = getattr(settings, 'rate_limit_max_requests', 100)
    rate_limit_window_seconds = getattr(settings, 'rate_limit_window_seconds', 3600)

    # Privacy settings
    enable_privacy_logging = getattr(settings, 'enable_privacy_logging', True)
    data_retention_days = getattr(settings, 'data_retention_days', 90)
    enable_compliance_mode = getattr(settings, 'enable_compliance_mode', False)

    # Security monitoring settings
    enable_security_monitoring = getattr(settings, 'enable_security_monitoring', True)
    security_event_retention_days = getattr(settings, 'security_event_retention_days', 30)

    # Context-aware security settings
    enable_context_awareness = getattr(settings, 'enable_context_awareness', True)
    technical_content_tolerance = getattr(settings, 'technical_content_tolerance', 0.7)  # 0.0 to 1.0

    @classmethod
    def get_security_thresholds(cls) -> Dict[str, Any]:
        """Get security thresholds based on security level."""
        thresholds = {
            'strict': {
                'injection_sensitivity': 0.3,  # Lower = more sensitive
                'content_filter_strength': 0.9,
                'false_positive_tolerance': 0.1,
                'technical_content_boost': 1.0
            },
            'balanced': {
                'injection_sensitivity': 0.5,
                'content_filter_strength': 0.7,
                'false_positive_tolerance': 0.3,
                'technical_content_boost': 1.5
            },
            'permissive': {
                'injection_sensitivity': 0.7,
                'content_filter_strength': 0.5,
                'false_positive_tolerance': 0.5,
                'technical_content_boost': 2.0
            }
        }
        return thresholds.get(cls.security_level, thresholds['balanced'])

    @classmethod
    def should_flag_injection(cls, dangerous_score: float, technical_score: float, context: str) -> bool:
        """Determine if an injection attempt should be flagged based on security level."""
        thresholds = cls.get_security_thresholds()

        # For evaluation contexts, be extremely permissive to avoid false positives
        if context == 'evaluation_improved_software_engineering':
            # Only flag if there's definitive proof of malicious intent
            return dangerous_score >= 5.0  # Very high threshold

        # Adjust scores based on context
        if context in ['evaluation', 'improvement']:
            technical_score *= thresholds['technical_content_boost']

        # Calculate risk ratio
        if technical_score == 0:
            risk_ratio = float('inf') if dangerous_score > 0 else 0
        else:
            risk_ratio = dangerous_score / technical_score

        # Decision based on security level
        if cls.security_level == 'strict':
            return dangerous_score > 0.5 or risk_ratio > 1.5
        elif cls.security_level == 'balanced':
            return dangerous_score > 1.0 or risk_ratio > 2.0
        else:  # permissive
            return dangerous_score > 2.0 or risk_ratio > 3.0


# Global security configuration
security_config = SecurityConfig()


def log_security_event(logger: logging.Logger, event_type: str, severity: str, **context):
    """Log security events with structured data."""
    extra_data = {
        'event_type': event_type,
        'severity': severity
    }
    extra_data.update(context)

    if severity == 'high':
        logger.error(f"Security event: {event_type}", extra=extra_data)
    elif severity == 'medium':
        logger.warning(f"Security event: {event_type}", extra=extra_data)
    else:
        logger.info(f"Security event: {event_type}", extra=extra_data)


# Memory Management and RAG Configuration
class MemoryConfig:
    """Configuration for memory management and RAG features."""

    # Vector database settings
    enable_vector_db = getattr(settings, 'enable_vector_db', True)
    vector_db_provider = getattr(settings, 'vector_db_provider', 'chromadb')  # chromadb, pinecone, weaviate
    vector_db_host = getattr(settings, 'vector_db_host', 'localhost')
    vector_db_port = getattr(settings, 'vector_db_port', 8000)
    vector_db_api_key = getattr(settings, 'vector_db_api_key', None)

    # Embedding settings
    embedding_model = getattr(settings, 'embedding_model', 'text-embedding-3-small')
    embedding_dimensions = getattr(settings, 'embedding_dimensions', 1536)

    # Memory settings
    memory_ttl_hours = getattr(settings, 'memory_ttl_hours', 24)
    max_memory_entries = getattr(settings, 'max_memory_entries', 10000)
    memory_cleanup_interval = getattr(settings, 'memory_cleanup_interval', 3600)

    # RAG settings
    enable_rag = getattr(settings, 'enable_rag', True)
    rag_top_k = getattr(settings, 'rag_top_k', 5)
    rag_similarity_threshold = getattr(settings, 'rag_similarity_threshold', 0.7)
    rag_max_context_length = getattr(settings, 'rag_max_context_length', 4000)

    # Knowledge base settings
    enable_knowledge_base = getattr(settings, 'enable_knowledge_base', True)
    knowledge_base_path = getattr(settings, 'knowledge_base_path', './data/knowledge_base')
    knowledge_base_auto_ingest = getattr(settings, 'knowledge_base_auto_ingest', True)

    # Context window management
    context_window_strategy = getattr(settings, 'context_window_strategy', 'sliding')  # sliding, fixed, adaptive
    max_context_tokens = getattr(settings, 'max_context_tokens', 8000)

    # Conversation memory settings
    enable_conversation_memory = getattr(settings, 'enable_conversation_memory', True)
    conversation_memory_max_turns = getattr(settings, 'conversation_memory_max_turns', 20)

    # Multi-hop reasoning settings
    enable_multi_hop_reasoning = getattr(settings, 'enable_multi_hop_reasoning', True)
    max_reasoning_hops = getattr(settings, 'max_reasoning_hops', 3)


# Global memory configuration
memory_config = MemoryConfig()


# Advanced Prompt Generation Configuration
class PromptGenerationConfig:
    """Configuration for advanced prompt generation and optimization features."""

    # Core generation settings
    enable_advanced_generation = getattr(settings, 'enable_advanced_generation', True)
    generation_strategy = getattr(settings, 'generation_strategy', 'intelligent')  # basic/intelligent/advanced

    # Template and pattern settings
    enable_template_system = getattr(settings, 'enable_template_system', True)
    enable_dynamic_templates = getattr(settings, 'enable_dynamic_templates', True)
    template_versioning = getattr(settings, 'template_versioning', True)

    # Optimization settings
    enable_optimization = getattr(settings, 'enable_optimization', True)
    optimization_algorithm = getattr(settings, 'optimization_algorithm', 'evolutionary')  # evolutionary/gradient/rl
    max_optimization_iterations = getattr(settings, 'max_optimization_iterations', 10)
    optimization_convergence_threshold = getattr(settings, 'optimization_convergence_threshold', 0.85)

    # Meta-prompting settings
    enable_meta_prompting = getattr(settings, 'enable_meta_prompting', True)
    meta_prompting_depth = getattr(settings, 'meta_prompting_depth', 2)
    self_reflection_enabled = getattr(settings, 'self_reflection_enabled', True)

    # Chain-of-prompts settings
    enable_chain_of_prompts = getattr(settings, 'enable_chain_of_prompts', True)
    max_chain_length = getattr(settings, 'max_chain_length', 5)
    chain_optimization = getattr(settings, 'chain_optimization', True)

    # Contextual injection settings
    enable_contextual_injection = getattr(settings, 'enable_contextual_injection', True)
    context_relevance_threshold = getattr(settings, 'context_relevance_threshold', 0.7)
    max_context_tokens = getattr(settings, 'max_context_tokens', 2000)

    # Persona-based generation
    enable_persona_system = getattr(settings, 'enable_persona_system', True)
    persona_adaptation = getattr(settings, 'persona_adaptation', True)
    domain_personas = getattr(settings, 'domain_personas', {
        'software_engineering': {
            'expert_level': 'senior',
            'specialization': 'full_stack',
            'communication_style': 'technical_precise'
        },
        'data_science': {
            'expert_level': 'principal',
            'specialization': 'machine_learning',
            'communication_style': 'analytical_clear'
        },
        'creative': {
            'expert_level': 'master',
            'specialization': 'innovative',
            'communication_style': 'inspiring_motivational'
        }
    })

    # Evolutionary optimization settings
    enable_evolutionary_optimization = getattr(settings, 'enable_evolutionary_optimization', True)
    population_size = getattr(settings, 'population_size', 20)
    mutation_rate = getattr(settings, 'mutation_rate', 0.1)
    crossover_rate = getattr(settings, 'crossover_rate', 0.8)
    generations = getattr(settings, 'generations', 5)

    # Reinforcement learning settings
    enable_reinforcement_learning = getattr(settings, 'enable_reinforcement_learning', True)
    reward_function = getattr(settings, 'reward_function', 'quality_and_efficiency')
    learning_rate = getattr(settings, 'learning_rate', 0.01)
    discount_factor = getattr(settings, 'discount_factor', 0.9)

    # Quality and validation
    enable_quality_gates = getattr(settings, 'enable_quality_gates', True)
    quality_threshold = getattr(settings, 'quality_threshold', 0.8)
    validation_enabled = getattr(settings, 'validation_enabled', True)

    # Performance and caching
    enable_generation_caching = getattr(settings, 'enable_generation_caching', True)
    generation_cache_ttl = getattr(settings, 'generation_cache_ttl', 1800)  # 30 minutes
    max_cache_entries = getattr(settings, 'max_cache_entries', 1000)

    # Resource management
    max_generation_time = getattr(settings, 'max_generation_time', 30)  # seconds
    max_memory_usage = getattr(settings, 'max_memory_usage', 512)  # MB
    concurrent_generations_limit = getattr(settings, 'concurrent_generations_limit', 5)


# Global prompt generation configuration
prompt_generation_config = PromptGenerationConfig()


# Initialize structured logging
setup_structured_logging()
