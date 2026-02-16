"""Custom exceptions for the Multi-Agent Prompt Engineering System.

This module defines a comprehensive exception hierarchy for the agentic system,
providing detailed error information for better debugging and error handling.

Hierarchy
~~~~~~~~~
AgenticSystemError (base)
├── ClassificationError
├── ImprovementError
├── EvaluationError
├── ConfigurationError
├── LLMServiceError
├── DomainError
├── WorkflowError
└── InputValidationError
"""

from typing import Dict, Any, Optional, List
from datetime import datetime


def _lazy_is_retryable(error: Optional[Exception]) -> bool:
    """Lazy wrapper to avoid importing agents.utils at module scope."""
    if error is None:
        return False
    try:
        from agents.utils import is_retryable_error
        return is_retryable_error(error)
    except ImportError:
        return False

class AgenticSystemError(Exception):
    """Base exception for all agentic system errors.

    Provides detailed error context including error codes, timestamps,
    and additional metadata for comprehensive error tracking.
    """

    def __init__(self, message: str, error_code: str = "UNKNOWN_ERROR",
                 details: Optional[Dict[str, Any]] = None, cause: Optional[Exception] = None,
                 security_events: Optional[List[Dict[str, Any]]] = None):
        """
        Initialize the exception with detailed context.

        Args:
            message: Human-readable error message
            error_code: Machine-readable error code for categorization
            details: Additional context information (agent state, inputs, etc.)
            cause: Original exception that caused this error
            security_events: List of security events associated with the error
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        self.cause = cause
        self.timestamp = datetime.now()
        self.security_events = security_events or []

    def __str__(self) -> str:
        """String representation including error code and timestamp."""
        return f"[{self.error_code}] {self.message} (at {self.timestamp.isoformat()})"

    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for logging/serialization."""
        result = {
            "error_type": self.__class__.__name__,
            "error_code": self.error_code,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
            "details": self.details,
            "cause": str(self.cause) if self.cause else None
        }
        if self.security_events:
            result["security_events"] = self.security_events
        return result

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(code={self.error_code!r}, message={self.message!r})"


class ClassificationError(AgenticSystemError):
    """Raised when prompt classification fails.

    Common causes:
    - LLM API failures during classification
    - Invalid or malformed prompts
    - Unsupported domains
    - Model configuration issues
    """

    def __init__(self, message: str, prompt: Optional[str] = None,
                 confidence: Optional[float] = None, cause: Optional[Exception] = None):
        """Initialize classification error with prompt context."""
        error_code = "CLASSIFICATION_FAILED"
        details = {
            "prompt_length": len(prompt) if prompt else 0,
            "confidence_score": confidence,
            "prompt_preview": prompt[:100] + "..." if prompt and len(prompt) > 100 else prompt
        }
        super().__init__(message, error_code, details, cause)


class ImprovementError(AgenticSystemError):
    """Raised when prompt improvement fails.

    Common causes:
    - LLM API failures during improvement
    - Domain expertise loading failures
    - Template processing errors
    - Resource constraints during improvement
    """

    def __init__(self, message: str, domain: Optional[str] = None,
                 prompt_type: Optional[str] = None, original_prompt: Optional[str] = None,
                 cause: Optional[Exception] = None):
        """Initialize improvement error with domain and prompt context."""
        error_code = "IMPROVEMENT_FAILED"
        details = {
            "domain": domain,
            "prompt_type": prompt_type,
            "original_prompt_length": len(original_prompt) if original_prompt else 0,
            "original_prompt_preview": original_prompt[:100] + "..." if original_prompt and len(original_prompt) > 100 else original_prompt
        }
        super().__init__(message, error_code, details, cause)


class EvaluationError(AgenticSystemError):
    """Raised when prompt evaluation fails.

    Common causes:
    - LLM API failures during evaluation
    - Evaluation criteria parsing errors
    - Threshold validation failures
    - Comparison analysis errors
    """

    def __init__(self, message: str, domain: Optional[str] = None,
                 iteration: Optional[int] = None, threshold: Optional[float] = None,
                 security_events: Optional[List[Dict[str, Any]]] = None,
                 cause: Optional[Exception] = None):
        """Initialize evaluation error with evaluation context."""
        error_code = "EVALUATION_FAILED"
        details = {
            "domain": domain,
            "current_iteration": iteration,
            "evaluation_threshold": threshold,
            "security_events": security_events or []
        }
        super().__init__(message, error_code, details, cause)


class ConfigurationError(AgenticSystemError):
    """Raised when there's a configuration-related error.

    Common causes:
    - Missing or invalid API keys
    - Model configuration issues
    - Invalid environment variables
    - Domain configuration problems
    """

    def __init__(self, message: str, config_key: Optional[str] = None,
                 config_value: Optional[str] = None, cause: Optional[Exception] = None):
        """Initialize configuration error with config context."""
        error_code = "CONFIGURATION_ERROR"
        details = {
            "config_key": config_key,
            "config_value_preview": config_value[:20] + "..." if config_value and len(config_value) > 20 else config_value
        }
        super().__init__(message, error_code, details, cause)


class LLMServiceError(AgenticSystemError):
    """Raised when there's an error communicating with LLM services.

    Common causes:
    - API rate limits
    - Network connectivity issues
    - Authentication failures
    - Model availability issues
    - Token limits exceeded
    """

    def __init__(self, message: str, provider: Optional[str] = None,
                 model: Optional[str] = None, request_type: Optional[str] = None,
                 cause: Optional[Exception] = None):
        """Initialize LLM service error with service context."""
        error_code = "LLM_SERVICE_ERROR"
        details = {
            "provider": provider,
            "model": model,
            "request_type": request_type,
            "is_retryable": _lazy_is_retryable(cause)
        }
        super().__init__(message, error_code, details, cause)


class DomainError(AgenticSystemError):
    """Raised when there's a domain-related error.

    Common causes:
    - Unknown or unsupported domains
    - Domain agent creation failures
    - Domain expertise loading issues
    """

    def __init__(self, message: str, domain: Optional[str] = None,
                 available_domains: Optional[list] = None, cause: Optional[Exception] = None):
        """Initialize domain error with domain context."""
        error_code = "DOMAIN_ERROR"
        details = {
            "requested_domain": domain,
            "available_domains": available_domains or []
        }
        super().__init__(message, error_code, details, cause)


class WorkflowError(AgenticSystemError):
    """Raised when there's a workflow orchestration error.

    Common causes:
    - Agent communication failures
    - Workflow step sequencing issues
    - State management problems
    - Resource exhaustion during workflow
    """

    def __init__(self, message: str, workflow_id: Optional[str] = None,
                 current_step: Optional[str] = None, completed_steps: Optional[list] = None,
                 cause: Optional[Exception] = None):
        """Initialize workflow error with workflow context."""
        error_code = "WORKFLOW_ERROR"
        details = {
            "workflow_id": workflow_id,
            "current_step": current_step,
            "completed_steps": completed_steps or [],
            "workflow_progress": f"{len(completed_steps) if completed_steps else 0} steps completed"
        }
        super().__init__(message, error_code, details, cause)


# Error code constants for consistent error categorization
ERROR_CODES = {
    # Classification errors
    "CLASSIFICATION_FAILED": "Prompt classification could not be completed",
    "DOMAIN_NOT_FOUND": "Requested domain is not supported",

    # Improvement errors
    "IMPROVEMENT_FAILED": "Prompt improvement could not be completed",
    "EXPERT_AGENT_ERROR": "Domain expert agent encountered an error",

    # Evaluation errors
    "EVALUATION_FAILED": "Prompt evaluation could not be completed",
    "THRESHOLD_NOT_MET": "Prompt quality did not meet evaluation threshold",

    # Configuration errors
    "CONFIGURATION_ERROR": "System configuration is invalid or incomplete",
    "API_KEY_MISSING": "Required API key is not configured",

    # LLM service errors
    "LLM_SERVICE_ERROR": "LLM service communication failed",
    "RATE_LIMIT_EXCEEDED": "API rate limit has been exceeded",
    "MODEL_NOT_AVAILABLE": "Requested model is not available",

    # Workflow errors
    "WORKFLOW_ERROR": "Workflow orchestration failed",
    "STEP_TIMEOUT": "Workflow step exceeded timeout limit",

    # General errors
    "UNKNOWN_ERROR": "An unexpected error occurred",
    "VALIDATION_ERROR": "Input validation failed"
}


class InputValidationError(AgenticSystemError):
    """Raised when user-supplied input fails validation.

    Common causes:
    - Empty or whitespace-only prompt
    - Prompt exceeding maximum allowed length
    - Invalid parameter values (e.g. unsupported prompt_type)
    """

    def __init__(self, message: str, field: Optional[str] = None,
                 value: Optional[Any] = None, cause: Optional[Exception] = None):
        error_code = "VALIDATION_ERROR"
        details = {
            "field": field,
            "value_preview": str(value)[:80] if value is not None else None,
        }
        super().__init__(message, error_code, details, cause)
