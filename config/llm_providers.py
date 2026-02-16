"""
Multi-LLM Provider System for CortexaAI.

Provides a unified interface to multiple LLM providers with:
- Automatic fallback routing
- Smart model selection based on task complexity
- Provider health monitoring
- Cost-aware routing
"""

from typing import Optional, Dict, Any, List
from langchain_core.language_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from config.config import settings, get_logger

logger = get_logger(__name__)


# Provider registry: maps provider name -> initialization function
_PROVIDER_REGISTRY: Dict[str, Dict[str, Any]] = {}


def _init_google(model_name: str, api_key: str, temperature: float = 0.1, **kwargs) -> BaseChatModel:
    """Initialize Google Gemini provider."""
    return ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=api_key,
        temperature=temperature,
        **kwargs,
    )


def _init_openai(model_name: str, api_key: str, temperature: float = 0.1, **kwargs) -> BaseChatModel:
    """Initialize OpenAI provider."""
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        model=model_name,
        api_key=api_key,
        temperature=temperature,
        **kwargs,
    )


def _init_anthropic(model_name: str, api_key: str, temperature: float = 0.1, **kwargs) -> BaseChatModel:
    """Initialize Anthropic provider."""
    from langchain_anthropic import ChatAnthropic
    return ChatAnthropic(
        model=model_name,
        api_key=api_key,
        temperature=temperature,
        **kwargs,
    )


def _init_groq(model_name: str, api_key: str, temperature: float = 0.1, **kwargs) -> BaseChatModel:
    """Initialize Groq provider (ultra-fast inference)."""
    from langchain_groq import ChatGroq
    return ChatGroq(
        model=model_name,
        api_key=api_key,
        temperature=temperature,
        **kwargs,
    )


def _init_deepseek(model_name: str, api_key: str, temperature: float = 0.1, **kwargs) -> BaseChatModel:
    """Initialize DeepSeek provider via OpenAI-compatible endpoint."""
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        model=model_name,
        api_key=api_key,
        base_url="https://api.deepseek.com/v1",
        temperature=temperature,
        **kwargs,
    )


def _init_openrouter(model_name: str, api_key: str, temperature: float = 0.1, **kwargs) -> BaseChatModel:
    """Initialize OpenRouter provider (100+ models)."""
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        model=model_name,
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        temperature=temperature,
        default_headers={
            "HTTP-Referer": "https://github.com/Abood991B/CortexaAI",
            "X-Title": "CortexaAI",
        },
        **kwargs,
    )


# Provider definitions
PROVIDER_CONFIGS: Dict[str, Dict[str, Any]] = {
    "google": {
        "init_fn": _init_google,
        "default_model": "gemma-3-27b-it",
        "api_key_env": "google_api_key",
        "free_tier": True,
        "models": {
            "gemma-3-27b-it": {"context_window": 8192, "cost_per_1k_tokens": 0.0},
            "gemma-3-12b-it": {"context_window": 8192, "cost_per_1k_tokens": 0.0},
            "gemma-3-4b-it": {"context_window": 8192, "cost_per_1k_tokens": 0.0},
            "gemini-2.0-flash": {"context_window": 1048576, "cost_per_1k_tokens": 0.0},
            "gemini-2.5-flash-lite": {"context_window": 1048576, "cost_per_1k_tokens": 0.0},
        },
    },
    "openai": {
        "init_fn": _init_openai,
        "default_model": "gpt-4o-mini",
        "api_key_env": "openai_api_key",
        "free_tier": False,
        "models": {
            "gpt-4o-mini": {"context_window": 128000, "cost_per_1k_tokens": 0.15},
            "gpt-4o": {"context_window": 128000, "cost_per_1k_tokens": 2.50},
            "gpt-4-turbo": {"context_window": 128000, "cost_per_1k_tokens": 10.0},
        },
    },
    "anthropic": {
        "init_fn": _init_anthropic,
        "default_model": "claude-3-haiku-20240307",
        "api_key_env": "anthropic_api_key",
        "free_tier": False,
        "models": {
            "claude-3-haiku-20240307": {"context_window": 200000, "cost_per_1k_tokens": 0.25},
            "claude-3-sonnet-20240229": {"context_window": 200000, "cost_per_1k_tokens": 3.0},
        },
    },
    "groq": {
        "init_fn": _init_groq,
        "default_model": "llama-3.3-70b-versatile",
        "api_key_env": "groq_api_key",
        "free_tier": True,
        "models": {
            "llama-3.3-70b-versatile": {"context_window": 128000, "cost_per_1k_tokens": 0.0},
            "mixtral-8x7b-32768": {"context_window": 32768, "cost_per_1k_tokens": 0.0},
            "gemma2-9b-it": {"context_window": 8192, "cost_per_1k_tokens": 0.0},
        },
    },
    "deepseek": {
        "init_fn": _init_deepseek,
        "default_model": "deepseek-chat",
        "api_key_env": "deepseek_api_key",
        "free_tier": False,
        "models": {
            "deepseek-chat": {"context_window": 64000, "cost_per_1k_tokens": 0.14},
            "deepseek-reasoner": {"context_window": 64000, "cost_per_1k_tokens": 0.55},
        },
    },
    "openrouter": {
        "init_fn": _init_openrouter,
        "default_model": "google/gemini-2.0-flash-exp:free",
        "api_key_env": "openrouter_api_key",
        "free_tier": True,
        "models": {
            "google/gemini-2.0-flash-exp:free": {"context_window": 1048576, "cost_per_1k_tokens": 0.0},
            "meta-llama/llama-3.3-70b-instruct:free": {"context_window": 128000, "cost_per_1k_tokens": 0.0},
            "qwen/qwen-2.5-72b-instruct:free": {"context_window": 32768, "cost_per_1k_tokens": 0.0},
        },
    },
}

# Provider priority for fallback (free tiers first)
FALLBACK_ORDER = ["google", "groq", "openrouter", "deepseek", "openai", "anthropic"]


class LLMProvider:
    """
    Unified LLM provider with automatic fallback, health monitoring,
    and smart routing capabilities.
    """

    def __init__(self):
        self._instances: Dict[str, BaseChatModel] = {}
        self._health: Dict[str, bool] = {}
        self._call_counts: Dict[str, int] = {}
        self._error_counts: Dict[str, int] = {}

    def get_available_providers(self) -> List[str]:
        """Return list of providers that have API keys configured."""
        available = []
        for provider_name, config in PROVIDER_CONFIGS.items():
            api_key = getattr(settings, config["api_key_env"], None)
            if api_key:
                available.append(provider_name)
        return available

    def get_model(
        self,
        provider: Optional[str] = None,
        model_name: Optional[str] = None,
        temperature: float = 0.1,
        **kwargs,
    ) -> BaseChatModel:
        """
        Get an LLM instance. Falls back through providers if the primary is unavailable.

        Args:
            provider: Provider name (uses default if None)
            model_name: Model name (uses provider default if None)
            temperature: Sampling temperature
            **kwargs: Additional provider-specific arguments

        Returns:
            BaseChatModel instance ready for use

        Raises:
            RuntimeError: If no provider is available
        """
        provider = provider or settings.default_model_provider
        model_name = model_name or settings.default_model_name

        # Try primary provider first
        try:
            return self._create_or_get(provider, model_name, temperature, **kwargs)
        except Exception as e:
            logger.warning(f"Primary provider '{provider}' failed: {e}")
            self._health[provider] = False
            self._error_counts[provider] = self._error_counts.get(provider, 0) + 1

        # Fallback through other providers
        for fallback_provider in FALLBACK_ORDER:
            if fallback_provider == provider:
                continue
            api_key = getattr(settings, PROVIDER_CONFIGS[fallback_provider]["api_key_env"], None)
            if not api_key:
                continue

            try:
                fallback_model = PROVIDER_CONFIGS[fallback_provider]["default_model"]
                logger.info(f"Falling back to '{fallback_provider}' with model '{fallback_model}'")
                return self._create_or_get(
                    fallback_provider, fallback_model, temperature, **kwargs
                )
            except Exception as e:
                logger.warning(f"Fallback provider '{fallback_provider}' failed: {e}")
                self._health[fallback_provider] = False
                continue

        raise RuntimeError(
            "No LLM provider available. Please configure at least one API key in .env. "
            "See .env.example for setup instructions."
        )

    def _create_or_get(
        self,
        provider: str,
        model_name: str,
        temperature: float,
        **kwargs,
    ) -> BaseChatModel:
        """Create or retrieve a cached LLM instance."""
        cache_key = f"{provider}:{model_name}:{temperature}"

        if cache_key in self._instances:
            return self._instances[cache_key]

        config = PROVIDER_CONFIGS.get(provider)
        if not config:
            raise ValueError(f"Unknown provider: {provider}")

        api_key = getattr(settings, config["api_key_env"], None)
        if not api_key:
            raise ValueError(f"API key not configured for provider: {provider}")

        instance = config["init_fn"](model_name, api_key, temperature, **kwargs)
        self._instances[cache_key] = instance
        self._health[provider] = True
        self._call_counts[provider] = self._call_counts.get(provider, 0) + 1

        logger.info(f"Initialized LLM: {provider}/{model_name} (temp={temperature})")
        return instance

    def get_provider_status(self) -> Dict[str, Any]:
        """Get status of all configured providers."""
        status = {}
        for provider_name, config in PROVIDER_CONFIGS.items():
            api_key = getattr(settings, config["api_key_env"], None)
            status[provider_name] = {
                "configured": bool(api_key),
                "healthy": self._health.get(provider_name, True) if api_key else False,
                "call_count": self._call_counts.get(provider_name, 0),
                "error_count": self._error_counts.get(provider_name, 0),
                "free_tier": config["free_tier"],
                "default_model": config["default_model"],
                "available_models": list(config["models"].keys()),
            }
        return status

    def reset_health(self, provider: Optional[str] = None):
        """Reset health status for providers."""
        if provider:
            self._health[provider] = True
        else:
            self._health = {p: True for p in PROVIDER_CONFIGS}

    async def verify_provider(self, provider_name: str) -> Dict[str, Any]:
        """
        Verify a provider by making a lightweight API call.
        Returns dict with 'available', 'latency_ms', and optional 'error'.
        """
        import asyncio, time as _time

        config = PROVIDER_CONFIGS.get(provider_name)
        if not config:
            return {"available": False, "error": "Unknown provider"}

        api_key = getattr(settings, config["api_key_env"], None)
        if not api_key:
            return {"available": False, "error": "No API key configured"}

        # Use the user-configured model for the default provider
        if provider_name == settings.default_model_provider:
            model_name = settings.default_model_name
        else:
            model_name = config["default_model"]

        try:
            model = self._create_or_get(
                provider_name, model_name, 0.0
            )
            start = _time.perf_counter()
            # Use a trivial invoke to validate connectivity
            resp = await model.ainvoke("Say OK")
            latency = round((_time.perf_counter() - start) * 1000)
            self._health[provider_name] = True
            return {"available": True, "latency_ms": latency, "model": model_name}
        except Exception as exc:
            self._health[provider_name] = False
            self._error_counts[provider_name] = self._error_counts.get(provider_name, 0) + 1
            return {"available": False, "error": str(exc)[:120]}


# Global singleton
llm_provider = LLMProvider()


def get_llm(
    provider: Optional[str] = None,
    model_name: Optional[str] = None,
    temperature: float = 0.1,
    **kwargs,
) -> BaseChatModel:
    """
    Convenience function to get an LLM instance.

    Usage:
        llm = get_llm()  # Uses default provider
        llm = get_llm("groq", "llama-3.3-70b-versatile")
        llm = get_llm("openrouter", "meta-llama/llama-3.3-70b-instruct:free")
    """
    return llm_provider.get_model(provider, model_name, temperature, **kwargs)
