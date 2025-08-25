"""Configuration management for Multi-Agent Prompt Engineering System."""

import os
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # API Keys
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    anthropic_api_key: Optional[str] = Field(default=None, env="ANTHROPIC_API_KEY")
    google_api_key: Optional[str] = Field(default=None, env="GOOGLE_API_KEY")

    # LangSmith Configuration
    langsmith_api_key: Optional[str] = Field(default=None, env="LANGSMITH_API_KEY")
    langsmith_project: str = Field(default="prompt-engineering-system", env="LANGSMITH_PROJECT")
    langsmith_endpoint: str = Field(default="https://api.smith.langchain.com", env="LANGSMITH_ENDPOINT")

    # Model Configuration
    default_model_provider: str = Field(default="openai", env="DEFAULT_MODEL_PROVIDER")
    default_model_name: str = Field(default="gpt-4", env="DEFAULT_MODEL_NAME")

    # System Configuration
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    max_evaluation_iterations: int = Field(default=3, env="MAX_EVALUATION_ITERATIONS")
    evaluation_threshold: float = Field(default=0.8, env="EVALUATION_THRESHOLD")

    # Server Configuration
    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=8000, env="PORT")

    class Config:
        """Pydantic configuration."""
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()


def get_model_config(provider: str = None, model_name: str = None):
    """Get model configuration for the specified provider."""
    provider = provider or settings.default_model_provider
    model_name = model_name or settings.default_model_name

    configs = {
        "openai": {
            "model_name": model_name if provider == "openai" else "gpt-4",
            "api_key": settings.openai_api_key,
        },
        "anthropic": {
            "model_name": model_name if provider == "anthropic" else "claude-3-sonnet-20240229",
            "api_key": settings.anthropic_api_key,
        },
        "google": {
            "model_name": model_name if provider == "google" else "gemini-2.0-flash",
            "api_key": settings.google_api_key,
        }
    }

    return configs.get(provider, configs["openai"])


def setup_langsmith():
    """Set up LangSmith tracing if API key is available."""
    if settings.langsmith_api_key:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = settings.langsmith_api_key
        os.environ["LANGCHAIN_PROJECT"] = settings.langsmith_project
        os.environ["LANGCHAIN_ENDPOINT"] = settings.langsmith_endpoint
        return True
    return False
