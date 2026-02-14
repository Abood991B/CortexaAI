"""Shared pytest fixtures for the CortexaAI test suite."""

import os
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


# ─── Environment Fixtures ───────────────────────────────────────

@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Ensure tests never use real API keys."""
    for key in [
        "GOOGLE_API_KEY",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GROQ_API_KEY",
        "DEEPSEEK_API_KEY",
        "OPENROUTER_API_KEY",
    ]:
        monkeypatch.delenv(key, raising=False)


@pytest.fixture
def fake_google_key(monkeypatch):
    """Provide a fake Google API key."""
    monkeypatch.setenv("GOOGLE_API_KEY", "test-google-key-12345")
    return "test-google-key-12345"


@pytest.fixture
def fake_groq_key(monkeypatch):
    """Provide a fake Groq API key."""
    monkeypatch.setenv("GROQ_API_KEY", "test-groq-key-12345")
    return "test-groq-key-12345"


@pytest.fixture
def fake_all_keys(monkeypatch):
    """Provide fake keys for all providers."""
    keys = {
        "GOOGLE_API_KEY": "test-google-key",
        "OPENAI_API_KEY": "test-openai-key",
        "ANTHROPIC_API_KEY": "test-anthropic-key",
        "GROQ_API_KEY": "test-groq-key",
        "DEEPSEEK_API_KEY": "test-deepseek-key",
        "OPENROUTER_API_KEY": "test-openrouter-key",
    }
    for k, v in keys.items():
        monkeypatch.setenv(k, v)
    return keys


# ─── Mock Agent Fixtures ────────────────────────────────────────

@pytest.fixture
def mock_evaluator():
    """Create a mock evaluator agent."""
    evaluator = AsyncMock()
    evaluator.evaluate_prompt = AsyncMock(return_value={
        "overall_score": 0.85,
        "criteria_scores": {
            "clarity": 0.9,
            "specificity": 0.8,
            "actionability": 0.85,
            "completeness": 0.8,
            "domain_relevance": 0.9,
            "constraint_quality": 0.85,
        },
        "passes_threshold": True,
        "strengths": ["Well-structured", "Domain-specific"],
        "weaknesses": [],
        "key_topics": ["sorting", "algorithms"],
    })
    return evaluator


@pytest.fixture
def mock_expert():
    """Create a mock expert agent."""
    expert = AsyncMock()
    expert.improve_prompt = AsyncMock(return_value={
        "improved_prompt": "Optimized: Write a well-documented, O(n log n) sorting function",
        "solution": "Optimized: Write a well-documented, O(n log n) sorting function",
    })
    return expert


@pytest.fixture
def mock_classifier():
    """Create a mock classifier agent."""
    classifier = AsyncMock()
    classifier.classify = AsyncMock(return_value={
        "domain": "software_engineering",
        "confidence": 0.95,
        "reasoning": "Technical programming task",
    })
    return classifier


# ─── Sample Data Fixtures ───────────────────────────────────────

@pytest.fixture
def sample_prompt():
    """A sample prompt for testing."""
    return "Write a Python function that sorts a list of integers"


@pytest.fixture
def sample_evaluation():
    """A sample evaluation result."""
    return {
        "overall_score": 0.75,
        "criteria_scores": {
            "clarity": 0.8,
            "specificity": 0.7,
            "actionability": 0.75,
            "completeness": 0.7,
            "domain_relevance": 0.8,
            "constraint_quality": 0.75,
        },
        "passes_threshold": False,
        "strengths": ["Clear intent"],
        "weaknesses": ["Lacks constraints", "No output format specified"],
        "key_topics": ["python", "sorting"],
    }
