"""
Prompt Complexity Scoring for CortexaAI.

Pre-processing complexity analysis to route prompts through the appropriate
pipeline depth (simple → single-pass, medium → 2 iterations, complex → 3+).
"""

import re
from typing import Dict, Any, List
from enum import Enum
from config.config import get_logger

logger = get_logger(__name__)


class ComplexityLevel(str, Enum):
    SIMPLE = "simple"
    MEDIUM = "medium"
    COMPLEX = "complex"


# Weights for complexity signals
_WEIGHTS = {
    "token_count": 0.20,
    "sentence_count": 0.10,
    "constraint_density": 0.25,
    "entity_density": 0.15,
    "question_count": 0.10,
    "nested_structure": 0.10,
    "technical_depth": 0.10,
}


def _count_tokens(text: str) -> int:
    """Approximate token count (words)."""
    return len(text.split())


def _count_sentences(text: str) -> int:
    """Count sentences using basic heuristic."""
    return max(1, len(re.split(r'[.!?]+', text.strip())) - 1) if text.strip() else 0


def _count_constraints(text: str) -> int:
    """Count explicit constraints / requirements."""
    constraint_patterns = [
        r"\bmust\b", r"\bshould\b", r"\brequire[ds]?\b", r"\bensure\b",
        r"\bconstraint\b", r"\blimit\b", r"\brestrict\b", r"\bno more than\b",
        r"\bat least\b", r"\bat most\b", r"\bexactly\b", r"\bmandatory\b",
        r"\bcritical\b", r"\bessential\b", r"\bnon-negotiable\b",
    ]
    text_lower = text.lower()
    return sum(1 for p in constraint_patterns if re.search(p, text_lower))


def _count_entities(text: str) -> int:
    """Count named-entity-like tokens (capitalised multi-word sequences, tech terms)."""
    named = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b", text)
    tech = re.findall(
        r"\b(?:API|REST|GraphQL|SQL|NoSQL|AWS|GCP|Azure|Docker|Kubernetes|"
        r"CI/CD|OAuth|JWT|HTTPS|WebSocket|gRPC|Redis|Kafka|"
        r"PostgreSQL|MongoDB|MySQL|React|Vue|Angular|Node\.js|"
        r"Python|Java|TypeScript|Go|Rust|C\+\+)\b",
        text,
        re.IGNORECASE,
    )
    return len(named) + len(tech)


def _count_questions(text: str) -> int:
    return text.count("?")


def _has_nested_structure(text: str) -> float:
    """Score for nested / hierarchical structure (bullets, numbered lists, headers)."""
    bullets = len(re.findall(r"^[\s]*[-*•]\s", text, re.MULTILINE))
    numbered = len(re.findall(r"^[\s]*\d+[.)]\s", text, re.MULTILINE))
    headers = len(re.findall(r"^#{1,6}\s|^\*\*[^*]+\*\*", text, re.MULTILINE))
    code_blocks = len(re.findall(r"```", text))
    total = bullets + numbered + headers + code_blocks
    if total >= 10:
        return 1.0
    if total >= 5:
        return 0.7
    if total >= 2:
        return 0.4
    return 0.0


def _technical_depth(text: str) -> float:
    """Score for technical depth (code, formulas, config, etc.)."""
    code_blocks = len(re.findall(r"```[\s\S]*?```", text))
    inline_code = len(re.findall(r"`[^`]+`", text))
    formulas = len(re.findall(r"\$[^$]+\$", text))
    json_like = len(re.findall(r"\{[^}]{10,}\}", text))
    total = code_blocks * 3 + inline_code + formulas * 2 + json_like
    if total >= 8:
        return 1.0
    if total >= 4:
        return 0.7
    if total >= 1:
        return 0.3
    return 0.0


def score_complexity(text: str) -> Dict[str, Any]:
    """
    Score the complexity of a prompt on a 0-1 scale.

    Returns:
        Dict with overall score, level, signal breakdown, and pipeline config.
    """
    if not text or not text.strip():
        return {
            "score": 0.0,
            "level": ComplexityLevel.SIMPLE,
            "signals": {},
            "recommended_iterations": 1,
            "skip_evaluation": True,
        }

    tokens = _count_tokens(text)
    sentences = _count_sentences(text)
    constraints = _count_constraints(text)
    entities = _count_entities(text)
    questions = _count_questions(text)
    nested = _has_nested_structure(text)
    tech = _technical_depth(text)

    # Normalise signals to 0-1
    signals = {
        "token_count": min(tokens / 300, 1.0),
        "sentence_count": min(sentences / 15, 1.0),
        "constraint_density": min(constraints / 8, 1.0),
        "entity_density": min(entities / 10, 1.0),
        "question_count": min(questions / 5, 1.0),
        "nested_structure": nested,
        "technical_depth": tech,
    }

    # Weighted score
    score = sum(signals[k] * _WEIGHTS[k] for k in _WEIGHTS)
    score = round(min(score, 1.0), 3)

    # Classify
    if score < 0.25:
        level = ComplexityLevel.SIMPLE
        iterations = 1
        skip_eval = tokens < 20
    elif score < 0.55:
        level = ComplexityLevel.MEDIUM
        iterations = 2
        skip_eval = False
    else:
        level = ComplexityLevel.COMPLEX
        iterations = 3
        skip_eval = False

    return {
        "score": score,
        "level": level.value,
        "signals": signals,
        "recommended_iterations": iterations,
        "skip_evaluation": skip_eval,
        "token_count": tokens,
    }


class ComplexityAnalyzer:
    """Facade for complexity scoring with caching."""

    def __init__(self):
        self._cache: Dict[str, Dict[str, Any]] = {}

    def analyze(self, text: str) -> Dict[str, Any]:
        """Analyze prompt complexity (cached)."""
        key = text[:200]
        if key in self._cache:
            return self._cache[key]
        result = score_complexity(text)
        self._cache[key] = result
        # Keep cache bounded
        if len(self._cache) > 500:
            oldest = next(iter(self._cache))
            del self._cache[oldest]
        return result

    def get_pipeline_config(self, text: str) -> Dict[str, Any]:
        """Get pipeline configuration based on complexity."""
        analysis = self.analyze(text)
        return {
            "max_iterations": analysis["recommended_iterations"],
            "skip_evaluation": analysis["skip_evaluation"],
            "complexity_level": analysis["level"],
            "enable_optimization": analysis["level"] == "complex",
        }


# Global instance
complexity_analyzer = ComplexityAnalyzer()
