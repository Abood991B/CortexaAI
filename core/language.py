"""
Multi-Language Support for CortexaAI.

Language detection, routing to language-aware templates,
and evaluation criteria adjustments for non-English prompts.
"""

import re
from typing import Dict, Optional, Tuple, List, Any
from config.config import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Language Detection
# ---------------------------------------------------------------------------

# Unicode block heuristics (fast, dependency-free)
_LANG_PATTERNS: Dict[str, re.Pattern] = {
    "arabic": re.compile(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]+"),
    "chinese": re.compile(r"[\u4E00-\u9FFF\u3400-\u4DBF]+"),
    "japanese": re.compile(r"[\u3040-\u309F\u30A0-\u30FF]+"),
    "korean": re.compile(r"[\uAC00-\uD7AF\u1100-\u11FF]+"),
    "cyrillic": re.compile(r"[\u0400-\u04FF]+"),            # Russian, Ukrainian, etc.
    "devanagari": re.compile(r"[\u0900-\u097F]+"),            # Hindi, Sanskrit
    "thai": re.compile(r"[\u0E00-\u0E7F]+"),
    "hebrew": re.compile(r"[\u0590-\u05FF]+"),
}

# Common stopwords for Latin-script languages
_LANG_STOPWORDS: Dict[str, set] = {
    "english": {"the", "is", "and", "of", "to", "in", "it", "for", "that", "with", "on", "as", "are", "was"},
    "spanish": {"el", "la", "de", "en", "y", "que", "es", "un", "los", "por", "con", "una", "del", "las", "se"},
    "french": {"le", "la", "de", "et", "en", "un", "une", "est", "les", "des", "du", "que", "dans", "pas", "pour"},
    "german": {"der", "die", "und", "in", "den", "von", "zu", "das", "mit", "ist", "auf", "für", "sich", "ein", "nicht"},
    "portuguese": {"de", "que", "e", "do", "da", "em", "um", "para", "com", "não", "uma", "os", "se", "na", "por"},
    "italian": {"di", "che", "il", "la", "in", "un", "per", "non", "una", "del", "con", "sono", "da", "dei", "le"},
    "dutch": {"de", "het", "van", "en", "een", "is", "dat", "op", "te", "in", "voor", "met", "niet", "er", "zijn"},
    "turkish": {"bir", "ve", "bu", "da", "için", "ile", "olan", "den", "gibi", "daha", "ya", "ne", "çok", "var", "her"},
}

# RTL languages
RTL_LANGUAGES = {"arabic", "hebrew"}

# Language metadata
LANGUAGE_METADATA: Dict[str, Dict[str, Any]] = {
    "english":    {"code": "en", "direction": "ltr", "name": "English"},
    "arabic":     {"code": "ar", "direction": "rtl", "name": "Arabic"},
    "chinese":    {"code": "zh", "direction": "ltr", "name": "Chinese"},
    "japanese":   {"code": "ja", "direction": "ltr", "name": "Japanese"},
    "korean":     {"code": "ko", "direction": "ltr", "name": "Korean"},
    "spanish":    {"code": "es", "direction": "ltr", "name": "Spanish"},
    "french":     {"code": "fr", "direction": "ltr", "name": "French"},
    "german":     {"code": "de", "direction": "ltr", "name": "German"},
    "portuguese": {"code": "pt", "direction": "ltr", "name": "Portuguese"},
    "italian":    {"code": "it", "direction": "ltr", "name": "Italian"},
    "dutch":      {"code": "nl", "direction": "ltr", "name": "Dutch"},
    "turkish":    {"code": "tr", "direction": "ltr", "name": "Turkish"},
    "cyrillic":   {"code": "ru", "direction": "ltr", "name": "Russian"},
    "devanagari": {"code": "hi", "direction": "ltr", "name": "Hindi"},
    "thai":       {"code": "th", "direction": "ltr", "name": "Thai"},
    "hebrew":     {"code": "he", "direction": "rtl", "name": "Hebrew"},
}


def detect_language(text: str) -> Tuple[str, float]:
    """
    Detect the primary language of a text string.

    Returns:
        Tuple of (language_name, confidence) where confidence is 0.0-1.0.
    """
    if not text or not text.strip():
        return ("english", 0.5)

    text_clean = text.strip()
    total_chars = len(text_clean)

    # Phase 1: Script-based detection (non-Latin scripts)
    script_scores: Dict[str, float] = {}
    for lang, pattern in _LANG_PATTERNS.items():
        matches = pattern.findall(text_clean)
        char_count = sum(len(m) for m in matches)
        if char_count > 0:
            script_scores[lang] = char_count / total_chars

    if script_scores:
        best_script = max(script_scores, key=script_scores.get)
        confidence = min(script_scores[best_script] * 2, 1.0)  # Boost confidence
        if confidence > 0.15:
            return (best_script, round(confidence, 2))

    # Phase 2: Stopword-based detection (Latin-script languages)
    words = set(re.findall(r"\b\w+\b", text_clean.lower()))
    if not words:
        return ("english", 0.3)

    stopword_scores: Dict[str, float] = {}
    for lang, stopwords in _LANG_STOPWORDS.items():
        overlap = words & stopwords
        if overlap:
            stopword_scores[lang] = len(overlap) / len(stopwords)

    if stopword_scores:
        best_lang = max(stopword_scores, key=stopword_scores.get)
        confidence = min(stopword_scores[best_lang] * 3, 0.95)
        return (best_lang, round(confidence, 2))

    # Default fallback
    return ("english", 0.3)


def get_language_metadata(language: str) -> Dict[str, Any]:
    """Get metadata for a detected language."""
    return LANGUAGE_METADATA.get(language, LANGUAGE_METADATA["english"])


def is_rtl(language: str) -> bool:
    """Check if the language is right-to-left."""
    return language in RTL_LANGUAGES


# ---------------------------------------------------------------------------
# Language-Aware Evaluation Adjustments
# ---------------------------------------------------------------------------

# Different languages have different structural norms
LANGUAGE_EVAL_ADJUSTMENTS: Dict[str, Dict[str, float]] = {
    "english":    {"clarity": 1.0, "specificity": 1.0, "structure": 1.0},
    "arabic":     {"clarity": 0.9, "specificity": 1.0, "structure": 0.85},
    "chinese":    {"clarity": 0.9, "specificity": 1.0, "structure": 0.9},
    "japanese":   {"clarity": 0.9, "specificity": 1.0, "structure": 0.85},
    "korean":     {"clarity": 0.9, "specificity": 1.0, "structure": 0.9},
    "spanish":    {"clarity": 1.0, "specificity": 0.95, "structure": 1.0},
    "french":     {"clarity": 1.0, "specificity": 0.95, "structure": 1.0},
    "german":     {"clarity": 0.95, "specificity": 1.0, "structure": 1.0},
}


def get_eval_adjustments(language: str) -> Dict[str, float]:
    """Get evaluation criteria adjustments for a language."""
    return LANGUAGE_EVAL_ADJUSTMENTS.get(language, {"clarity": 1.0, "specificity": 1.0, "structure": 1.0})


# ---------------------------------------------------------------------------
# Language-Aware Prompt Prefix
# ---------------------------------------------------------------------------

LANGUAGE_INSTRUCTIONS: Dict[str, str] = {
    "arabic": "يرجى تحسين هذا النص التالي بالعربية مع الحفاظ على المعنى الأصلي:\n\n",
    "chinese": "请优化以下中文文本，保持原始含义：\n\n",
    "japanese": "以下の日本語テキストを改善してください。元の意味を保持してください：\n\n",
    "korean": "다음 한국어 텍스트를 개선해 주세요. 원래 의미를 유지해 주세요:\n\n",
    "spanish": "Por favor, mejora el siguiente texto en español manteniendo el significado original:\n\n",
    "french": "Veuillez améliorer le texte suivant en français en conservant le sens original :\n\n",
    "german": "Bitte verbessern Sie den folgenden deutschen Text und behalten Sie die ursprüngliche Bedeutung bei:\n\n",
}


def get_language_instruction(language: str) -> str:
    """Get optimization instruction in the detected language."""
    return LANGUAGE_INSTRUCTIONS.get(language, "")


class LanguageProcessor:
    """Orchestrates language detection and processing adjustments."""

    def analyze(self, text: str) -> Dict[str, Any]:
        """Analyze text and return language metadata."""
        language, confidence = detect_language(text)
        metadata = get_language_metadata(language)
        return {
            "language": language,
            "confidence": confidence,
            "code": metadata["code"],
            "direction": metadata["direction"],
            "name": metadata["name"],
            "is_rtl": is_rtl(language),
            "eval_adjustments": get_eval_adjustments(language),
            "optimization_prefix": get_language_instruction(language),
        }

    def get_supported_languages(self) -> List[Dict[str, Any]]:
        """Return list of supported languages."""
        return [
            {"language": lang, **meta}
            for lang, meta in LANGUAGE_METADATA.items()
        ]


# Global instance
language_processor = LanguageProcessor()
