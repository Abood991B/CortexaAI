"""Classifier Agent for Multi-Agent Prompt Engineering System."""

from typing import Dict, List, Optional, Any
from config.config import (
    settings, get_logger, metrics, log_performance,
    cache_manager, perf_config, generate_prompt_cache_key, log_cache_performance,
    circuit_breakers, reliability_config, log_circuit_breaker_event, CircuitBreakerOpenException,
    security_manager, security_config, log_security_event
)
from agents.exceptions import ClassificationError, LLMServiceError, DomainError
from agents.utils import is_retryable_error
import json
import re
import asyncio

# Set up structured logging
logger = get_logger(__name__)


class DomainClassifier:
    """Agent responsible for classifying prompts into domains and creating new expert agents."""

    def __init__(self):
        """Initialize the classifier with known domains and their characteristics."""
        self.known_domains = {
            "software_engineering": {
                "keywords": [
                    "code", "coding", "programming", "software", "development", "developer",
                    "algorithm", "function", "class", "api", "database", "debug", "debugging",
                    "refactor", "deploy", "deployment", "devops", "testing", "test", "unit test",
                    "bug", "error", "exception", "git", "github", "repository", "repo",
                    "backend", "frontend", "fullstack", "framework", "library", "sdk",
                    "python", "javascript", "typescript", "java", "rust", "golang", "c++",
                    "react", "angular", "vue", "node", "django", "flask", "fastapi",
                    "docker", "kubernetes", "ci/cd", "pipeline", "microservice",
                    "rest", "graphql", "endpoint", "server", "client", "http", "websocket",
                    "sql", "nosql", "mongodb", "postgres", "redis", "orm",
                    "variable", "loop", "array", "object", "string", "integer",
                    "compile", "runtime", "syntax", "script", "terminal", "command line",
                    "html", "css", "dom", "component", "module", "package", "npm", "pip",
                    "authentication", "authorization", "oauth", "jwt", "encryption",
                    "cloud", "aws", "azure", "gcp", "lambda", "serverless",
                ],
                "description": "Software development, coding, and programming tasks"
            },
            "data_science": {
                "keywords": [
                    "data", "dataset", "dataframe", "csv", "analysis", "analytics",
                    "machine learning", "deep learning", "neural network", "ai model",
                    "statistics", "statistical", "regression", "classification", "clustering",
                    "visualization", "chart", "graph", "plot", "dashboard",
                    "prediction", "forecast", "predictive", "training data",
                    "feature engineering", "hyperparameter", "accuracy", "precision", "recall",
                    "tensorflow", "pytorch", "scikit-learn", "pandas", "numpy", "matplotlib",
                    "nlp", "natural language processing", "computer vision", "transformer",
                    "random forest", "gradient boosting", "xgboost", "decision tree",
                    "data pipeline", "etl", "data warehouse", "big data", "spark",
                    "correlation", "hypothesis", "p-value", "confidence interval",
                    "a/b test", "experiment", "metric", "kpi",
                ],
                "description": "Data analysis, machine learning, and statistical tasks"
            },
            "report_writing": {
                "keywords": [
                    "report", "summary", "summarize", "findings", "conclusion",
                    "executive summary", "presentation", "documentation", "white paper",
                    "brief", "memo", "memorandum", "proposal", "quarterly",
                    "annual report", "review", "assessment report", "audit",
                    "stakeholder report", "progress report", "status update",
                    "key findings", "recommendation", "highlight",
                    "outline", "structure", "section", "appendix",
                    "formal writing", "professional document", "deliverable",
                    "financial report", "quarterly report", "research paper",
                    "case study", "technical report", "incident report",
                ],
                "description": "Report writing, documentation, and presentation tasks"
            },
            "education": {
                "keywords": [
                    "teach", "teaching", "teacher", "learn", "learning", "learner",
                    "student", "lesson", "lesson plan", "curriculum", "syllabus",
                    "educational", "tutorial", "explanation", "explain",
                    "course", "class", "classroom", "assessment", "rubric", "quiz",
                    "exam", "test", "homework", "assignment", "grade", "grading",
                    "school", "university", "college", "academy",
                    "pedagogy", "instruction", "instructional", "training",
                    "lecture", "seminar", "workshop", "module",
                    "beginner", "intermediate", "advanced level",
                    "study guide", "flashcard", "exercise",
                ],
                "description": "Educational content creation and teaching materials"
            },
            "business_strategy": {
                "keywords": [
                    "business", "strategy", "strategic", "marketing", "market",
                    "management", "growth", "competitive", "competition", "competitor",
                    "revenue", "roi", "profit", "profitability", "stakeholder",
                    "startup", "entrepreneur", "venture", "investor", "investment",
                    "product launch", "go-to-market", "gtm", "pricing",
                    "customer acquisition", "retention", "churn", "ltv", "cac",
                    "swot", "market research", "target audience", "persona",
                    "brand", "branding", "positioning", "differentiation",
                    "sales", "pipeline", "funnel", "conversion", "lead",
                    "b2b", "b2c", "saas", "subscription", "recurring revenue",
                    "partnership", "alliance", "merger", "acquisition",
                    "kpi", "okr", "roadmap", "milestone", "quarter",
                    "budget", "forecast", "financial", "valuation",
                ],
                "description": "Business strategy, planning, and management tasks"
            },
            "creative_writing": {
                "keywords": [
                    "write", "writing", "writer", "story", "stories", "storytelling",
                    "creative", "novel", "fiction", "non-fiction", "nonfiction",
                    "content", "blog", "blog post", "article", "essay",
                    "narrative", "narrator", "character", "protagonist", "dialogue",
                    "copy", "copywriting", "headline", "tagline", "slogan",
                    "audience", "engagement", "social media", "seo", "caption",
                    "poem", "poetry", "lyric", "script", "screenplay", "scene",
                    "tone", "voice", "style", "genre", "draft",
                    "edit", "proofread", "revision", "rewrite",
                    "persuasive", "compelling", "hook", "call to action",
                    "email copy", "newsletter", "landing page",
                ],
                "description": "Creative writing, content creation, and copywriting tasks"
            }
        }

        self.created_agents = {}  # Track dynamically created agents

        # Pre-compile keyword lookup structures for O(1) matching
        self._single_word_kws: Dict[str, Dict[str, bool]] = {}  # domain -> {kw: True}
        self._multi_word_kws: Dict[str, List[str]] = {}  # domain -> [kw, ...]
        for domain_name, domain_info in self.known_domains.items():
            singles: Dict[str, bool] = {}
            multis: List[str] = []
            for kw in domain_info.get("keywords", []):
                if " " in kw:
                    multis.append(kw)
                else:
                    singles[kw] = True
            self._single_word_kws[domain_name] = singles
            self._multi_word_kws[domain_name] = multis

    async def classify_prompt(self, prompt: str) -> Dict[str, Any]:
        """
        Classify a prompt into a domain with security, caching and retry mechanism.

        Delegates to :meth:`_fast_keyword_classify`, which uses pre-compiled
        keyword lookups (O(1) single-word matching via dict, plus substring
        search for multi-word keywords).  The best-scoring domain is returned
        when any keyword hit exists (``best_score > 0``); otherwise the
        ``"general"`` domain is returned as a fallback.  No LLM call is made.

        Args:
            prompt: The prompt to classify

        Returns:
            Dict containing classification results

        Raises:
            ClassificationError: If classification fails after retries
        """
        # Input sanitization
        if security_config.enable_input_sanitization:
            sanitized_result = security_manager.sanitize_input(prompt, "classification")

            if not sanitized_result['is_safe'] and security_config.enable_injection_detection:
                high_severity_events = [e for e in sanitized_result['security_events'] if e['severity'] == 'high']
                if high_severity_events:
                    log_security_event(logger, "unsafe_input_blocked", "high",
                                     context="classification", events=high_severity_events)
                    raise ClassificationError(
                        "Input contains potentially unsafe content",
                        prompt=prompt,
                        security_events=high_severity_events
                    )

            sanitized_prompt = sanitized_result['sanitized_text']
        else:
            sanitized_prompt = prompt

        # Check cache first if caching is enabled
        if perf_config.enable_caching:
            cache_key = generate_prompt_cache_key(sanitized_prompt, prefix="classification")
            cached_result = cache_manager.get(cache_key)
            if cached_result:
                log_cache_performance(logger, "prompt_classification", True, prompt_length=len(sanitized_prompt))
                return cached_result

        # ── Fast keyword path — always returns a result (no LLM needed) ──
        fast_result = self._fast_keyword_classify(sanitized_prompt)
        logger.info(f"Keyword classification: {fast_result['domain']} (confidence {fast_result['confidence']:.2f})")
        if perf_config.enable_caching:
            cache_manager.set(cache_key, fast_result, perf_config.cache_ttl)
        return fast_result

    def _fast_keyword_classify(self, prompt: str) -> Dict[str, Any]:
        """High-speed keyword classification using pre-compiled lookups.

        **Always** returns a result so the LLM is never needed for domain
        classification.

        Algorithm:
        1. Extract unique words from the prompt.
        2. For each known domain, compute a weighted score:
           - Single-word keywords matched via O(1) dict look-up → **1 point** each.
           - Multi-word keywords matched via substring search → **2 points** each.
        3. The domain with the highest weighted score wins.
        4. If ``best_score > 0`` the winning domain is returned with a
           confidence derived from the score and the gap to the runner-up.
        5. If ``best_score == 0`` (no keyword hits at all) the ``"general"``
           domain is returned with a low confidence (0.55).
        """
        prompt_lower = prompt.lower()
        # Extract unique words from the prompt for O(1) single-word matching
        prompt_words = set(re.findall(r'[a-z0-9#+/.]+', prompt_lower))

        scores: Dict[str, float] = {}
        matched_kws: Dict[str, List[str]] = {}

        for domain_name in self.known_domains:
            hits: List[str] = []
            weighted_score = 0.0

            # O(1) single-word matching via set intersection
            singles = self._single_word_kws.get(domain_name, {})
            for word in prompt_words:
                if word in singles:
                    hits.append(word)
                    weighted_score += 1.0

            # Multi-word keywords still need substring search (small list)
            for kw in self._multi_word_kws.get(domain_name, []):
                if kw in prompt_lower:
                    hits.append(kw)
                    weighted_score += 2.0

            scores[domain_name] = weighted_score
            matched_kws[domain_name] = hits

        sorted_domains = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        best_domain, best_score = sorted_domains[0] if sorted_domains else ("general", 0.0)
        runner_up_score = sorted_domains[1][1] if len(sorted_domains) > 1 else 0.0

        if best_score == 0:
            # No keyword matches — return general
            return {
                "domain": "general",
                "confidence": 0.55,
                "is_new_domain": False,
                "new_domain_name": None,
                "new_domain_description": None,
                "key_topics": [],
                "reasoning": "No strong keyword matches — using general domain"
            }

        gap = best_score - runner_up_score
        num_hits = len(matched_kws.get(best_domain, []))

        # Confidence based on weighted score & gap
        if best_score >= 6 and gap >= 3:
            confidence = min(0.85 + best_score * 0.01, 0.95)
        elif best_score >= 4 and gap >= 2:
            confidence = min(0.75 + best_score * 0.02, 0.92)
        elif best_score >= 2:
            confidence = min(0.60 + best_score * 0.03, 0.85)
        else:
            confidence = 0.55 + best_score * 0.03

        return {
            "domain": best_domain,
            "confidence": round(min(confidence, 0.95), 2),
            "is_new_domain": False,
            "new_domain_name": None,
            "new_domain_description": None,
            "key_topics": matched_kws[best_domain][:5],
            "reasoning": f"Keyword match ({num_hits} hits, weighted_score={best_score:.1f}, gap={gap:.1f})"
        }

    def _get_fallback_classification_result(self, prompt: str) -> Dict[str, Any]:
        """Keyword-based fallback classification covering all known domains.
        
        Returns the domain with the highest keyword overlap.  Falls back to
        ``general`` if no domain scores above zero.
        """
        prompt_lower = prompt.lower()

        best_domain = "general"
        best_score = 0

        for domain_name, domain_info in self.known_domains.items():
            keywords = domain_info.get("keywords", [])
            # Count distinct keyword matches (not repeated hits)
            score = sum(1 for kw in keywords if kw in prompt_lower)
            if score > best_score:
                best_score = score
                best_domain = domain_name

        confidence = min(0.3 + best_score * 0.08, reliability_config.fallback_response_quality)

        return {
            "domain": best_domain,
            "confidence": round(confidence, 2),
            "is_new_domain": False,
            "key_topics": [kw for kw in self.known_domains.get(best_domain, {}).get("keywords", [])[:3]
                           if kw in prompt_lower] or [],
            "reasoning": f"Fallback keyword classification (matched {best_score} keywords)"
        }

    async def classify_prompt_type(self, prompt: str) -> str:
        """Classify whether a prompt is ``raw`` or ``structured``.
        
        Uses a weighted heuristic that checks for:
        - Structural markers (headings, numbered/bullet lists, colons)
        - Task-oriented vocabulary
        - Multi-line formatting
        - Overall length
        """
        prompt_lower = prompt.lower()

        # Weighted structural indicators
        structural_keywords = [
            "requirements:", "specifications:", "objective:", "task:",
            "constraints:", "deliverables:", "output format:",
            "context:", "background:", "instructions:"
        ]
        keyword_hits = sum(1 for kw in structural_keywords if kw in prompt_lower)

        # Line-level formatting signals
        lines = prompt.split('\n')
        non_empty_lines = [l.strip() for l in lines if l.strip()]
        list_lines = sum(
            1 for line in non_empty_lines
            if line.startswith(('-', '*', '•'))
            or (len(line) >= 2 and line[0].isdigit() and line[1] in '.)')
        )
        heading_lines = sum(1 for line in non_empty_lines if line.startswith('#') or line.endswith(':'))

        # Composite score (each feature worth 1 point)
        score = (
            keyword_hits
            + min(list_lines, 4)     # cap list contribution at 4
            + min(heading_lines, 3)  # cap heading contribution at 3
            + (1 if len(non_empty_lines) > 5 else 0)
            + (1 if len(prompt) > 300 else 0)
        )

        return "structured" if score >= 3 else "raw"

    def get_available_domains(self) -> Dict[str, Dict]:
        """Get all available domains (known + dynamically created)."""
        return self.known_domains.copy()

    def has_domain(self, domain: str) -> bool:
        """Check if a domain exists."""
        return domain in self.known_domains

    def get_domain_info(self, domain: str) -> Optional[Dict]:
        """Get information about a specific domain."""
        return self.known_domains.get(domain)
