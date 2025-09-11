"""Classifier Agent for Multi-Agent Prompt Engineering System."""

from typing import Dict, List, Optional, Any
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI
from config.config import (
    settings, get_model_config, get_logger, metrics, log_performance,
    cache_manager, perf_config, generate_prompt_cache_key, log_cache_performance,
    circuit_breakers, reliability_config, log_circuit_breaker_event, CircuitBreakerOpenException,
    security_manager, security_config, log_security_event
)
from agents.exceptions import ClassificationError, LLMServiceError, DomainError
from agents.utils import is_retryable_error, sanitize_json_response
import json
import asyncio

# Set up structured logging
logger = get_logger(__name__)


class DomainClassifier:
    """Agent responsible for classifying prompts into domains and creating new expert agents."""

    def __init__(self):
        """Initialize the classifier with known domains and their characteristics."""
        self.known_domains = {
            "software_engineering": {
                "keywords": ["code", "programming", "software", "development", "algorithm", "function", "class", "api", "database", "debug", "refactor"],
                "description": "Software development, coding, and programming tasks"
            },
            "data_science": {
                "keywords": ["data", "analysis", "machine learning", "statistics", "visualization", "dataset", "model", "prediction", "analytics"],
                "description": "Data analysis, machine learning, and statistical tasks"
            },
            "report_writing": {
                "keywords": ["report", "summary", "analysis", "findings", "conclusion", "executive", "presentation", "documentation"],
                "description": "Report writing, documentation, and presentation tasks"
            },
            "education": {
                "keywords": ["teaching", "learning", "student", "lesson", "curriculum", "educational", "tutorial", "explanation"],
                "description": "Educational content creation and teaching materials"
            },
            "business_strategy": {
                "keywords": ["business", "strategy", "marketing", "management", "growth", "market", "competitive", "analysis"],
                "description": "Business strategy, and management tasks"
            }
        }

        self.created_agents = {}  # Track dynamically created agents
        self._setup_classifier_chain()

    def _setup_classifier_chain(self):
        """Set up the LangChain for domain classification."""
        model_config = get_model_config(provider="google")
        model = ChatGoogleGenerativeAI(
            model=model_config["model_name"],
            google_api_key=model_config["api_key"],
            temperature=0.1
        )

        classification_prompt = PromptTemplate.from_template("""
        You are a domain classification expert. Analyze the following prompt and determine its primary domain.

        PROMPT TO ANALYZE:
        {prompt}

        KNOWN DOMAINS:
        {domains}

        TASK:
        1. Determine the most appropriate domain for this prompt from the known domains above
        2. If none of the known domains fit well, suggest a new domain name and description
        3. Provide a confidence score (0.0 to 1.0) for your classification
        4. Extract key topics/keywords from the prompt

        Respond in JSON format with the following structure:
        {{
            "domain": "domain_name",
            "confidence": 0.95,
            "is_new_domain": false,
            "new_domain_name": null,
            "new_domain_description": null,
            "key_topics": ["topic1", "topic2"],
            "reasoning": "Brief explanation of classification"
        }}

        If suggesting a new domain, set is_new_domain to true and provide new_domain_name and new_domain_description.
        """)

        self.classifier_chain = (
            {
                "prompt": RunnablePassthrough(),
                "domains": lambda x: json.dumps(self.known_domains, indent=2)
            }
            | classification_prompt
            | model
            | JsonOutputParser()
        )

    async def classify_prompt(self, prompt: str) -> Dict[str, Any]:
        """
        Classify a prompt into a domain with security, caching and retry mechanism.

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
                # Block potentially unsafe prompts
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

        max_retries = getattr(settings, 'max_llm_retries', 3)
        retry_delay = getattr(settings, 'llm_retry_delay', 1.0)

        # Use circuit breaker if enabled
        if reliability_config.enable_circuit_breakers:
            try:
                result = await circuit_breakers["classification"].call(
                    self._classify_with_fallback, sanitized_prompt, cache_key, max_retries, retry_delay
                )
                return result
            except CircuitBreakerOpenException as cboe:
                logger.warning(f"Circuit breaker open for classification: {cboe}")
                log_circuit_breaker_event(logger, "classification", "circuit_open",
                                         prompt_length=len(sanitized_prompt), failure_count=cboe.failure_count)

                # Return fallback result when circuit is open
                if reliability_config.enable_fallbacks:
                    return self._get_fallback_classification_result(sanitized_prompt)
                else:
                    raise ClassificationError(
                        "Classification circuit breaker is open",
                        prompt=sanitized_prompt,
                        circuit_breaker="classification"
                    )

        # Original retry logic without circuit breaker
        for attempt in range(max_retries + 1):
            try:
                logger.info(f"Classifying prompt: {sanitized_prompt[:100]}... (attempt {attempt + 1})")

                result = await self.classifier_chain.ainvoke(sanitized_prompt)

                # Handle new domain creation
                if result.get("is_new_domain", False):
                    new_domain = result["new_domain_name"]
                    new_description = result["new_domain_description"]

                    logger.info(f"Creating new domain: {new_domain}")
                    self._create_new_domain(new_domain, new_description, result.get("key_topics", []))

                    # Update the result to use the new domain
                    result["domain"] = new_domain

                # Cache the result if caching is enabled
                if perf_config.enable_caching:
                    cache_manager.set(cache_key, result, perf_config.cache_ttl)

                logger.info(f"Classification result: {result}")
                log_cache_performance(logger, "prompt_classification", False, prompt_length=len(sanitized_prompt))
                return result

            except ValueError as ve:
                if "No generations found in stream" in str(ve):
                    logger.error(f"Model streaming error: {ve}. This may be due to an invalid model name, API key, or model availability.")
                    # This is a configuration issue, not retryable
                    return {
                        "domain": "general",
                        "confidence": 0.5,
                        "is_new_domain": False,
                        "key_topics": [],
                        "reasoning": f"Model error: {str(ve)}. Please check the model name '{self.classifier_chain.steps[1].model}' and API key."
                    }
                else:
                    # Re-raise unexpected ValueError
                    raise ClassificationError(
                        "Unexpected classification error",
                        prompt=sanitized_prompt,
                        cause=ve
                    )

            except Exception as e:
                error_msg = f"Error classifying prompt: {str(e)}"

                if attempt < max_retries:
                    # Determine if error is retryable
                    is_retryable = self._is_retryable_error(e)

                    if is_retryable:
                        logger.warning(f"{error_msg}. Retrying in {retry_delay}s... (attempt {attempt + 1}/{max_retries})")
                        await asyncio.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                        continue
                    else:
                        logger.error(f"{error_msg}. Error is not retryable.")
                        break
                else:
                    logger.error(f"{error_msg}. All {max_retries + 1} attempts failed.")

                    # Raise custom exception with detailed context
                    raise ClassificationError(
                        f"Failed to classify prompt after {max_retries + 1} attempts",
                        prompt=sanitized_prompt,
                        cause=e
                    )

        # Fallback: return degraded result if all retries failed
        logger.warning("Returning fallback classification result due to repeated failures")
        return self._get_fallback_classification_result(sanitized_prompt)

    def _get_fallback_classification_result(self, prompt: str) -> Dict[str, Any]:
        """Get a fallback classification result when all else fails."""
        # Simple keyword-based fallback classification
        prompt_lower = prompt.lower()

        # Check for software engineering keywords
        se_keywords = ["code", "programming", "software", "development", "algorithm", "function", "class", "api", "database", "debug"]
        if any(keyword in prompt_lower for keyword in se_keywords):
            return {
                "domain": "software_engineering",
                "confidence": reliability_config.fallback_response_quality,
                "is_new_domain": False,
                "key_topics": ["programming"],
                "reasoning": "Fallback classification based on keywords"
            }

        # Check for data science keywords
        ds_keywords = ["data", "analysis", "machine learning", "statistics", "visualization", "dataset", "model"]
        if any(keyword in prompt_lower for keyword in ds_keywords):
            return {
                "domain": "data_science",
                "confidence": reliability_config.fallback_response_quality,
                "is_new_domain": False,
                "key_topics": ["data", "analysis"],
                "reasoning": "Fallback classification based on keywords"
            }

        # Default fallback
        return {
            "domain": "general",
            "confidence": reliability_config.fallback_response_quality,
            "is_new_domain": False,
            "key_topics": [],
            "reasoning": "Fallback classification - unable to determine specific domain"
        }

    async def _classify_with_fallback(self, prompt: str, cache_key: str,
                                    max_retries: int, retry_delay: float) -> Dict[str, Any]:
        """Classify with circuit breaker protection and fallback."""
        for attempt in range(max_retries + 1):
            try:
                logger.info(f"Classifying prompt: {prompt[:100]}... (attempt {attempt + 1})")

                result = await self.classifier_chain.ainvoke(prompt)

                # Handle new domain creation
                if result.get("is_new_domain", False):
                    new_domain = result["new_domain_name"]
                    new_description = result["new_domain_description"]

                    logger.info(f"Creating new domain: {new_domain}")
                    self._create_new_domain(new_domain, new_description, result.get("key_topics", []))

                    # Update the result to use the new domain
                    result["domain"] = new_domain

                # Cache the result if caching is enabled
                if perf_config.enable_caching:
                    cache_manager.set(cache_key, result, perf_config.cache_ttl)

                logger.info(f"Classification result: {result}")
                log_cache_performance(logger, "prompt_classification", False, prompt_length=len(prompt))
                return result

            except ValueError as ve:
                if "No generations found in stream" in str(ve):
                    logger.error(f"Model streaming error: {ve}. This may be due to an invalid model name, API key, or model availability.")
                    # This is a configuration issue, not retryable
                    return self._get_fallback_classification_result(prompt)
                else:
                    # Re-raise unexpected ValueError
                    raise ClassificationError(
                        "Unexpected classification error",
                        prompt=prompt,
                        cause=ve
                    )

            except Exception as e:
                error_msg = f"Error classifying prompt: {str(e)}"

                if attempt < max_retries:
                    # Determine if error is retryable
                    is_retryable = self._is_retryable_error(e)

                    if is_retryable:
                        logger.warning(f"{error_msg}. Retrying in {retry_delay}s... (attempt {attempt + 1}/{max_retries})")
                        await asyncio.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                        continue
                    else:
                        logger.error(f"{error_msg}. Error is not retryable.")
                        break
                else:
                    logger.error(f"{error_msg}. All {max_retries + 1} attempts failed.")
                    # Return fallback instead of raising exception
                    return self._get_fallback_classification_result(prompt)

        # If we get here, return fallback
        return self._get_fallback_classification_result(prompt)

    def _is_retryable_error(self, error: Exception) -> bool:
        return is_retryable_error(error)

    def _create_new_domain(self, domain_name: str, description: str, keywords: List[str]):
        """
        Create a new domain dynamically.

        Args:
            domain_name: Name of the new domain
            description: Description of the domain
            keywords: Key topics/keywords for the domain
        """
        self.known_domains[domain_name] = {
            "keywords": keywords,
            "description": description
        }

        # Mark that we've created an agent for this domain
        self.created_agents[domain_name] = {
            "created": True,
            "description": description
        }

        logger.info(f"New domain created: {domain_name}")

    async def classify_prompt_type(self, prompt: str) -> str:
        """Classify whether a prompt is raw or structured based on heuristics."""
        # Simple heuristic: if it has clear sections, formatting, or specific keywords, it's structured
        structured_indicators = [
            "requirements:", "specifications:", "please", "i need", "create",
            "develop", "build", "implement", "task:", "objective:",
            "1.", "2.", "3.", "-", "*", "•"
        ]

        prompt_lower = prompt.lower()
        structured_score = sum(1 for indicator in structured_indicators if indicator in prompt_lower)

        # Check for structured formatting
        lines = prompt.split('\n')
        formatted_lines = sum(1 for line in lines if line.strip().startswith(('-', '*', '•', '1.', '2.', '3.')))

        if structured_score > 3 or formatted_lines > 2 or len(lines) > 5:
            return "structured"
        else:
            return "raw"

    def get_available_domains(self) -> Dict[str, Dict]:
        """Get all available domains (known + dynamically created)."""
        return self.known_domains.copy()

    def has_domain(self, domain: str) -> bool:
        """Check if a domain exists."""
        return domain in self.known_domains

    def get_domain_info(self, domain: str) -> Optional[Dict]:
        """Get information about a specific domain."""
        return self.known_domains.get(domain)


# Global classifier instance
classifier = DomainClassifier()
