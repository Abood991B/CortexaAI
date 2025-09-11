"""Base Expert Prompt Engineer Agent framework."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI
from agents.exceptions import ImprovementError, LLMServiceError, ConfigurationError
from agents.utils import is_retryable_error, sanitize_json_response
import asyncio

from config.config import (
    settings, get_model_config, get_logger, metrics, log_performance,
    cache_manager, perf_config, generate_prompt_cache_key, log_cache_performance,
    security_manager, security_config, log_security_event,
    memory_config, prompt_generation_config
)
from agents.memory import memory_manager

# Set up structured logging
logger = get_logger(__name__)


class BaseExpertAgent(ABC):
    """Base class for all domain-specific expert prompt engineering agents."""

    def __init__(self, domain: str, domain_description: str):
        """
        Initialize the expert agent.

        Args:
            domain: The domain this agent specializes in
            domain_description: Description of the domain
        """
        self.domain = domain
        self.domain_description = domain_description
        self.expertise_areas = self._define_expertise_areas()
        self.improvement_templates = self._define_improvement_templates()
        self._setup_improvement_chain()

    async def improve_prompt_with_memory(self, original_prompt: str, user_id: str,
                                       prompt_type: str = "raw", key_topics: List[str] = None) -> Dict[str, Any]:
        """
        Improve a prompt using domain expertise with RAG-enhanced context and memory.

        Args:
            original_prompt: The original prompt to improve
            user_id: User identifier for memory retrieval
            prompt_type: Type of prompt ("raw" or "structured")
            key_topics: Key topics identified by classifier

        Returns:
            Dict containing the improved prompt and analysis
        """
        # Generate RAG context
        rag_context = await memory_manager.generate_rag_context(
            user_id=user_id,
            domain=self.domain,
            query=original_prompt
        )

        # Get conversation context
        conversation_context = memory_manager.get_conversation_context(user_id)

        # Combine context for enhanced improvement
        enhanced_context = self._build_enhanced_context(
            original_prompt, rag_context, conversation_context
        )

        # Use standard improvement with enhanced context
        result = await self.improve_prompt_with_context(
            original_prompt=original_prompt,
            context=enhanced_context,
            prompt_type=prompt_type,
            key_topics=key_topics
        )

        # Store the improvement in memory for future reference
        await memory_manager.store_memory(
            user_id=user_id,
            content=f"Prompt: {original_prompt}\nImprovement: {result.get('improved_prompt', '')}",
            metadata={
                'domain': self.domain,
                'prompt_type': prompt_type,
                'improvement_score': result.get('effectiveness_score', 0),
                'type': 'prompt_improvement'
            }
        )

        return result

    async def improve_prompt_with_context(self, original_prompt: str, context: Dict[str, Any],
                                        prompt_type: str = "raw", key_topics: List[str] = None) -> Dict[str, Any]:
        """
        Improve a prompt using additional context information.

        Args:
            original_prompt: The original prompt to improve
            context: Additional context from RAG and conversation history
            prompt_type: Type of prompt ("raw" or "structured")
            key_topics: Key topics identified by classifier

        Returns:
            Dict containing the improved prompt and analysis
        """
        # Input sanitization
        if security_config.enable_input_sanitization:
            sanitized_result = security_manager.sanitize_input(original_prompt, f"improvement_{self.domain}")

            if not sanitized_result['is_safe'] and security_config.enable_injection_detection:
                high_severity_events = [e for e in sanitized_result['security_events'] if e['severity'] == 'high']
                if high_severity_events:
                    log_security_event(logger, "unsafe_input_blocked", "high",
                                     context=f"improvement_{self.domain}", events=high_severity_events)
                    raise ImprovementError(
                        "Input contains potentially unsafe content",
                        domain=self.domain,
                        prompt_type=prompt_type,
                        original_prompt=original_prompt,
                        security_events=high_severity_events
                    )

            sanitized_prompt = sanitized_result['sanitized_text']
        else:
            sanitized_prompt = original_prompt

        # Check cache first if caching is enabled
        if perf_config.enable_caching:
            cache_key = generate_prompt_cache_key(sanitized_prompt, self.domain, f"{prompt_type}_context")
            cached_result = cache_manager.get(cache_key)
            if cached_result:
                log_cache_performance(logger, "prompt_improvement_context", True,
                                    domain=self.domain, prompt_type=prompt_type)
                return cached_result

        max_retries = getattr(settings, 'max_llm_retries', 3)
        retry_delay = getattr(settings, 'llm_retry_delay', 1.0)

        for attempt in range(max_retries + 1):
            try:
                logger.info(f"Improving {prompt_type} prompt with context in domain {self.domain} (attempt {attempt + 1})")

                # Use context-enhanced improvement chain
                input_data = {
                    "original_prompt": sanitized_prompt,
                    "prompt_type": prompt_type,
                    "key_topics": key_topics or [],
                    "context": context
                }

                result = await self.improvement_with_context_chain.ainvoke(input_data)

                # Cache the result if caching is enabled
                if perf_config.enable_caching:
                    cache_manager.set(cache_key, result, perf_config.cache_ttl)

                logger.info(f"Context-enhanced prompt improvement completed for {self.domain}")
                log_cache_performance(logger, "prompt_improvement_context", False,
                                    domain=self.domain, prompt_type=prompt_type)
                return result

            except Exception as e:
                error_msg = f"Error improving prompt with context in {self.domain}: {str(e)}"

                if attempt < max_retries:
                    is_retryable = self._is_retryable_error(e)

                    if is_retryable:
                        logger.warning(f"{error_msg}. Retrying in {retry_delay}s... (attempt {attempt + 1}/{max_retries})")
                        await asyncio.sleep(retry_delay)
                        retry_delay *= 2
                        continue
                    else:
                        logger.error(f"{error_msg}. Error is not retryable.")
                        break
                else:
                    logger.error(f"{error_msg}. All {max_retries + 1} attempts failed.")
                    raise ImprovementError(
                        f"Failed to improve prompt after {max_retries + 1} attempts",
                        domain=self.domain,
                        prompt_type=prompt_type,
                        original_prompt=original_prompt,
                        cause=e
                    )

        # Fallback result
        logger.warning(f"Returning fallback result for {self.domain} due to repeated failures")
        return {
            "improved_prompt": original_prompt,
            "improvements_made": [],
            "key_additions": [],
            "structure_analysis": "Improvement failed after multiple attempts",
            "effectiveness_score": 0.5,
            "reasoning": "Unable to improve prompt due to processing errors"
        }

    def _build_enhanced_context(self, original_prompt: str, rag_context: Dict[str, Any],
                              conversation_context: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Build enhanced context for prompt improvement."""
        context_parts = []

        # Add RAG context
        for part in rag_context.get('context_parts', []):
            context_parts.append({
                'type': part['type'],
                'content': part['content'],
                'relevance': part['relevance'],
                'source': 'rag'
            })

        # Add recent conversation context
        for turn in conversation_context[-3:]:  # Last 3 turns
            context_parts.append({
                'type': 'conversation',
                'content': f"User: {turn['message']}\nResponse: {turn['response']}",
                'relevance': 0.6,
                'source': 'conversation'
            })

        return {
            'original_prompt': original_prompt,
            'context_parts': context_parts,
            'rag_metadata': {
                'memories_count': rag_context.get('memories_count', 0),
                'knowledge_count': rag_context.get('knowledge_count', 0),
                'total_context_length': rag_context.get('total_context_length', 0)
            },
            'conversation_turns': len(conversation_context)
        }

    async def improve_prompt(self, original_prompt: str, prompt_type: str = "raw",
                      key_topics: List[str] = None) -> Dict[str, Any]:
        """
        Improve a prompt using domain expertise with security, caching and retry mechanism.

        Args:
            original_prompt: The original prompt to improve
            prompt_type: Type of prompt ("raw" or "structured")
            key_topics: Key topics identified by classifier

        Returns:
            Dict containing the improved prompt and analysis

        Raises:
            ImprovementError: If prompt improvement fails after retries
        """
        # Input sanitization
        if security_config.enable_input_sanitization:
            sanitized_result = security_manager.sanitize_input(original_prompt, f"improvement_{self.domain}")

            if not sanitized_result['is_safe'] and security_config.enable_injection_detection:
                # Block potentially unsafe prompts
                high_severity_events = [e for e in sanitized_result['security_events'] if e['severity'] == 'high']
                if high_severity_events:
                    log_security_event(logger, "unsafe_input_blocked", "high",
                                     context=f"improvement_{self.domain}", events=high_severity_events)
                    raise ImprovementError(
                        "Input contains potentially unsafe content",
                        domain=self.domain,
                        prompt_type=prompt_type,
                        original_prompt=original_prompt,
                        security_events=high_severity_events
                    )

            sanitized_prompt = sanitized_result['sanitized_text']
        else:
            sanitized_prompt = original_prompt

        # Check cache first if caching is enabled
        if perf_config.enable_caching:
            cache_key = generate_prompt_cache_key(sanitized_prompt, self.domain, prompt_type)
            cached_result = cache_manager.get(cache_key)
            if cached_result:
                log_cache_performance(logger, "prompt_improvement", True,
                                    domain=self.domain, prompt_type=prompt_type)
                return cached_result

        max_retries = getattr(settings, 'max_llm_retries', 3)
        retry_delay = getattr(settings, 'llm_retry_delay', 1.0)

        for attempt in range(max_retries + 1):
            try:
                logger.info(f"Improving {prompt_type} prompt in domain {self.domain} (attempt {attempt + 1})")

                input_data = {
                    "original_prompt": sanitized_prompt,
                    "prompt_type": prompt_type,
                    "key_topics": key_topics or []
                }

                result = await self.improvement_chain.ainvoke(input_data)

                # Cache the result if caching is enabled
                if perf_config.enable_caching:
                    cache_manager.set(cache_key, result, perf_config.cache_ttl)

                logger.info(f"Prompt improvement completed for {self.domain}")
                log_cache_performance(logger, "prompt_improvement", False,
                                    domain=self.domain, prompt_type=prompt_type)
                return result

            except Exception as e:
                error_msg = f"Error improving prompt in {self.domain}: {str(e)}"

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
                    raise ImprovementError(
                        f"Failed to improve prompt after {max_retries + 1} attempts",
                        domain=self.domain,
                        prompt_type=prompt_type,
                        original_prompt=original_prompt,
                        cause=e
                    )

        # Fallback: return degraded result if all retries failed
        logger.warning(f"Returning fallback result for {self.domain} due to repeated failures")
        return {
            "improved_prompt": original_prompt,  # Return original if improvement fails
            "improvements_made": [],
            "key_additions": [],
            "structure_analysis": "Improvement failed after multiple attempts",
            "effectiveness_score": 0.5,
            "reasoning": "Unable to improve prompt due to processing errors"
        }


    def _is_retryable_error(self, error: Exception) -> bool:
        return is_retryable_error(error)

    @abstractmethod
    def _define_expertise_areas(self) -> List[str]:
        """Define the specific areas of expertise for this domain."""
        pass

    @abstractmethod
    def _define_improvement_templates(self) -> Dict[str, str]:
        """Define improvement templates specific to this domain."""
        pass

    def _setup_improvement_chain(self):
        """Set up the LangChain for prompt improvement."""
        model_config = get_model_config()
        self.model = ChatGoogleGenerativeAI(
            model=model_config["model_name"],
            google_api_key=model_config["api_key"],
            temperature=0.4  # Slightly higher creativity for prompt improvement
        )

        # General improvement prompt template
        improvement_prompt = PromptTemplate.from_template("""
        You are an expert prompt engineer specializing in {domain}.

        DOMAIN EXPERTISE: {domain_description}
        EXPERTISE AREAS: {expertise_areas}

        ORIGINAL PROMPT:
        {original_prompt}

        PROMPT TYPE: {prompt_type}
        KEY TOPICS: {key_topics}

        PROMPT ENGINEERING PRINCIPLES:
        - **Clarity and Precision:** Leave no room for interpretation. Use precise language.
        - **Completeness:** Include all necessary context, constraints, and desired output formats.
        - **Structure:** Organize the prompt logically with clear sections.
        - **Actionability:** The prompt must be directly usable and lead to a high-quality result.
        - **Persona and Tone:** Define the persona the model should adopt and the tone of the response.

        TASK:
        Dramatically improve this prompt using your domain expertise and the principles above. Your goal is to create a world-class prompt that is ready for immediate use. Focus on:
        1. Injecting deep, domain-specific context and best practices.
        2. Eliminating all ambiguity and adding precise, actionable requirements.
        3. Re-structuring the prompt for maximum clarity and effectiveness.
        4. Specifying the desired output format, tone, and persona.
        5. Adding examples, edge cases, and constraints to guide the model.

        {improvement_instructions}

        Respond in JSON format with the following structure:
        {{
            "improved_prompt": "The fully improved and optimized prompt",
            "improvements_made": [
                "Specific improvement 1",
                "Specific improvement 2"
            ],
            "key_additions": [
                "Important context or requirements added",
                "Domain-specific best practices included"
            ],
            "structure_analysis": "Analysis of how the prompt was structured and improved",
            "effectiveness_score": 0.98,
            "reasoning": "Explanation of why these improvements will be effective"
        }}
        """)

        self.improvement_chain = (
            {
                "domain": lambda x: self.domain,
                "domain_description": lambda x: self.domain_description,
                "expertise_areas": lambda x: ", ".join(self.expertise_areas),
                "original_prompt": lambda x: x["original_prompt"],
                "prompt_type": lambda x: x["prompt_type"],
                "key_topics": lambda x: ", ".join(x.get("key_topics", [])),
                "improvement_instructions": lambda x: self.improvement_templates.get(
                    x["prompt_type"], self.improvement_templates.get("default", "")
                )
            }
            | improvement_prompt
            | self.model
            | JsonOutputParser()
        )

        # Set up context-enhanced improvement chain
        self._setup_context_improvement_chain()

    def _setup_context_improvement_chain(self):
        """Set up the LangChain for context-enhanced prompt improvement."""
        model_config = get_model_config()
        context_model = ChatGoogleGenerativeAI(
            model=model_config["model_name"],
            google_api_key=model_config["api_key"],
            temperature=0.2  # Lower temperature for context-aware improvements
        )

        # Context-enhanced improvement prompt template
        context_improvement_prompt = PromptTemplate.from_template("""
        You are an expert prompt engineer specializing in {domain}.

        DOMAIN EXPERTISE: {domain_description}
        EXPERTISE AREAS: {expertise_areas}

        ORIGINAL PROMPT:
        {original_prompt}

        PROMPT TYPE: {prompt_type}
        KEY TOPICS: {key_topics}

        ADDITIONAL CONTEXT:
        You have access to relevant context from previous interactions and knowledge:

        {context_parts}

        RAG METADATA:
        - Memories retrieved: {memories_count}
        - Knowledge entries: {knowledge_count}
        - Context length: {total_context_length} characters
        - Conversation turns: {conversation_turns}

        TASK:
        Improve this prompt using your domain expertise and the provided context. Focus on:
        1. Adding missing context and specificity based on previous interactions
        2. Removing ambiguity and unclear requirements
        3. Structuring the prompt for better results using learned patterns
        4. Adding relevant domain-specific best practices from knowledge base
        5. Leveraging conversation history for continuity
        6. Optimizing wording for clarity and effectiveness

        {improvement_instructions}

        Respond in JSON format with the following structure:
        {{
            "improved_prompt": "The fully improved and optimized prompt with context",
            "improvements_made": [
                "Specific improvement 1",
                "Specific improvement 2",
                "Context-based improvement"
            ],
            "key_additions": [
                "Important context or requirements added",
                "Domain-specific best practices included",
                "Conversation continuity maintained"
            ],
            "structure_analysis": "Analysis of how the prompt was structured and improved with context",
            "effectiveness_score": 0.95,
            "context_utilization": {{
                "memories_used": {memories_count},
                "knowledge_applied": {knowledge_count},
                "conversation_continuity": "high|medium|low"
            }},
            "reasoning": "Explanation of why these improvements will be effective, including context benefits"
        }}
        """)

        self.improvement_with_context_chain = (
            {
                "domain": lambda x: self.domain,
                "domain_description": lambda x: self.domain_description,
                "expertise_areas": lambda x: ", ".join(self.expertise_areas),
                "original_prompt": lambda x: x["original_prompt"],
                "prompt_type": lambda x: x["prompt_type"],
                "key_topics": lambda x: ", ".join(x.get("key_topics", [])),
                "context_parts": lambda x: self._format_context_parts(x["context"]["context_parts"]),
                "memories_count": lambda x: x["context"]["rag_metadata"]["memories_count"],
                "knowledge_count": lambda x: x["context"]["rag_metadata"]["knowledge_count"],
                "total_context_length": lambda x: x["context"]["rag_metadata"]["total_context_length"],
                "conversation_turns": lambda x: x["context"]["conversation_turns"],
                "improvement_instructions": lambda x: self.improvement_templates.get(
                    x["prompt_type"], self.improvement_templates.get("default", "")
                )
            }
            | context_improvement_prompt
            | context_model
            | JsonOutputParser()
        )

    def _format_context_parts(self, context_parts: List[Dict[str, Any]]) -> str:
        """Format context parts for inclusion in prompts."""
        if not context_parts:
            return "No additional context available."

        formatted_parts = []
        for i, part in enumerate(context_parts, 1):
            formatted_parts.append(f"""
[{i}] {part['type'].upper()} CONTEXT (Relevance: {part.get('relevance', 0):.2f})
{part['content']}
""")

        return "\n".join(formatted_parts)

    def get_domain_info(self) -> Dict[str, Any]:
        """Get information about this expert agent's domain and capabilities."""
        return {
            "domain": self.domain,
            "description": self.domain_description,
            "expertise_areas": self.expertise_areas,
            "supported_prompt_types": list(self.improvement_templates.keys())
        }


class SoftwareEngineeringExpert(BaseExpertAgent):
    """Expert agent for software engineering prompts."""

    def _define_expertise_areas(self) -> List[str]:
        return [
            "Code generation and optimization",
            "Algorithm design and implementation",
            "API design and documentation",
            "Database schema and queries",
            "Debugging and troubleshooting",
            "Code refactoring and best practices",
            "Software architecture patterns",
            "Testing strategies and frameworks"
        ]

    def _define_improvement_templates(self) -> Dict[str, str]:
        return {
            "default": """
            For software engineering prompts, ensure the final prompt is production-ready. It must include:
            - **Language and Framework:** Specify the exact programming language, version, and any required frameworks.
            - **Input/Output:** Define the precise format for inputs and outputs, including data structures and examples.
            - **Performance:** State clear performance constraints (e.g., time complexity, memory usage).
            - **Error Handling:** Detail how errors and edge cases should be managed.
            - **Code Style:** Enforce a specific code style (e.g., PEP 8, Google Style Guide) and documentation standard (e.g., JSDoc, Sphinx).
            - **Testing:** Mandate the inclusion of unit tests, integration tests, and validation criteria.
            - **Security:** Address potential security vulnerabilities and best practices.
            """,
            "raw": """
            This is a raw, unstructured prompt. Re-engineer it into a professional-grade prompt by:
            - **Defining a Persona:** Start with "You are an expert [Language] developer..."
            - **Structuring the Task:** Use clear headings like "## Task," "## Requirements," "## Constraints," and "## Output Format."
            - **Adding Technical Depth:** Inject specific library/framework requirements, version numbers, and architectural patterns.
            - **Providing Examples:** Include at least one clear example of input and expected output.
            """,
            "structured": """
            This is a semi-structured prompt. Elevate it to an exceptional standard by:
            - **Enhancing Specificity:** Replace vague terms with precise technical specifications.
            - **Incorporating Best Practices:** Add requirements for logging, monitoring, and configuration management.
            - **Considering Scalability:** Introduce considerations for how the solution will perform under load.
            - **Mandating Documentation:** Require detailed inline comments and a README file.
            """
        }


class DataScienceExpert(BaseExpertAgent):
    """Expert agent for data science prompts."""

    def _define_expertise_areas(self) -> List[str]:
        return [
            "Data analysis and visualization",
            "Machine learning model development",
            "Statistical analysis and interpretation",
            "Data preprocessing and feature engineering",
            "Model evaluation and validation",
            "Big data processing techniques",
            "Data storytelling and presentation",
            "Research methodology and experimental design"
        ]

    def _define_improvement_templates(self) -> Dict[str, str]:
        return {
            "default": """
            For data science prompts, ensure the final prompt is scientifically rigorous and ready for analysis. It must include:
            - **Data Source:** Specify the exact source, schema, and format of the data.
            - **Objective:** State a clear, measurable analytical objective and the key metrics for success.
            - **Methodology:** Define the required statistical methods, machine learning models, or analytical techniques.
            - **Data Quality:** Detail preprocessing steps, handling of missing values, and data validation criteria.
            - **Visualization:** Specify the types of visualizations and the format of the final report or dashboard.
            - **Reproducibility:** Mandate the use of seeds for random processes and clear documentation of the analysis steps.
            - **Ethical Considerations:** Address potential biases in the data and the ethical implications of the analysis.
            """,
            "raw": """
            This is a raw, unstructured data science prompt. Re-engineer it into a professional-grade analytical request by:
            - **Defining a Persona:** Start with "You are a senior data scientist..."
            - **Structuring the Analysis:** Use clear headings like "## Objective," "## Dataset," "## Methodology," and "## Deliverables."
            - **Adding Analytical Depth:** Inject specific statistical tests, model architectures, or feature engineering techniques.
            - **Providing a Hypothesis:** State a clear hypothesis to be tested or a question to be answered.
            """,
            "structured": """
            This is a semi-structured data science prompt. Elevate it to an exceptional standard by:
            - **Enhancing Rigor:** Add requirements for cross-validation, hyperparameter tuning, and model interpretability.
            - **Incorporating Business Context:** Frame the analysis within a broader business problem or goal.
            - **Considering Deployment:** Introduce considerations for model deployment, monitoring, and maintenance.
            - **Mandating a Narrative:** Require the final output to be a compelling data story, not just a set of charts.
            """
        }


# Registry of available expert agents
EXPERT_AGENT_REGISTRY = {
    "software_engineering": SoftwareEngineeringExpert,
    "data_science": DataScienceExpert,
}


def create_expert_agent(domain: str, description: str = None) -> BaseExpertAgent:
    """
    Factory function to create an expert agent for a domain.

    Args:
        domain: The domain to create an agent for
        description: Optional description of the domain

    Returns:
        An instance of the appropriate expert agent
    """
    if domain in EXPERT_AGENT_REGISTRY:
        agent_class = EXPERT_AGENT_REGISTRY[domain]
        return agent_class(domain, description or f"Expert in {domain}")

    # For unknown domains, create a generic expert agent
    return GenericExpertAgent(domain, description or f"Expert in {domain}")


class GenericExpertAgent(BaseExpertAgent):
    """Generic expert agent for unknown domains."""

    def _define_expertise_areas(self) -> List[str]:
        return [
            "General prompt optimization",
            "Clarity and specificity improvement",
            "Structure and organization enhancement",
            "Context and detail addition",
            "Best practices application"
        ]

    def _define_improvement_templates(self) -> Dict[str, str]:
        return {
            "default": """
            For general prompts, ensure the final prompt is comprehensive and unambiguous. It must include:
            - **Objective:** A crystal-clear statement of the desired outcome.
            - **Context:** All relevant background information required to complete the task.
            - **Audience and Tone:** Specify the target audience and the desired tone of the response.
            - **Constraints:** List any limitations, constraints, or negative requirements.
            - **Output Format:** Define the exact structure and format of the expected output.
            - **Success Criteria:** Provide clear, measurable criteria for what constitutes a successful response.
            """,
            "raw": """
            This is a raw, unstructured prompt. Re-engineer it into a high-quality, structured prompt by:
            - **Defining a Persona:** Start with "You are a helpful and knowledgeable assistant..."
            - **Structuring the Request:** Use clear headings like "## Goal," "## Context," "## Requirements," and "## Deliverable."
            - **Adding Specificity:** Replace generalities with specific details and examples.
            - **Clarifying the 'Why':** Briefly explain the purpose behind the request to provide better context.
            """,
            "structured": """
            This is a semi-structured prompt. Elevate it to an exceptional standard by:
            - **Enhancing Clarity:** Rephrase any ambiguous sentences for maximum clarity.
            - **Providing Examples:** Add a clear example of what is expected.
            - **Adding Negative Constraints:** Specify what should *not* be included in the response.
            - **Mandating a Review:** Ask the model to review its own response against the requirements before finalizing.
            """
        }
