"""Base Expert Prompt Engineer Agent framework."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnablePassthrough
from config.llm_providers import get_llm
from agents.exceptions import ImprovementError, LLMServiceError, ConfigurationError
from agents.utils import is_retryable_error
import asyncio

from config.config import (
    settings, get_logger, metrics, log_performance,
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
        self.model = get_llm(temperature=0.4)

        # General improvement prompt template — compact yet comprehensive
        improvement_prompt = PromptTemplate.from_template("""You are an **elite prompt engineer** specialising in **{domain}**.

DOMAIN: {domain_description}
EXPERTISE: {expertise_areas}

ORIGINAL PROMPT:
{original_prompt}

METADATA: type={prompt_type} | topics={key_topics}

━━━ CHAIN-OF-THOUGHT METHODOLOGY ━━━

**Phase 1 — DIAGNOSE** (internal reasoning):
• What is ambiguous, vague, or missing? (constraints, edge-cases, output format, persona, success criteria)
• Executor view: What questions would I ask before starting?
• Critic view: Where could the output go wrong or be misinterpreted?
• Expert view: What {domain} standards/frameworks are missing?

**Phase 2 — PLAN**: Map each weakness to a specific fix. Decide structure, persona, and examples.

**Phase 3 — EXECUTE**: Write the improved prompt using Chain-of-Density (broad → progressively add constraints and specifics).

**Phase 4 — SELF-CRITIQUE**: Re-read and verify:
• Could two people interpret this differently? → Fix it.
• Anything assumed? → Make explicit.
• Could I execute immediately without questions? → Add detail.
• Score ≥ 0.90 on clarity, specificity, structure, completeness, actionability, domain alignment? → If not, iterate.

━━━ PROMPT-ENGINEERING PRINCIPLES (apply ALL) ━━━
1. **Role Anchoring** — Open with vivid expert persona
2. **Clarity** — Concrete metrics/numbers, no vague qualifiers ("good", "some", "etc.")
3. **Structure** — ## headings, numbered steps, bullet lists, grouped sections
4. **Completeness** — All context, constraints, output format, success criteria, edge-cases
5. **Actionability** — Strong verbs, measurable deliverables
6. **Few-Shot** — ≥1 concrete input→output example (MANDATORY for complex tasks)
7. **Negative Constraints** — Explicitly state what to AVOID
8. **Output Format** — Define exact response structure
9. **Domain Practices** — {domain} conventions, terminology, standards
10. **Meta-Verification** — End with self-check instruction
11. **Scope Boundaries** — Define IN/OUT scope
12. **Quality Gates** — Explicit acceptance criteria

ANTI-PATTERNS TO REMOVE:
• "please", "kindly", filler words → precise imperatives
• "etc.", "and so on" → enumerate explicitly
• "make it good" → measurable criteria
• Redundancy → compress; Implicit assumptions → make explicit or remove

{improvement_instructions}

━━━ OUTPUT (strict JSON — no markdown fences) ━━━
{{
    "improved_prompt": "<the fully rewritten prompt>",
    "improvements_made": ["<specific change 1>", "..."],
    "key_additions": ["<important context/requirement added>", "..."],
    "structure_analysis": "<how the prompt structure was improved>",
    "effectiveness_score": <float 0.0-1.0>,
    "reasoning": "<why these changes make the prompt more effective>"
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
        context_model = get_llm(temperature=0.2)

        # Context-enhanced improvement prompt template — compact
        context_improvement_prompt = PromptTemplate.from_template("""You are an **elite prompt engineer** specialising in **{domain}**.

DOMAIN: {domain_description}
EXPERTISE: {expertise_areas}

ORIGINAL PROMPT:
{original_prompt}

METADATA: type={prompt_type} | topics={key_topics}

━━━ RAG & CONVERSATION CONTEXT ━━━
{context_parts}

RAG stats: {memories_count} memories · {knowledge_count} knowledge entries ·
           {total_context_length} chars · {conversation_turns} conversation turns

━━━ CHAIN-OF-THOUGHT ━━━

**Phase 1 — CONTEXT MINING**: Extract useful signals from RAG context:
• Past interactions revealing user's style, preferences, or pain points
• Knowledge entries with domain conventions to embed
• Conversation continuity to maintain

**Phase 2 — DIAGNOSE**: Ambiguities, missing context, unclear constraints, absent output format. Gaps RAG can fill.

**Phase 3 — EXECUTE**: Write improved prompt:
• Inject retrieved knowledge directly (don't just reference it)
• Apply all prompt-engineering principles below
• Structure for maximum skim-ability and precision

**Phase 4 — SELF-CRITIQUE**: Verify every relevant RAG insight incorporated, prompt is self-contained, no ambiguity remains.

━━━ PRINCIPLES (apply ALL) ━━━
1. Role Anchoring — vivid expert persona
2. Clarity — concrete metrics, no vague qualifiers
3. Structure — ## headings, numbered steps, bullet lists
4. Completeness — all context, constraints, output format, success criteria
5. Actionability — strong verbs, measurable deliverables
6. Few-Shot — input/output examples when valuable
7. Negative Constraints — state what to AVOID
8. Output Format — define exact response structure
9. Domain Practices — professional {domain} conventions
10. Meta-Verification — end with self-check instruction

{improvement_instructions}

━━━ OUTPUT (strict JSON — no markdown fences) ━━━
{{
    "improved_prompt": "<the fully rewritten prompt>",
    "improvements_made": ["<specific change>", "..."],
    "key_additions": ["<context-based addition>", "..."],
    "structure_analysis": "<how structure was improved with context>",
    "effectiveness_score": <float 0.0-1.0>,
    "context_utilization": {{
        "memories_used": {memories_count},
        "knowledge_applied": {knowledge_count},
        "conversation_continuity": "high|medium|low"
    }},
    "reasoning": "<why these changes help, including context benefits>"
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
            For software engineering prompts, ensure the final prompt is production-ready and uses
            state-of-the-art prompt engineering techniques:
            - **Language and Framework:** Specify the exact programming language, version, and required frameworks.
            - **Input/Output Contracts:** Define the precise format for inputs and outputs with typed examples.
            - **Performance Requirements:** State clear performance constraints (time complexity, memory, latency).
            - **Error Handling & Edge Cases:** Detail how errors, boundary conditions, and failures are managed.
            - **Code Style & Standards:** Enforce a specific code style (PEP 8, Google Style Guide) and docs standard.
            - **Testing Mandate:** Require unit tests, integration tests, test coverage targets, and test data.
            - **Security Considerations:** Address OWASP Top 10 or relevant security patterns.
            - **Few-Shot Example:** Include at least one concrete input → expected output example.
            - **Negative Constraints:** Explicitly state anti-patterns and approaches to avoid.
            - **Self-Verification Step:** End with "Before submitting, verify: [checklist]."
            """,
            "raw": """
            This is a raw, unstructured prompt. Apply full prompt engineering transformation:
            - **Role Anchor:** Open with "You are a principal software engineer at a top tech company with 15+ years
              of experience in [relevant tech stack]..."
            - **Structured Layout:** Use clear headings: ## Context, ## Task, ## Requirements, ## Constraints,
              ## Output Format, ## Examples, ## What NOT To Do, ## Verification Checklist.
            - **Technical Depth:** Inject specific library/framework requirements, version numbers, architectural
              patterns (e.g., "Use repository pattern with dependency injection").
            - **Concrete Example:** Include a worked input/output example that demonstrates expected quality.
            - **Chain-of-Thought Trigger:** Add "Think through your approach step-by-step before coding."
            - **Negative Constraints:** Add "Do NOT: use deprecated APIs, skip error handling, hardcode values."
            """,
            "structured": """
            This is a semi-structured prompt. Elevate it to an exceptional standard:
            - **Enhance Specificity:** Replace vague terms with precise technical specifications and numeric targets.
            - **Incorporate Production Practices:** Add requirements for logging, monitoring, config management,
              graceful degradation, and observability.
            - **Scalability Section:** Introduce considerations for load, concurrency, and horizontal scaling.
            - **Documentation Mandate:** Require inline comments, docstrings, README, and architecture decision records.
            - **Self-Review Gate:** End with "Before finalizing, confirm: all edge cases handled, tests pass,
              no hardcoded secrets, code is idiomatic [language]."
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


class ReportWritingExpert(BaseExpertAgent):
    """Expert agent for report writing and documentation prompts."""

    def _define_expertise_areas(self) -> List[str]:
        return [
            "Executive summary composition",
            "Technical documentation authoring",
            "Business report structuring",
            "Findings and recommendations writing",
            "Data-driven narrative construction",
            "Stakeholder communication",
            "Academic and research report formatting",
            "Presentation deck scripting"
        ]

    def _define_improvement_templates(self) -> Dict[str, str]:
        return {
            "default": """
            For report writing prompts, ensure the final prompt produces a publication-ready document. It must include:
            - **Purpose and Audience:** Specify who will read the report and what decisions it should inform.
            - **Structure:** Define the report sections (Executive Summary, Introduction, Methodology, Findings, Recommendations, Appendices).
            - **Data Requirements:** List the data sources, metrics, and KPIs to reference.
            - **Tone and Formality:** Specify the writing style (e.g., formal business, academic, journalistic).
            - **Visuals:** Mandate inclusion of charts, tables, or infographics where appropriate.
            - **Length and Format:** Set page limits, citation style, and formatting requirements.
            """,
            "raw": """
            This is a raw report writing prompt. Transform it into a professional content brief by:
            - **Defining a Persona:** Start with "You are a senior business analyst preparing a report for..."
            - **Structuring Sections:** Use headings like "## Report Objective," "## Target Audience," "## Required Sections," "## Deliverable Format."
            - **Adding Specificity:** Include exact metrics, timeframes, and comparison criteria.
            - **Providing a Template:** Outline the expected table of contents.
            """,
            "structured": """
            This is a semi-structured report prompt. Enhance it by:
            - **Adding Stakeholder Context:** Clarify who the decision-makers are and what they need.
            - **Mandating Evidence:** Require data citations and sourcing for every claim.
            - **Including Actionable Recommendations:** Each finding should map to a specific recommendation.
            - **Requiring an Executive Summary:** Mandate a standalone summary that captures all key points.
            """
        }


class EducationExpert(BaseExpertAgent):
    """Expert agent for educational content and teaching prompts."""

    def _define_expertise_areas(self) -> List[str]:
        return [
            "Curriculum design and lesson planning",
            "Learning objective formulation (Bloom's taxonomy)",
            "Assessment and rubric creation",
            "Differentiated instruction strategies",
            "Interactive and engaging content design",
            "Tutorial and explainer writing",
            "Student engagement techniques",
            "Educational technology integration"
        ]

    def _define_improvement_templates(self) -> Dict[str, str]:
        return {
            "default": """
            For education prompts, ensure the final prompt produces pedagogically sound content. It must include:
            - **Learning Objectives:** Use Bloom's Taxonomy verbs (Understand, Apply, Analyze, Evaluate, Create).
            - **Target Audience:** Specify grade level, skill level, or learner profile.
            - **Content Structure:** Organize into Introduction, Core Concepts, Examples, Practice Activities, Assessment.
            - **Engagement:** Include interactive elements, analogies, real-world scenarios, or discussion questions.
            - **Assessment:** Define how mastery will be measured (quiz, project, rubric criteria).
            - **Accessibility:** Ensure content accommodates different learning styles (visual, auditory, kinesthetic).
            """,
            "raw": """
            This is a raw educational prompt. Transform it into a structured lesson plan by:
            - **Defining a Persona:** Start with "You are an experienced educator specializing in..."
            - **Using Bloom's Taxonomy:** Frame objectives with specific cognitive levels.
            - **Adding Scaffolding:** Break complex topics into progressive, manageable steps.
            - **Including Examples:** Provide at least two worked examples for each concept.
            """,
            "structured": """
            This is a semi-structured education prompt. Enhance it by:
            - **Adding Differentiation:** Include tiered activities for beginner, intermediate, and advanced learners.
            - **Incorporating Assessment:** Add formative checks and a summative assessment with rubric.
            - **Mandating Engagement Hooks:** Require an opening hook, mid-lesson activity, and closing reflection.
            - **Requiring Real-World Connections:** Link each concept to a practical, real-world application.
            """
        }


class BusinessStrategyExpert(BaseExpertAgent):
    """Expert agent for business strategy and management prompts."""

    def _define_expertise_areas(self) -> List[str]:
        return [
            "Strategic planning and competitive analysis",
            "Market research and positioning",
            "Business model innovation",
            "Financial analysis and forecasting",
            "Go-to-market strategy development",
            "Organizational design and change management",
            "Risk assessment and mitigation",
            "Growth strategy and scaling"
        ]

    def _define_improvement_templates(self) -> Dict[str, str]:
        return {
            "default": """
            For business strategy prompts, ensure the final prompt produces actionable strategic insights. It must include:
            - **Business Context:** Specify industry, company size, market position, and competitive landscape.
            - **Strategic Objective:** Define the specific business goal (growth, efficiency, market entry, etc.).
            - **Analytical Frameworks:** Mandate use of frameworks (SWOT, Porter's Five Forces, PESTEL, Value Chain).
            - **Data Requirements:** Specify market data, financial metrics, and KPIs needed.
            - **Deliverable Format:** Define the output (strategy document, presentation, one-pager, financial model).
            - **Timeline and Constraints:** Set implementation timeline, budget constraints, and risk tolerance.
            """,
            "raw": """
            This is a raw business strategy prompt. Transform it into a professional strategic brief by:
            - **Defining a Persona:** Start with "You are a management consultant at a top-tier firm..."
            - **Structuring the Analysis:** Use headings like "## Strategic Objective," "## Market Analysis," "## Options," "## Recommendation."
            - **Adding Quantitative Rigor:** Include revenue targets, market size estimates, and ROI projections.
            - **Mandating Frameworks:** Require at least two analytical frameworks in the analysis.
            """,
            "structured": """
            This is a semi-structured business prompt. Enhance it by:
            - **Adding Competitive Intelligence:** Require analysis of top 3-5 competitors.
            - **Including Scenario Planning:** Mandate best-case, base-case, and worst-case scenarios.
            - **Requiring Actionable Recommendations:** Each insight must map to a specific action item with owner and timeline.
            - **Mandating Risk Assessment:** Include a risk matrix with mitigation strategies.
            """
        }


class CreativeWritingExpert(BaseExpertAgent):
    """Expert agent for creative writing and content creation prompts."""

    def _define_expertise_areas(self) -> List[str]:
        return [
            "Narrative structure and storytelling",
            "Character development and dialogue",
            "Content marketing and copywriting",
            "Creative ideation and brainstorming",
            "Tone and voice calibration",
            "Audience engagement techniques",
            "SEO-optimized content creation",
            "Multi-format content adaptation"
        ]

    def _define_improvement_templates(self) -> Dict[str, str]:
        return {
            "default": """
            For creative writing prompts, ensure the final prompt inspires high-quality, engaging output. It must include:
            - **Genre and Style:** Specify the genre, tone, and literary style (e.g., persuasive, narrative, poetic).
            - **Audience:** Define the target reader persona and their expectations.
            - **Structure:** Outline the desired structure (hook, body, conclusion) or narrative arc.
            - **Voice and Tone:** Provide specific adjectives for the desired voice (e.g., witty, authoritative, empathetic).
            - **Constraints:** Set word count, format, and any content restrictions.
            - **Inspiration:** Provide reference examples or comparable works for style guidance.
            """,
            "raw": """
            This is a raw creative prompt. Transform it into a detailed creative brief by:
            - **Defining a Persona:** Start with "You are an award-winning writer specializing in..."
            - **Setting the Scene:** Provide world-building details, character sketches, or brand context.
            - **Adding Sensory Details:** Require vivid descriptions that appeal to multiple senses.
            - **Including Examples:** Reference specific works or writers as style benchmarks.
            """,
            "structured": """
            This is a semi-structured creative prompt. Enhance it by:
            - **Deepening Character/Brand Voice:** Add detailed persona traits and communication preferences.
            - **Adding Emotional Arc:** Specify the emotional journey the reader should experience.
            - **Mandating a Hook:** Require an opening that immediately captures attention.
            - **Requiring Revision Criteria:** Define what "excellent" looks like with specific quality markers.
            """
        }


# Registry of available expert agents
EXPERT_AGENT_REGISTRY = {
    "software_engineering": SoftwareEngineeringExpert,
    "data_science": DataScienceExpert,
    "report_writing": ReportWritingExpert,
    "education": EducationExpert,
    "business_strategy": BusinessStrategyExpert,
    "creative_writing": CreativeWritingExpert,
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
            "Best practices application",
            "Chain-of-Thought reasoning",
            "Meta-prompting techniques"
        ]

    def _define_improvement_templates(self) -> Dict[str, str]:
        return {
            "default": """
            For general prompts, apply systematic prompt engineering to make the output
            comprehensive, unambiguous, and immediately executable:
            - **Objective:** A crystal-clear statement of the desired outcome with measurable success criteria.
            - **Context:** All relevant background information — the reader should never need to ask for more.
            - **Persona & Tone:** Define who the model should be and the exact voice/register to use.
            - **Constraints:** List all limitations, negative requirements ("Do NOT..."), and scope boundaries.
            - **Output Format:** Define the exact structure (headings, code blocks, tables, JSON, etc.).
            - **Examples:** Provide at least one input → output example to anchor expected quality.
            - **Edge Cases:** Explicitly address boundary conditions, ambiguous scenarios, and fallbacks.
            - **Verification:** End with a self-check instruction: "Before finalizing, verify all requirements are met."
            """,
            "raw": """
            This is a raw, unstructured prompt. Apply a full prompt engineering transformation:
            - **Role Anchor:** Start with a vivid persona: "You are a world-class expert in..."
            - **Structured Layout:** Use clear headings: ## Goal, ## Context, ## Requirements,
              ## Constraints, ## Output Format, ## Examples, ## What NOT To Do.
            - **Specificity Injection:** Replace every vague phrase with a concrete, measurable detail.
            - **Chain-of-Thought:** Add "Think step-by-step before producing your answer."
            - **Negative Constraints:** Add "Do NOT: provide generic filler, make assumptions not stated,
              skip edge cases, or produce incomplete output."
            - **Self-Verification:** End with "Before submitting, verify: [specific checklist items]."
            """,
            "structured": """
            This is a semi-structured prompt. Elevate it to an exceptional standard:
            - **Enhance Clarity:** Rephrase any ambiguous sentence so two readers would interpret identically.
            - **Add Examples:** Include a concrete worked example of expected input and output.
            - **Negative Constraints:** Specify what should *not* be included in the response.
            - **Self-Review Gate:** Ask the model to review its response against all requirements before finalizing.
            - **Meta-Verification:** Add "Before finalizing: Does this fully address the goal? Are all constraints
              met? Would the requester need to ask any follow-up questions?"
            """
        }
