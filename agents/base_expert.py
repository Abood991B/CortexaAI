"""Base Expert Prompt Engineer Agent framework."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI
from config.config import settings, get_model_config
import logging

# Set up logging
logging.basicConfig(level=settings.log_level)
logger = logging.getLogger(__name__)


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
            temperature=0.3  # Slightly higher creativity for prompt improvement
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

        TASK:
        Improve this prompt using your domain expertise. Focus on:
        1. Adding missing context and specificity
        2. Removing ambiguity and unclear requirements
        3. Structuring the prompt for better results
        4. Adding relevant domain-specific best practices
        5. Optimizing wording for clarity and effectiveness

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
            "effectiveness_score": 0.95,
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

    def improve_prompt(self, original_prompt: str, prompt_type: str = "raw",
                      key_topics: List[str] = None) -> Dict[str, Any]:
        """
        Improve a prompt using domain expertise.

        Args:
            original_prompt: The original prompt to improve
            prompt_type: Type of prompt ("raw" or "structured")
            key_topics: Key topics identified by classifier

        Returns:
            Dict containing the improved prompt and analysis
        """
        try:
            logger.info(f"Improving {prompt_type} prompt in domain {self.domain}")

            input_data = {
                "original_prompt": original_prompt,
                "prompt_type": prompt_type,
                "key_topics": key_topics or []
            }

            result = self.improvement_chain.invoke(input_data)

            logger.info(f"Prompt improvement completed for {self.domain}")
            return result

        except Exception as e:
            logger.error(f"Error improving prompt in {self.domain}: {e}")
            return {
                "improved_prompt": original_prompt,  # Return original if improvement fails
                "improvements_made": [],
                "key_additions": [],
                "structure_analysis": f"Improvement failed: {str(e)}",
                "effectiveness_score": 0.5,
                "reasoning": "Unable to improve prompt due to processing error"
            }

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
            For software engineering prompts, ensure to include:
            - Specific programming language requirements
            - Clear input/output specifications
            - Performance and efficiency considerations
            - Error handling requirements
            - Code style and documentation standards
            - Testing and validation criteria
            """,
            "raw": """
            This appears to be a raw, unstructured prompt. Structure it by adding:
            - Clear problem statement
            - Specific technical requirements
            - Expected deliverables and format
            - Constraints and edge cases to consider
            - Quality criteria for the solution
            """,
            "structured": """
            This appears to be a semi-structured prompt. Enhance it by adding:
            - Missing technical specifications
            - Best practices for the specific technology
            - Performance and scalability considerations
            - Security implications if applicable
            - Testing and validation requirements
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
            For data science prompts, ensure to include:
            - Clear data sources and formats
            - Specific analysis objectives and metrics
            - Model performance requirements
            - Data quality and preprocessing needs
            - Visualization and reporting requirements
            - Statistical methodology specifications
            """,
            "raw": """
            This appears to be a raw data science prompt. Structure it by adding:
            - Specific dataset characteristics and sources
            - Clear analytical objectives and success criteria
            - Required methodologies and techniques
            - Expected outputs and deliverables
            - Data quality and validation requirements
            """,
            "structured": """
            This appears to be a semi-structured data science prompt. Enhance it by adding:
            - Missing data specifications and constraints
            - Statistical rigor and validation methods
            - Computational efficiency considerations
            - Reproducibility and documentation requirements
            - Ethical considerations for data usage
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
            For general prompts, focus on:
            - Adding specific context and requirements
            - Improving clarity and reducing ambiguity
            - Structuring for better understanding
            - Adding relevant constraints and criteria
            - Optimizing for the intended audience and purpose
            """,
            "raw": """
            This appears to be a raw, unstructured prompt. Structure it by adding:
            - Clear objectives and goals
            - Specific requirements and constraints
            - Expected outcomes and deliverables
            - Success criteria and evaluation methods
            - Relevant context and background information
            """,
            "structured": """
            This appears to be a semi-structured prompt. Enhance it by adding:
            - Missing details and specifications
            - Clearer success criteria
            - Better organization and flow
            - Additional context if needed
            - Quality and completeness improvements
            """
        }
