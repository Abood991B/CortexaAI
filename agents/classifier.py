"""Classifier Agent for Multi-Agent Prompt Engineering System."""

from typing import Dict, List, Optional, Any
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI
from config.config import settings, get_model_config
import json
import logging

# Set up logging
logging.basicConfig(level=settings.log_level)
logger = logging.getLogger(__name__)


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
                "keywords": ["business", "strategy", "marketing", "management", "planning", "growth", "market", "competitive", "analysis"],
                "description": "Business strategy, planning, and management tasks"
            }
        }

        self.created_agents = {}  # Track dynamically created agents
        self._setup_classifier_chain()

    def _setup_classifier_chain(self):
        """Set up the LangChain for domain classification."""
        model_config = get_model_config()
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

    def classify_prompt(self, prompt: str) -> Dict[str, Any]:
        """
        Classify a prompt into a domain.

        Args:
            prompt: The prompt to classify

        Returns:
            Dict containing classification results
        """
        try:
            logger.info(f"Classifying prompt: {prompt[:100]}...")

            result = self.classifier_chain.invoke(prompt)

            # Handle new domain creation
            if result.get("is_new_domain", False):
                new_domain = result["new_domain_name"]
                new_description = result["new_domain_description"]

                logger.info(f"Creating new domain: {new_domain}")
                self._create_new_domain(new_domain, new_description, result.get("key_topics", []))

                # Update the result to use the new domain
                result["domain"] = new_domain

            logger.info(f"Classification result: {result}")
            return result

        except ValueError as ve:
            if "No generations found in stream" in str(ve):
                logger.error(f"Model streaming error: {ve}. This may be due to an invalid model name, API key, or model availability.")
                return {
                    "domain": "general",
                    "confidence": 0.5,
                    "is_new_domain": False,
                    "key_topics": [],
                    "reasoning": f"Model error: {str(ve)}. Please check the model name '{self.classifier_chain.steps[1].model}' and API key."
                }
            else:
                raise  # Re-raise if it's a different ValueError

        except Exception as e:
            logger.error(f"Error classifying prompt: {e}")
            return {
                "domain": "general",
                "confidence": 0.5,
                "is_new_domain": False,
                "key_topics": [],
                "reasoning": f"Classification failed: {str(e)}"
            }

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
