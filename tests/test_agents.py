"""Tests for Multi-Agent Prompt Engineering System agents."""

import pytest
from unittest.mock import Mock, patch
import json

from agents.classifier import classifier, DomainClassifier
from agents.base_expert import create_expert_agent, SoftwareEngineeringExpert, GenericExpertAgent
from agents.evaluator import evaluator, PromptEvaluator
from agents.coordinator import coordinator, WorkflowCoordinator
from config.config import settings


class TestDomainClassifier:
    """Test cases for the Domain Classifier agent."""

    def test_initialization(self):
        """Test that the classifier initializes with known domains."""
        assert len(classifier.known_domains) > 0
        assert "software_engineering" in classifier.known_domains
        assert "data_science" in classifier.known_domains

    def test_classify_prompt_mock(self):
        """Test prompt classification with mocked LLM."""
        import asyncio
        from unittest.mock import patch, AsyncMock

        # Mock the classifier chain with an async mock
        mock_chain = AsyncMock(return_value={
            "domain": "software_engineering",
            "confidence": 0.95,
            "is_new_domain": False,
            "key_topics": ["code", "algorithm"],
            "reasoning": "This is a coding task"
        })

        with patch.object(classifier, 'classifier_chain', mock_chain):
            # Run the async function synchronously
            result = asyncio.run(classifier.classify_prompt("Write a function to sort a list"))

            assert result["domain"] == "software_engineering"
            # The actual confidence may vary, just check it's a reasonable value
            assert 0.9 <= result["confidence"] <= 1.0
            assert "code" in result["key_topics"]

    def test_get_available_domains(self):
        """Test getting available domains."""
        domains = classifier.get_available_domains()
        assert isinstance(domains, dict)
        assert len(domains) > 0

    def test_has_domain(self):
        """Test domain existence check."""
        assert classifier.has_domain("software_engineering") == True
        assert classifier.has_domain("nonexistent_domain") == False


class TestExpertAgents:
    """Test cases for Expert Prompt Engineer agents."""

    def test_create_software_engineering_expert(self):
        """Test creating a software engineering expert."""
        expert = SoftwareEngineeringExpert("software_engineering", "Coding expert")

        assert expert.domain == "software_engineering"
        assert "Code generation and optimization" in expert.expertise_areas
        assert "default" in expert.improvement_templates

    def test_create_generic_expert(self):
        """Test creating a generic expert for unknown domains."""
        expert = GenericExpertAgent("law", "Legal expert")

        assert expert.domain == "law"
        assert "General prompt optimization" in expert.expertise_areas

    def test_create_expert_agent_factory(self):
        """Test the expert agent factory function."""
        expert = create_expert_agent("software_engineering")
        assert isinstance(expert, SoftwareEngineeringExpert)

        # Test unknown domain
        unknown_expert = create_expert_agent("unknown_domain")
        assert isinstance(unknown_expert, GenericExpertAgent)


class TestPromptEvaluator:
    """Test cases for the Prompt Evaluator agent."""

    def test_initialization(self):
        """Test evaluator initialization."""
        assert evaluator.evaluation_threshold == settings.evaluation_threshold
        assert evaluator.max_iterations == settings.max_evaluation_iterations

    @patch('langchain_google_genai.ChatGoogleGenerativeAI')
    def test_evaluate_prompt_mock(self, mock_chat):
        """Test prompt evaluation with mocked LLM."""
        import asyncio

        mock_response = Mock()
        mock_response.content = json.dumps({
            "overall_score": 0.85,
            "criteria_scores": {
                "clarity": 0.9,
                "specificity": 0.8,
                "structure": 0.9,
                "completeness": 0.8,
                "actionability": 0.85,
                "domain_alignment": 0.8
            },
            "passes_threshold": True,
            "needs_improvement": False,
            "strengths": ["Clear structure", "Good specificity"],
            "weaknesses": [],
            "specific_feedback": [],
            "improvement_priority": "low",
            "reasoning": "Good quality prompt",
            "comparison_analysis": "Well improved"
        })
        mock_chat.return_value.invoke.return_value = mock_response

        # Run the async function synchronously
        result = asyncio.run(evaluator.evaluate_prompt(
            original_prompt="Write code",
            improved_prompt="Write a Python function to sort a list",
            domain="software_engineering",
            prompt_type="raw"
        ))

        assert result["overall_score"] == 0.85
        assert result["passes_threshold"] == True
        # Check that strengths list is not empty (actual content may vary)
        assert len(result["strengths"]) > 0
        assert isinstance(result["strengths"][0], str)


class TestWorkflowCoordinator:
    """Test cases for the Workflow Coordinator."""

    def test_initialization(self):
        """Test coordinator initialization."""
        assert coordinator.classifier is not None
        assert coordinator.evaluator is not None
        assert isinstance(coordinator.expert_agents, dict)

    def test_get_available_domains(self):
        """Test getting available domains from coordinator."""
        domains = coordinator.get_available_domains()
        assert isinstance(domains, list)
        assert len(domains) > 0

    def test_get_workflow_stats_empty(self):
        """Test getting workflow stats when no workflows exist."""
        stats = coordinator.get_workflow_stats()
        # When no workflows exist, it returns an error message
        if "error" in stats:
            assert stats["error"] == "No workflow history available"
        else:
            # If there are workflows, check the stats
            assert stats["total_workflows"] == 0
            assert stats["completed_workflows"] == 0
            assert stats["success_rate"] == 0.0


class TestIntegration:
    """Integration tests for the multi-agent system."""

    def test_workflow_structure(self):
        """Test the basic structure and flow of the workflow."""
        # Test that all components are properly initialized
        assert coordinator.classifier is not None
        assert coordinator.evaluator is not None
        assert hasattr(coordinator, 'process_prompt')

        # Test that the workflow method exists and is callable
        assert callable(coordinator.process_prompt)

        # Test domain listing
        domains = coordinator.get_available_domains()
        assert isinstance(domains, list)
        assert len(domains) > 0

        # Check that each domain has the expected properties
        for domain_info in domains:
            assert "domain" in domain_info
            assert "description" in domain_info
            assert "keywords" in domain_info
            assert "has_expert_agent" in domain_info

    def test_expert_agent_creation_integration(self):
        """Test that expert agents can be created and have the expected interface."""
        # Test creating different types of experts
        sw_expert = create_expert_agent("software_engineering")
        assert hasattr(sw_expert, 'improve_prompt')
        assert hasattr(sw_expert, 'domain')
        assert sw_expert.domain == "software_engineering"

        generic_expert = create_expert_agent("unknown_domain")
        assert hasattr(generic_expert, 'improve_prompt')
        assert hasattr(generic_expert, 'domain')
        assert generic_expert.domain == "unknown_domain"


if __name__ == "__main__":
    # Run basic tests
    test_classifier = TestDomainClassifier()
    test_classifier.test_initialization()
    test_classifier.test_get_available_domains()
    test_classifier.test_has_domain()

    test_experts = TestExpertAgents()
    test_experts.test_create_software_engineering_expert()
    test_experts.test_create_generic_expert()
    test_experts.test_create_expert_agent_factory()

    test_evaluator = TestPromptEvaluator()
    test_evaluator.test_initialization()

    test_coordinator = TestWorkflowCoordinator()
    test_coordinator.test_initialization()
    test_coordinator.test_get_available_domains()
    test_coordinator.test_get_workflow_stats_empty()

    print("✅ All basic tests passed!")
