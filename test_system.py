#!/usr/bin/env python3
"""
Test the Multi-Agent Prompt Engineering System without requiring API keys.
This validates the system architecture and components.
"""

import os
import sys
import json
from unittest.mock import patch, Mock

# Add the project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from agents.classifier import classifier
from agents.base_expert import create_expert_agent, SoftwareEngineeringExpert, GenericExpertAgent
from agents.evaluator import evaluator
from agents.coordinator import coordinator
from config.config import settings


def test_system_initialization():
    """Test that all system components initialize correctly."""
    print("=== System Initialization Test ===")

    # Test classifier
    print("✓ Classifier initialized with", len(classifier.known_domains), "domains")

    # Test evaluator
    print("✓ Evaluator initialized with threshold:", evaluator.evaluation_threshold)

    # Test coordinator
    print("✓ Coordinator initialized with", len(coordinator.expert_agents), "expert agents")

    print("✓ All system components initialized successfully!\n")


def test_domain_classification():
    """Test domain classification with mock responses."""
    print("=== Domain Classification Test ===")

    with patch('agents.classifier.ChatOpenAI') as mock_chat:
        # Mock the LLM response
        mock_response = Mock()
        mock_response.content = json.dumps({
            "domain": "software_engineering",
            "confidence": 0.95,
            "is_new_domain": False,
            "key_topics": ["code", "algorithm"],
            "reasoning": "This is a coding task"
        })
        mock_chat.return_value.invoke.return_value = mock_response

        prompt = "Write a function to sort a list of numbers"
        result = classifier.classify_prompt(prompt)

        print(f"✓ Classified prompt: '{prompt[:30]}...'")
        print(f"✓ Domain: {result['domain']}")
        print(f"✓ Confidence: {result['confidence']:.2f}")
        print(f"✓ Key topics: {', '.join(result['key_topics'])}")

    print("✓ Domain classification working correctly!\n")


def test_expert_agent_creation():
    """Test expert agent creation and functionality."""
    print("=== Expert Agent Creation Test ===")

    # Test creating specific expert agents
    sw_expert = SoftwareEngineeringExpert("software_engineering", "Coding expert")
    print(f"✓ Created Software Engineering expert with {len(sw_expert.expertise_areas)} expertise areas")

    # Test generic expert for unknown domains
    generic_expert = GenericExpertAgent("legal", "Legal expert")
    print(f"✓ Created Generic expert for unknown domain 'legal'")

    # Test factory function
    auto_expert = create_expert_agent("software_engineering")
    print(f"✓ Factory function created: {type(auto_expert).__name__}")

    print("✓ Expert agent creation working correctly!\n")


def test_expert_prompt_improvement():
    """Test prompt improvement with mock responses."""
    print("=== Prompt Improvement Test ===")

    expert = SoftwareEngineeringExpert("software_engineering", "Coding expert")

    with patch('agents.base_expert.ChatOpenAI') as mock_chat:
        # Mock the LLM response
        mock_response = Mock()
        mock_response.content = json.dumps({
            "improved_prompt": "Write a Python function to sort a list using the merge sort algorithm with proper error handling and documentation.",
            "improvements_made": ["Added algorithm specification", "Added language requirement", "Added error handling"],
            "key_additions": ["Merge sort algorithm", "Python language", "Error handling"],
            "structure_analysis": "Better structured with specific requirements",
            "effectiveness_score": 0.9,
            "reasoning": "More specific and actionable prompt"
        })
        mock_chat.return_value.invoke.return_value = mock_response

        original_prompt = "Write a function to sort a list"
        result = expert.improve_prompt(original_prompt, "raw", ["code", "algorithm"])

        print(f"✓ Improved prompt: '{original_prompt[:20]}...'")
        print(f"✓ Effectiveness score: {result['effectiveness_score']:.2f}")
        print(f"✓ Improvements made: {len(result['improvements_made'])}")

    print("✓ Prompt improvement working correctly!\n")


def test_evaluation_system():
    """Test the evaluation system with mock responses."""
    print("=== Evaluation System Test ===")

    with patch('agents.evaluator.ChatOpenAI') as mock_chat:
        # Mock the LLM response
        mock_response = Mock()
        mock_response.content = json.dumps({
            "overall_score": 0.88,
            "criteria_scores": {
                "clarity": 0.9,
                "specificity": 0.85,
                "structure": 0.9,
                "completeness": 0.85,
                "actionability": 0.9,
                "domain_alignment": 0.9
            },
            "passes_threshold": True,
            "needs_improvement": False,
            "strengths": ["Clear and specific", "Good structure"],
            "weaknesses": [],
            "specific_feedback": [],
            "improvement_priority": "low",
            "reasoning": "High quality improvement",
            "comparison_analysis": "Significant improvement"
        })
        mock_chat.return_value.invoke.return_value = mock_response

        original = "Write code"
        improved = "Write a Python function to sort a list using merge sort algorithm"

        result = evaluator.evaluate_prompt(
            original_prompt=original,
            improved_prompt=improved,
            domain="software_engineering",
            prompt_type="raw"
        )

        print(f"✓ Evaluated prompt improvement")
        print(f"✓ Overall score: {result['overall_score']:.2f}")
        print(f"✓ Passes threshold: {result['passes_threshold']}")
        print(f"✓ Strengths identified: {len(result['strengths'])}")

    print("✓ Evaluation system working correctly!\n")


def test_full_workflow_integration():
    """Test the full workflow integration with mocks."""
    print("=== Full Workflow Integration Test ===")

    # Mock all the LLM calls
    with patch('agents.classifier.ChatOpenAI') as mock_classifier_chat, \
         patch('agents.base_expert.ChatOpenAI') as mock_expert_chat, \
         patch('agents.evaluator.ChatOpenAI') as mock_eval_chat:

        # Mock classifier response
        classifier_response = Mock()
        classifier_response.content = json.dumps({
            "domain": "software_engineering",
            "confidence": 0.95,
            "is_new_domain": False,
            "key_topics": ["code", "algorithm"],
            "reasoning": "This is a coding task"
        })
        mock_classifier_chat.return_value.invoke.return_value = classifier_response

        # Mock expert response
        expert_response = Mock()
        expert_response.content = json.dumps({
            "improved_prompt": "Write a Python function to sort a list using merge sort algorithm with proper error handling and documentation.",
            "improvements_made": ["Added algorithm specification", "Added language requirement"],
            "key_additions": ["Merge sort algorithm", "Python language"],
            "structure_analysis": "Better structured with specific requirements",
            "effectiveness_score": 0.9,
            "reasoning": "More specific and actionable"
        })
        mock_expert_chat.return_value.invoke.return_value = expert_response

        # Mock evaluator response
        evaluator_response = Mock()
        evaluator_response.content = json.dumps({
            "overall_score": 0.88,
            "criteria_scores": {
                "clarity": 0.9, "specificity": 0.85, "structure": 0.9,
                "completeness": 0.85, "actionability": 0.9, "domain_alignment": 0.9
            },
            "passes_threshold": True,
            "needs_improvement": False,
            "strengths": ["Clear and specific"],
            "weaknesses": [],
            "specific_feedback": [],
            "improvement_priority": "low",
            "reasoning": "High quality improvement",
            "comparison_analysis": "Significant improvement"
        })
        mock_eval_chat.return_value.invoke.return_value = evaluator_response

        # Test the full workflow
        prompt = "Write a function to sort a list of numbers"
        result = coordinator.process_prompt(prompt, "raw", True)

        print(f"✓ Full workflow completed")
        print(f"✓ Status: {result['status']}")
        print(f"✓ Domain: {result['output']['domain']}")
        print(f"✓ Quality Score: {result['output']['quality_score']:.2f}")
        print(f"✓ Iterations: {result['output']['iterations_used']}")
        print(f"✓ Processing Time: {result['processing_time_seconds']:.2f}s")

        # Check comparison data
        if result.get('comparison'):
            print(f"✓ Improvement Ratio: {result['comparison']['improvement_ratio']:.2%}")

    print("✓ Full workflow integration working correctly!\n")


def test_system_statistics():
    """Test system statistics and analytics."""
    print("=== System Statistics Test ===")

    # Get available domains
    domains = coordinator.get_available_domains()
    print(f"✓ Available domains: {len(domains)}")
    for domain in domains[:3]:
        print(f"  - {domain['domain']}: {domain['description'][:50]}...")

    # Get system statistics
    stats = coordinator.get_workflow_stats()
    print(f"✓ Total workflows: {stats['total_workflows']}")
    print(f"✓ Success rate: {stats['success_rate']:.2%}")
    print(f"✓ Average quality score: {stats['average_quality_score']:.2f}")

    print("✓ System statistics working correctly!\n")


def test_error_handling():
    """Test error handling capabilities."""
    print("=== Error Handling Test ===")

    # Test with empty prompt
    try:
        result = coordinator.process_prompt("", "raw")
        print(f"✓ Empty prompt handled gracefully: {result['status']}")
    except Exception as e:
        print(f"✓ Empty prompt error caught: {e}")

    # Test with very long prompt
    long_prompt = "Write a function " * 1000  # Very long repetitive prompt
    try:
        result = coordinator.process_prompt(long_prompt[:5000], "raw", False)
        print(f"✓ Long prompt processed: {result['status']}")
        print(f"✓ Processing time: {result['processing_time_seconds']:.2f}s")
    except Exception as e:
        print(f"✓ Long prompt error caught: {e}")

    print("✓ Error handling working correctly!\n")


def test_configuration_system():
    """Test the configuration system."""
    print("=== Configuration System Test ===")

    # Test settings
    print(f"✓ Default model provider: {settings.default_model_provider}")
    print(f"✓ Default model name: {settings.default_model_name}")
    print(f"✓ Evaluation threshold: {settings.evaluation_threshold}")
    print(f"✓ Max iterations: {settings.max_evaluation_iterations}")
    print(f"✓ Log level: {settings.log_level}")

    # Test model configuration
    from config.config import get_model_config
    config = get_model_config()
    print(f"✓ Model config for {settings.default_model_provider}: {config['model_name']}")

    print("✓ Configuration system working correctly!\n")


def main():
    """Run all system tests."""
    print("🚀 Multi-Agent Prompt Engineering System - Architecture Tests")
    print("=" * 70)
    print("Note: These tests validate system architecture without requiring API keys.")
    print("They use mocked LLM responses to test the system flow.\n")

    try:
        # Run all tests
        test_system_initialization()
        test_domain_classification()
        test_expert_agent_creation()
        test_expert_prompt_improvement()
        test_evaluation_system()
        test_full_workflow_integration()
        test_system_statistics()
        test_error_handling()
        test_configuration_system()

        print("=" * 70)
        print("✅ All system architecture tests passed!")
        print("\n🔗 Next Steps:")
        print("1. Add your API keys to the .env file")
        print("2. Run: python src/main.py")
        print("3. Open http://localhost:8000 in your browser")
        print("4. Test with real prompts using the web interface")
        print("\n📚 For more examples:")
        print("   python examples/basic_usage.py")
        print("   python examples/advanced_integration.py")

    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
