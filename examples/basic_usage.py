#!/usr/bin/env python3
"""
Basic usage examples for the Multi-Agent Prompt Engineering System.
"""

import os
import sys
from dotenv import load_dotenv

# Add the project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from agents.coordinator import coordinator
from src.workflow import process_prompt_with_langgraph

# Load environment variables
load_dotenv()


def example_1_basic_prompt_processing():
    """Example 1: Basic prompt processing with the coordinator."""
    print("=== Example 1: Basic Prompt Processing ===")

    prompt = "Write a function to sort a list of numbers"
    print(f"Original prompt: {prompt}")

    result = coordinator.process_prompt(
        prompt=prompt,
        prompt_type="raw",
        return_comparison=True
    )

    print(f"Status: {result['status']}")
    print(f"Domain: {result['output']['domain']}")
    print(f"Quality Score: {result['output']['quality_score']}")
    print(f"Iterations Used: {result['output']['iterations_used']}")
    print(f"Processing Time: {result['processing_time_seconds']:.2f} seconds")
    print(f"\nOptimized Prompt:\n{result['output']['optimized_prompt']}")

    if result.get('comparison'):
        print(f"\nImprovement Ratio: {result['comparison']['improvement_ratio']:.2%}")


def example_2_structured_prompt():
    """Example 2: Processing a structured prompt."""
    print("\n=== Example 2: Structured Prompt Processing ===")

    structured_prompt = """
    Task: Create a data analysis report
    Requirements:
    - Analyze sales data
    - Create visualizations
    - Provide insights
    """

    print(f"Original prompt:\n{structured_prompt}")

    result = coordinator.process_prompt(
        prompt=structured_prompt,
        prompt_type="structured",
        return_comparison=True
    )

    print(f"Status: {result['status']}")
    print(f"Domain: {result['output']['domain']}")
    print(f"Quality Score: {result['output']['quality_score']}")
    print(f"\nOptimized Prompt:\n{result['output']['optimized_prompt']}")


def example_3_langgraph_workflow():
    """Example 3: Using LangGraph workflow."""
    print("\n=== Example 3: LangGraph Workflow ===")

    prompt = "Create a machine learning model for predicting house prices"
    print(f"Original prompt: {prompt}")

    result = process_prompt_with_langgraph(
        prompt=prompt,
        prompt_type="auto"
    )

    print(f"Status: {result['status']}")
    print(f"Domain: {result['output']['domain']}")
    print(f"Quality Score: {result['output']['quality_score']}")
    print(f"Framework: {result['metadata']['framework']}")
    print(f"\nOptimized Prompt:\n{result['output']['optimized_prompt']}")


def example_4_batch_processing():
    """Example 4: Batch processing multiple prompts."""
    print("\n=== Example 4: Batch Processing ===")

    prompts = [
        "Write a Python script to scrape websites",
        "Design a user interface for a mobile app",
        "Analyze customer feedback data",
        "Create a marketing campaign strategy"
    ]

    for i, prompt in enumerate(prompts, 1):
        print(f"\n--- Prompt {i} ---")
        print(f"Input: {prompt}")

        result = coordinator.process_prompt(
            prompt=prompt,
            prompt_type="auto",
            return_comparison=False  # Skip comparison for brevity
        )

        print(f"Domain: {result['output']['domain']}")
        print(f"Score: {result['output']['quality_score']:.2f}")
        print(f"Optimized: {result['output']['optimized_prompt'][:100]}...")


def example_5_getting_system_info():
    """Example 5: Getting system information and statistics."""
    print("\n=== Example 5: System Information ===")

    # Get available domains
    domains = coordinator.get_available_domains()
    print(f"Available domains: {len(domains)}")
    for domain in domains[:3]:  # Show first 3
        print(f"  - {domain['domain']}: {domain['description']}")

    # Get system statistics
    stats = coordinator.get_workflow_stats()
    print(f"\nSystem Statistics:")
    print(f"  Total workflows: {stats['total_workflows']}")
    print(f"  Success rate: {stats['success_rate']:.2%}")
    print(f"  Average quality score: {stats['average_quality_score']:.2f}")
    print(f"  Average processing time: {stats['average_processing_time']:.2f}s")


def example_6_error_handling():
    """Example 6: Error handling and edge cases."""
    print("\n=== Example 6: Error Handling ===")

    # Test with empty prompt
    try:
        result = coordinator.process_prompt("", "raw")
        print(f"Empty prompt result: {result['status']}")
    except Exception as e:
        print(f"Error with empty prompt: {e}")

    # Test with very long prompt
    long_prompt = "Write a function " * 1000  # Very long repetitive prompt
    try:
        result = coordinator.process_prompt(long_prompt[:5000], "raw", False)
        print(f"Long prompt result: {result['status']}")
        print(f"Processing time: {result['processing_time_seconds']:.2f}s")
    except Exception as e:
        print(f"Error with long prompt: {e}")


def main():
    """Run all examples."""
    print("🚀 Multi-Agent Prompt Engineering System - Usage Examples")
    print("=" * 60)

    # Check if API key is configured
    if not os.getenv('OPENAI_API_KEY'):
        print("⚠️  Warning: OPENAI_API_KEY not found in environment variables.")
        print("   Please set your API key in the .env file to run these examples.")
        print("   Example usage will be demonstrated with mock data where possible.\n")

    try:
        # Run examples
        example_1_basic_prompt_processing()
        example_2_structured_prompt()
        example_3_langgraph_workflow()
        example_4_batch_processing()
        example_5_getting_system_info()
        example_6_error_handling()

        print("\n" + "=" * 60)
        print("✅ All examples completed successfully!")
        print("\nTo run the web interface:")
        print("  python src/main.py")
        print("\nTo run with Docker:")
        print("  docker-compose -f docker/docker-compose.yml up --build")

    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        print("Make sure you have:")
        print("  - Set up your .env file with API keys")
        print("  - Installed all dependencies: pip install -r requirements.txt")
        print("  - Checked your internet connection for API calls")


if __name__ == "__main__":
    main()
