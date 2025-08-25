#!/usr/bin/env python3
"""
Advanced integration examples for the Multi-Agent Prompt Engineering System.
"""

import os
import sys
import asyncio
import json
from typing import List, Dict, Any
from dotenv import load_dotenv

# Add the project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from agents.coordinator import coordinator
from src.workflow import process_prompt_with_langgraph

# Load environment variables
load_dotenv()


class PromptEngineeringAPI:
    """Example wrapper class for integrating with the prompt engineering system."""

    def __init__(self):
        self.coordinator = coordinator

    async def optimize_prompt_async(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Async version of prompt optimization."""
        return await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.coordinator.process_prompt(prompt, **kwargs)
        )

    def batch_optimize(self, prompts: List[str], **kwargs) -> List[Dict[str, Any]]:
        """Optimize multiple prompts in batch."""
        results = []
        for prompt in prompts:
            try:
                result = self.coordinator.process_prompt(prompt, **kwargs)
                results.append(result)
            except Exception as e:
                results.append({
                    "status": "error",
                    "error": str(e),
                    "input": prompt
                })
        return results

    def get_domain_statistics(self) -> Dict[str, Any]:
        """Get detailed domain usage statistics."""
        domains = self.coordinator.get_available_domains()
        stats = self.coordinator.get_workflow_stats()

        return {
            "total_domains": len(domains),
            "domains": domains,
            "workflow_stats": stats,
            "domain_distribution": stats.get("domain_distribution", {})
        }


def example_async_processing():
    """Example: Asynchronous prompt processing."""
    print("=== Example: Asynchronous Processing ===")

    api = PromptEngineeringAPI()

    async def process_async():
        prompts = [
            "Write a Python function for data validation",
            "Create a REST API design document",
            "Analyze sales performance metrics"
        ]

        tasks = [
            api.optimize_prompt_async(prompt, prompt_type="auto", return_comparison=False)
            for prompt in prompts
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for i, result in enumerate(results, 1):
            if isinstance(result, Exception):
                print(f"Prompt {i} failed: {result}")
            else:
                print(f"Prompt {i} - Domain: {result['output']['domain']}, "
                      f"Score: {result['output']['quality_score']:.2f}")

    # Run the async function
    asyncio.run(process_async())


def example_batch_processing():
    """Example: Batch processing with error handling."""
    print("\n=== Example: Batch Processing ===")

    api = PromptEngineeringAPI()

    prompts = [
        "Design a database schema",
        "Create a user authentication system",
        "Implement error logging",
        "",  # Empty prompt to test error handling
        "Build a recommendation engine"
    ]

    results = api.batch_optimize(
        prompts,
        prompt_type="auto",
        return_comparison=False
    )

    for i, result in enumerate(results, 1):
        if result.get("status") == "error":
            print(f"❌ Prompt {i}: {result.get('error', 'Unknown error')}")
        else:
            print(f"✅ Prompt {i}: {result['output']['domain']} "
                  f"(Score: {result['output']['quality_score']:.2f})")


def example_custom_workflow():
    """Example: Custom workflow with specific requirements."""
    print("\n=== Example: Custom Workflow ===")

    # Define custom processing requirements
    custom_prompts = [
        {
            "prompt": "Write a function to process CSV files",
            "requirements": {
                "domain": "software_engineering",
                "min_score": 0.85,
                "max_iterations": 2
            }
        },
        {
            "prompt": "Analyze customer churn data",
            "requirements": {
                "domain": "data_science",
                "min_score": 0.8,
                "max_iterations": 3
            }
        }
    ]

    for item in custom_prompts:
        print(f"\nProcessing: {item['prompt'][:50]}...")

        # Set custom evaluation threshold for this prompt
        original_threshold = coordinator.evaluator.evaluation_threshold
        coordinator.evaluator.evaluation_threshold = item["requirements"]["min_score"]
        coordinator.evaluator.max_iterations = item["requirements"]["max_iterations"]

        result = coordinator.process_prompt(
            prompt=item["prompt"],
            prompt_type="auto",
            return_comparison=True
        )

        # Restore original threshold
        coordinator.evaluator.evaluation_threshold = original_threshold
        coordinator.evaluator.max_iterations = 3

        print(f"Domain: {result['output']['domain']}")
        print(f"Score: {result['output']['quality_score']:.2f}")
        print(f"Threshold Met: {result['output']['passes_threshold']}")
        print(f"Iterations: {result['output']['iterations_used']}")


def example_langgraph_comparison():
    """Example: Comparing Coordinator vs LangGraph workflows."""
    print("\n=== Example: Coordinator vs LangGraph Comparison ===")

    prompt = "Create a machine learning pipeline for image classification"

    # Process with Coordinator
    print("1. Processing with Coordinator...")
    coord_result = coordinator.process_prompt(
        prompt=prompt,
        prompt_type="auto",
        return_comparison=False
    )

    # Process with LangGraph
    print("2. Processing with LangGraph...")
    lg_result = process_prompt_with_langgraph(
        prompt=prompt,
        prompt_type="auto"
    )

    # Compare results
    print("\nComparison:")
    print(f"Coordinator - Score: {coord_result['output']['quality_score']:.2f}, "
          f"Time: {coord_result['processing_time_seconds']:.2f}s")
    print(f"LangGraph   - Score: {lg_result['output']['quality_score']:.2f}, "
          f"Time: {lg_result.get('processing_time_seconds', 'N/A')}s")

    print(f"\nBoth passed threshold: "
          f"Coord={coord_result['output']['passes_threshold']}, "
          f"LG={lg_result['output']['passes_threshold']}")


def example_system_monitoring():
    """Example: System monitoring and analytics."""
    print("\n=== Example: System Monitoring ===")

    api = PromptEngineeringAPI()

    # Get system statistics
    stats = api.get_domain_statistics()

    print("📊 System Overview:")
    print(f"   Total Domains: {stats['total_domains']}")
    print(f"   Total Workflows: {stats['workflow_stats']['total_workflows']}")
    print(f"   Success Rate: {stats['workflow_stats']['success_rate']:.2%}")
    print(f"   Avg Quality Score: {stats['workflow_stats']['average_quality_score']:.2f}")
    print(f"   Avg Processing Time: {stats['workflow_stats']['average_processing_time']:.2f}s")

    print("\n🏷️  Available Domains:")
    for domain in stats["domains"][:5]:  # Show first 5
        status = "✅" if domain["has_expert_agent"] else "❌"
        print(f"   {status} {domain['domain']}: {domain['description']}")

    if stats["domain_distribution"]:
        print("\n📈 Domain Usage Distribution:")
        for domain, count in list(stats["domain_distribution"].items())[:5]:
            print(f"   {domain}: {count} workflows")


def example_integration_pattern():
    """Example: Integration pattern for larger applications."""
    print("\n=== Example: Application Integration Pattern ===")

    class ContentGenerator:
        """Example class showing how to integrate the prompt engineering system."""

        def __init__(self):
            self.prompt_engine = PromptEngineeringAPI()

        def generate_technical_content(self, topic: str, content_type: str) -> Dict[str, Any]:
            """Generate technical content with optimized prompts."""

            # Create a base prompt
            base_prompt = f"Create {content_type} about {topic}"

            # Optimize the prompt
            optimized_result = self.prompt_engine.coordinator.process_prompt(
                prompt=base_prompt,
                prompt_type="raw",
                return_comparison=True
            )

            return {
                "topic": topic,
                "content_type": content_type,
                "optimized_prompt": optimized_result["output"]["optimized_prompt"],
                "quality_score": optimized_result["output"]["quality_score"],
                "domain": optimized_result["output"]["domain"]
            }

        def batch_generate_content(self, topics: List[str], content_type: str) -> List[Dict[str, Any]]:
            """Generate content for multiple topics."""
            prompts = [f"Create {content_type} about {topic}" for topic in topics]
            return self.prompt_engine.batch_optimize(prompts, prompt_type="raw")

    # Usage example
    generator = ContentGenerator()

    # Generate single content
    result = generator.generate_technical_content("machine learning", "tutorial")
    print(f"Generated content for: {result['topic']}")
    print(f"Domain: {result['domain']}")
    print(f"Quality Score: {result['quality_score']:.2f}")

    # Batch generate content
    topics = ["neural networks", "data visualization", "API design"]
    batch_results = generator.batch_generate_content(topics, "comprehensive guide")

    print("\nBatch Results:")
    for i, result in enumerate(batch_results, 1):
        if result.get("status") == "error":
            print(f"  {i}. ❌ Failed: {result.get('error')}")
        else:
            domain = result["output"]["domain"]
            score = result["output"]["quality_score"]
            print(f"  {i}. ✅ {domain} (Score: {score:.2f})")


def example_error_recovery():
    """Example: Error recovery and retry logic."""
    print("\n=== Example: Error Recovery ===")

    def process_with_retry(prompt: str, max_retries: int = 3) -> Dict[str, Any]:
        """Process prompt with retry logic."""
        for attempt in range(max_retries):
            try:
                result = coordinator.process_prompt(
                    prompt=prompt,
                    prompt_type="auto",
                    return_comparison=False
                )
                return result
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    return {
                        "status": "error",
                        "error": str(e),
                        "attempts": max_retries
                    }
        return {"status": "error", "error": "Max retries exceeded"}

    # Test with potentially problematic prompts
    test_prompts = [
        "Write a simple function",  # Should work fine
        "",  # Empty prompt - should fail gracefully
        "Create" + " a" * 1000,  # Very long prompt
    ]

    for i, prompt in enumerate(test_prompts, 1):
        print(f"\nTest {i}: {'[EMPTY]' if not prompt.strip() else prompt[:30]}...")
        result = process_with_retry(prompt)

        if result["status"] == "error":
            print(f"❌ Failed after retries: {result.get('error', 'Unknown error')}")
        else:
            print(f"✅ Success: {result['output']['domain']} "
                  f"(Score: {result['output']['quality_score']:.2f})")


def main():
    """Run all advanced examples."""
    print("🚀 Multi-Agent Prompt Engineering System - Advanced Examples")
    print("=" * 70)

    if not os.getenv('OPENAI_API_KEY'):
        print("⚠️  Warning: OPENAI_API_KEY not found.")
        print("   Please set your API key in the .env file.\n")

    try:
        # Run examples
        example_async_processing()
        example_batch_processing()
        example_custom_workflow()
        example_langgraph_comparison()
        example_system_monitoring()
        example_integration_pattern()
        example_error_recovery()

        print("\n" + "=" * 70)
        print("✅ All advanced examples completed successfully!")
        print("\n🔗 Integration Resources:")
        print("   📚 Full API documentation: README.md")
        print("   🧪 Test suite: tests/test_agents.py")
        print("   🐳 Docker deployment: docker/docker-compose.yml")

    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
