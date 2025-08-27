#!/usr/bin/env python3
"""
Test Script for LangGraph Studio

This script provides test inputs for the Multi-Agent Prompt Engineering System
when running in LangGraph Studio.

Usage:
    python test_langgraph_studio.py

Or run in LangGraph Studio:
    1. Start LangGraph Studio: python run_langgraph_studio.py
    2. Open http://localhost:8123
    3. Use the test inputs below to test the workflow
"""

import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.workflow import process_prompt_with_langgraph

def test_software_engineering_prompt():
    """Test with a software engineering prompt."""
    prompt = "Write a Python function named `sort_list` that takes a list of integers as input and returns a new list with the integers sorted in ascending order. Include proper error handling, docstrings, and unit tests."

    print("🧪 Testing Software Engineering Prompt:")
    print(f"Input: {prompt}")
    print("-" * 50)

    try:
        result = process_prompt_with_langgraph(prompt, "structured")

        print("✅ Workflow completed successfully!")
        print(f"Domain: {result.get('output', {}).get('domain', 'N/A')}")
        print(f"Quality Score: {result.get('output', {}).get('quality_score', 0)}")
        print(f"Iterations Used: {result.get('output', {}).get('iterations_used', 0)}")
        print(f"Status: {result.get('status', 'N/A')}")

        print("\n📝 Optimized Prompt:")
        print(result.get('output', {}).get('optimized_prompt', 'N/A'))

        return result

    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def test_data_science_prompt():
    """Test with a data science prompt."""
    prompt = "Analyze the sales data and create visualizations to identify trends and patterns. Use Python and provide insights about customer behavior."

    print("\n🧪 Testing Data Science Prompt:")
    print(f"Input: {prompt}")
    print("-" * 50)

    try:
        result = process_prompt_with_langgraph(prompt, "raw")

        print("✅ Workflow completed successfully!")
        print(f"Domain: {result.get('output', {}).get('domain', 'N/A')}")
        print(f"Quality Score: {result.get('output', {}).get('quality_score', 0)}")
        print(f"Iterations Used: {result.get('output', {}).get('iterations_used', 0)}")
        print(f"Status: {result.get('status', 'N/A')}")

        print("\n📝 Optimized Prompt:")
        print(result.get('output', {}).get('optimized_prompt', 'N/A'))

        return result

    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def test_business_strategy_prompt():
    """Test with a business strategy prompt."""
    prompt = "Develop a marketing strategy for launching a new tech product. Include target audience analysis, competitive analysis, and budget recommendations."

    print("\n🧪 Testing Business Strategy Prompt:")
    print(f"Input: {prompt}")
    print("-" * 50)

    try:
        result = process_prompt_with_langgraph(prompt, "raw")

        print("✅ Workflow completed successfully!")
        print(f"Domain: {result.get('output', {}).get('domain', 'N/A')}")
        print(f"Quality Score: {result.get('output', {}).get('quality_score', 0)}")
        print(f"Iterations Used: {result.get('output', {}).get('iterations_used', 0)}")
        print(f"Status: {result.get('status', 'N/A')}")

        print("\n📝 Optimized Prompt:")
        print(result.get('output', {}).get('optimized_prompt', 'N/A'))

        return result

    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def test_education_prompt():
    """Test with an education prompt."""
    prompt = "Create a lesson plan for teaching basic algebra to high school students. Include objectives, activities, and assessment methods."

    print("\n🧪 Testing Education Prompt:")
    print(f"Input: {prompt}")
    print("-" * 50)

    try:
        result = process_prompt_with_langgraph(prompt, "structured")

        print("✅ Workflow completed successfully!")
        print(f"Domain: {result.get('output', {}).get('domain', 'N/A')}")
        print(f"Quality Score: {result.get('output', {}).get('quality_score', 0)}")
        print(f"Iterations Used: {result.get('output', {}).get('iterations_used', 0)}")
        print(f"Status: {result.get('status', 'N/A')}")

        print("\n📝 Optimized Prompt:")
        print(result.get('output', {}).get('optimized_prompt', 'N/A'))

        return result

    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def interactive_test():
    """Run interactive tests."""
    print("🎯 LangGraph Studio Test Suite")
    print("Multi-Agent Prompt Engineering System")
    print("=" * 60)

    # Test prompts
    test_prompts = [
        "Software Engineering",
        "Data Science",
        "Business Strategy",
        "Education"
    ]

    results = []

    print("Available test categories:")
    for i, category in enumerate(test_prompts, 1):
        print(f"{i}. {category}")

    print("\nRunning all tests...\n")

    # Run all tests
    results.append(test_software_engineering_prompt())
    results.append(test_data_science_prompt())
    results.append(test_business_strategy_prompt())
    results.append(test_education_prompt())

    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)

    successful = sum(1 for r in results if r is not None)
    total = len(results)

    print(f"✅ Successful: {successful}/{total}")
    print(f"❌ Failed: {total - successful}/{total}")

    if successful > 0:
        print("\n🎉 LangGraph workflow is working correctly!")
        print("\n📱 To visualize the graph in LangGraph Studio:")
        print("1. Run: python run_langgraph_studio.py")
        print("2. Open: http://localhost:8123")
        print("3. Select the 'prompt_engineering' graph")
        print("4. Test with the prompts above")
    else:
        print("\n❌ All tests failed. Check your configuration:")
        print("- Ensure GOOGLE_API_KEY is set in .env")
        print("- Check your internet connection")
        print("- Verify the virtual environment is activated")

def main():
    """Main function."""
    if len(sys.argv) > 1:
        # Run specific test
        test_type = sys.argv[1].lower()

        if test_type == "software" or test_type == "se":
            test_software_engineering_prompt()
        elif test_type == "data" or test_type == "ds":
            test_data_science_prompt()
        elif test_type == "business" or test_type == "biz":
            test_business_strategy_prompt()
        elif test_type == "education" or test_type == "edu":
            test_education_prompt()
        else:
            print(f"Unknown test type: {test_type}")
            print("Available types: software, data, business, education")
    else:
        # Run all tests
        interactive_test()

if __name__ == "__main__":
    main()
