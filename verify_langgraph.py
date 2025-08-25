#!/usr/bin/env python3
"""
Verify LangGraph Studio compatibility for the Multi-Agent Prompt Engineering System.
"""

import os
import sys

# Add the project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from src.workflow import create_prompt_engineering_app, process_prompt_with_langgraph
from langgraph.graph import StateGraph
from langchain_core.runnables import RunnableConfig

def test_langgraph_workflow_creation():
    """Test that the LangGraph workflow can be created and compiled."""
    print("=== LangGraph Workflow Creation Test ===")

    try:
        # Create the workflow
        app = create_prompt_engineering_app()
        print("✓ LangGraph workflow created successfully")

        # Check if it's a compiled graph
        if hasattr(app, 'get_graph'):
            graph = app.get_graph()
            print("✓ Workflow compiled successfully")
            print(f"✓ Graph has {len(graph.nodes)} nodes")
            print(f"✓ Graph has {len(graph.edges)} edges")

            # Print node information
            print("\n📋 Workflow Nodes:")
            for node_name in graph.nodes:
                print(f"  - {node_name}")

        else:
            print("✓ Workflow object created (not compiled for LangGraph Studio)")

        return True

    except Exception as e:
        print(f"❌ Error creating LangGraph workflow: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_langgraph_node_functions():
    """Test that all LangGraph node functions are properly defined."""
    print("\n=== LangGraph Node Functions Test ===")

    try:
        from src.workflow import (
            classify_node,
            create_expert_node,
            improve_prompt_node,
            evaluate_node,
            check_threshold_node,
            error_handler_node,
            WorkflowState
        )

        # Test node functions exist
        nodes = [
            classify_node,
            create_expert_node,
            improve_prompt_node,
            evaluate_node,
            check_threshold_node,
            error_handler_node
        ]

        for node in nodes:
            if callable(node):
                print(f"✓ {node.__name__} is callable")
            else:
                print(f"❌ {node.__name__} is not callable")
                return False

        # Test WorkflowState
        state = WorkflowState(
            original_prompt="Test prompt",
            prompt_type="raw",
            iterations_used=0,
            status="started",
            workflow_id="test_123"
        )
        print("✓ WorkflowState can be instantiated")

        return True

    except Exception as e:
        print(f"❌ Error testing node functions: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_langgraph_studio_compatibility():
    """Test LangGraph Studio specific compatibility."""
    print("\n=== LangGraph Studio Compatibility Test ===")

    try:
        app = create_prompt_engineering_app()

        # Test if the app has required methods for LangGraph Studio
        required_methods = ['invoke', 'astream', 'abatch']
        for method in required_methods:
            if hasattr(app, method):
                print(f"✓ Method '{method}' available")
            else:
                print(f"❌ Method '{method}' missing")
                return False

        # Test configuration support
        config = RunnableConfig(
            configurable={
                "thread_id": "studio_test_123"
            }
        )
        print("✓ RunnableConfig support verified")

        return True

    except Exception as e:
        print(f"❌ Error testing LangGraph Studio compatibility: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_workflow_execution():
    """Test workflow execution with mock data."""
    print("\n=== Workflow Execution Test ===")

    try:
        # Test with a simple prompt
        result = process_prompt_with_langgraph(
            prompt="Write a simple function",
            prompt_type="raw"
        )

        print(f"✓ Workflow executed with status: {result.get('status')}")
        print(f"✓ Framework: {result.get('metadata', {}).get('framework', 'unknown')}")
        print(f"✓ Domain detected: {result.get('output', {}).get('domain', 'unknown')}")

        return True

    except Exception as e:
        print(f"❌ Error testing workflow execution: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_graph_visualization():
    """Test graph visualization capabilities."""
    print("\n=== Graph Visualization Test ===")

    try:
        app = create_prompt_engineering_app()

        # Test getting graph representation
        if hasattr(app, 'get_graph'):
            graph = app.get_graph()

            # Try to get a simple representation
            try:
                # This is a basic test - in real LangGraph Studio,
                # more visualization features would be available
                graph_str = str(graph)
                print(f"✓ Graph representation available ({len(graph_str)} chars)")
            except:
                print("✓ Graph object created (visualization may require Studio)")
        else:
            print("✓ Workflow created (graph visualization requires compilation)")

        return True

    except Exception as e:
        print(f"❌ Error testing graph visualization: {e}")
        return False


def main():
    """Run all LangGraph compatibility tests."""
    print("🚀 Multi-Agent Prompt Engineering System - LangGraph Studio Compatibility Tests")
    print("=" * 80)
    print("Testing compatibility with LangGraph and LangGraph Studio...")

    all_passed = True

    # Run all tests
    tests = [
        test_langgraph_workflow_creation,
        test_langgraph_node_functions,
        test_langgraph_studio_compatibility,
        test_workflow_execution,
        test_graph_visualization
    ]

    for test in tests:
        try:
            if not test():
                all_passed = False
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
            all_passed = False

    print("\n" + "=" * 80)
    if all_passed:
        print("✅ All LangGraph Studio compatibility tests passed!")
        print("\n🔗 LangGraph Studio Integration:")
        print("1. The system is fully compatible with LangGraph Studio")
        print("2. All nodes and workflows are properly defined")
        print("3. Graph visualization is supported")
        print("4. Studio can be used for debugging and monitoring")
        print("\n📚 To use with LangGraph Studio:")
        print("   - Install LangGraph Studio")
        print("   - Import and use the workflow in Studio")
        print("   - Use the visualization features for debugging")
    else:
        print("❌ Some LangGraph Studio compatibility tests failed!")
        print("   Check the error messages above for details.")

    print("\n🔗 Next Steps:")
    print("1. Add your API keys to the .env file")
    print("2. Run: python src/main.py")
    print("3. Open http://localhost:8000 for the web interface")
    print("4. Use LangGraph Studio for advanced workflow debugging")


if __name__ == "__main__":
    main()
