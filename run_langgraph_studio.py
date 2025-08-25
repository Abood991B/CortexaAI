#!/usr/bin/env python3
"""
LangGraph Studio Runner for Multi-Agent Prompt Engineering System

This script provides a convenient way to run the LangGraph Studio
for visualizing and testing the multi-agent prompt engineering workflow.

Usage:
    python run_langgraph_studio.py

Requirements:
    pip install langgraph-cli
    pip install -e .

Environment Variables:
    GOOGLE_API_KEY - Your Google AI API key
    LANGSMITH_API_KEY - Optional: LangSmith API key for tracing

The script will:
1. Check for required dependencies
2. Set up environment variables
3. Launch LangGraph Studio
4. Open the studio in your default browser
"""

import os
import sys
import subprocess
import time
import webbrowser
from pathlib import Path

def check_requirements():
    """Check if required packages are installed."""
    try:
        import langgraph
        print("✅ langgraph package found")
    except ImportError:
        print("❌ langgraph package not found. Installing...")
        subprocess.run([sys.executable, "-m", "pip", "install", "langgraph-cli"], check=True)

    try:
        from dotenv import load_dotenv
        print("✅ python-dotenv package found")
    except ImportError:
        print("❌ python-dotenv package not found. Installing...")
        subprocess.run([sys.executable, "-m", "pip", "install", "python-dotenv"], check=True)

def setup_environment():
    """Set up environment variables from .env file."""
    # Load environment variables
    env_file = Path(".env")
    if env_file.exists():
        from dotenv import load_dotenv
        load_dotenv()
        print("✅ Environment variables loaded from .env")
    else:
        print("⚠️  .env file not found. Please create one with your API keys.")

    # Check for required API key
    if not os.getenv("GOOGLE_API_KEY"):
        print("❌ GOOGLE_API_KEY not found in environment variables.")
        print("Please add your Google AI API key to the .env file:")
        print("GOOGLE_API_KEY=your_api_key_here")
        sys.exit(1)

    # Set up LangSmith if available
    if os.getenv("LANGSMITH_API_KEY"):
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
        os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGSMITH_PROJECT", "prompt-engineering-system")
        print("✅ LangSmith tracing enabled")
    else:
        print("⚠️  LANGSMITH_API_KEY not found. Tracing will be disabled.")

def run_langgraph_studio():
    """Run LangGraph Studio."""
    print("\n🚀 Starting LangGraph Studio...")
    print("📊 Multi-Agent Prompt Engineering System")
    print("=" * 50)

    try:
        # Run langgraph dev command
        cmd = ["langgraph", "dev"]
        print(f"Running: {' '.join(cmd)}")
        print("\n📱 Studio will open in your browser automatically")
        print("🔗 You can also visit: http://localhost:8123")
        print("\n⚠️  Press Ctrl+C to stop the studio\n")

        # Start the process
        process = subprocess.Popen(cmd, cwd=os.getcwd())

        # Wait a moment for the server to start
        time.sleep(3)

        # Open browser
        try:
            webbrowser.open("http://localhost:8123")
        except Exception as e:
            print(f"⚠️  Could not open browser automatically: {e}")
            print("Please visit http://localhost:8123 manually")

        # Wait for the process to complete
        process.wait()

    except KeyboardInterrupt:
        print("\n\n🛑 LangGraph Studio stopped by user")
    except Exception as e:
        print(f"\n❌ Error running LangGraph Studio: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure you have installed langgraph-cli: pip install langgraph-cli")
        print("2. Check that your .env file exists with GOOGLE_API_KEY")
        print("3. Ensure you're in the correct directory")
        sys.exit(1)

def main():
    """Main function."""
    print("🎯 LangGraph Studio Runner")
    print("Multi-Agent Prompt Engineering System")
    print("=" * 50)

    # Check requirements
    check_requirements()

    # Setup environment
    setup_environment()

    # Run studio
    run_langgraph_studio()

if __name__ == "__main__":
    main()
