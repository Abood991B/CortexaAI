#!/usr/bin/env python3
"""
LangGraph Studio Runner for Multi-Agent Prompt Engineering System

This script provides a convenient way to run the LangGraph Studio
for visualizing and testing the multi-agent prompt engineering workflow.

Usage:
    python run_langgraph_studio.py [--port PORT]

Requirements:
    pip install langgraph-cli
    pip install python-dotenv

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
import argparse
import json
from pathlib import Path

def check_requirements():
    """Check if required packages are installed."""
    packages_to_check = [
        ("langgraph", "langgraph"),
        ("langgraph-cli", "langgraph_cli"),
        ("python-dotenv", "dotenv")
    ]
    
    missing_packages = []
    
    for package_name, import_name in packages_to_check:
        try:
            __import__(import_name)
            print(f"✅ {package_name} package found")
        except ImportError:
            print(f"⚠️  {package_name} package not found")
            missing_packages.append(package_name)
    
    if missing_packages:
        print(f"\n📦 Installing missing packages: {', '.join(missing_packages)}")
        for package in missing_packages:
            try:
                subprocess.run(
                    [sys.executable, "-m", "pip", "install", package],
                    check=True,
                    capture_output=True,
                    text=True
                )
                print(f"✅ Successfully installed {package}")
            except subprocess.CalledProcessError as e:
                print(f"❌ Failed to install {package}: {e}")
                print(f"Please install manually: pip install {package}")
                sys.exit(1)

def setup_environment(project_root):
    """Set up environment variables from .env file."""
    # Load environment variables
    env_file = project_root / ".env"
    env_example_file = project_root / ".env.example"
    
    if env_file.exists():
        from dotenv import load_dotenv
        load_dotenv(env_file)
        print(f"✅ Environment variables loaded from {env_file}")
    else:
        print(f"⚠️  .env file not found at {env_file}")
        if env_example_file.exists():
            print("💡 Found .env.example file. You can copy it to .env and add your API keys:")
            print(f"   cp {env_example_file} {env_file}")
        print("")

    # Check for at least one API key (Google, OpenAI, or Anthropic)
    api_keys = {
        "GOOGLE_API_KEY": os.getenv("GOOGLE_API_KEY"),
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        "ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY")
    }
    
    available_keys = [k for k, v in api_keys.items() if v]
    
    if not available_keys:
        print("❌ No API keys found in environment variables.")
        print("Please add at least one of the following to your .env file:")
        print("  - GOOGLE_API_KEY=your_google_api_key")
        print("  - OPENAI_API_KEY=your_openai_api_key")
        print("  - ANTHROPIC_API_KEY=your_anthropic_api_key")
        sys.exit(1)
    else:
        print(f"✅ Found API keys: {', '.join(available_keys)}")

    # Set up LangSmith if available
    if os.getenv("LANGSMITH_API_KEY"):
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
        os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGSMITH_PROJECT", "prompt-engineering-system")
        print("✅ LangSmith tracing enabled")
    else:
        print("ℹ️  LANGSMITH_API_KEY not found. Tracing will be disabled (optional).")

def check_langgraph_config(project_root):
    """Check if langgraph.json exists and is valid."""
    config_file = project_root / "langgraph.json"
    
    if not config_file.exists():
        print(f"❌ langgraph.json not found at {config_file}")
        print("Creating a default configuration...")
        
        default_config = {
            "dependencies": ["."],
            "graphs": {
                "prompt_engineering": "./src/workflow.py:prompt_engineering_app"
            },
            "env": ".env"
        }
        
        with open(config_file, 'w') as f:
            json.dump(default_config, f, indent=2)
        print(f"✅ Created {config_file}")
    else:
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
            print(f"✅ Found valid langgraph.json configuration")
            
            # Verify the graph module exists
            graphs = config.get("graphs", {})
            for name, path in graphs.items():
                module_path = path.split(":")[0]
                full_path = project_root / module_path.replace("./", "")
                if full_path.exists():
                    print(f"  ✅ Graph '{name}' module found: {module_path}")
                else:
                    print(f"  ⚠️  Graph '{name}' module not found: {module_path}")
        except json.JSONDecodeError as e:
            print(f"❌ Invalid langgraph.json: {e}")
            sys.exit(1)

def run_langgraph_studio(project_root, port=8123, allow_blocking=True):
    """Run LangGraph Studio."""
    print("\n🚀 Starting LangGraph Studio...")
    print("📊 Multi-Agent Prompt Engineering System")
    print("=" * 50)

    # Check if langgraph CLI is available
    try:
        result = subprocess.run(
            ["langgraph", "--version"],
            capture_output=True,
            text=True,
            check=False
        )
        if result.returncode != 0:
            raise FileNotFoundError("langgraph command not found")
    except (FileNotFoundError, subprocess.SubprocessError):
        print("❌ langgraph CLI not found. Please install it:")
        print("   pip install langgraph-cli")
        sys.exit(1)

    try:
        # Run langgraph dev command with specified port and allow-blocking flag
        cmd = ["langgraph", "dev", "--port", str(port)]
        if allow_blocking:
            cmd.append("--allow-blocking")
            print("ℹ️  Running with --allow-blocking flag to handle synchronous calls")
        
        print(f"Running: {' '.join(cmd)}")
        print(f"Working directory: {project_root}")
        print(f"\n📱 Studio will open in your browser automatically")
        print(f"🔗 You can also visit: http://localhost:{port}")
        print("\n⚠️  Press Ctrl+C to stop the studio\n")

        # Start the process with proper environment
        env = os.environ.copy()
        process = subprocess.Popen(
            cmd,
            cwd=str(project_root),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )

        # Monitor the output for startup
        startup_complete = False
        start_time = time.time()
        
        while time.time() - start_time < 30:  # 30 second timeout
            if process.poll() is not None:
                # Process ended prematurely
                output, _ = process.communicate()
                print(f"❌ LangGraph Studio failed to start:\n{output}")
                sys.exit(1)
            
            line = process.stdout.readline()
            if line:
                print(line.rstrip())
                if "Running on" in line or "Uvicorn running" in line or "Started server" in line:
                    startup_complete = True
                    break
            
            time.sleep(0.1)
        
        if startup_complete:
            # Wait a bit more for full initialization
            time.sleep(2)
            
            # Open browser
            try:
                webbrowser.open(f"http://localhost:{port}")
                print(f"\n✅ Browser opened at http://localhost:{port}")
            except Exception as e:
                print(f"⚠️  Could not open browser automatically: {e}")
                print(f"Please visit http://localhost:{port} manually")
        
        # Continue printing output
        for line in iter(process.stdout.readline, ''):
            if line:
                print(line.rstrip())
        
        # Wait for the process to complete
        process.wait()

    except KeyboardInterrupt:
        print("\n\n🛑 LangGraph Studio stopped by user")
        if 'process' in locals():
            process.terminate()
            time.sleep(1)
            if process.poll() is None:
                process.kill()
    except Exception as e:
        print(f"\n❌ Error running LangGraph Studio: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure you have installed langgraph-cli: pip install langgraph-cli")
        print("2. Check that your .env file exists with at least one API key")
        print("3. Ensure langgraph.json is properly configured")
        print("4. Check if the port is already in use")
        sys.exit(1)

def main():
    """Main function."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Run LangGraph Studio for CORTEXA System"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8123,
        help="Port to run the studio on (default: 8123)"
    )
    args = parser.parse_args()
    
    print("LangGraph Studio Runner")
    print("CORTEXA System")
    print("=" * 50)

    # Determine project root (go up one level from tools directory)
    script_path = Path(__file__).resolve()
    project_root = script_path.parent.parent
    
    print(f"📁 Project root: {project_root}")
    
    # Change to project root directory
    original_cwd = os.getcwd()
    os.chdir(project_root)
    
    try:
        # Check requirements
        check_requirements()
        
        # Check langgraph configuration
        check_langgraph_config(project_root)
        
        # Setup environment
        setup_environment(project_root)
        
        # Run studio
        run_langgraph_studio(project_root, port=args.port)
    finally:
        # Restore original working directory
        os.chdir(original_cwd)

if __name__ == "__main__":
    main()
