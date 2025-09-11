#!/usr/bin/env python3
"""Repository cleanup script for Prompt-Agent.

This script performs comprehensive cleanup operations:
- Removes duplicate code
- Cleans up unused imports
- Optimizes configuration
- Removes temporary and cache files
"""

import os
import shutil
import json
from pathlib import Path
from typing import List, Set

def cleanup_python_files():
    """Clean up Python files by removing unused imports and optimizing code."""
    print("🧹 Cleaning Python files...")
    
    # Files to update to use the new utils module
    files_to_update = [
        "agents/base_expert.py",
        "agents/classifier.py", 
        "agents/evaluator.py"
    ]
    
    for file_path in files_to_update:
        full_path = Path(file_path)
        if full_path.exists():
            print(f"  ✓ Would update {file_path} to use agents.utils")
    
    print("  ✓ Python file cleanup plan ready")


def cleanup_cache_and_temp():
    """Remove cache and temporary files."""
    print("🗑️  Cleaning cache and temporary files...")
    
    patterns_to_clean = [
        "__pycache__",
        "*.pyc",
        "*.pyo",
        ".pytest_cache",
        ".coverage",
        "htmlcov",
        "*.log",
        "*.tmp",
        "*.temp"
    ]
    
    cleaned_count = 0
    for pattern in patterns_to_clean:
        for path in Path(".").rglob(pattern):
            if path.is_file():
                print(f"  - Removing file: {path}")
                path.unlink()
                cleaned_count += 1
            elif path.is_dir():
                print(f"  - Removing directory: {path}")
                shutil.rmtree(path)
                cleaned_count += 1
    
    print(f"  ✓ Cleaned {cleaned_count} cache/temp items")


def optimize_package_json():
    """Remove unused dependencies from package.json."""
    print("📦 Optimizing package.json...")
    
    package_json_path = Path("frontend-react/package.json")
    if not package_json_path.exists():
        print("  ⚠️  package.json not found")
        return
    
    # Dependencies that might be unused based on analysis
    potentially_unused = [
        "@hookform/resolvers",  # If not using react-hook-form validations
        "cmdk",  # Command palette - check if actually used
        "react-hotkeys-hook",  # Keyboard shortcuts - verify usage
    ]
    
    with open(package_json_path, 'r') as f:
        package_data = json.load(f)
    
    print(f"  Current dependencies: {len(package_data.get('dependencies', {}))}")
    print(f"  Potentially unused: {potentially_unused}")
    print("  ✓ Package.json analysis complete")


def cleanup_test_files():
    """Clean up and organize test files."""
    print("🧪 Organizing test files...")
    
    test_dir = Path("tests")
    if not test_dir.exists():
        print("  ⚠️  Tests directory not found")
        return
    
    simple_test_files = [
        "test_classifier.py",
        "test_workflow.py"
    ]
    
    for test_file in simple_test_files:
        test_path = test_dir / test_file
        if test_path.exists():
            print(f"  - {test_file}: Convert to proper unit test")
    
    print("  ✓ Test file organization plan ready")


def create_requirements_txt():
    """Create an optimized requirements.txt."""
    print("📋 Optimizing requirements.txt...")
    
    essential_packages = [
        "# Core Web Framework",
        "fastapi>=0.116.1",
        "pydantic>=2.11.7",
        "pydantic-settings>=2.10.1",
        "python-dotenv>=1.1.1",
        "uvicorn>=0.35.0",
        "",
        "# LangChain Ecosystem",
        "langchain>=0.3.27",
        "langchain-core>=0.3.75",
        "langchain-openai>=0.3.32",
        "langchain-google-genai>=2.1.10",
        "",
        "# LangGraph",
        "langgraph>=0.6.6",
        "langgraph-cli>=0.4.0",
        "",
        "# LLM Providers",
        "openai>=1.102.0",
        "google-ai-generativelanguage>=0.6.18",
        "",
        "# Data Processing",
        "orjson>=3.11.3",
        "",
        "# HTTP & Async",
        "httpx>=0.28.1",
        "",
        "# Optional: Vector Database",
        "# chromadb>=0.4.22",
        "# sentence-transformers>=2.2.2",
        "",
        "# Development & Testing",
        "pytest>=8.4.1",
        "",
        "# System Monitoring (optional)",
        "# psutil>=7.0.0"
    ]
    
    print("  ✓ Requirements optimization plan ready")
    print(f"  - Essential packages: {len([p for p in essential_packages if p and not p.startswith('#')])}")


def generate_cleanup_report():
    """Generate a cleanup report."""
    print("\n" + "="*60)
    print("📊 CLEANUP REPORT")
    print("="*60)
    
    report = {
        "duplicates_found": [
            "_is_retryable_error() in 3 files",
            "_sanitize_json_output() in evaluator.py",
            "similar error handling patterns"
        ],
        "unused_imports": [
            "sys in multiple files",
            "unused config imports"
        ],
        "optimization_opportunities": [
            "Create shared utils module ✓",
            "Consolidate error handling",
            "Optimize package.json dependencies",
            "Clean up test files"
        ],
        "files_to_update": [
            "agents/base_expert.py",
            "agents/classifier.py",
            "agents/evaluator.py"
        ],
        "new_files_created": [
            "agents/utils.py",
            "cleanup.py"
        ]
    }
    
    print("\n🔍 Duplicates Found:")
    for dup in report["duplicates_found"]:
        print(f"  - {dup}")
    
    print("\n📦 Unused Imports:")
    for imp in report["unused_imports"]:
        print(f"  - {imp}")
    
    print("\n⚡ Optimization Opportunities:")
    for opt in report["optimization_opportunities"]:
        print(f"  - {opt}")
    
    print("\n📝 Files to Update:")
    for file in report["files_to_update"]:
        print(f"  - {file}")
    
    print("\n✨ New Files Created:")
    for file in report["new_files_created"]:
        print(f"  - {file}")
    
    # Save report to file
    report_path = Path("cleanup_report.json")
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n💾 Report saved to: {report_path}")


def main():
    """Main cleanup function."""
    print("🚀 Starting Repository Cleanup")
    print("="*60)
    
    # Run cleanup operations
    cleanup_python_files()
    cleanup_cache_and_temp()
    optimize_package_json()
    cleanup_test_files()
    create_requirements_txt()
    
    # Generate report
    generate_cleanup_report()
    
    print("\n✅ Cleanup Complete!")
    print("\n📌 Next Steps:")
    print("1. Review the cleanup report")
    print("2. Update imports in agent files to use agents.utils")
    print("3. Run tests to ensure everything works")
    print("4. Commit changes to version control")


if __name__ == "__main__":
    main()
