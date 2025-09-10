#!/usr/bin/env python3
"""
Performance Benchmarking Script for LangGraph Workflow in Multi-Agent Prompt Engineering System
"""

import asyncio
import time
import json
import sys
import os
from datetime import datetime
from typing import Dict, List, Any
import statistics
import httpx

# Add the parent directory to the path so we can import agents and config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from config.config import get_logger, settings
    IMPORTS_AVAILABLE = True
except ImportError as e:
    print(f"Error importing required modules: {e}")
    IMPORTS_AVAILABLE = False

logger = get_logger(__name__)


class LangGraphPerformanceBenchmark:
    """Comprehensive performance benchmarking suite for the LangGraph workflow."""

    def __init__(self):
        self.test_prompts = self._load_test_prompts()
        self.results = {}
        self.start_time = None
        self.base_url = f"http://localhost:{settings.port}"

    def _load_test_prompts(self) -> List[Dict[str, Any]]:
        """Load diverse test prompts for benchmarking."""
        return [
            {
                "prompt": "Write a function to validate email addresses using regex",
                "domain": "software_engineering",
                "complexity": "low"
            },
            {
                "prompt": "Create a REST API endpoint for user authentication with JWT tokens",
                "domain": "software_engineering",
                "complexity": "medium"
            },
            {
                "prompt": "Analyze the impact of climate change on global agriculture patterns",
                "domain": "data_science",
                "complexity": "high"
            },
            {
                "prompt": "Write a short story about a time traveler who accidentally changes history",
                "domain": "creative_writing",
                "complexity": "medium"
            },
            {
                "prompt": "Design a data visualization dashboard for sales performance metrics",
                "domain": "data_science",
                "complexity": "medium"
            }
        ]

    async def setup(self):
        """Initialize the system for benchmarking."""
        if not IMPORTS_AVAILABLE:
            raise RuntimeError("Required modules not available")
        logger.info("Setting up benchmark environment...")
        self.start_time = time.time()

    async def benchmark_langgraph_processing(self) -> Dict[str, Any]:
        """Benchmark LangGraph prompt processing performance."""
        logger.info("Running LangGraph prompt processing benchmark...")

        results = []
        async with httpx.AsyncClient() as client:
            for i, test_case in enumerate(self.test_prompts):
                logger.info(f"Testing prompt {i+1}/{len(self.test_prompts)}: {test_case['prompt'][:50]}...")

                start_time = time.time()
                try:
                    response = await client.post(
                        f"{self.base_url}/api/process-prompt",
                        json={
                            "prompt": test_case["prompt"],
                            "prompt_type": "auto",
                            "return_comparison": True,
                            "use_langgraph": True
                        },
                        timeout=120.0
                    )
                    response.raise_for_status()
                    result_data = response.json()

                    # The actual result is in the 'result' field of the status response
                    status_url = f"{self.base_url}/api/workflow-status/{result_data['workflow_id']}"
                    for _ in range(60):  # Poll for 60 seconds
                        await asyncio.sleep(1)
                        status_response = await client.get(status_url)
                        status_data = status_response.json()
                        if status_data['status'] == 'completed':
                            result = status_data['result']
                            break
                    else:
                        raise Exception("Workflow did not complete in time")


                    processing_time = time.time() - start_time

                    test_result = {
                        "prompt_id": i + 1,
                        "domain": test_case["domain"],
                        "complexity": test_case["complexity"],
                        "processing_time_seconds": processing_time,
                        "quality_score": result["output"]["quality_score"],
                        "iterations_used": result["output"]["iterations_used"],
                        "success": True
                    }
                    results.append(test_result)

                except Exception as e:
                    logger.error(f"Error processing prompt {i+1}: {e}")
                    processing_time = time.time() - start_time
                    results.append({
                        "prompt_id": i + 1,
                        "domain": test_case["domain"],
                        "complexity": test_case["complexity"],
                        "processing_time_seconds": processing_time,
                        "success": False,
                        "error": str(e)
                    })

        successful_results = [r for r in results if r["success"]]
        if successful_results:
            processing_times = [r["processing_time_seconds"] for r in successful_results]
            quality_scores = [r["quality_score"] for r in successful_results]
            summary = {
                "total_prompts": len(results),
                "successful_prompts": len(successful_results),
                "success_rate": len(successful_results) / len(results),
                "average_processing_time": statistics.mean(processing_times),
                "median_processing_time": statistics.median(processing_times),
                "min_processing_time": min(processing_times),
                "max_processing_time": max(processing_times),
                "processing_time_stddev": statistics.stdev(processing_times) if len(processing_times) > 1 else 0,
                "average_quality_score": statistics.mean(quality_scores),
                "median_quality_score": statistics.median(quality_scores),
                "quality_score_stddev": statistics.stdev(quality_scores) if len(quality_scores) > 1 else 0,
                "benchmark_duration": time.time() - self.start_time
            }
        else:
            summary = {
                "total_prompts": len(results),
                "successful_prompts": 0,
                "success_rate": 0.0,
                "error": "No successful results"
            }

        return {"summary": summary, "individual_results": results}

    async def benchmark_cache_performance(self) -> Dict[str, Any]:
        """Benchmark cache performance for the LangGraph workflow."""
        logger.info("Running LangGraph cache performance benchmark...")

        cache_test_prompts = [
            "Write a function to sort an array",
            "Create a REST API endpoint",
            "Analyze data with Python",
            "Write a function to sort an array",  # Repeat
            "Create a REST API endpoint",        # Repeat
            "Analyze data with Python"           # Repeat
        ]

        cache_results = []
        start_time = time.time()

        async with httpx.AsyncClient() as client:
            for i, prompt in enumerate(cache_test_prompts):
                prompt_start = time.time()
                try:
                    response = await client.post(
                        f"{self.base_url}/api/process-prompt",
                        json={
                            "prompt": prompt,
                            "prompt_type": "auto",
                            "return_comparison": False,
                            "use_langgraph": True
                        },
                        timeout=120.0
                    )
                    response.raise_for_status()
                    result_data = response.json()

                    status_url = f"{self.base_url}/api/workflow-status/{result_data['workflow_id']}"
                    for _ in range(60):  # Poll for 60 seconds
                        await asyncio.sleep(1)
                        status_response = await client.get(status_url)
                        status_data = status_response.json()
                        if status_data['status'] == 'completed':
                            result = status_data['result']
                            break
                    else:
                        raise Exception("Workflow did not complete in time")

                    processing_time = time.time() - prompt_start

                    cache_results.append({
                        "prompt_id": i + 1,
                        "prompt": prompt,
                        "processing_time_seconds": processing_time,
                        "quality_score": result["output"]["quality_score"],
                        "is_cached": processing_time < 5.0  # Rough heuristic
                    })

                except Exception as e:
                    cache_results.append({
                        "prompt_id": i + 1,
                        "prompt": prompt,
                        "error": str(e),
                        "processing_time_seconds": time.time() - prompt_start
                    })

        processing_times = [r["processing_time_seconds"] for r in cache_results if "error" not in r]
        cached_requests = [r for r in cache_results if r.get("is_cached", False)]

        summary = {
            "total_requests": len(cache_results),
            "unique_prompts": len(set(r["prompt"] for r in cache_results if "error" not in r)),
            "repeated_prompts": len([r for r in cache_results if "error" not in r]) - len(set(r["prompt"] for r in cache_results if "error" not in r)),
            "cache_hit_rate": len(cached_requests) / len([r for r in cache_results if "error" not in r]) if cache_results else 0,
            "average_processing_time": statistics.mean(processing_times) if processing_times else 0,
            "cache_benchmark_duration": time.time() - start_time
        }

        return {"summary": summary, "results": cache_results}

    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive benchmark report."""
        report = {
            "benchmark_timestamp": datetime.now().isoformat(),
            "system_info": {
                "python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
                "platform": sys.platform,
            },
            "configuration": {
                "max_evaluation_iterations": settings.max_evaluation_iterations,
                "evaluation_threshold": settings.evaluation_threshold,
                "log_level": settings.log_level
            }
        }
        for benchmark_name, result in self.results.items():
            report[benchmark_name] = result
        return report

    def save_report(self, report: Dict[str, Any], filename: str = "langgraph_benchmark_report.json"):
        """Save benchmark report to file."""
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        logger.info(f"Benchmark report saved to {filename}")

    async def run_full_benchmark(self):
        """Run complete benchmark suite."""
        logger.info("Starting LangGraph performance benchmark...")
        await self.setup()
        self.results["langgraph_processing"] = await self.benchmark_langgraph_processing()
        self.results["cache_performance"] = await self.benchmark_cache_performance()
        report = self.generate_report()
        self.save_report(report)
        self.print_summary(report)
        return report

    def print_summary(self, report: Dict[str, Any]):
        """Print benchmark summary to console."""
        print("\n" + "="*80)
        print("🚀 LANGGRAPH PERFORMANCE BENCHMARK SUMMARY")
        print("="*80)
        if "langgraph_processing" in report:
            pp = report["langgraph_processing"]["summary"]
            print("\n📊 LangGraph Prompt Processing Performance:")
            print(f"   Success Rate: {pp['success_rate']*100:.1f}%")
            print(f"   Average Processing Time: {pp['average_processing_time']:.2f}s")
            print(f"   Average Quality Score: {pp['average_quality_score']:.3f}")

        if "cache_performance" in report:
            cp = report["cache_performance"]["summary"]
            print("\n💾 Cache Performance:")
            print(f"   Cache Hit Rate: {cp['cache_hit_rate']*100:.1f}%")
            print(f"   Average Processing Time: {cp['average_processing_time']:.2f}s")

        print(f"\n📄 Full report saved to: langgraph_benchmark_report.json")
        print("="*80)

async def main():
    """Main benchmark function."""
    print("🧪 LangGraph Performance Benchmark")
    print("="*70)
    if not IMPORTS_AVAILABLE:
        print("\n❌ Required modules not available. Please ensure the project is properly set up.")
        return None
    try:
        benchmark = LangGraphPerformanceBenchmark()
        await benchmark.run_full_benchmark()
        print("\n✅ Benchmark completed successfully!")
    except Exception as e:
        print(f"\n❌ Benchmark failed: {e}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⚠️  Benchmark interrupted by user")
