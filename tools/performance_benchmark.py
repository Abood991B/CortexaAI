#!/usr/bin/env python3
"""
Performance Benchmarking Script for Multi-Agent Prompt Engineering System

This script provides comprehensive performance testing and benchmarking
to measure system capabilities, bottlenecks, and optimization opportunities.
"""

import asyncio
import time
import psutil
import json
import sys
import os
from datetime import datetime
from typing import Dict, List, Any
import statistics

# Add the parent directory to the path so we can import agents and config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    print("Warning: matplotlib/seaborn not available, plotting disabled")

try:
    from agents.coordinator import get_coordinator
    from config.config import get_logger, settings
    IMPORTS_AVAILABLE = True
except ImportError as e:
    print(f"Error importing required modules: {e}")
    IMPORTS_AVAILABLE = False

logger = get_logger(__name__)


class PerformanceBenchmark:
    """Comprehensive performance benchmarking suite."""

    def __init__(self):
        self.coordinator = None
        self.test_prompts = self._load_test_prompts()
        self.results = {}
        self.start_time = None

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
        self.coordinator = get_coordinator()
        self.start_time = time.time()

    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics."""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            return {
                "rss_mb": memory_info.rss / 1024 / 1024,
                "vms_mb": memory_info.vms / 1024 / 1024,
                "percent": process.memory_percent()
            }
        except ImportError:
            return {"error": "psutil not available"}

    def get_system_stats(self) -> Dict[str, Any]:
        """Get system performance statistics."""
        try:
            return {
                "cpu_percent": psutil.cpu_percent(interval=0.1),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_usage": psutil.disk_usage('/').percent
            }
        except ImportError:
            return {"error": "psutil not available"}

    async def benchmark_prompt_processing(self) -> Dict[str, Any]:
        """Benchmark prompt processing performance."""
        logger.info("Running prompt processing benchmark...")

        results = []
        memory_usage = []
        system_stats = []

        for i, test_case in enumerate(self.test_prompts):
            logger.info(f"Testing prompt {i+1}/{len(self.test_prompts)}: {test_case['prompt'][:50]}...")

            # Record memory before processing
            memory_before = self.get_memory_usage()
            system_before = self.get_system_stats()

            start_time = time.time()

            try:
                result = await self.coordinator.process_prompt(
                    prompt=test_case["prompt"],
                    prompt_type="auto",
                    return_comparison=True
                )

                processing_time = time.time() - start_time

                # Record memory after processing
                memory_after = self.get_memory_usage()
                system_after = self.get_system_stats()

                # Calculate metrics
                memory_delta = memory_after.get("rss_mb", 0) - memory_before.get("rss_mb", 0)

                test_result = {
                    "prompt_id": i + 1,
                    "domain": test_case["domain"],
                    "complexity": test_case["complexity"],
                    "processing_time_seconds": processing_time,
                    "quality_score": result["output"]["quality_score"],
                    "iterations_used": result["output"]["iterations_used"],
                    "memory_delta_mb": memory_delta,
                    "memory_peak_mb": memory_after.get("rss_mb", 0),
                    "system_cpu_percent": system_after.get("cpu_percent", 0),
                    "system_memory_percent": system_after.get("memory_percent", 0),
                    "success": True
                }

                results.append(test_result)
                memory_usage.append(memory_after.get("rss_mb", 0))
                system_stats.append(system_after)

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

        # Calculate summary statistics
        successful_results = [r for r in results if r["success"]]

        if successful_results:
            processing_times = [r["processing_time_seconds"] for r in successful_results]
            quality_scores = [r["quality_score"] for r in successful_results]
            memory_deltas = [r["memory_delta_mb"] for r in successful_results]

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
                "average_memory_delta": statistics.mean(memory_deltas),
                "peak_memory_usage": max(memory_usage) if memory_usage else 0,
                "benchmark_duration": time.time() - self.start_time
            }
        else:
            summary = {
                "total_prompts": len(results),
                "successful_prompts": 0,
                "success_rate": 0.0,
                "error": "No successful results"
            }

        return {
            "summary": summary,
            "individual_results": results,
            "memory_usage": memory_usage,
            "system_stats": system_stats
        }

    async def benchmark_cache_performance(self) -> Dict[str, Any]:
        """Benchmark cache performance."""
        logger.info("Running cache performance benchmark...")

        # Test with repeated prompts to check cache hits
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

        for i, prompt in enumerate(cache_test_prompts):
            prompt_start = time.time()

            try:
                result = await self.coordinator.process_prompt(
                    prompt=prompt,
                    prompt_type="auto",
                    return_comparison=False
                )

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

        # Calculate cache statistics
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

        return {
            "summary": summary,
            "results": cache_results
        }

    async def benchmark_security_performance(self) -> Dict[str, Any]:
        """Benchmark security filtering performance."""
        logger.info("Running security performance benchmark...")

        # Test prompts with varying security concerns
        security_test_prompts = [
            {
                "prompt": "Write a function to validate user input",
                "security_level": "low",
                "expected_safe": True
            },
            {
                "prompt": "Explain how to implement authentication in web applications",
                "security_level": "low",
                "expected_safe": True
            },
            {
                "prompt": "Show me how to create secure password hashing",
                "security_level": "low",
                "expected_safe": True
            },
            {
                "prompt": "How to bypass security restrictions in a system",
                "security_level": "high",
                "expected_safe": False
            },
            {
                "prompt": "Explain SQL injection vulnerabilities",
                "security_level": "medium",
                "expected_safe": True  # Educational context
            }
        ]

        security_results = []

        for i, test_case in enumerate(security_test_prompts):
            start_time = time.time()

            try:
                result = await self.coordinator.process_prompt(
                    prompt=test_case["prompt"],
                    prompt_type="auto",
                    return_comparison=False
                )

                processing_time = time.time() - start_time

                security_results.append({
                    "prompt_id": i + 1,
                    "security_level": test_case["security_level"],
                    "expected_safe": test_case["expected_safe"],
                    "processing_time_seconds": processing_time,
                    "quality_score": result["output"]["quality_score"],
                    "success": True
                })

            except Exception as e:
                processing_time = time.time() - start_time
                security_results.append({
                    "prompt_id": i + 1,
                    "security_level": test_case["security_level"],
                    "expected_safe": test_case["expected_safe"],
                    "processing_time_seconds": processing_time,
                    "error": str(e),
                    "success": False
                })

        # Calculate security statistics
        successful_results = [r for r in security_results if r["success"]]
        processing_times = [r["processing_time_seconds"] for r in security_results]

        summary = {
            "total_tests": len(security_results),
            "successful_tests": len(successful_results),
            "success_rate": len(successful_results) / len(security_results),
            "average_processing_time": statistics.mean(processing_times) if processing_times else 0,
            "security_filtering_efficiency": "high" if statistics.mean(processing_times) < 10 else "medium"
        }

        return {
            "summary": summary,
            "results": security_results
        }

    async def benchmark_concurrent_requests(self, num_concurrent: int = 5) -> Dict[str, Any]:
        """Benchmark concurrent request handling."""
        logger.info(f"Running concurrent requests benchmark with {num_concurrent} concurrent requests...")

        async def process_single_request(prompt: str, request_id: int):
            """Process a single request for concurrent testing."""
            start_time = time.time()
            try:
                result = await self.coordinator.process_prompt(
                    prompt=prompt,
                    prompt_type="auto",
                    return_comparison=False
                )
                processing_time = time.time() - start_time
                return {
                    "request_id": request_id,
                    "success": True,
                    "processing_time": processing_time,
                    "quality_score": result["output"]["quality_score"]
                }
            except Exception as e:
                processing_time = time.time() - start_time
                return {
                    "request_id": request_id,
                    "success": False,
                    "processing_time": processing_time,
                    "error": str(e)
                }

        # Create concurrent tasks
        tasks = []
        for i in range(num_concurrent):
            prompt = self.test_prompts[i % len(self.test_prompts)]["prompt"]
            task = process_single_request(prompt, i + 1)
            tasks.append(task)

        start_time = time.time()
        results = await asyncio.gather(*tasks)
        total_time = time.time() - start_time

        # Calculate concurrent statistics
        successful_results = [r for r in results if r["success"]]
        processing_times = [r["processing_time"] for r in results]

        summary = {
            "concurrent_requests": num_concurrent,
            "total_time_seconds": total_time,
            "average_time_per_request": total_time / num_concurrent,
            "successful_requests": len(successful_results),
            "success_rate": len(successful_results) / num_concurrent,
            "average_processing_time": statistics.mean(processing_times) if processing_times else 0,
            "throughput_requests_per_second": num_concurrent / total_time
        }

        return {
            "summary": summary,
            "results": results
        }

    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive benchmark report."""
        report = {
            "benchmark_timestamp": datetime.now().isoformat(),
            "system_info": {
                "python_version": f"{__import__('sys').version_info.major}.{__import__('sys').version_info.minor}",
                "platform": __import__('platform').platform(),
                "processor": __import__('platform').processor() or "Unknown"
            },
            "configuration": {
                "max_evaluation_iterations": settings.max_evaluation_iterations,
                "evaluation_threshold": settings.evaluation_threshold,
                "log_level": settings.log_level
            },
            "memory_baseline": self.get_memory_usage(),
            "system_baseline": self.get_system_stats()
        }

        # Add all benchmark results
        for benchmark_name, result in self.results.items():
            report[benchmark_name] = result

        return report

    def save_report(self, report: Dict[str, Any], filename: str = "benchmark_report.json"):
        """Save benchmark report to file."""
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        logger.info(f"Benchmark report saved to {filename}")

    async def run_full_benchmark(self):
        """Run complete benchmark suite."""
        logger.info("Starting comprehensive performance benchmark...")

        await self.setup()

        # Run all benchmarks
        self.results["prompt_processing"] = await self.benchmark_prompt_processing()
        self.results["cache_performance"] = await self.benchmark_cache_performance()
        self.results["security_performance"] = await self.benchmark_security_performance()
        self.results["concurrent_performance"] = await self.benchmark_concurrent_requests()

        # Generate and save report
        report = self.generate_report()
        self.save_report(report)

        # Print summary
        self.print_summary(report)

        return report

    def print_summary(self, report: Dict[str, Any]):
        """Print benchmark summary to console."""
        print("\n" + "="*80)
        print("🚀 PERFORMANCE BENCHMARK SUMMARY")
        print("="*80)

        if "prompt_processing" in report:
            pp = report["prompt_processing"]["summary"]
            print("\n📊 Prompt Processing Performance:")
            print(f"   Success Rate: {pp['success_rate']*100:.1f}%")
            print(f"   Average Processing Time: {pp['average_processing_time']:.2f}s")
            print(f"   Average Quality Score: {pp['average_quality_score']:.3f}")
            print(f"   Peak Memory Usage: {pp['peak_memory_usage']:.1f}MB")

        if "cache_performance" in report:
            cp = report["cache_performance"]["summary"]
            print("\n💾 Cache Performance:")
            print(f"   Cache Hit Rate: {cp['cache_hit_rate']*100:.1f}%")
            print(f"   Average Processing Time: {cp['average_processing_time']:.2f}s")

        if "security_performance" in report:
            sp = report["security_performance"]["summary"]
            print("\n🔒 Security Performance:")
            print(f"   Security Tests Success Rate: {sp['success_rate']*100:.1f}%")
            print(f"   Average Processing Time: {sp['average_processing_time']:.2f}s")
            print(f"   Security Filtering Efficiency: {sp['security_filtering_efficiency']}")

        if "concurrent_performance" in report:
            conc = report["concurrent_performance"]["summary"]
            print("\n⚡ Concurrent Performance:")
            print(f"   Throughput: {conc['throughput_requests_per_second']:.2f} req/sec")
            print(f"   Average Time per Request: {conc['average_time_per_request']:.2f}s")
            print(f"   Concurrent Requests Success Rate: {conc['success_rate']*100:.1f}%")

        print(f"\n📄 Full report saved to: benchmark_report.json")
        print("="*80)


async def main():
    """Main benchmark function."""
    print("🧪 Multi-Agent Prompt Engineering System - Performance Benchmark")
    print("="*70)

    if not IMPORTS_AVAILABLE:
        print("\n❌ Required modules not available. Please ensure the project is properly set up.")
        print("Missing imports: agents, config modules")
        return None

    try:
        benchmark = PerformanceBenchmark()
        report = await benchmark.run_full_benchmark()

        print("\n✅ Benchmark completed successfully!")
        return report
    except Exception as e:
        print(f"\n❌ Benchmark failed: {e}")
        return None


if __name__ == "__main__":
    try:
        report = asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⚠️  Benchmark interrupted by user")
    except Exception as e:
        print(f"\n❌ Benchmark failed: {e}")
        raise