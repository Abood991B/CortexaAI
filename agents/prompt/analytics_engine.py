"""
Analytics Engine - Handles performance tracking and analytics for prompts.
"""

from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
import statistics

from .prompt_models import (
    Environment, PerformanceRecord
)
from .prompt_registry import PromptRegistry
from config.config import get_logger

logger = get_logger(__name__)


class AnalyticsEngine:
    """Engine for tracking and analyzing prompt performance."""

    def __init__(self, registry: PromptRegistry):
        """Initialize the analytics engine.

        Args:
            registry: PromptRegistry instance to analyze
        """
        self.registry = registry
        self.performance_cache: Dict[str, List[PerformanceRecord]] = defaultdict(list)

    def record_performance_metric(self, prompt_id: str, version: str,
                                environment: Environment, metric_name: str,
                                value: float, metadata: Dict[str, Any] = None,
                                request_id: str = None):
        """Record a performance metric for a prompt.

        Args:
            prompt_id: ID of the prompt
            version: Version of the prompt
            environment: Environment where metric was recorded
            metric_name: Name of the metric
            value: Metric value
            metadata: Additional metadata
            request_id: Optional request ID for tracking
        """
        # Create performance record
        record = PerformanceRecord(
            id=f"perf_{prompt_id}_{version}_{metric_name}_{int(datetime.now().timestamp())}",
            prompt_id=prompt_id,
            prompt_version=version,
            environment=environment,
            metrics={metric_name: value},
            recorded_at=datetime.now(),
            request_id=request_id
        )

        # Add metadata if provided
        if metadata:
            record.metrics.update(metadata)

        # Store in cache
        cache_key = f"{prompt_id}_{version}_{environment.value}"
        self.performance_cache[cache_key].append(record)

        # Update registry with performance data
        self.registry.record_performance(
            prompt_id=prompt_id,
            version=version,
            environment=environment,
            metrics={metric_name: value},
            request_id=request_id
        )

        logger.debug(f"Recorded {metric_name}={value} for prompt {prompt_id} v{version}")

    def get_performance_metrics(self, prompt_id: str, version: str = None,
                              environment: Environment = None,
                              metric_names: List[str] = None,
                              time_range_hours: int = 24) -> Dict[str, Any]:
        """Get performance metrics for a prompt.

        Args:
            prompt_id: ID of the prompt
            version: Optional specific version
            environment: Optional environment filter
            metric_names: Optional list of metric names to include
            time_range_hours: Time range in hours to analyze

        Returns:
            Dictionary with performance metrics and statistics
        """
        if version is None:
            prompt = self.registry.get_prompt(prompt_id)
            if not prompt:
                return {"error": "Prompt not found"}
            version = prompt.current_version

        # Get time range
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=time_range_hours)

        # Collect records
        all_records = []
        cache_keys = []

        if environment:
            cache_keys.append(f"{prompt_id}_{version}_{environment.value}")
        else:
            # Check all environments
            for env in [Environment.DEVELOPMENT, Environment.STAGING, Environment.PRODUCTION]:
                cache_keys.append(f"{prompt_id}_{version}_{env.value}")

        for cache_key in cache_keys:
            records = self.performance_cache[cache_key]
            # Filter by time range
            filtered_records = [
                r for r in records
                if r.recorded_at >= start_time and r.recorded_at <= end_time
            ]
            all_records.extend(filtered_records)

        if not all_records:
            return {
                "prompt_id": prompt_id,
                "version": version,
                "time_range_hours": time_range_hours,
                "total_records": 0,
                "metrics": {},
                "error": "No performance data found"
            }

        # Aggregate metrics
        metrics_data = defaultdict(list)

        for record in all_records:
            for metric_name, value in record.metrics.items():
                if metric_names and metric_name not in metric_names:
                    continue
                metrics_data[metric_name].append({
                    'value': value,
                    'timestamp': record.recorded_at.isoformat(),
                    'environment': record.environment.value,
                    'request_id': record.request_id
                })

        # Calculate statistics for each metric
        metrics_stats = {}
        for metric_name, values in metrics_data.items():
            numeric_values = [v['value'] for v in values if isinstance(v['value'], (int, float))]

            if numeric_values:
                stats_dict = {
                    'count': len(numeric_values),
                    'min': min(numeric_values),
                    'max': max(numeric_values),
                    'mean': statistics.mean(numeric_values),
                    'median': statistics.median(numeric_values),
                    'values': values  # Include raw values for detailed analysis
                }

                # Add standard deviation if we have enough data
                if len(numeric_values) > 1:
                    stats_dict['std_dev'] = statistics.stdev(numeric_values)

                # Add percentiles
                if len(numeric_values) >= 10:
                    sorted_values = sorted(numeric_values)
                    stats_dict['p95'] = sorted_values[int(len(sorted_values) * 0.95)]
                    stats_dict['p99'] = sorted_values[int(len(sorted_values) * 0.99)]

                metrics_stats[metric_name] = stats_dict

        return {
            "prompt_id": prompt_id,
            "version": version,
            "time_range_hours": time_range_hours,
            "total_records": len(all_records),
            "metrics": metrics_stats,
            "environments": list(set(r.environment.value for r in all_records)),
            "time_range": {
                "start": start_time.isoformat(),
                "end": end_time.isoformat()
            }
        }

    def get_prompt_performance_comparison(self, prompt_ids: List[str],
                                        environment: Environment = None,
                                        time_range_hours: int = 24) -> Dict[str, Any]:
        """Compare performance across multiple prompts.

        Args:
            prompt_ids: List of prompt IDs to compare
            environment: Optional environment filter
            time_range_hours: Time range in hours to analyze

        Returns:
            Comparison data across prompts
        """
        comparison_data = {}

        for prompt_id in prompt_ids:
            metrics = self.get_performance_metrics(
                prompt_id=prompt_id,
                environment=environment,
                time_range_hours=time_range_hours
            )

            if "error" not in metrics:
                comparison_data[prompt_id] = metrics

        if not comparison_data:
            return {"error": "No valid prompt data found for comparison"}

        # Create comparison summary
        summary = {
            "prompts_compared": list(comparison_data.keys()),
            "time_range_hours": time_range_hours,
            "environment": environment.value if environment else "all",
            "comparison": {}
        }

        # Compare common metrics
        all_metrics = set()
        for prompt_data in comparison_data.values():
            all_metrics.update(prompt_data.get('metrics', {}).keys())

        for metric_name in all_metrics:
            metric_comparison = {}
            for prompt_id, prompt_data in comparison_data.items():
                if metric_name in prompt_data.get('metrics', {}):
                    metric_comparison[prompt_id] = prompt_data['metrics'][metric_name]

            if len(metric_comparison) > 1:
                summary["comparison"][metric_name] = metric_comparison

        return summary

    def detect_performance_anomalies(self, prompt_id: str, version: str = None,
                                   environment: Environment = None,
                                   sensitivity: float = 2.0,
                                   time_range_hours: int = 24) -> Dict[str, Any]:
        """Detect performance anomalies for a prompt.

        Args:
            prompt_id: ID of the prompt to analyze
            version: Optional specific version
            environment: Optional environment filter
            sensitivity: Sensitivity for anomaly detection (standard deviations)
            time_range_hours: Time range in hours to analyze

        Returns:
            Dictionary with detected anomalies
        """
        metrics_data = self.get_performance_metrics(
            prompt_id=prompt_id,
            version=version,
            environment=environment,
            time_range_hours=time_range_hours
        )

        if "error" in metrics_data:
            return metrics_data

        anomalies = {
            "prompt_id": prompt_id,
            "version": metrics_data.get("version"),
            "time_range_hours": time_range_hours,
            "anomalies_detected": [],
            "analysis": {}
        }

        # Analyze each metric for anomalies
        for metric_name, metric_stats in metrics_data.get('metrics', {}).items():
            if 'std_dev' not in metric_stats or metric_stats['std_dev'] == 0:
                continue

            values = [v['value'] for v in metric_stats['values']
                     if isinstance(v['value'], (int, float))]

            if len(values) < 10:  # Need minimum data for anomaly detection
                continue

            mean = metric_stats['mean']
            std_dev = metric_stats['std_dev']

            # Find anomalies (values outside sensitivity * std_dev)
            anomalous_values = []
            for i, value in enumerate(values):
                if abs(value - mean) > sensitivity * std_dev:
                    anomalous_values.append({
                        'index': i,
                        'value': value,
                        'deviation': abs(value - mean),
                        'timestamp': metric_stats['values'][i]['timestamp']
                    })

            if anomalous_values:
                anomalies["anomalies_detected"].append({
                    "metric": metric_name,
                    "anomalous_values": anomalous_values,
                    "threshold": sensitivity * std_dev,
                    "mean": mean,
                    "std_dev": std_dev
                })

        anomalies["total_anomalies"] = len(anomalies["anomalies_detected"])
        return anomalies

    def generate_performance_report(self, prompt_id: str, version: str = None,
                                  environment: Environment = None,
                                  time_range_hours: int = 24) -> Dict[str, Any]:
        """Generate a comprehensive performance report.

        Args:
            prompt_id: ID of the prompt to report on
            version: Optional specific version
            environment: Optional environment filter
            time_range_hours: Time range in hours to analyze

        Returns:
            Comprehensive performance report
        """
        # Get basic metrics
        metrics = self.get_performance_metrics(
            prompt_id=prompt_id,
            version=version,
            environment=environment,
            time_range_hours=time_range_hours
        )

        if "error" in metrics:
            return metrics

        # Get anomalies
        anomalies = self.detect_performance_anomalies(
            prompt_id=prompt_id,
            version=version,
            environment=environment,
            time_range_hours=time_range_hours
        )

        # Get prompt info
        prompt = self.registry.get_prompt(prompt_id)
        if not prompt:
            return {"error": "Prompt not found"}

        report = {
            "report_generated_at": datetime.now().isoformat(),
            "prompt_info": {
                "id": prompt_id,
                "name": prompt.name,
                "current_version": prompt.current_version,
                "status": prompt.status.value,
                "domain": prompt.metadata.domain,
                "author": prompt.created_by
            },
            "analysis_period": {
                "hours": time_range_hours,
                "start": (datetime.now() - timedelta(hours=time_range_hours)).isoformat(),
                "end": datetime.now().isoformat()
            },
            "performance_metrics": metrics,
            "anomalies": anomalies,
            "recommendations": self._generate_recommendations(metrics, anomalies)
        }

        return report

    def _generate_recommendations(self, metrics: Dict[str, Any],
                                anomalies: Dict[str, Any]) -> List[str]:
        """Generate performance recommendations based on metrics and anomalies."""
        recommendations = []

        # Check for high latency
        if 'response_time' in metrics.get('metrics', {}):
            rt_stats = metrics['metrics']['response_time']
            if rt_stats.get('mean', 0) > 5.0:  # 5 seconds threshold
                recommendations.append("Consider optimizing prompt for faster response times")

        # Check for high error rates
        if 'error_rate' in metrics.get('metrics', {}):
            er_stats = metrics['metrics']['error_rate']
            if er_stats.get('mean', 0) > 0.1:  # 10% error rate threshold
                recommendations.append("High error rate detected - review prompt robustness")

        # Check for performance degradation
        if anomalies.get('total_anomalies', 0) > 0:
            recommendations.append(f"Performance anomalies detected in {anomalies['total_anomalies']} metrics - investigate root causes")

        # Check for low throughput
        if 'throughput' in metrics.get('metrics', {}):
            tp_stats = metrics['metrics']['throughput']
            if tp_stats.get('mean', 0) < 10:  # 10 requests/minute threshold
                recommendations.append("Low throughput detected - consider prompt optimization")

        if not recommendations:
            recommendations.append("Performance is within normal parameters")

        return recommendations

    def get_performance_trends(self, prompt_id: str, metric_name: str,
                             environment: Environment = None,
                             time_range_hours: int = 168) -> Dict[str, Any]:  # 1 week
        """Analyze performance trends over time.

        Args:
            prompt_id: ID of the prompt to analyze
            metric_name: Name of the metric to analyze
            environment: Optional environment filter
            time_range_hours: Time range in hours to analyze

        Returns:
            Trend analysis data
        """
        metrics_data = self.get_performance_metrics(
            prompt_id=prompt_id,
            environment=environment,
            metric_names=[metric_name],
            time_range_hours=time_range_hours
        )

        if "error" in metrics_data or metric_name not in metrics_data.get('metrics', {}):
            return {"error": f"No data found for metric '{metric_name}'"}

        metric_stats = metrics_data['metrics'][metric_name]

        # Sort values by timestamp
        values = sorted(metric_stats['values'], key=lambda x: x['timestamp'])

        if len(values) < 5:  # Need minimum data for trend analysis
            return {
                "metric": metric_name,
                "data_points": len(values),
                "error": "Insufficient data for trend analysis"
            }

        # Calculate trend (simple linear regression)
        timestamps = [datetime.fromisoformat(v['timestamp']).timestamp() for v in values]
        metric_values = [v['value'] for v in values if isinstance(v['value'], (int, float))]

        if len(metric_values) != len(timestamps):
            return {"error": "Data inconsistency in trend analysis"}

        # Simple trend calculation
        n = len(timestamps)
        if n < 2:
            trend = 0
        else:
            # Calculate slope using simple linear regression
            x_mean = sum(timestamps) / n
            y_mean = sum(metric_values) / n

            numerator = sum((timestamps[i] - x_mean) * (metric_values[i] - y_mean) for i in range(n))
            denominator = sum((timestamps[i] - x_mean) ** 2 for i in range(n))

            trend = numerator / denominator if denominator != 0 else 0

        # Determine trend direction
        if trend > 0.01:
            trend_direction = "increasing"
        elif trend < -0.01:
            trend_direction = "decreasing"
        else:
            trend_direction = "stable"

        # Calculate trend strength
        trend_strength = abs(trend) * len(timestamps)  # Rough strength indicator

        return {
            "metric": metric_name,
            "data_points": len(values),
            "time_range_hours": time_range_hours,
            "trend": {
                "slope": trend,
                "direction": trend_direction,
                "strength": trend_strength,
                "description": self._describe_trend(trend_direction, trend_strength)
            },
            "summary": {
                "start_value": metric_values[0] if metric_values else None,
                "end_value": metric_values[-1] if metric_values else None,
                "change_percent": ((metric_values[-1] - metric_values[0]) / metric_values[0] * 100) if metric_values and metric_values[0] != 0 else 0
            }
        }

    def _describe_trend(self, direction: str, strength: float) -> str:
        """Generate human-readable trend description."""
        if direction == "stable":
            return "Stable performance with no significant trend"

        strength_desc = "strong" if strength > 1.0 else "moderate" if strength > 0.5 else "weak"

        if direction == "increasing":
            return f"{strength_desc.capitalize()} increasing trend - performance may be degrading"
        else:
            return f"{strength_desc.capitalize()} decreasing trend - performance is improving"

    def export_analytics_data(self, prompt_id: str, format: str = "json",
                            time_range_hours: int = 24) -> Optional[str]:
        """Export analytics data for external analysis.

        Args:
            prompt_id: ID of the prompt to export
            format: Export format ('json' or 'csv')
            time_range_hours: Time range in hours to export

        Returns:
            Exported data as string, or None if prompt not found
        """
        metrics_data = self.get_performance_metrics(
            prompt_id=prompt_id,
            time_range_hours=time_range_hours
        )

        if "error" in metrics_data:
            return None

        if format == "json":
            import json
            return json.dumps(metrics_data, indent=2, ensure_ascii=False)
        elif format == "csv":
            return self._export_metrics_as_csv(metrics_data)

        return None

    def _export_metrics_as_csv(self, metrics_data: Dict[str, Any]) -> str:
        """Export metrics data as CSV format."""
        lines = ["Performance Metrics Export", ""]

        # Add header info
        lines.append(f"Prompt ID,{metrics_data['prompt_id']}")
        lines.append(f"Version,{metrics_data.get('version', 'N/A')}")
        lines.append(f"Time Range (hours),{metrics_data['time_range_hours']}")
        lines.append("")

        # Add each metric as a separate section
        for metric_name, metric_stats in metrics_data.get('metrics', {}).items():
            lines.append(f"Metric: {metric_name}")
            lines.append("Statistic,Value")

            # Add basic stats
            for stat_name in ['count', 'min', 'max', 'mean', 'median', 'std_dev', 'p95', 'p99']:
                if stat_name in metric_stats:
                    lines.append(f"{stat_name},{metric_stats[stat_name]}")

            lines.append("")

            # Add individual values
            lines.append("Timestamp,Value,Environment")
            for value_data in metric_stats.get('values', []):
                timestamp = value_data.get('timestamp', '')
                value = value_data.get('value', '')
                environment = value_data.get('environment', '')
                lines.append(f"{timestamp},{value},{environment}")

            lines.append("")

        return "\n".join(lines)
