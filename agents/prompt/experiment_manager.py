"""
Experiment Manager - Handles A/B testing for prompt optimization.
"""

import math
import random
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta

# Optional imports for statistical calculations
try:
    from scipy import stats
    import numpy as np
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    stats = None
    np = None

from .prompt_models import (
    Experiment, ExperimentVariant, ExperimentResult, ExperimentStatus,
    Environment
)
from .prompt_registry import PromptRegistry
from config.config import get_logger

logger = get_logger(__name__)


class ExperimentManager:
    """Manages A/B testing experiments for prompt optimization."""

    def __init__(self, registry: PromptRegistry):
        """Initialize the experiment manager.

        Args:
            registry: PromptRegistry instance to manage
        """
        self.registry = registry
        self.active_experiments: Dict[str, Experiment] = {}
        self._load_active_experiments()

    def _load_active_experiments(self):
        """Load active experiments from storage."""
        # In a real implementation, you'd load from persistent storage
        # For now, we'll initialize empty
        self.active_experiments = {}

    def create_experiment(self, name: str, description: str,
                         variants: List[Dict[str, Any]], created_by: str,
                         duration_days: int = 7) -> str:
        """Create a new A/B testing experiment.

        Args:
            name: Name of the experiment
            description: Description of the experiment
            variants: List of variant configurations
            created_by: User creating the experiment
            duration_days: How long to run the experiment

        Returns:
            Experiment ID
        """
        experiment_id = f"experiment_{int(datetime.now().timestamp())}_{name.replace(' ', '_')}"

        # Create variant objects
        experiment_variants = []
        for i, variant_data in enumerate(variants):
            variant = ExperimentVariant(
                id=f"variant_{experiment_id}_{i}",
                prompt_id=variant_data['prompt_id'],
                prompt_version=variant_data['prompt_version'],
                weight=variant_data.get('weight', 1.0 / len(variants)),
                name=variant_data.get('name', f"Variant {i}"),
                description=variant_data.get('description', '')
            )
            experiment_variants.append(variant)

        # Validate that all prompts exist
        for variant in experiment_variants:
            if not self.registry.get_prompt(variant.prompt_id):
                raise ValueError(f"Prompt {variant.prompt_id} not found")
            if not self.registry.get_prompt_version(variant.prompt_id, variant.prompt_version):
                raise ValueError(f"Version {variant.prompt_version} not found for prompt {variant.prompt_id}")

        # Create experiment
        experiment = Experiment(
            id=experiment_id,
            name=name,
            description=description,
            variants=experiment_variants,
            status=ExperimentStatus.DRAFT,
            start_date=datetime.now(),
            end_date=datetime.now() + timedelta(days=duration_days),
            results={},
            created_by=created_by,
            created_at=datetime.now()
        )

        self.active_experiments[experiment_id] = experiment
        logger.info(f"Created experiment '{name}' with ID {experiment_id}")

        return experiment_id

    def start_experiment(self, experiment_id: str) -> bool:
        """Start an experiment.

        Args:
            experiment_id: ID of the experiment to start

        Returns:
            True if started successfully, False otherwise
        """
        if experiment_id not in self.active_experiments:
            logger.error(f"Experiment {experiment_id} not found")
            return False

        experiment = self.active_experiments[experiment_id]

        if experiment.status != ExperimentStatus.DRAFT:
            logger.error(f"Experiment {experiment_id} is not in DRAFT status")
            return False

        # Validate all variants are available
        for variant in experiment.variants:
            prompt = self.registry.get_prompt(variant.prompt_id)
            if not prompt:
                logger.error(f"Prompt {variant.prompt_id} for experiment {experiment_id} not found")
                return False

        experiment.status = ExperimentStatus.RUNNING
        experiment.start_date = datetime.now()

        logger.info(f"Started experiment '{experiment.name}' with ID {experiment_id}")
        return True

    def stop_experiment(self, experiment_id: str, stopped_by: str = "system") -> bool:
        """Stop an experiment and calculate final results.

        Args:
            experiment_id: ID of the experiment to stop
            stopped_by: User stopping the experiment

        Returns:
            True if stopped successfully, False otherwise
        """
        if experiment_id not in self.active_experiments:
            logger.error(f"Experiment {experiment_id} not found")
            return False

        experiment = self.active_experiments[experiment_id]

        if experiment.status != ExperimentStatus.RUNNING:
            logger.error(f"Experiment {experiment_id} is not running")
            return False

        # Calculate final results
        self._calculate_experiment_results(experiment)

        experiment.status = ExperimentStatus.COMPLETED
        experiment.end_date = datetime.now()

        logger.info(f"Stopped experiment '{experiment.name}' with ID {experiment_id}")
        return True

    def get_next_variant(self, experiment_id: str) -> Optional[ExperimentVariant]:
        """Get the next variant for traffic allocation using weighted random selection.

        Args:
            experiment_id: ID of the experiment

        Returns:
            Selected variant or None if experiment not found/running
        """
        if experiment_id not in self.active_experiments:
            return None

        experiment = self.active_experiments[experiment_id]

        if experiment.status != ExperimentStatus.RUNNING:
            return None

        # Weighted random selection
        variants = experiment.variants
        weights = [v.weight for v in variants]

        # Normalize weights
        total_weight = sum(weights)
        if total_weight == 0:
            # Equal weights if all are 0
            weights = [1.0 / len(variants)] * len(variants)
        else:
            weights = [w / total_weight for w in weights]

        # Select variant
        selected = random.choices(variants, weights=weights, k=1)[0]
        return selected

    def record_experiment_event(self, experiment_id: str, variant_id: str,
                               event_type: str, value: float = 1.0,
                               metadata: Dict[str, Any] = None) -> bool:
        """Record an event for an experiment variant.

        Args:
            experiment_id: ID of the experiment
            variant_id: ID of the variant
            event_type: Type of event (e.g., 'impression', 'conversion', 'quality_score')
            value: Numeric value for the event
            metadata: Additional metadata

        Returns:
            True if recorded successfully, False otherwise
        """
        if experiment_id not in self.active_experiments:
            return False

        experiment = self.active_experiments[experiment_id]

        if experiment.status != ExperimentStatus.RUNNING:
            return False

        # Find the variant
        variant = None
        for v in experiment.variants:
            if v.id == variant_id:
                variant = v
                break

        if not variant:
            return False

        # Initialize result if not exists
        if variant_id not in experiment.results:
            experiment.results[variant_id] = ExperimentResult(
                variant_id=variant_id,
                impressions=0,
                conversions=0,
                conversion_rate=0.0,
                confidence_interval=(0.0, 0.0),
                statistical_significance=0.0,
                performance_metrics={}
            )

        result = experiment.results[variant_id]

        # Update metrics based on event type
        if event_type == 'impression':
            result.impressions += int(value)
        elif event_type == 'conversion':
            result.conversions += int(value)
        else:
            # Store in performance metrics
            if event_type not in result.performance_metrics:
                result.performance_metrics[event_type] = []
            result.performance_metrics[event_type].append({
                'value': value,
                'timestamp': datetime.now().isoformat(),
                'metadata': metadata or {}
            })

        # Recalculate conversion rate and statistics
        self._update_experiment_statistics(result)

        return True

    def _update_experiment_statistics(self, result: ExperimentResult):
        """Update statistical measures for an experiment result."""
        if result.impressions == 0:
            result.conversion_rate = 0.0
            result.confidence_interval = (0.0, 0.0)
            result.statistical_significance = 0.0
            return

        # Calculate conversion rate
        result.conversion_rate = result.conversions / result.impressions

        # Calculate confidence interval (Wilson score interval)
        if result.conversions > 0:
            z = 1.96  # 95% confidence
            n = result.impressions
            p = result.conversion_rate

            # Wilson score interval formula
            denominator = 1 + z**2 / n
            center = (p + z**2 / (2 * n)) / denominator
            spread = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denominator

            result.confidence_interval = (max(0, center - spread), min(1, center + spread))
        else:
            result.confidence_interval = (0.0, 0.0)

        # Statistical significance would require comparison with baseline
        # For now, set to 0 (would be calculated against control variant)
        result.statistical_significance = 0.0

    def _calculate_experiment_results(self, experiment: Experiment):
        """Calculate final results for an experiment."""
        if not experiment.results:
            return

        # Find control variant (first variant is typically control)
        control_variant = experiment.variants[0]
        control_result = experiment.results.get(control_variant.id)

        if not control_result:
            return

        # Calculate statistical significance for each variant
        for variant in experiment.variants[1:]:  # Skip control
            result = experiment.results.get(variant.id)
            if result and control_result.impressions > 0 and result.impressions > 0:
                # Chi-square test for statistical significance
                try:
                    observed = np.array([
                        [control_result.conversions, control_result.impressions - control_result.conversions],
                        [result.conversions, result.impressions - result.conversions]
                    ])

                    chi2, p_value, _, _ = stats.chi2_contingency(observed)
                    result.statistical_significance = p_value

                except Exception as e:
                    logger.warning(f"Failed to calculate statistical significance: {e}")
                    result.statistical_significance = 1.0  # No significance

    def get_experiment_results(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """Get current results for an experiment.

        Args:
            experiment_id: ID of the experiment

        Returns:
            Experiment results dictionary or None if not found
        """
        if experiment_id not in self.active_experiments:
            return None

        experiment = self.active_experiments[experiment_id]

        # Calculate current results if experiment is running
        if experiment.status == ExperimentStatus.RUNNING:
            self._calculate_experiment_results(experiment)

        results = {
            'experiment_id': experiment_id,
            'name': experiment.name,
            'description': experiment.description,
            'status': experiment.status.value,
            'start_date': experiment.start_date.isoformat(),
            'end_date': experiment.end_date.isoformat() if experiment.end_date else None,
            'variants': []
        }

        for variant in experiment.variants:
            variant_result = experiment.results.get(variant.id)
            if variant_result:
                variant_data = {
                    'id': variant.id,
                    'name': variant.name,
                    'description': variant.description,
                    'weight': variant.weight,
                    'prompt_id': variant.prompt_id,
                    'prompt_version': variant.prompt_version,
                    'impressions': variant_result.impressions,
                    'conversions': variant_result.conversions,
                    'conversion_rate': variant_result.conversion_rate,
                    'confidence_interval': variant_result.confidence_interval,
                    'statistical_significance': variant_result.statistical_significance,
                    'performance_metrics': variant_result.performance_metrics
                }
            else:
                variant_data = {
                    'id': variant.id,
                    'name': variant.name,
                    'description': variant.description,
                    'weight': variant.weight,
                    'prompt_id': variant.prompt_id,
                    'prompt_version': variant.prompt_version,
                    'impressions': 0,
                    'conversions': 0,
                    'conversion_rate': 0.0,
                    'confidence_interval': (0.0, 0.0),
                    'statistical_significance': 0.0,
                    'performance_metrics': {}
                }

            results['variants'].append(variant_data)

        return results

    def list_experiments(self, status: ExperimentStatus = None) -> List[Dict[str, Any]]:
        """List experiments with optional status filter.

        Args:
            status: Optional status filter

        Returns:
            List of experiment summaries
        """
        experiments = []

        for experiment in self.active_experiments.values():
            if status and experiment.status != status:
                continue

            experiments.append({
                'id': experiment.id,
                'name': experiment.name,
                'description': experiment.description,
                'status': experiment.status.value,
                'start_date': experiment.start_date.isoformat(),
                'end_date': experiment.end_date.isoformat() if experiment.end_date else None,
                'created_by': experiment.created_by,
                'variant_count': len(experiment.variants),
                'total_impressions': sum(
                    result.impressions for result in experiment.results.values()
                )
            })

        return experiments

    def get_best_variant(self, experiment_id: str) -> Optional[ExperimentVariant]:
        """Get the best performing variant based on conversion rate.

        Args:
            experiment_id: ID of the experiment

        Returns:
            Best performing variant or None if experiment not found
        """
        if experiment_id not in self.active_experiments:
            return None

        experiment = self.active_experiments[experiment_id]

        if not experiment.results:
            return None

        # Find variant with highest conversion rate
        best_variant = None
        best_conversion_rate = 0.0

        for variant in experiment.variants:
            result = experiment.results.get(variant.id)
            if result and result.conversion_rate > best_conversion_rate:
                best_conversion_rate = result.conversion_rate
                best_variant = variant

        return best_variant

    def get_experiment_statistics(self, experiment_id: str) -> Dict[str, Any]:
        """Get comprehensive statistics for an experiment.

        Args:
            experiment_id: ID of the experiment

        Returns:
            Experiment statistics dictionary
        """
        if experiment_id not in self.active_experiments:
            return {"error": "Experiment not found"}

        experiment = self.active_experiments[experiment_id]

        total_impressions = sum(result.impressions for result in experiment.results.values())
        total_conversions = sum(result.conversions for result in experiment.results.values())

        stats = {
            'experiment_id': experiment_id,
            'name': experiment.name,
            'status': experiment.status.value,
            'duration_days': (datetime.now() - experiment.start_date).days,
            'total_impressions': total_impressions,
            'total_conversions': total_conversions,
            'overall_conversion_rate': total_conversions / total_impressions if total_impressions > 0 else 0,
            'variants_count': len(experiment.variants),
            'variants_with_data': len(experiment.results),
            'created_at': experiment.created_at.isoformat(),
            'performance_by_variant': {}
        }

        # Per-variant statistics
        for variant in experiment.variants:
            result = experiment.results.get(variant.id)
            if result:
                stats['performance_by_variant'][variant.name] = {
                    'impressions': result.impressions,
                    'conversions': result.conversions,
                    'conversion_rate': result.conversion_rate,
                    'confidence_interval': result.confidence_interval,
                    'statistical_significance': result.statistical_significance,
                    'improvement_over_baseline': 0.0  # Would be calculated against control
                }

        return stats

    def export_experiment_results(self, experiment_id: str, format: str = "json") -> Optional[str]:
        """Export experiment results.

        Args:
            experiment_id: ID of the experiment
            format: Export format ('json' or 'csv')

        Returns:
            Exported results as string, or None if experiment not found
        """
        results = self.get_experiment_results(experiment_id)
        if not results:
            return None

        if format == "json":
            import json
            return json.dumps(results, indent=2, ensure_ascii=False)
        elif format == "csv":
            return self._export_as_csv(results)

        return None

    def _export_as_csv(self, results: Dict[str, Any]) -> str:
        """Export experiment results as CSV."""
        lines = ["Experiment Results", ""]
        lines.append(f"Experiment ID,{results['experiment_id']}")
        lines.append(f"Name,{results['name']}")
        lines.append(f"Status,{results['status']}")
        lines.append("")

        lines.append("Variant Results")
        lines.append("Variant,Impressions,Conversions,Conversion Rate,Confidence Interval,Statistical Significance")

        for variant in results['variants']:
            ci_low, ci_high = variant['confidence_interval']
            lines.append(f"{variant['name']},{variant['impressions']},{variant['conversions']},"
                        f"{variant['conversion_rate']:.4f},[{ci_low:.4f}, {ci_high:.4f}],{variant['statistical_significance']:.4f}")

        return "\n".join(lines)
