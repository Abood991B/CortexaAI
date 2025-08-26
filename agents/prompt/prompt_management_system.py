"""
Prompt Management System - Unified interface for all prompt management operations.
"""

from typing import Dict, List, Optional, Any, Union
from datetime import datetime

from .prompt_models import (
    PromptMetadata, PromptStatus, ExperimentStatus,
    Environment, DeploymentStatus
)
from .prompt_registry import PromptRegistry
from .version_manager import VersionManager
from .experiment_manager import ExperimentManager
from .deployment_manager import DeploymentManager
from .analytics_engine import AnalyticsEngine
from .template_manager import TemplateManager
from config.config import get_logger

logger = get_logger(__name__)


class PromptManagementSystem:
    """Unified interface for comprehensive prompt management."""

    def __init__(self, storage_path: str = None):
        """Initialize the prompt management system.

        Args:
            storage_path: Base path for storing all data
        """
        self.registry = PromptRegistry(storage_path)
        self.version_manager = VersionManager(self.registry)
        self.experiment_manager = ExperimentManager(self.registry)
        self.deployment_manager = DeploymentManager(self.registry)
        self.analytics_engine = AnalyticsEngine(self.registry)
        self.template_manager = TemplateManager()

        logger.info("Prompt Management System initialized")

    # ================================
    # PROMPT REGISTRY OPERATIONS
    # ================================

    def create_prompt(self, name: str, content: str, metadata: PromptMetadata,
                     created_by: str, commit_message: str = "") -> str:
        """Create a new prompt in the system."""
        return self.registry.create_prompt(name, content, metadata, created_by, commit_message)

    def get_prompt(self, prompt_id: str) -> Optional[Dict[str, Any]]:
        """Get a prompt by ID with full details."""
        prompt = self.registry.get_prompt(prompt_id)
        if not prompt:
            return None

        # Get current version details
        current_version = self.registry.get_prompt_version(prompt_id)
        versions = self.registry.get_version_history(prompt_id)

        return {
            'id': prompt.id,
            'name': prompt.name,
            'current_version': prompt.current_version,
            'status': prompt.status.value,
            'metadata': {
                'domain': prompt.metadata.domain,
                'strategy': prompt.metadata.strategy,
                'author': prompt.metadata.author,
                'tags': prompt.metadata.tags,
                'description': prompt.metadata.description,
                'performance_metrics': prompt.metadata.performance_metrics
            },
            'versions': {
                'count': len(versions),
                'latest': {
                    'version': current_version.version if current_version else None,
                    'content': current_version.content if current_version else None,
                    'created_at': current_version.created_at.isoformat() if current_version else None,
                    'commit_message': current_version.commit_message if current_version else None
                }
            },
            'created_at': prompt.created_at.isoformat(),
            'updated_at': prompt.updated_at.isoformat()
        }

    def list_prompts(self, status: str = None, domain: str = None,
                    author: str = None) -> List[Dict[str, Any]]:
        """List prompts with optional filtering."""
        status_enum = PromptStatus(status) if status else None

        prompts = self.registry.list_prompts(status_enum, domain, author)

        result = []
        for prompt in prompts:
            versions = self.registry.get_version_history(prompt.id)
            result.append({
                'id': prompt.id,
                'name': prompt.name,
                'current_version': prompt.current_version,
                'status': prompt.status.value,
                'domain': prompt.metadata.domain,
                'author': prompt.created_by,
                'versions_count': len(versions),
                'tags': prompt.metadata.tags,
                'created_at': prompt.created_at.isoformat(),
                'updated_at': prompt.updated_at.isoformat()
            })

        return result

    def update_prompt_status(self, prompt_id: str, status: str, updated_by: str) -> bool:
        """Update the status of a prompt."""
        status_enum = PromptStatus(status)
        return self.registry.update_prompt_status(prompt_id, status_enum, updated_by)

    # ================================
    # VERSION CONTROL OPERATIONS
    # ================================

    def create_version(self, prompt_id: str, content: str, created_by: str,
                      commit_message: str = "", bump_type: str = "patch") -> Optional[str]:
        """Create a new version of a prompt."""
        return self.registry.create_version(prompt_id, content, created_by, commit_message, bump_type)

    def get_version_history(self, prompt_id: str) -> List[Dict[str, Any]]:
        """Get the version history for a prompt."""
        versions = self.registry.get_version_history(prompt_id)

        result = []
        for version in versions:
            result.append({
                'version': version.version,
                'created_by': version.created_by,
                'created_at': version.created_at.isoformat(),
                'commit_message': version.commit_message,
                'parent_version': version.parent_version,
                'content_length': len(version.content),
                'performance_metrics': version.performance_metrics
            })

        return result

    def compare_versions(self, prompt_id: str, version1: str, version2: str) -> Dict[str, Any]:
        """Compare two versions of a prompt."""
        return self.version_manager.compare_versions(prompt_id, version1, version2)

    def rollback_to_version(self, prompt_id: str, target_version: str, rolled_back_by: str) -> bool:
        """Rollback a prompt to a specific version."""
        return self.registry.rollback_to_version(prompt_id, target_version, rolled_back_by)

    # ================================
    # EXPERIMENT OPERATIONS
    # ================================

    def create_experiment(self, name: str, description: str, variants: List[Dict[str, Any]],
                         created_by: str, duration_days: int = 7) -> str:
        """Create a new A/B testing experiment."""
        return self.experiment_manager.create_experiment(
            name, description, variants, created_by, duration_days
        )

    def start_experiment(self, experiment_id: str) -> bool:
        """Start an experiment."""
        return self.experiment_manager.start_experiment(experiment_id)

    def get_experiment_results(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """Get current results for an experiment."""
        return self.experiment_manager.get_experiment_results(experiment_id)

    def list_experiments(self, status: str = None) -> List[Dict[str, Any]]:
        """List experiments with optional status filter."""
        status_enum = ExperimentStatus(status) if status else None
        return self.experiment_manager.list_experiments(status_enum)

    def record_experiment_event(self, experiment_id: str, variant_id: str,
                               event_type: str, value: float = 1.0,
                               metadata: Dict[str, Any] = None) -> bool:
        """Record an event for an experiment variant."""
        return self.experiment_manager.record_experiment_event(
            experiment_id, variant_id, event_type, value, metadata
        )

    # ================================
    # DEPLOYMENT OPERATIONS
    # ================================

    def deploy_prompt(self, prompt_id: str, version: str, environment: str,
                     deployed_by: str, description: str = "") -> str:
        """Deploy a prompt version to a specific environment."""
        env_enum = Environment(environment)
        return self.deployment_manager.deploy_prompt(
            prompt_id, version, env_enum, deployed_by, description
        )

    def get_deployment_status(self, deployment_id: str) -> Optional[Dict[str, Any]]:
        """Get the status of a deployment."""
        return self.deployment_manager.get_deployment_status(deployment_id)

    def list_deployments(self, prompt_id: str = None, environment: str = None,
                        status: str = None) -> List[Dict[str, Any]]:
        """List deployments with optional filters."""
        env_enum = Environment(environment) if environment else None
        status_enum = DeploymentStatus(status) if status else None
        return self.deployment_manager.list_deployments(prompt_id, env_enum, status_enum)

    def rollback_deployment(self, prompt_id: str, environment: str,
                           rolled_back_by: str, reason: str = "") -> Optional[str]:
        """Rollback to the previous version."""
        env_enum = Environment(environment)
        return self.deployment_manager.rollback_deployment(prompt_id, env_enum, rolled_back_by, reason)

    # ================================
    # ANALYTICS OPERATIONS
    # ================================

    def record_performance_metric(self, prompt_id: str, version: str, environment: str,
                                metric_name: str, value: float,
                                metadata: Dict[str, Any] = None, request_id: str = None):
        """Record a performance metric for a prompt."""
        env_enum = Environment(environment)
        self.analytics_engine.record_performance_metric(
            prompt_id, version, env_enum, metric_name, value, metadata, request_id
        )

    def get_performance_metrics(self, prompt_id: str, version: str = None,
                              environment: str = None, metric_names: List[str] = None,
                              time_range_hours: int = 24) -> Dict[str, Any]:
        """Get performance metrics for a prompt."""
        env_enum = Environment(environment) if environment else None
        return self.analytics_engine.get_performance_metrics(
            prompt_id, version, env_enum, metric_names, time_range_hours
        )

    def generate_performance_report(self, prompt_id: str, version: str = None,
                                  environment: str = None, time_range_hours: int = 24) -> Dict[str, Any]:
        """Generate a comprehensive performance report."""
        env_enum = Environment(environment) if environment else None
        return self.analytics_engine.generate_performance_report(
            prompt_id, version, env_enum, time_range_hours
        )

    def detect_performance_anomalies(self, prompt_id: str, version: str = None,
                                   environment: str = None, sensitivity: float = 2.0,
                                   time_range_hours: int = 24) -> Dict[str, Any]:
        """Detect performance anomalies for a prompt."""
        env_enum = Environment(environment) if environment else None
        return self.analytics_engine.detect_performance_anomalies(
            prompt_id, version, env_enum, sensitivity, time_range_hours
        )

    # ================================
    # TEMPLATE OPERATIONS
    # ================================

    def create_template(self, name: str, content: str, metadata: PromptMetadata,
                       parent_template: str = None, variables: Dict[str, Any] = None,
                       tags: List[str] = None) -> str:
        """Create a new template."""
        return self.template_manager.create_template(
            name, content, metadata, parent_template, variables, tags
        )

    def render_template(self, template_id: str, variables: Dict[str, Any] = None,
                       context: Dict[str, Any] = None) -> Optional[str]:
        """Render a template with variable substitution."""
        return self.template_manager.render_template(template_id, variables, context)

    def list_templates(self, domain: str = None, parent: str = None,
                      tags: List[str] = None) -> List[Dict[str, Any]]:
        """List templates with optional filtering."""
        return self.template_manager.list_templates(domain, parent, tags)

    def validate_template(self, template_id: str) -> Dict[str, Any]:
        """Validate a template for syntax and variable consistency."""
        return self.template_manager.validate_template(template_id)

    # ================================
    # INTEGRATED OPERATIONS
    # ================================

    def get_prompt_dashboard(self, prompt_id: str) -> Dict[str, Any]:
        """Get a comprehensive dashboard for a prompt."""
        prompt = self.get_prompt(prompt_id)
        if not prompt:
            return {"error": "Prompt not found"}

        # Get performance metrics for the last 24 hours
        performance = self.get_performance_metrics(prompt_id, time_range_hours=24)

        # Get recent deployments
        deployments = self.list_deployments(prompt_id=prompt_id)

        # Get active experiments
        experiments = []
        for exp in self.list_experiments():
            # Check if this prompt is used in the experiment
            exp_details = self.get_experiment_results(exp['id'])
            if exp_details:
                for variant in exp_details.get('variants', []):
                    if variant['prompt_id'] == prompt_id:
                        experiments.append(exp)
                        break

        # Get version statistics
        version_stats = self.version_manager.get_version_statistics(prompt_id)

        return {
            'prompt': prompt,
            'performance': performance,
            'recent_deployments': deployments[:5],  # Last 5 deployments
            'active_experiments': experiments,
            'version_statistics': version_stats,
            'dashboard_generated_at': datetime.now().isoformat()
        }

    def create_prompt_from_template(self, template_id: str, name: str,
                                   variables: Dict[str, Any], metadata: PromptMetadata,
                                   created_by: str) -> Optional[str]:
        """Create a prompt from a template."""
        # Render the template
        rendered_content = self.render_template(template_id, variables)
        if not rendered_content:
            logger.error(f"Failed to render template {template_id}")
            return None

        # Create the prompt
        prompt_id = self.create_prompt(
            name=name,
            content=rendered_content,
            metadata=metadata,
            created_by=created_by,
            commit_message=f"Created from template {template_id}"
        )

        logger.info(f"Created prompt {prompt_id} from template {template_id}")
        return prompt_id

    def optimize_prompt_workflow(self, prompt_id: str, optimization_goal: str,
                               optimizer: str) -> Dict[str, Any]:
        """Run a complete prompt optimization workflow."""
        prompt = self.get_prompt(prompt_id)
        if not prompt:
            return {"error": "Prompt not found"}

        workflow = {
            'prompt_id': prompt_id,
            'optimization_goal': optimization_goal,
            'steps': [],
            'results': {},
            'started_at': datetime.now().isoformat()
        }

        try:
            # Step 1: Create baseline performance measurement
            workflow['steps'].append("baseline_measurement")
            baseline = self.get_performance_metrics(prompt_id, time_range_hours=1)
            workflow['results']['baseline'] = baseline

            # Step 2: Create experiment for optimization
            workflow['steps'].append("create_experiment")

            # Get current version
            current_version = self.registry.get_prompt_version(prompt_id)
            if not current_version:
                raise ValueError("No current version found")

            # Create variants (simplified - in real implementation, you'd generate multiple variants)
            variants = [
                {
                    'prompt_id': prompt_id,
                    'prompt_version': current_version.version,
                    'name': 'Control',
                    'weight': 0.5
                },
                {
                    'prompt_id': prompt_id,
                    'prompt_version': current_version.version,  # Would be optimized version
                    'name': 'Optimized',
                    'weight': 0.5
                }
            ]

            experiment_id = self.create_experiment(
                name=f"Optimization: {prompt['name']}",
                description=f"Testing {optimization_goal} optimization",
                variants=variants,
                created_by=optimizer,
                duration_days=1
            )

            workflow['results']['experiment_id'] = experiment_id

            # Step 3: Start the experiment
            workflow['steps'].append("start_experiment")
            self.start_experiment(experiment_id)

            workflow['completed_at'] = datetime.now().isoformat()
            workflow['status'] = 'running'

        except Exception as e:
            workflow['error'] = str(e)
            workflow['status'] = 'failed'
            workflow['completed_at'] = datetime.now().isoformat()
            logger.error(f"Optimization workflow failed: {e}")

        return workflow

    def get_system_health(self) -> Dict[str, Any]:
        """Get the overall health status of the prompt management system."""
        health = {
            'timestamp': datetime.now().isoformat(),
            'components': {}
        }

        # Check registry health
        try:
            prompts_count = len(self.registry.list_prompts())
            health['components']['registry'] = {
                'status': 'healthy',
                'prompts_count': prompts_count
            }
        except Exception as e:
            health['components']['registry'] = {
                'status': 'unhealthy',
                'error': str(e)
            }

        # Check experiments
        try:
            experiments_count = len(self.list_experiments())
            health['components']['experiments'] = {
                'status': 'healthy',
                'active_experiments': experiments_count
            }
        except Exception as e:
            health['components']['experiments'] = {
                'status': 'unhealthy',
                'error': str(e)
            }

        # Check deployments
        try:
            deployments_count = len(self.list_deployments())
            health['components']['deployments'] = {
                'status': 'healthy',
                'total_deployments': deployments_count
            }
        except Exception as e:
            health['components']['deployments'] = {
                'status': 'unhealthy',
                'error': str(e)
            }

        # Check templates
        try:
            templates_count = len(self.list_templates())
            health['components']['templates'] = {
                'status': 'healthy',
                'templates_count': templates_count
            }
        except Exception as e:
            health['components']['templates'] = {
                'status': 'unhealthy',
                'error': str(e)
            }

        # Overall health
        unhealthy_components = [c for c in health['components'].values()
                              if c['status'] == 'unhealthy']
        health['overall_status'] = 'unhealthy' if unhealthy_components else 'healthy'

        return health

    def export_system_data(self, format: str = "json") -> Dict[str, Any]:
        """Export all system data."""
        export_data = {
            'export_timestamp': datetime.now().isoformat(),
            'system_health': self.get_system_health(),
            'data': {}
        }

        # Export prompts
        export_data['data']['prompts'] = []
        for prompt_summary in self.list_prompts():
            prompt_details = self.get_prompt(prompt_summary['id'])
            if prompt_details:
                export_data['data']['prompts'].append(prompt_details)

        # Export experiments
        export_data['data']['experiments'] = []
        for exp in self.list_experiments():
            exp_details = self.get_experiment_results(exp['id'])
            if exp_details:
                export_data['data']['experiments'].append(exp_details)

        # Export templates
        export_data['data']['templates'] = self.list_templates()

        # Export deployments
        export_data['data']['deployments'] = self.list_deployments()

        return export_data

    # ================================
    # COORDINATOR INTEGRATION METHODS
    # ================================

    def get_best_prompt_for_domain(self, domain: str) -> Optional[Dict[str, Any]]:
        """Get the best performing prompt for a domain."""
        try:
            # Get all prompts for the domain
            prompts = self.list_prompts(domain=domain)

            if not prompts:
                return None

            # For now, return the most recently updated prompt
            # In a real implementation, you'd analyze performance metrics
            best_prompt = max(prompts, key=lambda p: p['updated_at'])

            # Get the current version content
            current_version = self.registry.get_prompt_version(best_prompt['id'])
            if not current_version:
                return None

            return {
                'prompt_id': best_prompt['id'],
                'name': best_prompt['name'],
                'content': current_version.content,
                'version': current_version.version,
                'domain': domain,
                'performance_score': 0.9,  # Would be calculated from analytics
                'metadata': best_prompt
            }

        except Exception as e:
            logger.warning(f"Failed to get best prompt for domain {domain}: {e}")
            return None

    def record_performance(self, domain: str, prompt_content: str,
                          performance_score: float, metadata: Dict[str, Any] = None):
        """Record performance metrics for a prompt."""
        try:
            # Find the prompt by content (simplified - in real implementation, use ID)
            prompts = self.list_prompts(domain=domain)
            matching_prompt = None

            for prompt in prompts:
                current_version = self.registry.get_prompt_version(prompt['id'])
                if current_version and current_version.content.strip() == prompt_content.strip():
                    matching_prompt = prompt
                    break

            if not matching_prompt:
                # Create a new prompt for performance tracking
                prompt_id = self.create_prompt(
                    name=f"Performance Tracked Prompt - {domain}",
                    content=prompt_content,
                    metadata=PromptMetadata(
                        domain=domain,
                        strategy="performance_tracked",
                        author="system",
                        tags=["performance_tracked"],
                        description="Automatically created for performance tracking"
                    ),
                    created_by="system",
                    commit_message="Created for performance tracking"
                )
                matching_prompt = {'id': prompt_id}

            # Record the performance metric
            current_version = self.registry.get_prompt_version(matching_prompt['id'])
            if current_version:
                self.record_performance_metric(
                    prompt_id=matching_prompt['id'],
                    version=current_version.version,
                    environment="development",  # Default environment
                    metric_name="quality_score",
                    value=performance_score,
                    metadata=metadata or {}
                )

        except Exception as e:
            logger.warning(f"Failed to record performance for domain {domain}: {e}")

    def get_registry_stats(self) -> Dict[str, Any]:
        """Get registry statistics."""
        try:
            prompts = self.list_prompts()
            total_prompts = len(prompts)

            # Count by domain
            domain_count = {}
            for prompt in prompts:
                domain = prompt.get('domain', 'unknown')
                domain_count[domain] = domain_count.get(domain, 0) + 1

            # Count by status
            status_count = {}
            for prompt in prompts:
                status = prompt.get('status', 'unknown')
                status_count[status] = status_count.get(status, 0) + 1

            return {
                'total_prompts': total_prompts,
                'prompts_by_domain': domain_count,
                'prompts_by_status': status_count,
                'storage_info': {
                    'prompts_file': 'data/prompts/prompts.json',
                    'versions_file': 'data/prompts/versions.json'
                }
            }

        except Exception as e:
            logger.error(f"Failed to get registry stats: {e}")
            return {'error': str(e)}

    def get_experiment_stats(self) -> Dict[str, Any]:
        """Get experiment statistics."""
        try:
            experiments = self.list_experiments()
            total_experiments = len(experiments)

            # Count by status
            status_count = {}
            for exp in experiments:
                status = exp.get('status', 'unknown')
                status_count[status] = status_count.get(status, 0) + 1

            return {
                'total_experiments': total_experiments,
                'experiments_by_status': status_count,
                'active_experiments': status_count.get('running', 0)
            }

        except Exception as e:
            logger.error(f"Failed to get experiment stats: {e}")
            return {'error': str(e)}

    def get_deployment_stats(self) -> Dict[str, Any]:
        """Get deployment statistics."""
        try:
            deployments = self.list_deployments()
            total_deployments = len(deployments)

            # Count by environment
            env_count = {}
            for deployment in deployments:
                environment = deployment.get('environment', 'unknown')
                env_count[environment] = env_count.get(environment, 0) + 1

            # Count by status
            status_count = {}
            for deployment in deployments:
                status = deployment.get('status', 'unknown')
                status_count[status] = status_count.get(status, 0) + 1

            return {
                'total_deployments': total_deployments,
                'deployments_by_environment': env_count,
                'deployments_by_status': status_count
            }

        except Exception as e:
            logger.error(f"Failed to get deployment stats: {e}")
            return {'error': str(e)}

    def get_analytics_stats(self) -> Dict[str, Any]:
        """Get analytics statistics."""
        try:
            # This would normally query the analytics engine
            return {
                'metrics_collected': 0,  # Would be actual count
                'performance_trends': {},
                'anomaly_detection_enabled': False
            }

        except Exception as e:
            logger.error(f"Failed to get analytics stats: {e}")
            return {'error': str(e)}

    def get_available_templates(self) -> List[Dict[str, Any]]:
        """Get list of available templates."""
        try:
            templates = self.list_templates()

            # Add any coordinator-specific templates
            coordinator_templates = [
                {
                    'id': 'coordinator_fallback_template',
                    'name': 'Coordinator Fallback Template',
                    'domain': 'general',
                    'description': 'Template for graceful fallback scenarios',
                    'variables': ['prompt_content', 'domain', 'fallback_reason'],
                    'tags': ['coordinator', 'fallback', 'system']
                }
            ]

            return templates + coordinator_templates

        except Exception as e:
            logger.warning(f"Failed to get available templates: {e}")
            return []
