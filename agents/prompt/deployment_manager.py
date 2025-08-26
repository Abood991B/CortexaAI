"""
Deployment Manager - Handles prompt deployments and rollback mechanisms.
"""

from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from pathlib import Path

from .prompt_models import (
    Deployment, DeploymentStatus, Environment,
    PromptStatus
)
from .prompt_registry import PromptRegistry
from config.config import get_logger

logger = get_logger(__name__)


class DeploymentManager:
    """Manages prompt deployments across different environments."""

    def __init__(self, registry: PromptRegistry):
        """Initialize the deployment manager.

        Args:
            registry: PromptRegistry instance to manage
        """
        self.registry = registry
        self.active_deployments: Dict[str, Deployment] = {}
        self._load_active_deployments()

    def _load_active_deployments(self):
        """Load active deployments from storage."""
        # In a real implementation, you'd load from persistent storage
        # For now, we'll initialize empty
        self.active_deployments = {}

    def deploy_prompt(self, prompt_id: str, version: str, environment: Environment,
                     deployed_by: str, description: str = "") -> str:
        """Deploy a prompt version to a specific environment.

        Args:
            prompt_id: ID of the prompt to deploy
            version: Version to deploy
            environment: Target environment
            deployed_by: User performing the deployment
            description: Optional deployment description

        Returns:
            Deployment ID
        """
        # Validate prompt and version exist
        prompt = self.registry.get_prompt(prompt_id)
        if not prompt:
            raise ValueError(f"Prompt {prompt_id} not found")

        version_obj = self.registry.get_prompt_version(prompt_id, version)
        if not version_obj:
            raise ValueError(f"Version {version} not found for prompt {prompt_id}")

        # Check if prompt is active
        if prompt.status != PromptStatus.ACTIVE:
            logger.warning(f"Deploying prompt {prompt_id} which is not in ACTIVE status")

        # Create deployment ID
        deployment_id = f"deployment_{prompt_id}_{version}_{environment.value}_{int(datetime.now().timestamp())}"

        # Check for existing deployment to this environment
        existing_deployment = self._get_current_deployment(prompt_id, environment)
        rollback_version = None
        if existing_deployment:
            rollback_version = existing_deployment.prompt_version
            logger.info(f"Existing deployment found for {prompt_id} in {environment.value}, "
                       f"rollback version will be {rollback_version}")

        # Create deployment record
        deployment = Deployment(
            id=deployment_id,
            prompt_id=prompt_id,
            prompt_version=version,
            environment=environment,
            status=DeploymentStatus.PENDING,
            deployed_by=deployed_by,
            deployed_at=datetime.now(),
            rollback_version=rollback_version,
            error_message=""
        )

        # Add to active deployments
        self.active_deployments[deployment_id] = deployment

        # Start deployment process
        self._execute_deployment(deployment)

        logger.info(f"Initiated deployment of prompt {prompt_id} version {version} "
                   f"to {environment.value} by {deployed_by}")

        return deployment_id

    def _execute_deployment(self, deployment: Deployment):
        """Execute the deployment process.

        Args:
            deployment: Deployment to execute
        """
        try:
            # Set status to in progress
            deployment.status = DeploymentStatus.IN_PROGRESS

            # Perform deployment steps
            self._validate_deployment(deployment)
            self._prepare_deployment(deployment)
            self._apply_deployment(deployment)
            self._verify_deployment(deployment)

            # Mark as successful
            deployment.status = DeploymentStatus.SUCCESSFUL

            logger.info(f"Successfully deployed {deployment.prompt_id} version "
                       f"{deployment.prompt_version} to {deployment.environment.value}")

        except Exception as e:
            deployment.status = DeploymentStatus.FAILED
            deployment.error_message = str(e)
            logger.error(f"Deployment {deployment.id} failed: {e}")

    def _validate_deployment(self, deployment: Deployment):
        """Validate deployment prerequisites."""
        # Check if the prompt version exists and is valid
        version_obj = self.registry.get_prompt_version(
            deployment.prompt_id,
            deployment.prompt_version
        )

        if not version_obj:
            raise ValueError(f"Prompt version {deployment.prompt_version} not found")

        # Check if version has required metadata
        if not version_obj.metadata.domain:
            raise ValueError("Prompt version missing required domain metadata")

        # Validate environment-specific requirements
        if deployment.environment == Environment.PRODUCTION:
            # Production deployments require approval
            if not self._has_production_approval(deployment):
                raise ValueError("Production deployment requires approval")

        logger.info(f"Validated deployment {deployment.id}")

    def _prepare_deployment(self, deployment: Deployment):
        """Prepare for deployment."""
        # Create backup of current version if exists
        current_deployment = self._get_current_deployment(
            deployment.prompt_id,
            deployment.environment
        )

        if current_deployment:
            logger.info(f"Prepared rollback for deployment {deployment.id} "
                       f"to version {current_deployment.prompt_version}")

        # Prepare deployment artifacts
        # In a real implementation, this might involve:
        # - Creating deployment packages
        # - Updating configuration files
        # - Preparing container images
        # - Setting up load balancers

        logger.info(f"Prepared deployment artifacts for {deployment.id}")

    def _apply_deployment(self, deployment: Deployment):
        """Apply the deployment."""
        # Simulate deployment time
        import time
        time.sleep(0.1)  # Simulate deployment overhead

        # In a real implementation, this would involve:
        # - Updating load balancers
        # - Deploying to containers
        # - Updating service discovery
        # - Running health checks

        logger.info(f"Applied deployment {deployment.id}")

    def _verify_deployment(self, deployment: Deployment):
        """Verify deployment success."""
        # Perform health checks
        if not self._run_health_checks(deployment):
            raise ValueError("Health checks failed after deployment")

        # Validate that the deployed version is responding correctly
        if not self._validate_deployed_version(deployment):
            raise ValueError("Deployed version validation failed")

        logger.info(f"Verified deployment {deployment.id}")

    def _has_production_approval(self, deployment: Deployment) -> bool:
        """Check if deployment has required production approval."""
        # In a real implementation, this would check an approval workflow
        # For now, we'll simulate approval based on deployment metadata
        return True  # Simplified for demonstration

    def _run_health_checks(self, deployment: Deployment) -> bool:
        """Run health checks on deployed version."""
        # In a real implementation, this would:
        # - Check service health endpoints
        # - Validate response times
        # - Test critical functionality
        return True  # Simplified for demonstration

    def _validate_deployed_version(self, deployment: Deployment) -> bool:
        """Validate that the deployed version is working correctly."""
        # In a real implementation, this would:
        # - Test prompt generation with sample inputs
        # - Validate output quality
        # - Check performance metrics
        return True  # Simplified for demonstration

    def rollback_deployment(self, prompt_id: str, environment: Environment,
                           rolled_back_by: str, reason: str = "") -> Optional[str]:
        """Rollback to the previous version.

        Args:
            prompt_id: ID of the prompt to rollback
            environment: Environment to rollback
            rolled_back_by: User performing the rollback
            reason: Reason for rollback

        Returns:
            Deployment ID if rollback initiated, None otherwise
        """
        current_deployment = self._get_current_deployment(prompt_id, environment)
        if not current_deployment:
            logger.error(f"No current deployment found for {prompt_id} in {environment.value}")
            return None

        if not current_deployment.rollback_version:
            logger.error(f"No rollback version available for {prompt_id} in {environment.value}")
            return None

        # Create rollback deployment
        rollback_deployment_id = f"rollback_{prompt_id}_{current_deployment.rollback_version}_{environment.value}_{int(datetime.now().timestamp())}"

        rollback_deployment = Deployment(
            id=rollback_deployment_id,
            prompt_id=prompt_id,
            prompt_version=current_deployment.rollback_version,
            environment=environment,
            status=DeploymentStatus.PENDING,
            deployed_by=rolled_back_by,
            deployed_at=datetime.now(),
            rollback_version=None,  # No rollback for rollback deployments
            error_message=f"Rollback reason: {reason}"
        )

        # Add to active deployments
        self.active_deployments[rollback_deployment_id] = rollback_deployment

        # Execute rollback
        self._execute_deployment(rollback_deployment)

        logger.info(f"Initiated rollback of {prompt_id} in {environment.value} "
                   f"to version {current_deployment.rollback_version} by {rolled_back_by}")

        return rollback_deployment_id

    def get_deployment_status(self, deployment_id: str) -> Optional[Dict[str, Any]]:
        """Get the status of a deployment.

        Args:
            deployment_id: ID of the deployment

        Returns:
            Deployment status dictionary or None if not found
        """
        deployment = self.active_deployments.get(deployment_id)
        if not deployment:
            return None

        return {
            'id': deployment.id,
            'prompt_id': deployment.prompt_id,
            'prompt_version': deployment.prompt_version,
            'environment': deployment.environment.value,
            'status': deployment.status.value,
            'deployed_by': deployment.deployed_by,
            'deployed_at': deployment.deployed_at.isoformat(),
            'rollback_version': deployment.rollback_version,
            'error_message': deployment.error_message
        }

    def list_deployments(self, prompt_id: str = None, environment: Environment = None,
                        status: DeploymentStatus = None) -> List[Dict[str, Any]]:
        """List deployments with optional filters.

        Args:
            prompt_id: Optional filter by prompt ID
            environment: Optional filter by environment
            status: Optional filter by status

        Returns:
            List of deployment summaries
        """
        deployments = []

        for deployment in self.active_deployments.values():
            # Apply filters
            if prompt_id and deployment.prompt_id != prompt_id:
                continue
            if environment and deployment.environment != environment:
                continue
            if status and deployment.status != status:
                continue

            deployments.append({
                'id': deployment.id,
                'prompt_id': deployment.prompt_id,
                'prompt_version': deployment.prompt_version,
                'environment': deployment.environment.value,
                'status': deployment.status.value,
                'deployed_by': deployment.deployed_by,
                'deployed_at': deployment.deployed_at.isoformat(),
                'rollback_version': deployment.rollback_version
            })

        # Sort by deployment time (most recent first)
        deployments.sort(key=lambda x: x['deployed_at'], reverse=True)

        return deployments

    def _get_current_deployment(self, prompt_id: str, environment: Environment) -> Optional[Deployment]:
        """Get the current active deployment for a prompt in an environment."""
        # Find the most recent successful deployment
        recent_deployments = [
            d for d in self.active_deployments.values()
            if (d.prompt_id == prompt_id and
                d.environment == environment and
                d.status == DeploymentStatus.SUCCESSFUL)
        ]

        if not recent_deployments:
            return None

        # Return the most recent one
        return max(recent_deployments, key=lambda d: d.deployed_at)

    def get_deployment_history(self, prompt_id: str, environment: Environment) -> List[Dict[str, Any]]:
        """Get deployment history for a prompt in an environment.

        Args:
            prompt_id: ID of the prompt
            environment: Target environment

        Returns:
            List of deployments in chronological order
        """
        deployments = self.list_deployments(
            prompt_id=prompt_id,
            environment=environment
        )

        return deployments

    def promote_deployment(self, prompt_id: str, from_environment: Environment,
                          to_environment: Environment, promoted_by: str) -> Optional[str]:
        """Promote a deployment from one environment to another.

        Args:
            prompt_id: ID of the prompt to promote
            from_environment: Source environment
            to_environment: Target environment
            promoted_by: User performing the promotion

        Returns:
            New deployment ID if promotion initiated, None otherwise
        """
        source_deployment = self._get_current_deployment(prompt_id, from_environment)
        if not source_deployment:
            logger.error(f"No current deployment found for {prompt_id} in {from_environment.value}")
            return None

        # Deploy to target environment
        deployment_id = self.deploy_prompt(
            prompt_id=prompt_id,
            version=source_deployment.prompt_version,
            environment=to_environment,
            deployed_by=promoted_by,
            description=f"Promoted from {from_environment.value}"
        )

        logger.info(f"Promoted {prompt_id} from {from_environment.value} "
                   f"to {to_environment.value} by {promoted_by}")

        return deployment_id

    def get_deployment_statistics(self, environment: Environment = None) -> Dict[str, Any]:
        """Get deployment statistics.

        Args:
            environment: Optional filter by environment

        Returns:
            Deployment statistics dictionary
        """
        deployments = self.list_deployments(environment=environment)

        if not deployments:
            return {"error": "No deployments found"}

        total_deployments = len(deployments)
        successful_deployments = len([d for d in deployments if d['status'] == 'successful'])
        failed_deployments = len([d for d in deployments if d['status'] == 'failed'])
        pending_deployments = len([d for d in deployments if d['status'] == 'pending'])

        success_rate = successful_deployments / total_deployments if total_deployments > 0 else 0

        # Group by environment
        by_environment = {}
        for deployment in deployments:
            env = deployment['environment']
            if env not in by_environment:
                by_environment[env] = {'total': 0, 'successful': 0, 'failed': 0}
            by_environment[env]['total'] += 1
            if deployment['status'] == 'successful':
                by_environment[env]['successful'] += 1
            elif deployment['status'] == 'failed':
                by_environment[env]['failed'] += 1

        return {
            'total_deployments': total_deployments,
            'successful_deployments': successful_deployments,
            'failed_deployments': failed_deployments,
            'pending_deployments': pending_deployments,
            'success_rate': success_rate,
            'deployments_by_environment': by_environment
        }
