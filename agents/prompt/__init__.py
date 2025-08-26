"""
Prompt Engineering Agent Module for the Agentic System.

This module provides comprehensive prompt management, generation, and optimization capabilities
including version control, A/B testing, deployment pipelines, performance analytics, and template management.

## Core Components:

### Prompt Management & Versioning System
- **PromptRegistry**: Central storage and retrieval system for prompts
- **VersionManager**: Git-like version control with branching and merging
- **ExperimentManager**: A/B testing framework with statistical significance
- **DeploymentManager**: Multi-environment deployment with rollback capabilities
- **AnalyticsEngine**: Performance monitoring and anomaly detection
- **TemplateManager**: Template system with variable substitution and inheritance

### Unified Interface
- **PromptManagementSystem**: Single entry point for all prompt management operations

## Key Features:

### Version Control
- Semantic versioning (major.minor.patch)
- Branching and merging capabilities
- Version comparison and diff analysis
- Rollback functionality
- Version lineage tracking

### A/B Testing
- Statistical significance testing
- Weighted traffic allocation
- Real-time performance monitoring
- Automated experiment management
- Confidence interval calculations

### Deployment Pipeline
- Multi-environment support (development, staging, production)
- Automated health checks
- Rollback mechanisms
- Environment promotion
- Deployment history tracking

### Performance Analytics
- Real-time metrics collection
- Anomaly detection with statistical analysis
- Trend analysis and forecasting
- Comprehensive reporting
- Automated recommendations

### Template Management
- Hierarchical template inheritance
- Variable substitution (${VAR_NAME} or $VAR_NAME formats)
- Conditional rendering
- Template validation
- Version control for templates

## Usage Examples:

```python
from agents.prompt import PromptManagementSystem

# Initialize the system
pms = PromptManagementSystem()

# Create a prompt
prompt_id = pms.create_prompt(
    name="Customer Support Assistant",
    content="You are a helpful customer support assistant...",
    metadata=PromptMetadata(...),
    created_by="admin"
)

# Create a new version
new_version = pms.create_version(
    prompt_id=prompt_id,
    content="You are an advanced customer support assistant...",
    created_by="admin",
    commit_message="Enhanced response capabilities"
)

# Deploy to production
deployment_id = pms.deploy_prompt(
    prompt_id=prompt_id,
    version=new_version,
    environment="production",
    deployed_by="admin"
)

# Run A/B test
experiment_id = pms.create_experiment(
    name="Response Quality Test",
    description="Testing improved response quality",
    variants=[...],
    created_by="researcher"
)

# Get performance metrics
metrics = pms.get_performance_metrics(prompt_id, time_range_hours=24)

# Get comprehensive dashboard
dashboard = pms.get_prompt_dashboard(prompt_id)
```
"""

from .prompt_registry import PromptRegistry
from .version_manager import VersionManager
from .experiment_manager import ExperimentManager
from .deployment_manager import DeploymentManager
from .analytics_engine import AnalyticsEngine
from .template_manager import TemplateManager
from .prompt_generator import PromptGenerator
from .prompt_management_system import PromptManagementSystem

__all__ = [
    'PromptRegistry',
    'VersionManager',
    'ExperimentManager',
    'DeploymentManager',
    'AnalyticsEngine',
    'TemplateManager',
    'PromptGenerator',
    'PromptManagementSystem'
]
