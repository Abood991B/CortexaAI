"""
Data models and schemas for the Prompt Management & Versioning system.
"""

from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, asdict
import json


class PromptStatus(Enum):
    """Status of a prompt in the registry."""
    DRAFT = "draft"
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"


class ExperimentStatus(Enum):
    """Status of an A/B testing experiment."""
    DRAFT = "draft"
    RUNNING = "running"
    COMPLETED = "completed"
    STOPPED = "stopped"


class DeploymentStatus(Enum):
    """Status of a prompt deployment."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    SUCCESSFUL = "successful"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


class Environment(Enum):
    """Deployment environment."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


@dataclass
class PromptMetadata:
    """Metadata associated with a prompt."""
    domain: str
    strategy: str
    author: str
    tags: List[str]
    description: str = ""
    performance_metrics: Dict[str, float] = None
    dependencies: List[str] = None
    configuration: Dict[str, Any] = None

    def __post_init__(self):
        if self.performance_metrics is None:
            self.performance_metrics = {}
        if self.dependencies is None:
            self.dependencies = []
        if self.configuration is None:
            self.configuration = {}


@dataclass
class PromptVersion:
    """Represents a specific version of a prompt."""
    id: str
    prompt_id: str
    version: str
    content: str
    metadata: PromptMetadata
    performance_metrics: Dict[str, float]
    created_by: str
    created_at: datetime
    parent_version: Optional[str] = None
    commit_message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            'id': self.id,
            'prompt_id': self.prompt_id,
            'version': self.version,
            'content': self.content,
            'metadata': asdict(self.metadata),
            'performance_metrics': self.performance_metrics,
            'created_by': self.created_by,
            'created_at': self.created_at.isoformat(),
            'parent_version': self.parent_version,
            'commit_message': self.commit_message
        }


@dataclass
class Prompt:
    """Represents a prompt in the registry."""
    id: str
    name: str
    current_version: str
    status: PromptStatus
    metadata: PromptMetadata
    versions: List[str]  # List of version IDs
    created_by: str
    created_at: datetime
    updated_at: datetime

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            'id': self.id,
            'name': self.name,
            'current_version': self.current_version,
            'status': self.status.value,
            'metadata': asdict(self.metadata),
            'versions': self.versions,
            'created_by': self.created_by,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        }


@dataclass
class ExperimentVariant:
    """Represents a variant in an A/B test."""
    id: str
    prompt_id: str
    prompt_version: str
    weight: float
    name: str
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            'id': self.id,
            'prompt_id': self.prompt_id,
            'prompt_version': self.prompt_version,
            'weight': self.weight,
            'name': self.name,
            'description': self.description
        }


@dataclass
class ExperimentResult:
    """Results from an A/B test."""
    variant_id: str
    impressions: int
    conversions: int
    conversion_rate: float
    confidence_interval: tuple
    statistical_significance: float
    performance_metrics: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            'variant_id': self.variant_id,
            'impressions': self.impressions,
            'conversions': self.conversions,
            'conversion_rate': self.conversion_rate,
            'confidence_interval': list(self.confidence_interval),
            'statistical_significance': self.statistical_significance,
            'performance_metrics': self.performance_metrics
        }


@dataclass
class Experiment:
    """Represents an A/B testing experiment."""
    id: str
    name: str
    description: str
    variants: List[ExperimentVariant]
    status: ExperimentStatus
    start_date: datetime
    end_date: Optional[datetime]
    results: Dict[str, ExperimentResult]
    created_by: str
    created_at: datetime

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'variants': [variant.to_dict() for variant in self.variants],
            'status': self.status.value,
            'start_date': self.start_date.isoformat(),
            'end_date': self.end_date.isoformat() if self.end_date else None,
            'results': {k: v.to_dict() for k, v in self.results.items()},
            'created_by': self.created_by,
            'created_at': self.created_at.isoformat()
        }


@dataclass
class Deployment:
    """Represents a prompt deployment."""
    id: str
    prompt_id: str
    prompt_version: str
    environment: Environment
    status: DeploymentStatus
    deployed_by: str
    deployed_at: datetime
    rollback_version: Optional[str] = None
    error_message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            'id': self.id,
            'prompt_id': self.prompt_id,
            'prompt_version': self.prompt_version,
            'environment': self.environment.value,
            'status': self.status.value,
            'deployed_by': self.deployed_by,
            'deployed_at': self.deployed_at.isoformat(),
            'rollback_version': self.rollback_version,
            'error_message': self.error_message
        }


@dataclass
class PerformanceRecord:
    """Performance metrics for a prompt."""
    id: str
    prompt_id: str
    prompt_version: str
    environment: Environment
    metrics: Dict[str, float]
    recorded_at: datetime
    request_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            'id': self.id,
            'prompt_id': self.prompt_id,
            'prompt_version': self.prompt_version,
            'environment': self.environment.value,
            'metrics': self.metrics,
            'recorded_at': self.recorded_at.isoformat(),
            'request_id': self.request_id
        }
