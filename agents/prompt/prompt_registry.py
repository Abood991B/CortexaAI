"""
Prompt Registry - Central storage and retrieval system for prompts.
"""

import json
import os
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import threading
from pathlib import Path

from .prompt_models import (
    Prompt, PromptVersion, PromptMetadata, PromptStatus,
    Environment, PerformanceRecord
)
from config.config import get_logger

logger = get_logger(__name__)


class PromptRegistry:
    """Central registry for managing prompts, versions, and metadata."""

    def __init__(self, storage_path: str = None):
        """Initialize the prompt registry.

        Args:
            storage_path: Path to store prompt data (defaults to ./data/prompts)
        """
        if storage_path is None:
            storage_path = os.path.join(os.getcwd(), "data", "prompts")

        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # In-memory cache for faster access
        self._prompts_cache: Dict[str, Prompt] = {}
        self._versions_cache: Dict[str, PromptVersion] = {}
        self._lock = threading.RLock()

        # Storage file paths
        self.prompts_file = self.storage_path / "prompts.json"
        self.versions_file = self.storage_path / "versions.json"
        self.performance_file = self.storage_path / "performance.json"

        # Load existing data
        self._load_data()

    def _load_data(self):
        """Load existing prompt data from storage."""
        try:
            # Load prompts
            if self.prompts_file.exists():
                with open(self.prompts_file, 'r', encoding='utf-8') as f:
                    prompts_data = json.load(f)
                    for prompt_data in prompts_data.values():
                        prompt = self._dict_to_prompt(prompt_data)
                        self._prompts_cache[prompt.id] = prompt

            # Load versions
            if self.versions_file.exists():
                with open(self.versions_file, 'r', encoding='utf-8') as f:
                    versions_data = json.load(f)
                    for version_data in versions_data.values():
                        version = self._dict_to_version(version_data)
                        self._versions_cache[version.id] = version

            logger.info(f"Loaded {len(self._prompts_cache)} prompts and {len(self._versions_cache)} versions")

        except Exception as e:
            logger.error(f"Error loading prompt data: {e}")
            # Initialize empty if loading fails
            self._prompts_cache = {}
            self._versions_cache = {}

    def _save_data(self):
        """Save prompt data to storage."""
        try:
            # Save prompts
            prompts_data = {pid: prompt.to_dict() for pid, prompt in self._prompts_cache.items()}
            with open(self.prompts_file, 'w', encoding='utf-8') as f:
                json.dump(prompts_data, f, indent=2, ensure_ascii=False)

            # Save versions
            versions_data = {vid: version.to_dict() for vid, version in self._versions_cache.items()}
            with open(self.versions_file, 'w', encoding='utf-8') as f:
                json.dump(versions_data, f, indent=2, ensure_ascii=False)

        except Exception as e:
            logger.error(f"Error saving prompt data: {e}")
            raise

    def _dict_to_prompt(self, data: Dict[str, Any]) -> Prompt:
        """Convert dictionary to Prompt object."""
        metadata = PromptMetadata(**data['metadata'])
        return Prompt(
            id=data['id'],
            name=data['name'],
            current_version=data['current_version'],
            status=PromptStatus(data['status']),
            metadata=metadata,
            versions=data['versions'],
            created_by=data['created_by'],
            created_at=datetime.fromisoformat(data['created_at']),
            updated_at=datetime.fromisoformat(data['updated_at'])
        )

    def _dict_to_version(self, data: Dict[str, Any]) -> PromptVersion:
        """Convert dictionary to PromptVersion object."""
        metadata = PromptMetadata(**data['metadata'])
        return PromptVersion(
            id=data['id'],
            prompt_id=data['prompt_id'],
            version=data['version'],
            content=data['content'],
            metadata=metadata,
            performance_metrics=data['performance_metrics'],
            created_by=data['created_by'],
            created_at=datetime.fromisoformat(data['created_at']),
            parent_version=data.get('parent_version'),
            commit_message=data.get('commit_message', '')
        )

    def create_prompt(self, name: str, content: str, metadata: PromptMetadata,
                     created_by: str, commit_message: str = "") -> str:
        """Create a new prompt in the registry.

        Args:
            name: Name of the prompt
            content: Initial prompt content
            metadata: Prompt metadata
            created_by: User creating the prompt
            commit_message: Optional commit message

        Returns:
            Prompt ID
        """
        with self._lock:
            prompt_id = f"prompt_{int(datetime.now().timestamp())}_{name.replace(' ', '_')}"

            # Create initial version
            version_id = f"version_{prompt_id}_v1.0.0"
            version = PromptVersion(
                id=version_id,
                prompt_id=prompt_id,
                version="1.0.0",
                content=content,
                metadata=metadata,
                performance_metrics={},
                created_by=created_by,
                created_at=datetime.now(),
                commit_message=commit_message
            )

            # Create prompt
            prompt = Prompt(
                id=prompt_id,
                name=name,
                current_version="1.0.0",
                status=PromptStatus.DRAFT,
                metadata=metadata,
                versions=[version_id],
                created_by=created_by,
                created_at=datetime.now(),
                updated_at=datetime.now()
            )

            # Store in cache
            self._prompts_cache[prompt_id] = prompt
            self._versions_cache[version_id] = version

            # Save to disk
            self._save_data()

            logger.info(f"Created prompt '{name}' with ID {prompt_id}")
            return prompt_id

    def get_prompt(self, prompt_id: str) -> Optional[Prompt]:
        """Get a prompt by ID."""
        with self._lock:
            return self._prompts_cache.get(prompt_id)

    def get_prompt_version(self, prompt_id: str, version: str = None) -> Optional[PromptVersion]:
        """Get a specific version of a prompt."""
        with self._lock:
            if version is None:
                # Get current version
                prompt = self.get_prompt(prompt_id)
                if not prompt:
                    return None
                version = prompt.current_version

            # Find the version
            for version_obj in self._versions_cache.values():
                if version_obj.prompt_id == prompt_id and version_obj.version == version:
                    return version_obj

            return None

    def list_prompts(self, status: PromptStatus = None, domain: str = None,
                    author: str = None) -> List[Prompt]:
        """List prompts with optional filtering."""
        with self._lock:
            prompts = list(self._prompts_cache.values())

            if status:
                prompts = [p for p in prompts if p.status == status]

            if domain:
                prompts = [p for p in prompts if p.metadata.domain == domain]

            if author:
                prompts = [p for p in prompts if p.created_by == author]

            return prompts

    def update_prompt_status(self, prompt_id: str, status: PromptStatus,
                           updated_by: str) -> bool:
        """Update the status of a prompt."""
        with self._lock:
            prompt = self.get_prompt(prompt_id)
            if not prompt:
                return False

            prompt.status = status
            prompt.updated_at = datetime.now()

            self._save_data()
            logger.info(f"Updated prompt {prompt_id} status to {status.value}")
            return True

    def create_version(self, prompt_id: str, content: str, created_by: str,
                      commit_message: str = "", bump_type: str = "patch") -> Optional[str]:
        """Create a new version of an existing prompt.

        Args:
            prompt_id: ID of the prompt to version
            content: New prompt content
            created_by: User creating the version
            commit_message: Optional commit message
            bump_type: Version bump type (major, minor, patch)

        Returns:
            New version string if successful, None otherwise
        """
        with self._lock:
            prompt = self.get_prompt(prompt_id)
            if not prompt:
                return None

            # Get current version
            current_version = prompt.current_version
            version_parts = current_version.split('.')
            if len(version_parts) != 3:
                version_parts = ['1', '0', '0']

            major, minor, patch = map(int, version_parts)

            # Bump version based on type
            if bump_type == "major":
                major += 1
                minor = 0
                patch = 0
            elif bump_type == "minor":
                minor += 1
                patch = 0
            else:  # patch
                patch += 1

            new_version = f"{major}.{minor}.{patch}"

            # Create new version
            version_id = f"version_{prompt_id}_{new_version}"
            version = PromptVersion(
                id=version_id,
                prompt_id=prompt_id,
                version=new_version,
                content=content,
                metadata=prompt.metadata,
                performance_metrics={},
                created_by=created_by,
                created_at=datetime.now(),
                parent_version=current_version,
                commit_message=commit_message
            )

            # Update prompt
            prompt.current_version = new_version
            prompt.versions.append(version_id)
            prompt.updated_at = datetime.now()

            # Store in cache
            self._versions_cache[version_id] = version

            # Save to disk
            self._save_data()

            logger.info(f"Created version {new_version} for prompt {prompt_id}")
            return new_version

    def get_version_history(self, prompt_id: str) -> List[PromptVersion]:
        """Get the version history for a prompt."""
        with self._lock:
            versions = []
            for version_id in self._versions_cache:
                version = self._versions_cache[version_id]
                if version.prompt_id == prompt_id:
                    versions.append(version)

            # Sort by creation date (newest first)
            versions.sort(key=lambda v: v.created_at, reverse=True)
            return versions

    def rollback_to_version(self, prompt_id: str, target_version: str,
                          rolled_back_by: str) -> bool:
        """Rollback a prompt to a specific version."""
        with self._lock:
            prompt = self.get_prompt(prompt_id)
            if not prompt:
                return False

            # Check if version exists
            target_version_obj = None
            for version in self._versions_cache.values():
                if version.prompt_id == prompt_id and version.version == target_version:
                    target_version_obj = version
                    break

            if not target_version_obj:
                return False

            # Create a rollback version
            rollback_version = f"{target_version}.rollback.{int(datetime.now().timestamp())}"
            version_id = f"version_{prompt_id}_{rollback_version}"

            rollback_version_obj = PromptVersion(
                id=version_id,
                prompt_id=prompt_id,
                version=rollback_version,
                content=target_version_obj.content,
                metadata=target_version_obj.metadata,
                performance_metrics=target_version_obj.performance_metrics,
                created_by=rolled_back_by,
                created_at=datetime.now(),
                parent_version=prompt.current_version,
                commit_message=f"Rollback to version {target_version}"
            )

            # Update prompt
            prompt.current_version = rollback_version
            prompt.versions.append(version_id)
            prompt.updated_at = datetime.now()

            # Store in cache
            self._versions_cache[version_id] = rollback_version_obj

            # Save to disk
            self._save_data()

            logger.info(f"Rolled back prompt {prompt_id} to version {target_version}")
            return True

    def record_performance(self, prompt_id: str, version: str, environment: Environment,
                          metrics: Dict[str, float], request_id: str = None):
        """Record performance metrics for a prompt version."""
        with self._lock:
            record = PerformanceRecord(
                id=f"perf_{prompt_id}_{version}_{int(datetime.now().timestamp())}",
                prompt_id=prompt_id,
                prompt_version=version,
                environment=environment,
                metrics=metrics,
                recorded_at=datetime.now(),
                request_id=request_id
            )

            # Update version performance metrics
            version_obj = self.get_prompt_version(prompt_id, version)
            if version_obj:
                version_obj.performance_metrics.update(metrics)

            # Save performance data
            self._save_performance_record(record)

    def _save_performance_record(self, record: PerformanceRecord):
        """Save performance record to file."""
        try:
            performance_data = {}
            if self.performance_file.exists():
                with open(self.performance_file, 'r', encoding='utf-8') as f:
                    performance_data = json.load(f)

            performance_data[record.id] = record.to_dict()

            with open(self.performance_file, 'w', encoding='utf-8') as f:
                json.dump(performance_data, f, indent=2, ensure_ascii=False)

        except Exception as e:
            logger.error(f"Error saving performance record: {e}")

    def search_prompts(self, query: str, domain: str = None) -> List[Prompt]:
        """Search prompts by name, description, or tags."""
        with self._lock:
            results = []
            query_lower = query.lower()

            for prompt in self._prompts_cache.values():
                # Search in name
                if query_lower in prompt.name.lower():
                    results.append(prompt)
                    continue

                # Search in description
                if query_lower in prompt.metadata.description.lower():
                    results.append(prompt)
                    continue

                # Search in tags
                if any(query_lower in tag.lower() for tag in prompt.metadata.tags):
                    results.append(prompt)
                    continue

                # Filter by domain if specified
                if domain and prompt.metadata.domain != domain:
                    continue

            return results

    def get_prompt_statistics(self, prompt_id: str) -> Dict[str, Any]:
        """Get statistics for a prompt."""
        with self._lock:
            prompt = self.get_prompt(prompt_id)
            if not prompt:
                return {}

            versions = self.get_version_history(prompt_id)

            return {
                'total_versions': len(versions),
                'current_version': prompt.current_version,
                'status': prompt.status.value,
                'domain': prompt.metadata.domain,
                'author': prompt.created_by,
                'created_at': prompt.created_at.isoformat(),
                'last_updated': prompt.updated_at.isoformat(),
                'tags': prompt.metadata.tags,
                'performance_metrics': prompt.metadata.performance_metrics
            }
