"""
Version Manager - Handles version control operations for prompts.
"""

from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import json
import re
from pathlib import Path

from .prompt_models import PromptVersion, PromptMetadata
from .prompt_registry import PromptRegistry
from config.config import get_logger

logger = get_logger(__name__)


class VersionManager:
    """Manages version control operations for prompts."""

    def __init__(self, registry: PromptRegistry):
        """Initialize the version manager.

        Args:
            registry: PromptRegistry instance to manage
        """
        self.registry = registry

    def create_branch(self, prompt_id: str, branch_name: str,
                     from_version: str = None, created_by: str = "system") -> Optional[str]:
        """Create a new branch from a specific version.

        Args:
            prompt_id: ID of the prompt to branch
            branch_name: Name for the new branch
            from_version: Version to branch from (defaults to current)
            created_by: User creating the branch

        Returns:
            Branch version string if successful, None otherwise
        """
        prompt = self.registry.get_prompt(prompt_id)
        if not prompt:
            logger.error(f"Prompt {prompt_id} not found")
            return None

        # Get the source version
        if from_version is None:
            from_version = prompt.current_version

        source_version = self.registry.get_prompt_version(prompt_id, from_version)
        if not source_version:
            logger.error(f"Version {from_version} not found for prompt {prompt_id}")
            return None

        # Create branch version (e.g., "1.0.0-branch-feature")
        branch_version = f"{from_version}-branch-{branch_name}"

        # Create new version for the branch
        new_version = self.registry.create_version(
            prompt_id=prompt_id,
            content=source_version.content,
            created_by=created_by,
            commit_message=f"Created branch '{branch_name}' from version {from_version}",
            bump_type="patch"  # Don't bump semantic version for branches
        )

        if new_version:
            # Override the version string to include branch name
            # Note: This is a simplified approach. In a real implementation,
            # you'd want to track branches separately from semantic versions
            logger.info(f"Created branch '{branch_name}' for prompt {prompt_id}")
            return branch_version

        return None

    def merge_branch(self, prompt_id: str, branch_version: str, target_version: str,
                    merged_by: str, merge_message: str = "") -> bool:
        """Merge a branch into a target version.

        Args:
            prompt_id: ID of the prompt
            branch_version: Branch version to merge
            target_version: Target version to merge into
            merged_by: User performing the merge
            merge_message: Optional merge commit message

        Returns:
            True if merge successful, False otherwise
        """
        prompt = self.registry.get_prompt(prompt_id)
        if not prompt:
            logger.error(f"Prompt {prompt_id} not found")
            return False

        # Get versions
        branch_version_obj = self.registry.get_prompt_version(prompt_id, branch_version)
        target_version_obj = self.registry.get_prompt_version(prompt_id, target_version)

        if not branch_version_obj or not target_version_obj:
            logger.error(f"Branch version {branch_version} or target version {target_version} not found")
            return False

        # Simple merge strategy: create new version with merged content
        # In a real implementation, you'd want conflict resolution logic
        merged_content = self._merge_content(
            target_version_obj.content,
            branch_version_obj.content
        )

        commit_msg = merge_message or f"Merged branch {branch_version} into {target_version}"

        new_version = self.registry.create_version(
            prompt_id=prompt_id,
            content=merged_content,
            created_by=merged_by,
            commit_message=commit_msg,
            bump_type="minor"  # Merge typically warrants minor version bump
        )

        if new_version:
            logger.info(f"Merged branch {branch_version} into {target_version} for prompt {prompt_id}")
            return True

        return False

    def _merge_content(self, base_content: str, branch_content: str) -> str:
        """Simple content merge strategy.

        In a real implementation, this would include conflict resolution,
        diff analysis, and potentially user interaction for conflicts.
        """
        if base_content == branch_content:
            return base_content

        # Simple strategy: if content is different, combine them with a marker
        return f"""# Merged Content

## Base Version
{base_content}

## Branch Changes
{branch_content}

## Note
This is an automated merge. Please review and resolve any conflicts manually.
"""

    def compare_versions(self, prompt_id: str, version1: str, version2: str) -> Dict[str, Any]:
        """Compare two versions of a prompt.

        Args:
            prompt_id: ID of the prompt
            version1: First version to compare
            version2: Second version to compare

        Returns:
            Dictionary containing comparison results
        """
        v1 = self.registry.get_prompt_version(prompt_id, version1)
        v2 = self.registry.get_prompt_version(prompt_id, version2)

        if not v1 or not v2:
            return {"error": "One or both versions not found"}

        # Simple text-based diff
        diff_result = self._calculate_diff(v1.content, v2.content)

        return {
            "prompt_id": prompt_id,
            "versions": {
                "version1": version1,
                "version2": version2
            },
            "comparison": {
                "content_changed": v1.content != v2.content,
                "metadata_changed": v1.metadata != v2.metadata,
                "performance_changed": v1.performance_metrics != v2.performance_metrics,
                "diff": diff_result
            },
            "details": {
                "version1": {
                    "created_by": v1.created_by,
                    "created_at": v1.created_at.isoformat(),
                    "commit_message": v1.commit_message
                },
                "version2": {
                    "created_by": v2.created_by,
                    "created_at": v2.created_at.isoformat(),
                    "commit_message": v2.commit_message
                }
            }
        }

    def _calculate_diff(self, content1: str, content2: str) -> Dict[str, Any]:
        """Calculate simple diff between two content strings."""
        lines1 = content1.splitlines()
        lines2 = content2.splitlines()

        # Simple line-based diff
        added = []
        removed = []
        common = []

        i = j = 0
        while i < len(lines1) and j < len(lines2):
            if lines1[i] == lines2[j]:
                common.append(lines1[i])
                i += 1
                j += 1
            elif lines1[i] in lines2[j:]:
                # Line was moved or added later
                while j < len(lines2) and lines1[i] != lines2[j]:
                    added.append(lines2[j])
                    j += 1
            else:
                removed.append(lines1[i])
                i += 1

        # Handle remaining lines
        while i < len(lines1):
            removed.append(lines1[i])
            i += 1
        while j < len(lines2):
            added.append(lines2[j])
            j += 1

        return {
            "lines_added": len(added),
            "lines_removed": len(removed),
            "lines_common": len(common),
            "total_changes": len(added) + len(removed),
            "added_preview": added[:5],  # First 5 added lines
            "removed_preview": removed[:5]  # First 5 removed lines
        }

    def get_version_lineage(self, prompt_id: str, version: str) -> List[Dict[str, Any]]:
        """Get the lineage/ancestry of a version.

        Args:
            prompt_id: ID of the prompt
            version: Version to get lineage for

        Returns:
            List of ancestor versions with metadata
        """
        lineage = []
        current_version = version

        while current_version:
            version_obj = self.registry.get_prompt_version(prompt_id, current_version)
            if not version_obj:
                break

            lineage.append({
                "version": current_version,
                "created_by": version_obj.created_by,
                "created_at": version_obj.created_at.isoformat(),
                "commit_message": version_obj.commit_message,
                "parent_version": version_obj.parent_version
            })

            current_version = version_obj.parent_version

        return lineage

    def tag_version(self, prompt_id: str, version: str, tag: str,
                   tagged_by: str) -> bool:
        """Tag a specific version for easy reference.

        Args:
            prompt_id: ID of the prompt
            version: Version to tag
            tag: Tag name (e.g., 'production', 'stable', 'experimental')
            tagged_by: User creating the tag

        Returns:
            True if tagging successful, False otherwise
        """
        prompt = self.registry.get_prompt(prompt_id)
        if not prompt:
            return False

        version_obj = self.registry.get_prompt_version(prompt_id, version)
        if not version_obj:
            return False

        # In a real implementation, you'd want a separate tags table
        # For now, we'll store tags in the metadata
        if 'tags' not in version_obj.metadata.configuration:
            version_obj.metadata.configuration['tags'] = []

        tags = version_obj.metadata.configuration['tags']
        if tag not in tags:
            tags.append(tag)

            # Create a new version with the tag
            tagged_content = f"{version_obj.content}\n\n---\nTagged as: {tag}"
            tagged_version = self.registry.create_version(
                prompt_id=prompt_id,
                content=tagged_content,
                created_by=tagged_by,
                commit_message=f"Tagged version {version} as '{tag}'",
                bump_type="patch"
            )

            if tagged_version:
                logger.info(f"Tagged version {version} of prompt {prompt_id} as '{tag}'")
                return True

        return False

    def validate_version_format(self, version: str) -> bool:
        """Validate that a version string follows semantic versioning.

        Args:
            version: Version string to validate

        Returns:
            True if valid, False otherwise
        """
        # Semantic versioning pattern: major.minor.patch
        pattern = r'^\d+\.\d+\.\d+$'
        return bool(re.match(pattern, version))

    def get_version_statistics(self, prompt_id: str) -> Dict[str, Any]:
        """Get statistics about versions for a prompt.

        Args:
            prompt_id: ID of the prompt

        Returns:
            Dictionary with version statistics
        """
        versions = self.registry.get_version_history(prompt_id)

        if not versions:
            return {"error": "No versions found"}

        stats = {
            "total_versions": len(versions),
            "current_version": versions[0].version if versions else None,
            "version_range": {
                "oldest": versions[-1].version if versions else None,
                "newest": versions[0].version if versions else None
            },
            "authors": list(set(v.created_by for v in versions)),
            "creation_timeline": {
                "first_version": versions[-1].created_at.isoformat() if versions else None,
                "latest_version": versions[0].created_at.isoformat() if versions else None
            }
        }

        return stats

    def export_version_history(self, prompt_id: str, format: str = "json") -> Optional[str]:
        """Export version history for a prompt.

        Args:
            prompt_id: ID of the prompt
            format: Export format ('json' or 'markdown')

        Returns:
            Exported data as string, or None if prompt not found
        """
        versions = self.registry.get_version_history(prompt_id)

        if not versions:
            return None

        if format == "json":
            return json.dumps([v.to_dict() for v in versions], indent=2, ensure_ascii=False)
        elif format == "markdown":
            return self._export_as_markdown(versions)

        return None

    def _export_as_markdown(self, versions: List[PromptVersion]) -> str:
        """Export versions as markdown format."""
        lines = ["# Version History\n"]

        for version in versions:
            lines.extend([
                f"## Version {version.version}",
                f"- **Created by:** {version.created_by}",
                f"- **Created at:** {version.created_at.isoformat()}",
                f"- **Commit message:** {version.commit_message or 'No message'}",
                f"- **Parent version:** {version.parent_version or 'None'}",
                "",
                "### Content",
                "```",
                version.content,
                "```",
                "",
                "---",
                ""
            ])

        return "\n".join(lines)
