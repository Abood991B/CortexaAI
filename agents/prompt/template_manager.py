"""
Template Manager - Handles prompt template management and dynamic content injection.
"""

import json
import os
from typing import Dict, List, Optional, Any, Union, Set
from datetime import datetime
from pathlib import Path
from string import Template
import re

from .prompt_models import PromptMetadata
from config.config import get_logger

logger = get_logger(__name__)


class TemplateManager:
    """Manages prompt templates with variable substitution and inheritance."""

    def __init__(self, template_dir: str = None):
        """Initialize the template manager.

        Args:
            template_dir: Directory to store templates (defaults to ./templates)
        """
        if template_dir is None:
            template_dir = os.path.join(os.getcwd(), "templates")

        self.template_dir = Path(template_dir)
        self.template_dir.mkdir(parents=True, exist_ok=True)

        # In-memory template storage
        self.templates: Dict[str, Dict[str, Any]] = {}
        self.template_hierarchy: Dict[str, Set[str]] = {}  # parent -> children

        # Load existing templates
        self._load_templates()

    def _load_templates(self):
        """Load existing templates from storage."""
        try:
            template_file = self.template_dir / "templates.json"
            if template_file.exists():
                with open(template_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.templates = data.get('templates', {})
                    self.template_hierarchy = {
                        k: set(v) for k, v in data.get('hierarchy', {}).items()
                    }

            logger.info(f"Loaded {len(self.templates)} templates")

        except Exception as e:
            logger.error(f"Error loading templates: {e}")
            self.templates = {}
            self.template_hierarchy = {}

    def _save_templates(self):
        """Save templates to storage."""
        try:
            template_file = self.template_dir / "templates.json"
            data = {
                'templates': self.templates,
                'hierarchy': {k: list(v) for k, v in self.template_hierarchy.items()},
                'last_updated': datetime.now().isoformat()
            }

            with open(template_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

        except Exception as e:
            logger.error(f"Error saving templates: {e}")
            raise

    def create_template(self, name: str, content: str, metadata: PromptMetadata,
                       parent_template: str = None, variables: Dict[str, Any] = None,
                       tags: List[str] = None) -> str:
        """Create a new template.

        Args:
            name: Template name
            content: Template content with variable placeholders
            metadata: Template metadata
            parent_template: Optional parent template to inherit from
            variables: Default variable values
            tags: Optional tags for categorization

        Returns:
            Template ID
        """
        template_id = f"template_{int(datetime.now().timestamp())}_{name.replace(' ', '_')}"

        # Validate parent exists if specified
        if parent_template and parent_template not in self.templates:
            raise ValueError(f"Parent template '{parent_template}' not found")

        # Extract variables from content
        detected_vars = self._extract_variables(content)

        template = {
            'id': template_id,
            'name': name,
            'content': content,
            'metadata': {
                'domain': metadata.domain,
                'strategy': metadata.strategy,
                'author': metadata.author,
                'tags': tags or metadata.tags,
                'description': metadata.description,
                'performance_metrics': metadata.performance_metrics,
                'dependencies': metadata.dependencies,
                'configuration': metadata.configuration
            },
            'parent_template': parent_template,
            'variables': variables or {},
            'detected_variables': list(detected_vars),
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat(),
            'version': '1.0.0'
        }

        self.templates[template_id] = template

        # Update hierarchy
        if parent_template:
            if parent_template not in self.template_hierarchy:
                self.template_hierarchy[parent_template] = set()
            self.template_hierarchy[parent_template].add(template_id)

        self._save_templates()
        logger.info(f"Created template '{name}' with ID {template_id}")

        return template_id

    def get_template(self, template_id: str) -> Optional[Dict[str, Any]]:
        """Get a template by ID."""
        return self.templates.get(template_id)

    def update_template(self, template_id: str, content: str = None,
                       variables: Dict[str, Any] = None,
                       metadata: Dict[str, Any] = None) -> bool:
        """Update an existing template.

        Args:
            template_id: Template to update
            content: New content (optional)
            variables: New variables (optional)
            metadata: New metadata (optional)

        Returns:
            True if updated successfully, False otherwise
        """
        if template_id not in self.templates:
            return False

        template = self.templates[template_id]

        if content is not None:
            template['content'] = content
            template['detected_variables'] = list(self._extract_variables(content))

        if variables is not None:
            template['variables'].update(variables)

        if metadata is not None:
            template['metadata'].update(metadata)

        template['updated_at'] = datetime.now().isoformat()

        # Increment version
        current_version = template.get('version', '1.0.0')
        version_parts = current_version.split('.')
        if len(version_parts) == 3:
            major, minor, patch = map(int, version_parts)
            patch += 1
            template['version'] = f"{major}.{minor}.{patch}"

        self._save_templates()
        logger.info(f"Updated template {template_id}")

        return True

    def delete_template(self, template_id: str) -> bool:
        """Delete a template.

        Args:
            template_id: Template to delete

        Returns:
            True if deleted successfully, False otherwise
        """
        if template_id not in self.templates:
            return False

        # Check if template has children
        if template_id in self.template_hierarchy:
            children = self.template_hierarchy[template_id]
            if children:
                logger.warning(f"Cannot delete template {template_id} - has {len(children)} child templates")
                return False

        # Remove from parent's children
        template = self.templates[template_id]
        parent = template.get('parent_template')
        if parent and parent in self.template_hierarchy:
            self.template_hierarchy[parent].discard(template_id)

        # Delete template
        del self.templates[template_id]
        self._save_templates()

        logger.info(f"Deleted template {template_id}")
        return True

    def list_templates(self, domain: str = None, parent: str = None,
                      tags: List[str] = None) -> List[Dict[str, Any]]:
        """List templates with optional filtering.

        Args:
            domain: Filter by domain
            parent: Filter by parent template
            tags: Filter by tags

        Returns:
            List of templates
        """
        templates = list(self.templates.values())

        if domain:
            templates = [t for t in templates if t['metadata'].get('domain') == domain]

        if parent:
            templates = [t for t in templates if t.get('parent_template') == parent]

        if tags:
            templates = [t for t in templates if any(tag in t['metadata'].get('tags', [])
                                                    for tag in tags)]

        return templates

    def render_template(self, template_id: str, variables: Dict[str, Any] = None,
                       context: Dict[str, Any] = None) -> Optional[str]:
        """Render a template with variable substitution.

        Args:
            template_id: Template to render
            variables: Variable values for substitution
            context: Additional context for conditional rendering

        Returns:
            Rendered template content or None if template not found
        """
        template = self.get_template(template_id)
        if not template:
            return None

        # Get base content
        content = template['content']

        # Inherit from parent if exists
        if template.get('parent_template'):
            parent_content = self.render_template(
                template['parent_template'],
                variables,
                context
            )
            if parent_content:
                content = parent_content + "\n\n" + content

        # Merge variables
        template_vars = template.get('variables', {}).copy()
        if variables:
            template_vars.update(variables)

        # Apply conditional rendering
        content = self._apply_conditionals(content, context or {})

        # Substitute variables
        try:
            rendered = self._substitute_variables(content, template_vars)
            return rendered
        except Exception as e:
            logger.error(f"Error rendering template {template_id}: {e}")
            return None

    def _extract_variables(self, content: str) -> Set[str]:
        """Extract variable names from template content."""
        # Find variables in ${VAR_NAME} or $VAR_NAME format
        var_pattern = r'\$\{([^}]+)\}|\$([A-Za-z_][A-Za-z0-9_]*)'
        variables = set()

        for match in re.finditer(var_pattern, content):
            var_name = match.group(1) or match.group(2)
            if var_name:
                variables.add(var_name)

        return variables

    def _substitute_variables(self, content: str, variables: Dict[str, Any]) -> str:
        """Substitute variables in template content."""
        # Handle both ${VAR} and $VAR formats
        result = content

        # First pass: ${VAR_NAME} format
        for var_name, var_value in variables.items():
            pattern = r'\$\{' + re.escape(var_name) + r'\}'
            result = re.sub(pattern, str(var_value), result)

        # Second pass: $VAR_NAME format (but avoid $VAR_NAME in middle of words)
        for var_name, var_value in variables.items():
            pattern = r'\$' + re.escape(var_name) + r'(?=\W|$)'
            result = re.sub(pattern, str(var_value), result)

        return result

    def _apply_conditionals(self, content: str, context: Dict[str, Any]) -> str:
        """Apply conditional rendering based on context."""
        # Simple conditional blocks: {#if CONDITION}content{/if}
        result = content

        # Find conditional blocks
        conditional_pattern = r'\{#if\s+(\w+)\}(.*?)\{/if\}'

        def replace_conditional(match):
            condition = match.group(1)
            content_block = match.group(2)

            if condition in context and context[condition]:
                return content_block
            else:
                return ""

        result = re.sub(conditional_pattern, replace_conditional, result, flags=re.DOTALL)

        return result

    def validate_template(self, template_id: str) -> Dict[str, Any]:
        """Validate a template for syntax and variable consistency.

        Args:
            template_id: Template to validate

        Returns:
            Validation results
        """
        template = self.get_template(template_id)
        if not template:
            return {"valid": False, "errors": ["Template not found"]}

        errors = []
        warnings = []

        content = template['content']
        detected_vars = self._extract_variables(content)
        defined_vars = set(template.get('variables', {}).keys())

        # Check for undefined variables
        undefined_vars = detected_vars - defined_vars
        if undefined_vars:
            warnings.extend([f"Undefined variable: {var}" for var in undefined_vars])

        # Check for unused variables
        unused_vars = defined_vars - detected_vars
        if unused_vars:
            warnings.extend([f"Unused variable: {var}" for var in unused_vars])

        # Check template syntax
        try:
            # Try to render with dummy values
            dummy_vars = {var: f"dummy_{var}" for var in detected_vars}
            self._substitute_variables(content, dummy_vars)
        except Exception as e:
            errors.append(f"Template syntax error: {e}")

        # Check for circular inheritance
        if self._has_circular_inheritance(template_id):
            errors.append("Circular inheritance detected")

        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "detected_variables": list(detected_vars),
            "defined_variables": list(defined_vars)
        }

    def _has_circular_inheritance(self, template_id: str, visited: Set[str] = None) -> bool:
        """Check for circular inheritance in template hierarchy."""
        if visited is None:
            visited = set()

        if template_id in visited:
            return True

        visited.add(template_id)

        template = self.get_template(template_id)
        if not template:
            return False

        parent = template.get('parent_template')
        if parent:
            return self._has_circular_inheritance(parent, visited.copy())

        return False

    def get_template_lineage(self, template_id: str) -> List[Dict[str, Any]]:
        """Get the inheritance lineage of a template.

        Args:
            template_id: Template to analyze

        Returns:
            List of ancestor templates
        """
        lineage = []
        current_id = template_id

        while current_id:
            template = self.get_template(current_id)
            if not template:
                break

            lineage.append({
                "id": current_id,
                "name": template['name'],
                "version": template.get('version', '1.0.0')
            })

            current_id = template.get('parent_template')

        return lineage

    def clone_template(self, template_id: str, new_name: str,
                      modifications: Dict[str, Any] = None) -> Optional[str]:
        """Clone a template with optional modifications.

        Args:
            template_id: Template to clone
            new_name: Name for the cloned template
            modifications: Optional modifications to apply

        Returns:
            New template ID or None if cloning failed
        """
        original = self.get_template(template_id)
        if not original:
            return None

        # Start with original data
        new_template = original.copy()
        new_template['name'] = new_name
        new_template['created_at'] = datetime.now().isoformat()
        new_template['updated_at'] = datetime.now().isoformat()
        new_template['version'] = '1.0.0'

        # Apply modifications
        if modifications:
            if 'content' in modifications:
                new_template['content'] = modifications['content']
                new_template['detected_variables'] = list(
                    self._extract_variables(modifications['content'])
                )
            if 'variables' in modifications:
                new_template['variables'].update(modifications['variables'])
            if 'metadata' in modifications:
                new_template['metadata'].update(modifications['metadata'])

        # Generate new ID
        new_id = f"template_{int(datetime.now().timestamp())}_{new_name.replace(' ', '_')}"
        new_template['id'] = new_id

        self.templates[new_id] = new_template
        self._save_templates()

        logger.info(f"Cloned template {template_id} to {new_id} with name '{new_name}'")

        return new_id

    def export_template(self, template_id: str, format: str = "json") -> Optional[str]:
        """Export a template.

        Args:
            template_id: Template to export
            format: Export format ('json' or 'yaml')

        Returns:
            Exported template as string or None if not found
        """
        template = self.get_template(template_id)
        if not template:
            return None

        if format == "json":
            return json.dumps(template, indent=2, ensure_ascii=False)
        elif format == "yaml":
            try:
                import yaml
                return yaml.dump(template, default_flow_style=False, allow_unicode=True)
            except ImportError:
                logger.warning("PyYAML not available, falling back to JSON")
                return json.dumps(template, indent=2, ensure_ascii=False)

        return None

    def search_templates(self, query: str, domain: str = None) -> List[Dict[str, Any]]:
        """Search templates by name, content, or metadata.

        Args:
            query: Search query
            domain: Optional domain filter

        Returns:
            List of matching templates
        """
        results = []
        query_lower = query.lower()

        for template in self.templates.values():
            # Filter by domain if specified
            if domain and template['metadata'].get('domain') != domain:
                continue

            # Search in name
            if query_lower in template['name'].lower():
                results.append(template)
                continue

            # Search in content
            if query_lower in template['content'].lower():
                results.append(template)
                continue

            # Search in description
            if query_lower in template['metadata'].get('description', '').lower():
                results.append(template)
                continue

            # Search in tags
            if any(query_lower in tag.lower() for tag in template['metadata'].get('tags', [])):
                results.append(template)
                continue

        return results

    def get_template_statistics(self) -> Dict[str, Any]:
        """Get statistics about templates."""
        total_templates = len(self.templates)
        domains = {}
        tags = {}

        for template in self.templates.values():
            # Count by domain
            domain = template['metadata'].get('domain', 'unknown')
            domains[domain] = domains.get(domain, 0) + 1

            # Count tags
            template_tags = template['metadata'].get('tags', [])
            for tag in template_tags:
                tags[tag] = tags.get(tag, 0) + 1

        # Find templates with inheritance
        inherited_templates = len([t for t in self.templates.values()
                                 if t.get('parent_template')])

        return {
            'total_templates': total_templates,
            'templates_with_inheritance': inherited_templates,
            'inheritance_ratio': inherited_templates / total_templates if total_templates > 0 else 0,
            'domains': domains,
            'tags': tags,
            'hierarchy_depth': self._calculate_max_hierarchy_depth()
        }

    def _calculate_max_hierarchy_depth(self) -> int:
        """Calculate the maximum depth of template hierarchy."""
        max_depth = 0

        for template_id in self.templates:
            depth = len(self.get_template_lineage(template_id))
            max_depth = max(max_depth, depth)

        return max_depth
