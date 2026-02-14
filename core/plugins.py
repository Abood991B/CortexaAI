"""
Plugin Architecture for CortexaAI.

Load, register and manage plugins that can extend experts, evaluators,
or add custom processing steps to the prompt engineering pipeline.
"""

import importlib
import importlib.util
import os
import time
from typing import Dict, Any, List, Optional, Callable

from config.config import get_logger

logger = get_logger(__name__)


class PluginType:
    EXPERT = "expert"
    EVALUATOR = "evaluator"
    PRE_PROCESSOR = "pre_processor"
    POST_PROCESSOR = "post_processor"
    PROVIDER = "provider"


class PluginMetadata:
    """Metadata descriptor for a plugin."""

    def __init__(
        self,
        name: str,
        version: str,
        plugin_type: str,
        description: str = "",
        author: str = "",
        entry_point: Optional[str] = None,
    ):
        self.name = name
        self.version = version
        self.plugin_type = plugin_type
        self.description = description
        self.author = author
        self.entry_point = entry_point

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "plugin_type": self.plugin_type,
            "description": self.description,
            "author": self.author,
            "entry_point": self.entry_point,
        }


class PluginManager:
    """Manage loadable plugins for CortexaAI."""

    def __init__(self):
        self._plugins: Dict[str, Dict[str, Any]] = {}
        self._hooks: Dict[str, List[Callable]] = {
            "pre_process": [],
            "post_process": [],
            "pre_evaluate": [],
            "post_evaluate": [],
        }

    def _get_db(self):
        from core.database import db
        return db

    # ── Registration ─────────────────────────────────────────────────────
    def register(
        self,
        name: str,
        version: str,
        plugin_type: str,
        description: str = "",
        author: str = "",
        config: Optional[Dict[str, Any]] = None,
        handler: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """Register a plugin programmatically."""
        meta = PluginMetadata(name, version, plugin_type, description, author)
        entry = {
            "metadata": meta,
            "config": config or {},
            "handler": handler,
            "enabled": True,
            "registered_at": time.time(),
        }
        self._plugins[name] = entry

        # Persist to DB
        try:
            import uuid as _uuid
            plugin_id = str(_uuid.uuid4())[:12]
            db = self._get_db()
            db.execute(
                """INSERT OR REPLACE INTO plugin_registry
                   (id, name, version, plugin_type, description, author, config, enabled, registered_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, 1, ?)""",
                (plugin_id, name, version, plugin_type, description, author,
                 str(config or {}), time.time()),
            )
        except Exception:
            pass  # DB not available — in-memory only

        logger.info(f"Registered plugin: {name} v{version} ({plugin_type})")
        return meta.to_dict()

    def unregister(self, name: str) -> bool:
        if name in self._plugins:
            del self._plugins[name]
            try:
                db = self._get_db()
                db.execute("DELETE FROM plugin_registry WHERE name = ?", (name,))
            except Exception:
                pass
            return True
        return False

    # ── Enable / Disable ─────────────────────────────────────────────────
    def enable(self, name: str) -> bool:
        if name in self._plugins:
            self._plugins[name]["enabled"] = True
            return True
        return False

    def disable(self, name: str) -> bool:
        if name in self._plugins:
            self._plugins[name]["enabled"] = False
            return True
        return False

    # ── Hook system ──────────────────────────────────────────────────────
    def add_hook(self, hook_name: str, callback: Callable) -> None:
        if hook_name not in self._hooks:
            self._hooks[hook_name] = []
        self._hooks[hook_name].append(callback)

    def run_hooks(self, hook_name: str, data: Any) -> Any:
        """Run all hooks for a given hook_name, passing data through each."""
        for callback in self._hooks.get(hook_name, []):
            try:
                result = callback(data)
                if result is not None:
                    data = result
            except Exception as e:
                logger.error(f"Plugin hook error ({hook_name}): {e}")
        return data

    # ── Load from directory ──────────────────────────────────────────────
    def load_from_directory(self, directory: str) -> List[str]:
        """Load all plugins from a directory. Each plugin is a .py file with a
        `register_plugin(manager)` function."""
        loaded = []
        if not os.path.isdir(directory):
            logger.warning(f"Plugin directory not found: {directory}")
            return loaded

        for fname in os.listdir(directory):
            if not fname.endswith(".py") or fname.startswith("_"):
                continue
            path = os.path.join(directory, fname)
            try:
                spec = importlib.util.spec_from_file_location(
                    f"plugin_{fname[:-3]}", path
                )
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                if hasattr(module, "register_plugin"):
                    module.register_plugin(self)
                    loaded.append(fname)
                    logger.info(f"Loaded plugin from: {fname}")
            except Exception as e:
                logger.error(f"Failed to load plugin {fname}: {e}")

        return loaded

    # ── Query ────────────────────────────────────────────────────────────
    def list_plugins(self) -> List[Dict[str, Any]]:
        return [
            {
                **p["metadata"].to_dict(),
                "enabled": p["enabled"],
                "has_handler": p["handler"] is not None,
            }
            for p in self._plugins.values()
        ]

    def get_plugin(self, name: str) -> Optional[Dict[str, Any]]:
        p = self._plugins.get(name)
        if not p:
            return None
        return {
            **p["metadata"].to_dict(),
            "enabled": p["enabled"],
            "config": p["config"],
        }

    def get_plugins_by_type(self, plugin_type: str) -> List[Dict[str, Any]]:
        return [
            p["metadata"].to_dict()
            for p in self._plugins.values()
            if p["metadata"].plugin_type == plugin_type and p["enabled"]
        ]

    def get_handler(self, name: str) -> Optional[Callable]:
        p = self._plugins.get(name)
        if p and p["enabled"]:
            return p["handler"]
        return None


# Global instance
plugin_manager = PluginManager()
