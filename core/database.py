"""
Persistent Storage Layer for CortexaAI.

SQLite-backed persistence for workflows, optimization history, A/B tests,
prompt versions, templates, and user data.  Designed with an abstract base
so a PostgreSQL (or Redis) adapter can be swapped in later.
"""

import sqlite3
import json
import uuid
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

from config.config import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Database Manager
# ---------------------------------------------------------------------------

class Database:
    """SQLite-backed persistence layer with auto-migration."""

    def __init__(self, db_path: str = "data/cortexaai.db"):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_tables()
        logger.info(f"Database initialised at {db_path}")

    # -- connection helper --------------------------------------------------

    @contextmanager
    def connection(self):
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # -- schema -------------------------------------------------------------

    def _init_tables(self):
        with self.connection() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS workflows (
                    id TEXT PRIMARY KEY,
                    status TEXT NOT NULL DEFAULT 'running',
                    domain TEXT,
                    original_prompt TEXT,
                    optimized_prompt TEXT,
                    quality_score REAL DEFAULT 0,
                    iterations_used INTEGER DEFAULT 0,
                    processing_time REAL DEFAULT 0,
                    prompt_type TEXT DEFAULT 'auto',
                    result_json TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS optimization_runs (
                    id TEXT PRIMARY KEY,
                    workflow_id TEXT,
                    domain TEXT,
                    strategy TEXT,
                    initial_score REAL DEFAULT 0,
                    final_score REAL DEFAULT 0,
                    improvement_pct REAL DEFAULT 0,
                    total_iterations INTEGER DEFAULT 0,
                    total_time REAL DEFAULT 0,
                    result_json TEXT,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS ab_tests (
                    id TEXT PRIMARY KEY,
                    winner TEXT,
                    score_a REAL DEFAULT 0,
                    score_b REAL DEFAULT 0,
                    confidence REAL DEFAULT 0,
                    reasoning TEXT,
                    result_json TEXT,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS prompt_versions (
                    id TEXT PRIMARY KEY,
                    prompt_text TEXT NOT NULL,
                    domain TEXT,
                    quality_score REAL DEFAULT 0,
                    parent_version TEXT,
                    strategy_used TEXT DEFAULT 'iterative',
                    improvements_json TEXT,
                    metadata_json TEXT,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS templates (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    domain TEXT NOT NULL,
                    description TEXT,
                    template_text TEXT NOT NULL,
                    variables_json TEXT,
                    tags_json TEXT,
                    usage_count INTEGER DEFAULT 0,
                    rating REAL DEFAULT 0,
                    author TEXT DEFAULT 'system',
                    is_public INTEGER DEFAULT 1,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS api_keys (
                    id TEXT PRIMARY KEY,
                    key_hash TEXT UNIQUE NOT NULL,
                    name TEXT NOT NULL,
                    owner TEXT,
                    scopes_json TEXT DEFAULT '["read","write"]',
                    rate_limit INTEGER DEFAULT 100,
                    is_active INTEGER DEFAULT 1,
                    usage_count INTEGER DEFAULT 0,
                    last_used_at TEXT,
                    created_at TEXT NOT NULL,
                    expires_at TEXT
                );

                CREATE TABLE IF NOT EXISTS marketplace_items (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    description TEXT,
                    prompt_text TEXT,
                    domain TEXT,
                    author TEXT DEFAULT 'anonymous',
                    tags TEXT,
                    price REAL DEFAULT 0.0,
                    downloads INTEGER DEFAULT 0,
                    rating REAL DEFAULT 0,
                    rating_count INTEGER DEFAULT 0,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS regression_suites (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    domain TEXT,
                    description TEXT,
                    test_cases TEXT NOT NULL,
                    baseline TEXT,
                    created_at TEXT NOT NULL,
                    last_run TEXT
                );

                CREATE TABLE IF NOT EXISTS plugin_registry (
                    id TEXT PRIMARY KEY,
                    name TEXT UNIQUE NOT NULL,
                    version TEXT NOT NULL,
                    plugin_type TEXT NOT NULL,
                    description TEXT DEFAULT '',
                    author TEXT DEFAULT '',
                    config TEXT,
                    enabled INTEGER DEFAULT 1,
                    registered_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS error_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    error_type TEXT NOT NULL,
                    error_code TEXT,
                    severity TEXT DEFAULT 'medium',
                    message TEXT,
                    provider TEXT,
                    domain TEXT,
                    workflow_id TEXT,
                    stack_trace TEXT,
                    context_json TEXT,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS metrics_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    metrics_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_workflows_status ON workflows(status);
                CREATE INDEX IF NOT EXISTS idx_workflows_domain ON workflows(domain);
                CREATE INDEX IF NOT EXISTS idx_workflows_created ON workflows(created_at);
                CREATE INDEX IF NOT EXISTS idx_templates_domain ON templates(domain);
                CREATE INDEX IF NOT EXISTS idx_error_log_type ON error_log(error_type);
                CREATE INDEX IF NOT EXISTS idx_error_log_created ON error_log(created_at);
                CREATE INDEX IF NOT EXISTS idx_api_keys_hash ON api_keys(key_hash);
                CREATE INDEX IF NOT EXISTS idx_marketplace_domain ON marketplace_items(domain);
            """)

    # ======================================================================
    # Generic SQL Utilities (used by feature modules)
    # ======================================================================

    def execute(self, sql: str, params: tuple = ()) -> None:
        """Execute a write query (INSERT, UPDATE, DELETE)."""
        with self.connection() as conn:
            conn.execute(sql, params)

    def fetch_one(self, sql: str, params: tuple = ()):
        """Fetch a single row. Returns tuple or None."""
        with self.connection() as conn:
            row = conn.execute(sql, params).fetchone()
            return tuple(row) if row else None

    def fetch_all(self, sql: str, params: tuple = ()) -> list:
        """Fetch all rows. Returns list of tuples."""
        with self.connection() as conn:
            rows = conn.execute(sql, params).fetchall()
            return [tuple(r) for r in rows]

    # ======================================================================
    # Workflow CRUD
    # ======================================================================

    def save_workflow(self, data: Dict[str, Any]) -> str:
        wid = data.get("workflow_id", f"wf_{uuid.uuid4().hex[:12]}")
        now = datetime.now().isoformat()
        with self.connection() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO workflows
                   (id, status, domain, original_prompt, optimized_prompt,
                    quality_score, iterations_used, processing_time, prompt_type,
                    result_json, created_at, updated_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    wid,
                    data.get("status", "completed"),
                    data.get("output", {}).get("domain", data.get("domain")),
                    data.get("input", {}).get("original_prompt", data.get("original_prompt", "")),
                    data.get("output", {}).get("optimized_prompt", data.get("optimized_prompt", "")),
                    data.get("output", {}).get("quality_score", data.get("quality_score", 0)),
                    data.get("output", {}).get("iterations_used", data.get("iterations_used", 0)),
                    data.get("processing_time_seconds", data.get("processing_time", 0)),
                    data.get("input", {}).get("prompt_type", data.get("prompt_type", "auto")),
                    json.dumps(data, default=str),
                    data.get("timestamp", data.get("created_at", now)),
                    now,
                ),
            )
        return wid

    def get_workflow(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        with self.connection() as conn:
            row = conn.execute("SELECT * FROM workflows WHERE id=?", (workflow_id,)).fetchone()
            if row:
                result = json.loads(row["result_json"]) if row["result_json"] else {}
                result["workflow_id"] = row["id"]
                return result
        return None

    def get_workflows(self, limit: int = 50, offset: int = 0, status: str = None, domain: str = None) -> List[Dict[str, Any]]:
        with self.connection() as conn:
            query = "SELECT * FROM workflows WHERE 1=1"
            params: list = []
            if status:
                query += " AND status=?"
                params.append(status)
            if domain:
                query += " AND domain=?"
                params.append(domain)
            query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
            params.extend([limit, offset])
            rows = conn.execute(query, params).fetchall()
            results = []
            for r in rows:
                try:
                    data = json.loads(r["result_json"]) if r["result_json"] else {}
                except Exception:
                    data = {}
                data["workflow_id"] = r["id"]
                results.append(data)
            return results

    def count_workflows(self, status: str = None, domain: str = None) -> int:
        with self.connection() as conn:
            query = "SELECT COUNT(*) as cnt FROM workflows WHERE 1=1"
            params: list = []
            if status:
                query += " AND status=?"
                params.append(status)
            if domain:
                query += " AND domain=?"
                params.append(domain)
            row = conn.execute(query, params).fetchone()
            return row["cnt"] if row else 0

    # ======================================================================
    # Optimization Runs
    # ======================================================================

    def save_optimization_run(self, data: Dict[str, Any]) -> str:
        rid = data.get("run_id", f"opt_{uuid.uuid4().hex[:12]}")
        now = datetime.now().isoformat()
        with self.connection() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO optimization_runs
                   (id, workflow_id, domain, strategy, initial_score, final_score,
                    improvement_pct, total_iterations, total_time, result_json, created_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    rid,
                    data.get("workflow_id"),
                    data.get("domain"),
                    data.get("strategy"),
                    data.get("initial_score", 0),
                    data.get("final_score", 0),
                    data.get("improvement_percentage", 0),
                    data.get("iterations", data.get("total_iterations", 0)),
                    data.get("time_seconds", data.get("total_time_seconds", 0)),
                    json.dumps(data, default=str),
                    data.get("created_at", now),
                ),
            )
        return rid

    def get_optimization_runs(self, limit: int = 20) -> List[Dict[str, Any]]:
        with self.connection() as conn:
            rows = conn.execute(
                "SELECT * FROM optimization_runs ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
            return [json.loads(r["result_json"]) for r in rows if r["result_json"]]

    # ======================================================================
    # A/B Tests
    # ======================================================================

    def save_ab_test(self, data: Dict[str, Any]) -> str:
        tid = data.get("test_id", f"ab_{uuid.uuid4().hex[:12]}")
        now = datetime.now().isoformat()
        with self.connection() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO ab_tests
                   (id, winner, score_a, score_b, confidence, reasoning, result_json, created_at)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (
                    tid,
                    data.get("winner"),
                    data.get("score_a", 0),
                    data.get("score_b", 0),
                    data.get("confidence", 0),
                    data.get("reasoning", ""),
                    json.dumps(data, default=str),
                    data.get("created_at", now),
                ),
            )
        return tid

    # ======================================================================
    # Templates
    # ======================================================================

    def save_template(self, data: Dict[str, Any]) -> str:
        tid = data.get("id", f"tpl_{uuid.uuid4().hex[:12]}")
        now = datetime.now().isoformat()
        with self.connection() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO templates
                   (id, name, domain, description, template_text, variables_json,
                    tags_json, usage_count, rating, author, is_public, created_at, updated_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    tid,
                    data.get("name", "Untitled"),
                    data.get("domain", "general"),
                    data.get("description", ""),
                    data.get("template", data.get("template_text", "")),
                    json.dumps(data.get("variables", [])),
                    json.dumps(data.get("tags", [])),
                    data.get("usage_count", 0),
                    data.get("rating", 0),
                    data.get("author", "system"),
                    1 if data.get("is_public", True) else 0,
                    data.get("created_at", now),
                    now,
                ),
            )
        return tid

    def get_templates(self, domain: str = None, limit: int = 50) -> List[Dict[str, Any]]:
        with self.connection() as conn:
            query = "SELECT * FROM templates WHERE 1=1"
            params: list = []
            if domain:
                query += " AND domain=?"
                params.append(domain)
            query += " ORDER BY usage_count DESC LIMIT ?"
            params.append(limit)
            rows = conn.execute(query, params).fetchall()
            return [
                {
                    "id": r["id"],
                    "name": r["name"],
                    "domain": r["domain"],
                    "description": r["description"],
                    "template": r["template_text"],
                    "variables": json.loads(r["variables_json"]) if r["variables_json"] else [],
                    "tags": json.loads(r["tags_json"]) if r["tags_json"] else [],
                    "usage_count": r["usage_count"],
                    "rating": r["rating"],
                    "author": r["author"],
                    "is_public": bool(r["is_public"]),
                    "created_at": r["created_at"],
                }
                for r in rows
            ]

    def get_template(self, template_id: str) -> Optional[Dict[str, Any]]:
        with self.connection() as conn:
            r = conn.execute("SELECT * FROM templates WHERE id=?", (template_id,)).fetchone()
            if r:
                return {
                    "id": r["id"],
                    "name": r["name"],
                    "domain": r["domain"],
                    "description": r["description"],
                    "template": r["template_text"],
                    "variables": json.loads(r["variables_json"]) if r["variables_json"] else [],
                    "tags": json.loads(r["tags_json"]) if r["tags_json"] else [],
                    "usage_count": r["usage_count"],
                    "rating": r["rating"],
                    "author": r["author"],
                    "created_at": r["created_at"],
                }
        return None

    def increment_template_usage(self, template_id: str):
        with self.connection() as conn:
            conn.execute("UPDATE templates SET usage_count = usage_count + 1 WHERE id=?", (template_id,))

    # ======================================================================
    # API Keys
    # ======================================================================

    def save_api_key(self, data: Dict[str, Any]) -> str:
        kid = data.get("id", f"key_{uuid.uuid4().hex[:12]}")
        now = datetime.now().isoformat()
        with self.connection() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO api_keys
                   (id, key_hash, name, owner, scopes_json, rate_limit, is_active, created_at, expires_at)
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                (
                    kid,
                    data["key_hash"],
                    data.get("name", "default"),
                    data.get("owner"),
                    json.dumps(data.get("scopes", ["read", "write"])),
                    data.get("rate_limit", 100),
                    1 if data.get("is_active", True) else 0,
                    now,
                    data.get("expires_at"),
                ),
            )
        return kid

    def get_api_key_by_hash(self, key_hash: str) -> Optional[Dict[str, Any]]:
        with self.connection() as conn:
            r = conn.execute("SELECT * FROM api_keys WHERE key_hash=? AND is_active=1", (key_hash,)).fetchone()
            if r:
                return {
                    "id": r["id"],
                    "key_hash": r["key_hash"],
                    "name": r["name"],
                    "owner": r["owner"],
                    "scopes": json.loads(r["scopes_json"]) if r["scopes_json"] else [],
                    "rate_limit": r["rate_limit"],
                    "is_active": bool(r["is_active"]),
                    "usage_count": r["usage_count"],
                    "last_used_at": r["last_used_at"],
                    "created_at": r["created_at"],
                    "expires_at": r["expires_at"],
                }
        return None

    def increment_api_key_usage(self, key_hash: str):
        now = datetime.now().isoformat()
        with self.connection() as conn:
            conn.execute("UPDATE api_keys SET usage_count = usage_count + 1, last_used_at=? WHERE key_hash=?", (now, key_hash))

    def list_api_keys(self) -> List[Dict[str, Any]]:
        with self.connection() as conn:
            rows = conn.execute("SELECT id, name, owner, is_active, usage_count, created_at, expires_at FROM api_keys ORDER BY created_at DESC").fetchall()
            return [dict(r) for r in rows]

    def revoke_api_key(self, key_id: str):
        with self.connection() as conn:
            conn.execute("UPDATE api_keys SET is_active=0 WHERE id=?", (key_id,))

    # ======================================================================
    # Marketplace
    # ======================================================================

    def save_marketplace_item(self, data: Dict[str, Any]) -> str:
        mid = data.get("id", f"mkt_{uuid.uuid4().hex[:12]}")
        now = datetime.now().isoformat()
        with self.connection() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO marketplace_items
                   (id, title, description, prompt_text, domain,
                    author, tags, price, downloads, rating, rating_count, created_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    mid,
                    data.get("title", "Untitled"),
                    data.get("description", ""),
                    data.get("prompt_text", ""),
                    data.get("domain", "general"),
                    data.get("author", "anonymous"),
                    ",".join(data.get("tags", [])) if isinstance(data.get("tags"), list) else data.get("tags", ""),
                    data.get("price", 0.0),
                    data.get("downloads", 0),
                    data.get("rating", 0),
                    data.get("rating_count", 0),
                    data.get("created_at", now),
                ),
            )
        return mid

    def get_marketplace_items(self, domain: str = None, search: str = None, limit: int = 50) -> List[Dict[str, Any]]:
        with self.connection() as conn:
            query = "SELECT * FROM marketplace_items WHERE 1=1"
            params: list = []
            if domain:
                query += " AND domain=?"
                params.append(domain)
            if search:
                query += " AND (title LIKE ? OR description LIKE ?)"
                params.extend([f"%{search}%", f"%{search}%"])
            query += " ORDER BY downloads DESC LIMIT ?"
            params.append(limit)
            rows = conn.execute(query, params).fetchall()
            return [
                {
                    "id": r["id"],
                    "title": r["title"],
                    "description": r["description"],
                    "prompt_text": r["prompt_text"],
                    "domain": r["domain"],
                    "author": r["author"],
                    "tags": r["tags"].split(",") if r["tags"] else [],
                    "price": r["price"],
                    "downloads": r["downloads"],
                    "rating": r["rating"],
                    "rating_count": r["rating_count"],
                    "created_at": r["created_at"],
                }
                for r in rows
            ]

    def increment_marketplace_download(self, item_id: str):
        with self.connection() as conn:
            conn.execute("UPDATE marketplace_items SET downloads = downloads + 1 WHERE id=?", (item_id,))

    def rate_marketplace_item(self, item_id: str, rating: float):
        with self.connection() as conn:
            r = conn.execute("SELECT rating, rating_count FROM marketplace_items WHERE id=?", (item_id,)).fetchone()
            if r:
                new_count = r["rating_count"] + 1
                new_rating = ((r["rating"] * r["rating_count"]) + rating) / new_count
                conn.execute("UPDATE marketplace_items SET rating=?, rating_count=? WHERE id=?", (new_rating, new_count, item_id))

    # ======================================================================
    # Regression Suites
    # ======================================================================

    def save_regression_suite(self, data: Dict[str, Any]) -> str:
        sid = data.get("id", f"reg_{uuid.uuid4().hex[:12]}")
        now = datetime.now().isoformat()
        with self.connection() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO regression_suites
                   (id, name, domain, description, test_cases, baseline,
                    created_at, last_run)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (
                    sid,
                    data.get("name", "Untitled Suite"),
                    data.get("domain"),
                    data.get("description", ""),
                    json.dumps(data.get("test_cases", data.get("cases", []))),
                    json.dumps(data.get("baseline", data.get("baseline_scores", {}))),
                    data.get("created_at", now),
                    data.get("last_run"),
                ),
            )
        return sid

    def get_regression_suites(self) -> List[Dict[str, Any]]:
        with self.connection() as conn:
            rows = conn.execute("SELECT * FROM regression_suites ORDER BY created_at DESC").fetchall()
            return [
                {
                    "id": r["id"],
                    "name": r["name"],
                    "domain": r["domain"],
                    "description": r["description"],
                    "test_cases": json.loads(r["test_cases"]) if r["test_cases"] else [],
                    "baseline": json.loads(r["baseline"]) if r["baseline"] else {},
                    "last_run": r["last_run"],
                    "created_at": r["created_at"],
                }
                for r in rows
            ]

    def get_regression_suite(self, suite_id: str) -> Optional[Dict[str, Any]]:
        with self.connection() as conn:
            r = conn.execute("SELECT * FROM regression_suites WHERE id=?", (suite_id,)).fetchone()
            if r:
                return {
                    "id": r["id"],
                    "name": r["name"],
                    "domain": r["domain"],
                    "description": r["description"],
                    "test_cases": json.loads(r["test_cases"]) if r["test_cases"] else [],
                    "baseline": json.loads(r["baseline"]) if r["baseline"] else {},
                    "last_run": r["last_run"],
                    "created_at": r["created_at"],
                }
        return None

    # ======================================================================
    # Plugin Registry
    # ======================================================================

    def save_plugin(self, data: Dict[str, Any]) -> str:
        pid = data.get("id", f"plg_{uuid.uuid4().hex[:12]}")
        now = str(time.time())
        with self.connection() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO plugin_registry
                   (id, name, version, plugin_type, description, author, config, enabled, registered_at)
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                (
                    pid,
                    data["name"],
                    data.get("version", "1.0.0"),
                    data.get("plugin_type", "expert"),
                    data.get("description", ""),
                    data.get("author", ""),
                    json.dumps(data.get("config", {})),
                    1 if data.get("enabled", True) else 0,
                    now,
                ),
            )
        return pid

    def get_plugins(self, enabled_only: bool = True) -> List[Dict[str, Any]]:
        with self.connection() as conn:
            query = "SELECT * FROM plugin_registry"
            if enabled_only:
                query += " WHERE enabled=1"
            rows = conn.execute(query).fetchall()
            return [
                {
                    "id": r["id"],
                    "name": r["name"],
                    "version": r["version"],
                    "plugin_type": r["plugin_type"],
                    "description": r["description"],
                    "author": r["author"],
                    "config": json.loads(r["config"]) if r["config"] else {},
                    "enabled": bool(r["enabled"]),
                    "registered_at": r["registered_at"],
                }
                for r in rows
            ]

    # ======================================================================
    # Error Log
    # ======================================================================

    def log_error(self, data: Dict[str, Any]):
        now = datetime.now().isoformat()
        with self.connection() as conn:
            conn.execute(
                """INSERT INTO error_log
                   (error_type, error_code, severity, message, provider, domain,
                    workflow_id, stack_trace, context_json, created_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (
                    data.get("error_type", "unknown"),
                    data.get("error_code"),
                    data.get("severity", "medium"),
                    data.get("message", ""),
                    data.get("provider"),
                    data.get("domain"),
                    data.get("workflow_id"),
                    data.get("stack_trace"),
                    json.dumps(data.get("context", {})),
                    now,
                ),
            )

    def get_error_analytics(self, hours: int = 24) -> Dict[str, Any]:
        cutoff = datetime.now().timestamp() - (hours * 3600)
        cutoff_iso = datetime.fromtimestamp(cutoff).isoformat()
        with self.connection() as conn:
            total = conn.execute("SELECT COUNT(*) as cnt FROM error_log WHERE created_at >= ?", (cutoff_iso,)).fetchone()["cnt"]
            by_type = conn.execute(
                "SELECT error_type, COUNT(*) as cnt FROM error_log WHERE created_at >= ? GROUP BY error_type ORDER BY cnt DESC",
                (cutoff_iso,),
            ).fetchall()
            by_severity = conn.execute(
                "SELECT severity, COUNT(*) as cnt FROM error_log WHERE created_at >= ? GROUP BY severity",
                (cutoff_iso,),
            ).fetchall()
            by_provider = conn.execute(
                "SELECT provider, COUNT(*) as cnt FROM error_log WHERE created_at >= ? AND provider IS NOT NULL GROUP BY provider ORDER BY cnt DESC",
                (cutoff_iso,),
            ).fetchall()
            recent = conn.execute(
                "SELECT * FROM error_log WHERE created_at >= ? ORDER BY created_at DESC LIMIT 20",
                (cutoff_iso,),
            ).fetchall()

            return {
                "total_errors": total,
                "period_hours": hours,
                "by_type": {r["error_type"]: r["cnt"] for r in by_type},
                "by_severity": {r["severity"]: r["cnt"] for r in by_severity},
                "by_provider": {r["provider"]: r["cnt"] for r in by_provider},
                "recent_errors": [
                    {
                        "error_type": r["error_type"],
                        "error_code": r["error_code"],
                        "severity": r["severity"],
                        "message": r["message"],
                        "provider": r["provider"],
                        "workflow_id": r["workflow_id"],
                        "created_at": r["created_at"],
                    }
                    for r in recent
                ],
            }

    # ======================================================================
    # Metrics Snapshots
    # ======================================================================

    def save_metrics_snapshot(self, metrics_data: Dict[str, Any]):
        now = datetime.now().isoformat()
        with self.connection() as conn:
            conn.execute(
                "INSERT INTO metrics_snapshots (metrics_json, created_at) VALUES (?,?)",
                (json.dumps(metrics_data, default=str), now),
            )

    # ======================================================================
    # Utility
    # ======================================================================

    def get_dashboard_stats(self) -> Dict[str, Any]:
        with self.connection() as conn:
            total = conn.execute("SELECT COUNT(*) as cnt FROM workflows").fetchone()["cnt"]
            completed = conn.execute("SELECT COUNT(*) as cnt FROM workflows WHERE status='completed'").fetchone()["cnt"]
            errors = conn.execute("SELECT COUNT(*) as cnt FROM workflows WHERE status='error'").fetchone()["cnt"]
            avg_score = conn.execute("SELECT AVG(quality_score) as avg FROM workflows WHERE status='completed' AND quality_score > 0").fetchone()["avg"] or 0
            avg_time = conn.execute("SELECT AVG(processing_time) as avg FROM workflows WHERE status='completed'").fetchone()["avg"] or 0
            domains = conn.execute("SELECT domain, COUNT(*) as cnt FROM workflows WHERE domain IS NOT NULL GROUP BY domain ORDER BY cnt DESC").fetchall()
            templates_count = conn.execute("SELECT COUNT(*) as cnt FROM templates").fetchone()["cnt"]
            marketplace_count = conn.execute("SELECT COUNT(*) as cnt FROM marketplace_items").fetchone()["cnt"]
            return {
                "total_workflows": total,
                "completed_workflows": completed,
                "error_workflows": errors,
                "success_rate": completed / total if total > 0 else 0,
                "average_quality_score": round(avg_score, 3),
                "average_processing_time": round(avg_time, 2),
                "domain_distribution": {r["domain"]: r["cnt"] for r in domains},
                "total_templates": templates_count,
                "marketplace_items": marketplace_count,
            }


# Global database instance
db = Database()
