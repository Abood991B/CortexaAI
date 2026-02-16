"""
API Key Authentication for CortexaAI.

Generate, verify, and manage API keys for securing endpoints.
Keys are stored hashed (SHA-256) in the SQLite database.
"""

import hashlib
import secrets
import time
from typing import Optional, Dict, Any, List
from enum import Enum

from config.config import get_logger

logger = get_logger(__name__)


class APIScope(str, Enum):
    READ = "read"
    WRITE = "write"
    ADMIN = "admin"


# Fixed application-wide salt – prevents rainbow-table attacks.
# For a per-key salt, the salt would need to be stored alongside the hash,
# but a fixed salt already eliminates pre-computed tables.
_KEY_SALT = "cortexaai_v1_key_salt"


def _hash_key(key: str) -> str:
    """Hash an API key with a salt using SHA-256."""
    return hashlib.sha256(f"{_KEY_SALT}:{key}".encode()).hexdigest()


def _generate_key(prefix: str = "cxa") -> str:
    """Generate a random API key with prefix."""
    random_part = secrets.token_urlsafe(32)
    return f"{prefix}_{random_part}"


class AuthManager:
    """Manage API keys with DB-backed storage."""

    def __init__(self):
        self._rate_limits: Dict[str, List[float]] = {}  # key_hash -> timestamps
        self._default_rpm = 60  # requests per minute

    def _get_db(self):
        from core.database import db
        return db

    def create_key(
        self,
        name: str,
        scopes: Optional[List[str]] = None,
        rate_limit_rpm: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Create a new API key. Returns the raw key (only shown once)."""
        import uuid as _uuid
        raw_key = _generate_key()
        key_hash = _hash_key(raw_key)
        scope_str = ",".join(scopes or [APIScope.READ, APIScope.WRITE])
        rpm = rate_limit_rpm or self._default_rpm
        key_id = str(_uuid.uuid4())[:12]

        db = self._get_db()
        db.execute(
            """INSERT INTO api_keys (id, key_hash, name, scopes_json, rate_limit, created_at, is_active)
               VALUES (?, ?, ?, ?, ?, ?, 1)""",
            (key_id, key_hash, name, scope_str, rpm, time.time()),
        )
        logger.info(f"Created API key '{name}' with scopes: {scope_str}")
        return {
            "api_key": raw_key,
            "name": name,
            "scopes": scope_str.split(","),
            "rate_limit_rpm": rpm,
            "message": "Save this key — it will not be shown again.",
        }

    def verify_key(self, raw_key: str) -> Optional[Dict[str, Any]]:
        """Verify an API key. Returns key info if valid, None otherwise."""
        if not raw_key:
            return None
        key_hash = _hash_key(raw_key)
        db = self._get_db()
        row = db.fetch_one(
            "SELECT key_hash, name, scopes_json, rate_limit, is_active FROM api_keys WHERE key_hash = ?",
            (key_hash,),
        )
        if not row:
            return None
        if not row[4]:  # is_active
            return None

        # Defer timestamp update to avoid mixing concerns
        return {
            "key_hash": row[0],
            "name": row[1],
            "scopes": row[2].split(","),
            "rate_limit_rpm": row[3],
            "is_active": True,
        }

    def update_last_used(self, key_hash: str):
        """Update last_used_at timestamp for a key."""
        db = self._get_db()
        db.execute(
            "UPDATE api_keys SET last_used_at = ? WHERE key_hash = ?",
            (time.time(), key_hash),
        )

    def check_rate_limit(self, key_hash: str, rpm_limit: int) -> bool:
        """Check if key is within rate limit. Returns True if allowed."""
        now = time.time()
        window_start = now - 60.0

        if key_hash not in self._rate_limits:
            self._rate_limits[key_hash] = []

        # Prune old entries
        self._rate_limits[key_hash] = [
            t for t in self._rate_limits[key_hash] if t > window_start
        ]

        if len(self._rate_limits[key_hash]) >= rpm_limit:
            return False

        self._rate_limits[key_hash].append(now)
        return True

    def has_scope(self, key_info: Dict[str, Any], required_scope: str) -> bool:
        """Check if key has a specific scope."""
        scopes = key_info.get("scopes", [])
        if APIScope.ADMIN in scopes:
            return True
        return required_scope in scopes

    def revoke_key(self, name: str) -> bool:
        """Revoke an API key by name."""
        db = self._get_db()
        db.execute("UPDATE api_keys SET is_active = 0 WHERE name = ?", (name,))
        logger.info(f"Revoked API key: {name}")
        return True

    def list_keys(self) -> List[Dict[str, Any]]:
        """List all API keys (hashed, no raw keys)."""
        db = self._get_db()
        rows = db.fetch_all(
            "SELECT name, scopes_json, rate_limit, is_active, created_at, last_used_at FROM api_keys ORDER BY created_at DESC"
        )
        return [
            {
                "name": r[0],
                "scopes": r[1].split(",") if r[1] else ["read"],
                "rate_limit_rpm": r[2],
                "is_active": bool(r[3]),
                "created_at": r[4],
                "last_used": r[5],
            }
            for r in rows
        ]

    def delete_key(self, name: str) -> bool:
        """Permanently delete an API key."""
        db = self._get_db()
        db.execute("DELETE FROM api_keys WHERE name = ?", (name,))
        return True


# Global instance
auth_manager = AuthManager()


# Dependency for FastAPI
async def require_api_key(request) -> Optional[Dict[str, Any]]:
    """FastAPI dependency to verify API key from X-API-Key header.
    Returns key info dict on success, None if auth is disabled.
    Raises HTTPException(401) when auth is enabled but key is missing/invalid.
    """
    from fastapi import HTTPException
    from config.config import settings

    # If auth is not enforced, return None (allow)
    if not getattr(settings, "require_api_key", False):
        return None

    api_key = request.headers.get("X-API-Key", "")
    if not api_key:
        raise HTTPException(status_code=401, detail="API key required. Provide X-API-Key header.")

    key_info = auth_manager.verify_key(api_key)
    if not key_info:
        raise HTTPException(status_code=401, detail="Invalid or inactive API key.")

    # Rate limit check
    if not auth_manager.check_rate_limit(key_info["key_hash"], key_info["rate_limit_rpm"]):
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Try again later.")

    return key_info
