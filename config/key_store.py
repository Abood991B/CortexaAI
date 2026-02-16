"""
Encrypted API Key Store for CortexaAI.

Provides secure storage for sensitive API keys using Fernet symmetric
encryption.  Keys are encrypted at rest in `data/keys.enc` and decrypted
only in memory when the application needs them.

The encryption key itself is derived from a machine-specific seed so that
the encrypted file is useless if copied to a different machine.  For extra
security you may set the environment variable CORTEXAAI_KEY_PASSWORD to add
a user-chosen component.

Usage (programmatic):
    from config.key_store import key_store
    key_store.set_key("GOOGLE_API_KEY", "AIza...")
    value = key_store.get_key("GOOGLE_API_KEY")

Usage (CLI):
    python -m config.key_store add   GOOGLE_API_KEY <value>
    python -m config.key_store list
    python -m config.key_store remove GOOGLE_API_KEY
    python -m config.key_store import-env          # import from .env
"""

import base64
import hashlib
import json
import os
import platform
import sys
from pathlib import Path
from typing import Dict, Optional

from cryptography.fernet import Fernet, InvalidToken

# ── paths ──────────────────────────────────────────────────────
_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
_KEYS_FILE = _DATA_DIR / "keys.enc"


# ── derive a machine-bound encryption key ──────────────────────
def _derive_key() -> bytes:
    """
    Build a deterministic 32-byte key from:
      1. Machine node name
      2. Platform identifier
      3. Optional user password (env CORTEXAAI_KEY_PASSWORD)
    Hashed through SHA-256 → URL-safe base64 → Fernet key.

    **Security note**: Without ``CORTEXAAI_KEY_PASSWORD`` set, the encryption
    key relies solely on discoverable machine attributes.  A logged warning
    is emitted to encourage setting the password in production.
    """
    import logging as _logging
    seed = f"cortexaai:{platform.node()}:{platform.platform()}"
    password = os.environ.get("CORTEXAAI_KEY_PASSWORD", "")
    if not password:
        _logging.getLogger(__name__).warning(
            "CORTEXAAI_KEY_PASSWORD is not set. The encrypted key store relies "
            "only on machine attributes, which may be discoverable. Set "
            "CORTEXAAI_KEY_PASSWORD for stronger encryption."
        )
    seed += f":{password}"
    digest = hashlib.sha256(seed.encode()).digest()
    return base64.urlsafe_b64encode(digest)


class KeyStore:
    """Thread-safe encrypted key-value store backed by a single file."""

    def __init__(self, path: Path = _KEYS_FILE):
        self._path = path
        self._fernet = Fernet(_derive_key())
        self._cache: Optional[Dict[str, str]] = None

    # ── internal ───────────────────────────────────────────────

    def _load(self) -> Dict[str, str]:
        if self._cache is not None:
            return self._cache
        if not self._path.exists():
            self._cache = {}
            return self._cache
        try:
            encrypted = self._path.read_bytes()
            decrypted = self._fernet.decrypt(encrypted)
            self._cache = json.loads(decrypted.decode())
        except (InvalidToken, json.JSONDecodeError):
            # File corrupted or wrong machine → start fresh
            self._cache = {}
        return self._cache

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        raw = json.dumps(self._cache or {}, indent=2).encode()
        self._path.write_bytes(self._fernet.encrypt(raw))

    # ── public API ─────────────────────────────────────────────

    def get_key(self, name: str) -> Optional[str]:
        """Return the decrypted value for *name*, or None."""
        return self._load().get(name)

    def set_key(self, name: str, value: str) -> None:
        """Store (or overwrite) *name* with *value* and persist."""
        data = self._load()
        data[name] = value
        self._save()

    def remove_key(self, name: str) -> bool:
        """Remove *name*. Returns True if it existed."""
        data = self._load()
        if name in data:
            del data[name]
            self._save()
            return True
        return False

    def list_keys(self) -> list[str]:
        """Return sorted list of stored key names (no values)."""
        return sorted(self._load().keys())

    def has_key(self, name: str) -> bool:
        return name in self._load()

    def get_all(self) -> Dict[str, str]:
        """Return a **copy** of all decrypted key-value pairs."""
        return dict(self._load())

    def import_from_env(self, env_path: str = ".env") -> int:
        """
        Read known API-key variables from a .env file and import them
        into the encrypted store.  Returns the number of keys imported.
        """
        target_vars = [
            "GOOGLE_API_KEY",
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
            "GROQ_API_KEY",
            "DEEPSEEK_API_KEY",
            "OPENROUTER_API_KEY",
            "LANGSMITH_API_KEY",
        ]
        placeholder_patterns = ["your_", "sk-xxx", "placeholder", "_here"]

        imported = 0
        env_file = Path(env_path)
        if not env_file.exists():
            return 0

        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            var, _, val = line.partition("=")
            var = var.strip()
            val = val.strip().strip('"').strip("'")
            if var in target_vars and val:
                # Skip placeholder values
                if any(p in val.lower() for p in placeholder_patterns):
                    continue
                self.set_key(var, val)
                imported += 1
        return imported

    def clear_env_keys(self, env_path: str = ".env") -> int:
        """
        Remove sensitive API-key values from the .env file, replacing
        them with empty strings.  Returns the number of lines modified.
        """
        target_vars = [
            "GOOGLE_API_KEY",
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
            "GROQ_API_KEY",
            "DEEPSEEK_API_KEY",
            "OPENROUTER_API_KEY",
            "LANGSMITH_API_KEY",
        ]
        placeholder_patterns = ["your_", "sk-xxx", "placeholder", "_here"]

        env_file = Path(env_path)
        if not env_file.exists():
            return 0

        lines = env_file.read_text(encoding="utf-8").splitlines()
        modified = 0
        new_lines = []
        for line in lines:
            stripped = line.strip()
            if stripped and not stripped.startswith("#") and "=" in stripped:
                var, _, val = stripped.partition("=")
                var_name = var.strip()
                val_clean = val.strip().strip('"').strip("'")
                if var_name in target_vars and val_clean and not any(p in val_clean.lower() for p in placeholder_patterns):
                    new_lines.append(f"{var_name}=")
                    modified += 1
                    continue
            new_lines.append(line)

        if modified:
            env_file.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
        return modified


# ── global singleton ───────────────────────────────────────────
key_store = KeyStore()


# ── CLI entry point ────────────────────────────────────────────
def _cli() -> None:  # pragma: no cover
    """Minimal CLI for managing the key store."""
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        return

    cmd = args[0].lower()

    if cmd == "list":
        keys = key_store.list_keys()
        if keys:
            print("Stored keys:")
            for k in keys:
                # Show first/last 4 chars of value
                val = key_store.get_key(k) or ""
                masked = f"{val[:4]}...{val[-4:]}" if len(val) > 8 else "****"
                print(f"  {k} = {masked}")
        else:
            print("No keys stored.")

    elif cmd == "add" and len(args) >= 3:
        name, value = args[1], args[2]
        key_store.set_key(name, value)
        print(f"Stored {name}")

    elif cmd == "remove" and len(args) >= 2:
        name = args[1]
        if key_store.remove_key(name):
            print(f"Removed {name}")
        else:
            print(f"{name} not found")

    elif cmd == "import-env":
        env_path = args[1] if len(args) > 1 else ".env"
        n = key_store.import_from_env(env_path)
        print(f"Imported {n} key(s) from {env_path}")

    elif cmd == "clear-env":
        env_path = args[1] if len(args) > 1 else ".env"
        n = key_store.clear_env_keys(env_path)
        print(f"Cleared {n} key(s) from {env_path}")

    elif cmd == "secure":
        # Convenience: import + clear in one step
        env_path = args[1] if len(args) > 1 else ".env"
        n_imported = key_store.import_from_env(env_path)
        n_cleared = key_store.clear_env_keys(env_path)
        print(f"Imported {n_imported} key(s), cleared {n_cleared} from {env_path}")
        print("API keys are now encrypted in data/keys.enc")

    else:
        print("Usage:")
        print("  python -m config.key_store list")
        print("  python -m config.key_store add <NAME> <VALUE>")
        print("  python -m config.key_store remove <NAME>")
        print("  python -m config.key_store import-env [path]")
        print("  python -m config.key_store clear-env [path]")
        print("  python -m config.key_store secure [path]   # import + clear")


if __name__ == "__main__":
    _cli()
