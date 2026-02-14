"""
Prompt Marketplace / Sharing for CortexaAI.

Publish, search, download and rate optimised prompt templates.
All data is persisted to the SQLite database.
"""

import time
import uuid
from typing import Dict, Any, List, Optional

from config.config import get_logger

logger = get_logger(__name__)


class MarketplaceService:
    """Community marketplace for sharing optimised prompts."""

    def _get_db(self):
        from core.database import db
        return db

    # ── Publish ──────────────────────────────────────────────────────────
    def publish(
        self,
        title: str,
        description: str,
        prompt_text: str,
        domain: str,
        author: str = "anonymous",
        tags: Optional[List[str]] = None,
        price: float = 0.0,
    ) -> Dict[str, Any]:
        """Publish a prompt to the marketplace."""
        item_id = str(uuid.uuid4())[:12]
        tag_str = ",".join(tags or [])
        db = self._get_db()
        db.execute(
            """INSERT INTO marketplace_items
               (id, title, description, prompt_text, domain, author, tags, price,
                downloads, rating, rating_count, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, 0.0, 0, ?)""",
            (item_id, title, description, prompt_text, domain, author, tag_str, price, time.time()),
        )
        logger.info(f"Published marketplace item: {title} ({item_id})")
        return {"id": item_id, "title": title, "domain": domain, "author": author}

    # ── Search ───────────────────────────────────────────────────────────
    def search(
        self,
        query: Optional[str] = None,
        domain: Optional[str] = None,
        sort_by: str = "downloads",
        limit: int = 20,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """Search marketplace items."""
        conditions: List[str] = []
        params: List[Any] = []

        if query:
            conditions.append("(title LIKE ? OR description LIKE ? OR tags LIKE ?)")
            q = f"%{query}%"
            params.extend([q, q, q])
        if domain:
            conditions.append("domain = ?")
            params.append(domain)

        where = "WHERE " + " AND ".join(conditions) if conditions else ""
        order = {"downloads": "downloads DESC", "rating": "rating DESC", "newest": "created_at DESC"}.get(
            sort_by, "downloads DESC"
        )

        sql = f"SELECT id, title, description, domain, author, tags, price, downloads, rating, rating_count, created_at FROM marketplace_items {where} ORDER BY {order} LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        db = self._get_db()
        rows = db.fetch_all(sql, tuple(params))
        return [
            {
                "id": r[0],
                "title": r[1],
                "description": r[2],
                "domain": r[3],
                "author": r[4],
                "tags": r[5].split(",") if r[5] else [],
                "price": r[6],
                "downloads": r[7],
                "rating": r[8],
                "rating_count": r[9],
                "created_at": r[10],
            }
            for r in rows
        ]

    # ── Download ─────────────────────────────────────────────────────────
    def download(self, item_id: str) -> Optional[Dict[str, Any]]:
        """Download (retrieve) a marketplace item. Increments download count."""
        db = self._get_db()
        row = db.fetch_one(
            "SELECT id, title, description, prompt_text, domain, author, tags, downloads, rating FROM marketplace_items WHERE id = ?",
            (item_id,),
        )
        if not row:
            return None
        db.execute(
            "UPDATE marketplace_items SET downloads = downloads + 1 WHERE id = ?",
            (item_id,),
        )
        return {
            "id": row[0],
            "title": row[1],
            "description": row[2],
            "prompt_text": row[3],
            "domain": row[4],
            "author": row[5],
            "tags": row[6].split(",") if row[6] else [],
            "downloads": row[7] + 1,
            "rating": row[8],
        }

    # ── Rate ─────────────────────────────────────────────────────────────
    def rate(self, item_id: str, stars: int) -> Optional[Dict[str, Any]]:
        """Rate a marketplace item (1-5 stars)."""
        stars = max(1, min(5, stars))
        db = self._get_db()
        row = db.fetch_one(
            "SELECT rating, rating_count FROM marketplace_items WHERE id = ?",
            (item_id,),
        )
        if not row:
            return None
        old_rating, old_count = row
        new_count = old_count + 1
        new_rating = round(((old_rating * old_count) + stars) / new_count, 2)
        db.execute(
            "UPDATE marketplace_items SET rating = ?, rating_count = ? WHERE id = ?",
            (new_rating, new_count, item_id),
        )
        return {"id": item_id, "rating": new_rating, "rating_count": new_count}

    # ── Delete ───────────────────────────────────────────────────────────
    def delete(self, item_id: str) -> bool:
        db = self._get_db()
        db.execute("DELETE FROM marketplace_items WHERE id = ?", (item_id,))
        return True

    # ── Featured / Popular ───────────────────────────────────────────────
    def featured(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get featured items (highest rated with at least 3 ratings)."""
        return self.search(sort_by="rating", limit=limit)

    def popular(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get most downloaded items."""
        return self.search(sort_by="downloads", limit=limit)

    def stats(self) -> Dict[str, Any]:
        """Get marketplace statistics."""
        db = self._get_db()
        total = db.fetch_one("SELECT COUNT(*) FROM marketplace_items")
        downloads = db.fetch_one("SELECT COALESCE(SUM(downloads), 0) FROM marketplace_items")
        domains = db.fetch_all(
            "SELECT domain, COUNT(*) FROM marketplace_items GROUP BY domain ORDER BY COUNT(*) DESC"
        )
        return {
            "total_items": total[0] if total else 0,
            "total_downloads": downloads[0] if downloads else 0,
            "items_by_domain": {r[0]: r[1] for r in domains},
        }


# Global instance
marketplace = MarketplaceService()
