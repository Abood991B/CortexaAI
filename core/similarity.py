"""
Embedding Similarity Search for CortexaAI.

Lightweight vector-based similarity using TF-IDF / cosine similarity.
No external vector DB needed — all in-memory with optional SQLite persistence.
"""

import math
import re
import time
from typing import Dict, Any, List, Optional, Tuple
from collections import Counter

from config.config import get_logger

logger = get_logger(__name__)


# ── TF-IDF Vectorizer (zero dependencies) ───────────────────────────────


def _tokenize(text: str) -> List[str]:
    """Simple whitespace + punctuation tokenizer."""
    return re.findall(r"\b\w+\b", text.lower())


def _term_frequency(tokens: List[str]) -> Dict[str, float]:
    counts = Counter(tokens)
    total = len(tokens) if tokens else 1
    return {t: c / total for t, c in counts.items()}


def _cosine_similarity(vec_a: Dict[str, float], vec_b: Dict[str, float]) -> float:
    """Cosine similarity between two sparse vectors."""
    common = set(vec_a.keys()) & set(vec_b.keys())
    if not common:
        return 0.0
    dot = sum(vec_a[k] * vec_b[k] for k in common)
    mag_a = math.sqrt(sum(v * v for v in vec_a.values()))
    mag_b = math.sqrt(sum(v * v for v in vec_b.values()))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


class SimilarityEngine:
    """TF-IDF based similarity search for prompts."""

    def __init__(self):
        self._corpus: List[Dict[str, Any]] = []  # {id, text, domain, vector, metadata}
        self._idf: Dict[str, float] = {}
        self._doc_count = 0

    def _rebuild_idf(self):
        """Recalculate IDF values from entire corpus."""
        df: Dict[str, int] = {}
        n = len(self._corpus)
        for doc in self._corpus:
            tokens = set(_tokenize(doc["text"]))
            for t in tokens:
                df[t] = df.get(t, 0) + 1
        self._idf = {t: math.log((n + 1) / (count + 1)) + 1 for t, count in df.items()}
        self._doc_count = n

    def _vectorize(self, text: str) -> Dict[str, float]:
        """Convert text to TF-IDF vector."""
        tokens = _tokenize(text)
        tf = _term_frequency(tokens)
        return {t: tf_val * self._idf.get(t, 1.0) for t, tf_val in tf.items()}

    # ── Index Management ─────────────────────────────────────────────────
    def add_document(
        self,
        doc_id: str,
        text: str,
        domain: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Add a document to the similarity index."""
        entry = {
            "id": doc_id,
            "text": text,
            "domain": domain,
            "metadata": metadata or {},
            "added_at": time.time(),
        }
        # Check for duplicates
        self._corpus = [d for d in self._corpus if d["id"] != doc_id]
        self._corpus.append(entry)

        # Rebuild IDF periodically (every 10 docs or first 50)
        if len(self._corpus) <= 50 or len(self._corpus) % 10 == 0:
            self._rebuild_idf()

        return {"id": doc_id, "indexed": True, "corpus_size": len(self._corpus)}

    def remove_document(self, doc_id: str) -> bool:
        before = len(self._corpus)
        self._corpus = [d for d in self._corpus if d["id"] != doc_id]
        if len(self._corpus) < before:
            self._rebuild_idf()
            return True
        return False

    def clear(self):
        self._corpus.clear()
        self._idf.clear()
        self._doc_count = 0

    # ── Search ───────────────────────────────────────────────────────────
    def search(
        self,
        query: str,
        top_k: int = 5,
        domain: Optional[str] = None,
        min_score: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """Find similar documents to a query."""
        if not self._corpus:
            return []

        if not self._idf:
            self._rebuild_idf()

        query_vec = self._vectorize(query)
        results: List[Tuple[float, Dict[str, Any]]] = []

        candidates = self._corpus
        if domain:
            candidates = [d for d in candidates if d["domain"] == domain]

        for doc in candidates:
            doc_vec = self._vectorize(doc["text"])
            score = _cosine_similarity(query_vec, doc_vec)
            if score >= min_score:
                results.append((score, doc))

        results.sort(key=lambda x: x[0], reverse=True)

        return [
            {
                "id": doc["id"],
                "text": doc["text"][:300],
                "domain": doc["domain"],
                "similarity": round(score, 4),
                "metadata": doc["metadata"],
            }
            for score, doc in results[:top_k]
        ]

    # ── Deduplication ────────────────────────────────────────────────────
    def find_duplicates(self, threshold: float = 0.85) -> List[Dict[str, Any]]:
        """Find near-duplicate documents in the corpus."""
        if not self._idf:
            self._rebuild_idf()

        duplicates = []
        n = len(self._corpus)
        for i in range(n):
            vec_i = self._vectorize(self._corpus[i]["text"])
            for j in range(i + 1, n):
                vec_j = self._vectorize(self._corpus[j]["text"])
                sim = _cosine_similarity(vec_i, vec_j)
                if sim >= threshold:
                    duplicates.append({
                        "doc_a": self._corpus[i]["id"],
                        "doc_b": self._corpus[j]["id"],
                        "similarity": round(sim, 4),
                    })
        return duplicates

    # ── Bulk index from DB ───────────────────────────────────────────────
    def index_from_history(self, limit: int = 500) -> int:
        """Index historical workflows from the database."""
        try:
            from core.database import db
            rows = db.fetch_all(
                "SELECT id, input_prompt, domain FROM workflows ORDER BY created_at DESC LIMIT ?",
                (limit,),
            )
            count = 0
            for r in rows:
                if r[1]:
                    self.add_document(doc_id=r[0], text=r[1], domain=r[2] or "")
                    count += 1
            self._rebuild_idf()
            logger.info(f"Indexed {count} prompts from history")
            return count
        except Exception as e:
            logger.error(f"Failed to index from history: {e}")
            return 0

    # ── Stats ────────────────────────────────────────────────────────────
    def stats(self) -> Dict[str, Any]:
        domains = {}
        for doc in self._corpus:
            d = doc.get("domain", "unknown")
            domains[d] = domains.get(d, 0) + 1
        return {
            "corpus_size": len(self._corpus),
            "vocabulary_size": len(self._idf),
            "domains": domains,
        }


# Global instance
similarity_engine = SimilarityEngine()
