"""
Batch Processing for CortexaAI.

Process multiple prompts in a single API call with parallel or sequential
execution, progress tracking, and result aggregation.
"""

import asyncio
import time
import uuid
from typing import Dict, Any, List, Optional, Callable

from config.config import get_logger

logger = get_logger(__name__)


class BatchJob:
    """Represents a single batch processing job."""

    def __init__(self, batch_id: str, prompts: List[Dict[str, Any]], concurrency: int = 3):
        self.batch_id = batch_id
        self.prompts = prompts
        self.concurrency = concurrency
        self.results: List[Dict[str, Any]] = []
        self.status = "pending"
        self.started_at: Optional[float] = None
        self.completed_at: Optional[float] = None
        self.progress = 0
        self.total = len(prompts)
        self.errors: List[Dict[str, Any]] = []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "batch_id": self.batch_id,
            "status": self.status,
            "total": self.total,
            "progress": self.progress,
            "completed": len(self.results),
            "errors": len(self.errors),
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "elapsed_seconds": round(time.time() - self.started_at, 2) if self.started_at else 0,
        }


class BatchProcessor:
    """Manage and process batch prompt requests."""

    def __init__(self):
        self._jobs: Dict[str, BatchJob] = {}

    def create_batch(
        self,
        prompts: List[Dict[str, Any]],
        concurrency: int = 3,
    ) -> Dict[str, Any]:
        """Create a new batch job. Each prompt dict should have at least 'prompt' key."""
        batch_id = f"batch_{uuid.uuid4().hex[:10]}"
        job = BatchJob(batch_id, prompts, concurrency)
        self._jobs[batch_id] = job
        logger.info(f"Created batch {batch_id} with {len(prompts)} prompts (concurrency={concurrency})")
        return job.to_dict()

    def get_status(self, batch_id: str) -> Optional[Dict[str, Any]]:
        job = self._jobs.get(batch_id)
        if not job:
            return None
        return job.to_dict()

    def get_results(self, batch_id: str) -> Optional[Dict[str, Any]]:
        job = self._jobs.get(batch_id)
        if not job:
            return None
        return {
            **job.to_dict(),
            "results": job.results,
            "errors": job.errors,
        }

    async def run_batch(
        self,
        batch_id: str,
        processor_fn: Callable,
    ) -> Dict[str, Any]:
        """
        Execute a batch job using the provided processor function.

        Args:
            batch_id: The batch to run
            processor_fn: async fn(prompt: str, prompt_type: str) -> Dict
        """
        job = self._jobs.get(batch_id)
        if not job:
            return {"error": "Batch not found"}

        job.status = "running"
        job.started_at = time.time()

        semaphore = asyncio.Semaphore(job.concurrency)

        async def _process_one(index: int, item: Dict[str, Any]):
            async with semaphore:
                prompt_text = item.get("prompt", "")
                prompt_type = item.get("prompt_type", "auto")
                try:
                    result = await processor_fn(prompt_text, prompt_type)
                    job.results.append({
                        "index": index,
                        "prompt_preview": prompt_text[:100],
                        "status": "completed",
                        "result": result,
                    })
                except Exception as e:
                    logger.error(f"Batch {batch_id} item {index} failed: {e}")
                    job.errors.append({
                        "index": index,
                        "prompt_preview": prompt_text[:100],
                        "error": str(e),
                    })
                finally:
                    job.progress += 1

        tasks = [_process_one(i, p) for i, p in enumerate(job.prompts)]
        await asyncio.gather(*tasks)

        job.status = "completed"
        job.completed_at = time.time()
        logger.info(
            f"Batch {batch_id} completed: {len(job.results)} succeeded, {len(job.errors)} failed"
        )
        return self.get_results(batch_id)

    def list_batches(self) -> List[Dict[str, Any]]:
        return [job.to_dict() for job in self._jobs.values()]

    def cancel_batch(self, batch_id: str) -> bool:
        job = self._jobs.get(batch_id)
        if not job or job.status != "running":
            return False
        job.status = "cancelled"
        return True


# Global instance
batch_processor = BatchProcessor()
