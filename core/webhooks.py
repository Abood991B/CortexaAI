"""
Webhook Notifications for CortexaAI.

Send HTTP POST callbacks when workflow events occur (completed, failed, etc.).
Supports retry with exponential backoff.
"""

import asyncio
import json
import time
from typing import Dict, Any, List, Optional

from config.config import get_logger

logger = get_logger(__name__)

# Try to use the project's httpx dependency
try:
    import httpx
    _HAS_HTTPX = True
except ImportError:
    _HAS_HTTPX = False


class WebhookEvent:
    WORKFLOW_COMPLETED = "workflow.completed"
    WORKFLOW_FAILED = "workflow.failed"
    WORKFLOW_CANCELLED = "workflow.cancelled"
    BATCH_COMPLETED = "batch.completed"
    REGRESSION_COMPLETED = "regression.completed"


class WebhookManager:
    """Manage webhook registrations and deliver notifications."""

    def __init__(self):
        self._subscriptions: Dict[str, Dict[str, Any]] = {}  # id -> subscription
        self._delivery_log: List[Dict[str, Any]] = []

    def subscribe(
        self,
        url: str,
        events: Optional[List[str]] = None,
        secret: Optional[str] = None,
        name: str = "",
    ) -> Dict[str, Any]:
        """Register a webhook subscription."""
        import uuid
        sub_id = f"wh_{uuid.uuid4().hex[:8]}"
        sub = {
            "id": sub_id,
            "url": url,
            "events": events or [WebhookEvent.WORKFLOW_COMPLETED],
            "secret": secret,
            "name": name or url[:50],
            "created_at": time.time(),
            "active": True,
        }
        self._subscriptions[sub_id] = sub
        logger.info(f"Webhook subscription created: {sub_id} → {url}")
        return {k: v for k, v in sub.items() if k != "secret"}

    def unsubscribe(self, sub_id: str) -> bool:
        if sub_id in self._subscriptions:
            del self._subscriptions[sub_id]
            return True
        return False

    def list_subscriptions(self) -> List[Dict[str, Any]]:
        return [
            {k: v for k, v in s.items() if k != "secret"}
            for s in self._subscriptions.values()
        ]

    async def notify(self, event: str, payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Send webhook notifications for an event to all matching subscribers."""
        results = []
        for sub in self._subscriptions.values():
            if not sub.get("active"):
                continue
            if event not in sub.get("events", []):
                continue
            delivery = await self._deliver(sub, event, payload)
            results.append(delivery)
        return results

    async def _deliver(
        self,
        subscription: Dict[str, Any],
        event: str,
        payload: Dict[str, Any],
        max_retries: int = 3,
    ) -> Dict[str, Any]:
        """Deliver a webhook with retry logic."""
        url = subscription["url"]
        body = {
            "event": event,
            "timestamp": time.time(),
            "data": payload,
        }

        # Add HMAC signature if secret is configured
        headers = {"Content-Type": "application/json"}
        if subscription.get("secret"):
            import hashlib
            import hmac
            sig = hmac.new(
                subscription["secret"].encode(),
                json.dumps(body, sort_keys=True).encode(),
                hashlib.sha256,
            ).hexdigest()
            headers["X-Webhook-Signature"] = f"sha256={sig}"

        delivery = {
            "subscription_id": subscription["id"],
            "url": url,
            "event": event,
            "status": "pending",
            "attempts": 0,
            "timestamp": time.time(),
        }

        if not _HAS_HTTPX:
            delivery["status"] = "skipped"
            delivery["error"] = "httpx not installed"
            self._delivery_log.append(delivery)
            return delivery

        for attempt in range(max_retries):
            delivery["attempts"] = attempt + 1
            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    response = await client.post(url, json=body, headers=headers)
                    delivery["status_code"] = response.status_code
                    if response.status_code < 400:
                        delivery["status"] = "delivered"
                        self._delivery_log.append(delivery)
                        return delivery
                    else:
                        delivery["error"] = f"HTTP {response.status_code}"
            except Exception as e:
                delivery["error"] = str(e)

            # Exponential backoff
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)

        delivery["status"] = "failed"
        self._delivery_log.append(delivery)
        logger.warning(f"Webhook delivery failed after {max_retries} attempts: {url}")
        return delivery

    def get_delivery_log(self, limit: int = 50) -> List[Dict[str, Any]]:
        return self._delivery_log[-limit:]

    async def notify_workflow_completed(self, result: Dict[str, Any]):
        """Convenience: notify on workflow completion."""
        await self.notify(WebhookEvent.WORKFLOW_COMPLETED, result)

    async def notify_workflow_failed(self, error_info: Dict[str, Any]):
        """Convenience: notify on workflow failure."""
        await self.notify(WebhookEvent.WORKFLOW_FAILED, error_info)


# Global instance
webhook_manager = WebhookManager()
