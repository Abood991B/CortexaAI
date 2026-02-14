"""
Server-Sent Events (SSE) Streaming for CortexaAI.

Streams prompt-processing progress to the client in real-time so that each
workflow stage (classify → expert → improve → evaluate → finalize) is visible
as it completes.
"""

import asyncio
import json
import time
from typing import Any, AsyncGenerator, Dict, Optional

from config.config import get_logger

logger = get_logger(__name__)


async def _sse_event(event: str, data: Any) -> str:
    """Format a single SSE event."""
    payload = json.dumps(data) if not isinstance(data, str) else data
    return f"event: {event}\ndata: {payload}\n\n"


async def stream_workflow(
    prompt: str,
    prompt_type: str = "auto",
    use_langgraph: bool = False,
    coordinator=None,
    langgraph_fn=None,
) -> AsyncGenerator[str, None]:
    """
    Async generator that yields SSE events as a workflow progresses.

    Events emitted:
        started        – workflow begins
        classifying    – domain classification in progress
        classified     – domain found
        improving      – expert generating improvement
        improved       – improvement complete
        evaluating     – evaluation running
        evaluated      – evaluation result
        iterating      – additional iteration
        completed      – final result
        error          – error occurred
    """
    workflow_start = time.time()

    yield await _sse_event("started", {
        "message": "Workflow started",
        "prompt_preview": prompt[:120],
        "timestamp": time.time(),
    })

    try:
        # ── Step 1: Classification ───────────────────────────────────────
        yield await _sse_event("classifying", {"message": "Classifying domain..."})

        if use_langgraph and langgraph_fn:
            # LangGraph path — single call, limited intermediate detail
            result = await langgraph_fn(prompt=prompt, prompt_type=prompt_type)

            yield await _sse_event("classified", {
                "domain": result.get("output", {}).get("domain", "unknown"),
            })
            yield await _sse_event("improved", {
                "preview": result.get("output", {}).get("optimized_prompt", "")[:200],
            })
            yield await _sse_event("evaluated", {
                "score": result.get("output", {}).get("quality_score", 0),
            })
            yield await _sse_event("completed", {
                "result": result,
                "processing_time": round(time.time() - workflow_start, 2),
            })
            return

        if not coordinator:
            yield await _sse_event("error", {"message": "No coordinator available"})
            return

        # ── Step 1: Classify domain ──────────────────────────────────────
        classification = await coordinator.classifier.classify_prompt(prompt)
        domain = classification.get("domain", "general")

        yield await _sse_event("classified", {
            "domain": domain,
            "confidence": classification.get("confidence", 0),
        })

        # ── Step 2: Auto-detect prompt type ──────────────────────────────
        if prompt_type == "auto":
            prompt_type = await coordinator.classifier.classify_prompt_type(prompt)

        # ── Step 3: Expert improvement ───────────────────────────────────
        yield await _sse_event("improving", {
            "message": f"Expert ({domain}) generating improvement...",
        })

        expert = coordinator._get_or_create_expert_agent(domain, classification)
        improvement = await expert.improve_prompt(
            original_prompt=prompt,
            prompt_type=prompt_type,
            key_topics=classification.get("key_topics", []),
        )
        improved_prompt = improvement.get("improved_prompt", prompt)

        yield await _sse_event("improved", {
            "preview": improved_prompt[:200],
            "improvements_made": improvement.get("improvements_made", []),
        })

        # ── Step 4: Evaluate ─────────────────────────────────────────────
        yield await _sse_event("evaluating", {"message": "Evaluating quality..."})

        evaluation, iterations = await coordinator.evaluator.run_evaluation_loop(
            original_prompt=prompt,
            improved_prompt=improved_prompt,
            domain=domain,
            expert_agent=expert,
            prompt_type=prompt_type,
        )

        yield await _sse_event("evaluated", {
            "score": evaluation.get("overall_score", 0),
            "passes_threshold": evaluation.get("passes_threshold", False),
            "iterations": iterations,
        })

        # ── Step 5: Finalize ─────────────────────────────────────────────
        elapsed = round(time.time() - workflow_start, 2)
        final_result = {
            "status": "completed",
            "output": {
                "optimized_prompt": improved_prompt,
                "domain": domain,
                "quality_score": evaluation.get("overall_score", 0),
                "iterations_used": iterations,
            },
            "processing_time_seconds": elapsed,
        }
        yield await _sse_event("completed", {"result": final_result})

    except Exception as exc:
        logger.error(f"Streaming workflow error: {exc}")
        yield await _sse_event("error", {
            "message": str(exc),
            "timestamp": time.time(),
        })
