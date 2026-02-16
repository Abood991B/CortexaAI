"""
Server-Sent Events (SSE) Streaming for CortexaAI.

Streams prompt-processing progress to the client in real-time so that each
workflow stage (classify → expert → improve → evaluate → finalize) is visible
as it completes.
"""

import asyncio
import copy
import json
import time
from typing import Any, AsyncGenerator, Dict, Optional

from config.config import (
    get_logger, cache_manager, perf_config, generate_prompt_cache_key
)

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

    Both standard and LangGraph modes now run step-by-step so the client
    receives real-time progress for every stage.

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
        if not coordinator:
            yield await _sse_event("error", {"message": "No coordinator available"})
            return

        # ── Fast path: check workflow-level cache ────────────────────
        if perf_config.enable_caching:
            wf_cache_key = generate_prompt_cache_key(prompt, prefix="workflow")
            cached = cache_manager.get(wf_cache_key)
            if cached:
                cached_copy = copy.deepcopy(cached)
                cached_copy["workflow_id"] = f"sse_cached_{int(time.time())}"
                cached_copy["metadata"] = cached_copy.get("metadata", {})
                cached_copy["metadata"]["is_cache_hit"] = True
                try:
                    coordinator._record_workflow(cached_copy)
                except Exception:
                    pass
                yield await _sse_event("completed", {
                    "result": cached_copy,
                    "processing_time": 0.0,
                })
                return

        # ── Step 1: Classify domain + detect prompt type ─────────────
        yield await _sse_event("classifying", {"message": "Classifying domain..."})

        # Both classifiers are now pure heuristics (no LLM) — run sequentially.
        classification = await coordinator.classifier.classify_prompt(prompt)
        if prompt_type == "auto":
            prompt_type = await coordinator.classifier.classify_prompt_type(prompt)

        domain = classification.get("domain", "general")

        yield await _sse_event("classified", {
            "domain": domain,
            "confidence": classification.get("confidence", 0),
        })

        # ── Step 3: Expert improvement ───────────────────────────────────
        yield await _sse_event("improving", {
            "message": f"Expert ({domain}) generating improvement...",
        })

        if use_langgraph:
            # Use LangGraph-specific expert for higher-quality structured output
            from agents.langgraph_expert import get_langgraph_expert
            expert = get_langgraph_expert(domain, classification.get("reasoning", f"Expert in {domain}"))
        else:
            expert = coordinator._get_or_create_expert_agent(domain, classification)

        improvement = await expert.improve_prompt(
            original_prompt=prompt,
            prompt_type=prompt_type,
            key_topics=classification.get("key_topics", []),
        )
        improved_prompt = improvement.get("improved_prompt") or improvement.get("solution", prompt)

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
            "workflow_id": f"sse_{int(time.time())}",
            "status": "completed",
            "timestamp": time.time(),
            "processing_time_seconds": elapsed,
            "input": {
                "original_prompt": prompt,
                "prompt_type": prompt_type,
            },
            "output": {
                "optimized_prompt": improved_prompt,
                "domain": domain,
                "quality_score": evaluation.get("overall_score", 0),
                "iterations_used": iterations,
                "passes_threshold": evaluation.get("passes_threshold", False),
            },
            "analysis": {
                "classification": classification,
                "improvements": improvement,
                "evaluation": evaluation,
            },
            "comparison": {
                "side_by_side": {
                    "original": prompt,
                    "optimized": improved_prompt,
                },
                "improvement_ratio": evaluation.get("overall_score", 0),
            },
            "metadata": {
                "framework": "langgraph" if use_langgraph else "standard",
            },
        }

        # Record in coordinator history so the Dashboard updates
        try:
            coordinator._record_workflow(final_result)
        except Exception:
            pass

        # Cache the result for subsequent identical prompts
        if perf_config.enable_caching:
            try:
                wf_cache_key = generate_prompt_cache_key(prompt, prefix="workflow")
                cache_manager.set(wf_cache_key, final_result, perf_config.cache_ttl)
            except Exception:
                pass

        yield await _sse_event("completed", {
            "result": final_result,
            "processing_time": elapsed,
        })

    except Exception as exc:
        logger.error(f"Streaming workflow error: {exc}")
        yield await _sse_event("error", {
            "message": str(exc),
            "timestamp": time.time(),
        })


# ─────────────────────────────────────────────────────────────────────────────
#  Re-iterate: refine an already-optimized prompt
# ─────────────────────────────────────────────────────────────────────────────

async def stream_reiterate(
    original_prompt: str,
    current_prompt: str,
    domain: str = "general",
    use_langgraph: bool = False,
    user_feedback: str | None = None,
    coordinator=None,
) -> AsyncGenerator[str, None]:
    """SSE generator that takes an *already-optimized* prompt and refines it
    further.  Steps:

    1. Evaluate the current prompt to find weaknesses.
    2. Build feedback-enriched input for the expert.
    3. Run improve_prompt again.
    4. Re-evaluate the new version.
    5. Return the refined result.

    Events: started → evaluating → improving → evaluated → completed | error
    """
    workflow_start = time.time()

    yield await _sse_event("started", {
        "message": "Re-iteration started — analysing current prompt…",
        "timestamp": time.time(),
    })

    try:
        if not coordinator:
            yield await _sse_event("error", {"message": "No coordinator available"})
            return

        # ── Step 1: Fast heuristic evaluation (no LLM, instant) ──────────
        # Skip slow LLM-based pre-evaluation since user already sees the score
        yield await _sse_event("evaluating", {
            "message": "Analyzing current prompt for improvement areas…",
        })

        # Use fast heuristic evaluation instead of slow LLM evaluation
        pre_eval = coordinator.evaluator.heuristic_evaluate(
            original_prompt=original_prompt,
            improved_prompt=current_prompt,
            domain=domain,
        )

        weaknesses = pre_eval.get("weaknesses", [])
        specific_feedback = pre_eval.get("specific_feedback", [])
        criteria_scores = pre_eval.get("criteria_scores", {})

        yield await _sse_event("evaluated", {
            "score": pre_eval.get("overall_score", 0),
            "weaknesses": weaknesses,
            "message": f"Found {len(weaknesses)} area(s) to improve",
        })

        # ── Step 2: Build feedback-enriched prompt for expert ────────────
        yield await _sse_event("improving", {
            "message": "Generating refined improvement based on feedback…",
        })

        feedback_parts = []
        if user_feedback:
            feedback_parts.append(f"USER FEEDBACK (highest priority):\n{user_feedback}")
        if weaknesses:
            feedback_parts.append("WEAKNESSES TO FIX:\n" + "\n".join(f"- {w}" for w in weaknesses))
        if specific_feedback:
            feedback_parts.append("SPECIFIC FEEDBACK:\n" + "\n".join(f"- {f}" for f in specific_feedback))
        if criteria_scores:
            sorted_criteria = sorted(criteria_scores.items(), key=lambda x: x[1])
            low_criteria = [f"{k}: {v:.2f}" for k, v in sorted_criteria[:3]]
            feedback_parts.append(f"LOWEST SCORING CRITERIA: {', '.join(low_criteria)}")

        feedback_text = "\n\n".join(feedback_parts) if feedback_parts else "General refinement pass — tighten language and add missing structure."

        enriched_prompt = (
            f"[RE-ITERATION — Address ALL issues below in your improvement]\n"
            f"{feedback_text}\n\n"
            f"[CURRENT PROMPT TO IMPROVE]\n{current_prompt}"
        )

        # Create expert agent
        if use_langgraph:
            from agents.langgraph_expert import get_langgraph_expert
            expert = get_langgraph_expert(domain, f"Expert in {domain}")
        else:
            classification = {"domain": domain, "confidence": 0.9, "key_topics": [], "reasoning": "Re-iteration"}
            expert = coordinator._get_or_create_expert_agent(domain, classification)

        improvement = await expert.improve_prompt(
            original_prompt=enriched_prompt,
            prompt_type="structured",
            key_topics=[],
        )
        improved_prompt = improvement.get("improved_prompt") or improvement.get("solution", current_prompt)

        yield await _sse_event("improved", {
            "preview": improved_prompt[:200],
            "improvements_made": improvement.get("improvements_made", []),
        })

        # ── Step 3: Post-improvement evaluation ──────────────────────────
        yield await _sse_event("evaluating", {"message": "Re-evaluating refined prompt…"})

        post_eval = await coordinator.evaluator.evaluate_prompt(
            original_prompt=original_prompt,
            improved_prompt=improved_prompt,
            domain=domain,
            prompt_type="structured",
        )

        yield await _sse_event("evaluated", {
            "score": post_eval.get("overall_score", 0),
            "passes_threshold": post_eval.get("passes_threshold", False),
            "iterations": 1,
        })

        # ── Step 4: Finalize ─────────────────────────────────────────────
        elapsed = round(time.time() - workflow_start, 2)
        final_result = {
            "workflow_id": f"reiterate_{int(time.time())}",
            "status": "completed",
            "timestamp": time.time(),
            "processing_time_seconds": elapsed,
            "input": {
                "original_prompt": original_prompt,
                "previous_prompt": current_prompt,
                "prompt_type": "structured",
            },
            "output": {
                "optimized_prompt": improved_prompt,
                "domain": domain,
                "quality_score": post_eval.get("overall_score", 0),
                "iterations_used": 1,
                "passes_threshold": post_eval.get("passes_threshold", False),
            },
            "analysis": {
                "classification": {"domain": domain, "confidence": 0.9, "key_topics": [], "reasoning": "Re-iteration"},
                "improvements": improvement,
                "evaluation": post_eval,
                "pre_evaluation": pre_eval,
            },
            "comparison": {
                "side_by_side": {
                    "original": current_prompt,
                    "optimized": improved_prompt,
                },
                "improvement_ratio": post_eval.get("overall_score", 0),
            },
            "metadata": {
                "framework": "langgraph" if use_langgraph else "standard",
                "is_reiteration": True,
            },
        }

        try:
            coordinator._record_workflow(final_result)
        except Exception:
            pass

        yield await _sse_event("completed", {
            "result": final_result,
            "processing_time": elapsed,
        })

    except Exception as exc:
        logger.error(f"Re-iterate streaming error: {exc}")
        yield await _sse_event("error", {
            "message": str(exc),
            "timestamp": time.time(),
        })
