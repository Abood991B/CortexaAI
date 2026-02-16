"""Prompt processing endpoints: sync/async processing, streaming, re-iteration."""

import uuid
import asyncio
from datetime import datetime
from typing import Dict, Any

from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional

from src.deps import (
    logger,
    coordinator,
    active_workflows,
    MAX_PROMPT_LENGTH,
    WorkflowCancellationError,
    process_prompt_with_langgraph_cancellable,
    clear_workflow_caches,
    _cleanup_finished_workflows,
    PromptRequest,
    PromptResponse,
    cache_manager,
    generate_prompt_cache_key,
    perf_config,
    stream_workflow,
    process_prompt_with_langgraph,
)
import hashlib

router = APIRouter()


# ---------------------------------------------------------------------------
# Shared background-workflow helper
# ---------------------------------------------------------------------------

async def _run_background_workflow(
    workflow_id: str,
    cancellation_event: asyncio.Event,
    request: PromptRequest,
    execute_fn,
    label: str = "Workflow",
    post_process_fn=None,
):
    """Run *execute_fn* inside the standard cancellation / status / error
    boilerplate shared by all background workflow tasks.

    Args:
        workflow_id:       ID stored in *active_workflows*.
        cancellation_event: ``asyncio.Event`` that signals cancellation.
        request:           The original ``PromptRequest``.
        execute_fn:        ``async () -> dict`` that does the real work.
        label:             Human-friendly prefix for log messages.
        post_process_fn:   Optional ``async (result) -> None`` called after
                           success and before updating status.
    """
    try:
        if cancellation_event.is_set():
            logger.info(f"{label} {workflow_id} was cancelled before starting")
            active_workflows[workflow_id]["status"] = "cancelled"
            clear_workflow_caches(request.prompt, request.prompt_type)
            return

        result = await execute_fn()

        if cancellation_event.is_set():
            logger.info(f"{label} {workflow_id} was cancelled during execution")
            active_workflows[workflow_id]["status"] = "cancelled"
            clear_workflow_caches(request.prompt, request.prompt_type)
            return

        if post_process_fn is not None:
            post_process_fn(result)

        active_workflows[workflow_id]["status"] = "completed"
        active_workflows[workflow_id]["result"] = result
        logger.info(f"{label} {workflow_id} completed successfully")
        _cleanup_finished_workflows()

    except WorkflowCancellationError:
        logger.info(f"{label} {workflow_id} was cancelled during execution")
        active_workflows[workflow_id]["status"] = "cancelled"
        clear_workflow_caches(request.prompt, request.prompt_type)
    except Exception as e:
        if cancellation_event.is_set():
            logger.info(f"{label} {workflow_id} was cancelled during execution")
            active_workflows[workflow_id]["status"] = "cancelled"
            clear_workflow_caches(request.prompt, request.prompt_type)
            return
        logger.error(f"{label} {workflow_id} failed: {e}")
        active_workflows[workflow_id]["status"] = "failed"
        active_workflows[workflow_id]["error"] = str(e)


# ---------------------------------------------------------------------------
# POST /api/process-prompt
# ---------------------------------------------------------------------------

@router.post("/api/process-prompt", response_model=PromptResponse)
async def process_prompt(request: PromptRequest, background_tasks: BackgroundTasks) -> PromptResponse:
    """Process a prompt through the multi-agent workflow.
    If request.synchronous is True, run inline and return final result.
    """
    if len(request.prompt) > MAX_PROMPT_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=f"Prompt exceeds maximum length of {MAX_PROMPT_LENGTH:,} characters.",
        )

    # Synchronous mode: run inline and return final
    if request.synchronous:
        prompt_hash = hashlib.sha256(request.prompt.encode()).hexdigest()[:12]
        logger.info(f"Processing prompt synchronously (hash={prompt_hash})...")
        if request.use_langgraph:
            result = await process_prompt_with_langgraph(
                prompt=request.prompt,
                prompt_type=request.prompt_type,
            )
            coordinator._record_workflow(result)
        else:
            result = await coordinator.process_prompt(
                prompt=request.prompt,
                prompt_type=request.prompt_type,
                return_comparison=request.return_comparison,
            )

        response = PromptResponse(
            workflow_id=result.get("workflow_id", f"workflow_{uuid.uuid4().hex[:8]}"),
            status=result.get("status", "completed"),
            message="Workflow completed successfully.",
            timestamp=datetime.now().isoformat(),
            processing_time_seconds=result.get("processing_time_seconds", 0.0),
            input=request.model_dump(),
            output=result.get("output", {}),
            analysis=result.get("analysis", {}),
            comparison=result.get("comparison"),
            metadata=result.get("metadata", {}),
        )

        if request.use_langgraph and perf_config.enable_caching:
            cache_key = generate_prompt_cache_key(request.prompt, "langgraph_workflow")
            cache_manager.set(cache_key, response.model_dump(), perf_config.cache_ttl)

        return response

    # Default async/background behavior
    if perf_config.enable_caching and request.use_langgraph:
        cache_key = generate_prompt_cache_key(request.prompt, "langgraph_workflow")
        cached_result = cache_manager.get(cache_key)
        if cached_result:
            prompt_hash = hashlib.sha256(request.prompt.encode()).hexdigest()[:12]
            logger.info(f"Returning cached result for LangGraph workflow (hash={prompt_hash})...")
            return PromptResponse(**cached_result)

    workflow_id = f"workflow_{uuid.uuid4().hex[:8]}"
    cancellation_event = asyncio.Event()
    active_workflows[workflow_id] = {
        "status": "running",
        "task": None,
        "cancellation_event": cancellation_event,
        "start_time": datetime.now(),
    }

    try:
        prompt_hash = hashlib.sha256(request.prompt.encode()).hexdigest()[:12]
        logger.info(f"Starting workflow {workflow_id} (hash={prompt_hash})...")

        async def _execute():
            if request.use_langgraph:
                result = await process_prompt_with_langgraph_cancellable(
                    prompt=request.prompt,
                    prompt_type=request.prompt_type,
                    cancellation_event=cancellation_event,
                )
                coordinator._record_workflow(result)
            else:
                result = await coordinator.process_prompt(
                    prompt=request.prompt,
                    prompt_type=request.prompt_type,
                    return_comparison=request.return_comparison,
                )
            return result

        def _post(result):
            if perf_config.enable_caching and request.use_langgraph:
                cache_key = generate_prompt_cache_key(request.prompt, "langgraph_workflow")
                cacheable_response = PromptResponse(
                    workflow_id=workflow_id,
                    status="completed",
                    message="Workflow completed successfully.",
                    timestamp=datetime.now().isoformat(),
                    processing_time_seconds=result.get("processing_time_seconds", 0),
                    input=request.model_dump(),
                    output=result.get("output", {}),
                    analysis=result.get("analysis", {}),
                    metadata=result.get("metadata", {}),
                )
                cache_manager.set(cache_key, cacheable_response.model_dump(), perf_config.cache_ttl)

        background_tasks.add_task(
            _run_background_workflow,
            workflow_id, cancellation_event, request,
            _execute, "Workflow", _post,
        )

        return PromptResponse(
            workflow_id=workflow_id,
            status="running",
            message="Workflow started successfully.",
            timestamp=datetime.now().isoformat(),
            processing_time_seconds=0.0,
            input=request.model_dump(),
            output={},
            analysis={},
            metadata={},
        )

    except Exception as e:
        logger.error(f"Error starting workflow: {e}")
        active_workflows.pop(workflow_id, None)
        raise HTTPException(status_code=500, detail=f"Failed to start workflow: {str(e)}")


# ---------------------------------------------------------------------------
# POST /api/process-prompt-with-memory
# ---------------------------------------------------------------------------

@router.post("/api/process-prompt-with-memory", response_model=PromptResponse)
async def process_prompt_with_memory(request: PromptRequest, background_tasks: BackgroundTasks):
    """Process prompt with memory context using async workflow."""
    if not request.user_id:
        raise HTTPException(status_code=400, detail="User ID is required for memory-enhanced processing")

    if len(request.prompt) > MAX_PROMPT_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=f"Prompt exceeds maximum length of {MAX_PROMPT_LENGTH:,} characters.",
        )

    def _build_context_prompt(prompt: str, chat_history):
        """Build a context-aware prompt from chat history."""
        if not chat_history or len(chat_history) == 0:
            return prompt
        context_parts = ["Previous conversation context:"]
        for message in chat_history[-10:]:
            role = message.get("role", "user")
            content = message.get("content", "")
            prefix = "User" if role == "user" else "Assistant"
            context_parts.append(f"{prefix}: {content}")
        context_parts.append("\nCurrent request:")
        context_parts.append(f"User: {prompt}")
        context_parts.append(
            "\nPlease provide a response that takes into account the previous conversation context "
            "and continues the discussion appropriately."
        )
        return "\n".join(context_parts)

    # Synchronous mode
    if request.synchronous:
        prompt_hash = hashlib.sha256(request.prompt.encode()).hexdigest()[:12]
        logger.info(f"Processing memory workflow synchronously for user {request.user_id} (hash={prompt_hash})...")
        context_prompt = _build_context_prompt(request.prompt, request.chat_history)

        if request.advanced_mode:
            advanced_mode_result = await coordinator.handle_advanced_mode(
                prompt=request.prompt,
                chat_history=request.chat_history,
            )
            if advanced_mode_result["status"] == "needs_more_info":
                result = {
                    "output": {
                        "optimized_prompt": advanced_mode_result["content"],
                        "quality_score": 0,
                        "domain": "conversational",
                        "iterations_used": 1,
                    },
                    "analysis": {},
                    "metadata": {},
                    "processing_time_seconds": 0.1,
                    "status": "completed",
                }
            else:
                context_prompt = advanced_mode_result["content"]
                if request.use_langgraph:
                    result = await process_prompt_with_langgraph(
                        prompt=context_prompt,
                        prompt_type=request.prompt_type,
                    )
                    coordinator._record_workflow(result)
                else:
                    result = await coordinator.process_prompt(
                        prompt=context_prompt,
                        prompt_type=request.prompt_type,
                        return_comparison=request.return_comparison,
                    )
        elif request.use_langgraph:
            result = await process_prompt_with_langgraph(
                prompt=context_prompt,
                prompt_type=request.prompt_type,
            )
            coordinator._record_workflow(result)
        else:
            result = await coordinator.process_prompt(
                prompt=context_prompt,
                prompt_type=request.prompt_type,
                return_comparison=request.return_comparison,
            )

        if "metadata" not in result:
            result["metadata"] = {}
        result["metadata"]["original_prompt"] = request.prompt
        result["metadata"]["user_id"] = request.user_id
        result["metadata"]["has_context"] = bool(request.chat_history)

        return PromptResponse(
            workflow_id=result.get("workflow_id", f"workflow_{uuid.uuid4().hex[:8]}"),
            status=result.get("status", "completed"),
            message="Workflow completed successfully.",
            timestamp=datetime.now().isoformat(),
            processing_time_seconds=result.get("processing_time_seconds", 0.0),
            input=request.model_dump(),
            output=result.get("output", {}),
            analysis=result.get("analysis", {}),
            comparison=result.get("comparison"),
            metadata=result.get("metadata", {}),
        )

    # Default async/background behavior
    workflow_id = f"workflow_{uuid.uuid4().hex[:8]}"
    cancellation_event = asyncio.Event()
    active_workflows[workflow_id] = {
        "status": "running",
        "task": None,
        "cancellation_event": cancellation_event,
        "start_time": datetime.now(),
    }

    try:
        prompt_hash = hashlib.sha256(request.prompt.encode()).hexdigest()[:12]
        logger.info(f"Starting memory workflow {workflow_id} for user {request.user_id} (hash={prompt_hash})...")

        async def _execute():
            context_prompt = _build_context_prompt(request.prompt, request.chat_history)

            if request.advanced_mode:
                advanced_mode_result = await coordinator.handle_advanced_mode(
                    prompt=request.prompt,
                    chat_history=request.chat_history,
                )
                if advanced_mode_result["status"] == "needs_more_info":
                    return {
                        "output": {
                            "optimized_prompt": advanced_mode_result["content"],
                            "quality_score": 0,
                            "domain": "conversational",
                            "iterations_used": 1,
                        },
                        "analysis": {},
                        "metadata": {},
                        "processing_time_seconds": 0.1,
                    }
                else:
                    context_prompt = advanced_mode_result["content"]
                    if request.use_langgraph:
                        result = await process_prompt_with_langgraph_cancellable(
                            prompt=context_prompt,
                            prompt_type=request.prompt_type,
                            cancellation_event=cancellation_event,
                        )
                        coordinator._record_workflow(result)
                    else:
                        result = await coordinator.process_prompt(
                            prompt=context_prompt,
                            prompt_type=request.prompt_type,
                            return_comparison=request.return_comparison,
                        )
                    return result
            elif request.use_langgraph:
                result = await process_prompt_with_langgraph_cancellable(
                    prompt=context_prompt,
                    prompt_type=request.prompt_type,
                    cancellation_event=cancellation_event,
                )
                coordinator._record_workflow(result)
            else:
                result = await coordinator.process_prompt(
                    prompt=context_prompt,
                    prompt_type=request.prompt_type,
                    return_comparison=request.return_comparison,
                )
            return result

        def _post(result):
            if "metadata" not in result:
                result["metadata"] = {}
            result["metadata"]["original_prompt"] = request.prompt
            result["metadata"]["user_id"] = request.user_id
            result["metadata"]["has_context"] = bool(request.chat_history)

        background_tasks.add_task(
            _run_background_workflow,
            workflow_id, cancellation_event, request,
            _execute, "Memory workflow", _post,
        )

        return PromptResponse(
            workflow_id=workflow_id,
            status="running",
            message="Memory workflow started successfully.",
            timestamp=datetime.now().isoformat(),
            processing_time_seconds=0.0,
            input=request.model_dump(),
            output={},
            analysis={},
            metadata={},
        )

    except Exception as e:
        logger.error(f"Error starting memory workflow: {e}")
        active_workflows.pop(workflow_id, None)
        raise HTTPException(status_code=500, detail=f"Failed to start memory workflow: {str(e)}")


# ---------------------------------------------------------------------------
# POST /api/process-prompt/stream
# ---------------------------------------------------------------------------

@router.post("/api/process-prompt/stream")
async def process_prompt_stream(request: PromptRequest):
    """Stream prompt processing progress via Server-Sent Events."""
    if len(request.prompt) > MAX_PROMPT_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=f"Prompt exceeds maximum length of {MAX_PROMPT_LENGTH:,} characters.",
        )
    return StreamingResponse(
        stream_workflow(
            prompt=request.prompt,
            prompt_type=request.prompt_type,
            use_langgraph=request.use_langgraph,
            coordinator=coordinator,
            langgraph_fn=process_prompt_with_langgraph if request.use_langgraph else None,
            user_id=request.user_id,
        ),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ---------------------------------------------------------------------------
# POST /api/process-prompt/reiterate/stream
# ---------------------------------------------------------------------------

class ReiterateRequest(BaseModel):
    """Request model for re-iterating on an already-optimized prompt."""
    original_prompt: str
    optimized_prompt: str
    domain: str = "general"
    use_langgraph: bool = False
    user_feedback: Optional[str] = None
    user_id: Optional[str] = None


@router.post("/api/process-prompt/reiterate/stream")
async def reiterate_stream(request: ReiterateRequest):
    """Re-iterate on an optimized prompt via SSE."""
    from core.streaming import stream_reiterate

    return StreamingResponse(
        stream_reiterate(
            original_prompt=request.original_prompt,
            current_prompt=request.optimized_prompt,
            domain=request.domain,
            use_langgraph=request.use_langgraph,
            user_feedback=request.user_feedback,
            coordinator=coordinator,
            user_id=request.user_id,
        ),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
