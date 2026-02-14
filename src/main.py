"""Main application entry point for CortexaAI - Advanced Multi-Agent Prompt Engineering System."""

import uuid
import asyncio
import sys
import os
import json
import logging
import uvicorn
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request, Query
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import settings, setup_langsmith, metrics, get_logger
from config.llm_providers import llm_provider, PROVIDER_CONFIGS
from core.optimization import optimization_engine
from agents.coordinator import WorkflowCoordinator
from agents.classifier import DomainClassifier
from agents.evaluator import PromptEvaluator
from src.workflow import process_prompt_with_langgraph
import psutil
from inspect import iscoroutinefunction

# ── New feature imports ──────────────────────────────────────────────────
from core.database import db
from core.templates import template_engine
from core.error_recovery import error_analytics
from core.language import language_processor
from core.complexity import complexity_analyzer
from core.auth import auth_manager
from core.marketplace import marketplace
from core.finetuning import finetuning_manager
from core.prompt_builder import prompt_builder
from core.plugins import plugin_manager
from core.regression import regression_runner
from core.similarity import similarity_engine
from core.streaming import stream_workflow
from core.batch import batch_processor
from core.webhooks import webhook_manager

# Set up structured logging
logger = get_logger(__name__)

# Global instances for dependency injection
classifier_instance = DomainClassifier()
evaluator_instance = PromptEvaluator()
coordinator = WorkflowCoordinator(classifier_instance, evaluator_instance)

# In-memory storage for active workflows with cancellation tokens
active_workflows: Dict[str, Dict[str, Any]] = {}

class WorkflowCancellationError(Exception):
    """Exception raised when a workflow is cancelled."""
    pass

async def process_prompt_with_langgraph_cancellable(prompt: str, prompt_type: str, cancellation_event: asyncio.Event):
    """Wrapper for LangGraph processing with cancellation support."""
    return await process_prompt_with_langgraph(
        prompt=prompt, 
        prompt_type=prompt_type, 
        cancellation_event=cancellation_event
    )

async def check_cancellation(cancellation_event: asyncio.Event, workflow_id: str):
    """Check if workflow should be cancelled and raise exception if so."""
    if cancellation_event.is_set():
        logger.info(f"Workflow {workflow_id} cancellation detected")
        raise WorkflowCancellationError(f"Workflow {workflow_id} was cancelled")


def clear_workflow_caches(prompt: str, prompt_type: str = None):
    """Clear all cache entries related to a workflow prompt."""
    from config.config import cache_manager, generate_prompt_cache_key, generate_evaluation_cache_key
    
    # Clear classification cache
    cache_key_classification = generate_prompt_cache_key(prompt, prefix="classification")
    cache_manager.delete(cache_key_classification)
    
    # Clear prompt type classification cache if applicable
    if prompt_type:
        cache_key_prompt_type = generate_prompt_cache_key(prompt, prefix="prompt_type_classification")
        cache_manager.delete(cache_key_prompt_type)
    
    # Clear caches for common domains (we can't know the exact domain without classifying)
    common_domains = [
        "software_engineering", "data_science", "report_writing", 
        "education", "business_strategy", "general"
    ]
    
    for domain in common_domains:
        # Clear improvement caches
        for pt in ["raw", "structured", "auto"]:
            cache_key_improvement = generate_prompt_cache_key(prompt, domain, pt)
            cache_manager.delete(cache_key_improvement)
            
            # Clear context-based improvement caches
            cache_key_improvement_context = generate_prompt_cache_key(prompt, domain, f"{pt}_context")
            cache_manager.delete(cache_key_improvement_context)
    
    logger.info(f"Cleared all cache entries for prompt: {prompt[:50]}...")


# Initialize FastAPI app
app = FastAPI(
    title="CortexaAI",
    description="Advanced Multi-Agent Prompt Engineering System with optimization, A/B testing, streaming, batch processing, marketplace, and multi-LLM support",
    version="3.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Startup: initialize database, seed templates, load plugins ───────────
@app.on_event("startup")
async def startup_event():
    """Initialize persistent layers on server start."""
    # Database is auto-initialized on import; ensure templates are seeded
    template_engine._seed_defaults()
    # Load plugins from plugins/ directory if it exists
    plugins_dir = Path(__file__).parent.parent / "plugins"
    if plugins_dir.exists():
        plugin_manager.load_from_directory(str(plugins_dir))


# Pydantic models for API
class PromptRequest(BaseModel):
    """Request model for prompt processing."""
    prompt: str
    prompt_type: str = "auto"  # "auto", "raw", or "structured"
    return_comparison: bool = True
    use_langgraph: bool = False  # Whether to use LangGraph workflow
    chat_history: Optional[List[Dict[str, str]]] = None  # For memory-enhanced processing
    user_id: Optional[str] = None  # User identifier for memory context
    workflow_id: Optional[str] = None  # Add workflow_id for tracking
    advanced_mode: bool = False
    synchronous: bool = False  # If true, run inline and return final result
    # New feature fields
    callback_url: Optional[str] = None  # Webhook URL to POST result on completion
    language: Optional[str] = None  # Force a language (auto-detected if omitted)


class PromptResponse(BaseModel):
    """Response model for prompt processing."""
    workflow_id: str
    status: str
    message: Optional[str] = None  # Add an optional message field

    timestamp: str
    processing_time_seconds: Optional[float]
    input: Dict[str, Any]
    output: Dict[str, Any]
    analysis: Optional[Dict[str, Any]]
    comparison: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any]


class SystemStats(BaseModel):
    """Model for system statistics."""
    total_workflows: int
    completed_workflows: int
    error_workflows: int
    success_rate: float
    average_quality_score: float
    average_processing_time: float
    domain_distribution: Dict[str, int]


class WorkflowSummary(BaseModel):
    """Model for workflow summary."""
    workflow_id: str
    status: str
    domain: str
    prompt_preview: str
    created_at: str
    duration: float
    total_steps: int


class WorkflowDetails(BaseModel):
    """Model for detailed workflow information."""
    workflow_id: str
    status: str
    domain: str
    prompt: str
    created_at: str
    completed_at: Optional[str]
    duration: float
    agent_steps: List[Dict[str, Any]]
    final_output: Dict[str, Any]
    metadata: Dict[str, Any]


# API Routes
@app.post("/api/process-prompt", response_model=PromptResponse)
async def process_prompt(request: PromptRequest, background_tasks: BackgroundTasks) -> PromptResponse:
    """Process a prompt through the multi-agent workflow.
    If request.synchronous is True, run inline and return final result.
    """
    from config.config import cache_manager, generate_prompt_cache_key, perf_config

    # Synchronous mode: run inline and return final
    if request.synchronous:
        logger.info(f"Processing prompt synchronously: {request.prompt[:100]}...")
        if request.use_langgraph:
            result = await process_prompt_with_langgraph(
                prompt=request.prompt,
                prompt_type=request.prompt_type,
            )
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
            input=request.dict(),
            output=result.get("output", {}),
            analysis=result.get("analysis", {}),
            comparison=result.get("comparison"),
            metadata=result.get("metadata", {}),
        )

        # Cache LangGraph results if enabled
        if request.use_langgraph and perf_config.enable_caching:
            cache_key = generate_prompt_cache_key(request.prompt, "langgraph_workflow")
            cache_manager.set(cache_key, response.dict(), perf_config.cache_ttl)

        return response

    # Default async/background behavior
    # Check cache first if caching is enabled
    if perf_config.enable_caching and request.use_langgraph:
        cache_key = generate_prompt_cache_key(request.prompt, "langgraph_workflow")
        cached_result = cache_manager.get(cache_key)
        if cached_result:
            logger.info(f"Returning cached result for LangGraph workflow: {request.prompt[:100]}...")
            return PromptResponse(**cached_result)

    workflow_id = f"workflow_{uuid.uuid4().hex[:8]}"
    
    # Store the task in active_workflows with cancellation token
    cancellation_event = asyncio.Event()
    active_workflows[workflow_id] = {
        "status": "running",
        "task": None,
        "cancellation_event": cancellation_event,
        "start_time": datetime.now(),
        "grace_period_active": True
    }

    try:
        logger.info(f"Starting workflow {workflow_id} for prompt: {request.prompt[:100]}...")

        async def workflow_task():
            try:
                # Wait for grace period (3 seconds) to allow cancellation
                try:
                    await asyncio.wait_for(cancellation_event.wait(), timeout=3.0)
                    # If we reach here, the workflow was cancelled during grace period
                    logger.info(f"Workflow {workflow_id} was cancelled during grace period")
                    active_workflows[workflow_id]["status"] = "cancelled"
                    
                    # Clear all cache entries for cancelled workflows to prevent returning cached cancelled results
                    clear_workflow_caches(request.prompt, request.prompt_type)
                    
                    return
                except asyncio.TimeoutError:
                    # Grace period expired, disable cancellation option
                    active_workflows[workflow_id]["grace_period_active"] = False
                    logger.info(f"Workflow {workflow_id} grace period expired, proceeding with execution")
                
                # Check if cancelled after grace period (shouldn't happen, but safety check)
                if cancellation_event.is_set():
                    logger.info(f"Workflow {workflow_id} was cancelled after grace period")
                    active_workflows[workflow_id]["status"] = "cancelled"
                    
                    # Clear cache entries for cancelled workflows
                    clear_workflow_caches(request.prompt, request.prompt_type)
                    
                    return
                
                # Execute the workflow with periodic cancellation checks
                if request.use_langgraph:
                    result = await process_prompt_with_langgraph_cancellable(
                        prompt=request.prompt,
                        prompt_type=request.prompt_type,
                        cancellation_event=cancellation_event
                    )
                else:
                    # Use regular coordinator but with periodic cancellation checks
                    result = await coordinator.process_prompt(
                        prompt=request.prompt,
                        prompt_type=request.prompt_type,
                        return_comparison=request.return_comparison
                    )
                
                # Final check if cancelled
                if cancellation_event.is_set():
                    logger.info(f"Workflow {workflow_id} was cancelled during execution")
                    active_workflows[workflow_id]["status"] = "cancelled"
                    
                    # Clear cache entries to prevent stale cancelled results
                    clear_workflow_caches(request.prompt, request.prompt_type)
                    
                    return
                
                active_workflows[workflow_id]["status"] = "completed"
                active_workflows[workflow_id]["result"] = result
                logger.info(f"Workflow {workflow_id} completed successfully")

                # Cache the result if caching is enabled
                if perf_config.enable_caching and request.use_langgraph:
                    cache_key = generate_prompt_cache_key(request.prompt, "langgraph_workflow")
                    # Prepare a cacheable response
                    cacheable_response = PromptResponse(
                        workflow_id=workflow_id,
                        status="completed",
                        message="Workflow completed successfully.",
                        timestamp=datetime.now().isoformat(),
                        processing_time_seconds=result.get("processing_time_seconds", 0),
                        input=request.dict(),
                        output=result.get("output", {}),
                        analysis=result.get("analysis", {}),
                        metadata=result.get("metadata", {})
                    )
                    cache_manager.set(cache_key, cacheable_response.dict(), perf_config.cache_ttl)
                
            except WorkflowCancellationError:
                logger.info(f"Workflow {workflow_id} was cancelled during execution")
                active_workflows[workflow_id]["status"] = "cancelled"
                
                # Clear cache entries
                clear_workflow_caches(request.prompt, request.prompt_type)
                
                return
            except Exception as e:
                # Check if this was due to cancellation
                if cancellation_event.is_set():
                    logger.info(f"Workflow {workflow_id} was cancelled during execution")
                    active_workflows[workflow_id]["status"] = "cancelled"
                    
                    # Clear cache entries
                    clear_workflow_caches(request.prompt, request.prompt_type)
                    
                    return
                
                logger.error(f"Workflow {workflow_id} failed: {e}")
                active_workflows[workflow_id]["status"] = "failed"
                active_workflows[workflow_id]["error"] = str(e)

        # Run the workflow in the background
        background_tasks.add_task(workflow_task)
        active_workflows[workflow_id]["task"] = background_tasks

        return PromptResponse(
            workflow_id=workflow_id,
            status="running",
            message="Workflow started successfully.",
            timestamp=datetime.now().isoformat(),
            processing_time_seconds=0.0,
            input=request.dict(),
            output={},
            analysis={},
            metadata={}
        )

    except Exception as e:
        logger.error(f"Error starting workflow: {e}")
        active_workflows.pop(workflow_id, None)
        raise HTTPException(status_code=500, detail=f"Failed to start workflow: {str(e)}")


@app.get("/api/domains")
async def get_domains():
    """Return available domains from the coordinator."""
    try:
        domains = coordinator.get_available_domains()
        return domains
    except Exception as e:
        logger.error(f"Error getting domains: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get domains: {str(e)}")


@app.post("/api/cancel-workflow/{workflow_id}")
async def cancel_workflow(workflow_id: str):
    """Cancel a running workflow."""
    if workflow_id not in active_workflows:
        raise HTTPException(status_code=404, detail="Workflow not found")

    workflow = active_workflows[workflow_id]
    if workflow["status"] != "running":
        raise HTTPException(status_code=400, detail="Workflow is not running")

    # Check if still in grace period
    if not workflow.get("grace_period_active", False):
        return {"status": "too_late", "message": f"Workflow {workflow_id} is past the cancellation grace period and cannot be cancelled."}

    # Trigger the cancellation event
    cancellation_event = workflow.get("cancellation_event")
    if cancellation_event:
        cancellation_event.set()
        logger.info(f"User cancelled workflow {workflow_id} during grace period")
        return {"status": "cancelled", "message": f"Workflow {workflow_id} has been cancelled."}
    else:
        # Fallback for older workflows without cancellation events
        workflow["status"] = "cancelled"
        logger.info(f"User cancelled workflow {workflow_id} (fallback method)")
        return {"status": "cancelled", "message": f"Workflow {workflow_id} has been cancelled."}


@app.get("/api/workflow-status/{workflow_id}")
async def get_workflow_status(workflow_id: str):
    """Get the status of a running or completed workflow."""
    if workflow_id not in active_workflows:
        raise HTTPException(status_code=404, detail="Workflow not found")

    workflow = active_workflows[workflow_id]
    status = workflow.get("status")

    response = {
        "workflow_id": workflow_id,
        "status": status,
        "grace_period_active": workflow.get("grace_period_active", False)
    }
    
    if status == "completed" and "result" in workflow:
        response["result"] = workflow["result"]
    elif status == "failed" and "error" in workflow:
        response["error"] = workflow["error"]
    
    return response


@app.get("/api/stats", response_model=SystemStats)
async def get_system_stats():
    """Get system statistics and workflow history."""
    try:
        stats = coordinator.get_workflow_stats()
        # Handle case where there's no workflow history yet
        if "error" in stats:
            return SystemStats(
                total_workflows=0,
                completed_workflows=0,
                error_workflows=0,
                success_rate=0.0,
                average_quality_score=0.0,
                average_processing_time=0.0,
                domain_distribution={}
            )
        return SystemStats(**stats)
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")


@app.get("/api/history", response_model=List[Dict[str, Any]])
async def get_workflow_history(limit: int = 10):
    """Get recent workflow history."""
    try:
        history = coordinator.get_workflow_history(limit=limit)
        return history
    except Exception as e:
        logger.error(f"Error getting history: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get history: {str(e)}")


# Workflows API
@app.get("/api/workflows")
async def get_workflows(
    page: int = 1,
    limit: int = 20,
    status: Optional[str] = None,
    domain: Optional[str] = None
):
    """Get paginated list of workflows."""
    try:
        # Get workflows from coordinator history
        all_workflows = coordinator.get_workflow_history(limit=1000)  # Get more for filtering
        
        # Filter workflows based on parameters
        filtered_workflows = []
        for workflow in all_workflows:
            # Apply status filter
            if status and workflow.get("status") != status:
                continue
            
            # Apply domain filter
            workflow_domain = workflow.get("output", {}).get("domain")
            if domain and workflow_domain != domain:
                continue
            
            # Transform workflow data for API response
            workflow_data = {
                "workflow_id": workflow.get("workflow_id"),
                "status": workflow.get("status"),
                "domain": workflow_domain,
                "prompt_preview": workflow.get("input", {}).get("original_prompt", "")[:100] + "..." if len(workflow.get("input", {}).get("original_prompt", "")) > 100 else workflow.get("input", {}).get("original_prompt", ""),
                "created_at": workflow.get("timestamp"),
                "duration": workflow.get("processing_time_seconds", 0),
                "total_steps": workflow.get("output", {}).get("iterations_used", 1),
                "quality_score": workflow.get("output", {}).get("quality_score", 0),
                "processing_time": workflow.get("processing_time_seconds", 0)
            }
            filtered_workflows.append(workflow_data)
        
        # Sort by timestamp (most recent first)
        filtered_workflows.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        
        # Implement pagination
        start_idx = (page - 1) * limit
        end_idx = start_idx + limit
        paginated_workflows = filtered_workflows[start_idx:end_idx]
        
        total_count = len(filtered_workflows)
        total_pages = (total_count + limit - 1) // limit
        
        return {
            "data": paginated_workflows,
            "total": total_count,
            "page": page,
            "limit": limit,
            "pages": total_pages
        }
    except Exception as e:
        logger.error(f"Error getting workflows: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get workflows: {str(e)}")
@app.get("/api/workflows/{workflow_id}")
async def get_workflow_details(workflow_id: str):
    """Get detailed information about a specific workflow."""
    try:
        # Find workflow in coordinator history
        all_workflows = coordinator.get_workflow_history(limit=1000)
        workflow = next((w for w in all_workflows if w.get("workflow_id") == workflow_id), None)
        
        if not workflow:
            raise HTTPException(status_code=404, detail="Workflow not found")
        
        # Transform workflow data for detailed view
        workflow_details = {
            "workflow_id": workflow.get("workflow_id"),
            "status": workflow.get("status"),
            "domain": workflow.get("output", {}).get("domain"),
            "original_prompt": workflow.get("input", {}).get("original_prompt"),
            "optimized_prompt": workflow.get("output", {}).get("optimized_prompt"),
            "created_at": workflow.get("timestamp"),
            "completed_at": workflow.get("timestamp"),  # For now, same as created
            "duration": workflow.get("processing_time_seconds", 0),
            "quality_score": workflow.get("output", {}).get("quality_score", 0),
            "iterations_used": workflow.get("output", {}).get("iterations_used", 1),
            "processing_time": workflow.get("processing_time_seconds", 0),
            "agent_steps": [
                {
                    "agent_type": "Classifier",
                    "processing_time": workflow.get("processing_time_seconds", 0) * 0.2,
                    "output": f"Classified as {workflow.get('output', {}).get('domain', 'unknown')} domain"
                },
                {
                    "agent_type": "Expert",
                    "processing_time": workflow.get("processing_time_seconds", 0) * 0.6,
                    "output": "Generated optimized prompt with domain-specific improvements"
                },
                {
                    "agent_type": "Evaluator",
                    "processing_time": workflow.get("processing_time_seconds", 0) * 0.2,
                    "output": f"Quality score: {workflow.get('output', {}).get('quality_score', 0):.2f}/10"
                }
            ],
            "analysis": workflow.get("analysis", {}),
            "metadata": workflow.get("metadata", {})
        }
        
        return workflow_details
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting workflow details: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get workflow details: {str(e)}")


# Memory Management API
@app.post("/api/process-prompt-with-memory", response_model=PromptResponse)
async def process_prompt_with_memory(request: PromptRequest, background_tasks: BackgroundTasks):
    """Process prompt with memory context using async workflow.
    If request.synchronous is True, run inline and return final result.
    """
    if not request.user_id:
        raise HTTPException(status_code=400, detail="User ID is required for memory-enhanced processing")

    # Synchronous mode: run inline and return final
    if request.synchronous:
        logger.info(f"Processing memory workflow synchronously for user {request.user_id}: {request.prompt[:100]}...")

        # Build context-aware prompt if chat history provided
        context_prompt = request.prompt
        if request.chat_history and len(request.chat_history) > 0:
            context_parts = ["Previous conversation context:"]
            for message in request.chat_history[-10:]:
                role = message.get('role', 'user')
                content = message.get('content', '')
                prefix = "User" if role == 'user' else "Assistant"
                context_parts.append(f"{prefix}: {content}")
            context_parts.append("\nCurrent request:")
            context_parts.append(f"User: {request.prompt}")
            context_parts.append("\nPlease provide a response that takes into account the previous conversation context and continues the discussion appropriately.")
            context_prompt = "\n".join(context_parts)

        if request.advanced_mode:
            advanced_mode_result = await coordinator.handle_advanced_mode(
                prompt=request.prompt,
                chat_history=request.chat_history,
            )
            if advanced_mode_result['status'] == 'needs_more_info':
                result = {
                    "output": {
                        "optimized_prompt": advanced_mode_result['content'],
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
                context_prompt = advanced_mode_result['content']
                if request.use_langgraph:
                    result = await process_prompt_with_langgraph(
                        prompt=context_prompt,
                        prompt_type=request.prompt_type,
                    )
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
        else:
            result = await coordinator.process_prompt(
                prompt=context_prompt,
                prompt_type=request.prompt_type,
                return_comparison=request.return_comparison,
            )

        # Attach original metadata
        if 'metadata' not in result:
            result['metadata'] = {}
        result['metadata']['original_prompt'] = request.prompt
        result['metadata']['user_id'] = request.user_id
        result['metadata']['has_context'] = bool(request.chat_history)

        return PromptResponse(
            workflow_id=result.get("workflow_id", f"workflow_{uuid.uuid4().hex[:8]}"),
            status=result.get("status", "completed"),
            message="Workflow completed successfully.",
            timestamp=datetime.now().isoformat(),
            processing_time_seconds=result.get("processing_time_seconds", 0.0),
            input=request.dict(),
            output=result.get("output", {}),
            analysis=result.get("analysis", {}),
            comparison=result.get("comparison"),
            metadata=result.get("metadata", {}),
        )

    # Default async/background behavior
    
    # Generate unique workflow ID
    workflow_id = f"workflow_{uuid.uuid4().hex[:8]}"
    
    # Store the task in active_workflows with cancellation token
    cancellation_event = asyncio.Event()
    active_workflows[workflow_id] = {
        "status": "running",
        "task": None,
        "cancellation_event": cancellation_event,
        "start_time": datetime.now(),
        "grace_period_active": True
    }

    try:
        logger.info(f"Starting memory workflow {workflow_id} for user {request.user_id}: {request.prompt[:100]}...")

        async def workflow_task():
            try:
                # Wait for grace period (3 seconds) to allow cancellation
                try:
                    await asyncio.wait_for(cancellation_event.wait(), timeout=3.0)
                    # If we reach here, the workflow was cancelled during grace period
                    logger.info(f"Memory workflow {workflow_id} was cancelled during grace period")
                    active_workflows[workflow_id]["status"] = "cancelled"
                    # Clear cache entries for cancelled workflows
                    clear_workflow_caches(request.prompt, request.prompt_type)
                    return
                except asyncio.TimeoutError:
                    # Grace period expired, disable cancellation option
                    active_workflows[workflow_id]["grace_period_active"] = False
                    logger.info(f"Memory workflow {workflow_id} grace period expired, proceeding with execution")
                
                # Check if cancelled after grace period (shouldn't happen, but safety check)
                if cancellation_event.is_set():
                    logger.info(f"Memory workflow {workflow_id} was cancelled after grace period")
                    active_workflows[workflow_id]["status"] = "cancelled"
                    # Clear cache entries for cancelled workflows
                    clear_workflow_caches(request.prompt, request.prompt_type)
                    return
                
                # Build context from chat history if provided
                context_prompt = request.prompt
                if request.chat_history and len(request.chat_history) > 0:
                    # Create a context-aware prompt that includes conversation history
                    context_parts = ["Previous conversation context:"]
                    
                    for message in request.chat_history[-10:]:  # Use last 10 messages for context
                        role = message.get('role', 'user')
                        content = message.get('content', '')
                        if role == 'user':
                            context_parts.append(f"User: {content}")
                        elif role == 'assistant':
                            context_parts.append(f"Assistant: {content}")
                    
                    context_parts.append("\nCurrent request:")
                    context_parts.append(f"User: {request.prompt}")
                    context_parts.append("\nPlease provide a response that takes into account the previous conversation context and continues the discussion appropriately.")
                    
                    context_prompt = "\n".join(context_parts)

                if request.advanced_mode:
                    advanced_mode_result = await coordinator.handle_advanced_mode(
                        prompt=request.prompt,
                        chat_history=request.chat_history
                    )
                    if advanced_mode_result['status'] == 'needs_more_info':
                        result = {
                            "output": {
                                "optimized_prompt": advanced_mode_result['content'],
                                "quality_score": 0,
                                "domain": "conversational",
                                "iterations_used": 1,
                            },
                            "analysis": {},
                            "metadata": {},
                            "processing_time_seconds": 0.1,
                        }
                    else:
                        context_prompt = advanced_mode_result['content']
                        if request.use_langgraph:
                            result = await process_prompt_with_langgraph_cancellable(
                                prompt=context_prompt,
                                prompt_type=request.prompt_type,
                                cancellation_event=cancellation_event
                            )
                        else:
                            result = await coordinator.process_prompt(
                                prompt=context_prompt,
                                prompt_type=request.prompt_type,
                                return_comparison=request.return_comparison
                            )
                elif request.use_langgraph:
                    result = await process_prompt_with_langgraph_cancellable(
                        prompt=context_prompt,
                        prompt_type=request.prompt_type,
                        cancellation_event=cancellation_event
                    )
                else:
                    result = await coordinator.process_prompt(
                        prompt=context_prompt,
                        prompt_type=request.prompt_type,
                        return_comparison=request.return_comparison
                    )
                
                # Final check if cancelled
                if cancellation_event.is_set():
                    logger.info(f"Memory workflow {workflow_id} was cancelled during execution")
                    active_workflows[workflow_id]["status"] = "cancelled"
                    # Clear cache entries for cancelled workflows
                    clear_workflow_caches(request.prompt, request.prompt_type)
                    return
                
                # Store the original prompt in metadata for frontend display
                if 'metadata' not in result:
                    result['metadata'] = {}
                result['metadata']['original_prompt'] = request.prompt
                result['metadata']['user_id'] = request.user_id
                result['metadata']['has_context'] = bool(request.chat_history)
                
                active_workflows[workflow_id]["status"] = "completed"
                active_workflows[workflow_id]["result"] = result
                logger.info(f"Memory workflow {workflow_id} completed successfully")
                
            except WorkflowCancellationError:
                logger.info(f"Memory workflow {workflow_id} was cancelled during execution")
                active_workflows[workflow_id]["status"] = "cancelled"
                # Clear cache entries for cancelled workflows
                clear_workflow_caches(request.prompt, request.prompt_type)
                return
            except Exception as e:
                # Check if this was due to cancellation
                if cancellation_event.is_set():
                    logger.info(f"Memory workflow {workflow_id} was cancelled during execution")
                    active_workflows[workflow_id]["status"] = "cancelled"
                    # Clear cache entries for cancelled workflows
                    clear_workflow_caches(request.prompt, request.prompt_type)
                    return
                
                logger.error(f"Memory workflow {workflow_id} failed: {e}")
                active_workflows[workflow_id]["status"] = "failed"
                active_workflows[workflow_id]["error"] = str(e)

        # Run the workflow in the background
        background_tasks.add_task(workflow_task)
        active_workflows[workflow_id]["task"] = background_tasks

        return PromptResponse(
            workflow_id=workflow_id,
            status="running",
            message="Memory workflow started successfully.",
            timestamp=datetime.now().isoformat(),
            processing_time_seconds=0.0,
            input=request.dict(),
            output={},
            analysis={},
            metadata={}
        )

    except Exception as e:
        logger.error(f"Error starting memory workflow: {e}")
        active_workflows.pop(workflow_id, None)
        raise HTTPException(status_code=500, detail=f"Failed to start memory workflow: {str(e)}")


# Web Interface Routes
@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main web interface."""
    template_path = Path(__file__).parent.parent / "frontend-react" / "index.html"
    return template_path.read_text(encoding="utf-8")


@app.get("/metrics")
async def get_metrics():
    """Prometheus-compatible metrics endpoint."""
    lines = []

    # Get current metrics
    current_metrics = metrics.get_metrics()

    # Add system metrics
    lines.append("# HELP system_info System information")
    lines.append("# TYPE system_info gauge")
    lines.append('system_info{version="3.0.0",langsmith_enabled="' + str(bool(settings.langsmith_api_key)).lower() + '"} 1')

    # Add LLM call metrics
    lines.append("# HELP llm_calls_total Total number of LLM calls")
    lines.append("# TYPE llm_calls_total counter")
    lines.append(f'llm_calls_total {current_metrics.get("llm_calls_total", 0)}')

    lines.append("# HELP llm_calls_success Successful LLM calls")
    lines.append("# TYPE llm_calls_success counter")
    lines.append(f'llm_calls_success {current_metrics.get("llm_calls_success", 0)}')

    lines.append("# HELP llm_calls_error Failed LLM calls")
    lines.append("# TYPE llm_calls_error counter")
    lines.append(f'llm_calls_error {current_metrics.get("llm_calls_error", 0)}')

    # Add workflow metrics
    lines.append("# HELP workflows_completed Completed workflows")
    lines.append("# TYPE workflows_completed counter")
    lines.append(f'workflows_completed {current_metrics.get("workflows_completed", 0)}')

    lines.append("# HELP workflows_failed Failed workflows")
    lines.append("# TYPE workflows_failed counter")
    lines.append(f'workflows_failed {current_metrics.get("workflows_failed", 0)}')

    # Add retry metrics
    lines.append("# HELP retry_attempts_total Total retry attempts")
    lines.append("# TYPE retry_attempts_total counter")
    lines.append(f'retry_attempts_total {current_metrics.get("retry_attempts", 0)}')

    # Add performance histograms
    durations = current_metrics.get("llm_call_duration_seconds", [])
    if durations:
        lines.append("# HELP llm_call_duration_seconds LLM call duration in seconds")
        lines.append("# TYPE llm_call_duration_seconds histogram")
        lines.append(f'llm_call_duration_seconds_count {len(durations)}')
        lines.append(f'llm_call_duration_seconds_sum {sum(durations)}')

        # Calculate buckets
        buckets = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, float('inf')]
        bucket_counts = [0] * len(buckets)
        for duration in durations:
            for i, bucket in enumerate(buckets):
                if duration <= bucket:
                    bucket_counts[i] += 1
                    break

        for i, bucket in enumerate(buckets):
            lines.append(f'llm_call_duration_seconds_bucket{{le="{bucket}"}} {bucket_counts[i]}')

    # Add domain distribution
    domains = current_metrics.get("domains_processed", {})
    for domain, count in domains.items():
        lines.append(f'# HELP domain_processed_total Total workflows processed for domain {domain}')
        lines.append("# TYPE domain_processed_total counter")
        lines.append(f'domain_processed_total{{domain="{domain}"}} {count}')

    # Add memory and system metrics
    try:
        process = psutil.Process()
        memory_info = process.memory_info()

        lines.append("# HELP process_memory_bytes Process memory usage in bytes")
        lines.append("# TYPE process_memory_bytes gauge")
        lines.append(f'process_memory_bytes {memory_info.rss}')

        lines.append("# HELP system_memory_percent System memory usage percentage")
        lines.append("# TYPE system_memory_percent gauge")
        lines.append(f'system_memory_percent {psutil.virtual_memory().percent}')

        lines.append("# HELP system_cpu_percent System CPU usage percentage")
        lines.append("# TYPE system_cpu_percent gauge")
        lines.append(f'system_cpu_percent {psutil.cpu_percent(interval=0.1)}')

    except ImportError:
        # psutil not available, skip system metrics
        pass

    return "\n".join(lines)


@app.get("/health")
async def health_check():
    """Enhanced health check endpoint with detailed system status."""
    health_status = {
        "status": "healthy",
        "timestamp": time.time(),
        "version": "3.0.0",
        "uptime_seconds": time.time() - getattr(health_check, 'start_time', time.time()),
        "components": {},
        "metrics": {}
    }

    # Store start time for uptime calculation
    if not hasattr(health_check, 'start_time'):
        health_check.start_time = time.time()

    # Check LLM provider connectivity
    health_status["components"]["llm_providers"] = {}
    for provider in ["openai", "anthropic", "google"]:
        api_key_attr = f"{provider}_api_key"
        has_key = getattr(settings, api_key_attr) is not None
        health_status["components"]["llm_providers"][provider] = {
            "configured": has_key,
            "status": "available" if has_key else "not_configured"
        }

    # Check LangSmith
    health_status["components"]["langsmith"] = {
        "enabled": bool(settings.langsmith_api_key),
        "status": "enabled" if settings.langsmith_api_key else "disabled"
    }

    # Check coordinator
    try:
        domains = coordinator.get_available_domains()
        health_status["components"]["coordinator"] = {
            "status": "healthy",
            "available_domains": len(domains)
        }
    except Exception as e:
        health_status["components"]["coordinator"] = {
            "status": "unhealthy",
            "error": str(e)
        }
        health_status["status"] = "unhealthy"

    # Add key metrics
    current_metrics = metrics.get_metrics()
    health_status["metrics"] = {
        "total_workflows": current_metrics.get("workflows_completed", 0) + current_metrics.get("workflows_failed", 0),
        "successful_workflows": current_metrics.get("workflows_completed", 0),
        "failed_workflows": current_metrics.get("workflows_failed", 0),
        "llm_calls_total": current_metrics.get("llm_calls_total", 0),
        "retry_attempts": current_metrics.get("retry_attempts", 0)
    }

    # Add system resource info
    try:
        health_status["system"] = {
            "memory_percent": psutil.virtual_memory().percent,
            "cpu_percent": psutil.cpu_percent(interval=0.1),
            "active_connections": len(psutil.net_connections()) if hasattr(psutil, 'net_connections') else 0
        }
    except ImportError:
        health_status["system"] = {
            "note": "System monitoring not available (psutil not installed)"
        }

    # Readiness and liveness probes
    health_status["readiness"] = health_status["status"] == "healthy"
    health_status["liveness"] = True  # Basic liveness check

    return health_status


# ---------------------------------------------------------------------------
# LLM Provider Management Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/providers")
async def get_providers():
    """Get status of all configured LLM providers."""
    return {
        "providers": llm_provider.get_provider_status(),
        "default_provider": settings.default_model_provider,
        "default_model": settings.default_model_name,
        "available": llm_provider.get_available_providers(),
    }


@app.post("/api/providers/{provider_name}/reset")
async def reset_provider_health(provider_name: str):
    """Reset health status for a provider (useful after fixing API keys)."""
    if provider_name not in PROVIDER_CONFIGS:
        raise HTTPException(status_code=404, detail=f"Unknown provider: {provider_name}")
    llm_provider.reset_health(provider_name)
    return {"status": "ok", "message": f"Health reset for provider: {provider_name}"}


# ---------------------------------------------------------------------------
# Optimization Engine Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/optimization/dashboard")
async def get_optimization_dashboard():
    """Get optimization engine dashboard data (analytics, A/B tests, versions)."""
    return optimization_engine.get_dashboard_data()


@app.get("/api/optimization/analytics")
async def get_optimization_analytics():
    """Get detailed optimization analytics and performance metrics."""
    return optimization_engine.analytics.get_summary()


@app.get("/api/optimization/ab-tests")
async def get_ab_tests():
    """Get A/B test history and statistics."""
    return {
        "stats": optimization_engine.ab_testing.get_stats(),
        "history": optimization_engine.ab_testing.get_test_history(),
    }


@app.get("/api/optimization/versions")
async def get_prompt_versions():
    """Get prompt version statistics."""
    return optimization_engine.version_control.get_stats()


# ═══════════════════════════════════════════════════════════════════════════
#  NEW FEATURE ENDPOINTS (v3.0)
# ═══════════════════════════════════════════════════════════════════════════


# ---------------------------------------------------------------------------
# Streaming (SSE)
# ---------------------------------------------------------------------------

@app.post("/api/process-prompt/stream")
async def process_prompt_stream(request: PromptRequest):
    """Stream prompt processing progress via Server-Sent Events."""
    return StreamingResponse(
        stream_workflow(
            prompt=request.prompt,
            prompt_type=request.prompt_type,
            use_langgraph=request.use_langgraph,
            coordinator=coordinator,
            langgraph_fn=process_prompt_with_langgraph if request.use_langgraph else None,
        ),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ---------------------------------------------------------------------------
# Batch Processing
# ---------------------------------------------------------------------------

class BatchRequest(BaseModel):
    prompts: List[Dict[str, Any]]
    concurrency: int = Field(default=3, ge=1, le=10)


@app.post("/api/batch")
async def create_batch(req: BatchRequest, background_tasks: BackgroundTasks):
    """Create and start a batch processing job."""
    job = batch_processor.create_batch(req.prompts, req.concurrency)
    batch_id = job["batch_id"]

    async def _processor(prompt_text, prompt_type="auto"):
        return await coordinator.process_prompt(prompt=prompt_text, prompt_type=prompt_type)

    background_tasks.add_task(batch_processor.run_batch, batch_id, _processor)
    return job


@app.get("/api/batch/{batch_id}/status")
async def get_batch_status(batch_id: str):
    status = batch_processor.get_status(batch_id)
    if not status:
        raise HTTPException(404, "Batch not found")
    return status


@app.get("/api/batch/{batch_id}/results")
async def get_batch_results(batch_id: str):
    results = batch_processor.get_results(batch_id)
    if not results:
        raise HTTPException(404, "Batch not found")
    return results


@app.get("/api/batches")
async def list_batches():
    return batch_processor.list_batches()


# ---------------------------------------------------------------------------
# Prompt Templates
# ---------------------------------------------------------------------------

class TemplateCreateRequest(BaseModel):
    name: str
    domain: str
    template_text: str
    description: str = ""
    variables: Optional[List[str]] = None


class TemplateRenderRequest(BaseModel):
    template_id: str
    variables: Dict[str, str] = {}


@app.get("/api/templates")
async def list_templates(domain: Optional[str] = None, query: Optional[str] = None):
    if query:
        return template_engine.search(query)
    return template_engine.list_all(domain)


@app.post("/api/templates")
async def create_template(req: TemplateCreateRequest):
    return template_engine.create(
        name=req.name, domain=req.domain,
        template_text=req.template_text,
        description=req.description,
        variables=req.variables,
    )


@app.get("/api/templates/{template_id}")
async def get_template(template_id: str):
    t = template_engine.get(template_id)
    if not t:
        raise HTTPException(404, "Template not found")
    return t


@app.post("/api/templates/render")
async def render_template(req: TemplateRenderRequest):
    result = template_engine.render(req.template_id, req.variables)
    if not result:
        raise HTTPException(404, "Template not found")
    return result


# ---------------------------------------------------------------------------
# Complexity Scoring
# ---------------------------------------------------------------------------

class ComplexityRequest(BaseModel):
    text: str


@app.post("/api/complexity")
async def analyze_complexity(req: ComplexityRequest):
    return complexity_analyzer.analyze(req.text)


@app.post("/api/complexity/pipeline-config")
async def get_pipeline_config(req: ComplexityRequest):
    return complexity_analyzer.get_pipeline_config(req.text)


# ---------------------------------------------------------------------------
# Language Detection & Processing
# ---------------------------------------------------------------------------

class LanguageRequest(BaseModel):
    text: str


@app.post("/api/language/detect")
async def detect_language(req: LanguageRequest):
    return language_processor.analyze(req.text)


@app.get("/api/language/supported")
async def supported_languages():
    return language_processor.get_supported_languages()


# ---------------------------------------------------------------------------
# API Key Auth Management
# ---------------------------------------------------------------------------

class APIKeyCreateRequest(BaseModel):
    name: str
    scopes: Optional[List[str]] = None
    rate_limit_rpm: Optional[int] = None


@app.post("/api/auth/keys")
async def create_api_key(req: APIKeyCreateRequest):
    return auth_manager.create_key(req.name, req.scopes, req.rate_limit_rpm)


@app.get("/api/auth/keys")
async def list_api_keys():
    return auth_manager.list_keys()


@app.delete("/api/auth/keys/{name}")
async def revoke_api_key(name: str):
    auth_manager.revoke_key(name)
    return {"status": "revoked", "name": name}


@app.post("/api/auth/verify")
async def verify_api_key(request: Request):
    key = request.headers.get("X-API-Key", "")
    result = auth_manager.verify_key(key)
    if not result:
        raise HTTPException(401, "Invalid or inactive API key")
    return result


# ---------------------------------------------------------------------------
# Marketplace
# ---------------------------------------------------------------------------

class MarketplacePublishRequest(BaseModel):
    title: str
    description: str
    prompt_text: str
    domain: str
    author: str = "anonymous"
    tags: Optional[List[str]] = None
    price: float = 0.0


class MarketplaceRateRequest(BaseModel):
    stars: int = Field(ge=1, le=5)


@app.get("/api/marketplace")
async def marketplace_search(
    query: Optional[str] = None,
    domain: Optional[str] = None,
    sort_by: str = "downloads",
    limit: int = 20,
    offset: int = 0,
):
    return marketplace.search(query, domain, sort_by, limit, offset)


@app.post("/api/marketplace")
async def marketplace_publish(req: MarketplacePublishRequest):
    return marketplace.publish(
        title=req.title, description=req.description,
        prompt_text=req.prompt_text, domain=req.domain,
        author=req.author, tags=req.tags, price=req.price,
    )


@app.get("/api/marketplace/{item_id}")
async def marketplace_download(item_id: str):
    item = marketplace.download(item_id)
    if not item:
        raise HTTPException(404, "Marketplace item not found")
    return item


@app.post("/api/marketplace/{item_id}/rate")
async def marketplace_rate(item_id: str, req: MarketplaceRateRequest):
    result = marketplace.rate(item_id, req.stars)
    if not result:
        raise HTTPException(404, "Marketplace item not found")
    return result


@app.get("/api/marketplace/stats/overview")
async def marketplace_stats():
    return marketplace.stats()


@app.get("/api/marketplace/featured/list")
async def marketplace_featured(limit: int = 10):
    return marketplace.featured(limit)


# ---------------------------------------------------------------------------
# Fine-Tuning
# ---------------------------------------------------------------------------

class FineTuneJobRequest(BaseModel):
    provider: str = "openai"
    model: str = "gpt-4o-mini-2024-07-18"
    hyperparameters: Optional[Dict[str, Any]] = None


class TrainingDataRequest(BaseModel):
    prompts: List[Dict[str, str]]
    format_type: str = "openai"


@app.post("/api/finetuning/prepare")
async def prepare_training_data(req: TrainingDataRequest):
    return finetuning_manager.prepare_training_data(req.prompts, req.format_type)


@app.post("/api/finetuning/jobs")
async def create_finetune_job(req: FineTuneJobRequest):
    return finetuning_manager.create_job(req.provider, req.model, hyperparameters=req.hyperparameters)


@app.get("/api/finetuning/jobs")
async def list_finetune_jobs():
    return finetuning_manager.list_jobs()


@app.get("/api/finetuning/jobs/{job_id}")
async def get_finetune_job(job_id: str):
    job = finetuning_manager.get_job(job_id)
    if not job:
        raise HTTPException(404, "Fine-tuning job not found")
    return job


@app.post("/api/finetuning/jobs/{job_id}/simulate")
async def simulate_finetune(job_id: str):
    result = finetuning_manager.simulate_run(job_id)
    if not result:
        raise HTTPException(404, "Fine-tuning job not found")
    return result


@app.get("/api/finetuning/models/{provider}")
async def get_finetune_models(provider: str):
    return finetuning_manager.get_supported_models(provider)


@app.get("/api/finetuning/estimate")
async def estimate_finetune_cost(provider: str = "openai", samples: int = 100, epochs: int = 3):
    return finetuning_manager.estimate_cost(provider, samples, epochs)


# ---------------------------------------------------------------------------
# Visual Prompt Builder
# ---------------------------------------------------------------------------

class BuilderBlockRequest(BaseModel):
    block_type: str
    content: str
    label: Optional[str] = None
    order: Optional[int] = None


class BuilderAssembleRequest(BaseModel):
    variables: Optional[Dict[str, str]] = None


class BuilderReorderRequest(BaseModel):
    block_ids: List[str]


@app.post("/api/builder/sessions")
async def create_builder_session(domain: Optional[str] = None):
    return prompt_builder.create_session(domain)


@app.get("/api/builder/sessions/{session_id}")
async def get_builder_session(session_id: str):
    s = prompt_builder.get_session(session_id)
    if not s:
        raise HTTPException(404, "Builder session not found")
    return s


@app.post("/api/builder/sessions/{session_id}/blocks")
async def add_builder_block(session_id: str, req: BuilderBlockRequest):
    result = prompt_builder.add_block(session_id, req.block_type, req.content, req.label, req.order)
    if not result:
        raise HTTPException(404, "Session not found")
    return result


@app.put("/api/builder/sessions/{session_id}/blocks/{block_id}")
async def update_builder_block(session_id: str, block_id: str, req: BuilderBlockRequest):
    result = prompt_builder.update_block(session_id, block_id, req.content)
    if not result:
        raise HTTPException(404, "Block or session not found")
    return result


@app.delete("/api/builder/sessions/{session_id}/blocks/{block_id}")
async def remove_builder_block(session_id: str, block_id: str):
    if not prompt_builder.remove_block(session_id, block_id):
        raise HTTPException(404, "Block or session not found")
    return {"status": "removed"}


@app.post("/api/builder/sessions/{session_id}/reorder")
async def reorder_builder_blocks(session_id: str, req: BuilderReorderRequest):
    result = prompt_builder.reorder_blocks(session_id, req.block_ids)
    if not result:
        raise HTTPException(404, "Session not found")
    return result


@app.post("/api/builder/sessions/{session_id}/assemble")
async def assemble_prompt(session_id: str, req: BuilderAssembleRequest):
    result = prompt_builder.assemble(session_id, req.variables)
    if not result:
        raise HTTPException(404, "Session not found")
    return result


@app.get("/api/builder/presets")
async def list_builder_presets():
    return prompt_builder.list_presets()


@app.get("/api/builder/presets/{domain}")
async def get_builder_preset(domain: str):
    preset = prompt_builder.get_preset(domain)
    if not preset:
        raise HTTPException(404, "Preset not found")
    return preset


# ---------------------------------------------------------------------------
# Plugin System
# ---------------------------------------------------------------------------

class PluginRegisterRequest(BaseModel):
    name: str
    version: str = "1.0.0"
    plugin_type: str = "expert"
    description: str = ""
    author: str = ""
    config: Optional[Dict[str, Any]] = None


@app.get("/api/plugins")
async def list_plugins():
    return plugin_manager.list_plugins()


@app.post("/api/plugins")
async def register_plugin(req: PluginRegisterRequest):
    return plugin_manager.register(
        req.name, req.version, req.plugin_type, req.description, req.author, req.config,
    )


@app.get("/api/plugins/{name}")
async def get_plugin(name: str):
    p = plugin_manager.get_plugin(name)
    if not p:
        raise HTTPException(404, "Plugin not found")
    return p


@app.delete("/api/plugins/{name}")
async def unregister_plugin(name: str):
    if not plugin_manager.unregister(name):
        raise HTTPException(404, "Plugin not found")
    return {"status": "removed"}


@app.post("/api/plugins/{name}/enable")
async def enable_plugin(name: str):
    if not plugin_manager.enable(name):
        raise HTTPException(404, "Plugin not found")
    return {"status": "enabled"}


@app.post("/api/plugins/{name}/disable")
async def disable_plugin(name: str):
    if not plugin_manager.disable(name):
        raise HTTPException(404, "Plugin not found")
    return {"status": "disabled"}


# ---------------------------------------------------------------------------
# Regression Testing
# ---------------------------------------------------------------------------

class RegressionSuiteRequest(BaseModel):
    name: str
    domain: str
    description: str = ""
    test_cases: Optional[List[Dict[str, Any]]] = None


class RegressionCaseRequest(BaseModel):
    input_prompt: str
    expected_keywords: Optional[List[str]] = None
    min_score: float = 0.7


@app.get("/api/regression/suites")
async def list_regression_suites():
    return regression_runner.list_suites()


@app.post("/api/regression/suites")
async def create_regression_suite(req: RegressionSuiteRequest):
    return regression_runner.create_suite(req.name, req.domain, req.description, req.test_cases)


@app.get("/api/regression/suites/{suite_id}")
async def get_regression_suite(suite_id: str):
    s = regression_runner.get_suite(suite_id)
    if not s:
        raise HTTPException(404, "Suite not found")
    return s


@app.delete("/api/regression/suites/{suite_id}")
async def delete_regression_suite(suite_id: str):
    regression_runner.delete_suite(suite_id)
    return {"status": "deleted"}


@app.post("/api/regression/suites/{suite_id}/cases")
async def add_regression_case(suite_id: str, req: RegressionCaseRequest):
    result = regression_runner.add_test_case(suite_id, req.input_prompt, req.expected_keywords, req.min_score)
    if not result:
        raise HTTPException(404, "Suite not found")
    return result


@app.post("/api/regression/suites/{suite_id}/run")
async def run_regression_suite(suite_id: str):
    async def _processor(prompt_text):
        result = await coordinator.process_prompt(prompt=prompt_text, prompt_type="auto")
        return {
            "improved_prompt": result.get("output", {}).get("optimized_prompt", ""),
            "evaluation_score": result.get("output", {}).get("quality_score", 0),
        }
    return await regression_runner.run_suite(suite_id, _processor)


@app.post("/api/regression/suites/{suite_id}/baseline")
async def save_regression_baseline(suite_id: str, run_results: Dict[str, Any]):
    regression_runner.save_baseline(suite_id, run_results)
    return {"status": "baseline_saved"}


# ---------------------------------------------------------------------------
# Similarity Search
# ---------------------------------------------------------------------------

class SimilaritySearchRequest(BaseModel):
    query: str
    top_k: int = Field(default=5, ge=1, le=50)
    domain: Optional[str] = None
    min_score: float = 0.0


class SimilarityIndexRequest(BaseModel):
    doc_id: str
    text: str
    domain: str = ""
    metadata: Optional[Dict[str, Any]] = None


@app.post("/api/similarity/search")
async def similarity_search(req: SimilaritySearchRequest):
    return similarity_engine.search(req.query, req.top_k, req.domain, req.min_score)


@app.post("/api/similarity/index")
async def similarity_index(req: SimilarityIndexRequest):
    return similarity_engine.add_document(req.doc_id, req.text, req.domain, req.metadata)


@app.get("/api/similarity/duplicates")
async def find_duplicates(threshold: float = 0.85):
    return similarity_engine.find_duplicates(threshold)


@app.post("/api/similarity/reindex")
async def reindex_similarity(limit: int = 500):
    count = similarity_engine.index_from_history(limit)
    return {"indexed": count, "corpus_size": similarity_engine.stats()["corpus_size"]}


@app.get("/api/similarity/stats")
async def similarity_stats():
    return similarity_engine.stats()


# ---------------------------------------------------------------------------
# Error Analytics
# ---------------------------------------------------------------------------

@app.get("/api/errors/analytics")
async def get_error_analytics():
    return error_analytics.get_summary()


@app.get("/api/errors/recent")
async def get_recent_errors(limit: int = 20):
    return error_analytics.get_recent(limit)


# ---------------------------------------------------------------------------
# Webhook Management
# ---------------------------------------------------------------------------

class WebhookSubscribeRequest(BaseModel):
    url: str
    events: Optional[List[str]] = None
    secret: Optional[str] = None
    name: str = ""


@app.post("/api/webhooks")
async def subscribe_webhook(req: WebhookSubscribeRequest):
    return webhook_manager.subscribe(req.url, req.events, req.secret, req.name)


@app.get("/api/webhooks")
async def list_webhooks():
    return webhook_manager.list_subscriptions()


@app.delete("/api/webhooks/{sub_id}")
async def unsubscribe_webhook(sub_id: str):
    if not webhook_manager.unsubscribe(sub_id):
        raise HTTPException(404, "Subscription not found")
    return {"status": "removed"}


@app.get("/api/webhooks/log")
async def webhook_delivery_log(limit: int = 50):
    return webhook_manager.get_delivery_log(limit)


# ---------------------------------------------------------------------------
# Database Dashboard
# ---------------------------------------------------------------------------

@app.get("/api/dashboard")
async def get_dashboard():
    """Aggregated dashboard combining DB stats, cache stats, and optimisation."""
    return {
        "database": db.get_dashboard_stats(),
        "cache": {
            "stats": metrics.get_metrics(),
        },
        "optimization": optimization_engine.get_dashboard_data(),
        "plugins": plugin_manager.list_plugins(),
        "similarity": similarity_engine.stats(),
        "marketplace": marketplace.stats(),
        "version": "3.0.0",
    }


# ---------------------------------------------------------------------------
# Serve React Frontend (production build)
# ---------------------------------------------------------------------------

# Try to serve React build if it exists
_frontend_build = Path(__file__).parent.parent / "frontend-react" / "dist"
if _frontend_build.exists():
    app.mount("/assets", StaticFiles(directory=str(_frontend_build / "assets")), name="assets")

    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        """Serve the React SPA for any non-API route."""
        # Don't intercept API routes
        if full_path.startswith("api/") or full_path in ("docs", "redoc", "openapi.json", "health", "metrics"):
            raise HTTPException(status_code=404)
        index_file = _frontend_build / "index.html"
        if index_file.exists():
            return HTMLResponse(index_file.read_text(encoding="utf-8"))
        raise HTTPException(status_code=404, detail="Frontend not built. Run: cd frontend-react && npm run build")


def main():
    """Main entry point for running the application."""
    # Set up LangSmith if configured
    setup_langsmith()

    # Start the server
    logger.info("Starting CortexaAI - Advanced Multi-Agent Prompt Engineering System")
    logger.info(f"Server: http://{settings.host}:{settings.port}")
    logger.info(f"API Docs: http://{settings.host}:{settings.port}/docs")
    logger.info(f"Providers: {', '.join(llm_provider.get_available_providers()) or 'None (check .env)'}")

    uvicorn.run(
        "src.main:app",
        host=settings.host,
        port=settings.port,
        reload=True,
        log_level=settings.log_level.lower()
    )


if __name__ == "__main__":
    main()
