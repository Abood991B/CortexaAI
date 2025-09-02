"""Main application entry point for Multi-Agent Prompt Engineering System."""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
import uvicorn
import logging
from datetime import datetime
import json
import uuid

import sys
import os
from pathlib import Path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import settings, setup_langsmith, metrics, get_logger
from agents.coordinator import WorkflowCoordinator
from agents.classifier import DomainClassifier
from agents.evaluator import PromptEvaluator
from src.workflow import process_prompt_with_langgraph
import psutil
import time

# Set up structured logging
logger = get_logger(__name__)

# Global instances for dependency injection
classifier_instance = DomainClassifier()
evaluator_instance = PromptEvaluator()
coordinator = WorkflowCoordinator(classifier_instance, evaluator_instance)

# In-memory store for active workflows
active_workflows: Dict[str, Any] = {}


# Initialize FastAPI app
app = FastAPI(
    title="Multi-Agent Prompt Engineering System",
    description="A production-level system for improving and optimizing prompts using multiple AI agents",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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


class DomainInfo(BaseModel):
    """Model for domain information."""
    domain: str
    description: str
    keywords: List[str]
    has_expert_agent: bool
    agent_created: bool


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


class AnalyticsData(BaseModel):
    """Model for analytics data."""
    total_prompts: int
    success_rate: float
    avg_processing_time: float
    domain_breakdown: Dict[str, int]
    quality_trends: List[Dict[str, Any]]
    performance_metrics: Dict[str, Any]


class PromptMetadata(BaseModel):
    """Model for prompt metadata."""
    id: str
    title: str
    content: str
    domain: str
    tags: List[str]
    created_at: str
    updated_at: str
    version: str
    metadata: Dict[str, Any]
    versions: List[Dict[str, Any]]


class CreatePromptData(BaseModel):
    """Model for creating new prompts."""
    title: str
    content: str
    domain: str
    tags: List[str] = []
    metadata: Dict[str, Any] = {}


class Template(BaseModel):
    """Model for prompt templates."""
    id: str
    name: str
    description: str
    content: str
    variables: List[str]
    category: str
    created_at: str


class ExperimentResult(BaseModel):
    """Model for A/B test experiment results."""
    id: str
    name: str
    description: str
    status: str
    variants: List[Dict[str, Any]]
    metrics: Dict[str, Any]
    created_at: str
    completed_at: Optional[str]


# API Routes
@app.post("/api/process-prompt", response_model=PromptResponse)
async def process_prompt(request: PromptRequest, background_tasks: BackgroundTasks) -> PromptResponse:
    """Process a prompt through the multi-agent workflow."""
    workflow_id = f"workflow_{uuid.uuid4().hex[:8]}"
    
    # Store the task in active_workflows
    active_workflows[workflow_id] = {
        "status": "running",
        "task": None,  # Placeholder for the background task
    }

    try:
        logger.info(f"Starting workflow {workflow_id} for prompt: {request.prompt[:100]}...")

        async def workflow_task():
            try:
                if request.use_langgraph:
                    result = await process_prompt_with_langgraph(
                        prompt=request.prompt,
                        prompt_type=request.prompt_type
                    )
                else:
                    result = await coordinator.process_prompt(
                        prompt=request.prompt,
                        prompt_type=request.prompt_type,
                        return_comparison=request.return_comparison
                    )
                active_workflows[workflow_id]["status"] = "completed"
                active_workflows[workflow_id]["result"] = result
            except Exception as e:
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
            input=request.dict(),
            output={},
            analysis={},
            metadata={}
        )

    except Exception as e:
        logger.error(f"Error starting workflow: {e}")
        active_workflows.pop(workflow_id, None)
        raise HTTPException(status_code=500, detail=f"Failed to start workflow: {str(e)}")



@app.get("/api/domains", response_model=List[DomainInfo])
async def get_available_domains():
    """Get information about all available domains."""
    try:
        domains = coordinator.get_available_domains()
        return [DomainInfo(**domain) for domain in domains]
    except Exception as e:
        logger.error(f"Error getting domains: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get domains: {str(e)}")


@app.post("/api/cancel-workflow/{workflow_id}")
async def cancel_workflow(workflow_id: str):
    """Cancel a running workflow."""
    if workflow_id not in active_workflows:
        raise HTTPException(status_code=404, detail="Workflow not found")

    if active_workflows[workflow_id]["status"] != "running":
        raise HTTPException(status_code=400, detail="Workflow is not running")

    # For now, we just mark it as cancelled. 
    # In a real-world scenario, you'd need a more robust way to stop the task.
    active_workflows[workflow_id]["status"] = "cancelled"
    logger.info(f"User cancelled workflow {workflow_id}")

    return {"status": "cancelled", "message": f"Workflow {workflow_id} has been cancelled."}


@app.get("/api/workflow-status/{workflow_id}")
async def get_workflow_status(workflow_id: str):
    """Get the status of a running or completed workflow."""
    if workflow_id not in active_workflows:
        raise HTTPException(status_code=404, detail="Workflow not found")

    workflow = active_workflows[workflow_id]
    status = workflow.get("status")

    if status == "completed":
        return {"status": "completed", "result": workflow.get("result")}
    elif status == "failed":
        return {"status": "failed", "error": workflow.get("error")}
    elif status == "cancelled":
        return {"status": "cancelled"}
    else:
        return {"status": "running"}



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
        # Mock data for now - replace with actual database queries
        workflows = []
        for i in range(min(limit, 10)):  # Generate some mock data
            workflow_id = f"workflow_{uuid.uuid4().hex[:8]}"
            workflows.append({
                "workflow_id": workflow_id,
                "status": "completed" if i % 3 == 0 else "running" if i % 3 == 1 else "failed",
                "domain": ["technology", "business", "creative", "academic"][i % 4],
                "prompt_preview": f"Sample prompt {i + 1} for testing the workflow system...",
                "created_at": datetime.now().isoformat(),
                "duration": 2.5 + i * 0.3,
                "total_steps": 3 + i % 3
            })
        
        return {
            "data": workflows,
            "total": 50,  # Mock total
            "page": page,
            "limit": limit,
            "pages": 3
        }
    except Exception as e:
        logger.error(f"Error getting workflows: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get workflows: {str(e)}")


@app.get("/api/workflows/{workflow_id}")
async def get_workflow_details(workflow_id: str):
    """Get detailed information about a specific workflow."""
    try:
        # Mock detailed workflow data
        workflow_details = {
            "workflow_id": workflow_id,
            "status": "completed",
            "domain": "technology",
            "prompt": "Create a comprehensive guide for implementing microservices architecture",
            "created_at": datetime.now().isoformat(),
            "completed_at": datetime.now().isoformat(),
            "duration": 4.2,
            "agent_steps": [
                {
                    "agent_type": "Classifier",
                    "processing_time": 0.8,
                    "output": "Classified as technology domain with high confidence"
                },
                {
                    "agent_type": "Expert",
                    "processing_time": 2.1,
                    "output": "Generated comprehensive microservices guide with best practices"
                },
                {
                    "agent_type": "Evaluator",
                    "processing_time": 1.3,
                    "output": "Quality score: 8.5/10, completeness: 95%"
                }
            ],
            "final_output": {
                "improved_prompt": "Create a comprehensive, production-ready guide for implementing microservices architecture including deployment strategies, monitoring, and security considerations",
                "quality_score": 8.5,
                "improvements": ["Added deployment strategies", "Included monitoring guidance", "Enhanced security considerations"]
            },
            "metadata": {
                "model_used": "gpt-4",
                "total_tokens": 2500,
                "cost_estimate": 0.05
            }
        }
        return workflow_details
    except Exception as e:
        logger.error(f"Error getting workflow details: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get workflow details: {str(e)}")


# Analytics API
@app.get("/api/analytics", response_model=AnalyticsData)
async def get_analytics():
    """Get system analytics data."""
    try:
        # Mock analytics data
        analytics = {
            "total_prompts": 1247,
            "success_rate": 87.3,
            "avg_processing_time": 3.2,
            "domain_breakdown": {
                "technology": 45,
                "business": 32,
                "creative": 15,
                "academic": 8
            },
            "quality_trends": [
                {"date": "2024-01-01", "avg_quality": 7.2},
                {"date": "2024-01-02", "avg_quality": 7.8},
                {"date": "2024-01-03", "avg_quality": 8.1},
                {"date": "2024-01-04", "avg_quality": 8.3},
                {"date": "2024-01-05", "avg_quality": 8.5}
            ],
            "performance_metrics": {
                "avg_response_time": 2.1,
                "error_rate": 2.3,
                "throughput": 150
            }
        }
        return AnalyticsData(**analytics)
    except Exception as e:
        logger.error(f"Error getting analytics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get analytics: {str(e)}")


# Prompts Management API
@app.get("/api/prompts")
async def get_prompts(
    page: int = 1,
    limit: int = 20,
    domain: Optional[str] = None,
    tags: Optional[str] = None
):
    """Get paginated list of prompts."""
    try:
        # Mock prompts data
        prompts = []
        for i in range(min(limit, 10)):
            prompt_id = f"prompt_{uuid.uuid4().hex[:8]}"
            prompts.append({
                "id": prompt_id,
                "title": f"Sample Prompt {i + 1}",
                "content": f"This is a sample prompt content for testing prompt {i + 1}...",
                "domain": ["technology", "business", "creative", "academic"][i % 4],
                "tags": ["optimization", "testing", "sample"],
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
                "version": "1.0",
                "metadata": {
                    "author": "system",
                    "quality_score": 7.5 + i * 0.2
                },
                "versions": [
                    {"version": "1.0", "created_at": datetime.now().isoformat()}
                ]
            })
        
        return {
            "data": prompts,
            "total": 85,
            "page": page,
            "limit": limit,
            "pages": 5
        }
    except Exception as e:
        logger.error(f"Error getting prompts: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get prompts: {str(e)}")


@app.post("/api/prompts")
async def create_prompt(prompt_data: CreatePromptData):
    """Create a new prompt."""
    try:
        prompt_id = f"prompt_{uuid.uuid4().hex[:8]}"
        new_prompt = {
            "id": prompt_id,
            "title": prompt_data.title,
            "content": prompt_data.content,
            "domain": prompt_data.domain,
            "tags": prompt_data.tags,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "version": "1.0",
            "metadata": prompt_data.metadata,
            "versions": [
                {"version": "1.0", "created_at": datetime.now().isoformat()}
            ]
        }
        return PromptMetadata(**new_prompt)
    except Exception as e:
        logger.error(f"Error creating prompt: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create prompt: {str(e)}")


# Templates API
@app.get("/api/templates", response_model=List[Template])
async def get_templates():
    """Get available prompt templates."""
    try:
        # Mock templates data
        templates = [
            {
                "id": "template_1",
                "name": "Business Analysis Template",
                "description": "Template for business analysis and strategy prompts",
                "content": "Analyze the following business scenario: {scenario}\nConsider these factors: {factors}\nProvide recommendations for: {objectives}",
                "variables": ["scenario", "factors", "objectives"],
                "category": "business",
                "created_at": datetime.now().isoformat()
            },
            {
                "id": "template_2",
                "name": "Technical Documentation Template",
                "description": "Template for technical documentation and guides",
                "content": "Create documentation for: {topic}\nTarget audience: {audience}\nInclude: {sections}",
                "variables": ["topic", "audience", "sections"],
                "category": "technology",
                "created_at": datetime.now().isoformat()
            }
        ]
        return [Template(**template) for template in templates]
    except Exception as e:
        logger.error(f"Error getting templates: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get templates: {str(e)}")


# Experiments API
@app.get("/api/experiments", response_model=List[ExperimentResult])
async def get_experiments():
    """Get A/B test experiments."""
    try:
        # Mock experiments data
        experiments = [
            {
                "id": "exp_1",
                "name": "Prompt Length Optimization",
                "description": "Testing different prompt lengths for better results",
                "status": "completed",
                "variants": [
                    {"name": "Short", "conversion_rate": 0.75},
                    {"name": "Medium", "conversion_rate": 0.82},
                    {"name": "Long", "conversion_rate": 0.78}
                ],
                "metrics": {
                    "total_samples": 1000,
                    "confidence_level": 0.95,
                    "winner": "Medium"
                },
                "created_at": datetime.now().isoformat(),
                "completed_at": datetime.now().isoformat()
            }
        ]
        return [ExperimentResult(**exp) for exp in experiments]
    except Exception as e:
        logger.error(f"Error getting experiments: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get experiments: {str(e)}")


# Memory Management API
@app.post("/api/process-prompt-with-memory", response_model=PromptResponse)
async def process_prompt_with_memory(request: PromptRequest):
    """Process prompt with memory context."""
    try:
        if not request.user_id:
            raise HTTPException(status_code=400, detail="User ID is required for memory-enhanced processing")
        
        logger.info(f"Processing prompt with memory for user {request.user_id}: {request.prompt[:100]}...")
        
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
        
        if request.use_langgraph:
            # Use LangGraph workflow with context
            result = await process_prompt_with_langgraph(
                prompt=context_prompt,
                prompt_type=request.prompt_type
            )
        else:
            # Use Coordinator workflow with context
            result = await coordinator.process_prompt(
                prompt=context_prompt,
                prompt_type=request.prompt_type,
                return_comparison=request.return_comparison
            )
        
        # Store the original prompt in metadata for frontend display
        if 'metadata' not in result:
            result['metadata'] = {}
        result['metadata']['original_prompt'] = request.prompt
        result['metadata']['user_id'] = request.user_id
        result['metadata']['has_context'] = bool(request.chat_history)
        
        return PromptResponse(**result)
    except Exception as e:
        logger.error(f"Error processing prompt with memory: {e}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


# Web Interface Routes
@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main web interface."""
    template_path = Path(__file__).parent.parent / "frontend" / "index.html"
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
    lines.append('system_info{version="1.0.0",langsmith_enabled="' + str(bool(settings.langsmith_api_key)).lower() + '"} 1')

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
        "version": "1.0.0",
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


def main():
    """Main entry point for running the application."""
    # Set up LangSmith if configured
    setup_langsmith()

    # Start the server
    logger.info("Starting Multi-Agent Prompt Engineering System...")
    logger.info(f"Server will run on {settings.host}:{settings.port}")

    uvicorn.run(
        "src.main:app",
        host=settings.host,
        port=settings.port,
        reload=True,
        log_level=settings.log_level.lower()
    )


if __name__ == "__main__":
    main()
