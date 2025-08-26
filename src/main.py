"""Main application entry point for Multi-Agent Prompt Engineering System."""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
import uvicorn
import logging

import sys
import os
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

# Initialize FastAPI app
app = FastAPI(
    title="Multi-Agent Prompt Engineering System",
    description="A production-level system for improving and optimizing prompts using multiple AI agents",
    version="1.0.0"
)


# Pydantic models for API
class PromptRequest(BaseModel):
    """Request model for prompt processing."""
    prompt: str
    prompt_type: str = "auto"  # "auto", "raw", or "structured"
    return_comparison: bool = True
    use_langgraph: bool = False  # Whether to use LangGraph workflow


class PromptResponse(BaseModel):
    """Response model for prompt processing."""
    workflow_id: str
    status: str
    timestamp: str
    processing_time_seconds: Optional[float]
    input: Dict[str, Any]
    output: Dict[str, Any]
    analysis: Optional[Dict[str, Any]]
    comparison: Optional[Dict[str, Any]]
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


# API Routes
@app.post("/api/process-prompt", response_model=PromptResponse)
async def process_prompt(request: PromptRequest) -> PromptResponse:
    """Process a prompt through the multi-agent workflow."""
    try:
        logger.info(f"Processing prompt via API: {request.prompt[:100]}...")

        if request.use_langgraph:
            # Use LangGraph workflow
            result = await process_prompt_with_langgraph(
                prompt=request.prompt,
                prompt_type=request.prompt_type
            )
        else:
            # Use Coordinator workflow
            result = await coordinator.process_prompt(
                prompt=request.prompt,
                prompt_type=request.prompt_type,
                return_comparison=request.return_comparison
            )

        # Convert result to response format
        return PromptResponse(**result)

    except Exception as e:
        logger.error(f"Error processing prompt: {e}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


@app.get("/api/domains", response_model=List[DomainInfo])
async def get_available_domains():
    """Get information about all available domains."""
    try:
        domains = coordinator.get_available_domains()
        return [DomainInfo(**domain) for domain in domains]
    except Exception as e:
        logger.error(f"Error getting domains: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get domains: {str(e)}")


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


# Web Interface Routes
@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main web interface."""
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Multi-Agent Prompt Engineering System</title>
        <style>
            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
                background-color: #f5f5f5;
            }
            .container {
                background: white;
                padding: 30px;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                margin-bottom: 20px;
            }
            h1 {
                color: #333;
                text-align: center;
                margin-bottom: 30px;
            }
            .form-group {
                margin-bottom: 20px;
            }
            label {
                display: block;
                margin-bottom: 5px;
                font-weight: 600;
                color: #555;
            }
            textarea {
                width: 100%;
                min-height: 150px;
                padding: 12px;
                border: 2px solid #ddd;
                border-radius: 5px;
                font-family: 'Courier New', monospace;
                font-size: 14px;
                resize: vertical;
            }
            select, input[type="checkbox"] {
                padding: 10px;
                border: 2px solid #ddd;
                border-radius: 5px;
                font-size: 14px;
            }
            .checkbox-group {
                display: flex;
                align-items: center;
                gap: 10px;
            }
            button {
                background-color: #007bff;
                color: white;
                padding: 12px 30px;
                border: none;
                border-radius: 5px;
                cursor: pointer;
                font-size: 16px;
                font-weight: 600;
                transition: background-color 0.3s;
            }
            button:hover {
                background-color: #0056b3;
            }
            button:disabled {
                background-color: #6c757d;
                cursor: not-allowed;
            }
            .result {
                margin-top: 30px;
                padding: 20px;
                background-color: #f8f9fa;
                border-radius: 5px;
                border-left: 4px solid #28a745;
            }
            .error {
                border-left-color: #dc3545;
                background-color: #f8d7da;
            }
            .loading {
                text-align: center;
                padding: 20px;
            }
            .stats {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 20px;
                margin-top: 20px;
            }
            .stat-card {
                background: #f8f9fa;
                padding: 20px;
                border-radius: 5px;
                text-align: center;
            }
            .comparison {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 20px;
                margin-top: 20px;
            }
            .prompt-display {
                padding: 15px;
                background: #f8f9fa;
                border-radius: 5px;
                border: 1px solid #dee2e6;
            }
            .prompt-display h4 {
                margin-top: 0;
                color: #495057;
            }
            .code {
                font-family: 'Courier New', monospace;
                white-space: pre-wrap;
                word-wrap: break-word;
            }
            .hidden {
                display: none;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🚀 Multi-Agent Prompt Engineering System</h1>

            <form id="promptForm">
                <div class="form-group">
                    <label for="prompt">Enter your prompt:</label>
                    <textarea
                        id="prompt"
                        name="prompt"
                        placeholder="Enter a raw or structured prompt here...&#10;&#10;Examples:&#10;- Write a function to sort a list&#10;- Create a data analysis report&#10;- Draft a business strategy document"
                        required
                    ></textarea>
                </div>

                <div class="form-group">
                    <label for="promptType">Prompt Type:</label>
                    <select id="promptType" name="promptType">
                        <option value="auto">Auto-detect</option>
                        <option value="raw">Raw Prompt</option>
                        <option value="structured">Structured Prompt</option>
                    </select>
                </div>

                <div class="form-group">
                    <div class="checkbox-group">
                        <input type="checkbox" id="returnComparison" name="returnComparison" checked>
                        <label for="returnComparison">Return before/after comparison</label>
                    </div>
                </div>

                <div class="form-group">
                    <div class="checkbox-group">
                        <input type="checkbox" id="useLangGraph" name="useLangGraph">
                        <label for="useLangGraph">Use LangGraph workflow</label>
                    </div>
                </div>

                <button type="submit" id="submitBtn">Optimize Prompt</button>
            </form>

            <div id="loading" class="loading hidden">
                <h3>Processing your prompt with multiple AI agents...</h3>
                <p>This may take a few moments as our agents analyze, classify, improve, and evaluate your prompt.</p>
            </div>

            <div id="result" class="result hidden"></div>
        </div>

        <div class="container">
            <h2>📊 System Statistics</h2>
            <div id="stats" class="stats">
                <div class="stat-card">
                    <h3>Total Workflows</h3>
                    <div id="totalWorkflows">-</div>
                </div>
                <div class="stat-card">
                    <h3>Success Rate</h3>
                    <div id="successRate">-</div>
                </div>
                <div class="stat-card">
                    <h3>Avg Quality Score</h3>
                    <div id="avgQuality">-</div>
                </div>
                <div class="stat-card">
                    <h3>Avg Processing Time</h3>
                    <div id="avgTime">-</div>
                </div>
            </div>
        </div>

        <script>
            // Load initial stats
            loadStats();

            document.getElementById('promptForm').addEventListener('submit', async (e) => {
                e.preventDefault();

                const submitBtn = document.getElementById('submitBtn');
                const loading = document.getElementById('loading');
                const result = document.getElementById('result');

                const formData = new FormData(e.target);
                const requestData = {
                    prompt: formData.get('prompt'),
                    prompt_type: formData.get('promptType'),
                    return_comparison: formData.get('returnComparison') === 'on',
                    use_langgraph: formData.get('useLangGraph') === 'on'
                };

                // Show loading
                submitBtn.disabled = true;
                submitBtn.textContent = 'Processing...';
                loading.classList.remove('hidden');
                result.classList.add('hidden');

                try {
                    const response = await fetch('/api/process-prompt', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify(requestData)
                    });

                    const data = await response.json();

                    if (!response.ok) {
                        throw new Error(data.detail || 'Processing failed');
                    }

                    displayResult(data);

                } catch (error) {
                    displayError(error.message);
                } finally {
                    submitBtn.disabled = false;
                    submitBtn.textContent = 'Optimize Prompt';
                    loading.classList.add('hidden');
                    loadStats(); // Refresh stats
                }
            });

            function displayResult(data) {
                const result = document.getElementById('result');
                result.classList.remove('hidden', 'error');

                let html = `
                    <h3>✅ Prompt Optimization Complete</h3>
                    <p><strong>Workflow ID:</strong> ${data.workflow_id}</p>
                    <p><strong>Domain:</strong> ${data.output.domain}</p>
                    <p><strong>Quality Score:</strong> ${data.output.quality_score.toFixed(2)}/1.00</p>
                    <p><strong>Iterations:</strong> ${data.output.iterations_used}</p>
                    <p><strong>Processing Time:</strong> ${data.processing_time_seconds?.toFixed(2) || 'N/A'} seconds</p>
                `;

                if (data.comparison) {
                    html += `
                        <div class="comparison">
                            <div class="prompt-display">
                                <h4>📝 Original Prompt</h4>
                                <div class="code">${escapeHtml(data.comparison.side_by_side.original)}</div>
                            </div>
                            <div class="prompt-display">
                                <h4>✨ Optimized Prompt</h4>
                                <div class="code">${escapeHtml(data.comparison.side_by_side.optimized)}</div>
                            </div>
                        </div>
                        <p><strong>Improvement Ratio:</strong> ${(data.comparison.improvement_ratio * 100).toFixed(1)}%</p>
                    `;
                } else {
                    html += `
                        <div class="prompt-display">
                            <h4>✨ Optimized Prompt</h4>
                            <div class="code">${escapeHtml(data.output.optimized_prompt)}</div>
                        </div>
                    `;
                }

                if (data.analysis) {
                    html += `
                        <h4>🔍 Analysis</h4>
                        <p><strong>Classification:</strong> ${data.analysis.classification.reasoning}</p>
                        <p><strong>Key Topics:</strong> ${data.analysis.classification.key_topics.join(', ')}</p>
                    `;
                }

                result.innerHTML = html;
            }

            function displayError(message) {
                const result = document.getElementById('result');
                result.classList.remove('hidden');
                result.classList.add('error');

                result.innerHTML = `
                    <h3>❌ Processing Failed</h3>
                    <p><strong>Error:</strong> ${escapeHtml(message)}</p>
                    <p>Please check your input and try again.</p>
                `;
            }

            async function loadStats() {
                try {
                    const response = await fetch('/api/stats');
                    const stats = await response.json();

                    document.getElementById('totalWorkflows').textContent = stats.total_workflows;
                    document.getElementById('successRate').textContent = `${(stats.success_rate * 100).toFixed(1)}%`;
                    document.getElementById('avgQuality').textContent = stats.average_quality_score.toFixed(2);
                    document.getElementById('avgTime').textContent = `${stats.average_processing_time.toFixed(2)}s`;
                } catch (error) {
                    console.error('Failed to load stats:', error);
                }
            }

            function escapeHtml(text) {
                const div = document.createElement('div');
                div.textContent = text;
                return div.innerHTML;
            }
        </script>
    </body>
    </html>
    """


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
