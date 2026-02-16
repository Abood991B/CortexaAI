"""Workflow management endpoints: list, detail, cancel, status, history, stats, domains."""

from typing import Dict, List, Optional, Any

from fastapi import APIRouter, HTTPException

from src.deps import (
    logger,
    coordinator,
    active_workflows,
    clear_workflow_caches,
    db,
    SystemStats,
)

router = APIRouter()


# ---------------------------------------------------------------------------
# GET /api/domains
# ---------------------------------------------------------------------------

@router.get("/api/domains")
async def get_domains():
    """Return available domains from the coordinator."""
    try:
        return coordinator.get_available_domains()
    except Exception as e:
        logger.error(f"Error getting domains: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get domains: {str(e)}")


# ---------------------------------------------------------------------------
# POST /api/cancel-workflow/{workflow_id}
# ---------------------------------------------------------------------------

@router.post("/api/cancel-workflow/{workflow_id}")
async def cancel_workflow(workflow_id: str):
    """Cancel a running workflow."""
    if workflow_id not in active_workflows:
        raise HTTPException(status_code=404, detail="Workflow not found")

    workflow = active_workflows[workflow_id]
    if workflow["status"] != "running":
        raise HTTPException(status_code=400, detail="Workflow is not running")

    cancellation_event = workflow.get("cancellation_event")
    if cancellation_event:
        cancellation_event.set()
    workflow["status"] = "cancelled"
    logger.info(f"User cancelled workflow {workflow_id}")
    return {"status": "cancelled", "message": f"Workflow {workflow_id} has been cancelled."}


# ---------------------------------------------------------------------------
# GET /api/workflow-status/{workflow_id}
# ---------------------------------------------------------------------------

@router.get("/api/workflow-status/{workflow_id}")
async def get_workflow_status(workflow_id: str):
    """Get the status of a running or completed workflow."""
    if workflow_id not in active_workflows:
        # Check DB for completed workflows
        db_workflow = db.get_workflow(workflow_id)
        if db_workflow:
            status = db_workflow.get("status", "completed")
            resp: Dict[str, Any] = {
                "workflow_id": workflow_id,
                "status": status,
            }
            if status == "completed":
                resp["result"] = db_workflow.get("result", db_workflow.get("output"))
            elif status == "failed":
                resp["error"] = db_workflow.get("error")
            return resp
        raise HTTPException(status_code=404, detail="Workflow not found")

    workflow = active_workflows[workflow_id]
    status = workflow.get("status")
    response: Dict[str, Any] = {"workflow_id": workflow_id, "status": status}

    if status == "completed" and "result" in workflow:
        response["result"] = workflow["result"]
    elif status == "failed" and "error" in workflow:
        response["error"] = workflow["error"]

    return response


# ---------------------------------------------------------------------------
# GET /api/stats
# ---------------------------------------------------------------------------

@router.get("/api/stats", response_model=SystemStats)
async def get_system_stats(user_id: Optional[str] = None):
    """Get system statistics from persistent storage + in-memory with optional user filtering."""
    try:
        db_stats = db.get_dashboard_stats(user_id=user_id)
        if db_stats and db_stats.get("total_workflows", 0) > 0:
            return SystemStats(**db_stats)
        # If filtering by user_id, return empty stats instead of global fallback
        if user_id:
            return SystemStats(
                total_workflows=0,
                completed_workflows=0,
                error_workflows=0,
                success_rate=0.0,
                average_quality_score=0.0,
                average_processing_time=0.0,
                domain_distribution={},
            )
        stats = coordinator.get_workflow_stats()
        if "error" in stats:
            return SystemStats(
                total_workflows=0,
                completed_workflows=0,
                error_workflows=0,
                success_rate=0.0,
                average_quality_score=0.0,
                average_processing_time=0.0,
                domain_distribution={},
            )
        return SystemStats(**stats)
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")


# ---------------------------------------------------------------------------
# GET /api/history
# ---------------------------------------------------------------------------

@router.get("/api/history", response_model=List[Dict[str, Any]])
async def get_workflow_history(limit: int = 10, user_id: Optional[str] = None):
    """Get recent workflow history from persistent storage with optional user filtering."""
    if limit < 1 or limit > 500:
        raise HTTPException(status_code=400, detail="Limit must be between 1 and 500")
    try:
        db_workflows = db.get_workflows(limit=limit, user_id=user_id)
        if db_workflows:
            return _normalise_history(db_workflows)
        # If filtering by user_id, return empty list instead of global fallback
        if user_id:
            return []
        history = coordinator.get_workflow_history(limit=limit)
        return _normalise_history(history)
    except Exception as e:
        logger.error(f"Error getting history: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get history: {str(e)}")


def _normalise_history(items: list) -> List[Dict[str, Any]]:
    normalised: List[Dict[str, Any]] = []
    for item in items:
        original_prompt = item.get("input", {}).get("original_prompt", "")
        preview = (original_prompt[:100] + "...") if len(original_prompt) > 100 else original_prompt
        normalised.append({
            "workflow_id": item.get("workflow_id"),
            "status": item.get("status"),
            "prompt_preview": preview,
            "domain": item.get("output", {}).get("domain"),
            "quality_score": item.get("output", {}).get("quality_score"),
            "processing_time": item.get("processing_time_seconds", item.get("processing_time", 0)),
            "timestamp": item.get("timestamp"),
        })
    return normalised


# ---------------------------------------------------------------------------
# GET /api/workflows  (paginated, DB-level filtering)
# ---------------------------------------------------------------------------

@router.get("/api/workflows")
async def get_workflows(
    page: int = 1,
    limit: int = 20,
    status: Optional[str] = None,
    domain: Optional[str] = None,
):
    """Get paginated list of workflows using DB-level filtering."""
    if page < 1:
        raise HTTPException(status_code=400, detail="Page must be >= 1")
    if limit < 1 or limit > 100:
        raise HTTPException(status_code=400, detail="Limit must be between 1 and 100")
    try:
        offset = (page - 1) * limit
        db_workflows = db.get_workflows(limit=limit, offset=offset, status=status, domain=domain)
        total_count = db.count_workflows(status=status, domain=domain)
        total_pages = (total_count + limit - 1) // limit if total_count > 0 else 0

        paginated = []
        for wf in db_workflows:
            original_prompt = wf.get("input", {}).get("original_prompt", "")
            preview = (original_prompt[:100] + "...") if len(original_prompt) > 100 else original_prompt
            paginated.append({
                "workflow_id": wf.get("workflow_id"),
                "status": wf.get("status"),
                "domain": wf.get("output", {}).get("domain"),
                "prompt_preview": preview,
                "created_at": wf.get("timestamp"),
                "duration": wf.get("processing_time_seconds", 0),
                "total_steps": wf.get("output", {}).get("iterations_used", 1),
                "quality_score": wf.get("output", {}).get("quality_score", 0),
                "processing_time": wf.get("processing_time_seconds", 0),
            })

        return {
            "data": paginated,
            "total": total_count,
            "page": page,
            "limit": limit,
            "pages": total_pages,
        }
    except Exception as e:
        logger.error(f"Error getting workflows: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get workflows: {str(e)}")


# ---------------------------------------------------------------------------
# GET /api/workflows/{workflow_id}
# ---------------------------------------------------------------------------

@router.get("/api/workflows/{workflow_id}")
async def get_workflow_details(workflow_id: str):
    """Get detailed information about a specific workflow."""
    try:
        workflow = db.get_workflow(workflow_id)
        if not workflow:
            raise HTTPException(status_code=404, detail="Workflow not found")

        return {
            "workflow_id": workflow.get("workflow_id"),
            "status": workflow.get("status"),
            "domain": workflow.get("output", {}).get("domain"),
            "original_prompt": workflow.get("input", {}).get("original_prompt"),
            "optimized_prompt": workflow.get("output", {}).get("optimized_prompt"),
            "created_at": workflow.get("timestamp"),
            "completed_at": workflow.get("timestamp"),
            "duration": workflow.get("processing_time_seconds", 0),
            "quality_score": workflow.get("output", {}).get("quality_score", 0),
            "iterations_used": workflow.get("output", {}).get("iterations_used", 1),
            "processing_time": workflow.get("processing_time_seconds", 0),
            "analysis": workflow.get("analysis", {}),
            "metadata": workflow.get("metadata", {}),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting workflow details: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get workflow details: {str(e)}")
