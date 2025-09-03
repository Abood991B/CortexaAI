"""LangGraph workflow implementation for the Multi-Agent Prompt Engineering System."""

from typing import Dict, List, Optional, Any, Annotated, TypedDict
from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableConfig
import logging
import time
import asyncio
from functools import wraps

from agents.classifier import classifier
from agents.base_expert import create_expert_agent
from agents.evaluator import evaluator
from config.config import settings

# Set up logging
logging.basicConfig(level=settings.log_level)
logger = logging.getLogger(__name__)


class WorkflowState(TypedDict):
    """State object for the LangGraph workflow."""
    # Input
    original_prompt: str
    prompt_type: str

    # Classification results
    domain: Optional[str]
    classification_result: Optional[Dict[str, Any]]

    # Expert agent
    expert_agent: Optional[Any]  # Expert agent instance

    # Improvement results
    improved_prompt: Optional[str]
    improvement_result: Optional[Dict[str, Any]]

    # Evaluation results
    evaluation_result: Optional[Dict[str, Any]]
    iterations_used: int
    passes_threshold: bool

    # Final output
    final_prompt: Optional[str]

    # Metadata
    workflow_id: str
    status: str
    error_message: Optional[str]
    next_action: Optional[str]  # For conditional routing
    cancellation_event: Optional[asyncio.Event]


def cancellable_node(node_func):
    """Decorator to make a workflow node check for cancellation before running."""
    @wraps(node_func)
    async def wrapper(state: WorkflowState, config: RunnableConfig) -> Dict[str, Any]:
        cancellation_event = state.get("cancellation_event")
        if cancellation_event and cancellation_event.is_set():
            logger.info(f"Cancellation detected in node '{node_func.__name__}'. Halting workflow.")
            return {"status": "cancelled"}
        return await node_func(state, config)
    return wrapper

@cancellable_node
async def classify_node(state: WorkflowState, config: RunnableConfig) -> Dict[str, Any]:
    """Node for domain classification."""
    try:
        logger.info("Executing classification node")

        original_prompt = state["original_prompt"]

        # Classify the prompt (await the async function)
        classification_result = await classifier.classify_prompt(original_prompt)

        return {
            "domain": classification_result["domain"],
            "classification_result": classification_result,
            "status": "classification_completed"
        }

    except Exception as e:
        logger.error(f"Error in classification node: {e}")
        return {
            "domain": "general",
            "classification_result": {
                "domain": "general",
                "confidence": 0.5,
                "key_topics": [],
                "reasoning": f"Classification failed: {str(e)}"
            },
            "status": "classification_completed"
        }


@cancellable_node
async def create_expert_node(state: WorkflowState, config: RunnableConfig) -> Dict[str, Any]:
    """Node for creating/selecting expert agent."""
    try:
        logger.info("Executing expert creation node")

        domain = state.get("domain")
        classification_result = state.get("classification_result", {})

        if not domain:
            # Create generic expert agent
            expert_agent = create_expert_agent("general", "General prompt optimization")
            return {
                "expert_agent": expert_agent,
                "status": "expert_created"
            }

        # Get domain description from classification
        domain_description = classification_result.get("reasoning", f"Expert in {domain}")

        # Create expert agent
        expert_agent = create_expert_agent(domain, domain_description)

        return {
            "expert_agent": expert_agent,
            "status": "expert_created"
        }

    except Exception as e:
        logger.error(f"Error in expert creation node: {e}")
        # Create generic expert agent on error
        expert_agent = create_expert_agent("general", "General prompt optimization")
        return {
            "expert_agent": expert_agent,
            "status": "expert_created"
        }


@cancellable_node
async def improve_prompt_node(state: WorkflowState, config: RunnableConfig) -> Dict[str, Any]:
    """Node for prompt improvement."""
    try:
        logger.info("Executing prompt improvement node")

        original_prompt = state["original_prompt"]
        prompt_type = state.get("prompt_type", "raw")
        classification_result = state.get("classification_result", {})
        key_topics = classification_result.get("key_topics", [])

        # Get expert agent from state
        expert_agent = state.get("expert_agent")

        if not expert_agent:
            # Create a generic expert agent as fallback
            expert_agent = create_expert_agent("general", "General prompt optimization")

        # Improve the prompt (await the async function)
        improvement_result = await expert_agent.improve_prompt(
            original_prompt=original_prompt,
            prompt_type=prompt_type,
            key_topics=key_topics
        )

        improved_prompt = improvement_result.get("improved_prompt", original_prompt)

        return {
            "improved_prompt": improved_prompt,
            "improvement_result": improvement_result,
            "final_prompt": improved_prompt,  # Set as final for now
            "status": "improvement_completed"
        }

    except Exception as e:
        logger.error(f"Error in improvement node: {e}")
        original_prompt = state["original_prompt"]
        return {
            "improved_prompt": original_prompt,  # Return original on error
            "improvement_result": {
                "improved_prompt": original_prompt,
                "improvements_made": [],
                "structure_analysis": f"Improvement failed: {str(e)}",
                "effectiveness_score": 0.5
            },
            "final_prompt": original_prompt,
            "status": "improvement_completed"
        }


@cancellable_node
async def evaluate_node(state: WorkflowState, config: RunnableConfig) -> Dict[str, Any]:
    """Node for prompt evaluation."""
    try:
        logger.info("Executing evaluation node")

        original_prompt = state["original_prompt"]
        improved_prompt = state.get("improved_prompt", original_prompt)
        domain = state.get("domain", "general")
        prompt_type = state.get("prompt_type", "raw")
        current_iterations = state.get("iterations_used", 0)

        # Get expert agent from state
        expert_agent = state.get("expert_agent")

        if not expert_agent:
            # Create a generic expert agent as fallback
            expert_agent = create_expert_agent("general", "General prompt optimization")

        # Run evaluation loop (await the async function)
        evaluation_result, iterations_used = await evaluator.run_evaluation_loop(
            original_prompt=original_prompt,
            improved_prompt=improved_prompt,
            domain=domain,
            expert_agent=expert_agent,
            prompt_type=prompt_type
        )

        passes_threshold = evaluation_result.get("passes_threshold", False)

        return {
            "evaluation_result": evaluation_result,
            "iterations_used": current_iterations + iterations_used,
            "passes_threshold": passes_threshold,
            "status": "evaluation_completed"
        }

    except Exception as e:
        logger.error(f"Error in evaluation node: {e}")
        original_prompt = state["original_prompt"]
        return {
            "evaluation_result": {
                "overall_score": 0.5,
                "passes_threshold": False,
                "reasoning": f"Evaluation failed: {str(e)}"
            },
            "iterations_used": state.get("iterations_used", 0),
            "passes_threshold": False,
            "status": "evaluation_completed"
        }


@cancellable_node
async def check_threshold_node(state: WorkflowState, config: RunnableConfig) -> Dict[str, Any]:
    """Node to check if the prompt passes the evaluation threshold."""
    passes_threshold = state.get("passes_threshold", False)
    iterations_used = state.get("iterations_used", 0)
    status = state.get("status", "")

    # Update the state to indicate the next action
    if status == "error":
        state["next_action"] = "error"
        return {"next_action": "error"}

    # If we pass the threshold or have used max iterations, end
    if passes_threshold or iterations_used >= settings.max_evaluation_iterations:
        state["next_action"] = "end"
        return {"next_action": "end"}

    # Otherwise, continue with another improvement iteration
    state["next_action"] = "improve_again"
    return {"next_action": "improve_again"}


@cancellable_node
async def finalize_node(state: WorkflowState, config: RunnableConfig) -> Dict[str, Any]:
    """Final node to prepare the final result."""
    final_prompt = state.get("final_prompt", state.get("original_prompt", ""))
    evaluation_result = state.get("evaluation_result", {})
    domain = state.get("domain", "unknown")

    return {
        "final_prompt": final_prompt,
        "final_evaluation": evaluation_result,
        "final_domain": domain,
        "status": "workflow_completed"
    }


@cancellable_node
async def error_handler_node(state: WorkflowState, config: RunnableConfig) -> Dict[str, Any]:
    """Node for handling errors in the workflow."""
    error_message = state.get("error_message", "Unknown error occurred")

    logger.error(f"Workflow error: {error_message}")

    return {
        "status": "error",
        "error_message": error_message,
        "final_prompt": state.get("original_prompt")  # Return original prompt on error
    }


def create_workflow_graph() -> StateGraph:
    """Create the LangGraph workflow for prompt engineering."""
    # Initialize the state graph
    workflow = StateGraph(WorkflowState)

    # Add nodes
    workflow.add_node("classify", classify_node)
    workflow.add_node("create_expert", create_expert_node)
    workflow.add_node("improve", improve_prompt_node)
    workflow.add_node("evaluate", evaluate_node)
    workflow.add_node("check_threshold", check_threshold_node)
    workflow.add_node("finalize", finalize_node)
    workflow.add_node("error_handler", error_handler_node)

    # Define the workflow edges
    workflow.set_entry_point("classify")

    # Classification -> Expert Creation -> Initial Improvement
    workflow.add_edge("classify", "create_expert")
    workflow.add_edge("create_expert", "improve")

    # Improvement -> Evaluation -> Threshold Check
    workflow.add_edge("improve", "evaluate")
    workflow.add_edge("evaluate", "check_threshold")

    # Conditional edges from threshold check with proper routing
    def route_based_on_threshold(state: WorkflowState) -> str:
        """Route based on evaluation results and iteration count."""
        next_action = state.get("next_action", "end")
        iterations_used = state.get("iterations_used", 0)

        # Safety check: prevent infinite loops (hard limit)
        if iterations_used >= settings.max_evaluation_iterations:
            return "finalize"

        if next_action == "improve_again":
            return "improve"
        elif next_action == "error":
            return "error_handler"
        else:
            return "finalize"

    workflow.add_conditional_edges(
        "check_threshold",
        route_based_on_threshold,
        {
            "improve": "improve",        # Continue improving
            "finalize": "finalize",      # End workflow successfully
            "error_handler": "error_handler"  # Handle errors
        }
    )

    # Final edges
    workflow.add_edge("finalize", END)
    workflow.add_edge("error_handler", END)

    return workflow


def create_prompt_engineering_app():
    """Create the compiled LangGraph application."""
    workflow_graph = create_workflow_graph()

    # Compile the graph
    app = workflow_graph.compile()

    return app


# Global app instance
prompt_engineering_app = create_prompt_engineering_app()


async def process_prompt_with_langgraph(
    prompt: str, 
    prompt_type: str = "auto", 
    cancellation_event: Optional[asyncio.Event] = None
) -> Dict[str, Any]:
    """
    Process a prompt using the LangGraph workflow.

    Args:
        prompt: The input prompt to process
        prompt_type: Type of prompt ("auto", "raw", or "structured")

    Returns:
        Dict containing the workflow results
    """
    try:
        # Prepare initial state
        initial_state = {
            "original_prompt": prompt,
            "prompt_type": prompt_type,
            "iterations_used": 0,
            "status": "started",
            "workflow_id": f"lg_workflow_{int(__import__('time').time())}",
            "cancellation_event": cancellation_event
        }

        # Configure the run
        config = RunnableConfig(
            configurable={
                "thread_id": initial_state["workflow_id"]
            }
        )

        # Execute the workflow (use ainvoke for async nodes)
        logger.info(f"Starting LangGraph workflow for prompt processing")
        start_time = time.time()
        result = await prompt_engineering_app.ainvoke(initial_state, config)

        # Convert result to expected format
        processing_time = time.time() - start_time

        return {
            "workflow_id": result.get("workflow_id"),
            "status": result.get("status"),
            "timestamp": __import__('datetime').datetime.now().isoformat(),
            "processing_time_seconds": processing_time,
            "input": {
                "original_prompt": prompt,
                "prompt_type": prompt_type
            },
            "output": {
                "optimized_prompt": result.get("final_prompt", prompt),
                "domain": result.get("domain", "unknown"),
                "quality_score": result.get("evaluation_result", {}).get("overall_score", 0),
                "iterations_used": result.get("iterations_used", 0),
                "passes_threshold": result.get("passes_threshold", False)
            },
            "analysis": {
                "classification": result.get("classification_result", {}),
                "improvements": result.get("improvement_result", {}),
                "evaluation": result.get("evaluation_result", {})
            },
            "comparison": {
                "side_by_side": {
                    "original": prompt,
                    "optimized": result.get("final_prompt", prompt)
                },
                "improvement_ratio": result.get("evaluation_result", {}).get("overall_score", 0)
            },
            "metadata": {
                "langsmith_enabled": bool(settings.langsmith_api_key),
                "framework": "langgraph"
            }
        }

    except Exception as e:
        logger.error(f"Error in LangGraph processing: {e}")
        # Return proper format expected by API
        return {
            "workflow_id": f"lg_error_{int(__import__('time').time())}",
            "status": "error",
            "timestamp": __import__('datetime').datetime.now().isoformat(),
            "processing_time_seconds": 0.0,
            "input": {
                "original_prompt": prompt,
                "prompt_type": prompt_type
            },
            "output": {
                "optimized_prompt": prompt,  # Return original prompt on error
                "domain": "unknown",
                "quality_score": 0.0,
                "iterations_used": 0,
                "passes_threshold": False
            },
            "analysis": {
                "classification": {},
                "improvements": {},
                "evaluation": {}
            },
            "comparison": {},
            "metadata": {
                "langsmith_enabled": bool(settings.langsmith_api_key),
                "framework": "langgraph",
                "error": str(e)
            }
        }


if __name__ == "__main__":
    # Example usage
    test_prompt = "Write a function to sort a list of numbers"
    result = process_prompt_with_langgraph(test_prompt, "auto")
    print(f"Result: {result}")
