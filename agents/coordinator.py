"""Coordinator Agent for orchestrating the multi-agent prompt engineering workflow."""

from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import json
import logging

from agents.classifier import classifier, DomainClassifier
from agents.base_expert import create_expert_agent, BaseExpertAgent
from agents.evaluator import evaluator, PromptEvaluator
from config.config import settings, setup_langsmith

# Set up logging
logging.basicConfig(level=settings.log_level)
logger = logging.getLogger(__name__)


class WorkflowCoordinator:
    """Agent responsible for orchestrating the entire prompt engineering workflow."""

    def __init__(self):
        """Initialize the coordinator with references to all agents."""
        self.classifier = classifier
        self.evaluator = evaluator
        self.expert_agents = {}  # Cache for created expert agents
        self.workflow_history = []  # Track workflow executions
        self._setup_langsmith()

    def _setup_langsmith(self):
        """Set up LangSmith tracing if available."""
        if settings.langsmith_api_key:
            setup_langsmith()
            logger.info("LangSmith tracing enabled")
        else:
            logger.info("LangSmith not configured - tracing disabled")

    def process_prompt(self, prompt: str, prompt_type: str = "auto",
                      return_comparison: bool = True) -> Dict[str, Any]:
        """
        Process a prompt through the complete multi-agent workflow.

        Args:
            prompt: The input prompt to process
            prompt_type: Type of prompt ("auto", "raw", or "structured")
            return_comparison: Whether to include before/after comparison

        Returns:
            Dict containing the final optimized prompt and workflow metadata
        """
        start_time = datetime.now()
        workflow_id = f"workflow_{int(start_time.timestamp())}"

        logger.info(f"Starting workflow {workflow_id} for prompt processing")

        try:
            # Step 1: Determine prompt type if set to auto
            if prompt_type == "auto":
                prompt_type = self._classify_prompt_type(prompt)
                logger.info(f"Auto-detected prompt type: {prompt_type}")

            # Step 2: Classify domain
            logger.info("Step 1: Classifying domain...")
            classification_result = self.classifier.classify_prompt(prompt)
            domain = classification_result["domain"]

            # Step 3: Get or create expert agent
            logger.info(f"Step 2: Preparing expert agent for domain '{domain}'...")
            expert_agent = self._get_or_create_expert_agent(domain, classification_result)

            # Step 4: Improve prompt
            logger.info("Step 3: Improving prompt...")
            improvement_result = expert_agent.improve_prompt(
                original_prompt=prompt,
                prompt_type=prompt_type,
                key_topics=classification_result.get("key_topics", [])
            )

            improved_prompt = improvement_result.get("improved_prompt", prompt)

            # Step 5: Evaluate and iterate
            logger.info("Step 4: Evaluating and iterating...")
            final_evaluation, iterations_used = self.evaluator.run_evaluation_loop(
                original_prompt=prompt,
                improved_prompt=improved_prompt,
                domain=domain,
                expert_agent=expert_agent,
                prompt_type=prompt_type
            )

            # Get the final prompt from the evaluation loop
            final_prompt = improved_prompt
            if iterations_used > 1:
                # If we had multiple iterations, we need to get the final improved prompt
                # This would be handled by the evaluation loop returning the final prompt
                pass

            # Step 6: Prepare final result
            workflow_result = self._prepare_final_result(
                workflow_id=workflow_id,
                original_prompt=prompt,
                final_prompt=final_prompt,
                domain=domain,
                classification_result=classification_result,
                improvement_result=improvement_result,
                final_evaluation=final_evaluation,
                iterations_used=iterations_used,
                prompt_type=prompt_type,
                return_comparison=return_comparison,
                start_time=start_time
            )

            # Step 7: Record workflow
            self._record_workflow(workflow_result)

            logger.info(f"Workflow {workflow_id} completed successfully")
            return workflow_result

        except Exception as e:
            logger.error(f"Error in workflow {workflow_id}: {e}")
            error_result = self._prepare_error_result(workflow_id, prompt, str(e), start_time)
            self._record_workflow(error_result)
            return error_result

    def _classify_prompt_type(self, prompt: str) -> str:
        """Classify whether a prompt is raw or structured."""
        # Simple heuristic: if it has clear sections, formatting, or specific keywords, it's structured
        structured_indicators = [
            "requirements:", "specifications:", "please", "i need", "create",
            "develop", "build", "implement", "task:", "objective:",
            "1.", "2.", "3.", "-", "*", "•"
        ]

        prompt_lower = prompt.lower()
        structured_score = sum(1 for indicator in structured_indicators if indicator in prompt_lower)

        # Check for structured formatting
        lines = prompt.split('\n')
        formatted_lines = sum(1 for line in lines if line.strip().startswith(('-', '*', '•', '1.', '2.', '3.')))

        if structured_score > 3 or formatted_lines > 2 or len(lines) > 5:
            return "structured"
        else:
            return "raw"

    def _get_or_create_expert_agent(self, domain: str, classification_result: Dict[str, Any]) -> BaseExpertAgent:
        """Get an existing expert agent or create a new one for the domain."""
        if domain in self.expert_agents:
            logger.info(f"Using cached expert agent for domain '{domain}'")
            return self.expert_agents[domain]

        # Get domain description
        domain_description = classification_result.get("reasoning", f"Expert in {domain}")

        # Create new expert agent
        logger.info(f"Creating new expert agent for domain '{domain}'")
        expert_agent = create_expert_agent(domain, domain_description)

        # Cache the agent
        self.expert_agents[domain] = expert_agent

        return expert_agent

    def _prepare_final_result(self, workflow_id: str, original_prompt: str,
                            final_prompt: str, domain: str,
                            classification_result: Dict[str, Any],
                            improvement_result: Dict[str, Any],
                            final_evaluation: Dict[str, Any],
                            iterations_used: int, prompt_type: str,
                            return_comparison: bool, start_time: datetime) -> Dict[str, Any]:
        """Prepare the final result dictionary."""
        processing_time = (datetime.now() - start_time).total_seconds()

        result = {
            "workflow_id": workflow_id,
            "status": "completed",
            "timestamp": datetime.now().isoformat(),
            "processing_time_seconds": processing_time,
            "input": {
                "original_prompt": original_prompt,
                "prompt_type": prompt_type
            },
            "output": {
                "optimized_prompt": final_prompt,
                "domain": domain,
                "quality_score": final_evaluation.get("overall_score", 0),
                "iterations_used": iterations_used,
                "passes_threshold": final_evaluation.get("passes_threshold", False)
            },
            "analysis": {
                "classification": {
                    "domain": classification_result.get("domain"),
                    "confidence": classification_result.get("confidence", 0),
                    "key_topics": classification_result.get("key_topics", []),
                    "reasoning": classification_result.get("reasoning", "")
                },
                "improvements": {
                    "improvements_made": improvement_result.get("improvements_made", []),
                    "key_additions": improvement_result.get("key_additions", []),
                    "effectiveness_score": improvement_result.get("effectiveness_score", 0)
                },
                "evaluation": {
                    "overall_score": final_evaluation.get("overall_score", 0),
                    "criteria_scores": final_evaluation.get("criteria_scores", {}),
                    "strengths": final_evaluation.get("strengths", []),
                    "weaknesses": final_evaluation.get("weaknesses", []),
                    "reasoning": final_evaluation.get("reasoning", "")
                }
            },
            "metadata": {
                "langsmith_enabled": bool(settings.langsmith_api_key),
                "evaluation_threshold": settings.evaluation_threshold,
                "max_iterations": settings.max_evaluation_iterations
            }
        }

        if return_comparison:
            result["comparison"] = {
                "original_length": len(original_prompt),
                "optimized_length": len(final_prompt),
                "improvement_ratio": len(final_prompt) / len(original_prompt) if original_prompt else 0,
                "side_by_side": {
                    "original": original_prompt,
                    "optimized": final_prompt
                }
            }

        return result

    def _prepare_error_result(self, workflow_id: str, original_prompt: str,
                            error_message: str, start_time: datetime) -> Dict[str, Any]:
        """Prepare an error result dictionary."""
        processing_time = (datetime.now() - start_time).total_seconds()

        return {
            "workflow_id": workflow_id,
            "status": "error",
            "timestamp": datetime.now().isoformat(),
            "processing_time_seconds": processing_time,
            "input": {
                "original_prompt": original_prompt,
                "prompt_type": "unknown"
            },
            "error": {
                "message": error_message,
                "stage": "workflow_processing"
            },
            "metadata": {
                "langsmith_enabled": bool(settings.langsmith_api_key)
            }
        }

    def _record_workflow(self, workflow_result: Dict[str, Any]):
        """Record the workflow result in history."""
        self.workflow_history.append(workflow_result)

        # Keep only the last 100 workflows to prevent memory issues
        if len(self.workflow_history) > 100:
            self.workflow_history = self.workflow_history[-100:]

    def get_workflow_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent workflow history."""
        return self.workflow_history[-limit:] if limit > 0 else self.workflow_history

    def get_workflow_stats(self) -> Dict[str, Any]:
        """Get statistics about completed workflows."""
        if not self.workflow_history:
            return {"error": "No workflow history available"}

        completed_workflows = [w for w in self.workflow_history if w.get("status") == "completed"]
        error_workflows = [w for w in self.workflow_history if w.get("status") == "error"]

        if not completed_workflows:
            return {
                "total_workflows": len(self.workflow_history),
                "completed_workflows": 0,
                "error_workflows": len(error_workflows),
                "success_rate": 0.0,
                "average_quality_score": 0.0,
                "average_processing_time": 0.0
            }

        quality_scores = [w["output"]["quality_score"] for w in completed_workflows]
        processing_times = [w["processing_time_seconds"] for w in completed_workflows]

        return {
            "total_workflows": len(self.workflow_history),
            "completed_workflows": len(completed_workflows),
            "error_workflows": len(error_workflows),
            "success_rate": len(completed_workflows) / len(self.workflow_history),
            "average_quality_score": sum(quality_scores) / len(quality_scores),
            "average_processing_time": sum(processing_times) / len(processing_times),
            "domain_distribution": self._calculate_domain_distribution(completed_workflows)
        }

    def _calculate_domain_distribution(self, workflows: List[Dict[str, Any]]) -> Dict[str, int]:
        """Calculate domain distribution across workflows."""
        domain_count = {}
        for workflow in workflows:
            domain = workflow["output"]["domain"]
            domain_count[domain] = domain_count.get(domain, 0) + 1

        return dict(sorted(domain_count.items(), key=lambda x: x[1], reverse=True))

    def get_available_domains(self) -> List[Dict[str, Any]]:
        """Get information about all available domains."""
        domains_info = []
        for domain_name, domain_info in self.classifier.get_available_domains().items():
            has_agent = domain_name in self.expert_agents
            domains_info.append({
                "domain": domain_name,
                "description": domain_info.get("description", ""),
                "keywords": domain_info.get("keywords", []),
                "has_expert_agent": has_agent,
                "agent_created": has_agent
            })

        return domains_info


# Global coordinator instance
coordinator = WorkflowCoordinator()
