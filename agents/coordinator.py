"""Coordinator Agent for orchestrating the multi-agent prompt engineering workflow."""

from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import json
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
import time # Added for time.time()

from agents.classifier import DomainClassifier
from agents.base_expert import create_expert_agent, BaseExpertAgent
from agents.evaluator import PromptEvaluator
from agents.exceptions import (
    ClassificationError, ImprovementError, EvaluationError,
    WorkflowError, AgenticSystemError
)
from config.config import (
    settings, setup_langsmith, get_logger, metrics, log_workflow_event,
    cache_manager, perf_config, generate_prompt_cache_key,
    security_manager, security_config, log_security_event, rate_limiter,
    memory_config, prompt_generation_config
)
from agents.memory import memory_manager

# Prompt Management System removed
PROMPT_MANAGEMENT_AVAILABLE = False
PromptManagementSystem = None

# Set up structured logging
logger = get_logger(__name__)


class WorkflowCoordinator:
    """Agent responsible for orchestrating the entire prompt engineering workflow."""

    def __init__(self, classifier_instance: DomainClassifier, evaluator_instance: PromptEvaluator):
        """Initialize the coordinator with references to all agents."""
        self.classifier = classifier_instance
        self.evaluator = evaluator_instance
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

    async def process_prompt(self, prompt: str, prompt_type: str = "auto",
                      return_comparison: bool = True, client_ip: str = "unknown") -> Dict[str, Any]:
        """
        Process a prompt through the complete multi-agent workflow with comprehensive security.

        Args:
            prompt: The input prompt to process
            prompt_type: Type of prompt ("auto", "raw", or "structured")
            return_comparison: Whether to include before/after comparison
            client_ip: IP address of the client making the request

        Returns:
            Dict containing the final optimized prompt and workflow metadata
        """
        start_time = datetime.now()
        workflow_id = f"workflow_{int(start_time.timestamp())}"

        # Security: Rate limiting
        if security_config.enable_rate_limiting:
            if not rate_limiter.is_allowed(client_ip):
                log_security_event(logger, "rate_limit_exceeded", "medium",
                                 client_ip=client_ip, workflow_id=workflow_id)
                raise AgenticSystemError(
                    "Rate limit exceeded. Please try again later.",
                    error_code="RATE_LIMIT_EXCEEDED",
                    client_ip=client_ip
                )

        # Security: Input validation and sanitization
        if security_config.enable_input_sanitization:
            sanitized_result = security_manager.sanitize_input(prompt, "coordinator")

            # Log security events
            if sanitized_result['security_events']:
                log_security_event(logger, "input_security_events", "medium",
                                 workflow_id=workflow_id, client_ip=client_ip,
                                 events=sanitized_result['security_events'])

            # Block unsafe content
            if not sanitized_result['is_safe']:
                high_severity_events = [e for e in sanitized_result['security_events'] if e['severity'] == 'high']
                if high_severity_events:
                    log_security_event(logger, "unsafe_content_blocked", "high",
                                     workflow_id=workflow_id, client_ip=client_ip,
                                     events=high_severity_events)
                    raise AgenticSystemError(
                        "Input contains potentially unsafe content and has been blocked.",
                        error_code="UNSAFE_CONTENT_BLOCKED",
                        security_events=high_severity_events
                    )

            prompt = sanitized_result['sanitized_text']

        logger.info(f"Starting workflow {workflow_id} for prompt processing", extra={
            'workflow_id': workflow_id,
            'client_ip': client_ip,
            'prompt_length': len(prompt),
            'prompt_type': prompt_type
        })

        try:
            # Step 1: Determine prompt type if set to auto
            if prompt_type == "auto":
                prompt_type = await self.classifier.classify_prompt_type(prompt)
                logger.info(f"Auto-detected prompt type: {prompt_type}")

            # Step 2: Classify domain
            logger.info("Step 1: Classifying domain...")
            classification_result = await self.classifier.classify_prompt(prompt)
            domain = classification_result["domain"]

            # Step 3: Get or create expert agent
            logger.info(f"Step 2: Preparing expert agent for domain '{domain}'...")
            expert_agent = self._get_or_create_expert_agent(domain, classification_result)

            # Step 4: Improve prompt
            logger.info("Step 3: Improving prompt...")
            improvement_result = await expert_agent.improve_prompt(
                original_prompt=prompt,
                prompt_type=prompt_type,
                key_topics=classification_result.get("key_topics", [])
            )

            improved_prompt = improvement_result.get("improved_prompt", prompt)

            # Step 5: Evaluate and iterate
            logger.info("Step 4: Evaluating and iterating...")
            final_evaluation, iterations_used = await self.evaluator.run_evaluation_loop(
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

        except ClassificationError as ce:
            logger.error(f"Classification error in workflow {workflow_id}: {ce}")
            error_result = self._prepare_error_result(
                workflow_id, prompt,
                f"Domain classification failed: {ce.message}",
                start_time, ce.error_code, ce.to_dict()
            )
            self._record_workflow(error_result)
            return error_result

        except ImprovementError as ie:
            logger.error(f"Improvement error in workflow {workflow_id}: {ie}")
            error_result = self._prepare_error_result(
                workflow_id, prompt,
                f"Prompt improvement failed: {ie.message}",
                start_time, ie.error_code, ie.to_dict()
            )
            self._record_workflow(error_result)
            return error_result

        except EvaluationError as ee:
            logger.error(f"Evaluation error in workflow {workflow_id}: {ee}")
            error_result = self._prepare_error_result(
                workflow_id, prompt,
                f"Prompt evaluation failed: {ee.message}",
                start_time, ee.error_code, ee.to_dict()
            )
            self._record_workflow(error_result)
            return error_result

        except AgenticSystemError as ase:
            logger.error(f"System error in workflow {workflow_id}: {ase}")
            error_result = self._prepare_error_result(
                workflow_id, prompt,
                f"Agentic system error: {ase.message}",
                start_time, ase.error_code, ase.to_dict()
            )
            self._record_workflow(error_result)
            return error_result

        except Exception as e:
            logger.error(f"Unexpected error in workflow {workflow_id}: {e}")
            error_result = self._prepare_error_result(
                workflow_id, prompt,
                f"Unexpected error: {str(e)}",
                start_time, "UNKNOWN_ERROR", {"cause": str(e)}
            )
            self._record_workflow(error_result)
            return error_result

    async def process_prompt_with_memory(self, prompt: str, user_id: str,
                                       prompt_type: str = "auto", return_comparison: bool = True,
                                       client_ip: str = "unknown") -> Dict[str, Any]:
        """
        Process a prompt through the complete multi-agent workflow with memory and RAG.

        Args:
            prompt: The input prompt to process
            user_id: User identifier for memory retrieval
            prompt_type: Type of prompt ("auto", "raw", or "structured")
            return_comparison: Whether to include before/after comparison
            client_ip: IP address of the client making the request

        Returns:
            Dict containing the final optimized prompt and workflow metadata
        """
        start_time = datetime.now()
        workflow_id = f"memory_workflow_{int(start_time.timestamp())}"

        # Security: Rate limiting
        if security_config.enable_rate_limiting:
            if not rate_limiter.is_allowed(client_ip):
                log_security_event(logger, "rate_limit_exceeded", "medium",
                                 client_ip=client_ip, workflow_id=workflow_id)
                raise AgenticSystemError(
                    "Rate limit exceeded. Please try again later.",
                    error_code="RATE_LIMIT_EXCEEDED",
                    client_ip=client_ip
                )

        # Security: Input validation and sanitization
        if security_config.enable_input_sanitization:
            sanitized_result = security_manager.sanitize_input(prompt, "coordinator")

            # Log security events
            if sanitized_result['security_events']:
                log_security_event(logger, "input_security_events", "medium",
                                 workflow_id=workflow_id, client_ip=client_ip,
                                 events=sanitized_result['security_events'])

            # Block unsafe content
            if not sanitized_result['is_safe']:
                high_severity_events = [e for e in sanitized_result['security_events'] if e['severity'] == 'high']
                if high_severity_events:
                    log_security_event(logger, "unsafe_content_blocked", "high",
                                     workflow_id=workflow_id, client_ip=client_ip,
                                     events=high_severity_events)
                    raise AgenticSystemError(
                        "Input contains potentially unsafe content and has been blocked.",
                        error_code="UNSAFE_CONTENT_BLOCKED",
                        security_events=high_severity_events
                    )

            prompt = sanitized_result['sanitized_text']

        logger.info(f"Starting memory-enhanced workflow {workflow_id} for user {user_id}", extra={
            'workflow_id': workflow_id,
            'user_id': user_id,
            'client_ip': client_ip,
            'prompt_length': len(prompt),
            'prompt_type': prompt_type
        })

        try:
            # Step 1: Determine prompt type if set to auto
            if prompt_type == "auto":
                prompt_type = await self.classifier.classify_prompt_type(prompt)
                logger.info(f"Auto-detected prompt type: {prompt_type}")

            # Step 2: Classify domain
            logger.info("Step 1: Classifying domain...")
            classification_result = await self.classifier.classify_prompt(prompt)
            domain = classification_result["domain"]

            # Step 3: Get or create expert agent
            logger.info(f"Step 2: Preparing expert agent for domain '{domain}'...")
            expert_agent = self._get_or_create_expert_agent(domain, classification_result)

            # Step 4: Improve prompt with memory
            logger.info("Step 3: Improving prompt with memory...")
            improvement_result = await expert_agent.improve_prompt_with_memory(
                original_prompt=prompt,
                user_id=user_id,
                prompt_type=prompt_type,
                key_topics=classification_result.get("key_topics", [])
            )

            improved_prompt = improvement_result.get("improved_prompt", prompt)

            # Step 5: Store conversation memory
            memory_manager.update_conversation_memory(
                user_id=user_id,
                message=prompt,
                response=improved_prompt,
                metadata={
                    'domain': domain,
                    'prompt_type': prompt_type,
                    'workflow_id': workflow_id,
                    'improvement_score': improvement_result.get('effectiveness_score', 0)
                }
            )

            # Step 6: Evaluate and iterate
            logger.info("Step 4: Evaluating and iterating...")
            final_evaluation, iterations_used = await self.evaluator.run_evaluation_loop(
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

            # Step 7: Prepare final result with memory metadata
            workflow_result = self._prepare_memory_final_result(
                workflow_id=workflow_id,
                user_id=user_id,
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

            # Step 8: Record workflow
            self._record_workflow(workflow_result)

            logger.info(f"Memory-enhanced workflow {workflow_id} completed successfully")
            return workflow_result

        except ClassificationError as ce:
            logger.error(f"Classification error in memory workflow {workflow_id}: {ce}")
            error_result = self._prepare_error_result(
                workflow_id, prompt,
                f"Domain classification failed: {ce.message}",
                start_time, ce.error_code, ce.to_dict()
            )
            self._record_workflow(error_result)
            return error_result

        except ImprovementError as ie:
            logger.error(f"Improvement error in memory workflow {workflow_id}: {ie}")
            error_result = self._prepare_error_result(
                workflow_id, prompt,
                f"Prompt improvement failed: {ie.message}",
                start_time, ie.error_code, ie.to_dict()
            )
            self._record_workflow(error_result)
            return error_result

        except EvaluationError as ee:
            logger.error(f"Evaluation error in memory workflow {workflow_id}: {ee}")
            error_result = self._prepare_error_result(
                workflow_id, prompt,
                f"Prompt evaluation failed: {ee.message}",
                start_time, ee.error_code, ee.to_dict()
            )
            self._record_workflow(error_result)
            return error_result

        except AgenticSystemError as ase:
            logger.error(f"System error in memory workflow {workflow_id}: {ase}")
            error_result = self._prepare_error_result(
                workflow_id, prompt,
                f"Agentic system error: {ase.message}",
                start_time, ase.error_code, ase.to_dict()
            )
            self._record_workflow(error_result)
            return error_result

        except Exception as e:
            logger.error(f"Unexpected error in memory workflow {workflow_id}: {e}")
            error_result = self._prepare_error_result(
                workflow_id, prompt,
                f"Unexpected error: {str(e)}",
                start_time, "UNKNOWN_ERROR", {"cause": str(e)}
            )
            self._record_workflow(error_result)
            return error_result




    def _prepare_memory_final_result(self, workflow_id: str, user_id: str,
                                   original_prompt: str, final_prompt: str,
                                   domain: str, classification_result: Dict[str, Any],
                                   improvement_result: Dict[str, Any],
                                   final_evaluation: Dict[str, Any],
                                   iterations_used: int, prompt_type: str,
                                   return_comparison: bool, start_time: datetime) -> Dict[str, Any]:
        """Prepare the final result dictionary for memory-enhanced workflow."""
        processing_time = (datetime.now() - start_time).total_seconds()

        # Get memory statistics
        memory_stats = memory_manager.get_metrics()

        result = {
            "workflow_id": workflow_id,
            "status": "completed",
            "timestamp": datetime.now().isoformat(),
            "processing_time_seconds": processing_time,
            "input": {
                "original_prompt": original_prompt,
                "prompt_type": prompt_type,
                "user_id": user_id
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
                    "effectiveness_score": improvement_result.get("effectiveness_score", 0),
                    "context_utilization": improvement_result.get("context_utilization", {})
                },
                "evaluation": {
                    "overall_score": final_evaluation.get("overall_score", 0),
                    "criteria_scores": final_evaluation.get("criteria_scores", {}),
                    "strengths": final_evaluation.get("strengths", []),
                    "weaknesses": final_evaluation.get("weaknesses", []),
                    "reasoning": final_evaluation.get("reasoning", "")
                }
            },
            "memory": {
                "enabled": True,
                "user_id": user_id,
                "conversation_turns": len(memory_manager.get_conversation_context(user_id)),
                "memory_operations": memory_stats.get('memory_operations', 0),
                "rag_queries": memory_stats.get('rag_queries', 0),
                "cached_memories": memory_stats.get('cached_memories', 0),
                "vector_store_available": memory_stats.get('vector_store_available', False),
                "embedding_model_available": memory_stats.get('embedding_model_available', False)
            },
            "metadata": {
                "langsmith_enabled": bool(settings.langsmith_api_key),
                "evaluation_threshold": settings.evaluation_threshold,
                "max_iterations": settings.max_evaluation_iterations,
                "memory_enabled": memory_config.enable_rag,
                "rag_top_k": memory_config.rag_top_k
            }
        }

        if return_comparison:
            result["comparison"] = {
                "original_length": len(original_prompt),
                "optimized_length": len(final_prompt),
                "improvement_ratio": final_evaluation.get("scores", {}).get("overall_score", 0.5),
                "side_by_side": {
                    "original": original_prompt,
                    "optimized": final_prompt
                }
            }

        return result

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
                "improvement_ratio": final_evaluation.get("scores", {}).get("overall_score", 0.5),
                "side_by_side": {
                    "original": original_prompt,
                    "optimized": final_prompt
                }
            }

        return result


    def _prepare_error_result(self, workflow_id: str, original_prompt: str,
                            error_message: str, start_time: datetime,
                            error_code: str = "UNKNOWN_ERROR",
                            error_details: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Prepare an error result dictionary with detailed error information."""
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
                "error_code": error_code,
                "stage": "workflow_processing",
                "details": error_details or {}
            },
            "metadata": {
                "langsmith_enabled": bool(settings.langsmith_api_key),
                "error_handling_enabled": True
            }
        }

    def _record_workflow(self, workflow_result: Dict[str, Any]):
        """Record the workflow result in history and update domain learning."""
        self.workflow_history.append(workflow_result)

        # Keep only the last 100 workflows to prevent memory issues
        if len(self.workflow_history) > 100:
            self.workflow_history = self.workflow_history[-100:]
        
        # Extract domain information for self-learning
        domain = workflow_result.get("output", {}).get("domain")
        if domain and workflow_result.get("status") == "completed":
            self._update_domain_learning(domain, workflow_result)

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

        quality_scores = [w.get("output", {}).get("quality_score", 0) for w in completed_workflows if w.get("output", {}).get("quality_score")]
        processing_times = [w.get("processing_time_seconds", w.get("processing_time", 0)) for w in completed_workflows]

        return {
            "total_workflows": len(self.workflow_history),
            "completed_workflows": len(completed_workflows),
            "error_workflows": len(error_workflows),
            "success_rate": len(completed_workflows) / len(self.workflow_history),
            "average_quality_score": sum(quality_scores) / len(quality_scores) if quality_scores else 0.0,
            "average_processing_time": sum(processing_times) / len(processing_times) if processing_times else 0.0,
            "domain_distribution": self._calculate_domain_distribution(completed_workflows)
        }

    def _calculate_domain_distribution(self, workflows: List[Dict[str, Any]]) -> Dict[str, int]:
        """Calculate domain distribution across workflows."""
        domain_count = {}
        for workflow in workflows:
            domain = workflow.get("output", {}).get("domain")
            if domain:
                domain_count[domain] = domain_count.get(domain, 0) + 1

        return dict(sorted(domain_count.items(), key=lambda x: x[1], reverse=True))

    def get_available_domains(self) -> List[Dict[str, Any]]:
        """Get information about all available domains."""
        domains_info = []
        
        # Get base domains from classifier
        base_domains = self.classifier.get_available_domains()
        
        # Get learned domains from workflow history
        learned_domains = self._get_learned_domains()
        
        # Combine base and learned domains
        all_domains = {**base_domains, **learned_domains}
        
        for domain_name, domain_info in all_domains.items():
            has_agent = domain_name in self.expert_agents
            is_learned = domain_name in learned_domains
            
            domains_info.append({
                "domain": domain_name,
                "description": domain_info.get("description", ""),
                "keywords": domain_info.get("keywords", []),
                "has_expert_agent": has_agent,
                "agent_created": has_agent,
                "is_learned": is_learned,
                "usage_count": domain_info.get("usage_count", 0),
                "avg_quality_score": domain_info.get("avg_quality_score", 0),
                "last_used": domain_info.get("last_used", "")
            })

        return domains_info

    # Hybrid Integration Methods for Prompt Management System

    def enable_prompt_management(self, enable: bool = True):
        """Enable or disable prompt management integration."""
        self.prompt_management_enabled = enable
        logger.info(f"Prompt management {'enabled' if enable else 'disabled'}")

    def _update_domain_learning(self, domain: str, workflow_result: Dict[str, Any]):
        """Update domain learning data based on workflow results."""
        try:
            # Load existing learned domains
            learned_domains = self._load_learned_domains()
            
            # Initialize domain if not exists
            if domain not in learned_domains:
                learned_domains[domain] = {
                    "description": f"Learned domain: {domain}",
                    "keywords": [],
                    "usage_count": 0,
                    "quality_scores": [],
                    "avg_quality_score": 0,
                    "last_used": "",
                    "created_from_workflow": workflow_result.get("workflow_id"),
                    "learning_metadata": {
                        "first_seen": datetime.now().isoformat(),
                        "improvement_patterns": [],
                        "common_prompt_types": []
                    }
                }
                logger.info(f"New domain '{domain}' learned from workflow {workflow_result.get('workflow_id')}")
            
            # Update domain statistics
            domain_data = learned_domains[domain]
            domain_data["usage_count"] += 1
            domain_data["last_used"] = datetime.now().isoformat()
            
            # Update quality scores
            quality_score = workflow_result.get("output", {}).get("quality_score", 0)
            if quality_score > 0:
                domain_data["quality_scores"].append(quality_score)
                # Keep only last 50 scores to prevent memory bloat
                if len(domain_data["quality_scores"]) > 50:
                    domain_data["quality_scores"] = domain_data["quality_scores"][-50:]
                domain_data["avg_quality_score"] = sum(domain_data["quality_scores"]) / len(domain_data["quality_scores"])
            
            # Extract and update keywords from prompt analysis
            analysis = workflow_result.get("analysis", {})
            classification = analysis.get("classification", {})
            key_topics = classification.get("key_topics", [])
            
            if key_topics:
                existing_keywords = set(domain_data["keywords"])
                new_keywords = [topic for topic in key_topics if topic.lower() not in [k.lower() for k in existing_keywords]]
                domain_data["keywords"].extend(new_keywords[:5])  # Add up to 5 new keywords
                # Keep only most recent 20 keywords
                if len(domain_data["keywords"]) > 20:
                    domain_data["keywords"] = domain_data["keywords"][-20:]
            
            # Update learning metadata
            prompt_type = workflow_result.get("input", {}).get("prompt_type", "unknown")
            if prompt_type not in domain_data["learning_metadata"]["common_prompt_types"]:
                domain_data["learning_metadata"]["common_prompt_types"].append(prompt_type)
            
            # Track improvement patterns
            improvements = analysis.get("improvements", {})
            improvements_made = improvements.get("improvements_made", [])
            if improvements_made:
                domain_data["learning_metadata"]["improvement_patterns"].extend(improvements_made[:3])
                # Keep only last 10 improvement patterns
                if len(domain_data["learning_metadata"]["improvement_patterns"]) > 10:
                    domain_data["learning_metadata"]["improvement_patterns"] = domain_data["learning_metadata"]["improvement_patterns"][-10:]
            
            # Save updated learned domains
            self._save_learned_domains(learned_domains)
            
        except Exception as e:
            logger.error(f"Failed to update domain learning for '{domain}': {e}")
    
    def _load_learned_domains(self) -> Dict[str, Any]:
        """Load learned domains from persistent storage."""
        try:
            import json
            from pathlib import Path
            
            domains_file = Path("data") / "learned_domains.json"
            if domains_file.exists():
                with open(domains_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            logger.error(f"Failed to load learned domains: {e}")
            return {}
    
    def _save_learned_domains(self, learned_domains: Dict[str, Any]):
        """Save learned domains to persistent storage."""
        try:
            import json
            from pathlib import Path
            
            # Ensure data directory exists
            data_dir = Path("data")
            data_dir.mkdir(exist_ok=True)
            
            domains_file = data_dir / "learned_domains.json"
            with open(domains_file, 'w', encoding='utf-8') as f:
                json.dump(learned_domains, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            logger.error(f"Failed to save learned domains: {e}")
    
    def _get_learned_domains(self) -> Dict[str, Any]:
        """Get all learned domains."""
        return self._load_learned_domains()
    
    def get_domain_learning_stats(self) -> Dict[str, Any]:
        """Get statistics about domain learning."""
        try:
            learned_domains = self._load_learned_domains()
            
            stats = {
                "total_learned_domains": len(learned_domains),
                "domains_by_usage": [],
                "domains_by_quality": [],
                "recent_domains": [],
                "learning_summary": {
                    "total_keywords_learned": 0,
                    "total_improvement_patterns": 0,
                    "avg_domain_quality": 0
                }
            }
            
            if learned_domains:
                # Sort domains by usage
                stats["domains_by_usage"] = sorted(
                    [(domain, data["usage_count"]) for domain, data in learned_domains.items()],
                    key=lambda x: x[1], reverse=True
                )[:10]
                
                # Sort domains by quality
                stats["domains_by_quality"] = sorted(
                    [(domain, data["avg_quality_score"]) for domain, data in learned_domains.items() if data["avg_quality_score"] > 0],
                    key=lambda x: x[1], reverse=True
                )[:10]
                
                # Recent domains (last 30 days)
                from datetime import datetime, timedelta
                thirty_days_ago = datetime.now() - timedelta(days=30)
                
                recent_domains = []
                for domain, data in learned_domains.items():
                    if data.get("last_used"):
                        try:
                            last_used = datetime.fromisoformat(data["last_used"].replace('Z', '+00:00'))
                            if last_used > thirty_days_ago:
                                recent_domains.append((domain, data["last_used"]))
                        except:
                            continue
                
                stats["recent_domains"] = sorted(recent_domains, key=lambda x: x[1], reverse=True)[:10]
                
                # Learning summary
                total_keywords = sum(len(data.get("keywords", [])) for data in learned_domains.values())
                total_patterns = sum(len(data.get("learning_metadata", {}).get("improvement_patterns", [])) for data in learned_domains.values())
                avg_quality = sum(data.get("avg_quality_score", 0) for data in learned_domains.values()) / len(learned_domains)
                
                stats["learning_summary"] = {
                    "total_keywords_learned": total_keywords,
                    "total_improvement_patterns": total_patterns,
                    "avg_domain_quality": round(avg_quality, 2)
                }
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get domain learning stats: {e}")
            return {"error": str(e)}

def create_coordinator():
    """Create a new WorkflowCoordinator instance."""
    from agents.classifier import DomainClassifier
    from agents.evaluator import PromptEvaluator
    
    classifier = DomainClassifier()
    evaluator = PromptEvaluator()
    return WorkflowCoordinator(classifier, evaluator)

def get_coordinator_instance():
    """Get the singleton coordinator instance."""
    global _coordinator_instance
    if _coordinator_instance is None:
        _coordinator_instance = create_coordinator()
    return _coordinator_instance

# For backward compatibility, provide a coordinator instance
coordinator = create_coordinator()
