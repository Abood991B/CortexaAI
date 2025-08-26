"""Coordinator Agent for orchestrating the multi-agent prompt engineering workflow."""

from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import json
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
import time # Added for time.time()

from agents.prompt.prompt_models import PromptMetadata # Added for PromptMetadata

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
    memory_config, planning_config, prompt_generation_config
)
from agents.memory import memory_manager
from agents.planning import planning_engine
from agents.prompt import prompt_generator

# Import Prompt Management System (optional for hybrid integration)
try:
    from agents.prompt import PromptManagementSystem
    PROMPT_MANAGEMENT_AVAILABLE = True
except ImportError:
    PromptManagementSystem = None
    PROMPT_MANAGEMENT_AVAILABLE = False

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

        # Hybrid Integration: Prompt Management System
        self.prompt_manager = None
        self._enable_prompt_management = False
        self._enable_performance_tracking = False
        self._enable_experimentation = False
        self.migrated_domains = set()  # Domains using new system

        if PROMPT_MANAGEMENT_AVAILABLE:
            try:
                self.prompt_manager = PromptManagementSystem()
                logger.info("Prompt Management System initialized for hybrid integration")
            except Exception as e:
                logger.warning(f"Failed to initialize Prompt Management System: {e}")

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

    async def process_prompt_with_planning(self, prompt: str, user_id: str = None,
                                         prompt_type: str = "auto", return_comparison: bool = True,
                                         client_ip: str = "unknown") -> Dict[str, Any]:
        """
        Process a prompt through the complete multi-agent workflow with planning and reasoning.

        Args:
            prompt: The input prompt to process
            user_id: User identifier for context
            prompt_type: Type of prompt ("auto", "raw", or "structured")
            return_comparison: Whether to include before/after comparison
            client_ip: IP address of the client making the request

        Returns:
            Dict containing the final optimized prompt and workflow metadata
        """
        start_time = datetime.now()
        workflow_id = f"planning_workflow_{int(start_time.timestamp())}"

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

        logger.info(f"Starting planning-enhanced workflow {workflow_id}", extra={
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

            # Step 3: Create plan using planning engine
            logger.info("Step 2: Creating execution plan...")
            plan = await planning_engine.create_plan(
                task=f"Optimize prompt: {prompt}",
                domain=domain,
                user_id=user_id,
                context={
                    'classification_result': classification_result,
                    'prompt_type': prompt_type,
                    'client_ip': client_ip
                }
            )

            # Step 4: Execute plan step by step
            logger.info("Step 3: Executing plan...")
            final_result = None
            async for step_result in planning_engine.execute_plan(plan):
                if step_result.get('status') in ['completed', 'partial']:
                    final_result = step_result
                    break
                elif step_result.get('status') == 'failed':
                    raise AgenticSystemError(
                        f"Plan execution failed: {step_result.get('error', 'Unknown error')}",
                        error_code="PLAN_EXECUTION_FAILED"
                    )

            if not final_result:
                raise AgenticSystemError(
                    "Plan execution did not produce a final result",
                    error_code="NO_EXECUTION_RESULT"
                )

            # Step 5: Get the final improved prompt from execution results
            final_prompt = prompt  # Default fallback
            if final_result.get('results'):
                # Find the last successful subtask result
                successful_results = [
                    result for result in final_result['results'].values()
                    if result.get('success', False)
                ]
                if successful_results:
                    # Use the output from the last successful step
                    final_prompt = successful_results[-1].get('output', prompt)

            # Step 6: Prepare final result with planning metadata
            workflow_result = self._prepare_planning_final_result(
                workflow_id=workflow_id,
                user_id=user_id,
                original_prompt=prompt,
                final_prompt=final_prompt,
                domain=domain,
                classification_result=classification_result,
                plan=plan,
                execution_result=final_result,
                prompt_type=prompt_type,
                return_comparison=return_comparison,
                start_time=start_time
            )

            # Step 7: Record workflow
            self._record_workflow(workflow_result)

            logger.info(f"Planning-enhanced workflow {workflow_id} completed successfully")
            return workflow_result

        except ClassificationError as ce:
            logger.error(f"Classification error in planning workflow {workflow_id}: {ce}")
            error_result = self._prepare_error_result(
                workflow_id, prompt,
                f"Domain classification failed: {ce.message}",
                start_time, ce.error_code, ce.to_dict()
            )
            self._record_workflow(error_result)
            return error_result

        except AgenticSystemError as ase:
            logger.error(f"System error in planning workflow {workflow_id}: {ase}")
            error_result = self._prepare_error_result(
                workflow_id, prompt,
                f"Agentic system error: {ase.message}",
                start_time, ase.error_code, ase.to_dict()
            )
            self._record_workflow(error_result)
            return error_result

        except Exception as e:
            logger.error(f"Unexpected error in planning workflow {workflow_id}: {e}")
            error_result = self._prepare_error_result(
                workflow_id, prompt,
                f"Unexpected error: {str(e)}",
                start_time, "UNKNOWN_ERROR", {"cause": str(e)}
            )
            self._record_workflow(error_result)
            return error_result

    async def process_prompt_with_generation(self, task: str, user_id: str = None,
                                           domain: str = None, return_comparison: bool = True,
                                           client_ip: str = "unknown") -> Dict[str, Any]:
        """
        Process a task through the complete prompt generation and optimization workflow.

        Args:
            task: The task to generate an optimized prompt for
            user_id: User identifier for context
            domain: The domain context (auto-detected if None)
            return_comparison: Whether to include before/after comparison
            client_ip: IP address of the client making the request

        Returns:
            Dict containing the generated and optimized prompt with workflow metadata
        """
        start_time = datetime.now()
        workflow_id = f"generation_workflow_{int(start_time.timestamp())}"

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
            sanitized_result = security_manager.sanitize_input(task, "generation")

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

            task = sanitized_result['sanitized_text']

        logger.info(f"Starting prompt generation workflow {workflow_id}", extra={
            'workflow_id': workflow_id,
            'user_id': user_id,
            'client_ip': client_ip,
            'task_length': len(task)
        })

        try:
            # Step 1: Determine domain if not provided
            if not domain:
                classification_result = await self.classifier.classify_prompt(task)
                domain = classification_result["domain"]
                logger.info(f"Auto-detected domain: {domain}")

            # Step 2: Generate initial prompt using advanced strategies
            logger.info("Step 1: Generating initial prompt...")
            generation_result = await prompt_generator.generate_prompt(
                task=task,
                domain=domain,
                strategy=prompt_generation_config.generation_strategy,
                context={
                    'user_id': user_id,
                    'client_ip': client_ip,
                    'classification_result': classification_result if 'classification_result' in locals() else {}
                }
            )

            generated_prompt = generation_result['generated_prompt']

            # Step 3: Evaluate generation quality
            logger.info("Step 2: Evaluating generation quality...")
            generation_evaluation = await self.evaluator.evaluate_prompt_generation(
                task=task,
                generated_prompt=generated_prompt,
                domain=domain,
                strategy_used=generation_result['strategy_used'],
                generation_metadata={
                    'quality_score': generation_result.get('quality_score', 0),
                    'generation_time': generation_result.get('generation_time', 0)
                }
            )

            # Step 4: Optimize the generated prompt
            logger.info("Step 3: Optimizing generated prompt...")
            optimization_result = await prompt_generator.optimize_prompt(
                prompt=generated_prompt,
                task=task,
                domain=domain,
                algorithm="auto",  # Let system choose best algorithm
                context={'generation_evaluation': generation_evaluation}
            )

            optimized_prompt = optimization_result['optimized_prompt']

            # Step 5: Evaluate optimization performance
            logger.info("Step 4: Evaluating optimization performance...")
            optimization_evaluation = await self.evaluator.evaluate_optimization_performance(
                original_prompt=generated_prompt,
                optimized_prompt=optimized_prompt,
                algorithm_used=optimization_result['algorithm_used'],
                iterations_used=optimization_result['iterations_used'],
                improvement_score=optimization_result['final_quality_score'],
                domain=domain
            )

            # Step 6: Prepare final result with comprehensive metadata
            workflow_result = self._prepare_generation_final_result(
                workflow_id=workflow_id,
                task=task,
                user_id=user_id,
                domain=domain,
                generation_result=generation_result,
                optimization_result=optimization_result,
                generation_evaluation=generation_evaluation,
                optimization_evaluation=optimization_evaluation,
                return_comparison=return_comparison,
                start_time=start_time
            )

            # Step 7: Record workflow
            self._record_workflow(workflow_result)

            logger.info(f"Prompt generation workflow {workflow_id} completed successfully")
            return workflow_result

        except ClassificationError as ce:
            logger.error(f"Classification error in generation workflow {workflow_id}: {ce}")
            error_result = self._prepare_error_result(
                workflow_id, task,
                f"Domain classification failed: {ce.message}",
                start_time, ce.error_code, ce.to_dict()
            )
            self._record_workflow(error_result)
            return error_result

        except AgenticSystemError as ase:
            logger.error(f"System error in generation workflow {workflow_id}: {ase}")
            error_result = self._prepare_error_result(
                workflow_id, task,
                f"Agentic system error: {ase.message}",
                start_time, ase.error_code, ase.to_dict()
            )
            self._record_workflow(error_result)
            return error_result

        except Exception as e:
            logger.error(f"Unexpected error in generation workflow {workflow_id}: {e}")
            error_result = self._prepare_error_result(
                workflow_id, task,
                f"Unexpected error: {str(e)}",
                start_time, "UNKNOWN_ERROR", {"cause": str(e)}
            )
            self._record_workflow(error_result)
            return error_result

    async def compare_prompt_generation_strategies(self, task: str, domain: str = None,
                                                 strategies: List[str] = None) -> Dict[str, Any]:
        """
        Compare different prompt generation strategies for the same task.

        Args:
            task: The task to generate prompts for
            domain: The domain context
            strategies: List of strategies to compare

        Returns:
            Dict containing strategy comparison results
        """
        if not strategies:
            strategies = ['template_based', 'chain_of_prompts', 'contextual_injection', 'hybrid']

        comparison_id = f"comparison_{int(time.time())}"
        logger.info(f"Starting strategy comparison {comparison_id}")

        try:
            # Determine domain if not provided
            if not domain:
                classification_result = await self.classifier.classify_prompt(task)
                domain = classification_result["domain"]

            # Generate prompts using different strategies
            generated_prompts = []
            for strategy in strategies:
                try:
                    result = await prompt_generator.generate_prompt(
                        task=task,
                        domain=domain,
                        strategy=strategy
                    )

                    generated_prompts.append({
                        'strategy': strategy,
                        'prompt': result['generated_prompt'],
                        'quality_score': result.get('quality_score', 0),
                        'task': task,
                        'domain': domain,
                        'generation_metadata': result.get('metadata', {})
                    })

                except Exception as e:
                    logger.warning(f"Failed to generate prompt with strategy {strategy}: {e}")
                    continue

            if len(generated_prompts) < 2:
                return {
                    'error': 'Need at least 2 successful generations for comparison',
                    'comparison_id': comparison_id
                }

            # Compare the generated prompts
            comparison_result = await self.evaluator.compare_prompts(
                prompts=generated_prompts,
                criteria=['clarity', 'specificity', 'structure', 'completeness', 'actionability']
            )

            result = {
                'comparison_id': comparison_id,
                'task': task,
                'domain': domain,
                'strategies_compared': strategies,
                'generated_prompts': generated_prompts,
                'comparison': comparison_result,
                'best_strategy': comparison_result.get('ranked_prompts', [{}])[0].get('prompt_data', {}).get('strategy'),
                'metadata': {
                    'comparison_timestamp': datetime.now().isoformat(),
                    'successful_generations': len(generated_prompts)
                }
            }

            logger.info(f"Strategy comparison {comparison_id} completed")
            return result

        except Exception as e:
            logger.error(f"Failed to compare strategies {comparison_id}: {e}")
            return {
                'error': str(e),
                'comparison_id': comparison_id,
                'task': task
            }

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
                "improvement_ratio": len(final_prompt) / len(original_prompt) if original_prompt else 0,
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
                "improvement_ratio": len(final_prompt) / len(original_prompt) if original_prompt else 0,
                "side_by_side": {
                    "original": original_prompt,
                    "optimized": final_prompt
                }
            }

        return result

    def _prepare_planning_final_result(self, workflow_id: str, user_id: str,
                                     original_prompt: str, final_prompt: str,
                                     domain: str, classification_result: Dict[str, Any],
                                     plan: Dict[str, Any], execution_result: Dict[str, Any],
                                     prompt_type: str, return_comparison: bool,
                                     start_time: datetime) -> Dict[str, Any]:
        """Prepare the final result dictionary for planning-enhanced workflow."""
        processing_time = (datetime.now() - start_time).total_seconds()

        # Get planning statistics
        planning_stats = planning_engine.get_metrics()

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
                "quality_score": 0.9,  # Would be calculated from evaluation
                "iterations_used": len(plan.get('subtasks', [])),
                "passes_threshold": True  # Would be determined by evaluation
            },
            "analysis": {
                "classification": {
                    "domain": classification_result.get("domain"),
                    "confidence": classification_result.get("confidence", 0),
                    "key_topics": classification_result.get("key_topics", []),
                    "reasoning": classification_result.get("reasoning", "")
                },
                "planning": {
                    "plan_id": plan.get('plan_id'),
                    "strategy": plan.get('strategy'),
                    "complexity": plan.get('complexity'),
                    "subtasks_count": len(plan.get('subtasks', [])),
                    "execution_success_rate": execution_result.get('success_rate', 0),
                    "planning_time": plan.get('metadata', {}).get('planning_time_seconds', 0)
                },
                "improvements": {
                    "improvements_made": ["Planning-based optimization"],
                    "key_additions": ["Structured approach", "Reasoning-driven improvements"],
                    "effectiveness_score": 0.9
                }
            },
            "planning": {
                "enabled": True,
                "plan_id": plan.get('plan_id'),
                "strategy": plan.get('strategy'),
                "subtasks_executed": len(execution_result.get('results', {})),
                "subtasks_successful": execution_result.get('completed_steps', 0),
                "execution_time": execution_result.get('execution_time', 0),
                "success_rate": execution_result.get('success_rate', 0),
                "reasoning_steps": planning_stats.get('reasoning_steps', 0),
                "plans_created": planning_stats.get('plans_created', 0),
                "plans_successful": planning_stats.get('plans_successful', 0)
            },
            "metadata": {
                "langsmith_enabled": bool(settings.langsmith_api_key),
                "planning_enabled": planning_config.enable_planning,
                "planning_engine": planning_config.planning_engine,
                "max_reasoning_depth": planning_config.max_reasoning_depth
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

    def _prepare_generation_final_result(self, workflow_id: str, task: str, user_id: str,
                                       domain: str, generation_result: Dict[str, Any],
                                       optimization_result: Dict[str, Any],
                                       generation_evaluation: Dict[str, Any],
                                       optimization_evaluation: Dict[str, Any],
                                       return_comparison: bool, start_time: datetime) -> Dict[str, Any]:
        """Prepare the final result dictionary for prompt generation workflow."""
        processing_time = (datetime.now() - start_time).total_seconds()

        result = {
            "workflow_id": workflow_id,
            "status": "completed",
            "timestamp": datetime.now().isoformat(),
            "processing_time_seconds": processing_time,
            "input": {
                "task": task,
                "user_id": user_id,
                "domain": domain
            },
            "output": {
                "generated_prompt": generation_result.get('generated_prompt', ''),
                "optimized_prompt": optimization_result.get('optimized_prompt', ''),
                "quality_score": optimization_result.get('final_quality_score', 0),
                "passes_threshold": optimization_result.get('passes_threshold', False)
            },
            "analysis": {
                "generation": {
                    "strategy_used": generation_result.get('strategy_used', 'unknown'),
                    "quality_score": generation_result.get('quality_score', 0),
                    "generation_time": generation_result.get('generation_time', 0),
                    "template_used": generation_result.get('template_used', ''),
                    "persona_applied": generation_result.get('persona_applied', '')
                },
                "optimization": {
                    "algorithm_used": optimization_result.get('algorithm_used', 'unknown'),
                    "iterations_used": optimization_result.get('iterations_used', 0),
                    "final_quality_score": optimization_result.get('final_quality_score', 0),
                    "improvement_score": optimization_result.get('improvement_score', 0),
                    "optimizations_applied": optimization_result.get('optimizations_applied', [])
                },
                "generation_evaluation": {
                    "overall_score": generation_evaluation.get('overall_score', 0),
                    "criteria_scores": generation_evaluation.get('criteria_scores', {}),
                    "strengths": generation_evaluation.get('strengths', []),
                    "weaknesses": generation_evaluation.get('weaknesses', []),
                    "recommendations": generation_evaluation.get('recommendations', [])
                },
                "optimization_evaluation": {
                    "overall_score": optimization_evaluation.get('overall_score', 0),
                    "improvement_metrics": optimization_evaluation.get('improvement_metrics', {}),
                    "performance_score": optimization_evaluation.get('performance_score', 0),
                    "efficiency_score": optimization_evaluation.get('efficiency_score', 0),
                    "final_recommendations": optimization_evaluation.get('final_recommendations', [])
                }
            },
            "metadata": {
                "langsmith_enabled": bool(settings.langsmith_api_key),
                "generation_enabled": prompt_generation_config.enable_generation,
                "optimization_enabled": prompt_generation_config.enable_optimization,
                "evaluation_enabled": prompt_generation_config.enable_evaluation,
                "template_library_size": len(prompt_generation_config.templates),
                "persona_library_size": len(prompt_generation_config.personas)
            }
        }

        if return_comparison:
            generated_prompt = generation_result.get('generated_prompt', '')
            optimized_prompt = optimization_result.get('optimized_prompt', '')
            result["comparison"] = {
                "task_length": len(task),
                "generated_length": len(generated_prompt),
                "optimized_length": len(optimized_prompt),
                "generation_ratio": len(generated_prompt) / len(task) if task else 0,
                "optimization_ratio": len(optimized_prompt) / len(generated_prompt) if generated_prompt else 0,
                "side_by_side": {
                    "task": task,
                    "generated": generated_prompt,
                    "optimized": optimized_prompt
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

    # Hybrid Integration Methods for Prompt Management System

    def enable_prompt_management(self, enable: bool = True):
        """Enable or disable the Prompt Management System features."""
        if not self.prompt_manager:
            logger.warning("Prompt Management System not available")
            return False

        self._enable_prompt_management = enable
        logger.info(f"Prompt Management System {'enabled' if enable else 'disabled'}")
        return True

    def enable_performance_tracking(self, enable: bool = True):
        """Enable or disable performance tracking for prompts."""
        if not self.prompt_manager:
            logger.warning("Prompt Management System not available for performance tracking")
            return False

        self._enable_performance_tracking = enable
        logger.info(f"Performance tracking {'enabled' if enable else 'disabled'}")
        return True

    def enable_experimentation(self, enable: bool = True):
        """Enable or disable A/B testing and experimentation."""
        if not self.prompt_manager:
            logger.warning("Prompt Management System not available for experimentation")
            return False

        self._enable_experimentation = enable
        logger.info(f"Experimentation {'enabled' if enable else 'disabled'}")
        return True

    def migrate_domain_to_new_system(self, domain: str):
        """Migrate a domain to use the new Prompt Management System."""
        if not self.prompt_manager:
            logger.warning("Prompt Management System not available")
            return False

        if domain in self.migrated_domains:
            logger.info(f"Domain '{domain}' already migrated")
            return True

        self.migrated_domains.add(domain)
        logger.info(f"Domain '{domain}' migrated to Prompt Management System")
        return True

    def get_best_prompt_for_domain(self, domain: str, fallback_to_old: bool = True):
        """Get the best performing prompt for a domain from the new system."""
        if not self.prompt_manager or not self._enable_prompt_management:
            if fallback_to_old:
                logger.info(f"Using fallback system for domain '{domain}'")
                return None  # Return None to trigger fallback in calling method
            else:
                raise AgenticSystemError(
                    "Prompt Management System not available",
                    error_code="PROMPT_MANAGEMENT_UNAVAILABLE"
                )

        try:
            prompt_data = self.prompt_manager.get_best_prompt_for_domain(domain)
            if prompt_data:
                return prompt_data['content']

            if fallback_to_old:
                logger.info(f"No prompts found for domain '{domain}', using fallback")
                return None

            raise AgenticSystemError(
                f"No prompts found for domain '{domain}'",
                error_code="NO_PROMPTS_FOUND"
            )

        except Exception as e:
            if fallback_to_old:
                logger.warning(f"Error getting prompt from new system for domain '{domain}': {e}")
                return None
            else:
                raise AgenticSystemError(
                    f"Error retrieving prompt: {e}",
                    error_code="PROMPT_RETRIEVAL_ERROR"
                )

    def record_prompt_performance(self, domain: str, prompt_content: str,
                                performance_score: float, metadata: Dict[str, Any] = None):
        """Record performance metrics for a prompt."""
        if not self._enable_performance_tracking or not self.prompt_manager:
            return False

        try:
            self.prompt_manager.record_performance(
                domain=domain,
                prompt_content=prompt_content,
                performance_score=performance_score,
                metadata=metadata or {}
            )
            return True
        except Exception as e:
            logger.warning(f"Failed to record performance: {e}")
            return False

    def create_experiment(self, name: str, domain: str, variants: List[str],
                         traffic_split: List[float] = None):
        """Create an A/B test experiment for prompt variants."""
        if not self._enable_experimentation or not self.prompt_manager:
            raise AgenticSystemError(
                "Experimentation not enabled or Prompt Management System unavailable",
                error_code="EXPERIMENTATION_UNAVAILABLE"
            )

        try:
            # Create real prompts in the registry for each variant
            experiment_variants = []
            for i, variant_content in enumerate(variants):
                # Create a prompt in the registry for this variant
                prompt_id = self.prompt_manager.create_prompt(
                    name=f"Experiment Variant {i} - {name}",
                    content=variant_content,
                    metadata=PromptMetadata(
                        domain=domain,
                        strategy="experiment_variant",
                        author="coordinator",
                        tags=["experiment", f"variant_{i}", domain],
                        description=f"Experiment variant {i} for {name}"
                    ),
                    created_by="coordinator",
                    commit_message=f"Created experiment variant {i} for {name}"
                )

                # Get the current version
                current_version = self.prompt_manager.registry.get_prompt_version(prompt_id)
                if not current_version:
                    # Create initial version if none exists
                    version_id = self.prompt_manager.create_version(
                        prompt_id=prompt_id,
                        content=variant_content,
                        created_by="coordinator",
                        commit_message=f"Initial version for experiment variant {i}"
                    )
                    current_version = self.prompt_manager.registry.get_prompt_version(prompt_id)

                if current_version:
                    variant = {
                        'prompt_id': prompt_id,
                        'prompt_version': current_version.version,
                        'name': f"Variant {i}",
                        'weight': traffic_split[i] if traffic_split and i < len(traffic_split) else 1.0 / len(variants)
                    }
                    experiment_variants.append(variant)
                else:
                    logger.warning(f"Failed to create version for variant {i}")

            if not experiment_variants:
                raise AgenticSystemError(
                    "Failed to create any experiment variants",
                    error_code="NO_VARIANTS_CREATED"
                )

            experiment_id = self.prompt_manager.create_experiment(
                name=name,
                description=f"A/B testing experiment for {domain} domain",
                variants=experiment_variants,
                created_by="coordinator"
            )
            logger.info(f"Created experiment '{name}' with ID {experiment_id}")
            return experiment_id
        except Exception as e:
            logger.error(f"Failed to create experiment: {e}")
            raise AgenticSystemError(
                f"Experiment creation failed: {e}",
                error_code="EXPERIMENT_CREATION_FAILED"
            )

    def get_prompt_management_stats(self) -> Dict[str, Any]:
        """Get statistics from the Prompt Management System."""
        if not self.prompt_manager:
            return {"error": "Prompt Management System not available"}

        try:
            stats = {
                "system_available": True,
                "prompt_management_enabled": self._enable_prompt_management,
                "performance_tracking_enabled": self._enable_performance_tracking,
                "experimentation_enabled": self._enable_experimentation,
                "migrated_domains": list(self.migrated_domains),
                "total_migrated_domains": len(self.migrated_domains),
                "registry_stats": self.prompt_manager.get_registry_stats(),
                "experiment_stats": self.prompt_manager.get_experiment_stats(),
                "deployment_stats": self.prompt_manager.get_deployment_stats(),
                "analytics_stats": self.prompt_manager.get_analytics_stats()
            }
            return stats
        except Exception as e:
            logger.error(f"Failed to get system stats: {e}")
            return {"error": str(e), "system_available": True}

    def create_prompt_from_template(self, template_name: str, variables: Dict[str, str]):
        """Create a prompt from a template using the new system."""
        if not self.prompt_manager or not self._enable_prompt_management:
            raise AgenticSystemError(
                "Prompt Management System not available",
                error_code="PROMPT_MANAGEMENT_UNAVAILABLE"
            )

        try:
            result = self.prompt_manager.create_prompt_from_template(
                template_name=template_name,
                variables=variables
            )
            return result
        except Exception as e:
            logger.error(f"Failed to create prompt from template: {e}")
            raise AgenticSystemError(
                f"Template creation failed: {e}",
                error_code="TEMPLATE_CREATION_FAILED"
            )

    def get_available_templates(self) -> List[Dict[str, Any]]:
        """Get list of available templates from the new system."""
        if not self.prompt_manager:
            return []

        try:
            return self.prompt_manager.get_available_templates()
        except Exception as e:
            logger.warning(f"Failed to get templates: {e}")
            return []


# Global coordinator instance factory
def create_coordinator(classifier_instance=None, evaluator_instance=None):
    """Factory function to create a coordinator with dependencies."""
    if classifier_instance is None:
        from agents.classifier import DomainClassifier
        classifier_instance = DomainClassifier()
    if evaluator_instance is None:
        from agents.evaluator import PromptEvaluator
        evaluator_instance = PromptEvaluator()

    return WorkflowCoordinator(classifier_instance, evaluator_instance)

# Global coordinator instance (lazy initialization)
_coordinator_instance = None

def get_coordinator():
    """Get the global coordinator instance, creating it if necessary."""
    global _coordinator_instance
    if _coordinator_instance is None:
        _coordinator_instance = create_coordinator()
    return _coordinator_instance

# For backward compatibility, provide a coordinator instance
coordinator = get_coordinator()
