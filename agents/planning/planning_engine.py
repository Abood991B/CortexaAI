"""Planning Engine with Advanced Reasoning and Task Decomposition."""

from typing import Dict, List, Optional, Any, Tuple, AsyncGenerator
import asyncio
import json
import hashlib
import time
from datetime import datetime
from enum import Enum
import logging

from config.config import (
    planning_config, get_logger, log_performance,
    cache_manager, generate_cache_key
)
from agents.memory import memory_manager

# Set up structured logging
logger = get_logger(__name__)


class PlanningStrategy(Enum):
    """Planning strategy enumeration."""
    BASIC = "basic"
    ADVANCED = "advanced"
    EXPERT = "expert"


class ReasoningStrategy(Enum):
    """Reasoning strategy enumeration."""
    LOGICAL = "logical"
    STEP_BY_STEP = "step_by_step"
    HYPOTHESIS_DRIVEN = "hypothesis_driven"
    TREE_OF_THOUGHT = "tree_of_thought"


class TaskComplexity(Enum):
    """Task complexity levels."""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    EXPERT = "expert"


class PlanningEngine:
    """Advanced planning engine with multiple reasoning strategies and task decomposition."""

    def __init__(self):
        """Initialize the planning engine."""
        self.active_plans = {}
        self.plan_templates = self._load_plan_templates()
        self.reasoning_strategies = self._initialize_reasoning_strategies()

        # Metrics
        self._plans_created = 0
        self._plans_executed = 0
        self._plans_successful = 0
        self._reasoning_steps = 0

    def _load_plan_templates(self) -> Dict[str, Dict[str, Any]]:
        """Load planning templates for different domains and task types."""
        return {
            'software_engineering': {
                'code_generation': {
                    'strategy': 'functional_decomposition',
                    'reasoning': 'step_by_step',
                    'steps': [
                        'Analyze requirements',
                        'Design algorithm/structure',
                        'Implement core functionality',
                        'Add error handling',
                        'Test and validate',
                        'Optimize performance'
                    ]
                },
                'api_design': {
                    'strategy': 'architectural_design',
                    'reasoning': 'hypothesis_driven',
                    'steps': [
                        'Define API requirements',
                        'Design resource endpoints',
                        'Specify data models',
                        'Plan authentication/authorization',
                        'Design error handling',
                        'Create documentation'
                    ]
                },
                'debugging': {
                    'strategy': 'systematic_analysis',
                    'reasoning': 'logical',
                    'steps': [
                        'Reproduce the issue',
                        'Gather diagnostic information',
                        'Identify root cause',
                        'Develop fix',
                        'Test the solution',
                        'Verify deployment'
                    ]
                },
                'refactoring': {
                    'strategy': 'structural_improvement',
                    'reasoning': 'tree_of_thought',
                    'steps': [
                        'Analyze current code structure',
                        'Identify improvement areas',
                        'Plan refactoring approach',
                        'Implement changes incrementally',
                        'Test after each change',
                        'Validate overall improvement'
                    ]
                }
            },
            'data_science': {
                'analysis': {
                    'strategy': 'analytical_workflow',
                    'reasoning': 'hypothesis_driven',
                    'steps': [
                        'Define research question',
                        'Explore and understand data',
                        'Clean and preprocess data',
                        'Perform analysis',
                        'Interpret results',
                        'Communicate findings'
                    ]
                },
                'modeling': {
                    'strategy': 'experimental_design',
                    'reasoning': 'step_by_step',
                    'steps': [
                        'Define problem and metrics',
                        'Prepare and explore data',
                        'Select and train models',
                        'Evaluate and compare models',
                        'Tune hyperparameters',
                        'Validate and deploy model'
                    ]
                },
                'visualization': {
                    'strategy': 'design_process',
                    'reasoning': 'logical',
                    'steps': [
                        'Understand data and audience',
                        'Choose appropriate visualization types',
                        'Design visual layout',
                        'Create initial visualizations',
                        'Refine based on feedback',
                        'Finalize and document'
                    ]
                }
            },
            'general': {
                'problem_solving': {
                    'strategy': 'systematic_approach',
                    'reasoning': 'logical',
                    'steps': [
                        'Define the problem clearly',
                        'Gather relevant information',
                        'Generate potential solutions',
                        'Evaluate and select best solution',
                        'Implement the solution',
                        'Evaluate results and iterate'
                    ]
                },
                'creative': {
                    'strategy': 'creative_process',
                    'reasoning': 'tree_of_thought',
                    'steps': [
                        'Define creative objectives',
                        'Research and gather inspiration',
                        'Brainstorm multiple concepts',
                        'Develop and refine ideas',
                        'Create prototype or draft',
                        'Iterate based on feedback'
                    ]
                },
                'analytical': {
                    'strategy': 'analytical_methodology',
                    'reasoning': 'hypothesis_driven',
                    'steps': [
                        'Define analytical objectives',
                        'Identify data sources and methods',
                        'Collect and organize data',
                        'Apply analytical techniques',
                        'Interpret and validate results',
                        'Report findings and recommendations'
                    ]
                }
            }
        }

    def _initialize_reasoning_strategies(self) -> Dict[str, Any]:
        """Initialize different reasoning strategies."""
        return {
            'logical': LogicalReasoning(),
            'step_by_step': StepByStepReasoning(),
            'hypothesis_driven': HypothesisDrivenReasoning(),
            'tree_of_thought': TreeOfThoughtReasoning()
        }

    async def create_plan(self, task: str, domain: str = "general",
                         user_id: str = None, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Create a comprehensive plan for a given task.

        Args:
            task: The task to plan for
            domain: The domain context
            user_id: User identifier for context
            context: Additional context information

        Returns:
            Complete plan with subtasks, reasoning, and execution strategy
        """
        plan_id = self._generate_plan_id(task, user_id)
        start_time = time.time()

        try:
            logger.info(f"Creating plan {plan_id} for task in domain {domain}")

            # Assess task complexity
            complexity = await self._assess_complexity(task, domain)

            # Select planning strategy
            strategy = self._select_strategy(complexity, domain)

            # Generate reasoning context
            reasoning_context = await self._generate_reasoning_context(task, domain, user_id)

            # Decompose task into subtasks
            subtasks = await self._decompose_task(task, domain, complexity, reasoning_context)

            # Generate execution strategy
            execution_strategy = await self._generate_execution_strategy(subtasks, domain, strategy)

            # Create contingency plans
            contingencies = await self._generate_contingencies(subtasks, domain)

            # Compile complete plan
            plan = {
                'plan_id': plan_id,
                'task': task,
                'domain': domain,
                'user_id': user_id,
                'complexity': complexity.value,
                'strategy': strategy.value,
                'subtasks': subtasks,
                'execution_strategy': execution_strategy,
                'contingencies': contingencies,
                'reasoning_context': reasoning_context,
                'metadata': {
                    'created_at': datetime.now().isoformat(),
                    'planning_time_seconds': time.time() - start_time,
                    'config_version': planning_config.planning_engine
                }
            }

            # Cache plan for fast retrieval
            if planning_config.enable_planning_caching:
                cache_manager.set(f"plan:{plan_id}", plan, planning_config.planning_cache_ttl)

            self.active_plans[plan_id] = plan
            self._plans_created += 1

            logger.info(f"Plan {plan_id} created successfully with {len(subtasks)} subtasks")
            return plan

        except Exception as e:
            logger.error(f"Failed to create plan {plan_id}: {e}")
            raise PlanningError(f"Plan creation failed: {str(e)}", plan_id=plan_id)

    async def execute_plan(self, plan: Dict[str, Any]) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Execute a plan step by step.

        Args:
            plan: The plan to execute

        Yields:
            Execution results for each step
        """
        plan_id = plan['plan_id']
        logger.info(f"Executing plan {plan_id}")

        execution_state = {
            'current_step': 0,
            'completed_steps': [],
            'failed_steps': [],
            'results': {},
            'start_time': time.time()
        }

        try:
            for i, subtask in enumerate(plan['subtasks']):
                execution_state['current_step'] = i

                # Execute subtask
                step_result = await self._execute_subtask(subtask, plan, execution_state)

                # Update execution state
                if step_result['success']:
                    execution_state['completed_steps'].append(subtask['id'])
                    execution_state['results'][subtask['id']] = step_result
                else:
                    execution_state['failed_steps'].append(subtask['id'])

                    # Check if we should use contingency
                    if planning_config.enable_adaptive_planning:
                        contingency_result = await self._apply_contingency(
                            subtask, plan, execution_state, step_result
                        )
                        if contingency_result:
                            step_result = contingency_result

                # Yield progress update
                yield {
                    'plan_id': plan_id,
                    'step': i + 1,
                    'total_steps': len(plan['subtasks']),
                    'subtask': subtask,
                    'result': step_result,
                    'execution_state': execution_state.copy(),
                    'progress': (i + 1) / len(plan['subtasks'])
                }

                # Check if we should adapt the plan
                if planning_config.enable_adaptive_planning and i < len(plan['subtasks']) - 1:
                    adaptation = await self._check_plan_adaptation(plan, execution_state)
                    if adaptation:
                        plan = adaptation
                        logger.info(f"Plan {plan_id} adapted based on execution results")

            # Final execution summary
            execution_time = time.time() - execution_state['start_time']
            success_rate = len(execution_state['completed_steps']) / len(plan['subtasks'])

            final_result = {
                'plan_id': plan_id,
                'status': 'completed' if success_rate >= planning_config.goal_success_threshold else 'partial',
                'execution_time': execution_time,
                'success_rate': success_rate,
                'completed_steps': len(execution_state['completed_steps']),
                'failed_steps': len(execution_state['failed_steps']),
                'results': execution_state['results']
            }

            self._plans_executed += 1
            if success_rate >= planning_config.goal_success_threshold:
                self._plans_successful += 1

            yield final_result

        except Exception as e:
            logger.error(f"Plan execution failed for {plan_id}: {e}")
            yield {
                'plan_id': plan_id,
                'status': 'failed',
                'error': str(e),
                'execution_state': execution_state
            }

    async def _assess_complexity(self, task: str, domain: str) -> TaskComplexity:
        """Assess the complexity of a task."""
        # Simple heuristic-based complexity assessment
        task_length = len(task)
        keywords_complex = ['design', 'architect', 'optimize', 'implement', 'system', 'complex']
        keywords_simple = ['simple', 'basic', 'quick', 'straightforward']

        complexity_score = 0

        # Length-based scoring
        if task_length > 500:
            complexity_score += 2
        elif task_length > 200:
            complexity_score += 1

        # Keyword-based scoring
        task_lower = task.lower()
        complex_count = sum(1 for keyword in keywords_complex if keyword in task_lower)
        simple_count = sum(1 for keyword in keywords_simple if keyword in task_lower)

        complexity_score += complex_count - simple_count

        # Domain-based adjustment
        if domain in ['software_engineering', 'data_science']:
            complexity_score += 1

        # Determine complexity level
        if complexity_score >= 3:
            return TaskComplexity.EXPERT
        elif complexity_score >= 2:
            return TaskComplexity.COMPLEX
        elif complexity_score >= 1:
            return TaskComplexity.MODERATE
        else:
            return TaskComplexity.SIMPLE

    def _select_strategy(self, complexity: TaskComplexity, domain: str) -> PlanningStrategy:
        """Select the appropriate planning strategy."""
        if planning_config.planning_engine == 'expert' or complexity == TaskComplexity.EXPERT:
            return PlanningStrategy.EXPERT
        elif planning_config.planning_engine == 'advanced' or complexity == TaskComplexity.COMPLEX:
            return PlanningStrategy.ADVANCED
        else:
            return PlanningStrategy.BASIC

    async def _generate_reasoning_context(self, task: str, domain: str, user_id: str = None) -> Dict[str, Any]:
        """Generate reasoning context using available information."""
        context = {
            'task': task,
            'domain': domain,
            'timestamp': datetime.now().isoformat(),
            'reasoning_approach': planning_config.domain_planning_templates.get(
                domain, planning_config.domain_planning_templates['general']
            )['reasoning_approach']
        }

        # Add memory context if available
        if user_id and memory_manager:
            try:
                memory_context = await memory_manager.generate_rag_context(user_id, domain, task)
                context['memory_context'] = memory_context
            except Exception as e:
                logger.warning(f"Failed to get memory context: {e}")

        return context

    async def _decompose_task(self, task: str, domain: str, complexity: TaskComplexity,
                            reasoning_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Decompose a complex task into manageable subtasks."""
        if not planning_config.enable_task_decomposition:
            return [{
                'id': 'task_1',
                'description': task,
                'type': 'primary',
                'dependencies': [],
                'estimated_complexity': complexity.value,
                'success_criteria': ['Task completed successfully']
            }]

        # Use reasoning strategy to decompose task
        reasoning_strategy = self.reasoning_strategies[reasoning_context['reasoning_approach']]

        decomposition = await reasoning_strategy.decompose_task(
            task=task,
            domain=domain,
            max_subtasks=planning_config.max_subtasks,
            context=reasoning_context
        )

        # Assign IDs and dependencies
        for i, subtask in enumerate(decomposition):
            subtask['id'] = f"subtask_{i+1}"
            subtask['dependencies'] = []
            subtask['estimated_complexity'] = complexity.value

        # Add dependency relationships (simple linear for now)
        for i in range(1, len(decomposition)):
            decomposition[i]['dependencies'].append(decomposition[i-1]['id'])

        return decomposition

    async def _generate_execution_strategy(self, subtasks: List[Dict[str, Any]],
                                         domain: str, strategy: PlanningStrategy) -> Dict[str, Any]:
        """Generate execution strategy for the subtasks."""
        return {
            'strategy': strategy.value,
            'execution_order': [subtask['id'] for subtask in subtasks],
            'parallel_execution': False,  # Sequential for now
            'error_handling': 'fail_fast',  # Can be 'continue' or 'retry'
            'validation_points': [len(subtasks) // 2, len(subtasks)],  # Mid and end validation
            'monitoring': {
                'track_progress': True,
                'measure_quality': True,
                'adapt_on_failure': planning_config.enable_adaptive_planning
            }
        }

    async def _generate_contingencies(self, subtasks: List[Dict[str, Any]], domain: str) -> List[Dict[str, Any]]:
        """Generate contingency plans for potential failures."""
        contingencies = []

        for subtask in subtasks:
            contingency = {
                'subtask_id': subtask['id'],
                'trigger_conditions': ['failure', 'timeout', 'low_quality'],
                'actions': [
                    {
                        'type': 'retry',
                        'max_attempts': 2,
                        'backoff_seconds': 5
                    },
                    {
                        'type': 'alternative_approach',
                        'description': f"Use alternative method for {subtask['description']}"
                    },
                    {
                        'type': 'skip',
                        'condition': 'non_critical',
                        'compensation': 'Adjust subsequent steps'
                    }
                ]
            }
            contingencies.append(contingency)

        return contingencies

    async def _execute_subtask(self, subtask: Dict[str, Any], plan: Dict[str, Any],
                             execution_state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single subtask."""
        subtask_id = subtask['id']
        logger.info(f"Executing subtask {subtask_id}: {subtask['description']}")

        try:
            # Simulate subtask execution (in real implementation, this would call appropriate agents)
            # For now, return mock success
            result = {
                'subtask_id': subtask_id,
                'success': True,
                'output': f"Completed: {subtask['description']}",
                'quality_score': 0.9,
                'execution_time': 2.0,
                'metadata': {'method': 'simulated'}
            }

            self._reasoning_steps += 1
            return result

        except Exception as e:
            logger.error(f"Subtask {subtask_id} execution failed: {e}")
            return {
                'subtask_id': subtask_id,
                'success': False,
                'error': str(e),
                'execution_time': 0.5,
                'metadata': {'failure_reason': 'execution_error'}
            }

    async def _apply_contingency(self, subtask: Dict[str, Any], plan: Dict[str, Any],
                               execution_state: Dict[str, Any], failure_result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Apply contingency plan for failed subtask."""
        contingency = next(
            (c for c in plan['contingencies'] if c['subtask_id'] == subtask['id']),
            None
        )

        if not contingency:
            return None

        logger.info(f"Applying contingency for subtask {subtask['id']}")

        # Try alternative approach
        alternative_result = {
            'subtask_id': subtask['id'],
            'success': True,
            'output': f"Alternative approach: {subtask['description']}",
            'quality_score': 0.8,
            'execution_time': 3.0,
            'metadata': {'method': 'contingency_applied'}
        }

        return alternative_result

    async def _check_plan_adaptation(self, plan: Dict[str, Any], execution_state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Check if plan needs adaptation based on execution results."""
        if not planning_config.enable_adaptive_planning:
            return None

        # Simple adaptation logic
        success_rate = len(execution_state['completed_steps']) / (len(execution_state['completed_steps']) + len(execution_state['failed_steps']))

        if success_rate < planning_config.adaptation_threshold:
            logger.info(f"Plan adaptation triggered for {plan['plan_id']} (success rate: {success_rate})")

            # Create adapted plan with simplified approach
            adapted_plan = plan.copy()
            adapted_plan['metadata']['adapted'] = True
            adapted_plan['metadata']['adaptation_reason'] = 'low_success_rate'

            return adapted_plan

        return None

    def _generate_plan_id(self, task: str, user_id: str = None) -> str:
        """Generate a unique plan ID."""
        content = f"{task}:{user_id}:{time.time()}"
        return f"plan_{hashlib.md5(content.encode()).hexdigest()[:8]}"

    def get_metrics(self) -> Dict[str, Any]:
        """Get planning engine metrics."""
        total_plans = self._plans_created
        success_rate = self._plans_successful / max(total_plans, 1)

        return {
            'plans_created': self._plans_created,
            'plans_executed': self._plans_executed,
            'plans_successful': self._plans_successful,
            'success_rate': success_rate,
            'reasoning_steps': self._reasoning_steps,
            'active_plans': len(self.active_plans),
            'average_plan_complexity': 'medium',  # Would be calculated from actual data
            'most_used_strategy': planning_config.planning_engine
        }


# Reasoning Strategy Implementations
class BaseReasoningStrategy:
    """Base class for reasoning strategies."""

    async def decompose_task(self, task: str, domain: str, max_subtasks: int,
                           context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Decompose a task into subtasks."""
        raise NotImplementedError


class LogicalReasoning(BaseReasoningStrategy):
    """Logical reasoning strategy."""

    async def decompose_task(self, task: str, domain: str, max_subtasks: int,
                           context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Decompose task using logical reasoning."""
        # Simple logical decomposition
        return [
            {
                'description': f'Analyze {task}',
                'type': 'analysis',
                'estimated_effort': 'low'
            },
            {
                'description': f'Plan approach for {task}',
                'type': 'planning',
                'estimated_effort': 'medium'
            },
            {
                'description': f'Execute {task}',
                'type': 'execution',
                'estimated_effort': 'high'
            },
            {
                'description': f'Validate results of {task}',
                'type': 'validation',
                'estimated_effort': 'medium'
            }
        ]


class StepByStepReasoning(BaseReasoningStrategy):
    """Step-by-step reasoning strategy."""

    async def decompose_task(self, task: str, domain: str, max_subtasks: int,
                           context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Decompose task into sequential steps."""
        # More granular step-by-step approach
        steps = [
            {
                'description': f'Understand requirements for {task}',
                'type': 'requirements',
                'estimated_effort': 'low'
            },
            {
                'description': f'Break down {task} into components',
                'type': 'decomposition',
                'estimated_effort': 'medium'
            },
            {
                'description': f'Design solution for {task}',
                'type': 'design',
                'estimated_effort': 'medium'
            },
            {
                'description': f'Implement first component of {task}',
                'type': 'implementation',
                'estimated_effort': 'high'
            },
            {
                'description': f'Test implementation of {task}',
                'type': 'testing',
                'estimated_effort': 'medium'
            },
            {
                'description': f'Complete remaining components of {task}',
                'type': 'completion',
                'estimated_effort': 'high'
            }
        ]

        return steps[:max_subtasks]

    def _get_code_generation_template(self) -> Dict[str, Any]:
        """Get template for code generation tasks."""
        return {
            'strategy': 'functional_decomposition',
            'reasoning': 'step_by_step',
            'steps': [
                'Analyze requirements',
                'Design algorithm/structure',
                'Implement core functionality',
                'Add error handling',
                'Test and validate',
                'Optimize performance'
            ]
        }

    def _get_api_design_template(self) -> Dict[str, Any]:
        """Get template for API design tasks."""
        return {
            'strategy': 'architectural_design',
            'reasoning': 'hypothesis_driven',
            'steps': [
                'Define API requirements',
                'Design resource endpoints',
                'Specify data models',
                'Plan authentication/authorization',
                'Design error handling',
                'Create documentation'
            ]
        }

    def _get_debugging_template(self) -> Dict[str, Any]:
        """Get template for debugging tasks."""
        return {
            'strategy': 'systematic_analysis',
            'reasoning': 'logical',
            'steps': [
                'Reproduce the issue',
                'Gather diagnostic information',
                'Identify root cause',
                'Develop fix',
                'Test the solution',
                'Verify deployment'
            ]
        }

    def _get_refactoring_template(self) -> Dict[str, Any]:
        """Get template for code refactoring tasks."""
        return {
            'strategy': 'structural_improvement',
            'reasoning': 'tree_of_thought',
            'steps': [
                'Analyze current code structure',
                'Identify improvement areas',
                'Plan refactoring approach',
                'Implement changes incrementally',
                'Test after each change',
                'Validate overall improvement'
            ]
        }

    def _get_data_analysis_template(self) -> Dict[str, Any]:
        """Get template for data analysis tasks."""
        return {
            'strategy': 'analytical_workflow',
            'reasoning': 'hypothesis_driven',
            'steps': [
                'Define research question',
                'Explore and understand data',
                'Clean and preprocess data',
                'Perform analysis',
                'Interpret results',
                'Communicate findings'
            ]
        }

    def _get_modeling_template(self) -> Dict[str, Any]:
        """Get template for machine learning modeling tasks."""
        return {
            'strategy': 'experimental_design',
            'reasoning': 'step_by_step',
            'steps': [
                'Define problem and metrics',
                'Prepare and explore data',
                'Select and train models',
                'Evaluate and compare models',
                'Tune hyperparameters',
                'Validate and deploy model'
            ]
        }

    def _get_visualization_template(self) -> Dict[str, Any]:
        """Get template for data visualization tasks."""
        return {
            'strategy': 'design_process',
            'reasoning': 'logical',
            'steps': [
                'Understand data and audience',
                'Choose appropriate visualization types',
                'Design visual layout',
                'Create initial visualizations',
                'Refine based on feedback',
                'Finalize and document'
            ]
        }

    def _get_general_problem_template(self) -> Dict[str, Any]:
        """Get template for general problem-solving tasks."""
        return {
            'strategy': 'systematic_approach',
            'reasoning': 'logical',
            'steps': [
                'Define the problem clearly',
                'Gather relevant information',
                'Generate potential solutions',
                'Evaluate and select best solution',
                'Implement the solution',
                'Evaluate results and iterate'
            ]
        }

    def _get_creative_template(self) -> Dict[str, Any]:
        """Get template for creative tasks."""
        return {
            'strategy': 'creative_process',
            'reasoning': 'tree_of_thought',
            'steps': [
                'Define creative objectives',
                'Research and gather inspiration',
                'Brainstorm multiple concepts',
                'Develop and refine ideas',
                'Create prototype or draft',
                'Iterate based on feedback'
            ]
        }

    def _get_analytical_template(self) -> Dict[str, Any]:
        """Get template for analytical tasks."""
        return {
            'strategy': 'analytical_methodology',
            'reasoning': 'hypothesis_driven',
            'steps': [
                'Define analytical objectives',
                'Identify data sources and methods',
                'Collect and organize data',
                'Apply analytical techniques',
                'Interpret and validate results',
                'Report findings and recommendations'
            ]
        }


class HypothesisDrivenReasoning(BaseReasoningStrategy):
    """Hypothesis-driven reasoning strategy."""

    async def decompose_task(self, task: str, domain: str, max_subtasks: int,
                           context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Decompose task using hypothesis-driven approach."""
        return [
            {
                'description': f'Formulate hypothesis for {task}',
                'type': 'hypothesis',
                'estimated_effort': 'medium'
            },
            {
                'description': f'Design experiment to test hypothesis for {task}',
                'type': 'experiment_design',
                'estimated_effort': 'medium'
            },
            {
                'description': f'Execute experiment for {task}',
                'type': 'execution',
                'estimated_effort': 'high'
            },
            {
                'description': f'Analyze results of {task}',
                'type': 'analysis',
                'estimated_effort': 'medium'
            },
            {
                'description': f'Draw conclusions for {task}',
                'type': 'conclusion',
                'estimated_effort': 'low'
            }
        ]


class TreeOfThoughtReasoning(BaseReasoningStrategy):
    """Tree-of-thought reasoning strategy."""

    async def decompose_task(self, task: str, domain: str, max_subtasks: int,
                           context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Decompose task using tree-of-thought approach."""
        return [
            {
                'description': f'Generate multiple approaches for {task}',
                'type': 'brainstorming',
                'estimated_effort': 'medium'
            },
            {
                'description': f'Evaluate different approaches for {task}',
                'type': 'evaluation',
                'estimated_effort': 'high'
            },
            {
                'description': f'Select best approach for {task}',
                'type': 'selection',
                'estimated_effort': 'low'
            },
            {
                'description': f'Implement selected approach for {task}',
                'type': 'implementation',
                'estimated_effort': 'high'
            },
            {
                'description': f'Explore alternative branches if needed for {task}',
                'type': 'exploration',
                'estimated_effort': 'medium'
            }
        ]


class PlanningError(Exception):
    """Exception raised for planning-related errors."""

    def __init__(self, message: str, plan_id: str = None):
        super().__init__(message)
        self.plan_id = plan_id


# Global planning engine instance
planning_engine = PlanningEngine()
