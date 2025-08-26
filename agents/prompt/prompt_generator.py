"""Advanced Prompt Generator with Multiple Generation Strategies."""

from typing import Dict, List, Optional, Any, Tuple, AsyncGenerator
import asyncio
import json
import hashlib
import time
from datetime import datetime
from enum import Enum
import logging
import random
import re

from config.config import (
    prompt_generation_config, get_logger, cache_manager,
    generate_cache_key, security_manager, security_config
)

# Set up structured logging
logger = get_logger(__name__)


class GenerationStrategy(Enum):
    """Prompt generation strategy enumeration."""
    BASIC = "basic"
    TEMPLATE_BASED = "template_based"
    META_PROMPTING = "meta_prompting"
    CHAIN_OF_PROMPTS = "chain_of_prompts"
    CONTEXTUAL_INJECTION = "contextual_injection"
    PERSONA_BASED = "persona_based"
    HYBRID = "hybrid"


class OptimizationAlgorithm(Enum):
    """Prompt optimization algorithm enumeration."""
    EVOLUTIONARY = "evolutionary"
    GRADIENT_BASED = "gradient_based"
    REINFORCEMENT_LEARNING = "reinforcement_learning"
    AB_TESTING = "ab_testing"


class PromptGenerator:
    """Advanced prompt generator with multiple strategies and optimization algorithms."""

    def __init__(self):
        """Initialize the prompt generator."""
        self.template_library = self._load_template_library()
        self.persona_library = self._load_persona_library()
        self.meta_prompts = self._load_meta_prompts()
        self.generation_strategies = self._initialize_strategies()
        self.optimization_algorithms = self._initialize_optimization_algorithms()

        # Performance tracking
        self._generated_count = 0
        self._optimized_count = 0
        self._quality_scores = []

    def _load_template_library(self) -> Dict[str, Dict[str, Any]]:
        """Load comprehensive template library."""
        return {
            'software_engineering': {
                'code_generation': {
                    'template': """
You are a senior {domain} developer. Generate {output_type} for the following requirements:

Requirements:
{requirements}

Context:
{context}

Constraints:
{constraints}

Best Practices:
- Follow {domain} coding standards
- Include proper error handling
- Add comprehensive documentation
- Ensure performance optimization

Generate the {output_type}:
""",
                    'variables': ['domain', 'output_type', 'requirements', 'context', 'constraints']
                },
                'api_design': {
                    'template': """
Design a REST API for the following functionality:

Functionality: {functionality}
User Requirements: {requirements}
Technical Constraints: {constraints}

API Design Requirements:
- RESTful principles
- Clear resource naming
- Proper HTTP methods
- Comprehensive error responses
- Authentication/authorization
- Rate limiting considerations

Design the API specification:
""",
                    'variables': ['functionality', 'requirements', 'constraints']
                },
                'debugging': {
                    'template': """
Debug the following issue:

Issue Description: {issue_description}
Error Messages: {error_messages}
Code Context: {code_context}
Expected Behavior: {expected_behavior}
Actual Behavior: {actual_behavior}

Debugging Approach:
1. Analyze the error pattern
2. Identify root cause
3. Propose solution
4. Test the fix
5. Prevent recurrence

Provide debugging analysis:
""",
                    'variables': ['issue_description', 'error_messages', 'code_context', 'expected_behavior', 'actual_behavior']
                }
            },
            'data_science': {
                'analysis': {
                    'template': """
Perform data analysis on the following dataset and requirements:

Dataset: {dataset_description}
Analysis Objectives: {objectives}
Key Questions: {questions}
Available Tools: {tools}

Analysis Framework:
1. Data exploration and understanding
2. Statistical analysis
3. Visualization and insights
4. Recommendations

Provide comprehensive analysis:
""",
                    'variables': ['dataset_description', 'objectives', 'questions', 'tools']
                },
                'modeling': {
                    'template': """
Build a machine learning model for the following problem:

Problem: {problem_statement}
Data: {data_description}
Success Metrics: {metrics}
Constraints: {constraints}

Modeling Approach:
- Problem type identification
- Feature engineering
- Model selection and training
- Evaluation and validation
- Deployment considerations

Develop the modeling solution:
""",
                    'variables': ['problem_statement', 'data_description', 'metrics', 'constraints']
                }
            },
            'creative': {
                'content_creation': {
                    'template': """
Create {content_type} for the following brief:

Topic: {topic}
Target Audience: {audience}
Style: {style}
Length: {length}
Key Messages: {messages}

Content Requirements:
- Engaging and compelling
- Clear value proposition
- Call to action
- Brand consistency

Create the content:
""",
                    'variables': ['content_type', 'topic', 'audience', 'style', 'length', 'messages']
                }
            },
            'general': {
                'problem_solving': {
                    'template': """
Solve the following problem:

Problem: {problem_statement}
Context: {context}
Constraints: {constraints}
Available Resources: {resources}

Solution Approach:
1. Problem analysis
2. Solution design
3. Implementation steps
4. Validation and testing
5. Documentation

Provide complete solution:
""",
                    'variables': ['problem_statement', 'context', 'constraints', 'resources']
                }
            }
        }

    def _load_persona_library(self) -> Dict[str, Dict[str, Any]]:
        """Load comprehensive persona library."""
        return {
            'software_engineering': {
                'senior_developer': {
                    'role': 'Senior Software Engineer',
                    'expertise': '10+ years of full-stack development',
                    'style': 'technical, precise, practical',
                    'values': 'code quality, scalability, maintainability'
                },
                'architect': {
                    'role': 'Software Architect',
                    'expertise': 'system design and architecture patterns',
                    'style': 'strategic, comprehensive, forward-thinking',
                    'values': 'scalability, reliability, technical excellence'
                },
                'tech_lead': {
                    'role': 'Technical Lead',
                    'expertise': 'team leadership and technical direction',
                    'style': 'collaborative, mentoring, pragmatic',
                    'values': 'team success, code quality, knowledge sharing'
                }
            },
            'data_science': {
                'principal_data_scientist': {
                    'role': 'Principal Data Scientist',
                    'expertise': 'advanced ML, statistics, and data engineering',
                    'style': 'analytical, rigorous, innovative',
                    'values': 'data-driven insights, model accuracy, business impact'
                },
                'data_engineer': {
                    'role': 'Data Engineer',
                    'expertise': 'data pipelines, ETL, and big data technologies',
                    'style': 'systematic, efficient, scalable',
                    'values': 'data quality, performance, reliability'
                }
            },
            'creative': {
                'creative_director': {
                    'role': 'Creative Director',
                    'expertise': 'brand strategy, content creation, and design',
                    'style': 'inspiring, innovative, audience-focused',
                    'values': 'creativity, brand consistency, audience engagement'
                }
            }
        }

    def _load_meta_prompts(self) -> Dict[str, str]:
        """Load meta-prompting templates."""
        return {
            'self_improvement': """
Analyze this prompt and suggest improvements:

Original Prompt: {original_prompt}

Improvement Criteria:
1. Clarity and specificity
2. Context completeness
3. Task decomposition
4. Success criteria definition
5. Error handling considerations

Suggest improvements:
""",
            'prompt_engineer': """
You are an expert prompt engineer. Optimize this prompt for better results:

Current Prompt: {current_prompt}
Target Task: {target_task}
Expected Outcome: {expected_outcome}

Optimization Focus:
- Task clarity and specificity
- Context and constraints
- Output format definition
- Error handling and edge cases
- Performance considerations

Provide optimized prompt:
""",
            'quality_assessment': """
Evaluate the quality of this generated prompt:

Prompt: {generated_prompt}
Original Task: {original_task}
Target Audience: {target_audience}

Quality Dimensions:
1. Clarity and comprehensibility
2. Task specificity
3. Contextual completeness
4. Output structure definition
5. Error handling adequacy

Provide quality assessment:
"""
        }

    def _initialize_strategies(self) -> Dict[str, Any]:
        """Initialize generation strategies."""
        return {
            'basic': BasicGenerationStrategy(),
            'template_based': TemplateBasedStrategy(),
            'meta_prompting': MetaPromptingStrategy(),
            'chain_of_prompts': ChainOfPromptsStrategy(),
            'contextual_injection': ContextualInjectionStrategy(),
            'persona_based': PersonaBasedStrategy(),
            'hybrid': HybridStrategy()
        }

    def _initialize_optimization_algorithms(self) -> Dict[str, Any]:
        """Initialize optimization algorithms."""
        return {
            'evolutionary': EvolutionaryOptimizer(),
            'gradient_based': GradientBasedOptimizer(),
            'reinforcement_learning': ReinforcementLearningOptimizer(),
            'ab_testing': ABTestingOptimizer()
        }

    async def generate_prompt(self, task: str, domain: str = "general",
                            strategy: str = "intelligent", context: Dict[str, Any] = None,
                            persona: str = None, constraints: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Generate an optimized prompt using the specified strategy.

        Args:
            task: The task to generate a prompt for
            domain: The domain context
            strategy: Generation strategy to use
            context: Additional context information
            persona: Persona to use for generation
            constraints: Generation constraints

        Returns:
            Generated prompt with metadata
        """
        start_time = time.time()
        generation_id = self._generate_generation_id(task, domain)

        try:
            logger.info(f"Generating prompt {generation_id} for task in domain {domain}")

            # Select generation strategy
            if strategy == "intelligent":
                strategy = self._select_intelligent_strategy(task, domain)

            generation_strategy = self.generation_strategies[strategy]

            # Generate base prompt
            generated_prompt = await generation_strategy.generate(
                task=task,
                domain=domain,
                context=context or {},
                persona=persona,
                constraints=constraints or {},
                template_library=self.template_library,
                persona_library=self.persona_library
            )

            # Apply security validation
            if security_config.enable_input_sanitization:
                security_result = security_manager.sanitize_input(generated_prompt, "prompt_generation")
                if not security_result['is_safe']:
                    logger.warning(f"Generated prompt failed security validation: {generation_id}")
                    generated_prompt = security_result['sanitized_text']

            # Calculate quality score
            quality_score = self._assess_prompt_quality(generated_prompt, task, domain)

            result = {
                'generation_id': generation_id,
                'generated_prompt': generated_prompt,
                'strategy_used': strategy,
                'domain': domain,
                'persona': persona,
                'quality_score': quality_score,
                'generation_time': time.time() - start_time,
                'metadata': {
                    'created_at': datetime.now().isoformat(),
                    'task_complexity': self._assess_task_complexity(task),
                    'template_used': getattr(generation_strategy, 'template_used', None),
                    'optimization_applied': False
                }
            }

            # Cache successful generations
            if prompt_generation_config.enable_generation_caching:
                cache_key = generate_cache_key(f"gen:{task}:{domain}:{strategy}")
                cache_manager.set(cache_key, result, prompt_generation_config.generation_cache_ttl)

            self._generated_count += 1
            self._quality_scores.append(quality_score)

            logger.info(f"Prompt {generation_id} generated successfully with quality score {quality_score:.2f}")
            return result

        except Exception as e:
            logger.error(f"Failed to generate prompt {generation_id}: {e}")
            raise PromptGenerationError(f"Prompt generation failed: {str(e)}", generation_id=generation_id)

    async def optimize_prompt(self, prompt: str, task: str, domain: str = "general",
                            algorithm: str = "evolutionary", iterations: int = None,
                            context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Optimize an existing prompt using advanced algorithms.

        Args:
            prompt: The prompt to optimize
            task: The task the prompt is for
            domain: The domain context
            algorithm: Optimization algorithm to use
            iterations: Number of optimization iterations
            context: Additional context

        Returns:
            Optimized prompt with improvement metrics
        """
        start_time = time.time()
        optimization_id = self._generate_optimization_id(prompt, task)

        try:
            logger.info(f"Optimizing prompt {optimization_id} using {algorithm}")

            # Select optimization algorithm
            if algorithm == "auto":
                algorithm = self._select_optimization_algorithm(prompt, task, domain)

            optimizer = self.optimization_algorithms[algorithm]

            # Set iterations
            max_iterations = iterations or prompt_generation_config.max_optimization_iterations

            # Run optimization
            optimization_result = await optimizer.optimize(
                prompt=prompt,
                task=task,
                domain=domain,
                max_iterations=max_iterations,
                context=context or {},
                quality_threshold=prompt_generation_config.optimization_convergence_threshold
            )

            optimized_prompt = optimization_result['optimized_prompt']
            improvement_score = optimization_result['improvement_score']
            iterations_used = optimization_result['iterations_used']

            # Calculate final quality
            final_quality = self._assess_prompt_quality(optimized_prompt, task, domain)

            result = {
                'optimization_id': optimization_id,
                'original_prompt': prompt,
                'optimized_prompt': optimized_prompt,
                'algorithm_used': algorithm,
                'improvement_score': improvement_score,
                'final_quality_score': final_quality,
                'iterations_used': iterations_used,
                'optimization_time': time.time() - start_time,
                'metadata': {
                    'created_at': datetime.now().isoformat(),
                    'domain': domain,
                    'task': task,
                    'converged': iterations_used < max_iterations
                }
            }

            self._optimized_count += 1

            logger.info(f"Prompt {optimization_id} optimized with improvement score {improvement_score:.2f}")
            return result

        except Exception as e:
            logger.error(f"Failed to optimize prompt {optimization_id}: {e}")
            raise PromptOptimizationError(f"Prompt optimization failed: {str(e)}", optimization_id=optimization_id)

    async def generate_and_optimize(self, task: str, domain: str = "general",
                                  generation_strategy: str = "intelligent",
                                  optimization_algorithm: str = "auto",
                                  context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Generate and then optimize a prompt in one step.

        Args:
            task: The task to generate a prompt for
            domain: The domain context
            generation_strategy: Strategy for initial generation
            optimization_algorithm: Algorithm for optimization
            context: Additional context

        Returns:
            Generated and optimized prompt with full metadata
        """
        # Generate initial prompt
        generation_result = await self.generate_prompt(
            task=task,
            domain=domain,
            strategy=generation_strategy,
            context=context
        )

        # Optimize the generated prompt
        optimization_result = await self.optimize_prompt(
            prompt=generation_result['generated_prompt'],
            task=task,
            domain=domain,
            algorithm=optimization_algorithm,
            context=context
        )

        # Combine results
        combined_result = {
            'task': task,
            'domain': domain,
            'generation': generation_result,
            'optimization': optimization_result,
            'final_prompt': optimization_result['optimized_prompt'],
            'overall_quality_score': optimization_result['final_quality_score'],
            'total_time': generation_result['generation_time'] + optimization_result['optimization_time'],
            'metadata': {
                'generation_strategy': generation_result['strategy_used'],
                'optimization_algorithm': optimization_result['algorithm_used'],
                'created_at': datetime.now().isoformat()
            }
        }

        return combined_result

    def _select_intelligent_strategy(self, task: str, domain: str) -> str:
        """Intelligently select the best generation strategy."""
        task_complexity = self._assess_task_complexity(task)

        if task_complexity >= 0.8:  # High complexity
            return "hybrid"
        elif domain in self.template_library:
            return "template_based"
        elif task_complexity >= 0.6:  # Medium complexity
            return "chain_of_prompts"
        else:
            return "contextual_injection"

    def _select_optimization_algorithm(self, prompt: str, task: str, domain: str) -> str:
        """Select the best optimization algorithm."""
        prompt_length = len(prompt)
        current_quality = self._assess_prompt_quality(prompt, task, domain)

        if current_quality < 0.6:  # Low quality
            return "evolutionary"  # More thorough optimization
        elif prompt_length > 1000:  # Long prompts
            return "gradient_based"  # Faster convergence
        else:
            return "ab_testing"  # Comparative optimization

    def _assess_task_complexity(self, task: str) -> float:
        """Assess the complexity of a task on a 0-1 scale."""
        complexity_indicators = [
            len(task),  # Length
            task.count('and'),  # Multiple requirements
            task.count('or'),   # Multiple options
            task.count('with'), # Additional constraints
            task.count('but'),  # Complications
            task.count('however'),  # Complications
        ]

        # Normalize to 0-1 scale
        complexity_score = min(sum(complexity_indicators) / 50, 1.0)
        return complexity_score

    def _assess_prompt_quality(self, prompt: str, task: str, domain: str) -> float:
        """Assess the quality of a generated prompt."""
        quality_score = 0.0

        # Length appropriateness (20%)
        optimal_length = 200 + (len(task) * 2)
        length_ratio = min(len(prompt) / optimal_length, 2.0)
        length_score = 1.0 - abs(1.0 - length_ratio)
        quality_score += length_score * 0.2

        # Clarity indicators (30%)
        clarity_indicators = [
            prompt.count('?'),  # Questions for clarification
            prompt.count(':' ),  # Structure indicators
            prompt.count('- '),  # List items
            prompt.count('Example'),  # Examples provided
            prompt.count('Note:'),  # Additional notes
        ]
        clarity_score = min(sum(clarity_indicators) / 10, 1.0)
        quality_score += clarity_score * 0.3

        # Specificity (25%)
        specific_terms = ['specific', 'exactly', 'precisely', 'clearly', 'must', 'should']
        specificity_score = sum(1 for term in specific_terms if term in prompt.lower())
        specificity_score = min(specificity_score / 3, 1.0)
        quality_score += specificity_score * 0.25

        # Context completeness (25%)
        context_indicators = [
            'context' in prompt.lower(),
            'background' in prompt.lower(),
            'requirements' in prompt.lower(),
            'constraints' in prompt.lower(),
            'examples' in prompt.lower(),
        ]
        context_score = sum(context_indicators) / len(context_indicators)
        quality_score += context_score * 0.25

        return min(quality_score, 1.0)

    def _generate_generation_id(self, task: str, domain: str) -> str:
        """Generate a unique generation ID."""
        content = f"{task}:{domain}:{time.time()}"
        return f"gen_{hashlib.md5(content.encode()).hexdigest()[:8]}"

    def _generate_optimization_id(self, prompt: str, task: str) -> str:
        """Generate a unique optimization ID."""
        content = f"{prompt[:50]}:{task}:{time.time()}"
        return f"opt_{hashlib.md5(content.encode()).hexdigest()[:8]}"

    def get_metrics(self) -> Dict[str, Any]:
        """Get prompt generation metrics."""
        total_generations = self._generated_count
        avg_quality = sum(self._quality_scores) / max(len(self._quality_scores), 1)

        return {
            'prompts_generated': self._generated_count,
            'prompts_optimized': self._optimized_count,
            'average_quality_score': avg_quality,
            'quality_distribution': {
                'high': len([s for s in self._quality_scores if s >= 0.8]),
                'medium': len([s for s in self._quality_scores if 0.6 <= s < 0.8]),
                'low': len([s for s in self._quality_scores if s < 0.6])
            },
            'template_usage': self._calculate_template_usage(),
            'strategy_effectiveness': self._calculate_strategy_effectiveness()
        }

    def _calculate_template_usage(self) -> Dict[str, int]:
        """Calculate template usage statistics."""
        # This would track actual template usage in a real implementation
        return {
            'software_engineering': 0,
            'data_science': 0,
            'creative': 0,
            'general': 0
        }

    def _calculate_strategy_effectiveness(self) -> Dict[str, float]:
        """Calculate strategy effectiveness."""
        # This would track actual strategy performance in a real implementation
        return {
            'basic': 0.7,
            'template_based': 0.8,
            'meta_prompting': 0.85,
            'chain_of_prompts': 0.9,
            'contextual_injection': 0.82,
            'persona_based': 0.88,
            'hybrid': 0.92
        }


# Generation Strategy Implementations
class BaseGenerationStrategy:
    """Base class for prompt generation strategies."""

    async def generate(self, task: str, domain: str, context: Dict[str, Any],
                      persona: str = None, constraints: Dict[str, Any] = None,
                      template_library: Dict = None, persona_library: Dict = None) -> str:
        """Generate a prompt using this strategy."""
        raise NotImplementedError


class BasicGenerationStrategy(BaseGenerationStrategy):
    """Simple basic generation strategy."""

    async def generate(self, task: str, domain: str, context: Dict[str, Any],
                      persona: str = None, constraints: Dict[str, Any] = None,
                      template_library: Dict = None, persona_library: Dict = None) -> str:
        """Generate a basic prompt."""
        return f"Please help with the following task: {task}"


class TemplateBasedStrategy(BaseGenerationStrategy):
    """Template-based generation strategy."""

    async def generate(self, task: str, domain: str, context: Dict[str, Any],
                      persona: str = None, constraints: Dict[str, Any] = None,
                      template_library: Dict = None, persona_library: Dict = None) -> str:
        """Generate using templates."""
        if not template_library or domain not in template_library:
            return f"Please help with the following {domain} task: {task}"

        # Select appropriate template
        domain_templates = template_library[domain]
        best_template = self._select_best_template(task, domain_templates)

        if not best_template:
            return f"Please help with the following {domain} task: {task}"

        # Fill template variables
        template_data = best_template['template']
        variables = best_template['variables']

        # Extract or generate variable values
        variable_values = {}
        for var in variables:
            if var in context:
                variable_values[var] = context[var]
            else:
                variable_values[var] = self._generate_variable_value(var, task, domain)

        # Fill template
        filled_prompt = template_data
        for var, value in variable_values.items():
            filled_prompt = filled_prompt.replace(f"{{{var}}}", str(value))

        self.template_used = best_template.get('template_name', 'unknown')
        return filled_prompt

    def _select_best_template(self, task: str, templates: Dict) -> Dict:
        """Select the best template for the task."""
        # Simple keyword matching for now
        task_lower = task.lower()

        for template_name, template in templates.items():
            if template_name in task_lower:
                return template

        # Return first template as fallback
        return next(iter(templates.values())) if templates else None

    def _generate_variable_value(self, variable: str, task: str, domain: str) -> str:
        """Generate a value for a template variable."""
        # Simple value generation based on variable name
        if variable == 'requirements':
            return f"Complete the following task: {task}"
        elif variable == 'context':
            return f"This is a {domain} task that requires expertise in the field."
        elif variable == 'constraints':
            return "Ensure the solution is practical, efficient, and well-documented."
        elif variable == 'functionality':
            return task
        elif variable == 'objectives':
            return f"Achieve the following objectives: {task}"
        else:
            return f"Provide appropriate {variable} for this task."


class MetaPromptingStrategy(BaseGenerationStrategy):
    """Meta-prompting generation strategy."""

    async def generate(self, task: str, domain: str, context: Dict[str, Any],
                      persona: str = None, constraints: Dict[str, Any] = None,
                      template_library: Dict = None, persona_library: Dict = None) -> str:
        """Generate using meta-prompting."""
        # This would use meta-prompts to generate better prompts
        # Simplified implementation for now
        meta_prompt = f"""
Create an effective prompt for the following task:

Task: {task}
Domain: {domain}

The prompt should be:
1. Clear and specific
2. Include necessary context
3. Define expected output format
4. Consider potential edge cases
5. Provide examples if helpful

Generate the optimized prompt:
"""

        # In a real implementation, this would call an LLM with the meta-prompt
        return meta_prompt


class ChainOfPromptsStrategy(BaseGenerationStrategy):
    """Chain-of-prompts generation strategy."""

    async def generate(self, task: str, domain: str, context: Dict[str, Any],
                      persona: str = None, constraints: Dict[str, Any] = None,
                      template_library: Dict = None, persona_library: Dict = None) -> str:
        """Generate using chain-of-prompts approach."""
        # Break task into components and create a chain
        components = self._decompose_task(task)

        chain_prompt = f"""Please help with the following multi-step task:

Main Task: {task}

Step-by-step approach:
"""

        for i, component in enumerate(components, 1):
            chain_prompt += f"{i}. {component}\n"

        chain_prompt += "\nPlease provide a comprehensive solution addressing all steps."
        return chain_prompt

    def _decompose_task(self, task: str) -> List[str]:
        """Decompose task into components."""
        # Simple decomposition based on keywords
        if 'and' in task.lower():
            return [part.strip() for part in task.split('and')]
        elif ',' in task:
            return [part.strip() for part in task.split(',')]
        else:
            return [f"Analyze the requirements: {task}",
                   f"Design the solution approach",
                   f"Implement the solution",
                   f"Test and validate the results"]


class ContextualInjectionStrategy(BaseGenerationStrategy):
    """Contextual injection generation strategy."""

    async def generate(self, task: str, domain: str, context: Dict[str, Any],
                      persona: str = None, constraints: Dict[str, Any] = None,
                      template_library: Dict = None, persona_library: Dict = None) -> str:
        """Generate with contextual injection."""
        # Add relevant context to the prompt
        enhanced_prompt = f"""Context: This is a {domain} task requiring specialized knowledge.

Task: {task}
"""

        # Add domain-specific context
        if domain == 'software_engineering':
            enhanced_prompt += "\nConsider best practices, performance, and maintainability."
        elif domain == 'data_science':
            enhanced_prompt += "\nFocus on data quality, statistical validity, and practical insights."
        elif domain == 'creative':
            enhanced_prompt += "\nEmphasize creativity, audience engagement, and clear messaging."

        # Add any provided context
        if context:
            enhanced_prompt += f"\nAdditional Context: {context}"

        return enhanced_prompt


class PersonaBasedStrategy(BaseGenerationStrategy):
    """Persona-based generation strategy."""

    async def generate(self, task: str, domain: str, context: Dict[str, Any],
                      persona: str = None, constraints: Dict[str, Any] = None,
                      template_library: Dict = None, persona_library: Dict = None) -> str:
        """Generate using persona-based approach."""
        if not persona_library or domain not in persona_library:
            return f"Please help with the following task: {task}"

        # Select or use provided persona
        if persona and persona in persona_library[domain]:
            selected_persona = persona_library[domain][persona]
        else:
            selected_persona = next(iter(persona_library[domain].values()))

        # Generate persona-based prompt
        persona_prompt = f"""You are a {selected_persona['role']} with {selected_persona['expertise']}.

Your approach should be {selected_persona['style']}, and you value {selected_persona['values']}.

Task: {task}

Please provide a solution that reflects your expertise and approach:
"""

        return persona_prompt


class HybridStrategy(BaseGenerationStrategy):
    """Hybrid generation strategy combining multiple approaches."""

    async def generate(self, task: str, domain: str, context: Dict[str, Any],
                      persona: str = None, constraints: Dict[str, Any] = None,
                      template_library: Dict = None, persona_library: Dict = None) -> str:
        """Generate using hybrid approach."""
        # Combine template-based and contextual injection
        template_strategy = TemplateBasedStrategy()
        context_strategy = ContextualInjectionStrategy()

        # Get template-based prompt
        template_prompt = await template_strategy.generate(
            task, domain, context, persona, constraints,
            template_library, persona_library
        )

        # Get contextual prompt
        context_prompt = await context_strategy.generate(
            task, domain, context, persona, constraints,
            template_library, persona_library
        )

        # Combine them
        hybrid_prompt = f"""{context_prompt}

{template_prompt}

Please provide a comprehensive solution that combines both contextual understanding and structured approach.
"""

        return hybrid_prompt


# Optimization Algorithm Implementations
class BaseOptimizer:
    """Base class for prompt optimization algorithms."""

    async def optimize(self, prompt: str, task: str, domain: str,
                      max_iterations: int, context: Dict[str, Any],
                      quality_threshold: float) -> Dict[str, Any]:
        """Optimize a prompt."""
        raise NotImplementedError


class EvolutionaryOptimizer(BaseOptimizer):
    """Evolutionary optimization using genetic algorithms."""

    async def optimize(self, prompt: str, task: str, domain: str,
                      max_iterations: int, context: Dict[str, Any],
                      quality_threshold: float) -> Dict[str, Any]:
        """Optimize using evolutionary approach."""
        # Simplified evolutionary optimization
        population = self._generate_initial_population(prompt, 10)

        best_prompt = prompt
        best_score = self._evaluate_prompt(prompt, task, domain)

        for iteration in range(max_iterations):
            # Evaluate population
            scores = [self._evaluate_prompt(p, task, domain) for p in population]

            # Select best performers
            best_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:3]
            best_population = [population[i] for i in best_indices]

            # Create new generation
            new_population = []
            for _ in range(len(population)):
                parent1, parent2 = random.sample(best_population, 2)
                child = self._crossover(parent1, parent2)
                child = self._mutate(child)
                new_population.append(child)

            population = new_population

            # Update best
            current_best = max(population, key=lambda p: self._evaluate_prompt(p, task, domain))
            current_score = self._evaluate_prompt(current_best, task, domain)

            if current_score > best_score:
                best_prompt = current_best
                best_score = current_score

            if best_score >= quality_threshold:
                break

        return {
            'optimized_prompt': best_prompt,
            'improvement_score': best_score,
            'iterations_used': iteration + 1
        }

    def _generate_initial_population(self, prompt: str, size: int) -> List[str]:
        """Generate initial population of prompts."""
        population = [prompt]
        for _ in range(size - 1):
            mutated = self._mutate(prompt)
            population.append(mutated)
        return population

    def _mutate(self, prompt: str) -> str:
        """Mutate a prompt."""
        mutations = [
            lambda p: p + "\n\nPlease provide detailed reasoning for your approach.",
            lambda p: p + "\n\nConsider edge cases and error conditions.",
            lambda p: "IMPORTANT: " + p,
            lambda p: p.replace("Please", "You must"),
        ]
        mutation = random.choice(mutations)
        return mutation(prompt)

    def _crossover(self, parent1: str, parent2: str) -> str:
        """Create child from two parents."""
        split_point = len(parent1) // 2
        return parent1[:split_point] + parent2[split_point:]

    def _evaluate_prompt(self, prompt: str, task: str, domain: str) -> float:
        """Evaluate prompt quality."""
        # Simplified evaluation
        score = len(prompt) / 1000  # Prefer longer prompts (up to a point)
        score += prompt.count('?') * 0.1  # Questions are good
        score += prompt.count('Example') * 0.2  # Examples are valuable
        return min(score, 1.0)


class GradientBasedOptimizer(BaseOptimizer):
    """Gradient-based optimization."""

    async def optimize(self, prompt: str, task: str, domain: str,
                      max_iterations: int, context: Dict[str, Any],
                      quality_threshold: float) -> Dict[str, Any]:
        """Optimize using gradient-based approach."""
        # Simplified gradient descent approach
        current_prompt = prompt
        current_score = self._evaluate_prompt(current_prompt, task, domain)

        for iteration in range(max_iterations):
            # Generate candidate improvements
            candidates = [
                current_prompt + "\n\nProvide step-by-step reasoning.",
                current_prompt + "\n\nInclude specific examples.",
                current_prompt + "\n\nConsider performance implications.",
                current_prompt.replace("help", "assist me"),
            ]

            # Evaluate candidates
            candidate_scores = [self._evaluate_prompt(c, task, domain) for c in candidates]

            # Select best candidate
            best_index = max(range(len(candidate_scores)), key=lambda i: candidate_scores[i])
            best_candidate = candidates[best_index]
            best_score = candidate_scores[best_index]

            if best_score > current_score:
                current_prompt = best_candidate
                current_score = best_score

            if current_score >= quality_threshold:
                break

        return {
            'optimized_prompt': current_prompt,
            'improvement_score': current_score,
            'iterations_used': iteration + 1
        }

    def _evaluate_prompt(self, prompt: str, task: str, domain: str) -> float:
        """Evaluate prompt quality."""
        # Similar to evolutionary evaluation
        score = len(prompt) / 1000
        score += prompt.count('?') * 0.1
        score += prompt.count('Example') * 0.2
        return min(score, 1.0)


class ReinforcementLearningOptimizer(BaseOptimizer):
    """Reinforcement learning-based optimization."""

    async def optimize(self, prompt: str, task: str, domain: str,
                      max_iterations: int, context: Dict[str, Any],
                      quality_threshold: float) -> Dict[str, Any]:
        """Optimize using reinforcement learning."""
        # Simplified RL approach
        current_prompt = prompt
        current_score = self._evaluate_prompt(current_prompt, task, domain)

        # Simulate learning from rewards
        for iteration in range(max_iterations):
            # Generate action (improvement)
            action = self._select_action(current_prompt)

            # Apply action
            new_prompt = self._apply_action(current_prompt, action)

            # Evaluate reward
            new_score = self._evaluate_prompt(new_prompt, task, domain)

            if new_score > current_score:
                current_prompt = new_prompt
                current_score = new_score

            if current_score >= quality_threshold:
                break

        return {
            'optimized_prompt': current_prompt,
            'improvement_score': current_score,
            'iterations_used': iteration + 1
        }

    def _select_action(self, prompt: str) -> str:
        """Select an improvement action."""
        actions = [
            "add_structure",
            "add_examples",
            "add_constraints",
            "improve_clarity",
            "add_context"
        ]
        return random.choice(actions)

    def _apply_action(self, prompt: str, action: str) -> str:
        """Apply an action to the prompt."""
        if action == "add_structure":
            return prompt + "\n\nPlease structure your response clearly."
        elif action == "add_examples":
            return prompt + "\n\nProvide concrete examples in your response."
        elif action == "add_constraints":
            return prompt + "\n\nConsider any constraints or limitations."
        elif action == "improve_clarity":
            return prompt.replace("help", "clearly assist me")
        elif action == "add_context":
            return "Context: " + prompt
        return prompt

    def _evaluate_prompt(self, prompt: str, task: str, domain: str) -> float:
        """Evaluate prompt quality."""
        score = len(prompt) / 1000
        score += prompt.count('?') * 0.1
        score += prompt.count('Example') * 0.2
        return min(score, 1.0)


class ABTestingOptimizer(BaseOptimizer):
    """A/B testing-based optimization."""

    async def optimize(self, prompt: str, task: str, domain: str,
                      max_iterations: int, context: Dict[str, Any],
                      quality_threshold: float) -> Dict[str, Any]:
        """Optimize using A/B testing."""
        # Simplified A/B testing approach
        variants = [prompt]

        # Generate variants
        for _ in range(4):  # Create 4 variants
            variant = self._create_variant(prompt)
            variants.append(variant)

        # Evaluate all variants
        scores = [self._evaluate_prompt(v, task, domain) for v in variants]

        # Select best variant
        best_index = max(range(len(scores)), key=lambda i: scores[i])
        best_variant = variants[best_index]
        best_score = scores[best_index]

        return {
            'optimized_prompt': best_variant,
            'improvement_score': best_score,
            'iterations_used': 1  # Single evaluation pass
        }

    def _create_variant(self, prompt: str) -> str:
        """Create a variant of the prompt."""
        variations = [
            lambda p: p + "\n\nBe specific and provide examples.",
            lambda p: p + "\n\nExplain your reasoning step by step.",
            lambda p: "Please " + p.lower(),
            lambda p: p.replace(".", ".\n\n"),
        ]
        variation = random.choice(variations)
        return variation(prompt)

    def _evaluate_prompt(self, prompt: str, task: str, domain: str) -> float:
        """Evaluate prompt quality."""
        score = len(prompt) / 1000
        score += prompt.count('?') * 0.1
        score += prompt.count('Example') * 0.2
        return min(score, 1.0)


class PromptGenerationError(Exception):
    """Exception raised for prompt generation errors."""

    def __init__(self, message: str, generation_id: str = None):
        super().__init__(message)
        self.generation_id = generation_id


class PromptOptimizationError(Exception):
    """Exception raised for prompt optimization errors."""

    def __init__(self, message: str, optimization_id: str = None):
        super().__init__(message)
        self.optimization_id = optimization_id


# Global prompt generator instance
prompt_generator = PromptGenerator()
