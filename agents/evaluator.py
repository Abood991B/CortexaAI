"""Evaluator Agent for assessing prompt quality and running evaluation loops."""

from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI
from config.config import (
    settings, get_model_config, get_logger, metrics, log_performance,
    cache_manager, perf_config, generate_evaluation_cache_key, log_cache_performance,
    security_manager, security_config, log_security_event, prompt_generation_config
)
from agents.exceptions import EvaluationError, ImprovementError
import logging
import asyncio
import time

# Set up structured logging
logger = get_logger(__name__)


class PromptEvaluator:
    """Agent responsible for evaluating prompt quality and providing improvement feedback."""

    def __init__(self):
        """Initialize the evaluator agent."""
        self.evaluation_threshold = settings.evaluation_threshold
        self.max_iterations = settings.max_evaluation_iterations
        self._setup_evaluation_chain()

    def _setup_evaluation_chain(self):
        """Set up the LangChain for prompt evaluation."""
        model_config = get_model_config()
        self.model = ChatGoogleGenerativeAI(
            model=model_config["model_name"],
            google_api_key=model_config["api_key"],
            temperature=0.1  # Low temperature for consistent evaluation
        )

        evaluation_prompt = PromptTemplate.from_template("""
        You are an expert prompt evaluator. Your task is to assess the quality of a prompt improvement.

        DOMAIN: {domain}
        ORIGINAL PROMPT:
        {original_prompt}

        IMPROVED PROMPT:
        {improved_prompt}

        PROMPT TYPE: {prompt_type}
        IMPROVEMENTS CLAIMED: {improvements_made}

        EVALUATION CRITERIA:
        1. CLARITY: How clear and unambiguous is the prompt?
        2. SPECIFICITY: How specific are the requirements and expectations?
        3. STRUCTURE: How well is the prompt organized and structured?
        4. COMPLETENESS: How complete are the instructions and context?
        5. ACTIONABILITY: How actionable and implementable is the prompt?
        6. DOMAIN ALIGNMENT: How well does it align with domain best practices?

        TASK:
        Evaluate the improved prompt against the original and provide detailed feedback.
        Determine if the prompt meets quality standards or needs further improvement.

        Respond in JSON format with the following structure:
        {{
            "overall_score": 0.85,
            "criteria_scores": {{
                "clarity": 0.9,
                "specificity": 0.8,
                "structure": 0.9,
                "completeness": 0.8,
                "actionability": 0.85,
                "domain_alignment": 0.8
            }},
            "passes_threshold": true,
            "needs_improvement": false,
            "strengths": [
                "Strength 1",
                "Strength 2"
            ],
            "weaknesses": [
                "Weakness 1",
                "Weakness 2"
            ],
            "specific_feedback": [
                "Specific suggestion for improvement",
                "Another targeted recommendation"
            ],
            "improvement_priority": "high|medium|low",
            "reasoning": "Detailed explanation of evaluation and recommendations",
            "comparison_analysis": "Analysis of improvements made vs. what was needed"
        }}

        Set passes_threshold to true if overall_score >= {threshold}.
        Set needs_improvement to true if there are significant weaknesses to address.
        """)

        self.evaluation_chain = (
            {
                "domain": lambda x: x["domain"],
                "original_prompt": lambda x: x["original_prompt"],
                "improved_prompt": lambda x: x["improved_prompt"],
                "prompt_type": lambda x: x["prompt_type"],
                "improvements_made": lambda x: ", ".join(x.get("improvements_made", [])),
                "threshold": lambda x: self.evaluation_threshold
            }
            | evaluation_prompt
            | self.model
            | JsonOutputParser()
        )

    async def evaluate_prompt(self, original_prompt: str, improved_prompt: str,
                       domain: str, prompt_type: str = "raw",
                       improvements_made: List[str] = None) -> Dict[str, Any]:
        """
        Evaluate the quality of an improved prompt with security, caching and retry mechanism.

        Args:
            original_prompt: The original prompt
            improved_prompt: The improved prompt to evaluate
            domain: The domain of the prompt
            prompt_type: Type of prompt ("raw" or "structured")
            improvements_made: List of improvements that were made

        Returns:
            Dict containing evaluation results and feedback

        Raises:
            EvaluationError: If evaluation fails after retries
        """
        # Input sanitization
        if security_config.enable_input_sanitization:
            # Sanitize original prompt
            original_sanitized = security_manager.sanitize_input(original_prompt, f"evaluation_original_{domain}")

            # Sanitize improved prompt
            improved_sanitized = security_manager.sanitize_input(improved_prompt, f"evaluation_improved_{domain}")

            # The SecurityManager has already applied intelligent filtering based on security level and context
            # Only check for remaining high-severity events that passed all filtering
            original_events = [e for e in original_sanitized['security_events'] if e['severity'] == 'high']
            improved_events = [e for e in improved_sanitized['security_events'] if e['severity'] == 'high']

            # For evaluation contexts, be extremely permissive since SecurityManager already filtered
            if (original_events or improved_events) and security_config.enable_injection_detection:
                all_events = original_events + improved_events
                # Only block if there are genuine security threats that passed all filtering
                genuine_threats = [e for e in all_events if self._is_genuine_security_threat(e, f"evaluation_{domain}")]
                if genuine_threats:
                    log_security_event(logger, "unsafe_input_blocked", "high",
                                     context=f"evaluation_{domain}", events=genuine_threats)
                    raise EvaluationError(
                        "Input contains potentially unsafe content",
                        domain=domain,
                        security_events=genuine_threats
                    )

            sanitized_original = original_sanitized['sanitized_text']
            sanitized_improved = improved_sanitized['sanitized_text']
        else:
            sanitized_original = original_prompt
            sanitized_improved = improved_prompt

        # Check cache first if caching is enabled
        if perf_config.enable_caching:
            cache_key = generate_evaluation_cache_key(sanitized_original, sanitized_improved, domain)
            cached_result = cache_manager.get(cache_key)
            if cached_result:
                log_cache_performance(logger, "prompt_evaluation", True, domain=domain)
                return cached_result

        max_retries = getattr(settings, 'max_llm_retries', 3)
        retry_delay = getattr(settings, 'llm_retry_delay', 1.0)

        for attempt in range(max_retries + 1):
            try:
                logger.info(f"Evaluating prompt in domain: {domain} (attempt {attempt + 1})")

                evaluation_data = {
                    "domain": domain,
                    "original_prompt": sanitized_original,
                    "improved_prompt": sanitized_improved,
                    "prompt_type": prompt_type,
                    "improvements_made": improvements_made or []
                }

                result = await self.evaluation_chain.ainvoke(evaluation_data)

                # Ensure boolean values are properly set
                result["passes_threshold"] = result.get("passes_threshold", False)
                result["needs_improvement"] = result.get("needs_improvement", True)

                # Cache the result if caching is enabled (only for high-confidence results)
                if perf_config.enable_caching:
                    score = result.get("overall_score", 0)
                    if score >= 0.7:  # Only cache reasonably good evaluations
                        cache_manager.set(cache_key, result, perf_config.cache_ttl)

                logger.info(f"Evaluation completed. Score: {result.get('overall_score', 0)}")
                log_cache_performance(logger, "prompt_evaluation", False, domain=domain, score=score)
                return result

            except Exception as e:
                error_msg = f"Error evaluating prompt: {str(e)}"

                if attempt < max_retries:
                    # Determine if error is retryable
                    is_retryable = self._is_retryable_error(e)

                    if is_retryable:
                        logger.warning(f"{error_msg}. Retrying in {retry_delay}s... (attempt {attempt + 1}/{max_retries})")
                        await asyncio.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                        continue
                    else:
                        logger.error(f"{error_msg}. Error is not retryable.")
                        break
                else:
                    logger.error(f"{error_msg}. All {max_retries + 1} attempts failed.")

                    # Raise custom exception with detailed context
                    raise EvaluationError(
                        f"Failed to evaluate prompt after {max_retries + 1} attempts",
                        domain=domain,
                        iteration=0,  # Will be set by caller if available
                        threshold=self.evaluation_threshold,
                        cause=e
                    )

        # Fallback: return degraded result if all retries failed
        logger.warning(f"Returning fallback evaluation result for {domain} due to repeated failures")
        return {
            "overall_score": 0.5,
            "criteria_scores": {
                "clarity": 0.5,
                "specificity": 0.5,
                "structure": 0.5,
                "completeness": 0.5,
                "actionability": 0.5,
                "domain_alignment": 0.5
            },
            "passes_threshold": False,
            "needs_improvement": True,
            "strengths": [],
            "weaknesses": ["Evaluation failed after multiple attempts"],
            "specific_feedback": ["Unable to complete evaluation due to processing errors"],
            "improvement_priority": "high",
            "reasoning": "Evaluation could not be completed due to technical issues",
            "comparison_analysis": "Unable to perform comparison analysis"
        }

    def _is_retryable_error(self, error: Exception) -> bool:
        """Determine if an error is retryable based on its characteristics."""
        error_str = str(error).lower()
        retryable_indicators = [
            "rate limit", "timeout", "connection", "network",
            "temporary", "server error", "502", "503", "504",
            "internal server error", "service unavailable"
        ]

        # Check if any retryable indicator is in the error message
        is_retryable = any(indicator in error_str for indicator in retryable_indicators)

        logger.info(f"Error retryability check: {is_retryable} for error: {error_str}")
        return is_retryable

    async def run_evaluation_loop(self, original_prompt: str, improved_prompt: str,
                           domain: str, expert_agent: Any,
                           prompt_type: str = "raw") -> Tuple[Dict[str, Any], int]:
        """
        Run an evaluation loop until the prompt reaches acceptable quality.

        Args:
            original_prompt: The original prompt
            improved_prompt: Initial improved prompt
            domain: The domain of the prompt
            expert_agent: The expert agent that can improve prompts
            prompt_type: Type of prompt ("raw" or "structured")

        Returns:
            Tuple of (final_evaluation_result, iterations_used)
        """
        current_prompt = improved_prompt
        iteration = 0

        while iteration < self.max_iterations:
            iteration += 1

            logger.info(f"Evaluation iteration {iteration}/{self.max_iterations}")

            # Evaluate current prompt
            evaluation = await self.evaluate_prompt(
                original_prompt=original_prompt,
                improved_prompt=current_prompt,
                domain=domain,
                prompt_type=prompt_type
            )

            # Check if we meet the threshold
            if evaluation.get("passes_threshold", False):
                logger.info(f"Prompt passed evaluation threshold after {iteration} iterations")
                return evaluation, iteration

            # Check if we need improvement
            if not evaluation.get("needs_improvement", True):
                logger.info(f"Prompt deemed acceptable after {iteration} iterations")
                return evaluation, iteration

            # Get feedback for improvement
            feedback = evaluation.get("specific_feedback", [])
            if not feedback:
                logger.warning("No specific feedback provided, ending evaluation loop")
                return evaluation, iteration

            # Use expert agent to improve based on feedback
            try:
                improvement_result = await expert_agent.improve_prompt(
                    original_prompt=current_prompt,
                    prompt_type="structured",  # Treat as structured for re-improvement
                    key_topics=[domain]  # Pass domain as key topic
                )

                current_prompt = improvement_result.get("improved_prompt", current_prompt)

                logger.info(f"Completed iteration {iteration}, continuing evaluation loop")

            except ImprovementError as ie:
                logger.error(f"Improvement error in evaluation loop iteration {iteration}: {ie}")
                # Return current evaluation with error context
                evaluation["weaknesses"].append(f"Improvement failed: {str(ie)}")
                evaluation["specific_feedback"].append(f"Unable to improve further due to: {ie.error_code}")
                return evaluation, iteration

            except Exception as e:
                logger.error(f"Unexpected error in evaluation loop iteration {iteration}: {e}")
                # Continue with current prompt if improvement fails
                logger.warning(f"Continuing with current prompt due to error: {str(e)}")
                continue

        logger.warning(f"Maximum iterations ({self.max_iterations}) reached without meeting threshold")
        return evaluation, iteration

    def get_evaluation_summary(self, evaluations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate a summary of multiple evaluations.

        Args:
            evaluations: List of evaluation results

        Returns:
            Dict containing evaluation summary statistics
        """
        if not evaluations:
            return {"error": "No evaluations provided"}

        scores = [eval.get("overall_score", 0) for eval in evaluations]
        criteria_scores = {}

        # Aggregate criteria scores
        criteria_names = ["clarity", "specificity", "structure", "completeness", "actionability", "domain_alignment"]
        for criterion in criteria_names:
            criterion_scores = [eval.get("criteria_scores", {}).get(criterion, 0) for eval in evaluations]
            criteria_scores[criterion] = {
                "average": sum(criterion_scores) / len(criterion_scores) if criterion_scores else 0,
                "min": min(criterion_scores) if criterion_scores else 0,
                "max": max(criterion_scores) if criterion_scores else 0
            }

        passed_count = sum(1 for eval in evaluations if eval.get("passes_threshold", False))

        return {
            "total_evaluations": len(evaluations),
            "passed_evaluations": passed_count,
            "pass_rate": passed_count / len(evaluations) if evaluations else 0,
            "average_overall_score": sum(scores) / len(scores) if scores else 0,
            "score_range": {
                "min": min(scores) if scores else 0,
                "max": max(scores) if scores else 0
            },
            "criteria_averages": criteria_scores,
            "common_weaknesses": self._extract_common_weaknesses(evaluations),
            "common_strengths": self._extract_common_strengths(evaluations)
        }

    def _extract_common_weaknesses(self, evaluations: List[Dict[str, Any]]) -> List[str]:
        """Extract common weaknesses across evaluations."""
        all_weaknesses = []
        for eval in evaluations:
            all_weaknesses.extend(eval.get("weaknesses", []))

        # Simple frequency count (could be enhanced with more sophisticated analysis)
        weakness_count = {}
        for weakness in all_weaknesses:
            weakness_count[weakness] = weakness_count.get(weakness, 0) + 1

        # Return weaknesses that appear in more than one evaluation
        return [w for w, count in weakness_count.items() if count > 1]

    def _extract_common_strengths(self, evaluations: List[Dict[str, Any]]) -> List[str]:
        """Extract common strengths across evaluations."""
        all_strengths = []
        for eval in evaluations:
            all_strengths.extend(eval.get("strengths", []))

        # Simple frequency count
        strength_count = {}
        for strength in all_strengths:
            strength_count[strength] = strength_count.get(strength, 0) + 1

        # Return strengths that appear in more than one evaluation
        return [s for s, count in strength_count.items() if count > 1]

    async def evaluate_prompt_generation(self, task: str, generated_prompt: str,
                                       domain: str, strategy_used: str,
                                       generation_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate the effectiveness of prompt generation strategies.

        Args:
            task: Original task description
            generated_prompt: The generated prompt
            domain: The domain context
            strategy_used: The generation strategy that was used
            generation_metadata: Metadata from the generation process

        Returns:
            Dict containing generation effectiveness evaluation
        """
        if not prompt_generation_config.enable_quality_gates:
            # Return basic evaluation if quality gates are disabled
            return {
                'generation_score': 0.7,
                'strategy_effectiveness': 0.7,
                'task_alignment': 0.7,
                'recommendations': ['Enable quality gates for detailed evaluation']
            }

        try:
            logger.info(f"Evaluating prompt generation for domain {domain} using strategy {strategy_used}")

            # Evaluate generation effectiveness
            evaluation_data = {
                'task': task,
                'generated_prompt': generated_prompt,
                'domain': domain,
                'strategy_used': strategy_used,
                'generation_metadata': generation_metadata
            }

            # Use a simplified evaluation for generation quality
            generation_score = self._assess_generation_quality(
                task, generated_prompt, domain, generation_metadata
            )

            # Assess strategy effectiveness
            strategy_effectiveness = self._assess_strategy_effectiveness(
                strategy_used, generation_score, generation_metadata
            )

            # Assess task alignment
            task_alignment = self._assess_task_alignment(task, generated_prompt)

            # Generate recommendations
            recommendations = self._generate_generation_recommendations(
                generation_score, strategy_effectiveness, task_alignment, domain
            )

            result = {
                'generation_score': generation_score,
                'strategy_effectiveness': strategy_effectiveness,
                'task_alignment': task_alignment,
                'recommendations': recommendations,
                'metadata': {
                    'domain': domain,
                    'strategy_used': strategy_used,
                    'evaluation_timestamp': datetime.now().isoformat(),
                    'quality_threshold': prompt_generation_config.quality_threshold
                }
            }

            logger.info(f"Generation evaluation completed. Score: {generation_score:.2f}")
            return result

        except Exception as e:
            logger.error(f"Failed to evaluate prompt generation: {e}")
            return {
                'generation_score': 0.5,
                'strategy_effectiveness': 0.5,
                'task_alignment': 0.5,
                'recommendations': [f'Generation evaluation failed: {str(e)}'],
                'error': str(e)
            }

    async def evaluate_optimization_performance(self, original_prompt: str, optimized_prompt: str,
                                              algorithm_used: str, iterations_used: int,
                                              improvement_score: float, domain: str) -> Dict[str, Any]:
        """
        Evaluate the performance of prompt optimization algorithms.

        Args:
            original_prompt: The original prompt before optimization
            optimized_prompt: The optimized prompt
            algorithm_used: The optimization algorithm that was used
            iterations_used: Number of iterations used
            improvement_score: The improvement score achieved
            domain: The domain context

        Returns:
            Dict containing optimization performance evaluation
        """
        if not prompt_generation_config.enable_optimization:
            return {
                'algorithm_score': 0.7,
                'efficiency_score': 0.7,
                'improvement_quality': improvement_score,
                'recommendations': ['Optimization is disabled']
            }

        try:
            logger.info(f"Evaluating optimization performance for algorithm {algorithm_used}")

            # Assess algorithm effectiveness
            algorithm_score = self._assess_algorithm_effectiveness(
                algorithm_used, improvement_score, iterations_used
            )

            # Assess optimization efficiency
            efficiency_score = self._assess_optimization_efficiency(
                iterations_used, improvement_score, len(original_prompt), len(optimized_prompt)
            )

            # Assess improvement quality
            improvement_quality = self._assess_improvement_quality(
                original_prompt, optimized_prompt, improvement_score
            )

            # Generate optimization recommendations
            recommendations = self._generate_optimization_recommendations(
                algorithm_used, algorithm_score, efficiency_score, improvement_quality, domain
            )

            result = {
                'algorithm_score': algorithm_score,
                'efficiency_score': efficiency_score,
                'improvement_quality': improvement_quality,
                'overall_score': (algorithm_score + efficiency_score + improvement_quality) / 3,
                'recommendations': recommendations,
                'metadata': {
                    'algorithm_used': algorithm_used,
                    'iterations_used': iterations_used,
                    'original_length': len(original_prompt),
                    'optimized_length': len(optimized_prompt),
                    'domain': domain,
                    'evaluation_timestamp': datetime.now().isoformat()
                }
            }

            logger.info(f"Optimization evaluation completed. Overall score: {result['overall_score']:.2f}")
            return result

        except Exception as e:
            logger.error(f"Failed to evaluate optimization performance: {e}")
            return {
                'algorithm_score': 0.5,
                'efficiency_score': 0.5,
                'improvement_quality': improvement_score,
                'overall_score': 0.5,
                'recommendations': [f'Optimization evaluation failed: {str(e)}'],
                'error': str(e)
            }

    async def compare_prompts(self, prompts: List[Dict[str, Any]], criteria: List[str] = None) -> Dict[str, Any]:
        """
        Compare multiple prompts and rank them based on specified criteria.

        Args:
            prompts: List of prompt dictionaries with metadata
            criteria: List of criteria to evaluate on

        Returns:
            Dict containing comparison results and rankings
        """
        if not prompts or len(prompts) < 2:
            return {'error': 'At least 2 prompts required for comparison'}

        default_criteria = ['clarity', 'specificity', 'structure', 'completeness', 'actionability']
        evaluation_criteria = criteria or default_criteria

        try:
            logger.info(f"Comparing {len(prompts)} prompts based on {len(evaluation_criteria)} criteria")

            # Evaluate each prompt
            evaluations = []
            for i, prompt_data in enumerate(prompts):
                evaluation = await self._evaluate_single_prompt_for_comparison(
                    prompt_data, evaluation_criteria
                )
                evaluations.append({
                    'prompt_id': i,
                    'prompt_data': prompt_data,
                    'evaluation': evaluation
                })

            # Rank prompts based on overall scores
            ranked_prompts = sorted(evaluations,
                                  key=lambda x: x['evaluation']['overall_score'],
                                  reverse=True)

            # Calculate comparison statistics
            scores = [eval['evaluation']['overall_score'] for eval in evaluations]
            best_score = max(scores)
            worst_score = min(scores)
            score_range = best_score - worst_score

            # Generate comparison insights
            insights = self._generate_comparison_insights(ranked_prompts, evaluation_criteria)

            result = {
                'total_prompts': len(prompts),
                'criteria_used': evaluation_criteria,
                'ranked_prompts': ranked_prompts,
                'comparison_stats': {
                    'best_score': best_score,
                    'worst_score': worst_score,
                    'score_range': score_range,
                    'average_score': sum(scores) / len(scores),
                    'winning_margin': ranked_prompts[0]['evaluation']['overall_score'] - ranked_prompts[1]['evaluation']['overall_score'] if len(ranked_prompts) > 1 else 0
                },
                'insights': insights,
                'metadata': {
                    'comparison_timestamp': datetime.now().isoformat(),
                    'evaluation_method': 'multi_criteria_ranking'
                }
            }

            logger.info(f"Prompt comparison completed. Best score: {best_score:.2f}")
            return result

        except Exception as e:
            logger.error(f"Failed to compare prompts: {e}")
            return {
                'error': str(e),
                'total_prompts': len(prompts) if prompts else 0,
                'comparison_failed': True
            }


    def _assess_generation_quality(self, task: str, generated_prompt: str, domain: str,
                                 generation_metadata: Dict[str, Any]) -> float:
        """Assess the quality of generated prompts."""
        quality_score = 0.0

        # Task relevance (30%)
        task_words = set(task.lower().split())
        prompt_words = set(generated_prompt.lower().split())
        overlap = len(task_words.intersection(prompt_words))
        relevance_score = overlap / len(task_words) if task_words else 0
        quality_score += min(relevance_score, 1.0) * 0.3

        # Length appropriateness (20%)
        optimal_length = 200 + (len(task) * 1.5)
        length_ratio = len(generated_prompt) / optimal_length
        length_score = 1.0 - abs(1.0 - length_ratio)
        quality_score += max(0, length_score) * 0.2

        # Structure indicators (25%)
        structure_indicators = [
            '?' in generated_prompt,  # Questions
            ':' in generated_prompt,  # Structure markers
            '-' in generated_prompt,  # Lists
            'Example' in generated_prompt,  # Examples
            'Note:' in generated_prompt  # Notes
        ]
        structure_score = sum(structure_indicators) / len(structure_indicators)
        quality_score += structure_score * 0.25

        # Domain alignment (25%)
        domain_keywords = {
            'software_engineering': ['code', 'function', 'class', 'algorithm', 'debug', 'api', 'database'],
            'data_science': ['data', 'analysis', 'model', 'statistics', 'visualization', 'machine learning'],
            'creative': ['design', 'content', 'creative', 'audience', 'engagement', 'brand']
        }
        domain_words = domain_keywords.get(domain, [])
        domain_matches = sum(1 for word in domain_words if word in generated_prompt.lower())
        domain_score = min(domain_matches / 3, 1.0) if domain_words else 0.5
        quality_score += domain_score * 0.25

        return min(quality_score, 1.0)

    def _assess_strategy_effectiveness(self, strategy_used: str, generation_score: float,
                                     generation_metadata: Dict[str, Any]) -> float:
        """Assess the effectiveness of the generation strategy used."""
        base_effectiveness = {
            'basic': 0.6,
            'template_based': 0.8,
            'meta_prompting': 0.85,
            'chain_of_prompts': 0.9,
            'contextual_injection': 0.82,
            'persona_based': 0.88,
            'hybrid': 0.92
        }

        base_score = base_effectiveness.get(strategy_used, 0.7)

        # Adjust based on generation metadata
        if generation_metadata.get('quality_score', 0) > 0.8:
            base_score += 0.1  # Bonus for high-quality generation
        elif generation_metadata.get('quality_score', 0) < 0.5:
            base_score -= 0.1  # Penalty for low-quality generation

        return max(0, min(base_score, 1.0))

    def _assess_task_alignment(self, task: str, generated_prompt: str) -> float:
        """Assess how well the generated prompt aligns with the original task."""
        # Simple task alignment assessment
        task_keywords = set(task.lower().split()[:10])  # First 10 words
        prompt_keywords = set(generated_prompt.lower().split()[:20])  # First 20 words

        overlap = len(task_keywords.intersection(prompt_keywords))
        alignment_score = overlap / len(task_keywords) if task_keywords else 0

        return min(alignment_score, 1.0)

    def _generate_generation_recommendations(self, generation_score: float, strategy_effectiveness: float,
                                          task_alignment: float, domain: str) -> List[str]:
        """Generate recommendations for improving prompt generation."""
        recommendations = []

        if generation_score < 0.7:
            recommendations.append("Consider using a more sophisticated generation strategy")
            recommendations.append("Review template library for the specific domain")

        if strategy_effectiveness < 0.8:
            recommendations.append("Experiment with different generation strategies for better results")
            if domain in ['software_engineering', 'data_science']:
                recommendations.append("Use template-based or chain-of-prompts strategies for technical domains")

        if task_alignment < 0.7:
            recommendations.append("Improve task analysis and keyword extraction")
            recommendations.append("Consider using contextual injection for better task alignment")

        if not recommendations:
            recommendations.append("Current generation approach is effective")

        return recommendations

    def _assess_algorithm_effectiveness(self, algorithm: str, improvement_score: float, iterations: int) -> float:
        """Assess the effectiveness of optimization algorithms."""
        base_effectiveness = {
            'evolutionary': 0.85,
            'gradient_based': 0.8,
            'reinforcement_learning': 0.9,
            'ab_testing': 0.75
        }

        base_score = base_effectiveness.get(algorithm, 0.7)

        # Adjust based on performance
        if improvement_score > 0.8:
            base_score += 0.1
        elif improvement_score < 0.5:
            base_score -= 0.1

        # Efficiency bonus/penalty based on iterations
        if iterations <= 3:
            base_score += 0.05  # Efficient convergence
        elif iterations > 10:
            base_score -= 0.05  # Inefficient convergence

        return max(0, min(base_score, 1.0))

    def _assess_optimization_efficiency(self, iterations: int, improvement_score: float,
                                      original_length: int, optimized_length: int) -> float:
        """Assess the efficiency of the optimization process."""
        # Base efficiency on iterations vs improvement
        efficiency_score = improvement_score / max(iterations, 1)

        # Length optimization bonus
        if optimized_length < original_length * 1.1:  # Reasonable length increase
            efficiency_score += 0.1

        return min(efficiency_score, 1.0)

    def _assess_improvement_quality(self, original_prompt: str, optimized_prompt: str, improvement_score: float) -> float:
        """Assess the quality of improvements made."""
        # Check if optimization actually improved the prompt
        if len(optimized_prompt) > len(original_prompt) * 2:
            return improvement_score * 0.8  # Penalize excessive length

        if improvement_score < 0.1:
            return 0.3  # Minimum score for attempted optimization

        return improvement_score

    def _generate_optimization_recommendations(self, algorithm: str, algorithm_score: float,
                                             efficiency_score: float, improvement_quality: float, domain: str) -> List[str]:
        """Generate recommendations for optimization improvements."""
        recommendations = []

        if algorithm_score < 0.7:
            recommendations.append(f"Consider alternative algorithms to {algorithm}")
            recommendations.append("Review algorithm parameters and thresholds")

        if efficiency_score < 0.7:
            recommendations.append("Optimize algorithm convergence parameters")
            recommendations.append("Consider reducing maximum iterations for faster results")

        if improvement_quality < 0.7:
            recommendations.append("Review improvement criteria and scoring")
            recommendations.append("Consider domain-specific optimization strategies")

        if algorithm == 'evolutionary' and efficiency_score < 0.8:
            recommendations.append("Adjust population size or mutation rate for evolutionary optimization")

        if algorithm == 'gradient_based' and efficiency_score < 0.8:
            recommendations.append("Fine-tune learning rate for gradient-based optimization")

        return recommendations or ["Current optimization approach is effective"]

    async def _evaluate_single_prompt_for_comparison(self, prompt_data: Dict[str, Any], criteria: List[str]) -> Dict[str, Any]:
        """Evaluate a single prompt for comparison purposes."""
        prompt = prompt_data.get('prompt', '')
        task = prompt_data.get('task', '')
        domain = prompt_data.get('domain', 'general')

        # Simplified evaluation for comparison
        scores = {}
        for criterion in criteria:
            if criterion == 'clarity':
                scores[criterion] = self._score_clarity(prompt)
            elif criterion == 'specificity':
                scores[criterion] = self._score_specificity(prompt, task)
            elif criterion == 'structure':
                scores[criterion] = self._score_structure(prompt)
            elif criterion == 'completeness':
                scores[criterion] = self._score_completeness(prompt)
            elif criterion == 'actionability':
                scores[criterion] = self._score_actionability(prompt)
            else:
                scores[criterion] = 0.7  # Default score

        overall_score = sum(scores.values()) / len(scores)

        return {
            'criteria_scores': scores,
            'overall_score': overall_score,
            'evaluation_timestamp': datetime.now().isoformat()
        }

    def _generate_comparison_insights(self, ranked_prompts: List[Dict[str, Any]], criteria: List[str]) -> List[str]:
        """Generate insights from prompt comparison."""
        insights = []

        if len(ranked_prompts) >= 2:
            best_prompt = ranked_prompts[0]
            worst_prompt = ranked_prompts[-1]

            score_difference = best_prompt['evaluation']['overall_score'] - worst_prompt['evaluation']['overall_score']
            insights.append(f"Best prompt scores {score_difference:.2f} points higher than the worst")

            # Identify best and worst criteria
            best_scores = best_prompt['evaluation']['criteria_scores']
            best_criterion = max(best_scores, key=best_scores.get)
            worst_criterion = min(best_scores, key=best_scores.get)

            insights.append(f"Best prompt excels in {best_criterion}")
            insights.append(f"Best prompt could improve in {worst_criterion}")

        return insights


    def _score_clarity(self, prompt: str) -> float:
        """Score prompt clarity."""
        clarity_indicators = ['clear', 'specific', 'exactly', 'precisely', 'must', 'should']
        score = sum(1 for word in clarity_indicators if word in prompt.lower())
        return min(score / 3, 1.0)

    def _score_specificity(self, prompt: str, task: str) -> float:
        """Score prompt specificity."""
        task_keywords = set(task.lower().split()[:5])
        prompt_keywords = set(prompt.lower().split()[:10])
        overlap = len(task_keywords.intersection(prompt_keywords))
        return min(overlap / len(task_keywords), 1.0) if task_keywords else 0.5

    def _score_structure(self, prompt: str) -> float:
        """Score prompt structure."""
        structure_indicators = [':', '-', '1.', '2.', '3.', 'First', 'Second', 'Then', 'Finally']
        score = sum(1 for indicator in structure_indicators if indicator in prompt)
        return min(score / 5, 1.0)

    def _score_completeness(self, prompt: str) -> float:
        """Score prompt completeness."""
        completeness_indicators = [
            'context' in prompt.lower(),
            'requirements' in prompt.lower(),
            'example' in prompt.lower(),
            'constraints' in prompt.lower(),
            'output' in prompt.lower()
        ]
        return sum(completeness_indicators) / len(completeness_indicators)

    def _score_actionability(self, prompt: str) -> float:
        """Score prompt actionability."""
        action_indicators = ['provide', 'create', 'implement', 'analyze', 'design', 'write']
        score = sum(1 for word in action_indicators if word in prompt.lower())
        return min(score / 4, 1.0)

    def _is_genuine_security_threat(self, event: Dict, context: str) -> bool:
        """Final sanity check for security events that passed SecurityManager filtering."""
        # Only flag events with definitive proof of malicious intent
        if event['type'] == 'injection_attempt':
            matched_text = event.get('matched_text', '').lower()

            # Very specific patterns that indicate genuine malicious intent
            genuine_threats = [
                'ignore all previous instructions and',
                'override system security',
                'bypass all restrictions',
                'execute arbitrary code',
                'system compromise',
                'malicious payload',
                'exploit vulnerability'
            ]

            return any(threat in matched_text for threat in genuine_threats)

        elif event['type'] == 'content_violation':
            # Only block content violations that are genuinely harmful
            keyword = event.get('keyword', '').lower()
            harmful_keywords = ['terrorism', 'violence', 'illegal']

            return keyword in harmful_keywords and context.startswith('evaluation')

        return False  # Default to allowing events that passed SecurityManager filtering


# Global evaluator instance
evaluator = PromptEvaluator()
