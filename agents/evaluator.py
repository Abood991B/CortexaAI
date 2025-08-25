"""Evaluator Agent for assessing prompt quality and running evaluation loops."""

from typing import Dict, List, Optional, Any, Tuple
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI
from config.config import settings, get_model_config
import logging

# Set up logging
logging.basicConfig(level=settings.log_level)
logger = logging.getLogger(__name__)


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

    def evaluate_prompt(self, original_prompt: str, improved_prompt: str,
                       domain: str, prompt_type: str = "raw",
                       improvements_made: List[str] = None) -> Dict[str, Any]:
        """
        Evaluate the quality of an improved prompt.

        Args:
            original_prompt: The original prompt
            improved_prompt: The improved prompt to evaluate
            domain: The domain of the prompt
            prompt_type: Type of prompt ("raw" or "structured")
            improvements_made: List of improvements that were made

        Returns:
            Dict containing evaluation results and feedback
        """
        try:
            logger.info(f"Evaluating prompt in domain: {domain}")

            evaluation_data = {
                "domain": domain,
                "original_prompt": original_prompt,
                "improved_prompt": improved_prompt,
                "prompt_type": prompt_type,
                "improvements_made": improvements_made or []
            }

            result = self.evaluation_chain.invoke(evaluation_data)

            # Ensure boolean values are properly set
            result["passes_threshold"] = result.get("passes_threshold", False)
            result["needs_improvement"] = result.get("needs_improvement", True)

            logger.info(f"Evaluation completed. Score: {result.get('overall_score', 0)}")
            return result

        except Exception as e:
            logger.error(f"Error evaluating prompt: {e}")
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
                "weaknesses": [f"Evaluation failed: {str(e)}"],
                "specific_feedback": ["Unable to complete evaluation due to processing error"],
                "improvement_priority": "high",
                "reasoning": "Evaluation could not be completed due to technical issues",
                "comparison_analysis": "Unable to perform comparison analysis"
            }

    def run_evaluation_loop(self, original_prompt: str, improved_prompt: str,
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
            evaluation = self.evaluate_prompt(
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
                improvement_result = expert_agent.improve_prompt(
                    original_prompt=current_prompt,
                    prompt_type="structured",  # Treat as structured for re-improvement
                    key_topics=[domain]  # Pass domain as key topic
                )

                current_prompt = improvement_result.get("improved_prompt", current_prompt)

                logger.info(f"Completed iteration {iteration}, continuing evaluation loop")

            except Exception as e:
                logger.error(f"Error in evaluation loop iteration {iteration}: {e}")
                return evaluation, iteration

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


# Global evaluator instance
evaluator = PromptEvaluator()
