"""
Prompt Optimization Engine for CortexaAI.

Implements state-of-the-art prompt optimization with:
- Iterative refinement with quality convergence
- A/B testing of prompt variants
- Prompt version history and rollback
- Performance analytics and metrics
- Multiple optimization strategies (iterative, comparative, evolutionary)
- Persistent export/import of optimization history
"""

import time
import uuid
import hashlib
import json
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from dataclasses import dataclass, field, asdict
from pathlib import Path

from config.config import settings, get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

@dataclass
class PromptVersion:
    """A single versioned prompt with metadata."""
    version_id: str
    prompt_text: str
    domain: str
    quality_score: float
    created_at: str
    parent_version: Optional[str] = None
    strategy_used: str = "iterative"
    improvements: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ABTestResult:
    """Result of an A/B test between two prompt variants."""
    test_id: str
    variant_a: PromptVersion
    variant_b: PromptVersion
    winner: str  # "A" | "B" | "tie"
    score_a: float
    score_b: float
    criteria_comparison: Dict[str, Dict[str, float]]
    confidence: float
    created_at: str
    reasoning: str = ""


@dataclass
class OptimizationRun:
    """A complete optimization run with full history."""
    run_id: str
    original_prompt: str
    final_prompt: str
    domain: str
    strategy: str
    versions: List[PromptVersion]
    ab_tests: List[ABTestResult]
    total_iterations: int
    total_time_seconds: float
    initial_score: float
    final_score: float
    improvement_percentage: float
    created_at: str


# ---------------------------------------------------------------------------
# Version Control System
# ---------------------------------------------------------------------------

class PromptVersionControl:
    """Version control system for prompts with full history tracking."""

    def __init__(self):
        self._versions: Dict[str, PromptVersion] = {}
        self._history: Dict[str, List[str]] = {}  # prompt_hash -> [version_ids]

    def create_version(
        self,
        prompt_text: str,
        domain: str,
        quality_score: float,
        parent_version: Optional[str] = None,
        strategy_used: str = "iterative",
        improvements: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> PromptVersion:
        """Create a new prompt version."""
        version_id = f"v_{uuid.uuid4().hex[:12]}"
        version = PromptVersion(
            version_id=version_id,
            prompt_text=prompt_text,
            domain=domain,
            quality_score=quality_score,
            created_at=datetime.now().isoformat(),
            parent_version=parent_version,
            strategy_used=strategy_used,
            improvements=improvements or [],
            metadata=metadata or {},
        )

        self._versions[version_id] = version

        # Track history by prompt content hash
        prompt_hash = hashlib.md5(prompt_text.encode()).hexdigest()[:16]
        if prompt_hash not in self._history:
            self._history[prompt_hash] = []
        self._history[prompt_hash].append(version_id)

        logger.info(f"Created prompt version {version_id} (score: {quality_score:.2f})")
        return version

    def get_version(self, version_id: str) -> Optional[PromptVersion]:
        """Retrieve a specific version."""
        return self._versions.get(version_id)

    def get_version_history(self, version_id: str) -> List[PromptVersion]:
        """Get the full lineage of a version (all ancestors)."""
        history = []
        current = self._versions.get(version_id)
        while current:
            history.append(current)
            current = self._versions.get(current.parent_version) if current.parent_version else None
        return list(reversed(history))

    def get_best_version(self, domain: Optional[str] = None) -> Optional[PromptVersion]:
        """Get the highest-scoring version, optionally filtered by domain."""
        candidates = self._versions.values()
        if domain:
            candidates = [v for v in candidates if v.domain == domain]
        if not candidates:
            return None
        return max(candidates, key=lambda v: v.quality_score)

    def get_stats(self) -> Dict[str, Any]:
        """Get version control statistics."""
        versions = list(self._versions.values())
        if not versions:
            return {"total_versions": 0, "domains": {}, "avg_score": 0.0}

        domains: Dict[str, int] = {}
        for v in versions:
            domains[v.domain] = domains.get(v.domain, 0) + 1

        scores = [v.quality_score for v in versions]
        return {
            "total_versions": len(versions),
            "domains": domains,
            "avg_score": sum(scores) / len(scores),
            "best_score": max(scores),
            "worst_score": min(scores),
        }


# ---------------------------------------------------------------------------
# A/B Testing Engine
# ---------------------------------------------------------------------------

class ABTestingEngine:
    """A/B testing engine for comparing prompt variants."""

    def __init__(self):
        self._tests: List[ABTestResult] = []

    def create_test(
        self,
        variant_a: PromptVersion,
        variant_b: PromptVersion,
        evaluation_a: Dict[str, Any],
        evaluation_b: Dict[str, Any],
    ) -> ABTestResult:
        """
        Run an A/B test comparing two prompt variants.

        Args:
            variant_a: First prompt variant
            variant_b: Second prompt variant
            evaluation_a: Evaluation results for variant A
            evaluation_b: Evaluation results for variant B

        Returns:
            ABTestResult with comparison details
        """
        score_a = evaluation_a.get("overall_score", 0.0)
        score_b = evaluation_b.get("overall_score", 0.0)

        # Determine winner with significance margin
        margin = 0.05  # 5% minimum difference for a winner
        if score_a - score_b > margin:
            winner = "A"
        elif score_b - score_a > margin:
            winner = "B"
        else:
            winner = "tie"

        # Compare individual criteria
        criteria_a = evaluation_a.get("criteria_scores", {})
        criteria_b = evaluation_b.get("criteria_scores", {})
        criteria_comparison = {}
        for criterion in set(list(criteria_a.keys()) + list(criteria_b.keys())):
            criteria_comparison[criterion] = {
                "variant_a": criteria_a.get(criterion, 0.0),
                "variant_b": criteria_b.get(criterion, 0.0),
            }

        # Calculate confidence based on score difference magnitude
        score_diff = abs(score_a - score_b)
        confidence = min(1.0, score_diff / 0.2)  # Max confidence at 20% difference

        reasoning_parts = []
        if winner == "A":
            reasoning_parts.append(f"Variant A scored {score_a:.2f} vs B's {score_b:.2f}")
        elif winner == "B":
            reasoning_parts.append(f"Variant B scored {score_b:.2f} vs A's {score_a:.2f}")
        else:
            reasoning_parts.append(f"Scores too close: A={score_a:.2f}, B={score_b:.2f}")

        # Note which criteria each variant won
        a_wins = [c for c, s in criteria_comparison.items() if s["variant_a"] > s["variant_b"]]
        b_wins = [c for c, s in criteria_comparison.items() if s["variant_b"] > s["variant_a"]]
        if a_wins:
            reasoning_parts.append(f"A excelled in: {', '.join(a_wins)}")
        if b_wins:
            reasoning_parts.append(f"B excelled in: {', '.join(b_wins)}")

        result = ABTestResult(
            test_id=f"ab_{uuid.uuid4().hex[:12]}",
            variant_a=variant_a,
            variant_b=variant_b,
            winner=winner,
            score_a=score_a,
            score_b=score_b,
            criteria_comparison=criteria_comparison,
            confidence=confidence,
            created_at=datetime.now().isoformat(),
            reasoning="; ".join(reasoning_parts),
        )

        self._tests.append(result)
        logger.info(f"A/B test {result.test_id}: winner={winner} (A={score_a:.2f}, B={score_b:.2f})")
        return result

    def get_test_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent A/B test results."""
        results = []
        for test in self._tests[-limit:]:
            results.append({
                "test_id": test.test_id,
                "winner": test.winner,
                "score_a": test.score_a,
                "score_b": test.score_b,
                "confidence": test.confidence,
                "created_at": test.created_at,
                "reasoning": test.reasoning,
            })
        return results

    def get_stats(self) -> Dict[str, Any]:
        """Get A/B testing statistics."""
        if not self._tests:
            return {"total_tests": 0, "a_wins": 0, "b_wins": 0, "ties": 0}

        a_wins = sum(1 for t in self._tests if t.winner == "A")
        b_wins = sum(1 for t in self._tests if t.winner == "B")
        ties = sum(1 for t in self._tests if t.winner == "tie")
        avg_confidence = sum(t.confidence for t in self._tests) / len(self._tests)

        return {
            "total_tests": len(self._tests),
            "a_wins": a_wins,
            "b_wins": b_wins,
            "ties": ties,
            "avg_confidence": round(avg_confidence, 3),
        }


# ---------------------------------------------------------------------------
# Optimization Analytics
# ---------------------------------------------------------------------------

class OptimizationAnalytics:
    """Analytics engine for tracking optimization performance over time."""

    def __init__(self):
        self._runs: List[OptimizationRun] = []

    def record_run(self, run: OptimizationRun):
        """Record a completed optimization run."""
        self._runs.append(run)

    def get_summary(self) -> Dict[str, Any]:
        """Get overall optimization performance summary."""
        if not self._runs:
            return {
                "total_runs": 0,
                "avg_improvement": 0.0,
                "avg_iterations": 0,
                "avg_time_seconds": 0.0,
                "best_improvement": 0.0,
                "domain_breakdown": {},
            }

        improvements = [r.improvement_percentage for r in self._runs]
        times = [r.total_time_seconds for r in self._runs]
        iterations = [r.total_iterations for r in self._runs]

        domain_breakdown: Dict[str, Dict[str, Any]] = {}
        for run in self._runs:
            if run.domain not in domain_breakdown:
                domain_breakdown[run.domain] = {
                    "count": 0,
                    "avg_improvement": 0.0,
                    "total_improvement": 0.0,
                }
            domain_breakdown[run.domain]["count"] += 1
            domain_breakdown[run.domain]["total_improvement"] += run.improvement_percentage

        for domain, stats in domain_breakdown.items():
            stats["avg_improvement"] = round(
                stats["total_improvement"] / stats["count"], 1
            )

        return {
            "total_runs": len(self._runs),
            "avg_improvement": round(sum(improvements) / len(improvements), 1),
            "avg_iterations": round(sum(iterations) / len(iterations), 1),
            "avg_time_seconds": round(sum(times) / len(times), 2),
            "best_improvement": round(max(improvements), 1),
            "domain_breakdown": domain_breakdown,
            "recent_scores": [
                {"run_id": r.run_id, "initial": r.initial_score, "final": r.final_score}
                for r in self._runs[-10:]
            ],
        }

    def get_recent_runs(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent optimization runs."""
        return [
            {
                "run_id": r.run_id,
                "domain": r.domain,
                "strategy": r.strategy,
                "initial_score": r.initial_score,
                "final_score": r.final_score,
                "improvement": f"{r.improvement_percentage:.1f}%",
                "iterations": r.total_iterations,
                "time": f"{r.total_time_seconds:.1f}s",
                "created_at": r.created_at,
            }
            for r in self._runs[-limit:]
        ]


# ---------------------------------------------------------------------------
# Master Optimization Engine
# ---------------------------------------------------------------------------

class PromptOptimizationEngine:
    """
    Master optimization engine that orchestrates prompt improvement.

    Combines version control, A/B testing, and analytics for
    comprehensive prompt optimization workflows.
    """

    def __init__(self):
        self.version_control = PromptVersionControl()
        self.ab_testing = ABTestingEngine()
        self.analytics = OptimizationAnalytics()

    async def optimize(
        self,
        original_prompt: str,
        domain: str,
        evaluator,
        expert_agent,
        prompt_type: str = "raw",
        strategy: Optional[str] = None,
        max_iterations: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Run a full optimization pass on a prompt.

        Implements a multi-phase optimization pipeline:
          1. Evaluate original → establish baseline score
          2. Iterative improvement loop with adaptive early-stop
          3. Self-critique refinement pass (if score < 0.92)
          4. A/B test each candidate against the current best
          5. Record analytics and version history

        Prompt engineering techniques applied:
          - Chain-of-Thought: the expert agent decomposes reasoning steps
          - Self-Critique: a dedicated refinement pass targets weaknesses
          - Structured Decomposition: improvements are section-by-section
          - Negative Constraint Injection: weaknesses become "do not" rules
          - Meta-Verification: final prompts include a self-check instruction

        Args:
            original_prompt: The raw prompt to optimize
            domain: Classified domain of the prompt
            evaluator: PromptEvaluator instance
            expert_agent: Domain expert agent instance
            prompt_type: "raw" or "structured"
            strategy: Override optimization strategy
            max_iterations: Override max iterations

        Returns:
            Dict with optimized prompt, scores, versions, and analytics
        """
        strategy = strategy or getattr(settings, "optimization_strategy", "iterative")
        max_iterations = max_iterations or settings.max_evaluation_iterations
        start_time = time.time()
        run_id = f"opt_{uuid.uuid4().hex[:12]}"

        logger.info(f"Starting optimization run {run_id} (strategy={strategy}, domain={domain})")

        # --- Step 1: Create initial version & baseline ---
        initial_eval = await evaluator.evaluate_prompt(
            original_prompt=original_prompt,
            improved_prompt=original_prompt,
            domain=domain,
            prompt_type=prompt_type,
        )
        initial_score = initial_eval.get("overall_score", 0.5)

        v0 = self.version_control.create_version(
            prompt_text=original_prompt,
            domain=domain,
            quality_score=initial_score,
            strategy_used="original",
        )

        versions = [v0]
        ab_tests = []
        best_prompt = original_prompt
        best_score = initial_score
        best_version = v0
        best_eval = initial_eval

        # --- Step 2: Iterative improvement loop ---
        previous_score = initial_score
        for iteration in range(1, max_iterations + 1):
            logger.info(f"Optimization iteration {iteration}/{max_iterations} (best score: {best_score:.2f})")

            # Check convergence — already excellent
            if best_score >= settings.evaluation_threshold:
                logger.info(f"Score {best_score:.2f} meets threshold {settings.evaluation_threshold}. Converged.")
                break

            # Adaptive plateau detection — stop if gains are marginal
            if iteration > 1 and previous_score > 0:
                gain = (best_score - previous_score) / max(previous_score, 0.01)
                if gain < 0.015:  # < 1.5 % improvement
                    logger.info(f"Plateau detected (gain {gain:.3f}), moving to refinement phase.")
                    break

            previous_score = best_score

            # Improve prompt — the expert agent uses CoT + structured decomposition
            try:
                improvement_result = await expert_agent.improve_prompt(
                    original_prompt=best_prompt,
                    prompt_type=prompt_type,
                    key_topics=initial_eval.get("key_topics", []),
                )

                improved_prompt = improvement_result.get(
                    "improved_prompt",
                    improvement_result.get("solution", best_prompt),
                )

                if not improved_prompt or improved_prompt == best_prompt:
                    logger.info("No improvement produced, stopping.")
                    break

            except Exception as e:
                logger.warning(f"Improvement failed at iteration {iteration}: {e}")
                break

            # Evaluate improved prompt
            try:
                new_eval = await evaluator.evaluate_prompt(
                    original_prompt=original_prompt,
                    improved_prompt=improved_prompt,
                    domain=domain,
                    prompt_type=prompt_type,
                )
                new_score = new_eval.get("overall_score", 0.5)

            except Exception as e:
                logger.warning(f"Evaluation failed at iteration {iteration}: {e}")
                break

            # Create version
            new_version = self.version_control.create_version(
                prompt_text=improved_prompt,
                domain=domain,
                quality_score=new_score,
                parent_version=best_version.version_id,
                strategy_used=strategy,
                improvements=new_eval.get("strengths", []),
            )
            versions.append(new_version)

            # A/B test if enabled
            if getattr(settings, "enable_ab_testing", True):
                ab_result = self.ab_testing.create_test(
                    variant_a=best_version,
                    variant_b=new_version,
                    evaluation_a={"overall_score": best_score},
                    evaluation_b=new_eval,
                )
                ab_tests.append(ab_result)

            # Update best if improved
            if new_score > best_score:
                best_prompt = improved_prompt
                best_score = new_score
                best_version = new_version
                best_eval = new_eval
                logger.info(f"New best score: {best_score:.2f} (iteration {iteration})")
            else:
                logger.info(f"No improvement: {new_score:.2f} <= {best_score:.2f}")
                if strategy == "iterative":
                    break

        # --- Step 3: Self-Critique Refinement Pass ---
        # If the score is good but not excellent, run a targeted refinement
        # that converts weaknesses into negative constraints
        if best_score < 0.92 and best_eval.get("weaknesses"):
            logger.info(f"Running self-critique refinement pass (score={best_score:.2f})")
            try:
                weaknesses = best_eval.get("weaknesses", [])
                feedback = best_eval.get("specific_feedback", [])
                # Build a refinement-specific prompt that targets known weaknesses
                weakness_constraints = "\n".join(
                    f"  • FIX: {w}" for w in weaknesses[:5]
                )
                feedback_items = "\n".join(
                    f"  • {f}" for f in feedback[:5]
                )
                refinement_prompt_text = (
                    f"{best_prompt}\n\n"
                    f"---\n"
                    f"SELF-CRITIQUE — The following weaknesses were identified. "
                    f"Revise the prompt above to specifically address each one:\n"
                    f"{weakness_constraints}\n\n"
                    f"Actionable suggestions:\n{feedback_items}\n\n"
                    f"Produce the refined prompt only — no commentary."
                )

                refinement_result = await expert_agent.improve_prompt(
                    original_prompt=refinement_prompt_text,
                    prompt_type="structured",
                    key_topics=initial_eval.get("key_topics", []),
                )

                refined_prompt = refinement_result.get(
                    "improved_prompt",
                    refinement_result.get("solution", best_prompt),
                )

                if refined_prompt and refined_prompt != best_prompt:
                    refined_eval = await evaluator.evaluate_prompt(
                        original_prompt=original_prompt,
                        improved_prompt=refined_prompt,
                        domain=domain,
                        prompt_type="structured",
                    )
                    refined_score = refined_eval.get("overall_score", 0.5)

                    refined_version = self.version_control.create_version(
                        prompt_text=refined_prompt,
                        domain=domain,
                        quality_score=refined_score,
                        parent_version=best_version.version_id,
                        strategy_used="self_critique_refinement",
                        improvements=refined_eval.get("strengths", []),
                    )
                    versions.append(refined_version)

                    if refined_score > best_score:
                        logger.info(
                            f"Self-critique refinement improved score: "
                            f"{best_score:.2f} → {refined_score:.2f}"
                        )
                        best_prompt = refined_prompt
                        best_score = refined_score
                        best_version = refined_version
                        best_eval = refined_eval
                    else:
                        logger.info(
                            f"Self-critique did not improve: {refined_score:.2f} <= {best_score:.2f}"
                        )

            except Exception as e:
                logger.warning(f"Self-critique refinement failed: {e}")

        # --- Step 4: Record analytics ---
        total_time = time.time() - start_time
        improvement_pct = (
            ((best_score - initial_score) / initial_score * 100) if initial_score > 0 else 0
        )

        run = OptimizationRun(
            run_id=run_id,
            original_prompt=original_prompt,
            final_prompt=best_prompt,
            domain=domain,
            strategy=strategy,
            versions=versions,
            ab_tests=ab_tests,
            total_iterations=len(versions) - 1,
            total_time_seconds=round(total_time, 2),
            initial_score=initial_score,
            final_score=best_score,
            improvement_percentage=round(improvement_pct, 1),
            created_at=datetime.now().isoformat(),
        )
        self.analytics.record_run(run)

        logger.info(
            f"Optimization run {run_id} complete: {initial_score:.2f} → {best_score:.2f} "
            f"({improvement_pct:+.1f}%) in {total_time:.1f}s"
        )

        return {
            "run_id": run_id,
            "original_prompt": original_prompt,
            "optimized_prompt": best_prompt,
            "domain": domain,
            "strategy": strategy,
            "initial_score": initial_score,
            "final_score": best_score,
            "improvement_percentage": round(improvement_pct, 1),
            "iterations": len(versions) - 1,
            "time_seconds": round(total_time, 2),
            "versions": [
                {
                    "version_id": v.version_id,
                    "score": v.quality_score,
                    "strategy": v.strategy_used,
                }
                for v in versions
            ],
            "ab_tests": [
                {
                    "test_id": t.test_id,
                    "winner": t.winner,
                    "score_a": t.score_a,
                    "score_b": t.score_b,
                    "confidence": t.confidence,
                }
                for t in ab_tests
            ],
        }

    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get data for the optimization dashboard."""
        return {
            "analytics": self.analytics.get_summary(),
            "recent_runs": self.analytics.get_recent_runs(),
            "version_stats": self.version_control.get_stats(),
            "ab_test_stats": self.ab_testing.get_stats(),
            "ab_test_history": self.ab_testing.get_test_history(),
        }

    # ------------------------------------------------------------------
    # Persistence: Export / Import
    # ------------------------------------------------------------------

    def export_history(self, filepath: Optional[str] = None) -> Dict[str, Any]:
        """
        Export full optimization history as a JSON-serialisable dict.

        Args:
            filepath: Optional path to write JSON file. If None, returns dict only.

        Returns:
            Dict with versions, ab_tests, and analytics runs.
        """
        data = {
            "exported_at": datetime.now().isoformat(),
            "versions": [asdict(v) for v in self.version_control._versions.values()],
            "ab_tests": [
                {
                    "test_id": t.test_id,
                    "winner": t.winner,
                    "score_a": t.score_a,
                    "score_b": t.score_b,
                    "confidence": t.confidence,
                    "reasoning": t.reasoning,
                    "created_at": t.created_at,
                }
                for t in self.ab_testing._tests
            ],
            "runs": [
                {
                    "run_id": r.run_id,
                    "domain": r.domain,
                    "strategy": r.strategy,
                    "initial_score": r.initial_score,
                    "final_score": r.final_score,
                    "improvement_percentage": r.improvement_percentage,
                    "total_iterations": r.total_iterations,
                    "total_time_seconds": r.total_time_seconds,
                    "created_at": r.created_at,
                }
                for r in self.analytics._runs
            ],
        }

        if filepath:
            path = Path(filepath)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            logger.info(f"Optimization history exported to {filepath}")

        return data

    def import_history(self, filepath: str) -> bool:
        """
        Import optimization history from a JSON file.

        Args:
            filepath: Path to the JSON export file.

        Returns:
            True if import succeeded.
        """
        try:
            path = Path(filepath)
            if not path.exists():
                logger.warning(f"Import file not found: {filepath}")
                return False

            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Restore versions
            for v_data in data.get("versions", []):
                self.version_control.create_version(
                    prompt_text=v_data["prompt_text"],
                    domain=v_data["domain"],
                    quality_score=v_data["quality_score"],
                    parent_version=v_data.get("parent_version"),
                    strategy_used=v_data.get("strategy_used", "imported"),
                    improvements=v_data.get("improvements", []),
                    metadata={**v_data.get("metadata", {}), "imported": True},
                )

            logger.info(
                f"Imported {len(data.get('versions', []))} versions from {filepath}"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to import optimization history: {e}")
            return False

    # ------------------------------------------------------------------
    # Comparative Strategy
    # ------------------------------------------------------------------

    async def compare_strategies(
        self,
        prompt: str,
        domain: str,
        evaluator,
        expert_agent,
        prompt_type: str = "raw",
        strategies: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Run multiple optimization strategies on the same prompt and compare results.

        Args:
            prompt: The prompt to optimise.
            domain: Classified domain.
            evaluator: PromptEvaluator instance.
            expert_agent: Domain expert agent.
            prompt_type: "raw" or "structured".
            strategies: List of strategies to compare (default: iterative + evolutionary).

        Returns:
            Dict with per-strategy results and a recommended winner.
        """
        strategies = strategies or ["iterative"]
        results: Dict[str, Dict[str, Any]] = {}

        for strategy in strategies:
            try:
                result = await self.optimize(
                    original_prompt=prompt,
                    domain=domain,
                    evaluator=evaluator,
                    expert_agent=expert_agent,
                    prompt_type=prompt_type,
                    strategy=strategy,
                    max_iterations=settings.max_evaluation_iterations,
                )
                results[strategy] = result
            except Exception as e:
                logger.warning(f"Strategy '{strategy}' failed: {e}")
                results[strategy] = {"error": str(e), "final_score": 0}

        # Determine winner
        best_strategy = max(results, key=lambda s: results[s].get("final_score", 0))
        best_result = results[best_strategy]

        return {
            "strategies_tested": strategies,
            "results": results,
            "recommended_strategy": best_strategy,
            "best_score": best_result.get("final_score", 0),
            "best_prompt": best_result.get("optimized_prompt", prompt),
        }


# Global instance
optimization_engine = PromptOptimizationEngine()
