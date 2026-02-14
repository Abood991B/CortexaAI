"""Tests for the Prompt Optimization Engine."""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from core.optimization import (
    PromptVersionControl,
    ABTestingEngine,
    OptimizationAnalytics,
    PromptOptimizationEngine,
    PromptVersion,
    OptimizationRun,
)


class TestPromptVersionControl:
    """Tests for prompt version tracking."""

    def test_create_version(self):
        """Should create a version with correct fields."""
        vc = PromptVersionControl()
        version = vc.create_version(
            prompt_text="Write a sort function",
            domain="software_engineering",
            quality_score=0.75,
            strategy_used="iterative",
        )
        assert version.version_id.startswith("v_")
        assert version.prompt_text == "Write a sort function"
        assert version.domain == "software_engineering"
        assert version.quality_score == 0.75
        assert version.parent_version is None

    def test_create_version_with_parent(self):
        """Should track parent version for lineage."""
        vc = PromptVersionControl()
        v1 = vc.create_version("prompt v1", "general", 0.5)
        v2 = vc.create_version("prompt v2", "general", 0.8, parent_version=v1.version_id)
        assert v2.parent_version == v1.version_id

    def test_get_version(self):
        """Should retrieve a version by ID."""
        vc = PromptVersionControl()
        v = vc.create_version("test", "general", 0.5)
        retrieved = vc.get_version(v.version_id)
        assert retrieved is not None
        assert retrieved.version_id == v.version_id

    def test_get_version_nonexistent(self):
        """Should return None for unknown version."""
        vc = PromptVersionControl()
        assert vc.get_version("nonexistent") is None

    def test_get_version_history(self):
        """Should return full ancestor chain."""
        vc = PromptVersionControl()
        v1 = vc.create_version("v1", "general", 0.5)
        v2 = vc.create_version("v2", "general", 0.7, parent_version=v1.version_id)
        v3 = vc.create_version("v3", "general", 0.9, parent_version=v2.version_id)

        history = vc.get_version_history(v3.version_id)
        assert len(history) == 3
        assert history[0].version_id == v1.version_id
        assert history[2].version_id == v3.version_id

    def test_get_best_version(self):
        """Should return highest-scoring version."""
        vc = PromptVersionControl()
        vc.create_version("low", "general", 0.3)
        best = vc.create_version("high", "general", 0.95)
        vc.create_version("mid", "general", 0.6)

        result = vc.get_best_version()
        assert result.version_id == best.version_id

    def test_get_best_version_by_domain(self):
        """Should filter best version by domain."""
        vc = PromptVersionControl()
        vc.create_version("se high", "software_engineering", 0.9)
        ds_best = vc.create_version("ds high", "data_science", 0.95)
        vc.create_version("ds low", "data_science", 0.6)

        result = vc.get_best_version(domain="data_science")
        assert result.version_id == ds_best.version_id

    def test_get_stats(self):
        """Should return correct statistics."""
        vc = PromptVersionControl()
        vc.create_version("a", "general", 0.5)
        vc.create_version("b", "software_engineering", 0.8)
        vc.create_version("c", "general", 0.9)

        stats = vc.get_stats()
        assert stats["total_versions"] == 3
        assert stats["domains"]["general"] == 2
        assert stats["domains"]["software_engineering"] == 1
        assert stats["best_score"] == 0.9
        assert stats["worst_score"] == 0.5

    def test_get_stats_empty(self):
        """Should handle empty version store."""
        vc = PromptVersionControl()
        stats = vc.get_stats()
        assert stats["total_versions"] == 0


class TestABTestingEngine:
    """Tests for A/B testing."""

    def _make_version(self, text: str, score: float) -> PromptVersion:
        return PromptVersion(
            version_id=f"v_{text}",
            prompt_text=text,
            domain="general",
            quality_score=score,
            created_at=datetime.now().isoformat(),
        )

    def test_create_test_a_wins(self):
        """Variant A should win when score_a is significantly higher."""
        engine = ABTestingEngine()
        va = self._make_version("a", 0.9)
        vb = self._make_version("b", 0.7)

        result = engine.create_test(
            va, vb,
            evaluation_a={"overall_score": 0.9, "criteria_scores": {"clarity": 0.9}},
            evaluation_b={"overall_score": 0.7, "criteria_scores": {"clarity": 0.7}},
        )
        assert result.winner == "A"
        assert result.score_a == 0.9
        assert result.score_b == 0.7
        assert result.confidence > 0

    def test_create_test_b_wins(self):
        """Variant B should win when score_b is significantly higher."""
        engine = ABTestingEngine()
        va = self._make_version("a", 0.6)
        vb = self._make_version("b", 0.85)

        result = engine.create_test(
            va, vb,
            evaluation_a={"overall_score": 0.6},
            evaluation_b={"overall_score": 0.85},
        )
        assert result.winner == "B"

    def test_create_test_tie(self):
        """Should be a tie when scores are very close."""
        engine = ABTestingEngine()
        va = self._make_version("a", 0.82)
        vb = self._make_version("b", 0.84)

        result = engine.create_test(
            va, vb,
            evaluation_a={"overall_score": 0.82},
            evaluation_b={"overall_score": 0.84},
        )
        assert result.winner == "tie"

    def test_get_stats(self):
        """Should track test statistics."""
        engine = ABTestingEngine()
        va = self._make_version("a", 0.9)
        vb = self._make_version("b", 0.7)
        engine.create_test(va, vb, {"overall_score": 0.9}, {"overall_score": 0.7})
        engine.create_test(va, vb, {"overall_score": 0.7}, {"overall_score": 0.9})

        stats = engine.get_stats()
        assert stats["total_tests"] == 2
        assert stats["a_wins"] + stats["b_wins"] + stats["ties"] == 2

    def test_get_stats_empty(self):
        """Should handle no tests."""
        engine = ABTestingEngine()
        stats = engine.get_stats()
        assert stats["total_tests"] == 0


class TestOptimizationAnalytics:
    """Tests for optimization analytics."""

    def _make_run(self, initial: float, final: float, domain: str = "general") -> OptimizationRun:
        improvement = ((final - initial) / initial * 100) if initial > 0 else 0
        return OptimizationRun(
            run_id=f"opt_{domain}",
            original_prompt="original",
            final_prompt="final",
            domain=domain,
            strategy="iterative",
            versions=[],
            ab_tests=[],
            total_iterations=2,
            total_time_seconds=5.0,
            initial_score=initial,
            final_score=final,
            improvement_percentage=round(improvement, 1),
            created_at=datetime.now().isoformat(),
        )

    def test_record_and_summarize(self):
        """Should record runs and produce summary."""
        analytics = OptimizationAnalytics()
        analytics.record_run(self._make_run(0.5, 0.85))
        analytics.record_run(self._make_run(0.6, 0.9, "software_engineering"))

        summary = analytics.get_summary()
        assert summary["total_runs"] == 2
        assert summary["avg_improvement"] > 0
        assert "general" in summary["domain_breakdown"]
        assert "software_engineering" in summary["domain_breakdown"]

    def test_summary_empty(self):
        """Should handle empty analytics."""
        analytics = OptimizationAnalytics()
        summary = analytics.get_summary()
        assert summary["total_runs"] == 0
        assert summary["avg_improvement"] == 0.0

    def test_recent_runs(self):
        """Should return recent runs in correct format."""
        analytics = OptimizationAnalytics()
        analytics.record_run(self._make_run(0.5, 0.85))

        recent = analytics.get_recent_runs()
        assert len(recent) == 1
        assert "run_id" in recent[0]
        assert "improvement" in recent[0]


class TestPromptOptimizationEngine:
    """Tests for the master optimization engine."""

    def test_initialization(self):
        """Engine should initialize all sub-systems."""
        engine = PromptOptimizationEngine()
        assert engine.version_control is not None
        assert engine.ab_testing is not None
        assert engine.analytics is not None

    def test_get_dashboard_data(self):
        """Dashboard should include all sections."""
        engine = PromptOptimizationEngine()
        data = engine.get_dashboard_data()

        assert "analytics" in data
        assert "recent_runs" in data
        assert "version_stats" in data
        assert "ab_test_stats" in data
        assert "ab_test_history" in data

    @pytest.mark.asyncio
    async def test_optimize_basic_flow(self):
        """Should run optimization loop with mocked agents."""
        engine = PromptOptimizationEngine()

        # Mock evaluator
        mock_evaluator = AsyncMock()
        mock_evaluator.evaluate_prompt = AsyncMock(side_effect=[
            # Initial evaluation
            {"overall_score": 0.5, "criteria_scores": {}, "passes_threshold": False, "key_topics": []},
            # After first improvement
            {"overall_score": 0.85, "criteria_scores": {}, "passes_threshold": True, "strengths": ["Good structure"]},
        ])

        # Mock expert
        mock_expert = AsyncMock()
        mock_expert.improve_prompt = AsyncMock(return_value={
            "improved_prompt": "Enhanced: Write a well-documented sort function",
            "solution": "Enhanced: Write a well-documented sort function",
        })

        with patch("core.optimization.settings") as mock_settings:
            mock_settings.optimization_strategy = "iterative"
            mock_settings.max_evaluation_iterations = 3
            mock_settings.evaluation_threshold = 0.8
            mock_settings.enable_ab_testing = True

            result = await engine.optimize(
                original_prompt="Write a sort function",
                domain="software_engineering",
                evaluator=mock_evaluator,
                expert_agent=mock_expert,
                prompt_type="raw",
            )

        assert result["initial_score"] == 0.5
        assert result["final_score"] == 0.85
        assert result["improvement_percentage"] > 0
        assert len(result["versions"]) >= 2
        assert "run_id" in result
