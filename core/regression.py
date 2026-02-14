"""
Regression Testing for Prompts — CortexaAI.

Define test suites with canonical inputs, run them through the pipeline,
and compare results against saved baselines to detect regressions.
"""

import json
import time
import uuid
from typing import Dict, Any, List, Optional

from config.config import get_logger

logger = get_logger(__name__)


class RegressionTestRunner:
    """Manage and run prompt regression tests."""

    def _get_db(self):
        from core.database import db
        return db

    # ── Suite Management ─────────────────────────────────────────────────
    def create_suite(
        self,
        name: str,
        domain: str,
        description: str = "",
        test_cases: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Create a regression test suite."""
        suite_id = f"rs_{uuid.uuid4().hex[:8]}"
        cases = test_cases or []
        # Assign IDs to cases
        for i, case in enumerate(cases):
            if "id" not in case:
                case["id"] = f"tc_{i+1}"

        db = self._get_db()
        db.execute(
            """INSERT INTO regression_suites
               (id, name, domain, description, test_cases, baseline, created_at, last_run)
               VALUES (?, ?, ?, ?, ?, ?, ?, NULL)""",
            (suite_id, name, domain, description, json.dumps(cases), "{}", time.time()),
        )
        logger.info(f"Created regression suite: {name} ({suite_id}) with {len(cases)} cases")
        return {
            "id": suite_id,
            "name": name,
            "domain": domain,
            "description": description,
            "test_cases": cases,
            "case_count": len(cases),
        }

    def get_suite(self, suite_id: str) -> Optional[Dict[str, Any]]:
        db = self._get_db()
        row = db.fetch_one(
            "SELECT id, name, domain, description, test_cases, baseline, created_at, last_run FROM regression_suites WHERE id = ?",
            (suite_id,),
        )
        if not row:
            return None
        return {
            "id": row[0],
            "name": row[1],
            "domain": row[2],
            "description": row[3],
            "test_cases": json.loads(row[4]) if row[4] else [],
            "baseline": json.loads(row[5]) if row[5] else {},
            "created_at": row[6],
            "last_run": row[7],
        }

    def list_suites(self) -> List[Dict[str, Any]]:
        db = self._get_db()
        rows = db.fetch_all(
            "SELECT id, name, domain, description, created_at, last_run FROM regression_suites ORDER BY created_at DESC"
        )
        return [
            {
                "id": r[0],
                "name": r[1],
                "domain": r[2],
                "description": r[3],
                "created_at": r[4],
                "last_run": r[5],
            }
            for r in rows
        ]

    def delete_suite(self, suite_id: str) -> bool:
        db = self._get_db()
        db.execute("DELETE FROM regression_suites WHERE id = ?", (suite_id,))
        return True

    def add_test_case(
        self, suite_id: str, input_prompt: str, expected_keywords: Optional[List[str]] = None, min_score: float = 0.7
    ) -> Optional[Dict[str, Any]]:
        """Add a test case to a suite."""
        suite = self.get_suite(suite_id)
        if not suite:
            return None
        cases = suite["test_cases"]
        case = {
            "id": f"tc_{len(cases)+1}",
            "input": input_prompt,
            "expected_keywords": expected_keywords or [],
            "min_score": min_score,
        }
        cases.append(case)
        db = self._get_db()
        db.execute(
            "UPDATE regression_suites SET test_cases = ? WHERE id = ?",
            (json.dumps(cases), suite_id),
        )
        return case

    # ── Running Tests ────────────────────────────────────────────────────
    async def run_suite(self, suite_id: str, processor_fn=None) -> Dict[str, Any]:
        """
        Run all test cases in a suite.

        Args:
            suite_id: Suite to run
            processor_fn: async fn(prompt_text) -> Dict with 'improved_prompt', 'evaluation_score'

        Returns:
            Run results with pass/fail per case.
        """
        suite = self.get_suite(suite_id)
        if not suite:
            return {"error": "Suite not found"}

        results = []
        passed = 0
        failed = 0
        baseline = suite.get("baseline", {})

        for case in suite["test_cases"]:
            case_result = {
                "case_id": case["id"],
                "input": case["input"],
                "status": "skipped",
                "score": 0.0,
                "keywords_found": [],
                "keywords_missing": [],
            }

            if processor_fn:
                try:
                    output = await processor_fn(case["input"])
                    improved = output.get("improved_prompt", "")
                    score = output.get("evaluation_score", 0.0)

                    # Keyword check
                    found = []
                    missing = []
                    for kw in case.get("expected_keywords", []):
                        if kw.lower() in improved.lower():
                            found.append(kw)
                        else:
                            missing.append(kw)

                    # Score check
                    meets_score = score >= case.get("min_score", 0.7)

                    # Regression check (compare to baseline)
                    baseline_score = baseline.get(case["id"], {}).get("score", 0.0)
                    regression = baseline_score > 0 and score < baseline_score * 0.9

                    case_result.update({
                        "status": "passed" if (meets_score and not missing and not regression) else "failed",
                        "score": score,
                        "keywords_found": found,
                        "keywords_missing": missing,
                        "meets_min_score": meets_score,
                        "regression_detected": regression,
                        "baseline_score": baseline_score,
                        "improved_prompt_preview": improved[:200],
                    })

                    if case_result["status"] == "passed":
                        passed += 1
                    else:
                        failed += 1
                except Exception as e:
                    case_result["status"] = "error"
                    case_result["error"] = str(e)
                    failed += 1
            else:
                case_result["status"] = "skipped"

            results.append(case_result)

        run_result = {
            "suite_id": suite_id,
            "suite_name": suite["name"],
            "total_cases": len(suite["test_cases"]),
            "passed": passed,
            "failed": failed,
            "skipped": len(suite["test_cases"]) - passed - failed,
            "pass_rate": round(passed / max(1, len(suite["test_cases"])) * 100, 1),
            "results": results,
            "run_at": time.time(),
        }

        # Update last_run
        db = self._get_db()
        db.execute(
            "UPDATE regression_suites SET last_run = ? WHERE id = ?",
            (time.time(), suite_id),
        )

        return run_result

    # ── Baseline Management ──────────────────────────────────────────────
    def save_baseline(self, suite_id: str, run_results: Dict[str, Any]) -> bool:
        """Save current results as baseline for future comparison."""
        baseline = {}
        for case_result in run_results.get("results", []):
            baseline[case_result["case_id"]] = {
                "score": case_result.get("score", 0.0),
                "saved_at": time.time(),
            }
        db = self._get_db()
        db.execute(
            "UPDATE regression_suites SET baseline = ? WHERE id = ?",
            (json.dumps(baseline), suite_id),
        )
        return True


# Global instance
regression_runner = RegressionTestRunner()
