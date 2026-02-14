# Prompt Engineering System — Production-Grade Enhancement Report

**Date:** 2026-02-14  
**Scope:** Comprehensive analysis and enhancement of every prompt, template, workflow,
agent, and utility in the CortexaAI prompt engineering system.

---

## 1. Summary of Changes

### Files Modified (9)

| File | Changes |
|------|---------|
| `src/workflow.py` | Fixed 5 `__import__()` anti-patterns → proper module-level imports; fixed `check_threshold_node` direct state mutation; improved module docstring with architecture diagram; enhanced `evaluate_node` key resolution to handle both `solution` and `improved_prompt` keys |
| `agents/langgraph_expert.py` | Migrated `pydantic.v1` → `pydantic` v2; fixed shebang typo; rewrote system prompt with structured rubric, scoring band guidelines, strict JSON output format, and step-by-step transformation instructions |
| `agents/evaluator.py` | Rewrote evaluation prompt with rubric-based scoring bands (0.95-1.0 Exceptional → <0.50 Poor), evidence-based strength/weakness requirements, strict JSON output, and comparison analysis |
| `agents/classifier.py` | Rewrote classification prompt with structured sections and strict JSON output; enhanced fallback classifier to cover all 6 domains (was only 2); improved `classify_prompt_type` heuristic with weighted scoring |
| `agents/base_expert.py` | Rewrote both improvement prompts (standard + context-enhanced) with structured sections, explicit principles, strict JSON output format |
| `agents/coordinator.py` | Migrated `pydantic.v1` → `pydantic` v2; extracted `_handle_workflow_exception()` helper — eliminated ~50 lines of duplicated error handling; enhanced advanced-mode system prompt |
| `agents/utils.py` | Enhanced `sanitize_json_response` to handle trailing commas and bare markdown fences; enhanced `is_retryable_error` with HTTP status code checking and more indicators |
| `agents/exceptions.py` | Added `__repr__` to base exception; `to_dict()` now includes `security_events`; added `InputValidationError` class; improved module docstring with hierarchy diagram |
| `PROMPT_SYSTEM_ENHANCEMENT.md` | This summary document |

---

## 2. Detailed Changes

### 2.1 Workflow Architecture (`src/workflow.py`)

**Anti-pattern fixes:**
- Replaced 5 occurrences of `__import__('time')` and `__import__('datetime')` with proper
  module-level `import time` and `from datetime import datetime`.
- `check_threshold_node` no longer mutates `state` dict directly — it returns an update
  dict per LangGraph convention.

**Key resolution hardening:**
- `evaluate_node` now resolves the improved prompt from multiple possible keys:
  `improved_prompt` (BaseExpertAgent) → `solution` (LangGraphExpert) → `state["improved_prompt"]`
  → fallback to `original_prompt`.

**Documentation:**
- Module docstring rewritten with ASCII architecture diagram showing the 7-node graph
  flow and capability highlights.

### 2.2 LangGraph Expert (`agents/langgraph_expert.py`)

**Pydantic migration:** `from pydantic.v1 import ...` → `from pydantic import ...`

**System prompt rewrite:**
- Added structured sections with Unicode box-drawing separators for visual clarity.
- Explicit transformation steps numbered 1–4 with arrow notation to JSON fields.
- Scoring rubric explicitly references the 6 evaluation criteria.
- Output format section demands raw JSON (no markdown fences) for reliable parsing.
- Added critical warning against verbatim copying.

### 2.3 Evaluator (`agents/evaluator.py`)

**Evaluation prompt rewrite:**
- Scoring rubric with defined bands (Exceptional/Strong/Acceptable/Weak/Poor).
- Each criterion has detailed grading guidance (e.g. "Deduct for vague qualifiers").
- Evidence-based strength/weakness requirements.
- Strict JSON output instruction (no markdown fences).

### 2.4 Classifier (`agents/classifier.py`)

**Classification prompt rewrite:**
- Structured sections with clear instructions for new-domain proposals.
- Confidence threshold guidance (≥ 0.6 to use existing domain).
- Key-topic extraction capped at 5 terms.
- Strict JSON output.

**Fallback classifier enhancement:**
- Now scores ALL 6 known domains by keyword overlap (was hard-coded for only 2).
- Returns adaptive confidence based on match count.
- Extracts matched topics from winning domain.

**Prompt-type classifier enhancement:**
- Weighted composite scoring with structural keywords, list lines, heading lines,
  line count, and overall length.
- Capped contributions to prevent single-feature domination.

### 2.5 Base Expert (`agents/base_expert.py`)

**Improvement prompt rewrites (both standard and context-enhanced):**
- Structured sections with Unicode separators.
- Explicit prompt-engineering principles (6 items).
- Task instructions with bullet-point requirements.
- Strict JSON output with field documentation.

### 2.6 Coordinator (`agents/coordinator.py`)

**Pydantic migration:** `from pydantic.v1 import ...` → `from pydantic import ...`

**Error handling deduplication:**
- New `_handle_workflow_exception()` method handles all exception types uniformly.
- Replaced ~50 lines of duplicated try/except blocks in both `process_prompt()` and
  `process_prompt_with_memory()`.

**Advanced-mode prompt enhancement:**
- Rewritten from informal instructions to a professional workflow description.
- Explicit guidance on what clarifying questions should target.

### 2.7 Utils (`agents/utils.py`)

**`sanitize_json_response` improvements:**
- Now strips both ```` ```json ``` ```` and ```` ``` ``` ```` fences.
- Removes trailing commas before `}` and `]` (a frequent LLM error).
- Better boundary detection for nested JSON objects.

**`is_retryable_error` improvements:**
- Added HTTP status code attribute checking (`status_code`, `response.status_code`).
- Expanded retryable indicator list: `429`, `timed out`, `deadline exceeded`,
  `resource exhausted`, `overloaded`, `bad gateway`.

### 2.8 Exceptions (`agents/exceptions.py`)

- Added `__repr__` to `AgenticSystemError` for better debugging in logs/REPL.
- `to_dict()` now conditionally includes `security_events` field.
- New `InputValidationError` class for prompt validation failures.
- Module docstring shows the full exception hierarchy.

---

## 3. Design Principles Applied

1. **Single Responsibility** — each prompt has one clear purpose.
2. **Fail Gracefully** — fallbacks at every level; nodes never crash the pipeline.
3. **DRY** — extracted shared error handling; unified retry logic via `is_retryable_error`.
4. **Explicit > Implicit** — all JSON output instructions say "strict JSON, no markdown fences".
5. **Evidence-Based Scoring** — evaluation prompt demands evidence for strengths/weaknesses.
6. **Defence in Depth** — security sanitisation, rate limiting, circuit breakers, and input
   validation all remain intact; enhanced with `InputValidationError`.

---

## 4. Recommendations for Future Work

| Priority | Recommendation |
|----------|---------------|
| High | Add per-node **timeout** (`asyncio.wait_for`) to workflow nodes to prevent hangs |
| High | Implement **structured logging** with correlation IDs that flow through all nodes |
| Medium | Add **token-efficiency** as a 7th evaluation criterion (penalise unnecessarily long prompts) |
| Medium | Extend `classify_prompt_type` to use the LLM for borderline cases |
| Medium | Add **A/B testing** infrastructure to compare prompt templates against each other |
| Low | Migrate remaining `Dict[str, Any]` return types to Pydantic response models |
| Low | Add property-based tests (Hypothesis) for `sanitize_json_response` edge cases |
