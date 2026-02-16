# Changelog

All notable changes to CortexaAI are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [3.0.0] — 2026-02-14 (Production-Ready V3.0 Release)

### Added

- **16 Core Feature Modules** — full production feature suite in `core/`:
  - `auth.py` — API key authentication with CRUD management
  - `batch.py` — concurrent batch prompt processing with progress tracking
  - `complexity.py` — prompt complexity analysis and scoring
  - `database.py` — SQLite persistence layer with WAL mode
  - `error_recovery.py` — circuit breakers, dead letter queues, retry logic
  - `finetuning.py` — fine-tuning dataset generation from optimized prompts
  - `language.py` — multi-language prompt support and detection
  - `marketplace.py` — template marketplace with sharing and ratings
  - `plugins.py` — extensible plugin system for custom processors
  - `prompt_builder.py` — visual/programmatic prompt builder
  - `regression.py` — prompt regression testing across versions
  - `similarity.py` — prompt similarity search and deduplication
  - `streaming.py` — SSE streaming for real-time workflow progress
  - `templates.py` — prompt template management and versioning
  - `webhooks.py` — event-driven webhook notifications
- **86 API Routes** — comprehensive REST API covering all features
- **SQLite Database** — persistent storage for templates, API keys, analytics, and batch jobs
- **ARCHITECTURE.md** — System overview diagrams, component descriptions, data flow, and agent interaction maps
- **PREFLIGHT_CHECKLIST.md** — Pre-deployment validation checklist covering docs, security, deps, and config
- **.env.example** — Comprehensive environment variable template with all providers and settings documented
- **Prompt Engineering System Enhancement** — production-grade prompt rewrites for all agents:
  - Rubric-based evaluation with scoring bands (Exceptional → Poor)
  - Structured JSON output formats with strict schemas
  - Enhanced fallback classifier covering all 6 domains
  - Weighted scoring heuristic for prompt type classification
- **`InputValidationError`** exception class for input validation failures
- **Security events tracking** in exception `to_dict()` serialization
- **`__repr__`** methods on all custom exception classes

### Changed

- **Pydantic v1 → v2 migration** in `langgraph_expert.py` and `coordinator.py`
- **Workflow architecture** — replaced 5 `__import__()` anti-patterns with proper module-level imports
- **`check_threshold_node`** — no longer mutates state dict directly; returns update dict per LangGraph convention
- **`evaluate_node`** — hardened key resolution (`improved_prompt` → `solution` → fallback)
- **Coordinator** — extracted `_handle_workflow_exception()` helper, eliminating ~50 lines of duplicated error handling
- **`sanitize_json_response`** — handles trailing commas and bare markdown fences
- **`is_retryable_error`** — expanded with HTTP status code checking and additional indicators
- **Fallback classifier** — expanded from 2 to 6 domain patterns
- **README.md** — complete rewrite with accurate project structure, v3.0 features, deployment table
- **CONTRIBUTING.md** — updated to reflect current architecture and tooling
- **CHANGELOG.md** — consolidated all versions, documented as v3.0.0 production-ready
- **LICENSE** — updated copyright to 2025-2026 Abdulrahman Baidaq
- **Agent documentation** — updated header comments on all agent files
- **`pyproject.toml`** — updated to v3.0.0 with Production/Stable status
- **`.gitignore`** — reorganized with clear section headers and comprehensive coverage
- **Repository structure** — reorganized for professional layout:
  - `tools/` → `scripts/` (clearer naming convention)
  - Development reports moved to `docs/`
  - Runtime data cleaned from version control

### Removed

- **Dockerfile.hf** — Removed unused Hugging Face Spaces Dockerfile
- **.koyeb.yaml** — Removed unused Koyeb deployment configuration
- **docs/deployment/koyeb.md** — Removed Koyeb deployment guide
- **docs/deployment/huggingface.md** — Removed Hugging Face deployment guide

### Fixed

- State mutation bug in `check_threshold_node` (was modifying input state directly)
- Pydantic deprecation warnings from v1 compatibility imports
- Missing key handling in `evaluate_node` causing KeyError on certain agent outputs
- License copyright holder updated from generic to project author

---

## [2.0.0] — 2025-07-16

### Added

- **Multi-LLM Provider System** — support for 6 providers (Google Gemini, OpenAI, Anthropic, Groq, DeepSeek, OpenRouter) with automatic fallback routing and health monitoring
- **Prompt Optimization Engine** — iterative refinement with quality convergence and multiple optimization strategies
- **A/B Testing** — compare prompt variants with statistical confidence scoring and winner determination
- **Version Control** — full prompt version history with lineage tracking and rollback
- **Optimization Analytics** — run metrics, domain breakdown, improvement trends, and dashboard data
- **API Endpoints** — `/api/providers`, `/api/optimization/dashboard`, `/api/optimization/analytics`, `/api/optimization/ab-tests`, `/api/optimization/versions`
- **CI/CD Pipeline** — GitHub Actions workflow with lint, test, and security scan stages
- **Project Metadata** — `pyproject.toml` with Ruff, pytest, coverage, mypy, and bandit configurations
- **Comprehensive Test Suite** — tests for LLM providers, optimization engine, A/B testing, version control, analytics
- **Shared Test Fixtures** — `conftest.py` with mock agents, sample data, and environment isolation
- **Professional README** — badges, architecture diagram, provider comparison, API reference, project structure

### Changed

- Updated `.env.example` with 6 provider keys, documentation links, and optimization settings
- Updated `requirements.txt` with `langchain-anthropic`, `langchain-groq`, `pytest-asyncio`, `pytest-cov`, `ruff`
- Updated `config.py` Settings class with Groq, DeepSeek, OpenRouter key fields and optimization settings
- Updated `get_model_config()` to support all 6 providers with correct default models
- Updated `main.py` with new imports, FastAPI metadata, provider info logging, and SPA serving
- Updated `CONTRIBUTING.md` with CortexaAI branding, Python 3.10+ requirement, and updated commands

## [1.0.0] — 2025-06-01

### Added

- Initial multi-agent prompt engineering system
- Classifier, Expert, Evaluator agent pipeline
- LangGraph workflow orchestration
- FastAPI backend with async processing
- React + TypeScript + Tailwind CSS frontend
- Chat interface with session management
- Memory/RAG system for conversation context
- Security layer (PII detection, injection prevention, rate limiting)
- Caching, circuit breakers, dead letter queues
- Basic test suite
