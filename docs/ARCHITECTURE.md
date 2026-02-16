# CortexaAI — Architecture

## System Overview

```
┌──────────────────────────────────────────────────────────────────────────┐
│                          CLIENT LAYER                                    │
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │          React Frontend (Vite + TypeScript + Tailwind)          │    │
│  │                                                                 │    │
│  │  ┌──────────────┐  ┌──────────────────┐  ┌────────────────┐   │    │
│  │  │    Chat UI    │  │ Analytics Panel  │  │ Provider Mgmt  │   │    │
│  │  └──────────────┘  └──────────────────┘  └────────────────┘   │    │
│  └─────────────────────────────┬───────────────────────────────────┘    │
│                                │                                         │
│                     REST API (86 routes)                                 │
│                         + SSE Streaming                                  │
└────────────────────────────────┼─────────────────────────────────────────┘
                                 │
┌────────────────────────────────▼─────────────────────────────────────────┐
│                          API LAYER                                       │
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                    FastAPI Application                          │    │
│  │                    (src/main.py — 86 routes)                    │    │
│  │                                                                 │    │
│  │  Middleware: CORS │ Rate Limiting │ PII Detection │ Auth        │    │
│  └─────────────────────────────┬───────────────────────────────────┘    │
│                                │                                         │
└────────────────────────────────┼─────────────────────────────────────────┘
                                 │
┌────────────────────────────────▼─────────────────────────────────────────┐
│                       ORCHESTRATION LAYER                                │
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │              LangGraph Workflow Engine                          │    │
│  │              (src/workflow.py — 7-node StateGraph)              │    │
│  │                                                                 │    │
│  │  classify → create_expert → improve → evaluate → check         │    │
│  │       ↓                                            ↓            │    │
│  │  error_handler                              finalize → END      │    │
│  │                                        (loop if below threshold)│    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                          │
│  ┌──────────────────────┐  ┌───────────────────────────────────────┐    │
│  │  WorkflowCoordinator │  │  Cancellation & Progress Tracking    │    │
│  │  (agents/coordinator)│  │  (asyncio.Event + SSE streaming)     │    │
│  └──────────────────────┘  └───────────────────────────────────────┘    │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
                                 │
┌────────────────────────────────▼─────────────────────────────────────────┐
│                          AGENT LAYER                                     │
│                                                                          │
│  ┌──────────────┐   ┌──────────────────┐   ┌───────────────────┐       │
│  │  Classifier   │   │  Expert Agents    │   │    Evaluator      │       │
│  │  Agent        │   │                  │   │    Agent           │       │
│  │              │   │  ┌────────────┐  │   │                   │       │
│  │  • Domain     │──▶│  │ Software   │  │──▶│  • 6 criteria     │       │
│  │    detection  │   │  │ Data Sci   │  │   │    scoring        │       │
│  │  • Confidence │   │  │ Education  │  │   │  • Rubric-based   │       │
│  │    scoring    │   │  │ Business   │  │   │    evaluation     │       │
│  │  • Key topic  │   │  │ Creative   │  │   │  • Improvement    │       │
│  │    extraction │   │  │ Report     │  │   │    feedback       │       │
│  │  • 6 domains  │   │  │ LangGraph  │  │   │  • Plateau        │       │
│  │              │   │  │ (dynamic)  │  │   │    detection      │       │
│  └──────────────┘   │  └────────────┘  │   └───────────────────┘       │
│                      └──────────────────┘                                │
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │  Shared: agents/utils.py │ agents/exceptions.py │ memory/       │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
                                 │
┌────────────────────────────────▼─────────────────────────────────────────┐
│                        FEATURE MODULES (core/)                           │
│                                                                          │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐  │
│  │ Optimization │ │    Auth      │ │    Batch     │ │   Streaming  │  │
│  │ • A/B tests  │ │ • API keys   │ │ • Concurrent │ │ • SSE events │  │
│  │ • Versioning │ │ • CRUD mgmt  │ │ • Progress   │ │ • Real-time  │  │
│  │ • Analytics  │ │ • Rate limit │ │ • Tracking   │ │ • Per-node   │  │
│  └──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘  │
│                                                                          │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐  │
│  │  Templates   │ │  Plugins     │ │  Webhooks    │ │  Similarity  │  │
│  │ • Marketplace│ │ • Extensible │ │ • Event-     │ │ • Dedup      │  │
│  │ • Versioning │ │ • Custom     │ │   driven     │ │ • Semantic   │  │
│  │ • Sharing    │ │   processors │ │ • Callbacks  │ │   search     │  │
│  └──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘  │
│                                                                          │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐  │
│  │  Complexity  │ │  Language    │ │  Regression  │ │Error Recovery│  │
│  │ • Analysis   │ │ • Detection  │ │ • Testing    │ │ • Circuit    │  │
│  │ • Scoring    │ │ • Multi-lang │ │ • Comparison │ │   breakers   │  │
│  │              │ │              │ │              │ │ • DLQ        │  │
│  └──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘  │
│                                                                          │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐                    │
│  │  Database    │ │ Fine-tuning  │ │Prompt Builder│                    │
│  │ • SQLite WAL │ │ • Dataset    │ │ • Visual     │                    │
│  │ • Persistent │ │   generation │ │ • Programmat │                    │
│  └──────────────┘ └──────────────┘ └──────────────┘                    │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
                                 │
┌────────────────────────────────▼─────────────────────────────────────────┐
│                       INFRASTRUCTURE LAYER                               │
│                                                                          │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐  │
│  │  LLM Provider│ │   Caching    │ │   Security   │ │   Config     │  │
│  │  Manager     │ │              │ │              │ │              │  │
│  │ • 6 providers│ │ • In-memory  │ │ • PII detect │ │ • Pydantic   │  │
│  │ • Fallback   │ │ • TTL-based  │ │ • Injection  │ │   Settings   │  │
│  │ • Health     │ │ • Key-based  │ │   prevention │ │ • .env vars  │  │
│  │   monitoring │ │   invalidate │ │ • Content    │ │ • LangSmith  │  │
│  │ • Auto-route │ │              │ │   filtering  │ │   tracing    │  │
│  └──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘  │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## Component Descriptions

### Client Layer

| Component | File(s) | Purpose |
|-----------|---------|---------|
| **Chat UI** | `frontend-react/src/pages/PromptProcessor.tsx` | Main prompt input/output interface with real-time streaming |
| **Analytics Panel** | `frontend-react/src/pages/Dashboard.tsx` | System metrics, domain distribution, optimization trends |
| **Provider Management** | `frontend-react/src/components/` | Configure and monitor LLM provider status |
| **API Client** | `frontend-react/src/api/client.ts` | TypeScript HTTP client with type-safe requests |
| **React Hooks** | `frontend-react/src/hooks/` | `useApi` (data fetching), `useNotifications` (alerts) |

### API Layer

| Component | File | Purpose |
|-----------|------|---------|
| **FastAPI App** | `src/main.py` | 86 REST endpoints, CORS middleware, static file serving |
| **Rate Limiting** | `config/config.py` | Token-bucket rate limiter per client IP |
| **Authentication** | `core/auth.py` | API key validation via `X-API-Key` header |
| **PII Detection** | `config/config.py` | Scans prompts for personal identifiable information |

### Orchestration Layer

| Component | File | Purpose |
|-----------|------|---------|
| **LangGraph Workflow** | `src/workflow.py` | 7-node StateGraph: classify → improve → evaluate → loop/finalize |
| **WorkflowCoordinator** | `agents/coordinator.py` | High-level workflow orchestration with error handling |
| **Cancellation** | `src/main.py` | `asyncio.Event`-based workflow cancellation tokens |
| **SSE Streaming** | `core/streaming.py` | Server-Sent Events for real-time node progress |

### Agent Layer

| Agent | File | Responsibility |
|-------|------|---------------|
| **DomainClassifier** | `agents/classifier.py` | Classifies prompts into 6 domains with confidence scores |
| **BaseExpertAgent** | `agents/base_expert.py` | Abstract base with 6 concrete domain implementations |
| **LangGraphExpert** | `agents/langgraph_expert.py` | Structured output expert for the LangGraph pipeline |
| **PromptEvaluator** | `agents/evaluator.py` | Rubric-based quality scoring (6 criteria, 0.0–1.0) |
| **MemoryManager** | `agents/memory/memory_manager.py` | RAG context and conversation memory |

### Feature Modules

| Module | File | Purpose |
|--------|------|---------|
| **Optimization** | `core/optimization.py` | A/B testing, prompt versioning, analytics dashboard |
| **Batch Processing** | `core/batch.py` | Concurrent multi-prompt processing with progress |
| **Templates** | `core/templates.py` | Prompt template CRUD and marketplace |
| **Plugins** | `core/plugins.py` | Extensible plugin architecture for custom processors |
| **Webhooks** | `core/webhooks.py` | Event-driven HTTP callbacks on workflow events |
| **Database** | `core/database.py` | SQLite with WAL mode for persistent storage |
| **Error Recovery** | `core/error_recovery.py` | Circuit breakers, dead letter queues, retry logic |
| **Complexity** | `core/complexity.py` | Prompt complexity analysis and scoring |
| **Language** | `core/language.py` | Multi-language prompt detection and support |
| **Similarity** | `core/similarity.py` | Semantic prompt similarity search |
| **Regression** | `core/regression.py` | Prompt regression testing across versions |
| **Fine-tuning** | `core/finetuning.py` | Generate fine-tuning datasets from optimized prompts |
| **Prompt Builder** | `core/prompt_builder.py` | Programmatic/visual prompt construction |
| **Streaming** | `core/streaming.py` | SSE streaming for real-time workflow progress |

### Infrastructure Layer

| Component | File | Purpose |
|-----------|------|---------|
| **LLM Provider Manager** | `config/llm_providers.py` | Multi-provider routing with health monitoring and fallback |
| **Configuration** | `config/config.py` | Pydantic Settings from environment variables |
| **Caching** | `config/config.py` | In-memory TTL cache with key-based invalidation |
| **Security** | `config/config.py` | Input sanitization, content filtering, PII detection |

---

## Data Flow

### Prompt Processing Pipeline

```
User Input
    │
    ▼
┌─────────────────────────┐
│  1. INPUT VALIDATION     │  Security scan, PII detection, rate limiting
└────────────┬────────────┘
             ▼
┌─────────────────────────┐
│  2. CLASSIFICATION       │  DomainClassifier → {domain, confidence, topics}
└────────────┬────────────┘
             ▼
┌─────────────────────────┐
│  3. EXPERT CREATION      │  Create/load domain-specific ExpertAgent
└────────────┬────────────┘
             ▼
┌─────────────────────────┐
│  4. PROMPT IMPROVEMENT   │  Expert refines prompt with domain best practices
└────────────┬────────────┘
             ▼
┌─────────────────────────┐
│  5. QUALITY EVALUATION   │  Evaluator scores on 6 criteria (0.0–1.0)
└────────────┬────────────┘
             │
             ├── Score ≥ 0.8 ──▶ ┌──────────────────┐
             │                    │  6. FINALIZATION   │ → Return optimized prompt
             │                    └──────────────────┘
             │
             └── Score < 0.8 ──▶ Loop back to step 4 (max 3 iterations)
                                  with evaluator feedback
```

### LLM Provider Fallback Chain

```
Request → Primary Provider (Google Gemini)
              │
              ├── Success → Return response
              │
              └── Failure → Fallback Chain:
                    Groq → OpenRouter → DeepSeek → OpenAI → Anthropic
                    (prioritizes free-tier providers first)
```

---

## Agent Interaction Map

```
                    ┌─────────────┐
                    │   User      │
                    │   (Prompt)  │
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │ Coordinator │
                    │             │
                    │ Orchestrates│
                    │ all agents  │
                    └──┬───┬───┬──┘
                       │   │   │
            ┌──────────┘   │   └──────────┐
            ▼              ▼              ▼
     ┌────────────┐ ┌────────────┐ ┌────────────┐
     │ Classifier │ │   Expert   │ │ Evaluator  │
     │            │ │            │ │            │
     │ Input:     │ │ Input:     │ │ Input:     │
     │  prompt    │ │  prompt,   │ │  original, │
     │            │ │  domain,   │ │  improved, │
     │ Output:    │ │  topics    │ │  domain    │
     │  domain,   │ │            │ │            │
     │  confidence│ │ Output:    │ │ Output:    │
     │  topics    │ │  improved  │ │  scores,   │
     │            │ │  prompt    │ │  feedback,  │
     └────────────┘ └────────────┘ │  passes?   │
                                   └────────────┘
     Step 1             Step 2         Step 3
     (runs once)     (may loop)     (may loop)
```

### Communication Protocol

- **Classifier → Coordinator**: Returns `{domain, confidence, key_topics, reasoning}`
- **Coordinator → Expert**: Passes `{prompt, domain, prompt_type, key_topics}`
- **Expert → Coordinator**: Returns `{improved_prompt, improvements_made, domain_insights}`
- **Coordinator → Evaluator**: Passes `{original, improved, domain, prompt_type}`
- **Evaluator → Coordinator**: Returns `{overall_score, criteria_scores, feedback, passes_threshold}`
- **Coordinator → User**: Returns complete result with comparison, scores, and metadata

---

## Technology Stack

| Layer | Technology | Version |
|-------|-----------|---------|
| **Runtime** | Python | 3.10+ |
| **Web Framework** | FastAPI + Uvicorn | 0.116+ |
| **AI Orchestration** | LangGraph + LangChain | 0.6+ / 0.3+ |
| **LLM Access** | langchain-google-genai, langchain-openai, etc. | Latest |
| **Data Validation** | Pydantic v2 | 2.11+ |
| **Database** | SQLite (WAL mode) | Built-in |
| **Frontend** | React 18 + TypeScript + Vite | 18.2+ |
| **Styling** | Tailwind CSS + Radix UI | 3.4+ |
| **CI/CD** | GitHub Actions | v4 |
