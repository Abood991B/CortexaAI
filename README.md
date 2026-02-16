<p align="center">
  <img src="frontend-react/public/Cortexa Logo.png" alt="CortexaAI Logo" width="180"/>
</p>
  <h1 align="center">CortexaAI</h1>
  <p align="center">
    <strong>Multi-Agent Prompt Engineering Platform</strong>
  </p>
  <p align="center">
    <a href="https://github.com/Abood991B/CortexaAI/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License: MIT"></a>
    <img src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white" alt="Python 3.10+">
    <img src="https://img.shields.io/badge/FastAPI-0.116+-009688?logo=fastapi&logoColor=white" alt="FastAPI">
    <img src="https://img.shields.io/badge/LangGraph-0.6+-purple" alt="LangGraph">
    <img src="https://img.shields.io/badge/React-18-61DAFB?logo=react&logoColor=white" alt="React 18">
    <img src="https://img.shields.io/badge/Version-3.0.2-green" alt="Version 3.0.2">
  </p>
</p>

---

CortexaAI is an advanced multi-agent system that automatically classifies, optimizes, and evaluates prompts using domain-specific expert agents powered by LangChain and LangGraph. It supports **6 LLM providers**, **real-time SSE streaming**, **batch processing**, a **prompt marketplace**, and exposes **79 REST API endpoints** — all orchestrated through a 7-node LangGraph workflow with cancellation support, per-node timing, and adaptive iteration.

<img width="1920" height="869" alt="Main CortexaAI" src="https://github.com/user-attachments/assets/1c26c541-d392-4e0f-98ca-bed74db56d31" />
<img width="1920" height="1260" alt="Dasboard" src="https://github.com/user-attachments/assets/7b553656-5e7b-4701-815c-5d99bf570c91" />
<img width="1920" height="1632" alt="Templates" src="https://github.com/user-attachments/assets/405b4e96-69df-4b18-975f-f559a310e56c" />

## ✨ Key Features

### Multi-Agent Architecture

- **Domain Classifier** — Automatically detects the domain of any prompt (software engineering, data science, education, business strategy, creative writing, report writing) using keyword heuristics + LLM reasoning
- **Expert Agents** — Dynamically spawned, domain-specialized prompt engineers with structured output and multi-angle reasoning (Executor / Critic / Expert perspectives)
- **Evaluator** — Rigorous 6-criteria scoring engine (clarity, specificity, structure, completeness, actionability, domain alignment) with calibration anchors and evidence-based assessment
- **Coordinator** — Orchestrates the full classify → improve → evaluate loop with security, caching, rate limiting, and memory context support

### LangGraph Workflow

- **7-node StateGraph pipeline**: `classify → create_expert → improve → evaluate → check_threshold → finalize → END`
- Cancellable nodes with `asyncio.Event` signaling
- Per-node wall-clock timing metrics
- Adaptive iteration with plateau detection to short-circuit when improvements become marginal
- Optional post-evaluation optimization pass for prompts scoring below 0.95
- Graceful error handling with best-effort fallback at every node

### 6 LLM Providers

| Provider | Default Model | Free Tier | Notes |
|----------|--------------|-----------|-------|
| **Google** | `gemma-3-27b-it` | ✅ | Gemini & Gemma models |
| **Groq** | `llama-3.3-70b-versatile` | ✅ | Ultra-fast inference |
| **OpenRouter** | `gemini-2.0-flash-exp:free` | ✅ | 100+ models |
| **OpenAI** | `gpt-4o-mini` | ❌ | GPT-4o, GPT-4 Turbo |
| **Anthropic** | `claude-3-haiku-20240307` | ❌ | Claude 3 family |
| **DeepSeek** | `deepseek-chat` | ❌ | DeepSeek Chat & Reasoner |

- Automatic fallback routing (free tiers first)
- Provider health monitoring & verification
- Cost-aware model selection
- Smart routing based on task complexity

### Advanced Processing

- **Iterative refinement** with quality convergence detection and adaptive early-stopping
- **Performance analytics** — Track processing times, domain distributions, and quality scores
- **Real-time monitoring** — System metrics via `/api/metrics` and `/api/dashboard` endpoints

### Real-Time Streaming

- Server-Sent Events (SSE) for live progress updates
- Stage-by-stage events: `started → classifying → classified → improving → improved → evaluating → evaluated → completed`

### Conversation Mode

- **Interactive refinement** — AI asks clarifying questions before optimizing your prompt
- **Context retention** — Maintains chat history across conversation turns
- **Seamless transition** — Automatically switches to full optimization when enough context is gathered
- **Synchronous responses** — Instant replies without polling delays

### Batch Processing

- Process multiple prompts in a single API call
- Configurable concurrency (1–10 parallel workers)
- Progress tracking and result aggregation

### Additional Features

- **Prompt Templates Library** — Curated, domain-specific templates with variable substitution, real-time collaborative editing, and privacy controls (public/private templates)
- **Prompt Marketplace** — Publish, search, download, and rate community prompt templates with complete user isolation and live updates across all active sessions
- **Visual Prompt Builder** — Composable block-based assembly (Role → Context → Task → Constraints → Output Format → Examples)
- **Plugin Architecture** — Extend experts, evaluators, or add custom pipeline steps
- **Regression Testing** — Define test suites, run against baselines, detect prompt quality regressions
- **Similarity Search** — TF-IDF / cosine similarity for duplicate detection and prompt search
- **Multi-Language Support** — Auto-detection for Arabic, Chinese, Japanese, Korean, Cyrillic, Hindi, Thai, Hebrew, and Latin-script languages
- **Complexity Analysis** — Route prompts through appropriate pipeline depth (simple → single-pass, medium → 2 iterations, complex → 3+)
- **API Key Authentication** — SHA-256 hashed keys with scoped permissions (read/write/admin) and rate limiting
- **Webhook Notifications** — HTTP POST callbacks on workflow events with retry and exponential backoff
- **Memory Context** — Optional conversation memory for user-specific prompt processing
- **Error Recovery** — Structured error taxonomy, workflow-level retry with exponential backoff, circuit breakers, error analytics, and graceful audio notification handling
- **SQLite Persistence** — Workflows, templates, marketplace items, and user data stored with WAL journaling
- **LangSmith Integration** — Optional tracing and observability
- **Security** — Input sanitization, injection detection, rate limiting, content safety checks, complete user data isolation, and privacy-first template management

## 🛠️ Technology Stack

### Backend

- **Python 3.10+** — Core language
- **FastAPI** — High-performance async web framework
- **LangChain** / **LangChain Core** — LLM orchestration and prompt management
- **LangGraph** — Stateful multi-agent workflow graphs
- **Pydantic v2** — Data validation and settings management
- **Uvicorn** — ASGI server
- **SQLite** (with WAL) — Persistent storage
- **httpx** — Async HTTP client
- **SSE-Starlette** — Server-Sent Events
- **psutil** — System monitoring
- **orjson** — Fast JSON serialization

### Frontend

- **React 18** with **TypeScript**
- **Vite** — Build tool and dev server
- **TailwindCSS** — Utility-first CSS framework
- **Radix UI** — Accessible headless UI primitives (Dialog, Select, Tabs, Switch, Progress, etc.)
- **TanStack React Query** — Server state management
- **React Router v6** — Client-side routing
- **Recharts** — Data visualization
- **Framer Motion** — Animations
- **Axios** — HTTP client
- **Sonner** — Toast notifications
- **Zod** — Schema validation

### Development & Quality

- **Ruff** — Linter and formatter (replaces flake8, isort, black)
- **pytest** + **pytest-asyncio** — Async-first testing
- **pytest-cov** — Coverage reporting (60% minimum threshold)
- **Vitest** — Frontend testing
- **mypy** — Optional static type checking
- **Safety** — Dependency vulnerability scanning

## 📋 Prerequisites

- **Python** 3.10 or higher
- **Node.js** 18 or higher (for frontend)
- **Git**
- At least one LLM API key — Google Gemini recommended ([free tier](https://aistudio.google.com/))

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Abood991B/CortexaAI.git
cd CortexaAI
```

### 2. Backend Setup

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure Environment Variables

```bash
cp .env.example .env
```

Edit `.env` with your API keys:

```env
# Required: At least one provider
GOOGLE_API_KEY=your_google_api_key_here

# Optional: Additional providers
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
GROQ_API_KEY=your_groq_api_key_here
DEEPSEEK_API_KEY=your_deepseek_api_key_here
OPENROUTER_API_KEY=your_openrouter_api_key_here

# Optional: LangSmith tracing
LANGSMITH_API_KEY=your_langsmith_api_key_here
LANGSMITH_PROJECT=cortexaai

# Model configuration
DEFAULT_MODEL_PROVIDER=google
DEFAULT_MODEL_NAME=gemma-3-27b-it

# Server
HOST=0.0.0.0
PORT=8000
LOG_LEVEL=INFO

# Prompt optimization
EVALUATION_THRESHOLD=0.82
MAX_EVALUATION_ITERATIONS=1

# CORS (comma-separated)
CORS_ORIGINS=http://localhost:3000,http://localhost:5173
```

### 4. Frontend Setup

```bash
cd frontend-react
npm install
npm run build        # Production build
# OR
npm run dev          # Development server (hot reload on port 5173)
```

### 5. Start the Application

```bash
# From project root
python src/main.py
```

The server starts at `http://localhost:8000`:

- **Web UI**: `http://localhost:8000` (serves the built React SPA)
- **API Docs (Swagger)**: `http://localhost:8000/docs`
- **API Docs (ReDoc)**: `http://localhost:8000/redoc`

## 📖 Usage

### Web Interface

The React frontend provides three main pages:

- **Prompt Processor** (`/`) — Submit prompts for optimization, view real-time progress, compare original vs. optimized output
- **Dashboard** (`/dashboard`) — System statistics, workflow history, provider health status, domain distribution charts
- **Templates** (`/templates`) — Browse, create, and manage prompt templates with variable substitution

### API Usage

#### Process a Prompt (Synchronous)

```bash
curl -X POST http://localhost:8000/api/process-prompt \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Write a Python function to sort a list",
    "prompt_type": "auto",
    "synchronous": true,
    "return_comparison": true
  }'
```

#### Process a Prompt (Async with Status Polling)

```bash
# Start workflow
curl -X POST http://localhost:8000/api/process-prompt \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Create a marketing strategy for a SaaS product",
    "prompt_type": "auto"
  }'
# Returns: { "workflow_id": "workflow_...", "status": "running", ... }

# Poll status
curl http://localhost:8000/api/workflow-status/{workflow_id}
```

#### Stream Processing (SSE)

```bash
curl -X POST http://localhost:8000/api/process-prompt/stream \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Design a REST API for a todo app", "prompt_type": "auto"}'
```

#### LangGraph Workflow Mode

```bash
curl -X POST http://localhost:8000/api/process-prompt \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Build a machine learning pipeline",
    "prompt_type": "auto",
    "use_langgraph": true,
    "synchronous": true
  }'
```

#### Batch Processing

```bash
curl -X POST http://localhost:8000/api/batch \
  -H "Content-Type: application/json" \
  -d '{
    "prompts": [
      {"prompt": "Write a unit test for authentication"},
      {"prompt": "Create a data visualization dashboard"},
      {"prompt": "Draft a quarterly business report"}
    ],
    "concurrency": 3
  }'
```

### LangGraph Studio

Visualize and debug the workflow graph interactively:

```bash
python scripts/run_langgraph_studio.py --port 8123
```

This launches the LangGraph Studio UI connected to the `prompt_engineering` graph defined in `langgraph.json`.

## 🔌 API Documentation

Interactive API docs are available at `/docs` (Swagger UI) and `/redoc` when the server is running. Below is a summary of all endpoint groups:

### Core Prompt Processing

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/process-prompt` | Process a prompt (sync or async) |
| `POST` | `/api/process-prompt-with-memory` | Process with memory context |
| `POST` | `/api/process-prompt/stream` | Stream processing via SSE |
| `POST` | `/api/process-prompt/reiterate/stream` | Re-iterate and stream |

### Workflow Management

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/domains` | List available domains |
| `GET` | `/api/workflow-status/{id}` | Get workflow status |
| `POST` | `/api/cancel-workflow/{id}` | Cancel a running workflow |
| `GET` | `/api/workflows` | List all workflows |
| `GET` | `/api/workflows/{id}` | Get workflow detail |
| `GET` | `/api/stats` | System statistics |
| `GET` | `/api/history` | Workflow history |

### System & Monitoring

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check with provider verification |
| `GET` | `/metrics` | System and LLM metrics |
| `GET` | `/api/dashboard` | Dashboard aggregated data |
| `GET` | `/api/providers` | LLM provider status |
| `POST` | `/api/providers/{name}/reset` | Reset provider health |
| `GET` | `/api/cache/stats` | Cache statistics |
| `DELETE` | `/api/cache` | Clear all caches |
| `GET` | `/api/errors/analytics` | Error analytics |
| `GET` | `/api/errors/recent` | Recent errors |

### Analytics & Monitoring

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/optimization/dashboard` | Processing analytics overview |
| `GET` | `/api/optimization/analytics` | Detailed performance analytics |

### Templates

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/templates` | List/search templates |
| `POST` | `/api/templates` | Create template |
| `GET` | `/api/templates/{id}` | Get template |
| `PUT` | `/api/templates/{id}` | Update template |
| `DELETE` | `/api/templates/{id}` | Delete template |
| `POST` | `/api/templates/render` | Render with variables |

### Batch Processing

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/batch` | Create batch job |
| `GET` | `/api/batch/{id}/status` | Batch status |
| `GET` | `/api/batch/{id}/results` | Batch results |
| `GET` | `/api/batches` | List all batches |

### Marketplace

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/marketplace` | Search marketplace |
| `POST` | `/api/marketplace` | Publish prompt |
| `GET` | `/api/marketplace/{id}` | Get item detail |
| `POST` | `/api/marketplace/{id}/rate` | Rate an item |
| `GET` | `/api/marketplace/stats/overview` | Marketplace stats |
| `GET` | `/api/marketplace/featured/list` | Featured items |

### Visual Prompt Builder

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/builder/sessions` | Create builder session |
| `GET` | `/api/builder/sessions/{id}` | Get session |
| `POST` | `/api/builder/sessions/{id}/blocks` | Add block |
| `PUT` | `/api/builder/sessions/{id}/blocks/{bid}` | Update block |
| `DELETE` | `/api/builder/sessions/{id}/blocks/{bid}` | Delete block |
| `POST` | `/api/builder/sessions/{id}/reorder` | Reorder blocks |
| `POST` | `/api/builder/sessions/{id}/assemble` | Assemble prompt |
| `GET` | `/api/builder/presets` | List presets |
| `GET` | `/api/builder/presets/{domain}` | Domain presets |

### Plugins

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/plugins` | List plugins |
| `POST` | `/api/plugins` | Register plugin |
| `GET` | `/api/plugins/{name}` | Get plugin info |
| `DELETE` | `/api/plugins/{name}` | Remove plugin |
| `POST` | `/api/plugins/{name}/enable` | Enable plugin |
| `POST` | `/api/plugins/{name}/disable` | Disable plugin |

### Regression Testing

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/regression/suites` | List test suites |
| `POST` | `/api/regression/suites` | Create suite |
| `GET` | `/api/regression/suites/{id}` | Get suite |
| `DELETE` | `/api/regression/suites/{id}` | Delete suite |
| `POST` | `/api/regression/suites/{id}/cases` | Add test case |
| `POST` | `/api/regression/suites/{id}/run` | Run suite |
| `POST` | `/api/regression/suites/{id}/baseline` | Save baseline |

### Similarity Search

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/similarity/search` | Search similar prompts |
| `POST` | `/api/similarity/index` | Index a prompt |
| `GET` | `/api/similarity/duplicates` | Find duplicates |
| `POST` | `/api/similarity/reindex` | Rebuild index |
| `GET` | `/api/similarity/stats` | Index statistics |

### Other Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/complexity` | Analyze prompt complexity |
| `POST` | `/api/complexity/pipeline-config` | Get pipeline config for complexity |
| `POST` | `/api/language/detect` | Detect language |
| `GET` | `/api/language/supported` | List supported languages |
| `POST` | `/api/auth/keys` | Create API key |
| `GET` | `/api/auth/keys` | List API keys |
| `DELETE` | `/api/auth/keys/{name}` | Revoke API key |
| `POST` | `/api/auth/verify` | Verify API key |
| `POST` | `/api/webhooks` | Subscribe to events |
| `GET` | `/api/webhooks` | List subscriptions |
| `DELETE` | `/api/webhooks/{id}` | Unsubscribe |
| `GET` | `/api/webhooks/log` | Delivery log |

## ⚙️ Configuration

All configuration is managed via environment variables (loaded from `.env`):

| Variable | Default | Description |
|----------|---------|-------------|
| `GOOGLE_API_KEY` | — | Google AI API key |
| `OPENAI_API_KEY` | — | OpenAI API key |
| `ANTHROPIC_API_KEY` | — | Anthropic API key |
| `GROQ_API_KEY` | — | Groq API key |
| `DEEPSEEK_API_KEY` | — | DeepSeek API key |
| `OPENROUTER_API_KEY` | — | OpenRouter API key |
| `LANGSMITH_API_KEY` | — | LangSmith tracing key |
| `LANGSMITH_PROJECT` | `cortexaai` | LangSmith project name |
| `DEFAULT_MODEL_PROVIDER` | `google` | Default LLM provider |
| `DEFAULT_MODEL_NAME` | `gemma-3-27b-it` | Default model |
| `HOST` | `0.0.0.0` | Server bind host |
| `PORT` | `8000` | Server bind port |
| `LOG_LEVEL` | `INFO` | Logging level |
| `PRODUCTION_MODE` | `true` | Enable production logging (suppress third-party noise) |
| `EVALUATION_THRESHOLD` | `0.82` | Minimum quality score to pass |
| `MAX_EVALUATION_ITERATIONS` | `1` | Max improve/evaluate loops |
| `MAX_LLM_RETRIES` | `1` | LLM call retry count |
| `LLM_RETRY_DELAY` | `0.5` | Seconds between retries |
| `LLM_TIMEOUT_SECONDS` | `45` | LLM call timeout |
| `CORS_ORIGINS` | `localhost:3000,5173` | Allowed CORS origins |
| `CORTEXAAI_ENV` | `development` | Environment (enables auto-reload in dev) |

## 📁 Project Structure

```text
CortexaAI/
├── src/                          # FastAPI application
│   ├── main.py                   # App creation, lifespan, CORS, SPA serving
│   ├── deps.py                   # Shared state, Pydantic models, DI
│   ├── workflow.py               # LangGraph 7-node StateGraph workflow
│   └── routes/
│       ├── prompts.py            # Prompt processing endpoints
│       ├── workflows.py          # Workflow management endpoints
│       ├── system.py             # Health, metrics, providers, cache
│       └── features.py           # Templates, batch, marketplace, builder, etc.
│
├── agents/                       # Multi-agent system
│   ├── classifier.py             # Domain classifier with 7 domains
│   ├── base_expert.py            # Base expert agent framework
│   ├── langgraph_expert.py       # LangGraph-optimized expert agent
│   ├── evaluator.py              # 6-criteria prompt quality evaluator
│   ├── coordinator.py            # Workflow coordinator with security
│   ├── exceptions.py             # Custom exception hierarchy
│   ├── utils.py                  # Shared agent utilities
│   └── memory/
│       └── memory_manager.py     # Memory context management
│
├── config/
│   ├── config.py                 # Settings, metrics, caching, security
│   └── llm_providers.py          # 6-provider LLM abstraction layer
│
├── core/                         # Feature modules
│   ├── auth.py                   # API key authentication
│   ├── batch.py                  # Batch processing
│   ├── complexity.py             # Prompt complexity scoring
│   ├── database.py               # SQLite persistence layer
│   ├── error_recovery.py         # Error taxonomy & recovery
│   ├── language.py               # Multi-language detection
│   ├── marketplace.py            # Prompt marketplace
│   ├── optimization.py           # Performance analytics
│   ├── plugins.py                # Plugin architecture
│   ├── prompt_builder.py         # Visual block-based builder
│   ├── regression.py             # Regression test runner
│   ├── similarity.py             # TF-IDF similarity search
│   ├── streaming.py              # SSE streaming
│   ├── templates.py              # Template engine
│   └── webhooks.py               # Webhook notifications
│
├── frontend-react/               # React SPA
│   ├── src/
│   │   ├── App.tsx               # Router & query client setup
│   │   ├── api/client.ts         # Typed API client (Axios)
│   │   ├── pages/                # PromptProcessor, Dashboard, Templates
│   │   ├── components/           # Layout & Radix-based UI components
│   │   ├── hooks/                # useApi, useNotifications
│   │   └── types/api.ts          # TypeScript API type definitions
│   ├── package.json
│   ├── vite.config.ts
│   └── tailwind.config.js
│
├── scripts/
│   ├── run_langgraph_studio.py   # LangGraph Studio launcher
│   ├── performance_benchmark.py  # Performance benchmarking
│   └── langgraph_performance_benchmark.py
│
├── tests/                        # Test suite
│   ├── conftest.py               # Shared fixtures (mock agents, fake keys)
│   ├── test_main.py              # FastAPI app & endpoint tests
│   ├── test_agents.py            # Agent unit tests
│   ├── test_classifier.py        # Classifier tests
│   ├── test_workflow.py          # Workflow integration tests
│   ├── test_features.py          # Feature endpoint tests
│   ├── test_optimization.py      # Optimization engine tests
│   ├── test_llm_providers.py     # Provider tests
│   ├── test_system.py            # System endpoint tests
│   └── test_langgraph_studio.py  # Studio script tests
│
├── data/
│   └── learned_domains.json      # Domain learning data
│
├── langgraph.json                # LangGraph Studio configuration
├── pyproject.toml                # Project metadata & tool config
├── requirements.txt              # Python dependencies
├── CONTRIBUTING.md               # Contribution guidelines
└── LICENSE                       # MIT License
```

## 🧪 Testing

The project uses **pytest** with async support for the backend and **Vitest** for the frontend.

### Run Backend Tests

```bash
# Run all tests
python -m pytest tests/

# Run with coverage report
python -m pytest tests/ --cov=agents --cov=config --cov=core --cov=src --cov-report=html

# Run specific test categories
python -m pytest tests/ -m unit          # Unit tests only
python -m pytest tests/ -m "not slow"    # Skip slow tests

# Run a specific test file
python -m pytest tests/test_main.py -v
```

### Run Frontend Tests

```bash
cd frontend-react
npm test              # Run tests
npm run test:ui       # Run with Vitest UI
```

### Code Quality

```bash
# Lint and auto-fix
ruff check . --fix

# Format code
ruff format .

# Security audit
safety check -r requirements.txt

# Type checking (optional)
mypy src/ agents/ config/ core/
```

## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines covering:

- Development environment setup
- Code style (Ruff enforced — PEP 8, import sorting, type checking)
- Pull request process
- Issue reporting templates
- Testing requirements

### Quick Start for Contributors

```bash
# Fork & clone
git clone https://github.com/YOUR_USERNAME/CortexaAI.git
cd CortexaAI

# Setup
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # Add at least one API key

# Verify
python -m pytest tests/
ruff check .

# Create feature branch
git checkout -b feature/your-feature
```

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

## 👤 Author

### Abdulrahman Baidaq

- GitHub: [@Abood991B](https://github.com/Abood991B)
- Email: abdulrahman16baidaq@gmail.com

---

<p align="center">
  Built with ❤️ using LangChain, LangGraph, FastAPI, and React
</p>
