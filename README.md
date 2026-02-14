<p align="center">
  <img src="frontend-react/public/Cortexa Logo.png" alt="CortexaAI Logo" width="180"/>
</p>

<h1 align="center">CortexaAI</h1>
<p align="center"><strong>Production-Ready Multi-Agent Prompt Engineering Platform — V3.0</strong></p>

<p align="center">
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"/></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python 3.10+"/></a>
  <a href="https://fastapi.tiangolo.com/"><img src="https://img.shields.io/badge/FastAPI-0.116+-009688.svg?logo=fastapi&logoColor=white" alt="FastAPI"/></a>
  <a href="https://reactjs.org/"><img src="https://img.shields.io/badge/React-18.2+-61DAFB.svg?logo=react&logoColor=white" alt="React 18+"/></a>
  <a href="https://langchain-ai.github.io/langgraph/"><img src="https://img.shields.io/badge/LangGraph-0.6+-purple.svg" alt="LangGraph"/></a>
  <img src="https://img.shields.io/badge/LLM_Providers-6+-orange.svg" alt="6+ LLM Providers"/>
  <img src="https://img.shields.io/badge/API_Routes-84-brightgreen.svg" alt="84 API Routes"/>
  <a href="#docker-deployment"><img src="https://img.shields.io/badge/Docker-Ready-2496ED.svg?logo=docker&logoColor=white" alt="Docker Ready"/></a>
</p>

<p align="center">
  Transform raw prompts into optimized, production-grade prompts using AI agents that classify, enhance, evaluate, and A/B test — automatically.
</p>

---

## What is CortexaAI?

CortexaAI is a **production-ready, enterprise-grade multi-agent prompt optimization platform** built with LangGraph orchestration. It automatically transforms any raw prompt into a highly effective, domain-specific prompt through a pipeline of specialized AI agents, with built-in A/B testing, version control, batch processing, and 17 production feature modules.

**The Problem:** Writing effective prompts for LLMs is time-consuming, inconsistent, and requires expertise. A poorly written prompt can lead to vague, incorrect, or unhelpful AI responses.

**The Solution:** CortexaAI automates prompt engineering with a multi-agent system that:

1. **Classifies** the prompt domain (software engineering, data science, education, etc.)
2. **Routes** to a domain-specific expert agent
3. **Optimizes** the prompt with best practices for that domain
4. **Evaluates** the result against 6 quality criteria
5. **Iterates** until the quality threshold is met
6. **Tracks** versions and runs A/B tests between variants

> **Result:** Raw prompts become structured, detailed, actionable prompts that consistently score 90%+ on quality metrics.

---

## Key Features

| Feature | Description |
|---------|-------------|
| 🧠 **Multi-Agent Architecture** | Classifier → Expert → Evaluator pipeline with LangGraph orchestration |
| 🔄 **Iterative Optimization** | Automatic refinement loops until quality threshold is met |
| 📊 **A/B Testing** | Compare prompt variants with statistical confidence scoring |
| 📝 **Version Control** | Full history tracking with rollback for every prompt |
| 🤖 **6+ LLM Providers** | Google Gemini, OpenAI, Anthropic, Groq, DeepSeek, OpenRouter |
| 🔀 **Smart Fallback** | Automatic provider routing when a model is unavailable |
| 💬 **Chat Interface** | Conversational UI with session management and memory |
| 🛡️ **Security** | PII detection, injection prevention, API key auth, rate limiting |
| ⚡ **Performance** | Caching, circuit breakers, dead letter queues, streaming |
| 🔌 **Plugin System** | Extensible plugin architecture for custom processors |
| 📈 **Analytics Dashboard** | Real-time metrics, domain distribution, optimization trends |
| 🌐 **Webhook System** | Event-driven notifications for workflow completion |
| 📦 **Batch Processing** | Process multiple prompts concurrently with progress tracking |
| 🧬 **Prompt Templates** | Template marketplace with sharing and versioning |
| 🗄️ **SQLite Database** | Persistent storage for templates, API keys, and analytics |
| 🐳 **Docker Ready** | One-command deployment with Docker Compose |
| 📖 **84 API Routes** | Comprehensive REST API with interactive Swagger docs at `/docs` |

---

## Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                     React Frontend (Vite + Tailwind)           │
│  Chat Interface │ Analytics Dashboard │ Provider Management    │
└─────────────────────────────┬──────────────────────────────────┘
                              │ REST API (84 routes)
┌─────────────────────────────▼──────────────────────────────────┐
│                     FastAPI Backend                             │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              LangGraph Workflow Orchestrator              │  │
│  │                                                          │  │
│  │   ┌──────────┐   ┌──────────┐   ┌──────────────────┐   │  │
│  │   │Classifier│──▶│  Expert  │──▶│    Evaluator     │   │  │
│  │   │  Agent   │   │  Agent   │   │     Agent        │   │  │
│  │   └──────────┘   └──────────┘   └────────┬─────────┘   │  │
│  │                                    ▼ Passes threshold?  │  │
│  │                              Yes ──┘    No ──▶ Re-loop  │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                │
│  ┌───────────────┐ ┌────────────────┐ ┌────────────────────┐  │
│  │ Optimization  │ │  LLM Provider  │ │    Security &      │  │
│  │ Engine (A/B,  │ │  Manager (6+   │ │    Caching Layer   │  │
│  │  Versioning)  │ │  providers)    │ │                    │  │
│  └───────────────┘ └────────────────┘ └────────────────────┘  │
│                                                                │
│  ┌───────────────┐ ┌────────────────┐ ┌────────────────────┐  │
│  │ Plugin System │ │  Batch Engine  │ │  Template Market   │  │
│  └───────────────┘ └────────────────┘ └────────────────────┘  │
└────────────────────────────────────────────────────────────────┘
```

---

## Quick Start

### Prerequisites

- Python 3.10+
- Node.js 18+ (for frontend)
- At least one LLM API key (Google Gemini recommended — [free tier](https://aistudio.google.com/))

### 1. Clone & Setup

```bash
git clone https://github.com/Abood991B/CortexaAI.git
cd CortexaAI

# Create Python virtual environment
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure API Keys

```bash
cp .env.example .env
# Edit .env and add your API key(s)
# Minimum: GOOGLE_API_KEY (free at aistudio.google.com)
```

### 3. Start the Backend

```bash
python src/main.py
```

### 4. Start the Frontend

```bash
cd frontend-react
npm install
npm run dev
```

Open **http://localhost:5173** — start optimizing prompts!

### Docker (Alternative)

```bash
cp .env.example .env
docker compose up --build
```

---

## Supported LLM Providers

CortexaAI supports **6 providers** out of the box, with a modular system for adding more:

| Provider | Free Tier | Models | Setup |
|----------|-----------|--------|-------|
| **Google Gemma / Gemini** ⭐ | ✅ | gemma-3-27b-it (default), gemma-3-12b-it, gemini-2.0-flash | [aistudio.google.com](https://aistudio.google.com/) |
| **Groq** | ✅ | llama-3.3-70b, mixtral-8x7b | [console.groq.com](https://console.groq.com/) |
| **OpenRouter** | ✅ | 100+ models (free options) | [openrouter.ai](https://openrouter.ai/) |
| **DeepSeek** | Affordable | deepseek-chat, deepseek-reasoner | [platform.deepseek.com](https://platform.deepseek.com/) |
| **OpenAI** | Paid | gpt-4o-mini, gpt-4o | [platform.openai.com](https://platform.openai.com/) |
| **Anthropic** | Paid | claude-3-haiku, claude-3-sonnet | [console.anthropic.com](https://console.anthropic.com/) |

> **Smart Fallback:** If your primary provider fails, CortexaAI automatically routes to the next available provider — free tiers first.

---

## API Reference

### Core Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/process-prompt` | Optimize a prompt through the agent pipeline |
| `POST` | `/api/process-prompt-with-memory` | Optimize with conversation context |
| `GET` | `/api/workflow-status/{id}` | Check workflow status |
| `POST` | `/api/cancel-workflow/{id}` | Cancel a running workflow |

### Analytics & Optimization

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/optimization/dashboard` | Full optimization dashboard data |
| `GET` | `/api/optimization/analytics` | Performance metrics and trends |
| `GET` | `/api/optimization/ab-tests` | A/B test history and statistics |
| `GET` | `/api/optimization/versions` | Prompt version statistics |

### v3.0 Feature Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/batch` | Submit batch prompt processing jobs |
| `GET` | `/api/batch/{batch_id}/status` | Get batch job status |
| `GET` | `/api/batch/{batch_id}/results` | Get batch job results |
| `GET` | `/api/batches` | List all batch jobs |
| `GET` | `/api/templates` | Browse prompt template library |
| `POST` | `/api/templates` | Create new prompt template |
| `GET` | `/api/templates/{template_id}` | Get specific template |
| `POST` | `/api/templates/render` | Render template with variables |
| `POST` | `/api/webhooks` | Register webhook callbacks |
| `GET` | `/api/webhooks` | List webhook subscriptions |
| `DELETE` | `/api/webhooks/{sub_id}` | Delete webhook subscription |
| `GET` | `/api/webhooks/log` | View webhook delivery log |
| `GET` | `/api/plugins` | List installed plugins |
| `POST` | `/api/plugins` | Install a new plugin |
| `GET` | `/api/plugins/{name}` | Get plugin details |
| `DELETE` | `/api/plugins/{name}` | Uninstall plugin |
| `POST` | `/api/plugins/{name}/enable` | Enable plugin |
| `POST` | `/api/plugins/{name}/disable` | Disable plugin |
| `POST` | `/api/auth/keys` | Create API key |
| `GET` | `/api/auth/keys` | List API keys |
| `DELETE` | `/api/auth/keys/{name}` | Delete API key |
| `POST` | `/api/auth/verify` | Verify API key |
| `POST` | `/api/process-prompt/stream` | Stream workflow progress via SSE |
| `POST` | `/api/complexity` | Analyze prompt complexity |
| `POST` | `/api/complexity/pipeline-config` | Get pipeline config based on complexity |
| `POST` | `/api/language/detect` | Detect prompt language |
| `GET` | `/api/language/supported` | List supported languages |
| `GET` | `/api/marketplace` | Browse marketplace items |
| `POST` | `/api/marketplace` | Publish to marketplace |
| `GET` | `/api/marketplace/{item_id}` | Get marketplace item |
| `POST` | `/api/marketplace/{item_id}/rate` | Rate marketplace item |
| `GET` | `/api/marketplace/stats/overview` | Marketplace statistics |
| `GET` | `/api/marketplace/featured/list` | Featured marketplace items |
| `POST` | `/api/finetuning/prepare` | Prepare fine-tuning dataset |
| `POST` | `/api/finetuning/jobs` | Create fine-tuning job |
| `GET` | `/api/finetuning/jobs` | List fine-tuning jobs |
| `GET` | `/api/finetuning/jobs/{job_id}` | Get job details |
| `POST` | `/api/finetuning/jobs/{job_id}/simulate` | Simulate fine-tuned model |
| `GET` | `/api/finetuning/models/{provider}` | List available models for provider |
| `GET` | `/api/finetuning/estimate` | Estimate fine-tuning cost |
| `POST` | `/api/builder/sessions` | Create prompt builder session |
| `GET` | `/api/builder/sessions/{session_id}` | Get builder session |
| `POST` | `/api/builder/sessions/{session_id}/blocks` | Add block to prompt |
| `PUT` | `/api/builder/sessions/{session_id}/blocks/{block_id}` | Update block |
| `DELETE` | `/api/builder/sessions/{session_id}/blocks/{block_id}` | Delete block |
| `POST` | `/api/builder/sessions/{session_id}/reorder` | Reorder blocks |
| `POST` | `/api/builder/sessions/{session_id}/assemble` | Assemble final prompt |
| `GET` | `/api/builder/presets` | List builder presets |
| `GET` | `/api/builder/presets/{domain}` | Get domain-specific presets |
| `GET` | `/api/regression/suites` | List regression test suites |
| `POST` | `/api/regression/suites` | Create test suite |
| `GET` | `/api/regression/suites/{suite_id}` | Get suite details |
| `DELETE` | `/api/regression/suites/{suite_id}` | Delete test suite |
| `POST` | `/api/regression/suites/{suite_id}/cases` | Add test case |
| `POST` | `/api/regression/suites/{suite_id}/run` | Run regression tests |
| `POST` | `/api/regression/suites/{suite_id}/baseline` | Set baseline |
| `POST` | `/api/similarity/search` | Search similar prompts |
| `POST` | `/api/similarity/index` | Index prompt for similarity |
| `GET` | `/api/similarity/duplicates` | Find duplicate prompts |
| `POST` | `/api/similarity/reindex` | Rebuild similarity index |
| `GET` | `/api/similarity/stats` | Similarity index statistics |
| `GET` | `/api/errors/analytics` | Error analytics dashboard |
| `GET` | `/api/errors/recent` | Recent errors log |
| `GET` | `/api/dashboard` | Complete system dashboard |

### System

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/providers` | LLM provider status and availability |
| `GET` | `/api/stats` | System statistics |
| `GET` | `/api/domains` | Available prompt domains |
| `GET` | `/health` | Health check with component status |
| `GET` | `/metrics` | Prometheus-compatible metrics |
| `GET` | `/docs` | Interactive API documentation (Swagger) |

### Example Request

```bash
curl -X POST http://localhost:8000/api/process-prompt \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Write a Python function to sort data",
    "prompt_type": "auto",
    "use_langgraph": true,
    "synchronous": true
  }'
```

---

## Project Structure

```
CortexaAI/
├── src/                          # Application entry point
│   ├── main.py                   #   FastAPI app, 87 routes, middleware
│   └── workflow.py               #   LangGraph workflow orchestration
│
├── agents/                       # Multi-agent system
│   ├── classifier.py             #   Domain classification (6 domains)
│   ├── base_expert.py            #   Base expert + domain implementations
│   ├── langgraph_expert.py       #   LangGraph-specific expert agent
│   ├── evaluator.py              #   Quality evaluation (6 criteria)
│   ├── coordinator.py            #   Workflow coordination
│   ├── exceptions.py             #   Custom exception hierarchy
│   ├── utils.py                  #   Shared agent utilities
│   └── memory/                   #   RAG & conversation memory
│       └── memory_manager.py
│
├── core/                         # Feature modules (v3.0)
│   ├── optimization.py           #   A/B testing & prompt versioning
│   ├── auth.py                   #   API key authentication
│   ├── batch.py                  #   Batch prompt processing
│   ├── complexity.py             #   Prompt complexity analysis
│   ├── database.py               #   SQLite persistence layer
│   ├── error_recovery.py         #   Circuit breakers & DLQ
│   ├── finetuning.py             #   Fine-tuning dataset generation
│   ├── language.py               #   Multi-language support
│   ├── marketplace.py            #   Template marketplace
│   ├── plugins.py                #   Plugin system
│   ├── prompt_builder.py         #   Visual prompt builder
│   ├── regression.py             #   Prompt regression testing
│   ├── similarity.py             #   Prompt similarity search
│   ├── streaming.py              #   SSE streaming support
│   ├── templates.py              #   Template management
│   └── webhooks.py               #   Webhook notifications
│
├── config/                       # Configuration
│   ├── config.py                 #   Settings, security, caching
│   └── llm_providers.py          #   Multi-LLM provider system
│
├── frontend-react/               # React + TypeScript + Tailwind CSS
│   ├── src/
│   │   ├── pages/                #   PromptProcessor, SystemHealth
│   │   ├── components/           #   Reusable UI components
│   │   ├── hooks/                #   React Query hooks
│   │   └── api/                  #   API client
│   └── package.json
│
├── tests/                        # Pytest test suite (10 files)
│   ├── conftest.py               #   Shared fixtures & mocks
│   ├── test_agents.py
│   ├── test_classifier.py
│   ├── test_features.py
│   ├── test_llm_providers.py
│   ├── test_main.py
│   ├── test_optimization.py
│   ├── test_system.py
│   └── test_workflow.py
│
├── scripts/                      # Development utilities
│   ├── performance_benchmark.py  #   System benchmark suite
│   ├── langgraph_performance_benchmark.py
│   └── run_langgraph_studio.py   #   LangGraph Studio launcher
│
├── docs/                         # Documentation
│   ├── FEATURE_ANALYSIS.md       #   Feature analysis & roadmap
│   └── PROMPT_SYSTEM_ENHANCEMENT.md
│
├── data/                         # Runtime data (gitignored)
│   └── .gitkeep
│
├── .github/workflows/ci.yml     # CI/CD pipeline
├── Dockerfile                    # Multi-stage production build
├── Dockerfile.railway            # Railway-optimized Docker build
├── docker-compose.yml            # Docker Compose config
├── railway.json                  # Railway deployment configuration
├── langgraph.json                # LangGraph Studio config
├── pyproject.toml                # Project metadata & tool config
├── requirements.txt              # Python dependencies
├── .env.example                  # Environment variable template
├── ARCHITECTURE.md               # System architecture documentation
├── CHANGELOG.md                  # Version history
├── CONTRIBUTING.md               # Contribution guidelines
├── PREFLIGHT_CHECKLIST.md        # Pre-deployment verification
└── LICENSE                       # MIT License
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| **Backend** | Python 3.10+, FastAPI, Pydantic v2, Uvicorn |
| **AI Framework** | LangChain, LangGraph, LangSmith |
| **LLM Providers** | Google Gemini, OpenAI, Anthropic, Groq, DeepSeek, OpenRouter |
| **Database** | SQLite with WAL mode |
| **Frontend** | React 18, TypeScript, Vite, Tailwind CSS, Radix UI |
| **Data Layer** | React Query, Recharts, Framer Motion |
| **Infrastructure** | Docker, GitHub Actions CI/CD |
| **Deployment** | Railway (recommended), Docker Compose |
| **Quality** | Pytest, Ruff, mypy, Bandit |

---

## Deployment

CortexaAI deploys easily to Railway with Docker support.

### 🚀 Railway (Recommended)
**Best choice:** Generous free tier, Docker support, PostgreSQL included

[![Deploy on Railway](https://railway.app/button.svg)](https://railway.app/new/template/cortexaai)

**Why Railway?**
- $5/month credit (enough for small projects)
- 512MB RAM, 1GB disk free forever
- Auto-deploy from GitHub
- PostgreSQL database included
- Custom domains with SSL

**Quick Deploy:**
1. Click the button above
2. Connect your GitHub repo
3. Set `GOOGLE_API_KEY` in dashboard
4. Deploy!

### Docker (Local Development)

```bash
# Local development
docker compose up --build

# Production deploy (any Docker host)
docker build -f Dockerfile.railway -t cortexaai .
docker run -p 8000:8000 -e GOOGLE_API_KEY=your_key cortexaai
```

> See [RAILWAY_DEPLOYMENT.md](docs/RAILWAY_DEPLOYMENT.md) for complete Railway deployment guide.

---

## Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

```bash
# Run tests
pytest tests/ -v

# Lint
ruff check .

# Type check
mypy agents/ src/ config/ core/
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

<p align="center">
  <strong>Built by <a href="https://github.com/Abood991B">Abdulrahman Baidaq</a></strong><br/>
  <a href="https://linkedin.com/in/abdulrahman-baidaq">LinkedIn</a> · <a href="mailto:abdulrahman16baidaq@gmail.com">Email</a>
</p>