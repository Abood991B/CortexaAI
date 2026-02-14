# CortexaAI — Deployment Guide (Render.com)

This guide walks you through deploying CortexaAI to **Render.com** — the recommended platform for this project.

> **Why Render?** Free tier with Docker support, auto-deploy from GitHub, built-in health checks, zero-config SSL, and persistent environment variables. Perfect for a FastAPI + React application with LLM API integrations.

---

## Prerequisites

- A [Render.com](https://render.com) account (free)
- This repository pushed to GitHub
- At least one LLM API key (Google Gemini recommended — [free at aistudio.google.com](https://aistudio.google.com/))

---

## Step-by-Step Deployment

### 1. Push to GitHub

```bash
git add -A
git commit -m "v3.0.0: production-ready release"
git push origin main
```

### 2. Connect to Render

1. Go to [dashboard.render.com](https://dashboard.render.com)
2. Click **New** → **Web Service**
3. Connect your GitHub repository (`Abood991B/CortexaAI`)
4. Render will auto-detect the `render.yaml` blueprint

**Or use One-Click Deploy:**

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/Abood991B/CortexaAI)

### 3. Configure Environment Variables

In the Render dashboard, go to **Environment** and set these variables:

| Variable | Required | Description |
|----------|----------|-------------|
| `GOOGLE_API_KEY` | **Yes** | Google Gemini API key ([get one free](https://aistudio.google.com/)) |
| `DEFAULT_MODEL_PROVIDER` | No | Default: `google` |
| `DEFAULT_MODEL_NAME` | No | Default: `gemma-3-27b-it` |
| `LOG_LEVEL` | No | Default: `INFO` |
| `ENABLE_CACHING` | No | Default: `true` |
| `OPENAI_API_KEY` | No | Optional: OpenAI provider |
| `ANTHROPIC_API_KEY` | No | Optional: Anthropic provider |
| `GROQ_API_KEY` | No | Optional: Groq provider |
| `DEEPSEEK_API_KEY` | No | Optional: DeepSeek provider |
| `OPENROUTER_API_KEY` | No | Optional: OpenRouter provider |
| `LANGSMITH_API_KEY` | No | Optional: LangSmith tracing |

### 4. Deploy

Click **Create Web Service**. Render will:
1. Build the Docker image (multi-stage: frontend + backend)
2. Start the container on port 8000
3. Run the health check at `/health`
4. Assign a public URL: `https://cortexaai-xxxx.onrender.com`

First deploy takes ~5-8 minutes. Subsequent deploys are faster due to layer caching.

### 5. Verify Deployment

```bash
# Replace with your actual Render URL
export RENDER_URL=https://cortexaai-xxxx.onrender.com

# Health check
curl $RENDER_URL/health

# Test prompt processing
curl -X POST $RENDER_URL/api/process-prompt \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Write a Python function to sort data", "prompt_type": "auto", "use_langgraph": true, "synchronous": true}'

# Interactive API docs
open $RENDER_URL/docs
```

---

## Configuration Reference

### render.yaml

The deployment blueprint is already configured at the project root:

```yaml
services:
  - type: web
    name: cortexaai
    runtime: docker
    plan: free
    dockerfilePath: ./Dockerfile
    envVars:
      - key: PORT
        value: 8000
      - key: HOST
        value: 0.0.0.0
      - key: GOOGLE_API_KEY
        sync: false
      - key: DEFAULT_MODEL_PROVIDER
        value: google
      - key: DEFAULT_MODEL_NAME
        value: gemini-2.0-flash
      - key: LOG_LEVEL
        value: INFO
      - key: ENABLE_CACHING
        value: true
    healthCheckPath: /health
    autoDeploy: true
```

### Key Settings

| Setting | Value | Notes |
|---------|-------|-------|
| Runtime | Docker | Multi-stage build (Node 18 + Python 3.11) |
| Plan | Free | 512 MB RAM, shared CPU |
| Port | 8000 | Set via `PORT` env var |
| Health Check | `/health` | Checked every 30s |
| Auto-Deploy | Enabled | Deploys on every push to `main` |

---

## Post-Deployment Verification

Run these checks after deployment:

| Check | Command | Expected |
|-------|---------|----------|
| Health | `GET /health` | `{"status": "healthy", ...}` |
| API Docs | `GET /docs` | Swagger UI loads |
| Providers | `GET /api/providers` | At least 1 provider `available: true` |
| Process | `POST /api/process-prompt` | Returns optimized prompt |
| Frontend | `GET /` | React app loads |
| Stats | `GET /api/stats` | System statistics JSON |

---

## Rollback Procedures

### Revert to Previous Deploy

1. Go to Render Dashboard → your service → **Events**
2. Find the last successful deploy
3. Click **Rollback to this deploy**

### Manual Rollback via Git

```bash
# Find the last working commit
git log --oneline -10

# Reset to that commit
git revert HEAD
git push origin main
# Render will auto-deploy the reverted version
```

### Emergency: Suspend Service

1. Render Dashboard → your service → **Settings**
2. Click **Suspend Service** (stops billing and traffic)
3. Click **Resume** when ready

---

## Scaling (When You Outgrow Free Tier)

| Plan | RAM | CPU | Cost | Best For |
|------|-----|-----|------|----------|
| Free | 512 MB | Shared | $0 | Development, demos |
| Starter | 512 MB | 0.5 CPU | $7/mo | Light production |
| Standard | 2 GB | 1 CPU | $25/mo | Production workloads |
| Pro | 4 GB | 2 CPU | $85/mo | High traffic |

To upgrade: Dashboard → Service → **Settings** → Change Plan.

---

## Troubleshooting

### Build Fails

- Check Render build logs for missing dependencies
- Ensure `requirements.txt` and `package.json` are up to date
- Verify Dockerfile syntax: `docker build -t cortexaai .` locally

### App Crashes on Start

- Check that `GOOGLE_API_KEY` is set correctly in environment
- Verify the health check endpoint works: `GET /health`
- Check Render logs for Python import errors

### Slow Cold Starts

- Free tier services spin down after 15 minutes of inactivity
- First request after idle may take 30-60 seconds
- Upgrade to Starter plan ($7/mo) to avoid cold starts

### API Key Issues

- Ensure keys don't have leading/trailing spaces
- Google API keys: verify they're enabled for Generative AI API
- Test locally with the same key before deploying

---

## Alternative: Local Docker Deployment

If you prefer self-hosting:

```bash
# Clone and configure
git clone https://github.com/Abood991B/CortexaAI.git
cd CortexaAI
cp .env.example .env
# Edit .env with your API keys

# Build and run
docker compose up --build -d

# Verify
curl http://localhost:8000/health
```

The app will be available at `http://localhost:8000`.
