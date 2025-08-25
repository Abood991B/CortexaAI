# 🚀 Multi-Agent Prompt Engineering System

A production-level Multi-Agent Prompt Engineering System that improves and optimizes raw or written prompts using multiple AI agents. The system supports modular extensibility, dynamic domain expansion, and provides comprehensive evaluation and tracing capabilities.

![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)
![Python](https://img.shields.io/badge/python-3.11+-green.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 📋 Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [API Documentation](#api-documentation)
- [Docker Deployment](#docker-deployment)
- [Development](#development)
- [Testing](#testing)
- [Extensibility](#extensibility)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## ✨ Features

### Core Capabilities
- **Multi-Agent Architecture**: Classifier, Expert, Evaluator, and Coordinator agents
- **Dynamic Domain Expansion**: Automatically creates new expert agents for unknown domains
- **Raw & Structured Prompt Support**: Handles both unstructured and semi-structured prompts
- **Evaluation Loops**: Iterative improvement until quality threshold is met
- **LangSmith Integration**: Comprehensive tracing and debugging
- **Production Ready**: Docker support, health checks, and robust error handling

### Agent System
- **🔍 Classifier Agent**: Determines prompt domains and creates new specialized agents
- **🧠 Expert Agents**: Domain-specific prompt improvement (Software Engineering, Data Science, etc.)
- **📊 Evaluator Agent**: Quality assessment and feedback generation
- **🎯 Coordinator Agent**: Orchestrates the entire workflow

### Production Features
- **Docker Support**: Containerized deployment with docker-compose
- **Health Monitoring**: Built-in health checks and metrics
- **Error Handling**: Comprehensive error handling and recovery
- **Logging**: Structured logging with configurable levels
- **API Documentation**: Auto-generated OpenAPI documentation

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   User Input    │───▶│  Coordinator   │───▶│   Classifier    │
│                 │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                         │
                                                         ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Expert Agent  │◀───│   LangGraph    │───▶│   Evaluator     │
│                 │    │   Workflow     │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                         │
                                                         ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Optimized      │    │   LangSmith    │    │   Statistics    │
│   Output        │    │   Tracing      │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Workflow Flow

1. **Input Stage**: User provides raw or structured prompt
2. **Classification**: Classifier determines domain and creates expert agents
3. **Improvement**: Domain-specific expert improves the prompt
4. **Evaluation**: Evaluator assesses quality and provides feedback
5. **Iteration**: Process repeats until threshold is met or max iterations reached
6. **Output**: Final optimized prompt with analysis and comparison

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- OpenAI API key (or Anthropic/Google API keys)
- (Optional) LangSmith API key for tracing

### 1. Clone and Setup
```bash
git clone <repository-url>
cd multi-agent-prompt-engineering
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure Environment
```bash
cp .env.example .env
# Edit .env with your API keys
```

### 4. Run the Application
```bash
python src/main.py
```

### 5. Open Web Interface
Navigate to `http://localhost:8000` in your browser.

## 📦 Installation

### Local Development
```bash
# Clone repository
git clone <repository-url>
cd multi-agent-prompt-engineering

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy environment template
cp .env.example .env

# Configure your API keys in .env
```

### Docker Installation
```bash
# Build and run with Docker Compose
docker-compose -f docker/docker-compose.yml up --build

# Or build manually
docker build -f docker/Dockerfile -t prompt-engineering .
docker run -p 8000:8000 --env-file .env prompt-engineering
```

## ⚙️ Configuration

### Environment Variables

Create a `.env` file based on `.env.example`:

```env
# API Keys
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
GOOGLE_API_KEY=your_google_api_key_here

# LangSmith Configuration
LANGSMITH_API_KEY=your_langsmith_api_key_here
LANGSMITH_PROJECT=your_project_name
LANGSMITH_ENDPOINT=https://api.smith.langchain.com

# Model Configuration
DEFAULT_MODEL_PROVIDER=openai
DEFAULT_MODEL_NAME=gpt-4

# System Configuration
LOG_LEVEL=INFO
MAX_EVALUATION_ITERATIONS=3
EVALUATION_THRESHOLD=0.8

# Server Configuration
HOST=0.0.0.0
PORT=8000
```

### Model Configuration

The system supports multiple model providers:

- **OpenAI**: GPT-4, GPT-3.5-turbo
- **Anthropic**: Claude-3 models
- **Google**: Gemini models

Configure the default provider and model in your `.env` file.

## 🎯 Usage

### Web Interface

1. Open `http://localhost:8000` in your browser
2. Enter your prompt in the text area
3. Select prompt type (Auto-detect, Raw, or Structured)
4. Choose options (comparison, LangGraph workflow)
5. Click "Optimize Prompt"
6. View the optimized result with analysis

### REST API

#### Process a Prompt
```bash
curl -X POST "http://localhost:8000/api/process-prompt" \
     -H "Content-Type: application/json" \
     -d '{
       "prompt": "Write a function to sort a list",
       "prompt_type": "raw",
       "return_comparison": true,
       "use_langgraph": false
     }'
```

#### Get System Statistics
```bash
curl http://localhost:8000/api/stats
```

#### Get Available Domains
```bash
curl http://localhost:8000/api/domains
```

### Python API

```python
from agents.coordinator import coordinator

# Process a prompt
result = coordinator.process_prompt(
    prompt="Create a data analysis report",
    prompt_type="raw",
    return_comparison=True
)

print(f"Optimized: {result['output']['optimized_prompt']}")
print(f"Domain: {result['output']['domain']}")
print(f"Quality Score: {result['output']['quality_score']}")
```

## 📚 API Documentation

### Endpoints

#### `POST /api/process-prompt`
Process a prompt through the multi-agent workflow.

**Request Body:**
```json
{
  "prompt": "string",
  "prompt_type": "auto|raw|structured",
  "return_comparison": boolean,
  "use_langgraph": boolean
}
```

**Response:**
```json
{
  "workflow_id": "string",
  "status": "completed",
  "timestamp": "string",
  "processing_time_seconds": 2.34,
  "output": {
    "optimized_prompt": "string",
    "domain": "string",
    "quality_score": 0.88,
    "iterations_used": 1,
    "passes_threshold": true
  },
  "analysis": {
    "classification": {...},
    "improvements": {...},
    "evaluation": {...}
  }
}
```

#### `GET /api/domains`
Get information about available domains.

#### `GET /api/stats`
Get system statistics and metrics.

#### `GET /api/history`
Get recent workflow history.

#### `GET /health`
Health check endpoint.

## 🐳 Docker Deployment

### Production Deployment

1. **Build and Deploy:**
```bash
cd docker
docker-compose up --build -d
```

2. **Scale the Service:**
```bash
docker-compose up --scale prompt-engineering-system=3 -d
```

3. **View Logs:**
```bash
docker-compose logs -f prompt-engineering-system
```

### Docker Configuration Files

- `docker/Dockerfile`: Production container configuration
- `docker/docker-compose.yml`: Multi-service orchestration

### Environment for Docker

Ensure your `.env` file is in the project root (parent of docker directory).

## 🛠️ Development

### Project Structure

```
multi-agent-prompt-engineering/
├── agents/                 # Agent implementations
│   ├── classifier.py      # Domain classification agent
│   ├── base_expert.py     # Base expert agent framework
│   ├── evaluator.py       # Quality evaluation agent
│   └── coordinator.py     # Workflow coordinator
├── config/                # Configuration management
│   └── config.py         # Settings and model config
├── src/                   # Main application
│   ├── main.py           # FastAPI application
│   └── workflow.py       # LangGraph workflow
├── tests/                # Test suite
│   └── test_agents.py    # Agent unit tests
├── docker/               # Docker configuration
│   ├── Dockerfile        # Production container
│   └── docker-compose.yml # Service orchestration
├── docs/                 # Documentation
└── README.md            # This file
```

### Development Setup

```bash
# Install development dependencies
pip install -r requirements.txt
pip install pytest black isort mypy

# Run with auto-reload
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000

# Format code
black .
isort .
```

### LangGraph Studio

For development with LangGraph Studio:

1. Install LangGraph Studio
2. Run the application
3. Access the Studio interface for workflow debugging

## 🧪 Testing

### Run Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_agents.py

# Run with coverage
pytest --cov=agents --cov-report=html
```

### Test Structure

- **Unit Tests**: Individual agent functionality
- **Integration Tests**: Full workflow testing
- **Mock Tests**: LLM interaction testing with mocked responses

### Writing Tests

```python
import pytest
from unittest.mock import patch
from agents.classifier import classifier

def test_classification():
    with patch('agents.classifier.ChatOpenAI') as mock_chat:
        # Mock LLM response
        mock_response = Mock()
        mock_response.content = '{"domain": "software_engineering"}'
        mock_chat.return_value.invoke.return_value = mock_response

        result = classifier.classify_prompt("Write code")
        assert result["domain"] == "software_engineering"
```

## 🔧 Extensibility

### Adding New Domains

The system automatically creates new domains when encountered. To manually add a domain:

```python
from agents.classifier import classifier

# Add new domain
classifier.known_domains["legal"] = {
    "keywords": ["law", "legal", "contract", "regulation"],
    "description": "Legal document and analysis tasks"
}
```

### Creating Custom Expert Agents

```python
from agents.base_expert import BaseExpertAgent

class LegalExpert(BaseExpertAgent):
    def _define_expertise_areas(self):
        return ["Legal analysis", "Contract review", "Compliance checking"]

    def _define_improvement_templates(self):
        return {
            "default": "Legal-specific improvement instructions...",
            "raw": "Legal raw prompt improvements...",
            "structured": "Legal structured prompt improvements..."
        }

# Register the new expert
from agents.base_expert import EXPERT_AGENT_REGISTRY
EXPERT_AGENT_REGISTRY["legal"] = LegalExpert
```

### Custom Model Providers

Add support for new model providers in `config/config.py`:

```python
def get_model_config(provider=None, model_name=None):
    configs = {
        # ... existing providers
        "custom_provider": {
            "model_name": model_name or "custom-model",
            "api_key": settings.custom_api_key,
            "base_url": "https://custom-provider.com"
        }
    }
    return configs.get(provider, configs["openai"])
```

## 🔍 Troubleshooting

### Common Issues

#### API Key Errors
```
Error: OpenAI API key not found
```
**Solution**: Ensure `OPENAI_API_KEY` is set in your `.env` file.

#### Port Already in Use
```
Error: Port 8000 already in use
```
**Solution**: Change the port in `.env` or stop the conflicting service.

#### Model Not Available
```
Error: Model not found
```
**Solution**: Check model availability and update `DEFAULT_MODEL_NAME` in `.env`.

### Debug Mode

Enable debug logging:
```env
LOG_LEVEL=DEBUG
```

### Health Checks

Check system health:
```bash
curl http://localhost:8000/health
```

### Logs

View application logs:
```bash
# Docker
docker-compose logs -f

# Local
tail -f logs/app.log
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines

- Follow PEP 8 style guidelines
- Add tests for new features
- Update documentation
- Use type hints
- Write clear commit messages

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with [LangChain](https://langchain.com/) and [LangGraph](https://github.com/langchain-ai/langgraph)
- Powered by OpenAI, Anthropic, and Google AI models
- Tracing provided by [LangSmith](https://smith.langchain.com/)

## 📞 Support

For support and questions:
- Create an issue on GitHub
- Check the documentation
- Review the troubleshooting guide

---

**Happy Prompt Engineering! 🚀**
