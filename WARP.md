# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Development Commands

### Backend (Python)

#### Setup & Environment
```bash
# Create and activate virtual environment
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Setup environment variables
copy .env.example .env
# Edit .env with your API keys for Google, OpenAI, and Anthropic
```

#### Running the Application
```bash
# Start the FastAPI backend server with hot reload
python src/main.py

# Alternative: Using PowerShell script (Windows)
.\start_backend.ps1

# Start with system activation (recommended first run)
python system_activation.py
```

#### Testing
```bash
# Run all tests
python -m pytest tests/

# Run specific test file
python -m pytest tests/test_agents.py

# Run tests with coverage
python -m pytest --cov=src --cov-report=html tests/

# Run individual test modules
python -m pytest tests/test_classifier.py -v
python -m pytest tests/test_evaluator.py -v
python -m pytest tests/test_langgraph_studio.py -v
```

#### Benchmarking and Performance
```bash
# Run comprehensive performance benchmarks
python tools/performance_benchmark.py

# Run LangGraph Studio for workflow visualization
python tools/run_langgraph_studio.py
```

### Frontend (React + TypeScript)

#### Setup & Development
```bash
cd frontend-react

# Install dependencies
npm install

# Start development server
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview

# Lint code
npm run lint

# Run tests
npm run test

# Run tests with UI
npm run test:ui
```

### Docker Deployment
```bash
# Build and run with Docker Compose
cd docker
docker-compose up --build

# Run in detached mode
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### LangGraph Workflow Development
```bash
# The LangGraph configuration is in langgraph.json
# Main workflow entry point: src/workflow.py:prompt_engineering_app

# Run LangGraph Studio for visual workflow development
python tools/run_langgraph_studio.py

# Test LangGraph workflows specifically
python -m pytest tests/test_langgraph_studio.py
```

## Architecture Overview

### Multi-Agent System Design

This is a production-grade multi-agent system for intelligent prompt engineering with the following core architecture:

#### Agent Hierarchy
1. **WorkflowCoordinator** (`agents/coordinator.py`) - Main orchestrator that manages the entire workflow
2. **DomainClassifier** (`agents/classifier.py`) - Classifies prompts into domains and determines optimal processing strategy  
3. **BaseExpertAgent** (`agents/base_expert.py`) - Domain-specific expert agents that improve prompts based on specialized knowledge
4. **PromptEvaluator** (`agents/evaluator.py`) - Quality assessment agent that evaluates improvements and determines if further iteration is needed

#### Processing Flow
The system follows this multi-stage processing pipeline:
1. **Input Sanitization** - Security filtering and content validation
2. **Domain Classification** - Intelligent categorization of prompts
3. **Expert Agent Selection** - Dynamic creation or retrieval of domain-specific agents  
4. **Prompt Improvement** - Domain-aware optimization using specialized templates and techniques
5. **Quality Evaluation** - Iterative assessment and refinement up to configurable thresholds
6. **Result Packaging** - Comprehensive output with before/after comparison and metadata

#### LangGraph Workflow Integration
The system supports both traditional agent coordination and LangGraph workflow execution:
- **Traditional Mode**: Direct agent-to-agent communication via the coordinator
- **LangGraph Mode**: State-based workflow execution with visual debugging capabilities
- **Workflow Nodes**: Each agent operation is represented as a cancellable LangGraph node
- **State Management**: Centralized workflow state with proper error handling and rollback

### Key Architecture Components

#### Security Framework (`config/config.py`)
- **Multi-level Security**: Strict, balanced, and permissive modes
- **Content Filtering**: Context-aware detection that understands technical vs malicious content
- **Injection Prevention**: Sophisticated prompt injection detection with intelligent whitelisting
- **Rate Limiting**: Client-based throttling with configurable limits
- **Input Sanitization**: Domain-aware cleaning that preserves legitimate technical content

#### Performance & Reliability
- **Circuit Breaker Pattern**: Automatic failover when services become unstable
- **Intelligent Caching**: Multi-level caching with domain and context awareness
- **Retry Logic**: Exponential backoff with smart error classification
- **Memory Management**: RAG-enhanced context with conversation history
- **Metrics Collection**: Comprehensive performance tracking and analytics

#### Template & Configuration System
- **Dynamic Domain Creation**: Automatic detection and creation of new expert agents
- **Template Management**: Reusable prompt templates with variable substitution
- **A/B Testing Framework**: Built-in experimentation with statistical analysis
- **Configuration Hot-Reload**: Runtime configuration updates without service restart

### Critical Integration Points

#### LLM Provider Fallback Chain
The system uses a priority-based LLM provider strategy:
1. **Primary**: Google Gemini (`gemini-2.0-flash`)
2. **Fallback 1**: OpenAI GPT-4
3. **Fallback 2**: Anthropic Claude

#### Memory & Context Management (`agents/memory.py`)
- **RAG Integration**: Semantic search and context retrieval
- **Conversation Memory**: Multi-turn conversation state management
- **User Context**: Personalized prompt optimization based on user history
- **Memory Persistence**: Long-term storage of user preferences and successful patterns

#### API Design Philosophy
- **RESTful Endpoints**: Clean, resource-based API design
- **Async Processing**: All operations are fully asynchronous for scalability
- **Workflow Tracking**: Real-time status updates with cancellation support
- **Comprehensive Responses**: Detailed metadata, analysis, and comparison data

## Development Guidelines

### Agent Development Patterns
When extending the system with new agent types:
- Inherit from `BaseExpertAgent` for domain-specific agents
- Use the coordinator for inter-agent communication
- Implement proper error handling with custom exception types from `agents/exceptions.py`
- Add security sanitization for all user inputs
- Use the caching layer for expensive operations
- Follow the async/await pattern throughout

### LangGraph Node Development
When creating new workflow nodes:
- Use the `@cancellable_node` decorator for proper cancellation support
- Maintain the `WorkflowState` type for state consistency
- Handle errors gracefully and update state accordingly
- Use structured logging for debugging and monitoring

### Configuration Management
- Environment variables are managed through `config/config.py` using Pydantic
- Security settings should be configurable per environment
- Use the `get_logger()` function for structured JSON logging
- Metrics are collected automatically through the `metrics` global instance

### Testing Strategy
- Unit tests for individual agents in `tests/test_agents.py`
- Integration tests for the complete workflow in `tests/test_workflow.py`
- LangGraph-specific tests in `tests/test_langgraph_studio.py`
- Performance benchmarks should be added to `tools/performance_benchmark.py`

### Frontend Integration Notes
- The React frontend communicates with the FastAPI backend via REST API
- Workflow status is polled via `/api/workflow-status/{workflow_id}`
- Real-time updates can be implemented via the existing status polling mechanism
- The frontend supports both traditional and memory-enhanced workflows

### Deployment Considerations
- The system is designed for containerized deployment using the provided Docker configuration
- Environment variables must be properly configured for all LLM providers
- The health check endpoint `/health` provides comprehensive system status
- Metrics are exposed in Prometheus format at `/metrics`
- Log aggregation should capture the structured JSON logs for proper monitoring

### Security Best Practices
- Never hardcode API keys - use environment variables
- The security manager handles input sanitization automatically
- Rate limiting is enforced at the coordinator level
- All user inputs are logged for security auditing
- The system supports configurable security levels for different environments

## Important Notes

- **Cancellation Grace Period**: Workflows can only be cancelled within the first 3 seconds of execution
- **Evaluation Thresholds**: The system will iterate up to `MAX_EVALUATION_ITERATIONS` times or until the quality threshold is met
- **Domain Migration**: The system supports gradual migration of domains to new processing strategies
- **Memory Context**: When using memory-enhanced processing, conversation history is automatically included in prompt optimization
- **Template Variables**: Prompt templates support variable substitution using `{variable_name}` syntax
- **A/B Testing**: Experiments require at least 2 variants and traffic splits that sum to 1.0
