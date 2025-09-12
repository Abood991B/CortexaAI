# Cortexa

<div align="center">
  <img src="frontend-react/public/Cortexa Logo.png" alt="Cortexa Logo" width="200"/>
  
  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
  [![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
  [![FastAPI](https://img.shields.io/badge/FastAPI-0.116+-green.svg)](https://fastapi.tiangolo.com/)
  [![React](https://img.shields.io/badge/React-18.2+-61DAFB.svg)](https://reactjs.org/)
  
  **An Advanced Multi-Agent Prompt Engineering System**
</div>

## Description

Cortexa is a cutting-edge, production-ready multi-agent prompt engineering system that leverages the power of AI to automatically optimize, enhance, and evaluate prompts across various domains. Built with a sophisticated orchestration framework, Cortexa employs specialized AI agents that work collaboratively to transform raw prompts into highly effective, domain-specific optimized versions.

### What Problem Does Cortexa Solve?

In the era of Large Language Models (LLMs), the quality of prompts directly impacts the quality of AI-generated responses. Cortexa addresses the challenge of prompt engineering by:

- **Automating prompt optimization** - No more manual trial and error
- **Domain-specific enhancement** - Tailored improvements for different fields
- **Quality assurance** - Built-in evaluation and scoring mechanisms
- **Consistency** - Standardized prompt improvement across teams
- **Memory-enhanced processing** - Context-aware responses using conversation history

### Why Cortexa?

- **Save Time**: Reduce prompt engineering time from hours to seconds
- **Improve Quality**: Get better AI responses with optimized prompts
- **Scale Efficiently**: Handle multiple prompt optimization requests simultaneously
- **Learn Continuously**: System improves through memory and feedback loops

## Features

### Core Capabilities
- **Multi-Agent Architecture** - Specialized agents for classification, improvement, and evaluation
- **Domain Classification** - Automatic detection of prompt domains (Software Engineering, Data Science, Education, Business Strategy, etc.)
- **Quality Scoring** - Comprehensive evaluation metrics for prompt effectiveness
- **Iterative Refinement** - Multiple improvement cycles until quality threshold is met
- **Memory System** - Context retention across conversations for personalized responses
- **Async Processing** - Non-blocking workflow execution with background task management
- **LangGraph Integration** - Advanced workflow orchestration using LangGraph
- **Performance Monitoring** - Real-time metrics and workflow statistics
- **Security Features** - Input sanitization, PII detection, and content filtering
- **Modern React UI** - Beautiful, responsive frontend with real-time updates

### Advanced Features
- **Caching System** - Intelligent caching for improved performance
- **Circuit Breakers** - Fault tolerance and graceful degradation
- **Dead Letter Queue** - Failed request handling and retry mechanisms
- **Rate Limiting** - API protection and resource management
- **Workflow Cancellation** - Grace period for cancelling running workflows
- **Batch Processing** - Handle multiple prompts efficiently
- **Export Capabilities** - Export optimized prompts and analytics
- **API Documentation** - Interactive API documentation with FastAPI

## Installation

### Prerequisites
- Python 3.8 or higher
- Node.js 16+ and npm (for frontend)
- Git

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/cortexa.git
cd cortexa
```

### Step 2: Set Up Python Backend

#### Create Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

#### Install Python Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Configure Environment Variables
```bash
# Copy the example environment file
cp .env.example .env

# Edit .env and add your API keys
# Required: GOOGLE_API_KEY or OPENAI_API_KEY
# Optional: LANGSMITH_API_KEY for tracing
```

### Step 4: Set Up React Frontend
```bash
# Navigate to frontend directory
cd frontend-react

# Install dependencies
npm install

# Return to root directory
cd ..
```

## Usage

### Starting the Application

#### Run Both Backend and Frontend (Recommended)
```bash
# Terminal 1 - Start the backend server
python src/main.py

# Terminal 2 - Start the frontend development server
cd frontend-react
npm run dev
```

The application will be available at:
- Frontend: http://localhost:5173
- Backend API: http://localhost:8000
- API Documentation: http://localhost:8000/docs


### CLI Usage (LangGraph Studio)
```bash
# Run with LangGraph Studio for visual workflow debugging
python tools/run_langgraph_studio.py

# Performance benchmarking
python tools/performance_benchmark.py
```

## Technologies Used

### Backend
- **FastAPI** - Modern, fast web framework for building APIs
- **LangChain** - Framework for developing LLM applications
- **LangGraph** - Workflow orchestration and state management
- **Pydantic** - Data validation using Python type annotations
- **Google Generative AI** - Primary LLM provider (Gemini models)
- **OpenAI** - Alternative LLM provider support
- **SQLite** - Lightweight database for memory storage
- **Uvicorn** - Lightning-fast ASGI server
- **Python-dotenv** - Environment variable management

### Frontend
- **React 18** - Modern UI library
- **TypeScript** - Type-safe JavaScript
- **Vite** - Next-generation frontend tooling
- **Tailwind CSS** - Utility-first CSS framework
- **Radix UI** - Unstyled, accessible UI components
- **React Query** - Powerful data synchronization
- **Framer Motion** - Production-ready animation library
- **Recharts** - Composable charting library
- **React Router** - Declarative routing

### DevOps & Tools
- **Docker** - Containerization support
- **LangSmith** - LLM application monitoring and debugging
- **Pytest** - Testing framework
- **ESLint** - Code linting
- **Git** - Version control

## Contributing

We welcome contributions to Cortexa! Here's how you can help:

### Getting Started
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Make your changes
4. Write or update tests as needed
5. Ensure all tests pass (`pytest tests/`)
6. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
7. Push to the branch (`git push origin feature/AmazingFeature`)
8. Open a Pull Request

### Development Guidelines
- Follow PEP 8 for Python code
- Use ESLint configuration for TypeScript/JavaScript
- Write meaningful commit messages
- Add tests for new features
- Update documentation as needed
- Ensure backward compatibility

### Areas for Contribution
- Bug fixes and issue resolution
- New features and enhancements
- Documentation improvements
- Test coverage expansion
- UI/UX improvements
- Internationalization support
- Performance optimizations

### Code of Conduct
Please note that this project is released with a Contributor Code of Conduct. By participating in this project, you agree to abide by its terms.

## License

This project is licensed under the MIT License - see below for details:

```
MIT License

Copyright (c) 2024 Cortexa

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Contact

### Project Maintainer
- **GitHub**: [GitHub Profile](https://github.com/Abood991B)
- **Email**: abdulrahman16baidaq@gmail.com
- **LinkedIn**: [My LinkedIn](https://linkedin.com/in/abdulrahman-baidaq)

### Project Links
- **Repository**: [https://github.com/Abood991B/cortexa](https://github.com/Abood991B/cortexa)

## Acknowledgments

- Thanks to the LangChain team for the amazing framework
- Google AI for providing powerful language models
- The open-source community for continuous inspiration

---