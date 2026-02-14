# Contributing to CortexaAI

Thank you for your interest in contributing to CortexaAI! This document provides guidelines and information for contributors.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Contributing Guidelines](#contributing-guidelines)
- [Pull Request Process](#pull-request-process)
- [Issue Reporting](#issue-reporting)
- [Code Style](#code-style)
- [Testing](#testing)
- [Documentation](#documentation)

## Code of Conduct

This project adheres to a code of conduct that we expect all contributors to follow. Please be respectful and constructive in all interactions.

### Our Standards

- Use welcoming and inclusive language
- Be respectful of differing viewpoints and experiences
- Gracefully accept constructive criticism
- Focus on what is best for the community
- Show empathy towards other community members

## Getting Started

### Prerequisites

- Python 3.10 or higher
- Node.js 18 or higher
- Git
- At least one LLM API key (Google Gemini recommended — [free tier](https://aistudio.google.com/))

### Fork and Clone

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/Abood991B/CortexaAI.git
   cd CortexaAI
   ```

## Development Setup

### Backend Setup

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

4. Run tests:
   ```bash
   python -m pytest tests/
   ```

5. Start the backend:
   ```bash
   python src/main.py
   ```

### Frontend Setup

1. Navigate to frontend directory:
   ```bash
   cd frontend-react
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Start development server:
   ```bash
   npm run dev
   ```

4. Run tests:
   ```bash
   npm test
   ```

## Contributing Guidelines

### Types of Contributions

We welcome several types of contributions:

- **Bug fixes**: Fix issues in the codebase
- **Feature additions**: Add new functionality
- **Documentation improvements**: Enhance or fix documentation
- **Performance optimizations**: Improve system performance
- **Test coverage**: Add or improve tests
- **Security enhancements**: Improve security measures

### Before You Start

1. Check existing issues and pull requests to avoid duplication
2. Create an issue to discuss major changes before implementation
3. Ensure your development environment is properly set up
4. Read through the codebase to understand the architecture

## Pull Request Process

### 1. Create a Branch

Create a feature branch from `main`:
```bash
git checkout -b feature/your-feature-name
```

Use descriptive branch names:
- `feature/add-new-domain-classifier`
- `bugfix/fix-memory-leak-in-evaluator`
- `docs/update-api-documentation`

### 2. Make Changes

- Write clean, readable code
- Follow the existing code style
- Add tests for new functionality
- Update documentation as needed
- Ensure all tests pass

### 3. Commit Changes

Write clear, descriptive commit messages:
```bash
git commit -m "feat: add support for custom evaluation criteria

- Add CustomEvaluationCriteria class
- Update PromptEvaluator to use custom criteria
- Add tests for custom evaluation functionality
- Update API documentation"
```

Use conventional commit format:
- `feat:` for new features
- `fix:` for bug fixes
- `docs:` for documentation changes
- `test:` for test additions/changes
- `refactor:` for code refactoring
- `perf:` for performance improvements

### 4. Push and Create PR

```bash
git push origin feature/your-feature-name
```

Create a pull request with:
- Clear title and description
- Reference to related issues
- Screenshots for UI changes
- Test results and coverage information

### 5. Code Review

- Address reviewer feedback promptly
- Keep discussions constructive and professional
- Update your branch with the latest `main` if needed
- Ensure CI/CD checks pass

## Issue Reporting

### Bug Reports

When reporting bugs, include:

- **Environment**: OS, Python version, Node.js version
- **Steps to reproduce**: Clear, step-by-step instructions
- **Expected behavior**: What should happen
- **Actual behavior**: What actually happens
- **Error messages**: Full error messages and stack traces
- **Screenshots**: If applicable

### Feature Requests

For feature requests, provide:

- **Use case**: Why is this feature needed?
- **Proposed solution**: How should it work?
- **Alternatives**: Other approaches considered
- **Additional context**: Any other relevant information

## Code Style

### Python Code Style

- Follow PEP 8 guidelines
- Use type hints for all function parameters and return values
- Write docstrings for all public functions and classes
- Use meaningful variable and function names
- Keep functions focused and small

Example:
```python
async def classify_prompt(self, prompt: str) -> Dict[str, Any]:
    """
    Classify a prompt into a domain with security and caching.

    Args:
        prompt: The prompt to classify

    Returns:
        Dict containing classification results

    Raises:
        ClassificationError: If classification fails after retries
    """
```

### TypeScript Code Style

- Use TypeScript strict mode
- Follow React best practices
- Use meaningful component and variable names
- Write JSDoc comments for complex functions
- Use proper typing for all props and state

Example:
```typescript
interface PromptProcessorProps {
  onPromptSubmit: (prompt: string) => Promise<void>;
  isLoading: boolean;
}

const PromptProcessor: React.FC<PromptProcessorProps> = ({
  onPromptSubmit,
  isLoading
}) => {
  // Component implementation
};
```

### General Guidelines

- Use consistent indentation (2 spaces for TS/JS, 4 spaces for Python)
- Keep lines under 100 characters when possible
- Use meaningful comments for complex logic
- Remove unused imports and variables
- Use consistent naming conventions

## Testing

### Backend Testing

- Write unit tests for all new functions
- Include integration tests for API endpoints
- Test error handling and edge cases
- Aim for >80% code coverage

Run tests:
```bash
python -m pytest tests/ -v --cov=agents --cov=src --cov=config --cov=core --cov-report=html
```

### Frontend Testing

- Write unit tests for components
- Include integration tests for user workflows
- Test accessibility and responsive design
- Use React Testing Library best practices

Run tests:
```bash
npm test
npm run test:coverage
```

### Linting & Type Checking

```bash
# Python lint
ruff check .

# Python type check
mypy agents/ src/ config/ core/

# Frontend lint
cd frontend-react && npm run lint
```

### Test Guidelines

- Write descriptive test names
- Use arrange-act-assert pattern
- Mock external dependencies
- Test both success and failure scenarios

## Documentation

### Code Documentation

- Write clear docstrings and comments
- Document complex algorithms and business logic
- Keep README.md up to date
- Update API documentation for changes

### API Documentation

- Document all endpoints with examples
- Include request/response schemas
- Document error responses
- Provide usage examples

### User Documentation

- Update user guides for new features
- Include screenshots and examples
- Write clear installation instructions
- Document configuration options

## Release Process

### Version Numbering

We use Semantic Versioning (SemVer):
- `MAJOR.MINOR.PATCH`
- Major: Breaking changes
- Minor: New features (backward compatible)
- Patch: Bug fixes (backward compatible)

### Release Checklist

1. Update version numbers
2. Update CHANGELOG.md
3. Run full test suite
4. Update documentation
5. Create release notes
6. Tag the release
7. Deploy to production

## Getting Help

### Communication Channels

- **Issues**: For bug reports and feature requests
- **Discussions**: For questions and general discussion
- **Email**: For security issues or private matters

### Resources

- [Project Documentation](README.md)
- [API Documentation](http://localhost:8000/docs)
- [Frontend Application](http://localhost:5173)
- [Deployment Guides](docs/deployment/)
- [Changelog](CHANGELOG.md)

## Recognition

Contributors will be recognized in:
- CHANGELOG.md for significant contributions
- README.md contributors section
- Release notes for major features

## License

By contributing to this project, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to CortexaAI! Your contributions help make this project better for everyone.
