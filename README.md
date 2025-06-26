# 🤖 Multi-Model AI Comparison Tool

A production-ready web application that allows you to query and compare responses from multiple AI models (Claude, ChatGPT, and Gemini) simultaneously. Built with Streamlit and designed for both development and production environments.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](https://www.docker.com/)

## ✨ Features

### 🎯 Core Functionality
- **Multi-Model Querying**: Query Claude, ChatGPT, and Gemini APIs simultaneously
- **Side-by-Side Comparison**: View responses in multiple layouts (side-by-side, sequential, detailed analysis)
- **Dynamic Model Discovery**: Automatically fetch available models from each provider
- **Smart Model Defaults**: Configurable default models with use case optimization
- **Response Copying**: Enhanced clipboard functionality with multiple copy methods

### 🔍 Enhanced Debugging & Monitoring
- **Advanced Gemini Debugging**: Detailed error reporting with actionable troubleshooting tips
- **API Health Monitoring**: Real-time health checks for all providers
- **Performance Metrics**: Response time tracking and success rate monitoring
- **Request/Response Logging**: Comprehensive logging for debugging API issues

### 💰 Budget Management
- **Usage Tracking**: Monitor API costs and token consumption
- **Budget Limits**: Set daily and monthly spending limits with alerts
- **Cost Estimation**: Real-time cost estimates before making requests
- **Provider Comparison**: Cost breakdown by model and provider

### 🧪 Testing & Quality Assurance
- **Comprehensive Test Suite**: 10+ test categories covering all aspects
- **Load Testing**: Concurrent request handling validation
- **Error Scenario Testing**: Robust error handling verification
- **API Connectivity Tests**: Automated validation of API keys and endpoints

### 🚀 Production-Ready Deployment
- **Docker Support**: Full containerization with Docker Compose
- **SSL/TLS Security**: Automatic certificate generation and HTTPS support
- **Reverse Proxy**: Nginx configuration with rate limiting
- **Monitoring Stack**: Prometheus and Grafana integration
- **Health Checks**: Application and dependency health monitoring

## 🚀 Quick Start

### Prerequisites
- Python 3.9+ 
- Docker and Docker Compose (for containerized deployment)
- API keys for at least one provider (Claude, OpenAI, or Google)

### 1. Clone and Setup
```bash
git clone <repository-url>
cd multi-model-ai-comparison

# Create environment file
make setup-env
```

### 2. Configure API Keys
Edit the `.env` file with your API keys:
```bash
# Required: Add your API keys
ANTHROPIC_API_KEY=your_claude_api_key_here
OPENAI_API_KEY=your_openai_api_key_here  
GOOGLE_API_KEY=your_google_api_key_here

# Optional: Configure defaults
DEFAULT_CLAUDE_MODEL=claude-3-5-sonnet-20241022
DEFAULT_OPENAI_MODEL=gpt-4o
DEFAULT_GEMINI_MODEL=gemini-1.5-pro-latest

# Optional: Enable budget tracking
TRACK_USAGE=true
DAILY_BUDGET_USD=10.00
MONTHLY_BUDGET_USD=100.00
```

### 3. Test API Connectivity
```bash
# Test all APIs
make test-apis

# Test specific provider
make test-gemini-debug
```

### 4. Run the Application

#### Development Mode
```bash
# Local development
pip install -r requirements.txt
streamlit run app.py

# Or with Docker
make up-dev
```

#### Production Deployment
```bash
# Full production stack with nginx, SSL, monitoring
make up-prod

# Access at https://localhost
```

## 📖 Usage Guide

### Basic Usage
1. **Configure API Keys**: Add your API keys in the sidebar
2. **Select Models**: Choose models for each provider (auto-selected by default)
3. **Choose Strategy**: Select optimization strategy (default, fast, quality, cost-effective)
4. **Enter Prompt**: Type your question or prompt
5. **Compare Results**: View responses in your preferred layout

### Advanced Features

#### Model Selection Strategies
- **Default**: Balanced performance and cost
- **Fast**: Optimized for quick responses
- **Quality**: Best available models for highest quality
- **Cost-Effective**: Cheapest models to minimize costs

#### Comparison Modes
- **Side by Side**: View all responses in columns
- **Sequential**: View responses one after another
- **Detailed Analysis**: Comprehensive comparison with metrics

#### Budget Management
- Enable tracking in `.env`: `TRACK_USAGE=true`
- Set daily/monthly limits
- Monitor real-time usage in the dashboard
- Get cost estimates before sending requests

## 🛠️ Development

### Local Development Setup
```bash
# Setup development environment
make dev-setup

# Run with hot reload
make dev-run

# Run tests
make test-all

# Format code
make format

# Lint code
make lint
```

### Project Structure
```
├── app.py                    # Main Streamlit application
├── clipboard_utils.py        # Enhanced copy functionality
├── default_models.py         # Smart model configuration
├── budget_tracker.py         # Usage and cost tracking
├── monitoring.py            # Performance monitoring
├── test_suite.py            # Comprehensive testing
├── docker-compose.yml       # Production deployment
├── docker-compose.dev.yml   # Development environment
├── Makefile                 # Build and deployment commands
├── nginx.conf               # Nginx reverse proxy config
├── prometheus.yml           # Monitoring configuration
└── README.md               # This file
```

### Environment Variables

#### API Configuration
| Variable | Description | Default |
|----------|-------------|---------|
| `ANTHROPIC_API_KEY` | Claude API key | Required |
| `OPENAI_API_KEY` | OpenAI API key | Required |
| `GOOGLE_API_KEY` | Google/Gemini API key | Required |

#### Model Defaults
| Variable | Description | Default |
|----------|-------------|---------|
| `DEFAULT_CLAUDE_MODEL` | Default Claude model | `claude-3-5-sonnet-20241022` |
| `DEFAULT_OPENAI_MODEL` | Default OpenAI model | `gpt-4o` |
| `DEFAULT_GEMINI_MODEL` | Default Gemini model | `gemini-1.5-pro-latest` |
| `AUTO_SELECT_MODELS` | Auto-select defaults | `true` |

#### Budget Tracking
| Variable | Description | Default |
|----------|-------------|---------|
| `TRACK_USAGE` | Enable usage tracking | `false` |
| `DAILY_BUDGET_USD` | Daily spending limit | `10.00` |
| `MONTHLY_BUDGET_USD` | Monthly spending limit | `100.00` |
| `BUDGET_HARD_LIMIT` | Block requests over budget | `false` |

#### Application Settings
| Variable | Description | Default |
|----------|-------------|---------|
| `DEBUG_MODE` | Enable debug logging | `false` |
| `LOG_LEVEL` | Logging level | `INFO` |
| `HOST_PORT` | Application port | `8501` |
| `METRICS_ENABLED` | Enable metrics collection | `true` |

## 🐳 Deployment Options

### Development Deployment
```bash
# Local development with hot reload
make up-dev

# Access at http://localhost:8501
```

### Production Deployment
```bash
# Full production stack
make up-prod

# With monitoring (Prometheus + Grafana)
make up-monitoring

# Access at https://localhost
# Monitoring at http://localhost:9090 (Prometheus)
# Dashboards at http://localhost:3000 (Grafana)
```

### Docker Commands
```bash
# Build images
make build

# Start services
make up-prod

# View logs
make logs

# Stop services
make down

# Clean up
make clean
```

## 🧪 Testing

### Automated Testing
```bash
# Run all tests
make test-all

# API connectivity tests
make test-apis

# Load testing
make test-load

# Unit tests
make test-unit

# Output test results to file
python test_suite.py test --output test_results.json
```

### Test Categories
- **API Key Validation**: Verify API keys are valid and working
- **Model Discovery**: Test dynamic model fetching
- **Basic Connectivity**: Simple request/response validation
- **Response Format**: Ensure consistent response structure
- **Error Handling**: Validate graceful error handling
- **Concurrent Requests**: Test parallel request handling
- **Rate Limiting**: Verify rate limit compliance
- **Large Prompts**: Test handling of long inputs
- **Special Characters**: Unicode and symbol handling
- **Timeout Handling**: Network timeout resilience

## 🔧 Troubleshooting

### Common Issues

#### Gemini API Problems
- **Use the Debug Panel**: Access enhanced debugging in the sidebar
- **Check API Key Format**: Ensure you're using Google AI Studio API key
- **Verify Model Names**: Some models may not be available in your region

#### Configuration Issues
```bash
# Check environment configuration
make check-env

# Validate Docker Compose files
make validate-config

# Test API connectivity
make test-connectivity
```

#### Performance Issues
```bash
# View performance metrics
make metrics

# Check application health
make health

# Monitor in real-time
make up-monitoring
```

#### SSL/HTTPS Issues
```bash
# Generate self-signed certificates
make generate-ssl

# Check nginx configuration
make debug-nginx
```

### Getting Help
1. **Check Logs**: Use `make logs` to view application logs
2. **Run Diagnostics**: Use `make test-apis` for comprehensive testing
3. **Debug Panel**: Use the enhanced Gemini debug panel for API issues
4. **Health Checks**: Monitor application health with `make health`

## 📊 Monitoring & Observability

### Built-in Monitoring
- **Response Time Tracking**: Monitor API performance
- **Success Rate Monitoring**: Track request success rates
- **Cost Tracking**: Monitor API usage costs
- **Error Reporting**: Detailed error analysis and troubleshooting

### Production Monitoring
```bash
# Deploy with monitoring stack
make up-monitoring

# Access monitoring dashboards
# Prometheus: http://localhost:9090
# Grafana: http://localhost:3000 (admin/admin)
```

### Metrics Collection
- API response times and success rates
- Token usage and cost analysis
- Error patterns and troubleshooting data
- System resource utilization

## 🤝 Contributing

### Development Workflow
1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/your-feature`
3. **Make changes and add tests**
4. **Run the test suite**: `make test-all`
5. **Format code**: `make format`
6. **Lint code**: `make lint`
7. **Submit a pull request**

### Code Quality Standards
- **Python 3.9+ compatibility**
- **Type hints where appropriate**
- **Comprehensive test coverage**
- **Clear documentation and comments**
- **Follow existing code style**

### Testing Requirements
- All new features must include tests
- Existing tests must pass
- Code coverage should not decrease
- API integration tests must pass

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### Third-Party Services
This application integrates with third-party AI services:
- **Anthropic Claude**: Subject to [Anthropic's Terms](https://www.anthropic.com/terms)
- **OpenAI**: Subject to [OpenAI's Terms](https://openai.com/terms/)
- **Google AI**: Subject to [Google AI Terms](https://ai.google.dev/terms)

**Important**: Users are responsible for:
- Obtaining their own API keys
- Monitoring and managing API costs
- Complying with each service's terms of use
- Ensuring appropriate use of AI-generated content

## 🚀 Roadmap

### Planned Features
- [ ] **Additional AI Providers**: Integration with more AI services
- [ ] **Response Caching**: Smart caching to reduce API costs
- [ ] **Batch Processing**: Process multiple prompts simultaneously
- [ ] **Custom Model Fine-tuning**: Integration with custom models
- [ ] **Advanced Analytics**: Deeper insights into model performance
- [ ] **Team Collaboration**: Multi-user support and shared workspaces
- [ ] **API Rate Optimization**: Intelligent rate limiting and queueing
- [ ] **Export Capabilities**: Enhanced export options (PDF, CSV, etc.)

### Version History
- **v0.1.0**: Initial release with basic comparison functionality
- **v0.2.0**: Added enhanced debugging and monitoring (current)

## 🙋‍♂️ Support

### Getting Help
- **GitHub Issues**: [Report bugs or request features](https://github.com/yourusername/multi-model-ai-comparison/issues)
- **Discussions**: [Community discussions and Q&A](https://github.com/yourusername/multi-model-ai-comparison/discussions)
- **Documentation**: Check this README and inline documentation

### Reporting Issues
When reporting issues, please include:
- **Environment details** (OS, Python version, Docker version)
- **Configuration** (anonymized .env contents)
- **Steps to reproduce** the issue
- **Expected vs actual behavior**
- **Log output** (use `make logs`)

---

**Made with ❤️ for the AI community**