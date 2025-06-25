# Multi-Model AI Comparison Tool - Docker Setup

A containerized tool to query and compare responses from Claude, ChatGPT, and Gemini in parallel.

## 🚀 Quick Start

### Prerequisites
- Docker installed and running
- Make (optional, for convenience commands)

### 1. Clone/Setup Files
Ensure you have these files in your project directory:
```
multi-model-ai-comparison/
├── app.py              # Main application
├── Dockerfile          # Container definition
├── pyproject.toml      # Python dependencies
├── Makefile           # Docker commands
├── .dockerignore      # Docker build exclusions
└── README.md          # This file
```

### 2. Build and Run

**Option A: Manual API Key Entry**
```bash
# Build the Docker image
make build

# Run the container (background)
make run

# Or build and run in one command
make up
```

**Option B: Environment Variables (Recommended)**
```bash
# Set your API keys
export ANTHROPIC_API_KEY="your_claude_key_here"
export OPENAI_API_KEY="your_openai_key_here" 
export GOOGLE_API_KEY="your_gemini_key_here"

# Build and run with environment variables
make up-env
```

The application will be available at: **http://localhost:8501**

### 3. Stop and Clean Up
```bash
# Stop the container
make stop

# Remove containers and images
make clean
```

## 📋 Available Commands

### Basic Operations
```bash
make build              # Build the Docker image
make run               # Run container in background
make run-env           # Run with environment variables
make run-interactive   # Run container in foreground
make stop              # Stop the running container
make restart           # Restart the container
make up                # Build and run (convenience)
make up-env            # Build and run with env vars
make down              # Stop and remove containers
```

### Monitoring & Debugging
```bash
make logs              # Show container logs
make status            # Show container status
make health            # Check container health
make stats             # Show resource usage
make shell             # Open shell in container
```

### Cleanup
```bash
make clean-containers  # Remove containers only
make clean-images      # Remove images only
make clean             # Remove both containers and images
make clean-all-docker  # Remove ALL Docker resources (⚠️ DANGEROUS)
```

### Development
```bash
make dev-build         # Build with dev dependencies
make dev-run           # Run with volume mounting for development
```

### Import/Export
```bash
make export            # Export image to tar.gz file
make import            # Import image from tar.gz file
make info              # Show Docker system information
```

## 🔑 API Keys Setup

### Option 1: Environment Variables (Recommended)
```bash
# Set environment variables (Linux/Mac)
export ANTHROPIC_API_KEY="your_claude_key_here"
export OPENAI_API_KEY="your_openai_key_here"
export GOOGLE_API_KEY="your_gemini_key_here"

# Windows PowerShell
$env:ANTHROPIC_API_KEY="your_claude_key_here"
$env:OPENAI_API_KEY="your_openai_key_here"
$env:GOOGLE_API_KEY="your_gemini_key_here"

# Then run
make up-env
```

### Option 2: Manual Entry
1. Run `make up`
2. Open http://localhost:8501
3. Enter API keys in the sidebar
4. Check "Remember keys this session" for convenience

### Get API Keys:
- **Anthropic Claude**: https://console.anthropic.com
- **OpenAI ChatGPT**: https://platform.openai.com
- **Google Gemini**: https://ai.google.dev

## ✨ Features

### Environment Variable Integration
- ✅ **Auto-loads** API keys from environment variables
- ✅ **Manual override** - can still change keys in the UI
- ✅ **Source tracking** - shows whether key is from env var or manual entry
- ✅ **Secure** - environment variables are ideal for production

### Session Management
- ✅ **Results persistence** - survives UI interactions (Statistics, export, etc.)
- ✅ **Smart key handling** - choose between security and convenience
- ⚠️ **Refresh behavior** - page refresh clears everything (for security)

### Model Selection
- ✅ **Latest models** - Claude Opus 4, Sonnet 4, GPT-4o, Gemini 1.5 Pro
- ✅ **Fallback models** - older reliable models if latest aren't available
- ✅ **Error context** - helpful tips for common API issues

## 🔧 Configuration

### Environment Variables
The app recognizes these environment variables:
- `ANTHROPIC_API_KEY` - Your Claude API key
- `OPENAI_API_KEY` - Your OpenAI API key  
- `GOOGLE_API_KEY` - Your Google/Gemini API key

### Docker with Environment Variables
```bash
# Direct docker command
docker run -d \
  -p 8501:8501 \
  -e ANTHROPIC_API_KEY="your_key" \
  -e OPENAI_API_KEY="your_key" \
  -e GOOGLE_API_KEY="your_key" \
  multi-model-ai-comparison

# Or use make command
make run-env
```

### Change Port
Edit `HOST_PORT` in `Makefile`:
```makefile
HOST_PORT := 8502
```

### Development Mode
For active development with file watching:
```bash
# Run with volume mounting (changes reflect immediately)
make dev-run
```

## 🐛 Troubleshooting

### Container won't start
```bash
# Check if port 8501 is already in use
sudo lsof -i :8501

# Check container logs
make logs

# Check container status
make status
```

### Build issues
```bash
# Clean everything and rebuild
make clean
make build

# Check Docker system
make info
```

### API Key Issues
- **Environment variables not loading**: Check `make logs` for startup messages
- **Keys not persisting**: Enable "Remember keys this session" checkbox
- **Refresh clears everything**: This is intentional for security

### Common API Errors
- **Claude 503**: Temporary server overload - try again in a few minutes
- **OpenAI 429**: Rate limited - check your API plan
- **Gemini 404**: Model not available in your region/plan

## 📁 Project Structure

```
multi-model-ai-comparison/
├── app.py                 # Streamlit application with env var support
├── Dockerfile            # Multi-stage Docker build
├── pyproject.toml        # Python dependencies & config
├── Makefile             # Docker management commands (including run-env)
├── .dockerignore        # Build context exclusions
├── LICENSE              # MIT License with AI service disclaimers
└── README.md            # Setup instructions
```

## 🔒 Security Notes

- **Environment variables**: Preferred method for API keys
- **Session storage**: Optional, cleared on refresh for security
- **Container isolation**: API keys never persist in Docker images
- **No localStorage**: Keys never stored in browser storage
- **HTTPS recommended**: Use reverse proxy for production

## 🎯 Usage Tips

1. **Use environment variables** for seamless experience
2. **Start with one model** to verify setup  
3. **Enable session memory** for development convenience
4. **Try different models** for the same prompt to see differences
5. **Export results** before refreshing the page
6. **Monitor costs** - each API call costs money

## 🚢 Production Deployment

For production deployment, consider:
- Using environment variables for all API keys
- Adding reverse proxy (nginx) with SSL/TLS
- Implementing authentication/authorization
- Adding rate limiting and cost monitoring
- Using Docker Compose for multi-service setup
- Setting up log aggregation and monitoring

## 🤝 Contributing

1. Set environment variables: `export ANTHROPIC_API_KEY=...`
2. Make changes to `app.py`  
3. Test locally: `make dev-run`
4. Rebuild: `make build`
5. Test container: `make run-env`

Happy comparing! 🤖