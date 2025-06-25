# Enhanced Makefile for Multi-Model AI Comparison Tool

# Variables
IMAGE_NAME := multi-model-ai-comparison
CONTAINER_NAME := ai-comparison-app
HOST_PORT := 8501
CONTAINER_PORT := 8501
TAG := latest
ENV_FILE := .env
COMPOSE_FILE := docker-compose.yml
DEV_COMPOSE_FILE := docker-compose.dev.yml

# Colors for output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[1;33m
BLUE := \033[0;34m
NC := \033[0m # No Color

# Default target
.PHONY: help
help: ## Show this help message
	@echo "$(BLUE)Multi-Model AI Comparison Tool - Enhanced Commands$(NC)"
	@echo "=================================================="
	@echo ""
	@echo "$(YELLOW)🚀 Quick Start:$(NC)"
	@echo "  make setup-env     # Create .env file from template"
	@echo "  make test-apis     # Test API connectivity"
	@echo "  make up-prod       # Deploy to production"
	@echo ""
	@echo "$(YELLOW)📋 Available Commands:$(NC)"
	@awk 'BEGIN {FS = ":.*##"} /^[a-zA-Z_-]+:.*##/ { printf "  $(GREEN)%-15s$(NC) %s\n", $$1, $$2 }' $(MAKEFILE_LIST)

# =============================================================================
# 🔧 Setup & Configuration
# =============================================================================

.PHONY: setup-env
setup-env: ## Create .env file from template
	@if [ ! -f $(ENV_FILE) ]; then \
		echo "$(YELLOW)📝 Creating $(ENV_FILE) from template...$(NC)"; \
		cp .env.example $(ENV_FILE); \
		echo "$(GREEN)✅ $(ENV_FILE) created! Please edit it with your API keys.$(NC)"; \
	else \
		echo "$(YELLOW)⚠️  $(ENV_FILE) already exists$(NC)"; \
	fi

.PHONY: check-env
check-env: ## Check if environment is properly configured
	@echo "$(BLUE)🔍 Checking environment configuration...$(NC)"
	@python3 test_suite.py deploy --check-only

.PHONY: generate-ssl
generate-ssl: ## Generate self-signed SSL certificates
	@echo "$(BLUE)🔒 Generating SSL certificates...$(NC)"
	@python3 test_suite.py ssl

.PHONY: validate-config
validate-config: ## Validate configuration files
	@echo "$(BLUE)📋 Validating configuration files...$(NC)"
	@if [ -f $(COMPOSE_FILE) ]; then \
		docker-compose -f $(COMPOSE_FILE) config > /dev/null && \
		echo "$(GREEN)✅ docker-compose.yml is valid$(NC)" || \
		echo "$(RED)❌ docker-compose.yml has errors$(NC)"; \
	fi
	@if [ -f nginx.conf ]; then \
		nginx -t -c $(PWD)/nginx.conf > /dev/null 2>&1 && \
		echo "$(GREEN)✅ nginx.conf is valid$(NC)" || \
		echo "$(YELLOW)⚠️  nginx.conf validation failed (nginx not installed or config issues)$(NC)"; \
	fi

# =============================================================================
# 🧪 Testing
# =============================================================================

.PHONY: test-apis
test-apis: ## Run comprehensive API tests
	@echo "$(BLUE)🧪 Running API test suite...$(NC)"
	@python3 test_suite.py test --output test_results.json
	@echo "$(GREEN)✅ Test results saved to test_results.json$(NC)"

.PHONY: test-apis-yaml
test-apis-yaml: ## Run API tests and output YAML
	@echo "$(BLUE)🧪 Running API test suite (YAML output)...$(NC)"
	@python3 test_suite.py test --format yaml --output test_results.yaml
	@echo "$(GREEN)✅ Test results saved to test_results.yaml$(NC)"

.PHONY: test-connectivity
test-connectivity: ## Quick connectivity test for all APIs
	@echo "$(BLUE)🌐 Testing API connectivity...$(NC)"
	@python3 -c "import asyncio; from test_suite import APITestRunner; runner = APITestRunner(); asyncio.run(runner.test_basic_connectivity())"

.PHONY: test-gemini-debug
test-gemini-debug: ## Run detailed Gemini debugging tests
	@echo "$(BLUE)🔍 Running Gemini debug tests...$(NC)"
	@python3 -c "import asyncio; from test_suite import APITestRunner; runner = APITestRunner(); asyncio.run(runner.test_api_key_validation())"

.PHONY: test-load
test-load: ## Run load testing (requires API keys)
	@echo "$(BLUE)⚡ Running load tests...$(NC)"
	@python3 -c "import asyncio; from test_suite import APITestRunner; runner = APITestRunner(); asyncio.run(runner.test_concurrent_requests())"

.PHONY: test-unit
test-unit: ## Run unit tests
	@echo "$(BLUE)🔬 Running unit tests...$(NC)"
	@pytest tests/ -v --cov=app --cov-report=html --cov-report=term

.PHONY: test-all
test-all: test-unit test-apis ## Run all tests (unit + API)
	@echo "$(GREEN)✅ All tests completed$(NC)"

# =============================================================================
# 🐳 Docker Operations
# =============================================================================

.PHONY: build
build: ## Build the Docker image
	@echo "$(BLUE)🔨 Building Docker image: $(IMAGE_NAME):$(TAG)$(NC)"
	docker build -t $(IMAGE_NAME):$(TAG) .
	@echo "$(GREEN)✅ Build complete!$(NC)"

.PHONY: build-dev
build-dev: ## Build development Docker image
	@echo "$(BLUE)🔨 Building development Docker image...$(NC)"
	docker build -f Dockerfile.dev -t $(IMAGE_NAME):dev .
	@echo "$(GREEN)✅ Development build complete!$(NC)"

.PHONY: run
run: ## Run the container with port forwarding
	@echo "$(BLUE)🚀 Starting container: $(CONTAINER_NAME)$(NC)"
	@echo "$(YELLOW)📡 Accessible at: http://localhost:$(HOST_PORT)$(NC)"
	docker run -d \
		--name $(CONTAINER_NAME) \
		-p $(HOST_PORT):$(CONTAINER_PORT) \
		--restart unless-stopped \
		$(IMAGE_NAME):$(TAG)
	@echo "$(GREEN)✅ Container started!$(NC)"
	@echo "$(BLUE)🌐 Open http://localhost:$(HOST_PORT) in your browser$(NC)"

.PHONY: run-env
run-env: ## Run the container with API keys from environment variables
	@echo "$(BLUE)🚀 Starting container with environment variables: $(CONTAINER_NAME)$(NC)"
	@echo "$(YELLOW)🔑 Loading API keys from: ANTHROPIC_API_KEY, OPENAI_API_KEY, GOOGLE_API_KEY$(NC)"
	@echo "$(YELLOW)📡 Accessible at: http://localhost:$(HOST_PORT)$(NC)"
	docker run -d \
		--name $(CONTAINER_NAME) \
		-p $(HOST_PORT):$(CONTAINER_PORT) \
		--env-file $(ENV_FILE) \
		--restart unless-stopped \
		$(IMAGE_NAME):$(TAG)
	@echo "$(GREEN)✅ Container started with environment variables!$(NC)"
	@echo "$(BLUE)🌐 Open http://localhost:$(HOST_PORT) in your browser$(NC)"

# =============================================================================
# 🚀 Production Deployment
# =============================================================================

.PHONY: up-prod
up-prod: setup-env validate-config ## Deploy to production using Docker Compose
	@echo "$(BLUE)🚀 Deploying to production...$(NC)"
	@python3 test_suite.py deploy --env production
	@echo "$(GREEN)✅ Production deployment complete!$(NC)"
	@echo "$(BLUE)🌐 Application available at: https://localhost$(NC)"

.PHONY: up-dev
up-dev: ## Deploy to development environment
	@echo "$(BLUE)🔧 Deploying to development...$(NC)"
	@python3 test_suite.py deploy --env development
	@echo "$(GREEN)✅ Development deployment complete!$(NC)"

.PHONY: up-monitoring
up-monitoring: ## Deploy with monitoring stack (Prometheus + Grafana)
	@echo "$(BLUE)📊 Deploying with monitoring stack...$(NC)"
	docker-compose --profile monitoring up -d --build
	@echo "$(GREEN)✅ Monitoring stack deployed!$(NC)"
	@echo "$(BLUE)📊 Prometheus: http://localhost:9090$(NC)"
	@echo "$(BLUE)📈 Grafana: http://localhost:3000 (admin/admin)$(NC)"

.PHONY: down
down: ## Stop and remove all containers
	@echo "$(BLUE)🛑 Stopping all services...$(NC)"
	-docker-compose down --remove-orphans
	-docker-compose -f $(DEV_COMPOSE_FILE) down --remove-orphans
	@echo "$(GREEN)✅ All services stopped$(NC)"

.PHONY: down-volumes
down-volumes: ## Stop containers and remove volumes
	@echo "$(BLUE)🛑 Stopping services and removing volumes...$(NC)"
	-docker-compose down --volumes --remove-orphans
	@echo "$(GREEN)✅ Services stopped and volumes removed$(NC)"

# =============================================================================
# 📊 Monitoring & Debugging
# =============================================================================

.PHONY: logs
logs: ## Show container logs
	docker-compose logs -f

.PHONY: logs-app
logs-app: ## Show application logs only
	docker-compose logs -f multi-model-ai

.PHONY: status
status: ## Show container status
	@echo "$(BLUE)📊 Container Status:$(NC)"
	@docker-compose ps

.PHONY: health
health: ## Check application health
	@echo "$(BLUE)🏥 Checking application health...$(NC)"
	@curl -f http://localhost:$(HOST_PORT)/_stcore/health >/dev/null 2>&1 && \
		echo "$(GREEN)✅ Application is healthy$(NC)" || \
		echo "$(RED)❌ Application health check failed$(NC)"

.PHONY: metrics
metrics: ## Show performance metrics
	@echo "$(BLUE)📈 Performance Metrics:$(NC)"
	@docker stats --no-stream $(CONTAINER_NAME) 2>/dev/null || \
		echo "$(YELLOW)⚠️  Container not running$(NC)"

.PHONY: debug
debug: ## Enter debug shell in running container
	@echo "$(BLUE)🔍 Entering debug shell...$(NC)"
	docker-compose exec multi-model-ai /bin/bash

.PHONY: debug-nginx
debug-nginx: ## Debug nginx configuration
	@echo "$(BLUE)🔍 Testing nginx configuration...$(NC)"
	docker-compose exec nginx nginx -t

# =============================================================================
# 🧹 Cleanup Operations
# =============================================================================

.PHONY: clean-containers
clean-containers: ## Remove all containers for this app
	@echo "$(BLUE)🧹 Removing containers...$(NC)"
	-docker stop $(CONTAINER_NAME)
	-docker rm $(CONTAINER_NAME)
	-docker-compose down --remove-orphans
	@echo "$(GREEN)✅ Containers cleaned!$(NC)"

.PHONY: clean-images
clean-images: ## Remove the Docker image
	@echo "$(BLUE)🧹 Removing Docker image: $(IMAGE_NAME)$(NC)"
	-docker rmi $(IMAGE_NAME):$(TAG)
	-docker rmi $(IMAGE_NAME):dev
	@echo "$(GREEN)✅ Images cleaned!$(NC)"

.PHONY: clean-volumes
clean-volumes: ## Remove all volumes
	@echo "$(BLUE)🧹 Removing volumes...$(NC)"
	-docker volume rm $(shell docker volume ls -q | grep $(IMAGE_NAME) 2>/dev/null)
	@echo "$(GREEN)✅ Volumes cleaned!$(NC)"

.PHONY: clean
clean: clean-containers clean-images ## Clean up both containers and images
	@echo "$(GREEN)🧹 Full cleanup complete!$(NC)"

.PHONY: clean-all
clean-all: clean-containers clean-images clean-volumes ## Clean everything including volumes
	@echo "$(GREEN)🧹 Complete cleanup finished!$(NC)"

.PHONY: clean-docker-system
clean-docker-system: ## Clean up all Docker resources (DANGEROUS!)
	@echo "$(RED)⚠️  WARNING: This will remove ALL Docker containers, images, and volumes!$(NC)"
	@read -p "Are you sure? Type 'yes' to continue: " confirm && [ "$$confirm" = "yes" ]
	docker system prune -af --volumes
	@echo "$(GREEN)🧹 All Docker resources cleaned!$(NC)"

# =============================================================================
# 🔄 Development Workflow
# =============================================================================

.PHONY: dev-setup
dev-setup: setup-env build-dev ## Setup development environment
	@echo "$(GREEN)✅ Development environment ready!$(NC)"

.PHONY: dev-run
dev-run: ## Run development container with volume mounting
	@echo "$(BLUE)🔧 Starting development container...$(NC)"
	docker run --rm -it \
		--name $(CONTAINER_NAME)-dev \
		-p $(HOST_PORT):$(CONTAINER_PORT) \
		-v $(PWD):/app \
		--env-file $(ENV_FILE) \
		$(IMAGE_NAME):dev

.PHONY: dev-test
dev-test: ## Run tests in development environment
	@echo "$(BLUE)🧪 Running tests in development environment...$(NC)"
	docker-compose -f $(DEV_COMPOSE_FILE) run --rm multi-model-ai-dev \
		python -m pytest tests/ -v

.PHONY: format
format: ## Format code with black
	@echo "$(BLUE)🎨 Formatting code...$(NC)"
	black app.py test_suite.py
	@echo "$(GREEN)✅ Code formatted!$(NC)"

.PHONY: lint
lint: ## Lint code with flake8
	@echo "$(BLUE)🔍 Linting code...$(NC)"
	flake8 app.py test_suite.py
	@echo "$(GREEN)✅ Linting complete!$(NC)"

.PHONY: type-check
type-check: ## Check types with mypy
	@echo "$(BLUE)🔍 Type checking...$(NC)"
	mypy app.py test_suite.py
	@echo "$(GREEN)✅ Type checking complete!$(NC)"

.PHONY: pre-commit
pre-commit: format lint type-check test-unit ## Run all pre-commit checks
	@echo "$(GREEN)✅ All pre-commit checks passed!$(NC)"

# =============================================================================
# 📦 Export/Import Operations
# =============================================================================

.PHONY: export
export: ## Export the Docker image to a tar file
	@echo "$(BLUE)📦 Exporting Docker image...$(NC)"
	docker save $(IMAGE_NAME):$(TAG) | gzip > $(IMAGE_NAME)-$(TAG).tar.gz
	@echo "$(GREEN)✅ Image exported to $(IMAGE_NAME)-$(TAG).tar.gz$(NC)"

.PHONY: import
import: ## Import the Docker image from a tar file
	@if [ ! -f "$(IMAGE_NAME)-$(TAG).tar.gz" ]; then \
		echo "$(RED)❌ File $(IMAGE_NAME)-$(TAG).tar.gz not found!$(NC)"; \
		exit 1; \
	fi
	@echo "$(BLUE)📦 Importing Docker image...$(NC)"
	gunzip -c $(IMAGE_NAME)-$(TAG).tar.gz | docker load
	@echo "$(GREEN)✅ Image imported successfully$(NC)"

.PHONY: backup
backup: ## Backup volumes and configuration
	@echo "$(BLUE)💾 Creating backup...$(NC)"
	@mkdir -p backups/$(shell date +%Y%m%d_%H%M%S)
	@cp $(ENV_FILE) backups/$(shell date +%Y%m%d_%H%M%S)/ 2>/dev/null || true
	@cp $(COMPOSE_FILE) backups/$(shell date +%Y%m%d_%H%M%S)/
	@echo "$(GREEN)✅ Backup created in backups/$(NC)"

# =============================================================================
# 📊 Reporting & Analysis
# =============================================================================

.PHONY: report
report: ## Generate comprehensive system report
	@echo "$(BLUE)📊 Generating system report...$(NC)"
	@echo "# Multi-Model AI Comparison Tool - System Report" > system_report.md
	@echo "Generated: $(shell date)" >> system_report.md
	@echo "" >> system_report.md
	@echo "## Environment Status" >> system_report.md
	@make check-env >> system_report.md 2>&1 || true
	@echo "" >> system_report.md
	@echo "## Container Status" >> system_report.md
	@make status >> system_report.md 2>&1 || true
	@echo "" >> system_report.md
	@echo "## API Test Results" >> system_report.md
	@make test-apis >> system_report.md 2>&1 || true
	@echo "$(GREEN)✅ System report generated: system_report.md$(NC)"

.PHONY: info
info: ## Show comprehensive system information
	@echo "$(BLUE)📋 System Information:$(NC)"
	@echo ""
	@echo "$(YELLOW)Docker Version:$(NC)"
	@docker --version
	@echo ""
	@echo "$(YELLOW)Docker Compose Version:$(NC)"
	@docker-compose --version
	@echo ""
	@echo "$(YELLOW)System Resources:$(NC)"
	@docker system df
	@echo ""
	@echo "$(YELLOW)Application Images:$(NC)"
	@docker images $(IMAGE_NAME) --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}\t{{.CreatedAt}}" 2>/dev/null || \
		echo "No images found"
	@echo ""
	@echo "$(YELLOW)Running Containers:$(NC)"
	@docker ps --filter "name=$(CONTAINER_NAME)" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" 2>/dev/null || \
		echo "No containers running"

# =============================================================================
# 🎯 Quick Actions
# =============================================================================

.PHONY: quick-start
quick-start: setup-env build test-connectivity up-prod ## Complete quick start setup
	@echo "$(GREEN)🎉 Quick start complete! Application is running.$(NC)"

.PHONY: restart
restart: down up-prod ## Restart the application
	@echo "$(GREEN)✅ Application restarted$(NC)"

.PHONY: update
update: down build up-prod ## Update and restart the application
	@echo "$(GREEN)✅ Application updated and restarted$(NC)"

.PHONY: reset
reset: down-volumes clean up-prod ## Complete reset (removes all data)
	@echo "$(GREEN)✅ Application completely reset$(NC)"

# =============================================================================
# 📚 Documentation & Help
# =============================================================================

.PHONY: docs
docs: ## Generate documentation
	@echo "$(BLUE)📚 Generating documentation...$(NC)"
	@echo "Documentation would be generated here (not implemented)"

.PHONY: check-updates
check-updates: ## Check for available updates
	@echo "$(BLUE)🔍 Checking for updates...$(NC)"
	@echo "Update checking would be implemented here"

.PHONY: version
version: ## Show version information
	@echo "$(BLUE)Multi-Model AI Comparison Tool$(NC)"
	@echo "Version: 1.0.0"
	@echo "Build: $(shell date +%Y%m%d_%H%M%S)"

# Support for old command names (backwards compatibility)
.PHONY: up up-env
up: build run ## Build and run the container (alias for build + run)
up-env: build run-env ## Build and run with environment variables (alias for build + run-env)