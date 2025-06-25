# Makefile for Multi-Model AI Comparison Tool

# Variables
IMAGE_NAME := multi-model-ai-comparison
CONTAINER_NAME := ai-comparison-app
HOST_PORT := 8501
CONTAINER_PORT := 8501
TAG := latest

# Default target
.PHONY: help
help: ## Show this help message
	@echo "Multi-Model AI Comparison Tool - Docker Commands"
	@echo "================================================"
	@awk 'BEGIN {FS = ":.*##"} /^[a-zA-Z_-]+:.*##/ { printf "  %-15s %s\n", $$1, $$2 }' $(MAKEFILE_LIST)

# Build the Docker image
.PHONY: build
build: ## Build the Docker image
	@echo "Building Docker image: $(IMAGE_NAME):$(TAG)"
	docker build -t $(IMAGE_NAME):$(TAG) .
	@echo "✅ Build complete!"

# Run the container
.PHONY: run
run: ## Run the container with port forwarding
	@echo "Starting container: $(CONTAINER_NAME)"
	@echo "Accessible at: http://localhost:$(HOST_PORT)"
	docker run -d \
		--name $(CONTAINER_NAME) \
		-p $(HOST_PORT):$(CONTAINER_PORT) \
		--restart unless-stopped \
		$(IMAGE_NAME):$(TAG)
	@echo "✅ Container started!"
	@echo "🌐 Open http://localhost:$(HOST_PORT) in your browser"

# Run the container with environment variables
.PHONY: run-env
run-env: ## Run the container with API keys from environment variables
	@echo "Starting container with environment variables: $(CONTAINER_NAME)"
	@echo "Loading API keys from: ANTHROPIC_API_KEY, OPENAI_API_KEY, GOOGLE_API_KEY"
	@echo "Accessible at: http://localhost:$(HOST_PORT)"
	docker run -d \
		--name $(CONTAINER_NAME) \
		-p $(HOST_PORT):$(CONTAINER_PORT) \
		-e ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY}" \
		-e OPENAI_API_KEY="${OPENAI_API_KEY}" \
		-e GOOGLE_API_KEY="${GOOGLE_API_KEY}" \
		--restart unless-stopped \
		$(IMAGE_NAME):$(TAG)
	@echo "✅ Container started with environment variables!"
	@echo "🌐 Open http://localhost:$(HOST_PORT) in your browser"

# Run the container interactively
.PHONY: run-interactive
run-interactive: ## Run the container interactively (foreground)
	@echo "Starting container interactively: $(CONTAINER_NAME)"
	@echo "Accessible at: http://localhost:$(HOST_PORT)"
	@echo "Press Ctrl+C to stop"
	docker run --rm \
		--name $(CONTAINER_NAME) \
		-p $(HOST_PORT):$(CONTAINER_PORT) \
		$(IMAGE_NAME):$(TAG)

# Stop the container
.PHONY: stop
stop: ## Stop the running container
	@echo "Stopping container: $(CONTAINER_NAME)"
	-docker stop $(CONTAINER_NAME)
	@echo "✅ Container stopped!"

# Restart the container
.PHONY: restart
restart: stop run ## Restart the container

# Show container logs
.PHONY: logs
logs: ## Show container logs
	docker logs -f $(CONTAINER_NAME)

# Show container status
.PHONY: status
status: ## Show container status
	@echo "Container Status:"
	@docker ps -a --filter name=$(CONTAINER_NAME) --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

# Execute shell in running container
.PHONY: shell
shell: ## Execute shell in the running container
	docker exec -it $(CONTAINER_NAME) /bin/bash

# Clean up containers
.PHONY: clean-containers
clean-containers: ## Remove all containers for this app
	@echo "Removing containers..."
	-docker stop $(CONTAINER_NAME)
	-docker rm $(CONTAINER_NAME)
	@echo "✅ Containers cleaned!"

# Clean up images
.PHONY: clean-images
clean-images: ## Remove the Docker image
	@echo "Removing Docker image: $(IMAGE_NAME)"
	-docker rmi $(IMAGE_NAME):$(TAG)
	@echo "✅ Images cleaned!"

# Full cleanup
.PHONY: clean
clean: clean-containers clean-images ## Clean up both containers and images
	@echo "🧹 Full cleanup complete!"

# Clean up everything Docker-related (use with caution)
.PHONY: clean-all-docker
clean-all-docker: ## Clean up all Docker containers, images, and volumes (DANGEROUS!)
	@echo "⚠️  WARNING: This will remove ALL Docker containers, images, and volumes!"
	@read -p "Are you sure? Type 'yes' to continue: " confirm && [ "$$confirm" = "yes" ]
	docker system prune -af --volumes
	@echo "🧹 All Docker resources cleaned!"

# Development targets
.PHONY: dev-build
dev-build: ## Build image with development dependencies
	docker build -t $(IMAGE_NAME):dev --target dev .

.PHONY: dev-run
dev-run: ## Run development container with volume mounting
	docker run --rm -it \
		--name $(CONTAINER_NAME)-dev \
		-p $(HOST_PORT):$(CONTAINER_PORT) \
		-v $(PWD):/app \
		$(IMAGE_NAME):dev

# Health check
.PHONY: health
health: ## Check if the container is healthy
	@docker inspect --format='{{.State.Health.Status}}' $(CONTAINER_NAME) 2>/dev/null || echo "Container not running"

# Quick commands
.PHONY: up
up: build run ## Build and run the container

.PHONY: up-env  
up-env: build run-env ## Build and run with environment variables

.PHONY: down
down: clean-containers ## Stop and remove containers

# Show Docker resource usage
.PHONY: stats
stats: ## Show Docker resource usage
	docker stats $(CONTAINER_NAME) --no-stream

# Export/Import image
.PHONY: export
export: ## Export the Docker image to a tar file
	docker save $(IMAGE_NAME):$(TAG) | gzip > $(IMAGE_NAME)-$(TAG).tar.gz
	@echo "✅ Image exported to $(IMAGE_NAME)-$(TAG).tar.gz"

.PHONY: import
import: ## Import the Docker image from a tar file
	@if [ ! -f "$(IMAGE_NAME)-$(TAG).tar.gz" ]; then \
		echo "❌ File $(IMAGE_NAME)-$(TAG).tar.gz not found!"; \
		exit 1; \
	fi
	gunzip -c $(IMAGE_NAME)-$(TAG).tar.gz | docker load
	@echo "✅ Image imported successfully"

# Show environment info
.PHONY: info
info: ## Show Docker and system information
	@echo "Docker Version:"
	@docker --version
	@echo "\nDocker Info:"
	@docker system df
	@echo "\nImage Info:"
	@docker images $(IMAGE_NAME) --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}\t{{.CreatedAt}}"