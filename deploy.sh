#!/bin/bash
# Invoice Intelligence System - Quick Deployment Script

set -e

echo "==== Invoice Intelligence System - Quick Deployment ===="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Error: Docker is not installed${NC}"
    echo "Please install Docker from https://docs.docker.com/get-docker/"
    exit 1
fi

# Check if Docker daemon is running
if ! docker info > /dev/null 2>&1; then
    echo -e "${RED}Error: Docker daemon is not running${NC}"
    echo "Please start Docker Desktop or Docker service"
    exit 1
fi

echo -e "${GREEN}✓ Docker is installed and running${NC}"
echo ""

# Build Docker image
echo -e "${YELLOW}Building Docker image...${NC}"
docker build -t invoice-intelligence:latest .

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Docker image built successfully${NC}"
else
    echo -e "${RED}✗ Failed to build Docker image${NC}"
    exit 1
fi

echo ""

# Check if container is already running
if docker ps --format '{{.Names}}' | grep -q '^invoice-api$'; then
    echo -e "${YELLOW}Stopping existing invoice-api container...${NC}"
    docker stop invoice-api
    docker rm invoice-api
fi

# Run Docker container
echo -e "${YELLOW}Starting invoice-api container...${NC}"
docker run -d \
  -p 8000:8000 \
  -v "$(pwd)/temp_uploads:/app/temp_uploads" \
  --name invoice-api \
  --restart unless-stopped \
  invoice-intelligence:latest

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Container started successfully${NC}"
else
    echo -e "${RED}✗ Failed to start container${NC}"
    exit 1
fi

echo ""

# Wait for container to be healthy
echo -e "${YELLOW}Waiting for application to be ready...${NC}"
for i in {1..60}; do
    if curl -s http://localhost:8000/docs > /dev/null; then
        echo -e "${GREEN}✓ Application is ready!${NC}"
        break
    fi
    if [ $i -eq 60 ]; then
        echo -e "${RED}✗ Application failed to start within timeout${NC}"
        docker logs invoice-api
        exit 1
    fi
    echo -n "."
    sleep 1
done

echo ""
echo -e "${GREEN}==== Deployment Complete ====${NC}"
echo ""
echo "API is now running at:"
echo -e "  ${GREEN}http://localhost:8000${NC}"
echo ""
echo "API Documentation:"
echo -e "  ${GREEN}http://localhost:8000/docs${NC}"
echo ""
echo "Useful commands:"
echo "  View logs:      docker logs -f invoice-api"
echo "  Stop container: docker stop invoice-api"
echo "  Remove image:   docker rmi invoice-intelligence:latest"
echo ""
