

#!/bin/bash
# Invoice Intelligence System - Using Docker Compose

set -e

echo "==== Invoice Intelligence System - Docker Compose Deployment ===="
echo ""

# Check if docker-compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "Error: docker-compose is not installed"
    echo "Please install Docker Compose from https://docs.docker.com/compose/install/"
    exit 1
fi

echo "Starting services with Docker Compose..."
docker-compose up -d

echo ""
echo "Waiting for services to be ready..."
sleep 10

# Check if services are running
if docker-compose ps | grep -q "invoice-api.*Up"; then
    echo ""
    echo "✓ Services are running successfully!"
    echo ""
    echo "API is now available at:"
    echo "  http://localhost:8000"
    echo ""
    echo "API Documentation:"
    echo "  http://localhost:8000/docs"
    echo ""
    echo "Useful commands:"
    echo "  View logs:      docker-compose logs -f"
    echo "  Stop services:  docker-compose down"
    echo "  Restart:        docker-compose restart"
    echo ""
else
    echo "Failed to start services. Checking logs..."
    docker-compose logs
    exit 1
fi
