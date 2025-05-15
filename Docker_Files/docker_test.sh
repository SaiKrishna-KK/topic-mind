#!/bin/bash

# Colors for better readability
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}====== TopicMind Docker Test ======${NC}"

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Error: Docker is not installed.${NC}"
    echo "Please install Docker and try again."
    exit 1
fi

# Check if docker-compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo -e "${RED}Error: docker-compose is not installed.${NC}"
    echo "Please install docker-compose and try again."
    exit 1
fi

# Function to clean up
cleanup() {
    echo -e "\n${YELLOW}Stopping Docker containers...${NC}"
    docker-compose down
    echo -e "${GREEN}Docker containers stopped.${NC}"
    exit 0
}

# Set up trap
trap cleanup SIGINT SIGTERM

# Build and start Docker containers
echo -e "${GREEN}Building Docker image...${NC}"
docker-compose build

echo -e "${GREEN}Starting Docker containers...${NC}"
docker-compose up -d

echo -e "${YELLOW}Waiting for services to start...${NC}"
sleep 10  # Give some time for services to start

# Check if services are running
if docker-compose ps | grep -q "Up"; then
    echo -e "${GREEN}TopicMind is running in Docker!${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo -e "🌐 Access the web interface at: ${YELLOW}http://localhost:8501${NC}"
    echo -e "🔌 API endpoint is at: ${YELLOW}http://localhost:5001${NC}"
    
    # Get container OS info
    echo -e "\n${BLUE}Container OS Information:${NC}"
    docker-compose exec topicmind cat /etc/os-release
    
    # Get Python version
    echo -e "\n${BLUE}Python Version:${NC}"
    docker-compose exec topicmind python --version
    
    # Check if backend is healthy
    echo -e "\n${BLUE}Checking API Health:${NC}"
    if curl -s http://localhost:5001/health | grep -q "\"status\":\"ok\""; then
        echo -e "${GREEN}✓ API is healthy${NC}"
    else
        echo -e "${RED}✗ API health check failed${NC}"
        echo -e "${YELLOW}Showing recent logs:${NC}"
        docker-compose logs --tail 20 topicmind
    fi
    
    echo -e "\n${GREEN}========================================${NC}"
    echo -e "Press Ctrl+C to stop the Docker containers."
    
    # Keep the script running to allow for Ctrl+C
    while true; do
        sleep 1
    done
else
    echo -e "${RED}Failed to start Docker containers!${NC}"
    echo -e "${YELLOW}Showing logs:${NC}"
    docker-compose logs topicmind
    cleanup
fi 