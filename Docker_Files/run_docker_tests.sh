#!/bin/bash

# Colors for better readability
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}===========================================${NC}"
echo -e "${BLUE}    TopicMind Docker Compatibility Tests   ${NC}"
echo -e "${BLUE}===========================================${NC}"

# Create logs directory if it doesn't exist
mkdir -p logs/tests/individual_tests

# Container status check
echo -e "\n${YELLOW}Checking Docker container status...${NC}"
if docker-compose ps | grep -q "Up"; then
    echo -e "${GREEN}✅ Docker container is running.${NC}"
else
    echo -e "${RED}❌ Docker container is not running.${NC}"
    echo -e "${YELLOW}Starting Docker container...${NC}"
    docker-compose up -d
    sleep 10
    if docker-compose ps | grep -q "Up"; then
        echo -e "${GREEN}✅ Docker container is now running.${NC}"
    else
        echo -e "${RED}❌ Failed to start Docker container. Please check docker-compose logs.${NC}"
        exit 1
    fi
fi

# Check container logs for errors
echo -e "\n${YELLOW}Checking container logs for errors...${NC}"
docker-compose logs | grep -i "error" > logs/tests/individual_tests/docker_errors.log
ERROR_COUNT=$(cat logs/tests/individual_tests/docker_errors.log | wc -l)
if [ $ERROR_COUNT -gt 0 ]; then
    echo -e "${YELLOW}⚠️ Found $ERROR_COUNT error messages in logs.${NC}"
    echo -e "${YELLOW}See logs/tests/individual_tests/docker_errors.log for details.${NC}"
else
    echo -e "${GREEN}✅ No errors found in container logs.${NC}"
fi

# Port availability check
echo -e "\n${YELLOW}Checking port availability...${NC}"
if lsof -i :5001 | grep -q LISTEN; then
    echo -e "${GREEN}✅ API server (port 5001) is running.${NC}"
else
    echo -e "${RED}❌ API server (port 5001) is not running.${NC}"
fi

if lsof -i :8501 | grep -q LISTEN; then
    echo -e "${GREEN}✅ Streamlit UI (port 8501) is running.${NC}"
else
    echo -e "${RED}❌ Streamlit UI (port 8501) is not running.${NC}"
fi

# API health check
echo -e "\n${YELLOW}Running API tests...${NC}"
TIMEOUT_SECONDS=30
curl --max-time $TIMEOUT_SECONDS -s http://localhost:5001/health > /dev/null
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ API health endpoint is responding.${NC}"
    echo -e "${YELLOW}Running full API tests...${NC}"
    python tests/api_test.py 2>&1 | tee logs/tests/individual_tests/api_test_results.log
else
    echo -e "${RED}❌ API health endpoint is not responding (timeout after ${TIMEOUT_SECONDS}s).${NC}"
    echo -e "${YELLOW}This could be due to model loading delays.${NC}"
fi

# Basic UI check
echo -e "\n${YELLOW}Running basic UI test...${NC}"
curl --max-time $TIMEOUT_SECONDS -s http://localhost:8501 > /dev/null
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Streamlit UI is accessible.${NC}"
    echo -e "${YELLOW}For detailed UI testing, run: python tests/ui_test.py${NC}"
else
    echo -e "${RED}❌ Streamlit UI is not accessible.${NC}"
fi

# Environment variable handling check
echo -e "\n${YELLOW}Checking environment variable handling...${NC}"
if docker-compose logs | grep -q "OPENAI_API_KEY environment variable not found"; then
    echo -e "${GREEN}✅ Application correctly detects missing API key.${NC}"
else
    echo -e "${YELLOW}⚠️ No message found about missing API key.${NC}"
fi

# Summary
echo -e "\n${BLUE}===========================================${NC}"
echo -e "${BLUE}           Test Summary                    ${NC}"
echo -e "${BLUE}===========================================${NC}"
echo -e "Docker container status: ${GREEN}Running${NC}"
echo -e "API server (port 5001): $(lsof -i :5001 | grep -q LISTEN && echo "${GREEN}Running${NC}" || echo "${RED}Not running${NC}")"
echo -e "Streamlit UI (port 8501): $(lsof -i :8501 | grep -q LISTEN && echo "${GREEN}Running${NC}" || echo "${RED}Not running${NC}")"
echo -e "API health endpoint: $(curl --max-time $TIMEOUT_SECONDS -s http://localhost:5001/health > /dev/null && echo "${GREEN}Responding${NC}" || echo "${RED}Not responding${NC}")"
echo -e "Streamlit UI accessible: $(curl --max-time $TIMEOUT_SECONDS -s http://localhost:8501 > /dev/null && echo "${GREEN}Yes${NC}" || echo "${RED}No${NC}")"
echo -e "Error messages in logs: $([ $ERROR_COUNT -gt 0 ] && echo "${YELLOW}$ERROR_COUNT${NC}" || echo "${GREEN}None${NC}")"

echo -e "\n${BLUE}Complete test report available at:${NC}"
echo -e "${BLUE}logs/tests/individual_tests/docker_test_results.md${NC}"

# Generate timestamp
echo -e "\n${YELLOW}Test completed at: $(date)${NC}" 