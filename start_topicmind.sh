#!/bin/bash

# TopicMind Launch Script
# This script launches both the Flask backend and Streamlit frontend

# Set colors for better output readability
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${GREEN}==== TopicMind Launcher ====${NC}"

# Function to clean up processes when the script is terminated
cleanup() {
    echo -e "\n${YELLOW}Shutting down TopicMind services...${NC}"
    
    # Kill the backend process if it exists
    if [ -n "$BACKEND_PID" ]; then
        echo -e "Stopping backend server (PID: $BACKEND_PID)..."
        kill $BACKEND_PID 2>/dev/null || true
        wait $BACKEND_PID 2>/dev/null || true
        echo -e "${GREEN}Backend server stopped.${NC}"
    fi
    
    # Kill any streamlit processes
    echo -e "Stopping Streamlit frontend..."
    if [ -n "$FRONTEND_PID" ]; then
        kill $FRONTEND_PID 2>/dev/null || true
        wait $FRONTEND_PID 2>/dev/null || true
        echo -e "${GREEN}Streamlit frontend stopped.${NC}"
    else
        pkill -f "streamlit run" 2>/dev/null || true
    fi
    
    echo -e "${GREEN}TopicMind shutdown complete. Thank you for using TopicMind!${NC}"
    exit 0
}

# Set up trap to catch Ctrl+C and other termination signals
trap cleanup SIGINT SIGTERM

# Check if curl is available (needed for health check)
if ! command -v curl &> /dev/null; then
    echo -e "${YELLOW}Warning: curl not found. Health check will be skipped.${NC}"
    HAS_CURL=false
else
    HAS_CURL=true
fi

# Check if required files exist
if [ ! -f "app.py" ]; then
    echo -e "${RED}Error: app.py not found!${NC}"
    echo "Make sure you're running this script from the project root directory."
    exit 1
fi

if [ ! -f "frontend/streamlit_app.py" ]; then
    echo -e "${RED}Error: frontend/streamlit_app.py not found!${NC}"
    echo "Make sure the Streamlit frontend file exists."
    exit 1
fi

# Check if the logs directory exists, create if not
if [ ! -d "logs" ]; then
    echo -e "${YELLOW}Creating logs directory structure...${NC}"
    mkdir -p logs/gpt logs/semantic logs/summaries logs/eval
fi

# Check if .env file exists, warn if not
if [ ! -f ".env" ]; then
    echo -e "${YELLOW}Warning: .env file not found!${NC}"
    echo "You may need to set up your environment variables for OpenAI API."
    echo "Creating a basic .env file..."
    echo "OPENAI_API_KEY=your_openai_api_key_here" > .env
    echo "MODEL_DEVICE=cpu" >> .env
    echo -e "${YELLOW}Created basic .env file. Please edit it with your API key for full functionality.${NC}"
else
    echo -e "${GREEN}Found .env file.${NC}"
fi

# Check if ports are already in use
check_port() {
    local port=$1
    local service=$2
    
    if command -v lsof >/dev/null 2>&1; then
        if lsof -i :"$port" > /dev/null 2>&1; then
            echo -e "${YELLOW}Warning: Port $port is already in use!${NC}"
            
            # Get PID of process using this port
            local pid=$(lsof -t -i :"$port")
            
            # Check if we're in a terminal (interactive)
            if [ -t 0 ]; then
                echo -e "Do you want to kill the process using port $port? [y/N]: "
                read -r kill_process
                
                if [[ "$kill_process" =~ ^[Yy]$ ]]; then
                    kill_port_process "$port" "$pid"
                else
                    echo -e "${RED}Cannot start $service on port $port as it's already in use.${NC}"
                    echo -e "Please either kill the process using this port or edit app.py to use a different port."
                    exit 1
                fi
            else
                # Non-interactive mode - automatically kill the process
                echo -e "Automatically killing process using port $port in non-interactive mode..."
                kill_port_process "$port" "$pid"
            fi
        fi
    else
        echo -e "${YELLOW}Warning: 'lsof' command not available, cannot check if ports are in use.${NC}"
    fi
    
    return 0
}

# Helper function to kill a process using a specific port
kill_port_process() {
    local port=$1
    local pid=$2
    
    if [ -n "$pid" ]; then
        echo -e "Killing process (PID: $pid) using port $port..."
        kill -9 "$pid" 2>/dev/null
        sleep 1
        if ! lsof -i :"$port" > /dev/null 2>&1; then
            echo -e "${GREEN}Process killed successfully. Port $port is now available.${NC}"
        else
            echo -e "${RED}Failed to kill process using port $port.${NC}"
            echo -e "Please manually kill the process or use a different port."
            exit 1
        fi
    else
        echo -e "${RED}Failed to get PID of process using port $port.${NC}"
        return 1
    fi
}

# Add port conflict checking before starting services
echo -e "${BLUE}Checking if required ports are available...${NC}"
check_port 5002 "backend server"
check_port 8502 "Streamlit frontend"

# Set environment variables
export FLASK_APP=app.py
export FLASK_ENV=development

# Create log files
BACKEND_LOG="backend_output.log"
FRONTEND_LOG="frontend_output.log"
touch $BACKEND_LOG
touch $FRONTEND_LOG

# Start the Flask backend
echo -e "${BLUE}Starting Flask backend server...${NC}"

# Show steps of backend initialization
echo -e "${YELLOW}+ Loading dependencies and configuring environment${NC}"
echo -e "${YELLOW}+ Initializing Flask application${NC}"
echo -e "${YELLOW}+ Starting backend services:${NC}"
echo -e "  - Text preprocessor service"
echo -e "  - Embedding model service"
echo -e "  - Topic extraction service"
echo -e "  - Summarization service"
echo -e "  - API endpoints"

# Start the backend and redirect to log file
python app.py > $BACKEND_LOG 2>&1 &
BACKEND_PID=$!

# Check if backend process started
sleep 1
if ! ps -p $BACKEND_PID > /dev/null; then
    echo -e "${RED}Failed to start backend server!${NC}"
    echo "Check $BACKEND_LOG for details."
    tail -n 10 $BACKEND_LOG
    exit 1
fi

echo -e "${GREEN}Backend server started with PID: $BACKEND_PID${NC}"

# Wait for backend to initialize and check health
MAX_RETRIES=30
RETRY_COUNT=0
BACKEND_READY=false

echo -e "${BLUE}Waiting for backend to be fully initialized...${NC}"
echo -e "${YELLOW}Checking dependencies and loading models (this may take a minute)...${NC}"

# Display detailed initialization
echo -e "  → Checking TensorFlow and PyTorch modules"
echo -e "  → Loading NLTK resources"
echo -e "  → Initializing sentence embeddings model"
echo -e "  → Loading DistilBART summarization model"
echo -e "  → Creating API routes and endpoints"

if [ "$HAS_CURL" = true ]; then
    echo -n "Waiting for backend to be ready"
    while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
        # Check if the process is still running
        if ! ps -p $BACKEND_PID > /dev/null; then
            echo -e "${RED}\nBackend process terminated unexpectedly!${NC}"
            echo "Check $BACKEND_LOG for details."
            tail -n 20 $BACKEND_LOG
            exit 1
        fi
        
        # Try to connect to health endpoint
        HEALTH_CHECK=$(curl -s http://localhost:5002/health || echo "")
        if [[ $HEALTH_CHECK == *"\"status\":\"up\""* ]]; then
            echo -e "\n${GREEN}Backend is ready and healthy!${NC}"
            
            # Extract model status from health check if available
            if [[ $HEALTH_CHECK == *"\"models_loaded\""* ]]; then
                if [[ $HEALTH_CHECK == *"\"models_loaded\":true"* ]]; then
                    echo -e "  ✓ ${GREEN}Models loaded successfully${NC}"
                else
                    echo -e "  ✗ ${YELLOW}Models not fully loaded${NC}"
                fi
            fi
            
            BACKEND_READY=true
            # Extra sleep to ensure everything is ready
            sleep 2
            break
        fi
        
        echo -n "."
        sleep 1
        RETRY_COUNT=$((RETRY_COUNT + 1))
    done
else
    # If curl isn't available, just wait
    echo "Waiting 20 seconds for backend initialization (curl not available for health check)..."
    sleep 20
    BACKEND_READY=true
fi

if [ "$BACKEND_READY" = false ]; then
    echo -e "${RED}\nBackend health check failed after ${MAX_RETRIES} attempts!${NC}"
    echo "The Flask server might still be starting up. Check $BACKEND_LOG for details."
    tail -n 20 $BACKEND_LOG
    echo -e "${YELLOW}Continuing anyway, but Streamlit might not connect properly...${NC}"
fi

# Start the Streamlit frontend
echo -e "${BLUE}Starting Streamlit frontend...${NC}"

# Export required environment variables for Streamlit
export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
export STREAMLIT_SERVER_PORT=8502

# Start Streamlit
echo -e "${YELLOW}+ Initializing Streamlit interface${NC}"
echo -e "${YELLOW}+ Loading UI components${NC}"
echo -e "${YELLOW}+ Setting up API connections${NC}"

streamlit run frontend/streamlit_app.py --server.port=8502 --server.address=127.0.0.1 > $FRONTEND_LOG 2>&1 &
FRONTEND_PID=$!

# Check if frontend started successfully
sleep 3
if ! ps -p $FRONTEND_PID > /dev/null; then
    echo -e "${RED}Failed to start Streamlit frontend!${NC}"
    echo "Check $FRONTEND_LOG for details."
    tail -n 20 $FRONTEND_LOG
    kill $BACKEND_PID
    exit 1
fi

echo -e "${GREEN}Streamlit frontend started with PID: $FRONTEND_PID${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "TopicMind is now running!"
echo -e "Access the web interface at: ${YELLOW}http://localhost:8502${NC}"
echo -e "API endpoint is at: ${YELLOW}http://localhost:5002${NC}"
echo -e "Press Ctrl+C to shut down all services."
echo -e "${GREEN}========================================${NC}"

# Keep the script running to allow for clean termination
while true; do
    sleep 1
    # Check if either process has died
    if ! ps -p $BACKEND_PID > /dev/null; then
        echo -e "${RED}Backend server stopped unexpectedly!${NC}"
        echo "Check $BACKEND_LOG for details."
        cleanup
    fi
    if ! ps -p $FRONTEND_PID > /dev/null; then
        echo -e "${RED}Frontend stopped unexpectedly!${NC}"
        echo "Check $FRONTEND_LOG for details."
        cleanup
    fi
done 