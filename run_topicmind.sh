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
        echo "Stopping backend (PID: $BACKEND_PID)..."
        kill $BACKEND_PID 2>/dev/null || true
    fi
    
    # Kill any streamlit processes
    echo "Stopping Streamlit frontend..."
    pkill -f "streamlit run" 2>/dev/null || true
    
    echo -e "${GREEN}Shutdown complete.${NC}"
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

# Check if .env file exists, warn if not
if [ ! -f ".env" ]; then
    echo -e "${YELLOW}Warning: .env file not found!${NC}"
    echo "You may need to set up your environment variables for OpenAI API."
    echo "Create a .env file with OPENAI_API_KEY=your_key_here"
fi

# Start the Flask backend
echo -e "${BLUE}Starting Flask backend server...${NC}"
# Create a named pipe for real-time log viewing
PIPE=$(mktemp -u)
mkfifo $PIPE
# Start showing logs in background
cat $PIPE | grep --color=auto -E "PyTorch|TensorFlow|Hugging Face|SentenceTransformers|Model|^|$" &
LOG_PID=$!
# Start the backend and redirect to pipe
python app.py > $PIPE 2>&1 &
BACKEND_PID=$!

# Check if backend process started
sleep 1
if ! ps -p $BACKEND_PID > /dev/null; then
    echo -e "${RED}Failed to start backend server!${NC}"
    # Clean up pipe and log reader
    kill $LOG_PID 2>/dev/null || true
    rm -f $PIPE
    exit 1
fi

echo -e "${GREEN}Backend server started with PID: $BACKEND_PID${NC}"

# Wait for backend to initialize and check health
MAX_RETRIES=30
RETRY_COUNT=0
BACKEND_READY=false

echo -e "${BLUE}Waiting for backend to be fully initialized...${NC}"
echo -e "${YELLOW}Checking dependencies and loading models...${NC}"

if [ "$HAS_CURL" = true ]; then
    while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
        # Check if the process is still running
        if ! ps -p $BACKEND_PID > /dev/null; then
            echo -e "${RED}Backend process terminated unexpectedly!${NC}"
            # Clean up pipe and log reader
            kill $LOG_PID 2>/dev/null || true
            rm -f $PIPE
            exit 1
        fi
        
        # Try to connect to health endpoint
        HEALTH_CHECK=$(curl -s http://localhost:5001/health || echo "")
        if [[ $HEALTH_CHECK == *"\"status\":\"ok\""* ]]; then
            echo -e "${GREEN}Backend is ready and healthy!${NC}"
            BACKEND_READY=true
            break
        fi
        
        echo -n "."
        sleep 1
        RETRY_COUNT=$((RETRY_COUNT + 1))
    done
else
    # If curl isn't available, just wait a bit longer
    sleep 10
    BACKEND_READY=true
fi

# Clean up log display
kill $LOG_PID 2>/dev/null || true
rm -f $PIPE

if [ "$BACKEND_READY" = false ]; then
    echo -e "${RED}\nBackend health check failed after ${MAX_RETRIES} attempts!${NC}"
    echo "The Flask server might still be starting up. Check backend.log for details."
    kill $BACKEND_PID
    exit 1
fi

# Start the Streamlit frontend
echo -e "${BLUE}Starting Streamlit frontend...${NC}"
streamlit run frontend/streamlit_app.py > frontend.log 2>&1 &
FRONTEND_PID=$!

# Check if frontend started successfully
sleep 3
if ! ps -p $FRONTEND_PID > /dev/null; then
    echo -e "${RED}Failed to start Streamlit frontend!${NC}"
    echo "Check frontend.log for details."
    kill $BACKEND_PID
    exit 1
fi

echo -e "${GREEN}Streamlit frontend started!${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "TopicMind is now running!"
echo -e "Access the web interface at: ${YELLOW}http://localhost:8501${NC}"
echo -e "API endpoint is at: ${YELLOW}http://localhost:5001${NC}"
echo -e "Press Ctrl+C to shut down all services."
echo -e "${GREEN}========================================${NC}"

# Keep the script running to allow for clean termination
while true; do
    sleep 1
    # Check if either process has died
    if ! ps -p $BACKEND_PID > /dev/null; then
        echo -e "${RED}Backend server stopped unexpectedly!${NC}"
        echo "Check backend.log for details."
        cleanup
    fi
    if ! ps -p $FRONTEND_PID > /dev/null; then
        echo -e "${RED}Frontend stopped unexpectedly!${NC}"
        echo "Check frontend.log for details."
        cleanup
    fi
done 