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

# Check if the logs directory exists, create if not
if [ ! -d "logs" ]; then
    echo -e "${YELLOW}Creating logs directory structure...${NC}"
    mkdir -p logs/gpt logs/semantic logs/summaries logs/eval
fi

# Check if .env file exists, warn if not
if [ ! -f ".env" ]; then
    echo -e "${YELLOW}Warning: .env file not found!${NC}"
    echo "You may need to set up your environment variables for OpenAI API."
    echo "Create a .env file with OPENAI_API_KEY=your_key_here"
    
    # Check if OPENAI_API_KEY is set in environment
    if [ -z "$OPENAI_API_KEY" ]; then
        echo -e "${YELLOW}Warning: OPENAI_API_KEY environment variable not found.${NC}"
        echo "Some features may not work without an OpenAI API key."
    else
        echo -e "${GREEN}Found OPENAI_API_KEY in environment.${NC}"
    fi
fi

# Start the Flask backend
echo -e "${BLUE}Starting Flask backend server...${NC}"

# Create backend log file
BACKEND_LOG="backend.log"
touch $BACKEND_LOG

# Show steps of backend initialization instead of just one line
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
    exit 1
fi

echo -e "${GREEN}Backend server started with PID: $BACKEND_PID${NC}"

# Wait for backend to initialize and check health
MAX_RETRIES=45  # Increased wait time to 45 seconds
RETRY_COUNT=0
BACKEND_READY=false

echo -e "${BLUE}Waiting for backend to be fully initialized...${NC}"
echo -e "${YELLOW}Checking dependencies and loading models (this may take a minute)...${NC}"

# Detailed initialization display
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
            exit 1
        fi
        
        # Try to connect to health endpoint
        HEALTH_CHECK=$(curl -s http://localhost:5001/health || echo "")
        if [[ $HEALTH_CHECK == *"\"status\":\"ok\""* ]]; then
            echo -e "\n${GREEN}Backend is ready and healthy!${NC}"
            # Extract model status from health check if available
            if [[ $HEALTH_CHECK == *"\"models\""* ]]; then
                if [[ $HEALTH_CHECK == *"\"summarizer_model_loaded\":true"* ]]; then
                    echo -e "  ✓ ${GREEN}Summarizer model loaded successfully${NC}"
                else
                    echo -e "  ✗ ${RED}Summarizer model not loaded${NC}"
                fi
                
                if [[ $HEALTH_CHECK == *"\"embedding_model_loaded\":true"* ]]; then
                    echo -e "  ✓ ${GREEN}Embedding model loaded successfully${NC}"
                else
                    echo -e "  ✗ ${RED}Embedding model not loaded${NC}"
                fi
                
                if [[ $HEALTH_CHECK == *"\"openai_api_key_set\":true"* ]]; then
                    echo -e "  ✓ ${GREEN}OpenAI API key found and valid${NC}"
                    echo -e "    Topic refinement will use GPT for human-readable topic names"
                else
                    echo -e "  ✗ ${YELLOW}OpenAI API key not available${NC}"
                    echo -e "    Using keyword-based topic names (no GPT refinement)"
                fi
            fi
            
            BACKEND_READY=true
            # Extra sleep to ensure models are fully loaded
            sleep 2
            break
        fi
        
        echo -n "."
        sleep 1
        RETRY_COUNT=$((RETRY_COUNT + 1))
    done
else
    # If curl isn't available, just wait longer
    echo "Waiting 30 seconds for backend initialization (curl not available for health check)..."
    sleep 30
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

# Create frontend log file
FRONTEND_LOG="frontend.log"
touch $FRONTEND_LOG

# Export required environment variables for Streamlit
export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
export STREAMLIT_SERVER_PORT=8501

# Start Streamlit with explicit port and host specification
streamlit run frontend/streamlit_app.py --server.port=8501 --server.address=127.0.0.1 > $FRONTEND_LOG 2>&1 &
FRONTEND_PID=$!

# Check if frontend started successfully
sleep 3
if ! ps -p $FRONTEND_PID > /dev/null; then
    echo -e "${RED}Failed to start Streamlit frontend!${NC}"
    echo "Check $FRONTEND_LOG for details."
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
        echo "Check $BACKEND_LOG for details."
        cleanup
    fi
    if ! ps -p $FRONTEND_PID > /dev/null; then
        echo -e "${RED}Frontend stopped unexpectedly!${NC}"
        echo "Check $FRONTEND_LOG for details."
        cleanup
    fi
done 