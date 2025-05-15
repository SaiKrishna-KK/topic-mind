#!/bin/bash
set -e

echo "Starting TopicMind application..."

# Environment setup
export PYTHONUNBUFFERED=1
mkdir -p /app/models/cache
mkdir -p /app/logs/gpt /app/logs/semantic /app/logs/summaries /app/logs/eval

# Ensure required data directories exist
echo "Setting up data directories..."
mkdir -p /app/data/embeddings
mkdir -p /app/data/models

# Download NLTK data if needed
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

# Start the Flask backend API server in the background
echo "Starting Flask backend server on port 5001..."
python app.py &
API_PID=$!

# Function to check if the API is healthy
check_api_health() {
    for i in {1..50}; do
        echo "Checking API health ($i/50)..."
        if curl -s http://localhost:5001/health > /dev/null; then
            echo "✅ API is healthy and responding!"
            return 0
        fi
        echo "⏳ API not yet healthy. Retrying in 5 seconds..."
        sleep 5
    done
    echo "❌ API failed to become healthy within timeout period."
    return 1
}

# Check if API started successfully
echo "Waiting for API to become available..."
if check_api_health; then
    echo "Starting Streamlit frontend on port 8501..."
    streamlit run frontend/streamlit_app.py --server.port=8501 --server.address=0.0.0.0 &
    STREAMLIT_PID=$!
    
    # Monitor both processes
    echo "TopicMind is running! Backend on port 5001, Frontend on port 8501."
    
    # Wait for either process to exit
    wait $API_PID $STREAMLIT_PID
else
    echo "Failed to start API server. Check logs for details."
    # Kill the API process if it's running but not healthy
    kill -0 $API_PID 2>/dev/null && kill $API_PID
    exit 1
fi 