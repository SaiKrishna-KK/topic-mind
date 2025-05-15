FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy .env.example as .env (will be overridden if volume mounted)
COPY .env.example .env

# Copy application code
COPY . .

# Ensure logs directory exists
RUN mkdir -p logs/gpt logs/semantic logs/summaries logs/eval

# Expose ports for Flask API and Streamlit
EXPOSE 5001 8501

# Create a startup script that runs both services
RUN echo '#!/bin/bash\n\
python app.py & \n\
sleep 10\n\
streamlit run frontend/streamlit_app.py --server.port=8501 --server.address=0.0.0.0\n\
wait\n\
' > /app/docker_start.sh && chmod +x /app/docker_start.sh

# Set environment variable to make Streamlit accessible from outside container
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

CMD ["/app/docker_start.sh"] 