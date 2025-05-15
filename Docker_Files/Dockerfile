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

# Copy the startup script
COPY docker_start.sh /app/docker_start.sh
RUN chmod +x /app/docker_start.sh

# Set environment variable to make Streamlit accessible from outside container
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Use our custom startup script
CMD ["/app/docker_start.sh"] 