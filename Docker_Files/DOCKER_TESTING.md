# Testing TopicMind with Docker

This guide explains how to use Docker to test TopicMind across different operating systems and environments.

## Prerequisites

- Docker installed on your system
- Docker Compose installed on your system

## Quick Start

The easiest way to test TopicMind in Docker is to use the provided script:

```bash
./docker_test.sh
```

This script will:
1. Build a Docker image for TopicMind
2. Start the container
3. Show information about the container environment
4. Check if the API is healthy
5. Provide URLs to access the application

## Manual Testing

If you prefer to manage Docker manually, you can use the following commands:

### Building the Docker image

```bash
docker-compose build
```

### Starting the services

```bash
docker-compose up -d
```

### Checking the logs

```bash
docker-compose logs -f
```

### Stopping the services

```bash
docker-compose down
```

## Accessing TopicMind

Once the container is running:

- Web UI: [http://localhost:8501](http://localhost:8501)
- API: [http://localhost:5001](http://localhost:5001)
- Health Check: [http://localhost:5001/health](http://localhost:5001/health)

## Using Your OpenAI API Key

There are two ways to provide your OpenAI API key:

1. Create a `.env` file with your API key and uncomment the volume mount in `docker-compose.yml`:
   ```yaml
   volumes:
     - ./logs:/app/logs
     - ./.env:/app/.env  # Uncomment this line
   ```

2. Add the API key directly in the `docker-compose.yml` file:
   ```yaml
   environment:
     - OPENAI_API_KEY=your_key_here  # Replace with your actual API key
     - PYTHONUNBUFFERED=1
   ```

## Testing on Different Platforms

You can run the same Docker setup on various platforms:

- Windows (with Docker Desktop)
- macOS (with Docker Desktop)
- Linux (native Docker)
- Cloud environments (AWS, GCP, Azure)

The Docker container uses a consistent Linux environment regardless of the host OS, making it ideal for cross-platform testing.

## Troubleshooting

If you encounter issues:

1. Check the logs: `docker-compose logs`
2. Ensure ports 5001 and 8501 are not in use by other applications
3. Verify Docker has sufficient resources (memory, CPU)
4. Check if any required model files are failing to download

For permissions issues with logs, run: `chmod -R 777 logs/` to ensure the container can write to the logs directory. 