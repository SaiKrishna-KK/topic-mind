# Cross-Platform Deployment Guide for TopicMind

This guide provides instructions for deploying TopicMind across different operating systems and environments.

## Deployment Options

### 1. Docker (Recommended)

Docker provides the most consistent experience across platforms.

**Requirements:**
- Docker and Docker Compose installed
- 8GB+ of RAM allocated to Docker

**Steps:**

1. **Build and start the container:**
   ```bash
   docker-compose up -d
   ```

2. **Check logs:**
   ```bash
   docker-compose logs -f
   ```

3. **Stop the container:**
   ```bash
   docker-compose down
   ```

### 2. Native Installation

#### Linux/macOS

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Create environment file:**
   ```bash
   cp .env.example .env
   # Edit .env with your settings
   ```

3. **Start the application:**
   ```bash
   ./run_topicmind.sh
   ```

#### Windows

1. **Install dependencies:**
   ```cmd
   pip install -r requirements.txt
   ```

2. **Create environment file:**
   ```cmd
   copy .env.example .env
   # Edit .env with your settings
   ```

3. **Start the application:**
   ```cmd
   start_topicmind.bat
   ```

## Common Issues & Solutions

### Missing .env File

If the .env file is missing, the application will:
1. Look for .env.example and copy it (if available)
2. Log a warning about the missing API key
3. Continue with limited functionality

### Model Loading Errors

Symptoms:
- "empty mode stack" errors
- "Failed to load model" messages

Solutions:
- Ensure sufficient disk space (5GB+ free)
- Set `USE_SMALL_MODELS=true` in .env
- Increase API timeout: `API_TIMEOUT=300`
- Force CPU mode: `MODEL_DEVICE=cpu`

### Python 3.9 Compatibility

Issues:
- Type annotation errors with Union operator (`|`)

Solutions:
- Replace `type1 | type2` with `Union[type1, type2]`
- Import Union: `from typing import Union`

### OpenAI Client Errors

Issues:
- `proxies` parameter incompatibility

Solution:
- Use simplified client initialization:
  ```python
  client = OpenAI(api_key=api_key)
  ```

### Docker Disk Space Issues

When Docker runs out of disk space:

1. **Clean unused images:**
   ```bash
   docker system prune -a
   ```

2. **Increase Docker storage:**
   - Docker Desktop: Settings → Resources → Disk Image Size

## Testing Across Platforms

Use the automated test script:

```bash
./run_docker_tests.sh
```

Or for Windows:
```cmd
run_docker_tests.bat
```

## CI/CD Integration

A GitHub Actions workflow is included in `.github/workflows/docker-tests.yml` that tests:
- Ubuntu Linux
- macOS
- Windows

This ensures cross-platform compatibility with each code change. 