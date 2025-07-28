# Document Analyzer - Docker Setup

This project includes Docker configuration for easy deployment and consistent environments.

## Files Created

- `Dockerfile` - Main Docker image configuration
- `.dockerignore` - Excludes unnecessary files from Docker context
- `docker-compose.yml` - Docker Compose configuration for easier management

## Building and Running

### Option 1: Using Docker directly

```bash
# Build the image
docker build -t document-analyzer .

# Run the container
docker run -d \
  --name document-analyzer \
  -v $(pwd)/documents:/app/documents:ro \
  -v $(pwd)/output:/app/output \
  -v $(pwd)/analysis_results:/app/analysis_results \
  -p 8000:8000 \
  document-analyzer

# View logs
docker logs document-analyzer

# Stop the container
docker stop document-analyzer
```

### Option 2: Using Docker Compose (Recommended)

```bash
# Start the service
docker-compose up -d

# View logs
docker-compose logs -f

# Stop the service
docker-compose down

# Rebuild and restart
docker-compose up --build -d
```

## Configuration Options

### Environment Variables

- `LOG_LEVEL` - Set logging level (DEBUG, INFO, WARNING, ERROR)
- `PYTHONPATH` - Python path (set to /app by default)

### Volume Mounts

- `/app/documents` - Input PDF documents (mounted read-only)
- `/app/output` - Generated output files
- `/app/analysis_results` - Analysis result files
- `/app/chroma_db` - Persistent vector database

### Port Configuration

- Port 8000 is exposed for FastAPI server (if enabled)

## Different Run Modes

### 1. Document Processing (Default)
```bash
docker run document-analyzer python document_analyzer.py input.json output.json
```

### 2. FastAPI Server
```bash
docker run -p 8000:8000 document-analyzer uvicorn main:app --host 0.0.0.0 --port 8000
```

### 3. Interactive Development
```bash
docker run -it document-analyzer /bin/bash
```

## Resource Requirements

- **Memory**: 2GB limit, 512MB reserved
- **CPU**: 1.0 core limit, 0.5 core reserved
- **Storage**: Depends on document collection size

## Security Features

- Runs as non-root user (`appuser`)
- Read-only document access
- Isolated container environment
- Health checks included

## Troubleshooting

### Common Issues

1. **Permission errors**: Ensure your local directories have proper permissions
2. **Memory issues**: Increase memory limits in docker-compose.yml
3. **Port conflicts**: Change port mapping if 8000 is already in use

### Debug Commands

```bash
# Check container status
docker ps

# Access container shell
docker exec -it document-analyzer /bin/bash

# View detailed logs
docker logs --tail 100 -f document-analyzer

# Check resource usage
docker stats document-analyzer
```

## Development Workflow

1. Make code changes locally
2. Rebuild the container: `docker-compose up --build`
3. Test the changes
4. Commit and deploy

For production deployment, consider using a Docker registry and orchestration platform like Kubernetes.
