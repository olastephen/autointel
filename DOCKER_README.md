# 🐳 Docker Deployment Guide

## Car Analysis Intelligence Platform - Containerized Deployment

This guide explains how to deploy the Car Analysis Intelligence Platform using Docker for production hosting.

## 🚀 Quick Start

### Option 1: Docker Compose (Recommended)
```bash
# Clone the repository
git clone https://github.com/olastephen/autointel.git
cd autointel

# Start the entire stack
docker-compose up -d

# View logs
docker-compose logs -f car-analysis-dashboard
```

### Option 2: Docker Build & Run
```bash
# Build the image
docker build -t autointel-dashboard .

# Run the container
docker run -d \
  --name autointel-dashboard \
  -p 8501:8501 \
  -v $(pwd)/datasets:/app/datasets:ro \
  -v $(pwd)/logs:/app/logs \
  autointel-dashboard
```

## 📊 What Happens on Startup

1. **🔍 Dataset Validation**: Checks for required CSV files
2. **🔬 Analysis Execution**: Runs the complete analysis framework
3. **📈 Data Processing**: Processes sentiment, NER, keywords, n-grams, topics
4. **💾 Database Storage**: Stores results for dashboard access
5. **🌐 Dashboard Launch**: Starts Streamlit on port 8501

## 🔧 Configuration

### Environment Variables
```bash
# Streamlit Configuration
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0
PYTHONPATH=/app
PYTHONUNBUFFERED=1

# Optional: Database Configuration
DATABASE_URL=postgresql://user:pass@host:port/db
```

### Volume Mounts
```yaml
volumes:
  - ./datasets:/app/datasets:ro    # Read-only dataset access
  - ./logs:/app/logs              # Log persistence
```

## 🏭 Production Deployment

### Using Docker Compose with PostgreSQL
```bash
# Start with database
docker-compose up -d

# Scale dashboard (optional)
docker-compose up -d --scale car-analysis-dashboard=2
```

### Cloud Deployment Options

#### 1. **AWS ECS/Fargate**
```bash
# Push to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 123456789.dkr.ecr.us-east-1.amazonaws.com
docker tag autointel-dashboard:latest 123456789.dkr.ecr.us-east-1.amazonaws.com/autointel:latest
docker push 123456789.dkr.ecr.us-east-1.amazonaws.com/autointel:latest
```

#### 2. **Google Cloud Run**
```bash
# Build and deploy
gcloud builds submit --tag gcr.io/PROJECT-ID/autointel
gcloud run deploy --image gcr.io/PROJECT-ID/autointel --platform managed
```

#### 3. **Azure Container Instances**
```bash
# Create container group
az container create \
  --resource-group myResourceGroup \
  --name autointel-dashboard \
  --image autointel-dashboard:latest \
  --ports 8501
```

## 📈 Performance Optimization

### Resource Requirements
```yaml
# Minimum requirements
deploy:
  resources:
    limits:
      memory: 2G
      cpus: '1.0'
    reservations:
      memory: 1G
      cpus: '0.5'
```

### Multi-stage Build (Advanced)
```dockerfile
# Multi-stage build for smaller images
FROM python:3.11-slim as builder
WORKDIR /app
COPY requirements.txt .
RUN pip wheel --no-deps --wheel-dir wheels -r requirements.txt

FROM python:3.11-slim
COPY --from=builder /app/wheels /wheels
RUN pip install --no-index --find-links /wheels -r requirements.txt
```

## 🔍 Monitoring & Logs

### Health Checks
```bash
# Check container health
docker ps
docker logs autointel-dashboard

# Manual health check
curl http://localhost:8501/_stcore/health
```

### Log Management
```bash
# View real-time logs
docker-compose logs -f car-analysis-dashboard

# Export logs
docker logs autointel-dashboard > dashboard.log 2>&1
```

## 🛠️ Troubleshooting

### Common Issues

1. **Analysis Framework Fails**
   ```bash
   # Check dataset permissions
   ls -la datasets/
   
   # Verify Python path
   docker exec -it autointel-dashboard python -c "import sys; print(sys.path)"
   ```

2. **Memory Issues**
   ```bash
   # Increase memory limit
   docker run --memory=4g autointel-dashboard
   ```

3. **Port Conflicts**
   ```bash
   # Use different port
   docker run -p 8502:8501 autointel-dashboard
   ```

### Debug Mode
```bash
# Run with debug output
docker run -it --rm \
  -e PYTHONUNBUFFERED=1 \
  -e DEBUG=1 \
  autointel-dashboard
```

## 🔐 Security Considerations

### Production Hardening
```dockerfile
# Run as non-root user
RUN adduser --disabled-password --gecos '' appuser
USER appuser

# Remove unnecessary packages
RUN apt-get remove -y gcc g++ && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*
```

### Environment Secrets
```bash
# Use Docker secrets
echo "secret_password" | docker secret create db_password -
```

## 📋 Maintenance

### Updates
```bash
# Update image
docker-compose pull
docker-compose up -d

# Backup data
docker-compose exec postgres pg_dump -U autointel car_analysis > backup.sql
```

### Cleanup
```bash
# Remove old containers
docker system prune -a

# Clean volumes
docker volume prune
```

## 🌐 Reverse Proxy Setup

### Nginx Configuration
```nginx
server {
    listen 80;
    server_name autointel.yourdomain.com;
    
    location / {
        proxy_pass http://localhost:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
    }
}
```

## 📞 Support

For deployment issues or questions:
- 📧 GitHub Issues: [https://github.com/olastephen/autointel/issues](https://github.com/olastephen/autointel/issues)
- 📖 Documentation: Check README.md for application details
- 🐳 Docker Hub: [Coming Soon]

---

**🎯 Ready to deploy your Car Analysis Intelligence Platform!**
