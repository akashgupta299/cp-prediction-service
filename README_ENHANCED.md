# CP Prediction Service - Enhanced Version

A comprehensive, production-ready CP (Courier Partner) prediction service with advanced features including error handling, caching, monitoring, and scalability improvements.

## 🚀 Features

### Core Improvements
- **Comprehensive Error Handling**: Detailed error responses with proper HTTP status codes
- **Input Validation**: Pydantic models with custom validators
- **Caching**: TTL-based caching for improved performance
- **Memory Optimization**: Lazy loading with LRU eviction
- **Health Monitoring**: Health checks, metrics, and model status endpoints
- **Configuration Management**: Environment-based configuration
- **Database Integration**: Prediction history storage
- **A/B Testing**: Support for model versioning and testing

### Additional Features
- **Docker Support**: Complete containerization with Docker Compose
- **Load Balancing**: Nginx reverse proxy with rate limiting
- **Batch Processing**: Efficient batch prediction endpoint
- **Background Tasks**: Async database operations
- **Structured Logging**: Comprehensive logging with different levels
- **API Documentation**: Auto-generated OpenAPI docs

## 📁 Project Structure

```
cp-prediction-service/
├── app/
│   ├── __init__.py
│   ├── main_enhanced.py      # Enhanced main application
│   ├── config.py            # Configuration management
│   ├── models.py            # Pydantic models
│   ├── database.py          # Database operations
│   ├── model_manager.py     # Advanced model management
│   └── dummy_model.py       # Fallback dummy model
├── scripts/
│   ├── run_dev.sh          # Development server script
│   ├── run_prod.sh         # Production server script
│   └── test_api.py         # API testing script
├── models/                  # Model files directory
├── Dockerfile              # Container definition
├── docker-compose.yml      # Multi-service orchestration
├── requirements.txt        # Python dependencies
├── nginx.conf             # Nginx configuration
├── env.example            # Environment variables template
└── README_ENHANCED.md     # This file
```

## 🛠️ Installation & Setup

### Option 1: Docker (Recommended)

1. **Clone and setup**:
   ```bash
   git clone <repository>
   cd cp-prediction-service
   cp env.example .env  # Edit as needed
   ```

2. **Start with Docker Compose**:
   ```bash
   docker-compose up -d
   ```

3. **Access the service**:
   - API: http://localhost:8000
   - Documentation: http://localhost:8000/docs
   - Health Check: http://localhost:8000/health

### Option 2: Local Development

1. **Setup environment**:
   ```bash
   chmod +x scripts/run_dev.sh
   ./scripts/run_dev.sh
   ```

2. **Or manual setup**:
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   cd app && python -m uvicorn main_enhanced:app --reload
   ```

## 🔧 Configuration

### Environment Variables

Create a `.env` file from `env.example`:

```bash
# Server Configuration
HOST=0.0.0.0
PORT=8000
WORKERS=4

# Model Configuration
MODEL_DIR=models
MAX_MODELS_IN_MEMORY=5
MODEL_VERSION=v1

# Cache Configuration
CACHE_TTL_SECONDS=3600
CACHE_MAX_SIZE=10000

# Database Configuration
DATABASE_URL=postgresql://user:pass@localhost:5432/cp_predictions
ENABLE_PREDICTION_HISTORY=true

# A/B Testing
ENABLE_AB_TESTING=true
MODEL_A_WEIGHT=0.7
MODEL_B_WEIGHT=0.3
```

## 📊 API Endpoints

### Core Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/predict` | Single prediction |
| POST | `/predict/batch` | Batch predictions |
| GET | `/health` | Health check |
| GET | `/metrics` | Service metrics |
| GET | `/models/status` | Model status |

### Management Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/cache/clear` | Clear prediction cache |
| GET | `/history/{shipment_id}` | Get prediction history |
| GET | `/statistics` | Service statistics |

### Example Usage

**Single Prediction**:
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "address": "123 Main Street, Mumbai 280001",
    "shipment_id": "SHIP123"
  }'
```

**Batch Prediction**:
```bash
curl -X POST "http://localhost:8000/predict/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "requests": [
      {"address": "123 Main St, Mumbai 280001", "shipment_id": "SHIP001"},
      {"address": "456 Oak Ave, Delhi 280002", "shipment_id": "SHIP002"}
    ]
  }'
```

## 🧪 Testing

### Automated Testing
```bash
python scripts/test_api.py
```

### Manual Testing
1. **Health Check**: `curl http://localhost:8000/health`
2. **Metrics**: `curl http://localhost:8000/metrics`
3. **API Docs**: Visit http://localhost:8000/docs

## 📈 Monitoring & Observability

### Health Checks
- **Endpoint**: `/health`
- **Docker Health Check**: Built-in container health monitoring
- **Response**: Service status, uptime, memory usage, models loaded

### Metrics
- **Endpoint**: `/metrics`
- **Includes**: Request counts, response times, cache statistics, error rates
- **Format**: JSON (can be extended to Prometheus format)

### Logging
- **Structured Logging**: JSON format with timestamps and levels
- **Log Levels**: DEBUG, INFO, WARNING, ERROR
- **Context**: Request IDs, shipment IDs, performance metrics

## 🔄 Deployment

### Production Deployment

1. **Using Docker Compose**:
   ```bash
   docker-compose -f docker-compose.yml up -d
   ```

2. **Kubernetes** (example):
   ```yaml
   apiVersion: apps/v1
   kind: Deployment
   metadata:
     name: cp-prediction-service
   spec:
     replicas: 3
     selector:
       matchLabels:
         app: cp-prediction-service
     template:
       spec:
         containers:
         - name: app
           image: cp-prediction-service:latest
           ports:
           - containerPort: 8000
   ```

### Scaling Considerations

1. **Horizontal Scaling**: Multiple service instances behind load balancer
2. **Vertical Scaling**: Increase memory/CPU for model loading
3. **Database Scaling**: Use PostgreSQL with read replicas
4. **Caching**: Redis cluster for distributed caching

## 🔒 Security

### Implemented Security Features
- **Input Validation**: Comprehensive request validation
- **Error Sanitization**: Safe error messages
- **CORS Configuration**: Configurable allowed origins
- **Rate Limiting**: Nginx-based rate limiting
- **Security Headers**: Standard security headers

### Additional Security Recommendations
- **API Keys**: Implement API key authentication
- **HTTPS**: Use TLS certificates
- **Network Security**: VPC/firewall configuration
- **Secrets Management**: Use environment-specific secrets

## 🚨 Troubleshooting

### Common Issues

1. **Models not loading**:
   - Check model directory path
   - Verify file permissions
   - Check logs for loading errors

2. **High memory usage**:
   - Reduce `MAX_MODELS_IN_MEMORY`
   - Monitor model eviction logs
   - Check cache size settings

3. **Slow predictions**:
   - Enable caching
   - Check model loading times
   - Monitor database performance

4. **Database connection issues**:
   - Verify DATABASE_URL
   - Check database service status
   - Review connection pool settings

### Performance Tuning

1. **Cache Optimization**:
   ```bash
   # Increase cache size for high-traffic
   CACHE_MAX_SIZE=50000
   CACHE_TTL_SECONDS=7200
   ```

2. **Model Management**:
   ```bash
   # Keep more models in memory
   MAX_MODELS_IN_MEMORY=10
   ```

3. **Worker Configuration**:
   ```bash
   # Increase workers for CPU-bound tasks
   WORKERS=8
   ```

## 📋 Maintenance

### Regular Tasks
- **Log Rotation**: Configure log rotation for disk space
- **Database Cleanup**: Archive old prediction history
- **Cache Monitoring**: Monitor cache hit rates
- **Model Updates**: Deploy new model versions

### Monitoring Alerts
- **High Error Rate**: > 5% error rate
- **High Response Time**: > 1000ms average
- **Low Cache Hit Rate**: < 70% hit rate
- **High Memory Usage**: > 80% memory utilization

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License.
