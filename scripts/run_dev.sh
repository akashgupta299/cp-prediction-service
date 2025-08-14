#!/bin/bash
# Development server startup script

set -e

echo "Starting CP Prediction Service in development mode..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Set development environment variables
export LOG_LEVEL=DEBUG
export ENABLE_PREDICTION_HISTORY=true
export CACHE_TTL_SECONDS=300
export MAX_MODELS_IN_MEMORY=3

# Run database migrations if needed
echo "Setting up database..."
python -c "from app.database import db_manager; print('Database initialized')"

# Start the development server
echo "Starting development server on http://localhost:8000"
echo "API Documentation available at http://localhost:8000/docs"

cd app && python -m uvicorn main_enhanced:app --reload --host 0.0.0.0 --port 8000
