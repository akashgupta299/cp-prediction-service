#!/bin/bash
# Production server startup script

set -e

echo "Starting CP Prediction Service in production mode..."

# Load environment variables
if [ -f ".env" ]; then
    export $(cat .env | grep -v ^# | xargs)
fi

# Set production defaults
export LOG_LEVEL=${LOG_LEVEL:-INFO}
export WORKERS=${WORKERS:-4}
export HOST=${HOST:-0.0.0.0}
export PORT=${PORT:-8000}

# Run database migrations
echo "Running database setup..."
python -c "from app.database import db_manager; print('Database initialized')"

# Start the production server with multiple workers
echo "Starting production server with $WORKERS workers on $HOST:$PORT"

cd app && python -m uvicorn main_enhanced:app \
    --host $HOST \
    --port $PORT \
    --workers $WORKERS \
    --access-log \
    --loop uvloop \
    --http httptools
