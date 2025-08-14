#!/bin/bash
# Minimal MySQL Setup for CP Prediction Service - Only Predictions Table

set -e

echo "🗄️  Setting up minimal MySQL database (predictions table only)"
echo "============================================================="

# Check MySQL
if ! command -v mysql &> /dev/null; then
    echo "❌ MySQL not found. Please install MySQL first."
    exit 1
fi

# Get MySQL root password
echo "📝 Enter MySQL root password:"
read -s MYSQL_ROOT_PASSWORD

# Get application password
echo "📝 Enter password for cp_service user (or press Enter for default):"
read -s CP_SERVICE_PASSWORD
if [ -z "$CP_SERVICE_PASSWORD" ]; then
    CP_SERVICE_PASSWORD="cp_service_pass"
fi

# Test connection
echo "🔍 Testing MySQL connection..."
mysql -u root -p"$MYSQL_ROOT_PASSWORD" -e "SELECT VERSION();" > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "❌ Failed to connect to MySQL"
    exit 1
fi

# Run setup
echo "📊 Creating database and table..."
mysql -u root -p"$MYSQL_ROOT_PASSWORD" << EOF
-- Create database
CREATE DATABASE IF NOT EXISTS cp_predictions 
CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

USE cp_predictions;

-- Create user
DROP USER IF EXISTS 'cp_service'@'%';
CREATE USER 'cp_service'@'%' IDENTIFIED BY '$CP_SERVICE_PASSWORD';
GRANT SELECT, INSERT, UPDATE, DELETE ON cp_predictions.* TO 'cp_service'@'%';
FLUSH PRIVILEGES;

-- Create predictions table
CREATE TABLE IF NOT EXISTS predictions (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    shipment_id VARCHAR(100) NOT NULL,
    address TEXT NOT NULL,
    pincode VARCHAR(6) NOT NULL,
    predicted_cp VARCHAR(100) NOT NULL,
    confidence DECIMAL(5, 4) NOT NULL,
    model_version VARCHAR(50) DEFAULT NULL,
    processing_time_ms DECIMAL(8, 3) DEFAULT NULL,
    cached BOOLEAN DEFAULT FALSE,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    
    INDEX idx_shipment_id (shipment_id),
    INDEX idx_pincode (pincode),
    INDEX idx_timestamp (timestamp)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

DESCRIBE predictions;
EOF

if [ $? -eq 0 ]; then
    echo "✅ Database setup completed successfully!"
else
    echo "❌ Database setup failed"
    exit 1
fi

# Update .env file
echo "📝 Updating .env file..."
ENV_FILE=".env"
if [ ! -f "$ENV_FILE" ]; then
    cp env.example "$ENV_FILE"
fi

# Update database URL
if grep -q "DATABASE_URL=" "$ENV_FILE"; then
    sed -i.bak "s|DATABASE_URL=.*|DATABASE_URL=mysql+pymysql://cp_service:$CP_SERVICE_PASSWORD@localhost:3306/cp_predictions|" "$ENV_FILE"
else
    echo "DATABASE_URL=mysql+pymysql://cp_service:$CP_SERVICE_PASSWORD@localhost:3306/cp_predictions" >> "$ENV_FILE"
fi

echo "✅ Configuration updated"
echo ""
echo "🎉 Setup Complete!"
echo "Database: cp_predictions"
echo "Table: predictions"
echo "User: cp_service"
echo "Connection: mysql+pymysql://cp_service:$CP_SERVICE_PASSWORD@localhost:3306/cp_predictions"
