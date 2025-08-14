#!/bin/bash
# MySQL Database Setup Script for CP Prediction Service

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}🗄️  Setting up MySQL Database for CP Prediction Service${NC}"
echo "================================================="

# Check if MySQL is installed
if ! command -v mysql &> /dev/null; then
    echo -e "${RED}❌ MySQL is not installed. Please install MySQL first.${NC}"
    echo "Ubuntu/Debian: sudo apt-get install mysql-server"
    echo "CentOS/RHEL: sudo yum install mysql-server"
    echo "macOS: brew install mysql"
    exit 1
fi

# Check if MySQL is running
if ! pgrep mysql > /dev/null; then
    echo -e "${YELLOW}⚠️  MySQL service is not running. Starting MySQL...${NC}"
    # Try different service managers
    if command -v systemctl &> /dev/null; then
        sudo systemctl start mysql
    elif command -v service &> /dev/null; then
        sudo service mysql start
    elif command -v brew &> /dev/null; then
        brew services start mysql
    else
        echo -e "${RED}❌ Could not start MySQL service. Please start it manually.${NC}"
        exit 1
    fi
fi

# Prompt for MySQL root password
echo -e "${YELLOW}📝 Please enter MySQL root password:${NC}"
read -s MYSQL_ROOT_PASSWORD

# Prompt for application database password
echo -e "${YELLOW}📝 Enter password for cp_service database user (or press Enter for default):${NC}"
read -s CP_SERVICE_PASSWORD
if [ -z "$CP_SERVICE_PASSWORD" ]; then
    CP_SERVICE_PASSWORD="cp_service_password_123"
    echo -e "${YELLOW}ℹ️  Using default password: $CP_SERVICE_PASSWORD${NC}"
fi

# Test MySQL connection
echo -e "${GREEN}🔍 Testing MySQL connection...${NC}"
mysql -u root -p"$MYSQL_ROOT_PASSWORD" -e "SELECT VERSION();" > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Failed to connect to MySQL. Please check your root password.${NC}"
    exit 1
fi

echo -e "${GREEN}✅ MySQL connection successful${NC}"

# Create database and user
echo -e "${GREEN}📊 Creating database and user...${NC}"
mysql -u root -p"$MYSQL_ROOT_PASSWORD" << EOF
-- Create database
CREATE DATABASE IF NOT EXISTS cp_predictions 
CHARACTER SET utf8mb4 
COLLATE utf8mb4_unicode_ci;

-- Create user
DROP USER IF EXISTS 'cp_service'@'%';
CREATE USER 'cp_service'@'%' IDENTIFIED BY '$CP_SERVICE_PASSWORD';
GRANT SELECT, INSERT, UPDATE, DELETE ON cp_predictions.* TO 'cp_service'@'%';
FLUSH PRIVILEGES;

-- Show created database
SHOW DATABASES LIKE 'cp_predictions';
EOF

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Database and user created successfully${NC}"
else
    echo -e "${RED}❌ Failed to create database and user${NC}"
    exit 1
fi

# Run initialization script
echo -e "${GREEN}🚀 Running database initialization script...${NC}"
mysql -u root -p"$MYSQL_ROOT_PASSWORD" cp_predictions < sql/init_mysql.sql

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Database tables created successfully${NC}"
else
    echo -e "${RED}❌ Failed to create database tables${NC}"
    exit 1
fi

# Verify tables were created
echo -e "${GREEN}🔍 Verifying table creation...${NC}"
TABLE_COUNT=$(mysql -u root -p"$MYSQL_ROOT_PASSWORD" cp_predictions -e "SHOW TABLES;" | wc -l)
if [ $TABLE_COUNT -gt 1 ]; then
    echo -e "${GREEN}✅ Tables created successfully${NC}"
    mysql -u root -p"$MYSQL_ROOT_PASSWORD" cp_predictions -e "SHOW TABLES;"
else
    echo -e "${RED}❌ No tables found${NC}"
    exit 1
fi

# Create .env file with database configuration
echo -e "${GREEN}📝 Creating database configuration...${NC}"
ENV_FILE=".env"

# Backup existing .env if it exists
if [ -f "$ENV_FILE" ]; then
    cp "$ENV_FILE" "$ENV_FILE.backup.$(date +%Y%m%d_%H%M%S)"
    echo -e "${YELLOW}ℹ️  Existing .env backed up${NC}"
fi

# Create or update .env file
if [ ! -f "$ENV_FILE" ]; then
    cp env.example "$ENV_FILE"
fi

# Update database URL in .env file
if grep -q "DATABASE_URL=" "$ENV_FILE"; then
    sed -i.bak "s|DATABASE_URL=.*|DATABASE_URL=mysql+pymysql://cp_service:$CP_SERVICE_PASSWORD@localhost:3306/cp_predictions|" "$ENV_FILE"
else
    echo "DATABASE_URL=mysql+pymysql://cp_service:$CP_SERVICE_PASSWORD@localhost:3306/cp_predictions" >> "$ENV_FILE"
fi

echo -e "${GREEN}✅ Database configuration updated in .env file${NC}"

# Install Python MySQL dependencies
echo -e "${GREEN}📦 Installing Python MySQL dependencies...${NC}"
pip install pymysql mysqlclient

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Python dependencies installed${NC}"
else
    echo -e "${YELLOW}⚠️  Some Python dependencies may have failed to install${NC}"
    echo -e "${YELLOW}ℹ️  You may need to install system dependencies first:${NC}"
    echo "Ubuntu/Debian: sudo apt-get install default-libmysqlclient-dev build-essential"
    echo "CentOS/RHEL: sudo yum install mysql-devel gcc"
    echo "macOS: brew install mysql-client"
fi

# Test application database connection
echo -e "${GREEN}🧪 Testing application database connection...${NC}"
python3 -c "
import pymysql
try:
    connection = pymysql.connect(
        host='localhost',
        user='cp_service',
        password='$CP_SERVICE_PASSWORD',
        database='cp_predictions'
    )
    cursor = connection.cursor()
    cursor.execute('SELECT COUNT(*) FROM prediction_history')
    result = cursor.fetchone()
    print(f'✅ Connection successful. prediction_history table has {result[0]} records.')
    connection.close()
except Exception as e:
    print(f'❌ Connection failed: {e}')
    exit(1)
"

echo ""
echo -e "${GREEN}🎉 MySQL Database Setup Complete!${NC}"
echo "================================================="
echo -e "${GREEN}📊 Database Details:${NC}"
echo "  • Database Name: cp_predictions"
echo "  • Username: cp_service"
echo "  • Password: $CP_SERVICE_PASSWORD"
echo "  • Host: localhost"
echo "  • Port: 3306"
echo ""
echo -e "${GREEN}🔗 Connection String:${NC}"
echo "  DATABASE_URL=mysql+pymysql://cp_service:$CP_SERVICE_PASSWORD@localhost:3306/cp_predictions"
echo ""
echo -e "${GREEN}📝 Next Steps:${NC}"
echo "  1. Start your FastAPI application"
echo "  2. Check the /health endpoint"
echo "  3. Make some predictions to populate the database"
echo "  4. Use the queries in sql/sample_queries.sql to analyze data"
echo ""
echo -e "${YELLOW}⚠️  Security Note:${NC}"
echo "  • Change the default password in production"
echo "  • Use SSL connections for production deployments"
echo "  • Regularly backup your database"
echo ""
