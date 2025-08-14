# MySQL Database Setup for CP Prediction Service

This directory contains SQL scripts to set up and manage the MySQL database for the CP Prediction Service.

## Files Overview

- `init_mysql.sql` - Initial database setup and table creation
- `sample_queries.sql` - Common queries for data analysis
- `maintenance_queries.sql` - Database maintenance and optimization queries
- `README.md` - This documentation file

## Quick Setup

### 1. Database Initialization

```bash
# Connect to MySQL as root
mysql -u root -p

# Run the initialization script
source sql/init_mysql.sql;
```

### 2. Update Database Connection String

Update your `.env` file with the MySQL connection details:

```env
DATABASE_URL=mysql+pymysql://cp_service:your_password@localhost:3306/cp_predictions
```

### 3. Install MySQL Driver

```bash
pip install pymysql
# or
pip install mysqlclient
```

## Database Schema

### Main Tables

#### prediction_history
Stores all prediction requests and results.

| Column | Type | Description |
|--------|------|-------------|
| id | INT (PK) | Auto-incrementing primary key |
| shipment_id | VARCHAR(100) | Unique shipment identifier |
| address | TEXT | Full address used for prediction |
| pincode | VARCHAR(6) | Extracted 6-digit pincode |
| predicted_cp | VARCHAR(100) | Predicted courier partner location |
| confidence | DECIMAL(10,8) | Prediction confidence score (0-1) |
| model_version | VARCHAR(50) | Model version used |
| processing_time_ms | DECIMAL(10,3) | Processing time in milliseconds |
| cached | BOOLEAN | Whether result was from cache |
| timestamp | DATETIME | When prediction was made |

#### service_metrics
Stores service-level metrics for monitoring.

| Column | Type | Description |
|--------|------|-------------|
| id | INT (PK) | Auto-incrementing primary key |
| metric_name | VARCHAR(100) | Name of the metric |
| metric_value | DECIMAL(15,6) | Metric value |
| metric_type | ENUM | Type: counter, gauge, histogram |
| timestamp | DATETIME | When metric was recorded |

#### model_performance
Aggregated model performance data.

| Column | Type | Description |
|--------|------|-------------|
| id | INT (PK) | Auto-incrementing primary key |
| model_version | VARCHAR(50) | Model version |
| pincode_prefix | VARCHAR(2) | 2-digit pincode prefix |
| total_predictions | INT | Total predictions made |
| avg_confidence | DECIMAL(10,8) | Average confidence |
| avg_processing_time_ms | DECIMAL(10,3) | Average processing time |
| last_updated | DATETIME | Last update timestamp |

### Views

#### recent_predictions
Shows predictions from the last 24 hours.

#### prediction_stats
Daily aggregated prediction statistics.

## Common Operations

### 1. Check Recent Predictions
```sql
SELECT * FROM recent_predictions LIMIT 10;
```

### 2. Get Prediction Statistics
```sql
SELECT * FROM prediction_stats WHERE prediction_date >= DATE_SUB(NOW(), INTERVAL 7 DAY);
```

### 3. Find Low Confidence Predictions
```sql
SELECT shipment_id, address, predicted_cp, confidence 
FROM prediction_history 
WHERE confidence < 0.5 
ORDER BY timestamp DESC LIMIT 50;
```

### 4. Monitor Cache Performance
```sql
SELECT 
    DATE(timestamp) as date,
    COUNT(*) as total,
    SUM(CASE WHEN cached = TRUE THEN 1 ELSE 0 END) as cache_hits,
    ROUND(SUM(CASE WHEN cached = TRUE THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) as hit_rate
FROM prediction_history 
WHERE timestamp >= DATE_SUB(NOW(), INTERVAL 7 DAY)
GROUP BY DATE(timestamp);
```

## Performance Optimization

### Indexes
The following indexes are automatically created:
- `idx_shipment_id` - For shipment-based queries
- `idx_pincode` - For pincode-based analysis
- `idx_timestamp` - For time-based queries
- `idx_model_version` - For model performance analysis
- `idx_cached` - For cache analysis

### Maintenance Tasks

Run these periodically:

1. **Analyze Tables** (weekly):
   ```sql
   ANALYZE TABLE prediction_history;
   ```

2. **Optimize Tables** (monthly):
   ```sql
   OPTIMIZE TABLE prediction_history;
   ```

3. **Archive Old Data** (quarterly):
   ```sql
   -- Move data older than 6 months to archive table
   INSERT INTO prediction_history_archive 
   SELECT * FROM prediction_history 
   WHERE timestamp < DATE_SUB(NOW(), INTERVAL 6 MONTH);
   
   DELETE FROM prediction_history 
   WHERE timestamp < DATE_SUB(NOW(), INTERVAL 6 MONTH);
   ```

## Backup and Recovery

### Backup Commands
```bash
# Full database backup
mysqldump -u root -p cp_predictions > backup_$(date +%Y%m%d).sql

# Compressed backup
mysqldump -u root -p cp_predictions | gzip > backup_$(date +%Y%m%d).sql.gz

# Table-specific backup
mysqldump -u root -p cp_predictions prediction_history > predictions_$(date +%Y%m%d).sql
```

### Restore Commands
```bash
# Restore full database
mysql -u root -p cp_predictions < backup_20231215.sql

# Restore from compressed backup
gunzip < backup_20231215.sql.gz | mysql -u root -p cp_predictions
```

## Monitoring Queries

### Database Size
```sql
SELECT 
    ROUND(SUM(data_length + index_length) / 1024 / 1024, 2) AS 'DB Size in MB'
FROM information_schema.tables 
WHERE table_schema = 'cp_predictions';
```

### Table Growth
```sql
SELECT 
    table_name,
    table_rows,
    ROUND(((data_length + index_length) / 1024 / 1024), 2) AS 'Size in MB'
FROM information_schema.TABLES 
WHERE table_schema = 'cp_predictions'
ORDER BY (data_length + index_length) DESC;
```

### Connection Status
```sql
SHOW PROCESSLIST;
```

## Security Considerations

1. **User Permissions**: The `cp_service` user has only necessary permissions (SELECT, INSERT, UPDATE, DELETE).

2. **Connection Security**: Use SSL connections in production:
   ```env
   DATABASE_URL=mysql+pymysql://cp_service:password@localhost:3306/cp_predictions?ssl_ca=/path/to/ca.pem
   ```

3. **Password Security**: Use strong passwords and consider using environment variables.

4. **Network Security**: Restrict database access to application servers only.

## Troubleshooting

### Common Issues

1. **Connection Refused**:
   - Check if MySQL service is running
   - Verify connection credentials
   - Check firewall settings

2. **Slow Queries**:
   - Check `sample_queries.sql` for optimized queries
   - Run `EXPLAIN` on slow queries
   - Consider adding indexes

3. **Disk Space**:
   - Monitor table sizes with maintenance queries
   - Archive old data regularly
   - Use table compression if needed

4. **Lock Timeouts**:
   - Check for long-running transactions
   - Use maintenance queries to identify locks
   - Consider connection pooling

## Integration with Application

Update your `app/database.py` to use MySQL:

```python
# Update the database URL in config.py
DATABASE_URL=mysql+pymysql://cp_service:your_password@localhost:3306/cp_predictions

# Install required driver
pip install pymysql
```

The application will automatically create tables if they don't exist, but running the initialization script first is recommended for proper indexing and optimization.
