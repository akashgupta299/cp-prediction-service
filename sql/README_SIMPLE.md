# Simple MySQL Setup - Predictions Table Only

This directory contains minimal SQL scripts for setting up just the predictions table.

## Files

- `minimal_mysql.sql` - Creates database, user, and predictions table only
- `basic_queries.sql` - Basic queries for the predictions table
- `setup_minimal_mysql.sh` - Automated setup script

## Quick Setup

### Option 1: Automated Script
```bash
./scripts/setup_minimal_mysql.sh
```

### Option 2: Manual Setup
```bash
# Connect to MySQL
mysql -u root -p

# Run the minimal script
source sql/minimal_mysql.sql;
```

## Table Structure

### predictions
| Column | Type | Description |
|--------|------|-------------|
| id | BIGINT (PK) | Auto-incrementing ID |
| shipment_id | VARCHAR(100) | Shipment identifier |
| address | TEXT | Full address |
| pincode | VARCHAR(6) | 6-digit pincode |
| predicted_cp | VARCHAR(100) | Predicted CP location |
| confidence | DECIMAL(5,4) | Confidence score (0-1) |
| model_version | VARCHAR(50) | Model version used |
| processing_time_ms | DECIMAL(8,3) | Processing time |
| cached | BOOLEAN | Whether cached |
| timestamp | DATETIME | When prediction was made |

## Basic Queries

```sql
-- Get recent predictions
SELECT * FROM predictions ORDER BY timestamp DESC LIMIT 10;

-- Get prediction by shipment ID
SELECT * FROM predictions WHERE shipment_id = 'SHIP123';

-- Count total predictions
SELECT COUNT(*) FROM predictions;

-- Get average confidence
SELECT AVG(confidence) FROM predictions;
```

## Database Connection

Update your `.env` file:
```env
DATABASE_URL=mysql+pymysql://cp_service:your_password@localhost:3306/cp_predictions
```

That's it! Just one table, no views, no extra tables - exactly what you requested.
