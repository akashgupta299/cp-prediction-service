-- MySQL Maintenance Queries for CP Prediction Service
-- Run these queries periodically to maintain database performance

-- 1. Database size and table statistics
SELECT 
    table_name,
    table_rows,
    ROUND(((data_length + index_length) / 1024 / 1024), 2) AS table_size_mb,
    ROUND((data_length / 1024 / 1024), 2) AS data_size_mb,
    ROUND((index_length / 1024 / 1024), 2) AS index_size_mb
FROM information_schema.TABLES 
WHERE table_schema = 'cp_predictions'
ORDER BY table_size_mb DESC;

-- 2. Check index usage and performance
SELECT 
    table_name,
    index_name,
    column_name,
    cardinality,
    seq_in_index
FROM information_schema.STATISTICS 
WHERE table_schema = 'cp_predictions'
ORDER BY table_name, seq_in_index;

-- 3. Analyze table performance
ANALYZE TABLE prediction_history;
ANALYZE TABLE service_metrics;
ANALYZE TABLE model_performance;

-- 4. Check for slow queries or missing indexes
-- Enable slow query log first: SET GLOBAL slow_query_log = 'ON';
-- SET GLOBAL long_query_time = 1;

-- 5. Optimize tables (run during maintenance window)
OPTIMIZE TABLE prediction_history;
OPTIMIZE TABLE service_metrics;
OPTIMIZE TABLE model_performance;

-- 6. Data cleanup queries (run these carefully!)

-- Archive old predictions (older than 6 months) to archive table
CREATE TABLE IF NOT EXISTS prediction_history_archive LIKE prediction_history;

-- Move old data to archive (uncomment when ready)
-- INSERT INTO prediction_history_archive 
-- SELECT * FROM prediction_history 
-- WHERE timestamp < DATE_SUB(NOW(), INTERVAL 6 MONTH);

-- Delete old data after archiving (uncomment when ready)
-- DELETE FROM prediction_history 
-- WHERE timestamp < DATE_SUB(NOW(), INTERVAL 6 MONTH);

-- 7. Clean up service metrics older than 30 days
-- DELETE FROM service_metrics 
-- WHERE timestamp < DATE_SUB(NOW(), INTERVAL 30 DAY);

-- 8. Update table statistics for better query planning
UPDATE mysql.innodb_table_stats SET last_update = NOW() 
WHERE database_name = 'cp_predictions';

UPDATE mysql.innodb_index_stats SET last_update = NOW() 
WHERE database_name = 'cp_predictions';

-- 9. Check for corrupted tables
CHECK TABLE prediction_history;
CHECK TABLE service_metrics;
CHECK TABLE model_performance;

-- 10. Repair tables if needed (only if corruption detected)
-- REPAIR TABLE prediction_history;

-- 11. Monitor database connections and processes
SELECT 
    id,
    user,
    host,
    db,
    command,
    time,
    state,
    info
FROM information_schema.PROCESSLIST 
WHERE db = 'cp_predictions';

-- 12. Check database locks
SELECT 
    r.trx_id waiting_trx_id,
    r.trx_mysql_thread_id waiting_thread,
    r.trx_query waiting_query,
    b.trx_id blocking_trx_id,
    b.trx_mysql_thread_id blocking_thread,
    b.trx_query blocking_query
FROM information_schema.innodb_lock_waits w
INNER JOIN information_schema.innodb_trx b ON b.trx_id = w.blocking_trx_id
INNER JOIN information_schema.innodb_trx r ON r.trx_id = w.requesting_trx_id;

-- 13. Monitor table growth over time
SELECT 
    DATE(timestamp) as date,
    COUNT(*) as daily_records,
    SUM(COUNT(*)) OVER (ORDER BY DATE(timestamp)) as cumulative_records
FROM prediction_history 
WHERE timestamp >= DATE_SUB(NOW(), INTERVAL 30 DAY)
GROUP BY DATE(timestamp)
ORDER BY date;

-- 14. Create partitioning for large tables (advanced)
-- This is useful when prediction_history grows very large
-- ALTER TABLE prediction_history 
-- PARTITION BY RANGE (YEAR(timestamp)) (
--     PARTITION p2023 VALUES LESS THAN (2024),
--     PARTITION p2024 VALUES LESS THAN (2025),
--     PARTITION p2025 VALUES LESS THAN (2026),
--     PARTITION pmax VALUES LESS THAN MAXVALUE
-- );

-- 15. Backup commands (run from command line)
-- Full backup:
-- mysqldump -u root -p cp_predictions > cp_predictions_backup_$(date +%Y%m%d).sql

-- Table-specific backup:
-- mysqldump -u root -p cp_predictions prediction_history > prediction_history_backup_$(date +%Y%m%d).sql

-- Restore:
-- mysql -u root -p cp_predictions < cp_predictions_backup_20231215.sql
