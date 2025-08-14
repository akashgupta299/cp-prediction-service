-- Simple MySQL Queries for CP Prediction Service (Single Table)
-- All queries work with the 'predictions' table

-- ==========================================
-- BASIC QUERIES
-- ==========================================

-- 1. Get all recent predictions (last 24 hours)
SELECT * FROM recent_predictions LIMIT 20;

-- 2. Get prediction by shipment ID
SELECT * FROM predictions 
WHERE shipment_id = 'SHIP123456'
ORDER BY timestamp DESC;

-- 3. Get predictions by pincode
SELECT 
    shipment_id,
    predicted_cp,
    confidence,
    timestamp
FROM predictions 
WHERE pincode = '280001'
ORDER BY timestamp DESC
LIMIT 20;

-- 4. Get predictions by pincode prefix (first 2 digits)
SELECT 
    shipment_id,
    pincode,
    predicted_cp,
    confidence,
    timestamp
FROM predictions 
WHERE pincode LIKE '28%'
ORDER BY timestamp DESC
LIMIT 50;

-- ==========================================
-- STATISTICS QUERIES
-- ==========================================

-- 5. Daily prediction statistics
SELECT * FROM daily_stats 
WHERE prediction_date >= DATE_SUB(NOW(), INTERVAL 30 DAY)
ORDER BY prediction_date DESC;

-- 6. Model performance comparison
SELECT * FROM model_performance;

-- 7. Cache performance analysis
SELECT 
    cached,
    COUNT(*) as count,
    AVG(processing_time_ms) as avg_processing_time,
    AVG(confidence) as avg_confidence
FROM predictions 
WHERE timestamp >= DATE_SUB(NOW(), INTERVAL 7 DAY)
GROUP BY cached;

-- 8. Confidence distribution
SELECT 
    CASE 
        WHEN confidence >= 0.9 THEN 'Very High (0.9-1.0)'
        WHEN confidence >= 0.7 THEN 'High (0.7-0.9)'
        WHEN confidence >= 0.5 THEN 'Medium (0.5-0.7)'
        ELSE 'Low (0.0-0.5)'
    END as confidence_range,
    COUNT(*) as prediction_count,
    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM predictions), 2) as percentage
FROM predictions 
GROUP BY 
    CASE 
        WHEN confidence >= 0.9 THEN 'Very High (0.9-1.0)'
        WHEN confidence >= 0.7 THEN 'High (0.7-0.9)'
        WHEN confidence >= 0.5 THEN 'Medium (0.5-0.7)'
        ELSE 'Low (0.0-0.5)'
    END
ORDER BY MIN(confidence) DESC;

-- ==========================================
-- ANALYSIS QUERIES
-- ==========================================

-- 9. Most common predicted CPs
SELECT 
    predicted_cp,
    COUNT(*) as prediction_count,
    AVG(confidence) as avg_confidence,
    MIN(confidence) as min_confidence,
    MAX(confidence) as max_confidence
FROM predictions 
WHERE timestamp >= DATE_SUB(NOW(), INTERVAL 7 DAY)
GROUP BY predicted_cp
ORDER BY prediction_count DESC
LIMIT 20;

-- 10. Performance by pincode prefix
SELECT 
    LEFT(pincode, 2) as pincode_prefix,
    COUNT(*) as total_predictions,
    AVG(confidence) as avg_confidence,
    AVG(processing_time_ms) as avg_processing_time,
    COUNT(DISTINCT predicted_cp) as unique_cps,
    COUNT(DISTINCT shipment_id) as unique_shipments
FROM predictions 
WHERE timestamp >= DATE_SUB(NOW(), INTERVAL 7 DAY)
GROUP BY LEFT(pincode, 2)
ORDER BY total_predictions DESC;

-- 11. Low confidence predictions (need review)
SELECT 
    shipment_id,
    address,
    pincode,
    predicted_cp,
    confidence,
    model_version,
    timestamp
FROM predictions 
WHERE confidence < 0.5
  AND timestamp >= DATE_SUB(NOW(), INTERVAL 7 DAY)
ORDER BY confidence ASC, timestamp DESC
LIMIT 50;

-- 12. Processing time analysis
SELECT 
    MIN(processing_time_ms) as min_time,
    MAX(processing_time_ms) as max_time,
    AVG(processing_time_ms) as avg_time,
    COUNT(*) as total_predictions,
    SUM(CASE WHEN processing_time_ms > 100 THEN 1 ELSE 0 END) as slow_predictions
FROM predictions 
WHERE processing_time_ms IS NOT NULL
  AND timestamp >= DATE_SUB(NOW(), INTERVAL 24 HOUR);

-- ==========================================
-- A/B TESTING QUERIES (if enabled)
-- ==========================================

-- 13. A/B testing performance comparison
SELECT 
    ab_variant,
    COUNT(*) as total_predictions,
    AVG(confidence) as avg_confidence,
    AVG(processing_time_ms) as avg_processing_time,
    COUNT(DISTINCT predicted_cp) as unique_cps
FROM predictions 
WHERE ab_variant IS NOT NULL
  AND timestamp >= DATE_SUB(NOW(), INTERVAL 7 DAY)
GROUP BY ab_variant
ORDER BY ab_variant;

-- ==========================================
-- HOURLY PATTERNS
-- ==========================================

-- 14. Predictions by hour of day
SELECT 
    HOUR(timestamp) as hour_of_day,
    COUNT(*) as total_predictions,
    AVG(confidence) as avg_confidence,
    SUM(CASE WHEN cached = TRUE THEN 1 ELSE 0 END) as cache_hits,
    ROUND(SUM(CASE WHEN cached = TRUE THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) as cache_hit_rate
FROM predictions 
WHERE timestamp >= DATE_SUB(NOW(), INTERVAL 7 DAY)
GROUP BY HOUR(timestamp)
ORDER BY hour_of_day;

-- ==========================================
-- DATA QUALITY CHECKS
-- ==========================================

-- 15. Find potential duplicate predictions
SELECT 
    address,
    COUNT(*) as prediction_count,
    COUNT(DISTINCT predicted_cp) as different_predictions,
    GROUP_CONCAT(DISTINCT predicted_cp ORDER BY predicted_cp) as all_predictions
FROM predictions 
GROUP BY address
HAVING COUNT(*) > 1 AND COUNT(DISTINCT predicted_cp) > 1
ORDER BY prediction_count DESC, different_predictions DESC
LIMIT 20;

-- 16. Validate data integrity
SELECT 
    'Total Records' as check_type,
    COUNT(*) as count,
    'OK' as status
FROM predictions

UNION ALL

SELECT 
    'Records with NULL shipment_id' as check_type,
    COUNT(*) as count,
    CASE WHEN COUNT(*) = 0 THEN 'OK' ELSE 'ISSUE' END as status
FROM predictions 
WHERE shipment_id IS NULL

UNION ALL

SELECT 
    'Records with invalid confidence' as check_type,
    COUNT(*) as count,
    CASE WHEN COUNT(*) = 0 THEN 'OK' ELSE 'ISSUE' END as status
FROM predictions 
WHERE confidence < 0 OR confidence > 1

UNION ALL

SELECT 
    'Records with invalid pincode' as check_type,
    COUNT(*) as count,
    CASE WHEN COUNT(*) = 0 THEN 'OK' ELSE 'ISSUE' END as status
FROM predictions 
WHERE LENGTH(pincode) != 6 OR pincode NOT REGEXP '^[0-9]{6}$';

-- ==========================================
-- MAINTENANCE QUERIES
-- ==========================================

-- 17. Table size and growth
SELECT 
    DATE(timestamp) as date,
    COUNT(*) as daily_records,
    SUM(COUNT(*)) OVER (ORDER BY DATE(timestamp)) as cumulative_records
FROM predictions 
WHERE timestamp >= DATE_SUB(NOW(), INTERVAL 30 DAY)
GROUP BY DATE(timestamp)
ORDER BY date;

-- 18. Clean up old data (UNCOMMENT WHEN READY TO DELETE)
-- Delete predictions older than 6 months
-- DELETE FROM predictions 
-- WHERE timestamp < DATE_SUB(NOW(), INTERVAL 6 MONTH);

-- 19. Archive old data (create archive table first)
-- CREATE TABLE predictions_archive LIKE predictions;
-- INSERT INTO predictions_archive 
-- SELECT * FROM predictions 
-- WHERE timestamp < DATE_SUB(NOW(), INTERVAL 6 MONTH);

-- ==========================================
-- SEARCH QUERIES
-- ==========================================

-- 20. Search by address pattern
SELECT 
    shipment_id,
    address,
    predicted_cp,
    confidence,
    timestamp
FROM predictions 
WHERE address LIKE '%Mumbai%'
ORDER BY timestamp DESC
LIMIT 20;

-- 21. Recent predictions with high confidence
SELECT 
    shipment_id,
    address,
    pincode,
    predicted_cp,
    confidence,
    processing_time_ms,
    timestamp
FROM predictions 
WHERE confidence >= 0.8
  AND timestamp >= DATE_SUB(NOW(), INTERVAL 24 HOUR)
ORDER BY confidence DESC, timestamp DESC
LIMIT 50;
