-- Sample MySQL Queries for CP Prediction Service
-- Use these queries to interact with your prediction data

-- 1. Get recent predictions (last 24 hours)
SELECT 
    shipment_id,
    address,
    predicted_cp,
    confidence,
    processing_time_ms,
    timestamp
FROM prediction_history 
WHERE timestamp >= DATE_SUB(NOW(), INTERVAL 24 HOUR)
ORDER BY timestamp DESC
LIMIT 100;

-- 2. Get prediction history for a specific shipment
SELECT * FROM prediction_history 
WHERE shipment_id = 'SHIP123456'
ORDER BY timestamp DESC;

-- 3. Get predictions by pincode
SELECT 
    shipment_id,
    predicted_cp,
    confidence,
    timestamp
FROM prediction_history 
WHERE pincode = '280001'
ORDER BY timestamp DESC
LIMIT 50;

-- 4. Get daily prediction statistics
SELECT 
    DATE(timestamp) as date,
    COUNT(*) as total_predictions,
    AVG(confidence) as avg_confidence,
    AVG(processing_time_ms) as avg_processing_time,
    SUM(CASE WHEN cached = TRUE THEN 1 ELSE 0 END) as cache_hits,
    ROUND(SUM(CASE WHEN cached = TRUE THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) as cache_hit_rate
FROM prediction_history 
WHERE timestamp >= DATE_SUB(NOW(), INTERVAL 30 DAY)
GROUP BY DATE(timestamp)
ORDER BY date DESC;

-- 5. Get most common predicted CPs
SELECT 
    predicted_cp,
    COUNT(*) as prediction_count,
    AVG(confidence) as avg_confidence
FROM prediction_history 
WHERE timestamp >= DATE_SUB(NOW(), INTERVAL 7 DAY)
GROUP BY predicted_cp
ORDER BY prediction_count DESC
LIMIT 20;

-- 6. Get performance by pincode prefix
SELECT 
    LEFT(pincode, 2) as pincode_prefix,
    COUNT(*) as total_predictions,
    AVG(confidence) as avg_confidence,
    AVG(processing_time_ms) as avg_processing_time,
    COUNT(DISTINCT predicted_cp) as unique_cps
FROM prediction_history 
WHERE timestamp >= DATE_SUB(NOW(), INTERVAL 7 DAY)
GROUP BY LEFT(pincode, 2)
ORDER BY total_predictions DESC;

-- 7. Get low confidence predictions (might need review)
SELECT 
    shipment_id,
    address,
    predicted_cp,
    confidence,
    timestamp
FROM prediction_history 
WHERE confidence < 0.5
  AND timestamp >= DATE_SUB(NOW(), INTERVAL 7 DAY)
ORDER BY confidence ASC, timestamp DESC
LIMIT 100;

-- 8. Get processing time statistics
SELECT 
    MIN(processing_time_ms) as min_time,
    MAX(processing_time_ms) as max_time,
    AVG(processing_time_ms) as avg_time,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY processing_time_ms) as median_time,
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY processing_time_ms) as p95_time,
    COUNT(*) as total_predictions
FROM prediction_history 
WHERE processing_time_ms IS NOT NULL
  AND timestamp >= DATE_SUB(NOW(), INTERVAL 24 HOUR);

-- 9. Get model version performance comparison
SELECT 
    model_version,
    COUNT(*) as total_predictions,
    AVG(confidence) as avg_confidence,
    AVG(processing_time_ms) as avg_processing_time,
    MIN(timestamp) as first_used,
    MAX(timestamp) as last_used
FROM prediction_history 
WHERE model_version IS NOT NULL
GROUP BY model_version
ORDER BY first_used DESC;

-- 10. Clean up old data (run periodically to manage storage)
-- Delete predictions older than 90 days
-- DELETE FROM prediction_history 
-- WHERE timestamp < DATE_SUB(NOW(), INTERVAL 90 DAY);

-- 11. Update model performance table (run this periodically)
INSERT INTO model_performance (
    model_version, 
    pincode_prefix, 
    total_predictions, 
    avg_confidence, 
    avg_processing_time_ms
)
SELECT 
    model_version,
    LEFT(pincode, 2) as pincode_prefix,
    COUNT(*) as total_predictions,
    AVG(confidence) as avg_confidence,
    AVG(processing_time_ms) as avg_processing_time_ms
FROM prediction_history 
WHERE model_version IS NOT NULL
  AND timestamp >= DATE_SUB(NOW(), INTERVAL 1 DAY)
GROUP BY model_version, LEFT(pincode, 2)
ON DUPLICATE KEY UPDATE
    total_predictions = total_predictions + VALUES(total_predictions),
    avg_confidence = (avg_confidence + VALUES(avg_confidence)) / 2,
    avg_processing_time_ms = (avg_processing_time_ms + VALUES(avg_processing_time_ms)) / 2,
    last_updated = CURRENT_TIMESTAMP;

-- 12. Get cache effectiveness by time of day
SELECT 
    HOUR(timestamp) as hour_of_day,
    COUNT(*) as total_requests,
    SUM(CASE WHEN cached = TRUE THEN 1 ELSE 0 END) as cache_hits,
    ROUND(SUM(CASE WHEN cached = TRUE THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) as cache_hit_rate
FROM prediction_history 
WHERE timestamp >= DATE_SUB(NOW(), INTERVAL 7 DAY)
GROUP BY HOUR(timestamp)
ORDER BY hour_of_day;

-- 13. Find duplicate predictions (same address, different results)
SELECT 
    address,
    COUNT(DISTINCT predicted_cp) as different_predictions,
    COUNT(*) as total_predictions,
    GROUP_CONCAT(DISTINCT predicted_cp) as all_predictions
FROM prediction_history 
GROUP BY address
HAVING COUNT(DISTINCT predicted_cp) > 1
ORDER BY different_predictions DESC, total_predictions DESC
LIMIT 50;
