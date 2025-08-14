-- Basic MySQL Queries for Predictions Table Only

-- 1. Get recent predictions
SELECT * FROM predictions 
ORDER BY timestamp DESC 
LIMIT 20;

-- 2. Get prediction by shipment ID
SELECT * FROM predictions 
WHERE shipment_id = 'SHIP123456';

-- 3. Get predictions by pincode
SELECT * FROM predictions 
WHERE pincode = '280001'
ORDER BY timestamp DESC;

-- 4. Count total predictions
SELECT COUNT(*) as total_predictions FROM predictions;

-- 5. Get predictions from last 24 hours
SELECT * FROM predictions 
WHERE timestamp >= DATE_SUB(NOW(), INTERVAL 24 HOUR)
ORDER BY timestamp DESC;

-- 6. Get average confidence
SELECT AVG(confidence) as avg_confidence FROM predictions;

-- 7. Get predictions with low confidence
SELECT * FROM predictions 
WHERE confidence < 0.5
ORDER BY confidence ASC;

-- 8. Get cached vs non-cached predictions
SELECT 
    cached,
    COUNT(*) as count
FROM predictions 
GROUP BY cached;

-- 9. Get predictions by model version
SELECT 
    model_version,
    COUNT(*) as count
FROM predictions 
GROUP BY model_version;

-- 10. Delete old predictions (uncomment when needed)
-- DELETE FROM predictions 
-- WHERE timestamp < DATE_SUB(NOW(), INTERVAL 90 DAY);
