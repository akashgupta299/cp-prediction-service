-- Simple MySQL Database Initialization Script for CP Prediction Service
-- This script creates only one table to store all prediction data

-- Create database (run this as root or admin user)
CREATE DATABASE IF NOT EXISTS cp_predictions 
CHARACTER SET utf8mb4 
COLLATE utf8mb4_unicode_ci;

-- Use the database
USE cp_predictions;

-- Create user for the application (optional, for security)
-- Replace 'your_password' with a strong password
CREATE USER IF NOT EXISTS 'cp_service'@'%' IDENTIFIED BY 'your_password';
GRANT SELECT, INSERT, UPDATE, DELETE ON cp_predictions.* TO 'cp_service'@'%';
FLUSH PRIVILEGES;

-- Create single predictions table with all necessary fields
CREATE TABLE IF NOT EXISTS predictions (
    -- Primary key
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    
    -- Request information
    shipment_id VARCHAR(100) NOT NULL,
    address TEXT NOT NULL,
    pincode VARCHAR(6) NOT NULL,
    
    -- Prediction results
    predicted_cp VARCHAR(100) NOT NULL,
    confidence DECIMAL(5, 4) NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
    
    -- Model and performance info
    model_version VARCHAR(50) DEFAULT 'v1',
    processing_time_ms DECIMAL(8, 3) DEFAULT NULL,
    cached BOOLEAN DEFAULT FALSE,
    
    -- A/B Testing (optional)
    ab_variant VARCHAR(10) DEFAULT NULL,
    
    -- Metadata
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    
    -- Indexes for better query performance
    INDEX idx_shipment_id (shipment_id),
    INDEX idx_pincode (pincode),
    INDEX idx_pincode_prefix (pincode(2)),  -- For prefix-based queries
    INDEX idx_predicted_cp (predicted_cp),
    INDEX idx_timestamp (timestamp),
    INDEX idx_model_version (model_version),
    INDEX idx_confidence (confidence),
    INDEX idx_cached (cached),
    INDEX idx_ab_variant (ab_variant),
    
    -- Composite indexes for common query patterns
    INDEX idx_shipment_timestamp (shipment_id, timestamp),
    INDEX idx_pincode_timestamp (pincode, timestamp),
    INDEX idx_model_timestamp (model_version, timestamp),
    INDEX idx_cached_timestamp (cached, timestamp)
    
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Create useful views for common queries
CREATE OR REPLACE VIEW recent_predictions AS
SELECT 
    id,
    shipment_id,
    address,
    pincode,
    predicted_cp,
    confidence,
    model_version,
    processing_time_ms,
    cached,
    ab_variant,
    timestamp
FROM predictions 
WHERE timestamp >= DATE_SUB(NOW(), INTERVAL 24 HOUR)
ORDER BY timestamp DESC;

CREATE OR REPLACE VIEW daily_stats AS
SELECT 
    DATE(timestamp) as prediction_date,
    COUNT(*) as total_predictions,
    AVG(confidence) as avg_confidence,
    AVG(processing_time_ms) as avg_processing_time,
    SUM(CASE WHEN cached = TRUE THEN 1 ELSE 0 END) as cache_hits,
    ROUND(SUM(CASE WHEN cached = TRUE THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) as cache_hit_rate,
    COUNT(DISTINCT pincode) as unique_pincodes,
    COUNT(DISTINCT predicted_cp) as unique_cps,
    COUNT(DISTINCT shipment_id) as unique_shipments
FROM predictions 
GROUP BY DATE(timestamp)
ORDER BY prediction_date DESC;

CREATE OR REPLACE VIEW model_performance AS
SELECT 
    model_version,
    COUNT(*) as total_predictions,
    AVG(confidence) as avg_confidence,
    AVG(processing_time_ms) as avg_processing_time,
    MIN(timestamp) as first_used,
    MAX(timestamp) as last_used,
    COUNT(DISTINCT pincode) as unique_pincodes,
    COUNT(DISTINCT predicted_cp) as unique_cps
FROM predictions 
WHERE model_version IS NOT NULL
GROUP BY model_version
ORDER BY first_used DESC;

-- Insert a sample record to verify table structure
INSERT INTO predictions (
    shipment_id, 
    address, 
    pincode, 
    predicted_cp, 
    confidence, 
    model_version,
    processing_time_ms,
    cached
) VALUES (
    'SAMPLE_001',
    'Sample Address, Mumbai 400001',
    '400001',
    'SAMPLE_CP',
    0.8500,
    'v1',
    25.50,
    FALSE
);

-- Show table structure
DESCRIBE predictions;

-- Show sample data
SELECT * FROM predictions LIMIT 5;

-- Show views
SHOW FULL TABLES WHERE Table_type = 'VIEW';

-- Display table size information
SELECT 
    table_name,
    table_rows,
    ROUND(((data_length + index_length) / 1024 / 1024), 2) AS table_size_mb
FROM information_schema.TABLES 
WHERE table_schema = 'cp_predictions' AND table_name = 'predictions';

-- Success message
SELECT 'Database setup completed successfully!' as status;
