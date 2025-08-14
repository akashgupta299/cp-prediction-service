-- MySQL Database Initialization Script for CP Prediction Service
-- This script creates the database and tables for storing prediction history

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

-- Create prediction_history table
CREATE TABLE IF NOT EXISTS prediction_history (
    id INT AUTO_INCREMENT PRIMARY KEY,
    shipment_id VARCHAR(100) NOT NULL,
    address TEXT NOT NULL,
    pincode VARCHAR(6) NOT NULL,
    predicted_cp VARCHAR(100) NOT NULL,
    confidence DECIMAL(10, 8) NOT NULL,
    model_version VARCHAR(50) DEFAULT NULL,
    processing_time_ms DECIMAL(10, 3) DEFAULT NULL,
    cached BOOLEAN DEFAULT FALSE,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    
    -- Indexes for better query performance
    INDEX idx_shipment_id (shipment_id),
    INDEX idx_pincode (pincode),
    INDEX idx_timestamp (timestamp),
    INDEX idx_model_version (model_version),
    INDEX idx_cached (cached)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Create indexes for better performance
CREATE INDEX idx_shipment_timestamp ON prediction_history (shipment_id, timestamp);
CREATE INDEX idx_pincode_timestamp ON prediction_history (pincode, timestamp);

-- Create a table for storing service metrics (optional)
CREATE TABLE IF NOT EXISTS service_metrics (
    id INT AUTO_INCREMENT PRIMARY KEY,
    metric_name VARCHAR(100) NOT NULL,
    metric_value DECIMAL(15, 6) NOT NULL,
    metric_type ENUM('counter', 'gauge', 'histogram') DEFAULT 'gauge',
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    
    INDEX idx_metric_name (metric_name),
    INDEX idx_timestamp (timestamp)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Create a table for model performance tracking (optional)
CREATE TABLE IF NOT EXISTS model_performance (
    id INT AUTO_INCREMENT PRIMARY KEY,
    model_version VARCHAR(50) NOT NULL,
    pincode_prefix VARCHAR(2) NOT NULL,
    total_predictions INT DEFAULT 0,
    avg_confidence DECIMAL(10, 8) DEFAULT 0.0,
    avg_processing_time_ms DECIMAL(10, 3) DEFAULT 0.0,
    last_updated DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    
    UNIQUE KEY unique_model_prefix (model_version, pincode_prefix),
    INDEX idx_model_version (model_version),
    INDEX idx_pincode_prefix (pincode_prefix)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Insert some initial data (optional)
INSERT IGNORE INTO service_metrics (metric_name, metric_value, metric_type) VALUES
('service_start_time', UNIX_TIMESTAMP(), 'gauge'),
('total_predictions', 0, 'counter'),
('successful_predictions', 0, 'counter'),
('failed_predictions', 0, 'counter');

-- Create views for common queries
CREATE OR REPLACE VIEW recent_predictions AS
SELECT 
    shipment_id,
    address,
    pincode,
    predicted_cp,
    confidence,
    model_version,
    processing_time_ms,
    cached,
    timestamp
FROM prediction_history 
WHERE timestamp >= DATE_SUB(NOW(), INTERVAL 24 HOUR)
ORDER BY timestamp DESC;

CREATE OR REPLACE VIEW prediction_stats AS
SELECT 
    DATE(timestamp) as prediction_date,
    COUNT(*) as total_predictions,
    AVG(confidence) as avg_confidence,
    AVG(processing_time_ms) as avg_processing_time,
    SUM(CASE WHEN cached = TRUE THEN 1 ELSE 0 END) as cache_hits,
    COUNT(DISTINCT pincode) as unique_pincodes,
    COUNT(DISTINCT predicted_cp) as unique_cps
FROM prediction_history 
GROUP BY DATE(timestamp)
ORDER BY prediction_date DESC;

-- Show table structure
SHOW TABLES;
DESCRIBE prediction_history;
DESCRIBE service_metrics;
DESCRIBE model_performance;
