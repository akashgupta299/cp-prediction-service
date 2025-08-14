-- Direct MySQL setup with specified password
CREATE DATABASE IF NOT EXISTS cp_predictions 
CHARACTER SET utf8mb4 
COLLATE utf8mb4_unicode_ci;

USE cp_predictions;

DROP USER IF EXISTS 'cp_service'@'%';
CREATE USER 'cp_service'@'%' IDENTIFIED BY '$eRv!(e$@12';
GRANT SELECT, INSERT, UPDATE, DELETE ON cp_predictions.* TO 'cp_service'@'%';
FLUSH PRIVILEGES;

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
SELECT 'Database setup completed successfully!' as status;
