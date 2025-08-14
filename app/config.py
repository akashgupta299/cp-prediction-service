"""Configuration management for CP Prediction Service"""
import os
from typing import List
from pydantic import BaseSettings, Field


class Settings(BaseSettings):
    """Application settings with environment variable support"""
    
    # Server settings
    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=8000, env="PORT")
    workers: int = Field(default=1, env="WORKERS")
    
    # Model settings
    model_dir: str = Field(default="models", env="MODEL_DIR")
    max_models_in_memory: int = Field(default=5, env="MAX_MODELS_IN_MEMORY")
    model_version: str = Field(default="v1", env="MODEL_VERSION")
    
    # Cache settings
    cache_ttl_seconds: int = Field(default=3600, env="CACHE_TTL_SECONDS")  # 1 hour
    cache_max_size: int = Field(default=10000, env="CACHE_MAX_SIZE")
    
    # API settings
    batch_size_limit: int = Field(default=100, env="BATCH_SIZE_LIMIT")
    max_request_size: int = Field(default=1024*1024, env="MAX_REQUEST_SIZE")  # 1MB
    
    # Logging
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_format: str = Field(default="%(asctime)s - %(name)s - %(levelname)s - %(message)s", env="LOG_FORMAT")
    
    # Database settings
    database_url: str = Field(default="sqlite:///predictions.db", env="DATABASE_URL")
    enable_prediction_history: bool = Field(default=True, env="ENABLE_PREDICTION_HISTORY")
    
    # A/B Testing
    enable_ab_testing: bool = Field(default=False, env="ENABLE_AB_TESTING")
    model_a_weight: float = Field(default=0.5, env="MODEL_A_WEIGHT")
    model_b_weight: float = Field(default=0.5, env="MODEL_B_WEIGHT")
    
    # Security
    api_key_header: str = Field(default="X-API-Key", env="API_KEY_HEADER")
    allowed_origins: List[str] = Field(default=["*"], env="ALLOWED_ORIGINS")
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()
