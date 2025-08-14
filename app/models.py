"""Pydantic models for request/response validation"""
from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, Field, validator
import re


class PredictionRequest(BaseModel):
    """Request model for prediction endpoint"""
    address: str = Field(
        ..., 
        min_length=10, 
        max_length=500, 
        description="Full address with 6-digit pincode",
        example="123 Main Street, Mumbai 280001"
    )
    shipment_id: str = Field(
        ..., 
        min_length=1, 
        max_length=100, 
        description="Unique shipment identifier",
        example="SHIP123456"
    )
    
    @validator('address')
    def validate_address(cls, v):
        if not v.strip():
            raise ValueError('Address cannot be empty or whitespace only')
        # Check if pincode pattern exists
        if not re.search(r'\b\d{6}\b', v):
            raise ValueError('Address must contain a valid 6-digit pincode')
        return v.strip()
    
    @validator('shipment_id')
    def validate_shipment_id(cls, v):
        if not v.strip():
            raise ValueError('Shipment ID cannot be empty')
        return v.strip()


class PredictionResponse(BaseModel):
    """Response model for prediction endpoint"""
    shipment_id: str = Field(..., description="Unique shipment identifier")
    predicted_cp: str = Field(..., description="Predicted CP location")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Prediction confidence score")
    pincode: str = Field(..., description="Extracted pincode from address")
    model_version: Optional[str] = Field(None, description="Model version used")
    processing_time_ms: Optional[float] = Field(None, description="Processing time in milliseconds")
    timestamp: datetime = Field(default_factory=datetime.now, description="Prediction timestamp")
    cached: bool = Field(default=False, description="Whether result was from cache")


class BatchPredictionRequest(BaseModel):
    """Request model for batch prediction endpoint"""
    requests: List[PredictionRequest] = Field(
        ..., 
        min_items=1, 
        max_items=100, 
        description="List of prediction requests"
    )


class BatchPredictionResponse(BaseModel):
    """Response model for batch prediction endpoint"""
    results: List[PredictionResponse] = Field(..., description="List of prediction results")
    total_requests: int = Field(..., description="Total number of requests processed")
    successful_predictions: int = Field(..., description="Number of successful predictions")
    failed_predictions: int = Field(..., description="Number of failed predictions")
    total_processing_time_ms: float = Field(..., description="Total processing time")


class ErrorResponse(BaseModel):
    """Error response model"""
    error: str = Field(..., description="Error type")
    detail: str = Field(..., description="Error details")
    shipment_id: Optional[str] = Field(None, description="Related shipment ID if applicable")
    timestamp: datetime = Field(default_factory=datetime.now, description="Error timestamp")


class HealthResponse(BaseModel):
    """Health check response model"""
    status: str = Field(..., description="Service status")
    timestamp: datetime = Field(default_factory=datetime.now, description="Check timestamp")
    uptime_seconds: float = Field(..., description="Service uptime in seconds")
    models_available: List[str] = Field(..., description="Available model prefixes")
    memory_usage_mb: float = Field(..., description="Current memory usage in MB")
    cache_size: int = Field(..., description="Current cache size")


class MetricsResponse(BaseModel):
    """Metrics response model"""
    start_time: datetime = Field(..., description="Service start time")
    total_predictions: int = Field(..., description="Total predictions made")
    successful_predictions: int = Field(..., description="Successful predictions")
    failed_predictions: int = Field(..., description="Failed predictions")
    avg_response_time_ms: float = Field(..., description="Average response time")
    cache_hits: int = Field(..., description="Cache hits")
    cache_misses: int = Field(..., description="Cache misses")
    models_loaded: int = Field(..., description="Number of models loaded")


class ModelStatus(BaseModel):
    """Model status response model"""
    available_prefixes: List[str] = Field(..., description="Available model prefixes")
    total_models_loaded: int = Field(..., description="Total models in memory")
    model_access_times: dict = Field(..., description="Last access time for each model")
    memory_usage_per_model: dict = Field(..., description="Memory usage per model")
