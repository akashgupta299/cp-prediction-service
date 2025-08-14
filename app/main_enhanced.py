"""
Enhanced CP Prediction Service with comprehensive improvements:
- Error handling & logging
- Input validation & response models
- Caching implementation
- Memory optimization
- Health checks & monitoring
- Configuration management
- Database integration
- A/B testing support
"""

import asyncio
import logging
import time
import traceback
import re
from datetime import datetime
from typing import Dict, Any, List
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException, status, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import ValidationError

# Local imports
from .config import settings
from .models import (
    PredictionRequest, PredictionResponse, BatchPredictionRequest, 
    BatchPredictionResponse, ErrorResponse, HealthResponse, 
    MetricsResponse, ModelStatus
)
from .model_manager import model_manager
from .database import db_manager
from .dummy_model import DummyModel, DummyVectorizer, DummyEncoder

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format=settings.log_format
)
logger = logging.getLogger(__name__)

# Global metrics
app_metrics = {
    "start_time": datetime.now(),
    "total_predictions": 0,
    "successful_predictions": 0,
    "failed_predictions": 0,
    "total_response_time": 0.0,
    "cache_hits": 0,
    "cache_misses": 0,
    "models_loaded": 0
}

# Thread pool for CPU-intensive tasks
executor = ThreadPoolExecutor(max_workers=4)

# Pincode regex
PINCODE_REGEX = re.compile(r"\b(\d{6})\b")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    logger.info("Starting CP Prediction Service")
    logger.info(f"Configuration: {settings.dict()}")
    
    # Warm up some models
    common_prefixes = ["28", "30", "31", "32"]
    model_manager.warm_up_models(common_prefixes)
    
    yield
    
    logger.info("Shutting down CP Prediction Service")


# Create FastAPI app
app = FastAPI(
    title="CP Prediction Service",
    description="Enhanced CP prediction service with comprehensive improvements",
    version="2.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def extract_pincode(address: str) -> str:
    """Extract pincode from address with caching"""
    match = PINCODE_REGEX.search(address)
    if not match:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Pincode not found in address"
        )
    return match.group(1)


async def predict_cp_enhanced(
    address: str, 
    shipment_id: str,
    background_tasks: BackgroundTasks
) -> Dict[str, Any]:
    """Enhanced prediction with caching, logging, and database storage"""
    start_time = time.time()
    
    try:
        # Check cache first
        cached_result = model_manager.get_cached_prediction(address)
        if cached_result:
            app_metrics["cache_hits"] += 1
            logger.info(f"Cache hit for shipment: {shipment_id}")
            
            result = cached_result.copy()
            result["cached"] = True
            result["processing_time_ms"] = (time.time() - start_time) * 1000
            
            # Save to database in background
            if settings.enable_prediction_history:
                background_tasks.add_task(
                    save_prediction_to_db,
                    shipment_id=shipment_id,
                    address=address,
                    result=result,
                    cached=True
                )
            
            return result
        
        app_metrics["cache_misses"] += 1
        
        # Extract pincode
        pincode = extract_pincode(address)
        prefix = pincode[:2]
        
        # Get model (with A/B testing support)
        if settings.enable_ab_testing:
            bundle = model_manager.get_model_for_ab_testing(prefix)
        else:
            bundle = model_manager.get_model(prefix)
        
        if not bundle:
            # Try dummy model as fallback
            logger.warning(f"No model found for prefix {prefix}, using dummy model")
            bundle = {
                "model": DummyModel(),
                "vectorizer": DummyVectorizer(),
                "encoder": DummyEncoder(),
                "ab_variant": "fallback"
            }
        
        # Make prediction
        model = bundle['model']
        vectorizer = bundle['vectorizer']
        encoder = bundle['encoder']
        
        features = vectorizer.transform([address])
        probs = model.predict_proba(features)[0]
        best_idx = np.argmax(probs)
        predicted_cp = encoder.inverse_transform([best_idx])[0]
        confidence = float(probs[best_idx])
        
        # Prepare result
        processing_time = (time.time() - start_time) * 1000
        result = {
            'predicted_cp': predicted_cp,
            'confidence': confidence,
            'pincode': pincode,
            'model_version': settings.model_version,
            'processing_time_ms': processing_time,
            'cached': False,
            'ab_variant': bundle.get('ab_variant')
        }
        
        # Cache the result
        model_manager.cache_prediction(address, result)
        
        # Save to database in background
        if settings.enable_prediction_history:
            background_tasks.add_task(
                save_prediction_to_db,
                shipment_id=shipment_id,
                address=address,
                result=result,
                cached=False
            )
        
        logger.info(
            f"Prediction successful for shipment: {shipment_id}, "
            f"CP: {predicted_cp}, Confidence: {confidence:.3f}, "
            f"Time: {processing_time:.2f}ms"
        )
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction failed for shipment {shipment_id}: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error during prediction: {str(e)}"
        )


def save_prediction_to_db(
    shipment_id: str,
    address: str,
    result: Dict[str, Any],
    cached: bool
):
    """Save prediction to database (background task)"""
    try:
        db_manager.save_prediction(
            shipment_id=shipment_id,
            address=address,
            pincode=result.get('pincode', ''),
            predicted_cp=result.get('predicted_cp', ''),
            confidence=result.get('confidence', 0.0),
            model_version=result.get('model_version'),
            processing_time_ms=result.get('processing_time_ms'),
            cached=cached
        )
    except Exception as e:
        logger.error(f"Failed to save prediction to database: {str(e)}")


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    request: PredictionRequest,
    background_tasks: BackgroundTasks
):
    """Single prediction endpoint with comprehensive error handling"""
    start_time = time.time()
    
    try:
        app_metrics["total_predictions"] += 1
        
        result = await predict_cp_enhanced(
            request.address, 
            request.shipment_id,
            background_tasks
        )
        
        app_metrics["successful_predictions"] += 1
        response_time = (time.time() - start_time) * 1000
        app_metrics["total_response_time"] += response_time
        
        return PredictionResponse(
            shipment_id=request.shipment_id,
            predicted_cp=result['predicted_cp'],
            confidence=result['confidence'],
            pincode=result['pincode'],
            model_version=result.get('model_version'),
            processing_time_ms=result.get('processing_time_ms'),
            cached=result.get('cached', False)
        )
        
    except HTTPException:
        app_metrics["failed_predictions"] += 1
        raise
    except ValidationError as e:
        app_metrics["failed_predictions"] += 1
        logger.error(f"Validation error for shipment {request.shipment_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Validation error: {str(e)}"
        )
    except Exception as e:
        app_metrics["failed_predictions"] += 1
        logger.error(f"Unexpected error for shipment {request.shipment_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(
    request: BatchPredictionRequest,
    background_tasks: BackgroundTasks
):
    """Batch prediction endpoint with parallel processing"""
    start_time = time.time()
    
    if len(request.requests) > settings.batch_size_limit:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Batch size too large (max {settings.batch_size_limit})"
        )
    
    # Group requests by pincode prefix for efficient processing
    grouped_requests = {}
    for req in request.requests:
        try:
            pincode = extract_pincode(req.address)
            prefix = pincode[:2]
            if prefix not in grouped_requests:
                grouped_requests[prefix] = []
            grouped_requests[prefix].append(req)
        except Exception as e:
            logger.error(f"Error processing request {req.shipment_id}: {str(e)}")
            continue
    
    # Process each group in parallel
    tasks = []
    for prefix, group_requests in grouped_requests.items():
        task = asyncio.create_task(
            process_batch_for_prefix(prefix, group_requests, background_tasks)
        )
        tasks.append(task)
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Flatten results
    all_results = []
    for result in results:
        if isinstance(result, list):
            all_results.extend(result)
        elif isinstance(result, Exception):
            logger.error(f"Batch processing error: {str(result)}")
    
    total_processing_time = (time.time() - start_time) * 1000
    successful_predictions = len(all_results)
    failed_predictions = len(request.requests) - successful_predictions
    
    # Update metrics
    app_metrics["total_predictions"] += len(request.requests)
    app_metrics["successful_predictions"] += successful_predictions
    app_metrics["failed_predictions"] += failed_predictions
    app_metrics["total_response_time"] += total_processing_time
    
    return BatchPredictionResponse(
        results=all_results,
        total_requests=len(request.requests),
        successful_predictions=successful_predictions,
        failed_predictions=failed_predictions,
        total_processing_time_ms=total_processing_time
    )


async def process_batch_for_prefix(
    prefix: str, 
    requests: List[PredictionRequest],
    background_tasks: BackgroundTasks
) -> List[PredictionResponse]:
    """Process batch of requests for same prefix"""
    results = []
    
    for req in requests:
        try:
            result = await predict_cp_enhanced(req.address, req.shipment_id, background_tasks)
            
            results.append(PredictionResponse(
                shipment_id=req.shipment_id,
                predicted_cp=result['predicted_cp'],
                confidence=result['confidence'],
                pincode=result['pincode'],
                model_version=result.get('model_version'),
                processing_time_ms=result.get('processing_time_ms'),
                cached=result.get('cached', False)
            ))
            
        except Exception as e:
            logger.error(f"Failed to process request {req.shipment_id}: {str(e)}")
            continue
    
    return results


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Comprehensive health check endpoint"""
    try:
        uptime = (datetime.now() - app_metrics["start_time"]).total_seconds()
        status_info = model_manager.get_status()
        
        # Get memory usage
        try:
            import psutil
            process = psutil.Process()
            memory_usage = process.memory_info().rss / 1024 / 1024  # MB
        except ImportError:
            memory_usage = 0.0
        
        return HealthResponse(
            status="healthy",
            uptime_seconds=uptime,
            models_available=status_info["available_prefixes"],
            memory_usage_mb=memory_usage,
            cache_size=status_info["cache_size"]
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service unhealthy"
        )


@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    """Get service metrics"""
    avg_response_time = (
        app_metrics["total_response_time"] / max(app_metrics["total_predictions"], 1)
    )
    
    return MetricsResponse(
        start_time=app_metrics["start_time"],
        total_predictions=app_metrics["total_predictions"],
        successful_predictions=app_metrics["successful_predictions"],
        failed_predictions=app_metrics["failed_predictions"],
        avg_response_time_ms=avg_response_time,
        cache_hits=app_metrics["cache_hits"],
        cache_misses=app_metrics["cache_misses"],
        models_loaded=len(model_manager._models)
    )


@app.get("/models/status", response_model=ModelStatus)
async def models_status():
    """Get model status information"""
    status_info = model_manager.get_status()
    
    return ModelStatus(
        available_prefixes=status_info["available_prefixes"],
        total_models_loaded=status_info["models_loaded"],
        model_access_times=status_info["model_access_times"],
        memory_usage_per_model=status_info["memory_usage_per_model"]
    )


@app.post("/cache/clear")
async def clear_cache():
    """Clear prediction cache"""
    model_manager.clear_cache()
    return {"message": "Cache cleared successfully"}


@app.get("/history/{shipment_id}")
async def get_prediction_history(shipment_id: str):
    """Get prediction history for a shipment"""
    if not settings.enable_prediction_history:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Prediction history not enabled"
        )
    
    history = db_manager.get_prediction_history(shipment_id=shipment_id)
    return {"shipment_id": shipment_id, "history": history}


@app.get("/statistics")
async def get_statistics():
    """Get service statistics"""
    db_stats = db_manager.get_statistics()
    
    return {
        "service_metrics": app_metrics,
        "database_statistics": db_stats,
        "model_status": model_manager.get_status()
    }


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler"""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error="HTTP Exception",
            detail=exc.detail,
            timestamp=datetime.now()
        ).dict()
    )


@app.exception_handler(ValidationError)
async def validation_exception_handler(request, exc):
    """Custom validation exception handler"""
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=ErrorResponse(
            error="Validation Error",
            detail=str(exc),
            timestamp=datetime.now()
        ).dict()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """General exception handler"""
    logger.error(f"Unhandled exception: {str(exc)}")
    logger.error(traceback.format_exc())
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="Internal Server Error",
            detail="An unexpected error occurred",
            timestamp=datetime.now()
        ).dict()
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main_enhanced:app",
        host=settings.host,
        port=settings.port,
        workers=settings.workers,
        reload=True
    )
