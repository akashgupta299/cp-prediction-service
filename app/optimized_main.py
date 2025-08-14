#!/usr/bin/env python3
"""
Optimized CP Prediction Service with:
1. Memory Management & Model Loading
2. Async Model Loading & Caching
3. Database Connection Pooling
4. Batch Processing & Async I/O
5. Memory-Mapped Files
6. Response Caching
7. Performance Monitoring & Profiling
"""

import asyncio
import time
import hashlib
import logging
import psutil
import mmap
import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
import numpy as np
import pandas as pd

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn
from sqlalchemy import create_engine, text
from sqlalchemy.pool import QueuePool
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
import pymysql
from cachetools import TTLCache
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
DATABASE_URL = os.getenv("DATABASE_URL", "mysql+pymysql://cp_test:test123@localhost/cp_predictions")
MODELS_DIR = Path("../models")  # Relative to app directory (go up one level)
CACHE_TTL = 3600  # 1 hour
CACHE_MAXSIZE = 10000
BATCH_SIZE = 100
MAX_WORKERS = 4

# Performance monitoring
class PerformanceMetrics:
    def __init__(self):
        self.request_times = defaultdict(list)
        self.cache_hits = 0
        self.cache_misses = 0
        self.model_load_times = defaultdict(list)
        self.db_operation_times = defaultdict(list)
        self.memory_usage = []
    
    def record_request_time(self, operation: str, duration: float):
        self.request_times[operation].append(duration)
    
    def record_cache_hit(self):
        self.cache_hits += 1
    
    def record_cache_miss(self):
        self.cache_misses += 1
    
    def record_model_load_time(self, prefix: str, duration: float):
        self.model_load_times[prefix].append(duration)
    
    def record_db_operation_time(self, operation: str, duration: float):
        self.db_operation_times[operation].append(duration)
    
    def record_memory_usage(self):
        process = psutil.Process()
        memory_info = process.memory_info()
        self.memory_usage.append({
            'timestamp': time.time(),
            'rss_mb': memory_info.rss / 1024 / 1024,
            'vms_mb': memory_info.vms / 1024 / 1024
        })
    
    def get_stats(self):
        stats = {}
        for operation, times in self.request_times.items():
            if times:
                stats[operation] = {
                    'count': len(times),
                    'avg_ms': np.mean(times) * 1000,
                    'p95_ms': np.percentile(times, 95) * 1000,
                    'p99_ms': np.percentile(times, 99) * 1000
                }
        
        stats['cache'] = {
            'hits': self.cache_hits,
            'misses': self.cache_misses,
            'hit_rate': self.cache_hits / (self.cache_hits + self.cache_misses) if (self.cache_hits + self.cache_misses) > 0 else 0
        }
        
        stats['memory'] = {
            'current_rss_mb': psutil.Process().memory_info().rss / 1024 / 1024,
            'peak_rss_mb': max([m['rss_mb'] for m in self.memory_usage]) if self.memory_usage else 0
        }
        
        return stats

# Performance monitoring context manager
@contextmanager
def performance_monitor(operation_name: str, metrics: PerformanceMetrics):
    start_time = time.time()
    start_memory = psutil.Process().memory_info().rss
    try:
        yield
    finally:
        duration = time.time() - start_time
        memory_used = psutil.Process().memory_info().rss - start_memory
        metrics.record_request_time(operation_name, duration)
        logger.info(f"{operation_name}: {duration:.3f}s, Memory: {memory_used/1024/1024:.2f}MB")

# Pydantic models
class PredictionRequest(BaseModel):
    address: str = Field(..., description="Address to predict CP for")
    shipment_id: str = Field(..., description="Unique shipment identifier")

class BatchPredictionRequest(BaseModel):
    addresses: List[str] = Field(..., description="List of addresses to predict")
    shipment_ids: List[str] = Field(..., description="List of shipment IDs")

class PredictionResponse(BaseModel):
    shipment_id: str
    predicted_cp: str
    confidence: float
    processing_time_ms: float
    cached: bool = False

class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
    total_processing_time_ms: float
    batch_size: int

# Optimized Model Manager with Memory Management
class OptimizedModelManager:
    def __init__(self, models_dir: Path, max_workers: int = MAX_WORKERS):
        self.models_dir = models_dir
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.models_cache = {}
        self.loading_locks = {}
        self.model_load_times = {}
        self.metrics = PerformanceMetrics()
        
        # Pre-load all available models
        self.preload_all_models()
    
    def get_available_prefixes(self) -> List[str]:
        """Get list of available model prefixes"""
        prefixes = []
        if self.models_dir.exists():
            for item in self.models_dir.iterdir():
                if item.is_dir() and item.name.isdigit():
                    prefixes.append(item.name)
        return sorted(prefixes)
    
    def preload_all_models(self):
        """Pre-load all available models into memory at startup"""
        logger.info("🚀 Pre-loading all models into memory...")
        start_time = time.time()
        
        prefixes = self.get_available_prefixes()
        logger.info(f"📁 Found {len(prefixes)} model prefixes: {prefixes}")
        
        # Load models synchronously to avoid event loop conflicts
        for prefix in prefixes:
            try:
                # Load model synchronously
                model_data = self.load_model_sync(prefix)
                if model_data:
                    self.models_cache[prefix] = model_data
                    logger.info(f"✅ Loaded model for prefix {prefix}")
                else:
                    logger.warning(f"❌ Failed to load model for prefix {prefix}")
            except Exception as e:
                logger.error(f"❌ Error loading model for prefix {prefix}: {e}")
        
        total_time = time.time() - start_time
        logger.info(f"🎯 Model pre-loading completed in {total_time:.2f}s")
        logger.info(f"💾 Loaded {len(self.models_cache)} models into memory")
    
    def load_model_sync(self, prefix: str) -> Optional[Dict]:
        """Load model synchronously with memory mapping"""
        try:
            with performance_monitor(f"model_load_{prefix}", self.metrics):
                model_path = self.models_dir / prefix
                
                if not model_path.exists():
                    return None
                
                # Load model files with memory mapping
                model_data = {}
                
                # Load model
                model_file = model_path / "address_model.joblib"
                if model_file.exists():
                    with open(model_file, 'rb') as f:
                        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
                        model_data['model'] = joblib.load(mm)
                        mm.close()
                
                # Load vectorizer
                vectorizer_file = model_path / "address_vectorizer.joblib"
                if vectorizer_file.exists():
                    with open(vectorizer_file, 'rb') as f:
                        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
                        model_data['vectorizer'] = joblib.load(mm)
                        mm.close()
                
                # Load encoder
                encoder_file = model_path / "address_encoder.joblib"
                if encoder_file.exists():
                    with open(encoder_file, 'rb') as f:
                        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
                        model_data['encoder'] = joblib.load(mm)
                        mm.close()
                
                if len(model_data) == 3:
                    self.model_load_times[prefix] = time.time()
                    return model_data
                else:
                    logger.error(f"❌ Incomplete model data for prefix {prefix}")
                    return None
                    
        except Exception as e:
            logger.error(f"❌ Error loading model for prefix {prefix}: {e}")
            return None
    
    async def load_model_async(self, prefix: str) -> Optional[Dict]:
        """Load model asynchronously with memory mapping"""
        try:
            with performance_monitor(f"model_load_{prefix}", self.metrics):
                model_path = self.models_dir / prefix
                
                if not model_path.exists():
                    return None
                
                # Load model files with memory mapping
                model_data = {}
                
                # Load model
                model_file = model_path / "address_model.joblib"
                if model_file.exists():
                    with open(model_file, 'rb') as f:
                        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
                        model_data['model'] = joblib.load(mm)
                        mm.close()
                
                # Load vectorizer
                vectorizer_file = model_path / "address_vectorizer.joblib"
                if vectorizer_file.exists():
                    with open(vectorizer_file, 'rb') as f:
                        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
                        model_data['vectorizer'] = joblib.load(mm)
                        mm.close()
                
                # Load encoder
                encoder_file = model_path / "address_encoder.joblib"
                if encoder_file.exists():
                    with open(encoder_file, 'rb') as f:
                        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
                        model_data['encoder'] = joblib.load(mm)
                        mm.close()
                
                if len(model_data) == 3:
                    self.model_load_times[prefix] = time.time()
                    return model_data
                else:
                    logger.error(f"❌ Incomplete model data for prefix {prefix}")
                    return None
                    
        except Exception as e:
            logger.error(f"❌ Error loading model for prefix {prefix}: {e}")
            return None
    
    def get_model(self, prefix: str) -> Optional[Dict]:
        """Get model from cache"""
        return self.models_cache.get(prefix)
    
    def is_model_loaded(self, prefix: str) -> bool:
        """Check if model is loaded in memory"""
        return prefix in self.models_cache
    
    def get_model_stats(self) -> Dict:
        """Get model loading statistics"""
        return {
            'total_models': len(self.models_cache),
            'available_prefixes': list(self.models_cache.keys()),
            'memory_usage_mb': sum(
                self._estimate_model_memory(model) 
                for model in self.models_cache.values()
            ) / 1024 / 1024
        }
    
    def _estimate_model_memory(self, model_data: Dict) -> int:
        """Estimate memory usage of a model"""
        try:
            total_size = 0
            for component in model_data.values():
                if hasattr(component, 'nbytes'):
                    total_size += component.nbytes
                elif hasattr(component, '__sizeof__'):
                    total_size += component.__sizeof__()
            return total_size
        except:
            return 0

# Database Connection Pool Manager
class DatabaseManager:
    def __init__(self, database_url: str):
        self.database_url = database_url
        self.engine = None
        self.async_engine = None
        self.session_factory = None
        self.async_session_factory = None
        self.initialize_connections()
    
    def initialize_connections(self):
        """Initialize database connections with pooling"""
        try:
            logger.info(f"🔍 Initializing database connection with URL: {self.database_url}")
            
            # Synchronous engine with connection pooling
            self.engine = create_engine(
                self.database_url,
                poolclass=QueuePool,
                pool_size=20,           # Maintain 20 connections
                max_overflow=30,        # Allow up to 30 additional connections
                pool_pre_ping=True,     # Validate connections before use
                pool_recycle=3600,      # Recycle connections every hour
                pool_timeout=30         # Connection timeout
            )
            
            logger.info("🔍 Database engine created, testing connection...")
            
            # Test connection
            with self.engine.connect() as conn:
                result = conn.execute(text("SELECT 1"))
                logger.info(f"🔍 Connection test result: {result}")
            
            logger.info("✅ Database connection pool initialized successfully")
            
        except Exception as e:
            logger.warning(f"⚠️ Database connection failed: {e}")
            logger.warning("⚠️ Service will start without database functionality")
            self.engine = None
    
    def get_connection(self):
        """Get database connection from pool"""
        return self.engine.connect()
    
    def execute_query(self, query: str, params: tuple = None):
        """Execute query with connection from pool"""
        if not self.engine:
            logger.warning("⚠️ Database not available, skipping query")
            return None
        
        with self.engine.begin() as conn:
            try:
                if params:
                    result = conn.execute(text(query), params)
                else:
                    result = conn.execute(text(query))
                return result
            except Exception as e:
                # The transaction will be automatically rolled back by the context manager
                raise e
    
    def execute_many(self, query: str, params_list: List[tuple]):
        """Execute multiple queries in batch"""
        if not self.engine:
            logger.warning("⚠️ Database not available, skipping batch query")
            return
        
        with self.engine.begin() as conn:
            try:
                conn.execute(text(query), params_list)
                # The transaction will be automatically committed by the context manager
            except Exception as e:
                # The transaction will be automatically rolled back by the context manager
                raise e

# Response Cache Manager
class ResponseCache:
    def __init__(self, maxsize: int = CACHE_MAXSIZE, ttl: int = CACHE_TTL):
        self.cache = TTLCache(maxsize=maxsize, ttl=ttl)
        self.metrics = PerformanceMetrics()
    
    def get_cache_key(self, address: str) -> str:
        """Generate cache key for address"""
        return hashlib.md5(address.encode()).hexdigest()
    
    def get_cached_prediction(self, address: str) -> Optional[Dict]:
        """Get cached prediction if available"""
        key = self.get_cache_key(address)
        result = self.cache.get(key)
        
        if result:
            self.metrics.record_cache_hit()
            logger.debug(f"🎯 Cache HIT for address: {address[:50]}...")
        else:
            self.metrics.record_cache_miss()
            logger.debug(f"❌ Cache MISS for address: {address[:50]}...")
        
        return result
    
    def cache_prediction(self, address: str, prediction: Dict):
        """Cache prediction result"""
        key = self.get_cache_key(address)
        self.cache[key] = prediction
        logger.debug(f"💾 Cached prediction for address: {address[:50]}...")
    
    def get_cache_stats(self) -> Dict:
        """Get cache statistics"""
        return {
            'size': len(self.cache),
            'maxsize': self.cache.maxsize,
            'ttl': self.cache.ttl,
            'hits': self.metrics.cache_hits,
            'misses': self.metrics.cache_misses,
            'hit_rate': self.metrics.cache_hits / (self.metrics.cache_hits + self.metrics.cache_misses) if (self.metrics.cache_hits + self.metrics.cache_misses) > 0 else 0
        }

# Batch Prediction Processor
class BatchPredictor:
    def __init__(self, model_manager: OptimizedModelManager, cache: ResponseCache, db_manager: DatabaseManager):
        self.model_manager = model_manager
        self.cache = cache
        self.db_manager = db_manager
        self.metrics = PerformanceMetrics()
    
    async def process_batch(self, addresses: List[str], shipment_ids: List[str]) -> List[PredictionResponse]:
        """Process multiple addresses in a single batch"""
        start_time = time.time()
        
        with performance_monitor("batch_processing", self.metrics):
            # Process predictions
            predictions = []
            cached_count = 0
            
            for address, shipment_id in zip(addresses, shipment_ids):
                # Check cache first
                cached_result = self.cache.get_cached_prediction(address)
                if cached_result:
                    cached_count += 1
                    predictions.append(PredictionResponse(
                        shipment_id=shipment_id,
                        predicted_cp=cached_result['predicted_cp'],
                        confidence=cached_result['confidence'],
                        processing_time_ms=cached_result['processing_time_ms'],
                        cached=True
                    ))
                    continue
                
                # Process new prediction
                result = await self.predict_single_optimized(address, shipment_id)
                predictions.append(result)
            
            # Batch save to database
            await self.batch_save_predictions(predictions)
            
            total_time = (time.time() - start_time) * 1000
            logger.info(f"🎯 Batch processed {len(addresses)} addresses in {total_time:.2f}ms (cached: {cached_count})")
            
            return predictions
    
    async def predict_single_optimized(self, address: str, shipment_id: str) -> PredictionResponse:
        """Optimized single prediction with caching"""
        start_time = time.time()
        
        # Check cache first
        cached_result = self.cache.get_cached_prediction(address)
        if cached_result:
            return PredictionResponse(
                shipment_id=shipment_id,
                predicted_cp=cached_result['predicted_cp'],
                confidence=cached_result['confidence'],
                processing_time_ms=cached_result['processing_time_ms'],
                cached=True
            )
        
        # Process prediction
        with performance_monitor("prediction", self.metrics):
            result = self.predict_cp_optimized(address)
            
            # Cache the result
            self.cache.cache_prediction(address, {
                'predicted_cp': result['predicted_cp'],
                'confidence': result['confidence'],
                'processing_time_ms': (time.time() - start_time) * 1000
            })
            
            processing_time = (time.time() - start_time) * 1000
            
            return PredictionResponse(
                shipment_id=shipment_id,
                predicted_cp=result['predicted_cp'],
                confidence=result['confidence'],
                processing_time_ms=processing_time,
                cached=False
            )
    
    def predict_cp_optimized(self, address: str) -> Dict:
        """Optimized CP prediction using pre-loaded models"""
        try:
            # Extract pincode
            import re
            pincode_match = re.search(r'\b(\d{6})\b', address)
            if not pincode_match:
                return {'predicted_cp': 'INVALID', 'confidence': 0.95, 'pincode': None}
            
            pincode = pincode_match.group(1)
            prefix = pincode[:2]
            
            # Get model from cache
            model_data = self.model_manager.get_model(prefix)
            if not model_data:
                return {'predicted_cp': 'INVALID', 'confidence': 0.95, 'pincode': pincode}
            
            # Make prediction
            model = model_data['model']
            vectorizer = model_data['vectorizer']
            encoder = model_data['encoder']
            
            # Vectorize address
            features = vectorizer.transform([address])
            
            # Predict
            prediction = model.predict(features)[0]
            confidence = model.predict_proba(features).max()
            
            # Decode prediction
            predicted_cp = encoder.inverse_transform([prediction])[0]
            
            return {
                'predicted_cp': predicted_cp,
                'confidence': confidence,
                'pincode': pincode
            }
            
        except Exception as e:
            logger.error(f"❌ Prediction error: {e}")
            return {'predicted_cp': 'INVALID', 'confidence': 0.95, 'pincode': None}
    
    async def batch_save_predictions(self, predictions: List[PredictionResponse]):
        """Batch save predictions to database"""
        if not predictions:
            return
        
        with performance_monitor("batch_db_save", self.metrics):
            # Prepare batch insert
            insert_query = """
            INSERT INTO prediction_history (shipment_id, address, pincode, predicted_cp, confidence, 
                                  model_version, cached, timestamp, processing_time_ms)
            VALUES (:shipment_id, :address, :pincode, :predicted_cp, :confidence, 
                    :model_version, :cached, :timestamp, :processing_time_ms)
            """
            
            # Extract data for batch insert
            batch_data = []
            for pred in predictions:
                # Note: We don't have address and pincode in the response, 
                # so we'll use placeholder values for this example
                batch_data.append({
                    'shipment_id': str(pred.shipment_id),
                    'address': "address_placeholder",  # In real implementation, pass address
                    'pincode': "pincode_placeholder",  # In real implementation, pass pincode
                    'predicted_cp': str(pred.predicted_cp),
                    'confidence': float(pred.confidence),
                    'model_version': 'v1_optimized',
                    'cached': bool(pred.cached),
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'processing_time_ms': float(pred.processing_time_ms)
                })
            
            # Execute batch insert
            try:
                self.db_manager.execute_many(insert_query, batch_data)
                logger.info(f"💾 Batch saved {len(predictions)} predictions to database")
            except Exception as e:
                logger.error(f"❌ Batch save failed: {e}")
                # Log more details for debugging
                if batch_data:
                    logger.error(f"❌ First prediction data: {batch_data[0]}")
                    logger.error(f"❌ Data types: shipment_id={type(batch_data[0][0])}, confidence={type(batch_data[0][4])}, processing_time_ms={type(batch_data[0][8])}")
                else:
                    logger.error("❌ No batch data to save")
    
    async def save_single_prediction_to_db(self, prediction: PredictionResponse, address: str, pincode: str):
        """Save a single prediction to database"""
        logger.info(f"🔍 Attempting to save prediction for shipment {prediction.shipment_id}")
        
        if not self.db_manager or not self.db_manager.engine:
            logger.warning("⚠️ Database not available, skipping save")
            return
        
        try:
            insert_query = """
            INSERT INTO prediction_history (shipment_id, address, pincode, predicted_cp, confidence, 
                                  model_version, cached, timestamp, processing_time_ms)
            VALUES (:shipment_id, :address, :pincode, :predicted_cp, :confidence, 
                    :model_version, :cached, :timestamp, :processing_time_ms)
            """
            
            data = {
                'shipment_id': str(prediction.shipment_id),
                'address': str(address),
                'pincode': str(pincode),
                'predicted_cp': str(prediction.predicted_cp),
                'confidence': float(prediction.confidence),
                'model_version': 'v1_optimized',
                'cached': bool(prediction.cached),
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'processing_time_ms': float(prediction.processing_time_ms)
            }
            
            logger.info(f"🔍 Executing query with data: {data}")
            result = self.db_manager.execute_query(insert_query, data)
            logger.info(f"💾 Saved single prediction for shipment {prediction.shipment_id}, result: {result}")
            
        except Exception as e:
            logger.error(f"❌ Failed to save single prediction: {e}")
            logger.error(f"❌ Data: {data}")
            import traceback
            logger.error(f"❌ Traceback: {traceback.format_exc()}")

# Initialize FastAPI app
app = FastAPI(
    title="Optimized CP Prediction Service",
    description="High-performance CP prediction service with optimizations",
    version="2.0.0"
)

# Global instances
model_manager = None
cache_manager = None
db_manager = None
batch_predictor = None
performance_metrics = PerformanceMetrics()

@app.on_event("startup")
async def startup_event():
    """Initialize all components on startup"""
    global model_manager, cache_manager, db_manager, batch_predictor
    
    logger.info("🚀 Starting Optimized CP Prediction Service...")
    
    # Initialize components
    model_manager = OptimizedModelManager(MODELS_DIR)
    cache_manager = ResponseCache()
    db_manager = DatabaseManager(DATABASE_URL)
    batch_predictor = BatchPredictor(model_manager, cache_manager, db_manager)
    
    # Start memory monitoring
    asyncio.create_task(memory_monitor())
    
    logger.info("✅ Service initialization completed")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("🛑 Shutting down Optimized CP Prediction Service...")
    
    if model_manager and hasattr(model_manager, 'executor'):
        model_manager.executor.shutdown(wait=True)
    
    logger.info("✅ Service shutdown completed")

async def memory_monitor():
    """Monitor memory usage periodically"""
    while True:
        try:
            performance_metrics.record_memory_usage()
            await asyncio.sleep(60)  # Monitor every minute
        except Exception as e:
            logger.error(f"Memory monitoring error: {e}")

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check with detailed status"""
    try:
        # Check model status
        model_stats = model_manager.get_model_stats() if model_manager else {}
        
        # Check cache status
        cache_stats = cache_manager.get_cache_stats() if cache_manager else {}
        
        # Check database status
        db_status = "healthy"
        try:
            if db_manager and db_manager.engine:
                # Test database connection with a simple query
                result = db_manager.execute_query("SELECT 1")
                if result is None:
                    db_status = "unavailable"
            else:
                db_status = "unavailable"
        except Exception as e:
            logger.warning(f"⚠️ Database health check failed: {e}")
            db_status = "unhealthy"
        
        # Get performance metrics
        perf_stats = performance_metrics.get_stats()
        
        return {
            "status": "healthy",
            "timestamp": time.time(),
            "models": model_stats,
            "cache": cache_stats,
            "database": db_status,
            "performance": perf_stats
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/test-db")
async def test_database():
    """Test database connection and operations directly"""
    from datetime import datetime
    
    try:
        if not db_manager or not db_manager.engine:
            return {"status": "error", "message": "Database manager not available"}
        
        # Test connection
        try:
            result = db_manager.execute_query("SELECT 1")
            connection_status = "success" if result is not None else "failed"
        except Exception as e:
            connection_status = f"error: {e}"
        
        # Test insertion
        try:
            test_data = (
                'TEST_ENDPOINT_001',  # shipment_id
                'Test Address from Endpoint',  # address
                '110001',  # pincode
                'TEST_CP_ENDPOINT',  # predicted_cp
                0.95,  # confidence
                'v1_test',  # model_version
                False,  # cached
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),  # timestamp
                100.0  # processing_time_ms
            )
            
            insert_query = """
            INSERT INTO prediction_history (shipment_id, address, pincode, predicted_cp, confidence, 
                                  model_version, cached, timestamp, processing_time_ms)
            VALUES (:shipment_id, :address, :pincode, :predicted_cp, :confidence, 
                    :model_version, :cached, :timestamp, :processing_time_ms)
            """
            
            test_data_dict = {
                'shipment_id': test_data[0],
                'address': test_data[1],
                'pincode': test_data[2],
                'predicted_cp': test_data[3],
                'confidence': test_data[4],
                'model_version': test_data[5],
                'cached': test_data[6],
                'timestamp': test_data[7],
                'processing_time_ms': test_data[8]
            }
            
            db_manager.execute_query(insert_query, test_data_dict)
            insertion_status = "success"
        except Exception as e:
            insertion_status = f"error: {e}"
        
        # Get current count
        try:
            count_result = db_manager.execute_query("SELECT COUNT(*) FROM prediction_history")
            current_count = count_result.fetchone()[0] if count_result else "unknown"
        except Exception as e:
            current_count = f"error: {e}"
        
        return {
            "status": "completed",
            "connection_test": connection_status,
            "insertion_test": insertion_status,
            "current_count": current_count,
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Database test error: {e}")
        return {"status": "error", "message": str(e)}

# Single prediction endpoint
@app.post("/predict", response_model=PredictionResponse)
async def predict_cp(request: PredictionRequest):
    """Single CP prediction with optimizations"""
    try:
        with performance_monitor("single_prediction", performance_metrics):
            result = await batch_predictor.predict_single_optimized(
                request.address, 
                request.shipment_id
            )
            
            # Save to database (non-blocking)
            # Create a proper prediction record with address and pincode
            prediction_record = PredictionResponse(
                shipment_id=result.shipment_id,
                predicted_cp=result.predicted_cp,
                confidence=result.confidence,
                processing_time_ms=result.processing_time_ms,
                cached=result.cached
            )
            
            # Extract address and pincode from the request
            import re
            pincode_match = re.search(r'\b(\d{6})\b', request.address)
            pincode = pincode_match.group(1) if pincode_match else "unknown"
            
            # Save to database (synchronous for debugging)
            try:
                await batch_predictor.save_single_prediction_to_db(prediction_record, request.address, pincode)
                logger.info(f"✅ Database save completed for shipment {request.shipment_id}")
            except Exception as e:
                logger.error(f"❌ Database save failed for shipment {request.shipment_id}: {e}")
            
            return result
            
    except Exception as e:
        logger.error(f"❌ Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Batch prediction endpoint
@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_cp_batch(request: BatchPredictionRequest):
    """Batch CP prediction with optimizations"""
    try:
        if len(request.addresses) != len(request.shipment_ids):
            raise HTTPException(status_code=400, detail="Addresses and shipment IDs must have same length")
        
        if len(request.addresses) > 1000:
            raise HTTPException(status_code=400, detail="Maximum batch size is 1000")
        
        start_time = time.time()
        
        # Process in batches
        all_predictions = []
        for i in range(0, len(request.addresses), BATCH_SIZE):
            batch_addresses = request.addresses[i:i + BATCH_SIZE]
            batch_shipment_ids = request.shipment_ids[i:i + BATCH_SIZE]
            
            batch_predictions = await batch_predictor.process_batch(
                batch_addresses, 
                batch_shipment_ids
            )
            all_predictions.extend(batch_predictions)
        
        total_time = (time.time() - start_time) * 1000
        
        return BatchPredictionResponse(
            predictions=all_predictions,
            total_processing_time_ms=total_time,
            batch_size=len(request.addresses)
        )
        
    except Exception as e:
        logger.error(f"❌ Batch prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Performance metrics endpoint
@app.get("/metrics")
async def get_metrics():
    """Get detailed performance metrics"""
    try:
        return {
            "timestamp": time.time(),
            "performance": performance_metrics.get_stats(),
            "models": model_manager.get_model_stats() if model_manager else {},
            "cache": cache_manager.get_cache_stats() if cache_manager else {},
            "system": {
                "cpu_percent": psutil.cpu_percent(),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_usage": psutil.disk_usage('/').percent
            }
        }
    except Exception as e:
        logger.error(f"Metrics retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Cache management endpoints
@app.post("/cache/clear")
async def clear_cache():
    """Clear response cache"""
    try:
        if cache_manager:
            cache_manager.cache.clear()
            logger.info("🗑️ Cache cleared")
            return {"message": "Cache cleared successfully"}
        else:
            raise HTTPException(status_code=500, detail="Cache manager not initialized")
    except Exception as e:
        logger.error(f"Cache clear failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/cache/stats")
async def get_cache_stats():
    """Get cache statistics"""
    try:
        if cache_manager:
            return cache_manager.get_cache_stats()
        else:
            raise HTTPException(status_code=500, detail="Cache manager not initialized")
    except Exception as e:
        logger.error(f"Cache stats retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Model management endpoints
@app.get("/models/status")
async def get_models_status():
    """Get model loading status"""
    try:
        if model_manager:
            return model_manager.get_model_stats()
        else:
            raise HTTPException(status_code=500, detail="Model manager not initialized")
    except Exception as e:
        logger.error(f"Model status retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/models/reload")
async def reload_models():
    """Reload all models"""
    try:
        if model_manager:
            # Clear existing cache
            model_manager.models_cache.clear()
            
            # Pre-load all models again
            model_manager.preload_all_models()
            
            logger.info("🔄 Models reloaded successfully")
            return {"message": "Models reloaded successfully"}
        else:
            raise HTTPException(status_code=500, detail="Model manager not initialized")
    except Exception as e:
        logger.error(f"Model reload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(
        "optimized_main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,  # Disable reload for production
        workers=1      # Single worker for now, can be increased with load balancer
    )
