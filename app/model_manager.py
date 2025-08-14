"""Advanced model management with lazy loading and memory optimization"""
import gc
import time
import joblib
import psutil
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from functools import lru_cache
from cachetools import TTLCache
import hashlib
import random
from .config import settings

logger = logging.getLogger(__name__)


class ModelManager:
    """Advanced model manager with lazy loading and memory optimization"""
    
    def __init__(self):
        self._models: Dict[str, Dict[str, Any]] = {}
        self._model_access_times: Dict[str, float] = {}
        self._model_memory_usage: Dict[str, float] = {}
        self.model_dir = Path(settings.model_dir).resolve()
        
        # Initialize prediction cache
        self.prediction_cache = TTLCache(
            maxsize=settings.cache_max_size,
            ttl=settings.cache_ttl_seconds
        )
        
        # Load initial models
        self._load_initial_models()
    
    def _load_initial_models(self):
        """Load a few models at startup"""
        if not self.model_dir.exists():
            logger.warning(f"Model directory {self.model_dir} does not exist")
            return
        
        # Load first few models to warm up
        loaded_count = 0
        for sub_dir in self.model_dir.iterdir():
            if sub_dir.is_dir() and loaded_count < 3:  # Load first 3 models
                if self._load_model(sub_dir.name):
                    loaded_count += 1
        
        logger.info(f"Loaded {loaded_count} models at startup")
    
    def get_model(self, prefix: str) -> Optional[Dict[str, Any]]:
        """Get model with lazy loading and LRU eviction"""
        # Update access time if model is already loaded
        if prefix in self._models:
            self._model_access_times[prefix] = time.time()
            logger.debug(f"Model cache hit for prefix: {prefix}")
            return self._models[prefix]
        
        # Try to load the model
        if self._load_model(prefix):
            return self._models[prefix]
        
        return None
    
    def _load_model(self, prefix: str) -> bool:
        """Load a specific model"""
        model_path = self.model_dir / prefix
        if not model_path.exists():
            logger.warning(f"Model directory does not exist: {model_path}")
            return False
        
        try:
            # Check if we need to evict models
            if len(self._models) >= settings.max_models_in_memory:
                self._evict_lru_model()
            
            logger.info(f"Loading model for prefix: {prefix}")
            start_time = time.time()
            
            # Load model components
            model = joblib.load(model_path / "address_model.joblib")
            vectorizer = joblib.load(model_path / "address_vectorizer.joblib")
            encoder = joblib.load(model_path / "address_encoder.joblib")
            
            # Store model
            self._models[prefix] = {
                "model": model,
                "vectorizer": vectorizer,
                "encoder": encoder,
                "loaded_at": time.time()
            }
            
            self._model_access_times[prefix] = time.time()
            
            # Calculate memory usage
            load_time = time.time() - start_time
            memory_usage = self._estimate_model_memory_usage(prefix)
            self._model_memory_usage[prefix] = memory_usage
            
            logger.info(
                f"Successfully loaded model for prefix: {prefix} "
                f"(Load time: {load_time:.2f}s, Memory: {memory_usage:.2f}MB)"
            )
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model for prefix {prefix}: {str(e)}")
            return False
    
    def _evict_lru_model(self):
        """Evict least recently used model"""
        if not self._models:
            return
        
        # Find LRU model
        lru_prefix = min(
            self._model_access_times.keys(),
            key=lambda k: self._model_access_times[k]
        )
        
        memory_freed = self._model_memory_usage.get(lru_prefix, 0)
        logger.info(f"Evicting LRU model: {lru_prefix} (Memory freed: {memory_freed:.2f}MB)")
        
        # Remove model
        del self._models[lru_prefix]
        del self._model_access_times[lru_prefix]
        if lru_prefix in self._model_memory_usage:
            del self._model_memory_usage[lru_prefix]
        
        # Force garbage collection
        gc.collect()
    
    def _estimate_model_memory_usage(self, prefix: str) -> float:
        """Estimate memory usage of a model in MB"""
        try:
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except Exception:
            return 0.0
    
    def get_cache_key(self, address: str) -> str:
        """Generate cache key from address"""
        return hashlib.md5(address.lower().strip().encode()).hexdigest()
    
    def get_cached_prediction(self, address: str) -> Optional[Dict[str, Any]]:
        """Get cached prediction result"""
        cache_key = self.get_cache_key(address)
        return self.prediction_cache.get(cache_key)
    
    def cache_prediction(self, address: str, result: Dict[str, Any]):
        """Cache prediction result"""
        cache_key = self.get_cache_key(address)
        self.prediction_cache[cache_key] = result
        logger.debug(f"Cached prediction for address hash: {cache_key[:8]}...")
    
    def get_model_for_ab_testing(self, prefix: str) -> Optional[Dict[str, Any]]:
        """Get model with A/B testing support"""
        if not settings.enable_ab_testing:
            return self.get_model(prefix)
        
        # Simple A/B testing logic
        if random.random() < settings.model_a_weight:
            # Use model A (current model)
            model = self.get_model(prefix)
            if model:
                model["ab_variant"] = "A"
            return model
        else:
            # Use model B (could be a different version or fallback)
            model = self.get_model(prefix)
            if model:
                model["ab_variant"] = "B"
            return model
    
    def get_status(self) -> Dict[str, Any]:
        """Get model manager status"""
        return {
            "models_loaded": len(self._models),
            "available_prefixes": list(self._models.keys()),
            "model_access_times": self._model_access_times.copy(),
            "memory_usage_per_model": self._model_memory_usage.copy(),
            "cache_size": len(self.prediction_cache),
            "cache_info": {
                "maxsize": self.prediction_cache.maxsize,
                "ttl": self.prediction_cache.ttl,
                "currsize": self.prediction_cache.currsize
            }
        }
    
    def clear_cache(self):
        """Clear prediction cache"""
        self.prediction_cache.clear()
        logger.info("Prediction cache cleared")
    
    def warm_up_models(self, prefixes: list):
        """Pre-load specified models"""
        for prefix in prefixes:
            if prefix not in self._models:
                self._load_model(prefix)


# Global model manager instance
model_manager = ModelManager()
