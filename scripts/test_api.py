#!/usr/bin/env python3
"""
API Testing Script for CP Prediction Service
Tests all endpoints with various scenarios
"""

import asyncio
import httpx
import json
import time
from typing import List, Dict, Any


class CPPredictionTester:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=30.0)
    
    async def test_health_check(self) -> Dict[str, Any]:
        """Test health check endpoint"""
        print("🔍 Testing health check...")
        response = await self.client.get(f"{self.base_url}/health")
        assert response.status_code == 200
        result = response.json()
        print(f"✅ Health check passed: {result['status']}")
        return result
    
    async def test_single_prediction(self) -> Dict[str, Any]:
        """Test single prediction endpoint"""
        print("🔍 Testing single prediction...")
        
        test_data = {
            "address": "123 Main Street, Mumbai 280001",
            "shipment_id": "TEST_001"
        }
        
        response = await self.client.post(
            f"{self.base_url}/predict",
            json=test_data
        )
        
        assert response.status_code == 200
        result = response.json()
        
        print(f"✅ Single prediction successful:")
        print(f"   Predicted CP: {result['predicted_cp']}")
        print(f"   Confidence: {result['confidence']:.3f}")
        print(f"   Processing time: {result['processing_time_ms']:.2f}ms")
        
        return result
    
    async def test_batch_prediction(self) -> Dict[str, Any]:
        """Test batch prediction endpoint"""
        print("🔍 Testing batch prediction...")
        
        test_requests = [
            {"address": "456 Oak Avenue, Delhi 280002", "shipment_id": "BATCH_001"},
            {"address": "789 Pine Road, Bangalore 280003", "shipment_id": "BATCH_002"},
            {"address": "321 Elm Street, Chennai 280004", "shipment_id": "BATCH_003"},
        ]
        
        batch_data = {"requests": test_requests}
        
        response = await self.client.post(
            f"{self.base_url}/predict/batch",
            json=batch_data
        )
        
        assert response.status_code == 200
        result = response.json()
        
        print(f"✅ Batch prediction successful:")
        print(f"   Total requests: {result['total_requests']}")
        print(f"   Successful: {result['successful_predictions']}")
        print(f"   Failed: {result['failed_predictions']}")
        print(f"   Total time: {result['total_processing_time_ms']:.2f}ms")
        
        return result
    
    async def test_error_handling(self):
        """Test error handling"""
        print("🔍 Testing error handling...")
        
        # Test invalid address (no pincode)
        invalid_data = {
            "address": "Invalid address without pincode",
            "shipment_id": "ERROR_TEST_001"
        }
        
        response = await self.client.post(
            f"{self.base_url}/predict",
            json=invalid_data
        )
        
        assert response.status_code == 400
        print("✅ Error handling for invalid address works correctly")
        
        # Test missing fields
        response = await self.client.post(
            f"{self.base_url}/predict",
            json={"address": "Some address 123456"}  # Missing shipment_id
        )
        
        assert response.status_code == 422
        print("✅ Error handling for missing fields works correctly")
    
    async def test_caching(self):
        """Test caching functionality"""
        print("🔍 Testing caching...")
        
        test_data = {
            "address": "Cache Test Street, Mumbai 280001",
            "shipment_id": "CACHE_TEST_001"
        }
        
        # First request (cache miss)
        start_time = time.time()
        response1 = await self.client.post(f"{self.base_url}/predict", json=test_data)
        first_time = time.time() - start_time
        
        assert response1.status_code == 200
        result1 = response1.json()
        
        # Second request (should be cache hit)
        start_time = time.time()
        response2 = await self.client.post(f"{self.base_url}/predict", json=test_data)
        second_time = time.time() - start_time
        
        assert response2.status_code == 200
        result2 = response2.json()
        
        # Cache hit should be faster
        print(f"✅ Caching test:")
        print(f"   First request: {first_time*1000:.2f}ms")
        print(f"   Second request: {second_time*1000:.2f}ms")
        print(f"   Cached result: {result2.get('cached', False)}")
    
    async def test_metrics_endpoint(self):
        """Test metrics endpoint"""
        print("🔍 Testing metrics endpoint...")
        
        response = await self.client.get(f"{self.base_url}/metrics")
        assert response.status_code == 200
        
        result = response.json()
        print(f"✅ Metrics endpoint working:")
        print(f"   Total predictions: {result['total_predictions']}")
        print(f"   Success rate: {result['successful_predictions']}/{result['total_predictions']}")
    
    async def test_model_status(self):
        """Test model status endpoint"""
        print("🔍 Testing model status endpoint...")
        
        response = await self.client.get(f"{self.base_url}/models/status")
        assert response.status_code == 200
        
        result = response.json()
        print(f"✅ Model status endpoint working:")
        print(f"   Models loaded: {result['total_models_loaded']}")
        print(f"   Available prefixes: {result['available_prefixes']}")
    
    async def run_all_tests(self):
        """Run all tests"""
        print("🚀 Starting CP Prediction Service API Tests")
        print("=" * 50)
        
        try:
            await self.test_health_check()
            await self.test_single_prediction()
            await self.test_batch_prediction()
            await self.test_error_handling()
            await self.test_caching()
            await self.test_metrics_endpoint()
            await self.test_model_status()
            
            print("\n" + "=" * 50)
            print("🎉 All tests passed successfully!")
            
        except Exception as e:
            print(f"\n❌ Test failed: {str(e)}")
            raise
        finally:
            await self.client.aclose()


async def main():
    """Main test function"""
    tester = CPPredictionTester()
    await tester.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())
