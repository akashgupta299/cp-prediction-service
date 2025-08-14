#!/usr/bin/env python3
"""
Script to test 1000 addresses from parquet file and calculate accuracy
"""

import pandas as pd
import requests
import json
import time
import os
from datetime import datetime
import asyncio
import aiohttp
from typing import List, Dict
import numpy as np

# Configuration
BASE_URL = "http://localhost:8000"
PARQUET_FILE_PATH = "/Users/akash/Documents/cp_pred_github/cp-prediction-service/notebook/pan_India_test_data_2025-07-23.parquet"
BATCH_SIZE = 50  # Process in batches to avoid overwhelming the service
MAX_CONCURRENT_REQUESTS = 10

class AccuracyTester:
    def __init__(self, base_url: str, parquet_file_path: str):
        self.base_url = base_url
        self.parquet_file_path = parquet_file_path
        self.results = []
        self.session = None
        
    async def __aenter__(self):
        connector = aiohttp.TCPConnector(limit=100, limit_per_host=100)
        timeout = aiohttp.ClientTimeout(total=60)
        self.session = aiohttp.ClientSession(connector=connector, timeout=timeout)
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    def load_addresses_from_parquet(self, num_addresses: int = 1000) -> pd.DataFrame:
        """Load addresses from parquet file"""
        print(f"📖 Loading {num_addresses} addresses from parquet file...")
        
        if not os.path.exists(self.parquet_file_path):
            raise FileNotFoundError(f"Parquet file not found: {self.parquet_file_path}")
        
        # Read parquet file
        df = pd.read_parquet(self.parquet_file_path)
        print(f"📊 Loaded {len(df)} total addresses from parquet file")
        
        # Check required columns
        required_columns = ['full_address', 'cp_code']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Take first 1000 addresses
        df_subset = df.head(num_addresses).copy()
        print(f"🎯 Selected {len(df_subset)} addresses for testing")
        
        # Generate shipment IDs
        df_subset['shipment_id'] = [f"ACCURACY_TEST_{i}_{int(time.time())}" for i in range(len(df_subset))]
        
        return df_subset
    
    async def make_prediction_request(self, address: str, shipment_id: str) -> Dict:
        """Make a single prediction request"""
        payload = {
            "address": address,
            "shipment_id": shipment_id
        }
        
        try:
            async with self.session.post(f"{self.base_url}/predict", json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    return {
                        'success': True,
                        'data': data,
                        'error': None
                    }
                else:
                    error_text = await response.text()
                    return {
                        'success': False,
                        'data': None,
                        'error': f"HTTP {response.status}: {error_text}"
                    }
        except Exception as e:
            return {
                'success': False,
                'data': None,
                'error': str(e)
            }
    
    async def process_batch(self, batch_df: pd.DataFrame) -> List[Dict]:
        """Process a batch of addresses"""
        tasks = []
        
        for _, row in batch_df.iterrows():
            task = asyncio.create_task(
                self.make_prediction_request(row['full_address'], row['shipment_id'])
            )
            tasks.append(task)
        
        # Wait for all requests in batch to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        batch_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                batch_results.append({
                    'shipment_id': batch_df.iloc[i]['shipment_id'],
                    'address': batch_df.iloc[i]['full_address'],
                    'actual_cp': batch_df.iloc[i]['cp_code'],
                    'success': False,
                    'error': str(result)
                })
            else:
                batch_results.append({
                    'shipment_id': batch_df.iloc[i]['shipment_id'],
                    'address': batch_df.iloc[i]['full_address'],
                    'actual_cp': batch_df.iloc[i]['cp_code'],
                    'success': result['success'],
                    'predicted_cp': result['data']['predicted_cp'] if result['data'] else None,
                    'confidence': result['data']['confidence'] if result['data'] else None,
                    'error': result['error']
                })
        
        return batch_results
    
    async def run_accuracy_test(self, num_addresses: int = 1000) -> List[Dict]:
        """Run the complete accuracy test"""
        print(f"🚀 Starting accuracy test with {num_addresses} addresses...")
        
        # Load addresses from parquet
        df = self.load_addresses_from_parquet(num_addresses)
        
        # Process in batches
        all_results = []
        total_batches = (len(df) + BATCH_SIZE - 1) // BATCH_SIZE
        
        for batch_num in range(0, len(df), BATCH_SIZE):
            batch_df = df.iloc[batch_num:batch_num + BATCH_SIZE]
            batch_num_display = (batch_num // BATCH_SIZE) + 1
            
            print(f"📦 Processing batch {batch_num_display}/{total_batches} ({len(batch_df)} addresses)...")
            
            batch_results = await self.process_batch(batch_df)
            all_results.extend(batch_results)
            
            # Small delay between batches to avoid overwhelming the service
            if batch_num + BATCH_SIZE < len(df):
                await asyncio.sleep(0.1)
        
        print(f"✅ Completed accuracy test! Processed {len(all_results)} addresses")
        return all_results
    
    def calculate_accuracy(self, results: List[Dict]) -> Dict:
        """Calculate accuracy metrics from results"""
        print("📊 Calculating accuracy metrics...")
        
        # Filter successful predictions
        successful_results = [r for r in results if r['success'] and r['predicted_cp']]
        failed_results = [r for r in results if not r['success']]
        
        print(f"📈 Successful predictions: {len(successful_results)}")
        print(f"❌ Failed predictions: {len(failed_results)}")
        
        if not successful_results:
            return {
                'total_requests': len(results),
                'successful_requests': 0,
                'failed_requests': len(failed_results),
                'success_rate': 0.0,
                'accuracy': 0.0,
                'error_rate': 1.0
            }
        
        # Calculate accuracy
        correct_predictions = 0
        total_predictions = len(successful_results)
        
        for result in successful_results:
            if result['predicted_cp'] == result['actual_cp']:
                correct_predictions += 1
        
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
        error_rate = 1.0 - accuracy
        
        # Calculate confidence statistics
        confidences = [r['confidence'] for r in successful_results if r['confidence'] is not None]
        avg_confidence = np.mean(confidences) if confidences else 0.0
        
        # Analyze by CP code
        cp_analysis = {}
        for result in successful_results:
            actual_cp = result['actual_cp']
            predicted_cp = result['predicted_cp']
            
            if actual_cp not in cp_analysis:
                cp_analysis[actual_cp] = {
                    'count': 0,
                    'correct': 0,
                    'incorrect': 0,
                    'confidence_scores': []
                }
            
            cp_analysis[actual_cp]['count'] += 1
            
            if actual_cp == predicted_cp:
                cp_analysis[actual_cp]['correct'] += 1
            else:
                cp_analysis[actual_cp]['incorrect'] += 1
            
            if result['confidence'] is not None:
                cp_analysis[actual_cp]['confidence_scores'].append(result['confidence'])
        
        # Calculate accuracy for each CP
        for cp in cp_analysis:
            cp_analysis[cp]['accuracy'] = cp_analysis[cp]['correct'] / cp_analysis[cp]['count']
            cp_analysis[cp]['avg_confidence'] = np.mean(cp_analysis[cp]['confidence_scores']) if cp_analysis[cp]['confidence_scores'] else 0.0
        
        return {
            'total_requests': len(results),
            'successful_requests': len(successful_results),
            'failed_requests': len(failed_results),
            'success_rate': len(successful_results) / len(results),
            'accuracy': accuracy,
            'error_rate': error_rate,
            'correct_predictions': correct_predictions,
            'incorrect_predictions': total_predictions - correct_predictions,
            'avg_confidence': avg_confidence,
            'cp_analysis': cp_analysis
        }
    
    def print_results(self, results: List[Dict], metrics: Dict):
        """Print detailed results"""
        print("\n" + "="*80)
        print("🎯 ACCURACY TEST RESULTS")
        print("="*80)
        
        print(f"📊 Total Requests: {metrics['total_requests']}")
        print(f"✅ Successful: {metrics['successful_requests']}")
        print(f"❌ Failed: {metrics['failed_requests']}")
        print(f"📈 Success Rate: {metrics['success_rate']*100:.2f}%")
        print(f"🎯 Overall Accuracy: {metrics['accuracy']*100:.2f}%")
        print(f"❌ Error Rate: {metrics['error_rate']*100:.2f}%")
        print(f"🎯 Correct Predictions: {metrics['correct_predictions']}")
        print(f"❌ Incorrect Predictions: {metrics['incorrect_predictions']}")
        print(f"💪 Average Confidence: {metrics['avg_confidence']:.4f}")
        
        print("\n📊 Accuracy by CP Code:")
        print("-" * 60)
        print(f"{'CP Code':<15} {'Count':<8} {'Correct':<8} {'Accuracy':<10} {'Avg Conf':<10}")
        print("-" * 60)
        
        # Sort by accuracy (descending)
        sorted_cps = sorted(metrics['cp_analysis'].items(), 
                           key=lambda x: x[1]['accuracy'], reverse=True)
        
        for cp, stats in sorted_cps:
            print(f"{cp:<15} {stats['count']:<8} {stats['correct']:<8} "
                  f"{stats['accuracy']*100:<9.1f}% {stats['avg_confidence']:<9.4f}")
        
        # Show some examples of correct and incorrect predictions
        print("\n🎯 Sample Correct Predictions:")
        print("-" * 60)
        correct_examples = [r for r in results if r['success'] and r['predicted_cp'] == r['actual_cp']][:5]
        for i, example in enumerate(correct_examples, 1):
            print(f"{i}. Address: {example['address'][:50]}...")
            print(f"   Actual: {example['actual_cp']}, Predicted: {example['predicted_cp']}, "
                  f"Confidence: {example['confidence']:.4f}")
        
        print("\n❌ Sample Incorrect Predictions:")
        print("-" * 60)
        incorrect_examples = [r for r in results if r['success'] and r['predicted_cp'] != r['actual_cp']][:5]
        for i, example in enumerate(incorrect_examples, 1):
            print(f"{i}. Address: {example['address'][:50]}...")
            print(f"   Actual: {example['actual_cp']}, Predicted: {example['predicted_cp']}, "
                  f"Confidence: {example['confidence']:.4f}")
        
        if metrics['failed_requests'] > 0:
            print("\n💥 Sample Failed Requests:")
            print("-" * 60)
            failed_examples = [r for r in results if not r['success']][:5]
            for i, example in enumerate(failed_examples, 1):
                print(f"{i}. Address: {example['address'][:50]}...")
                print(f"   Error: {example['error']}")

async def main():
    """Main function"""
    print("🚀 Starting 1000 Address Accuracy Test")
    print("=" * 50)
    
    # Check if service is running
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=10)
        if response.status_code == 200:
            print("✅ Service is running and healthy")
        else:
            print(f"⚠️ Service responded with status: {response.status_code}")
    except Exception as e:
        print(f"❌ Cannot connect to service at {BASE_URL}: {e}")
        print("💡 Make sure the service is running: uvicorn app.optimized_main:app --host 0.0.0.0 --port 8000")
        return
    
    # Run accuracy test
    async with AccuracyTester(BASE_URL, PARQUET_FILE_PATH) as tester:
        try:
            results = await tester.run_accuracy_test(1000)
            
            # Calculate accuracy
            metrics = tester.calculate_accuracy(results)
            
            # Print results
            tester.print_results(results, metrics)
            
            # Save results
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Save detailed results
            results_file = f'accuracy_test_1000_results_{timestamp}.json'
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            # Save metrics
            metrics_file = f'accuracy_test_1000_metrics_{timestamp}.json'
            with open(metrics_file, 'w') as f:
                json.dump(metrics, f, indent=2, default=str)
            
            # Save to CSV
            results_df = pd.DataFrame(results)
            csv_file = f'accuracy_test_1000_results_{timestamp}.csv'
            results_df.to_csv(csv_file, index=False)
            
            print(f"\n💾 Results saved:")
            print(f"   Detailed: {results_file}")
            print(f"   Metrics: {metrics_file}")
            print(f"   CSV: {csv_file}")
            
        except Exception as e:
            print(f"❌ Error during accuracy test: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
