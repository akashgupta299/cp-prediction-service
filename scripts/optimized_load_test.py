#!/usr/bin/env python3
"""
Optimized Load Testing Script for the Enhanced CP Prediction Service
Tests all the new optimizations: caching, batch processing, connection pooling, etc.
"""

import asyncio
import aiohttp
import time
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import numpy as np
from collections import defaultdict
import random
import argparse
import sys
import statistics
from typing import List

# Configuration
BASE_URL = "http://localhost:8000"
TEST_ADDRESSES = [
    "123 Main Street, Delhi, Delhi 110001",
    "456 Park Avenue, Delhi, Delhi 120001", 
    "789 MG Road, Delhi, Delhi 130001",
    "321 Commercial Street, Delhi, Delhi 140001",
    "654 Brigade Road, Delhi, Delhi 150001",
    "987 Linking Road, Delhi, Delhi 160001",
    "147 FC Road, Delhi, Delhi 170001",
    "258 Residency Road, Delhi, Delhi 180001",
    "369 Civil Lines, Delhi, Delhi 190001",
    "741 Mall Road, Delhi, Delhi 200001",
    "852 Station Road, Delhi, Delhi 210001",
    "963 Airport Road, Delhi, Delhi 220001",
    "159 Sector 17, Delhi, Delhi 230001",
    "357 IT Park, Delhi, Delhi 240001",
    "486 Electronic City, Delhi, Delhi 250001",
    "624 Banjara Hills, Delhi, Delhi 260001",
    "793 Koramangala, Delhi, Delhi 270001",
    "135 Andheri West, Delhi, Delhi 280001",
    "246 Connaught Place, Delhi, Delhi 300001",
    "579 Park Street, Delhi, Delhi 310001",
    "ABC Street, Delhi, Delhi 320001",
    "XYZ Avenue, Delhi, Delhi 330001",
    "PQR Road, Delhi, Delhi 340001",
    "LMN Colony, Delhi, Delhi 360001",
    "RST Nagar, Delhi, Delhi 370001",
    "UVW Park, Delhi, Delhi 380001",
    "GHI Complex, Delhi, Delhi 390001",
    "JKL Apartment, Delhi, Delhi 410001",
    "DEF Building, Delhi, Delhi 420001",
    "MNO Tower, Delhi, Delhi 430001"
]

class OptimizedLoadTester:
    def __init__(self, base_url=BASE_URL):
        self.base_url = base_url
        self.results = []
        self.session = None
        self.cache_performance = defaultdict(list)
        self.batch_performance = defaultdict(list)
        
    async def __aenter__(self):
        connector = aiohttp.TCPConnector(limit=200, limit_per_host=200)
        timeout = aiohttp.ClientTimeout(total=30)
        self.session = aiohttp.ClientSession(connector=connector, timeout=timeout)
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def test_single_prediction(self, address: str, shipment_id: str, qps_level: int):
        """Test single prediction endpoint"""
        start_time = time.time()
        
        payload = {
            "address": address,
            "shipment_id": shipment_id
        }
        
        try:
            async with self.session.post(f"{self.base_url}/predict", json=payload) as response:
                end_time = time.time()
                response_time = (end_time - start_time) * 1000
                
                if response.status == 200:
                    data = await response.json()
                    success = True
                    error = None
                    cached = data.get('cached', False)
                    
                    # Record cache performance
                    self.cache_performance[qps_level].append({
                        'cached': cached,
                        'response_time': response_time
                    })
                    
                else:
                    data = None
                    success = False
                    error = f"HTTP {response.status}"
                    cached = False
                    
        except Exception as e:
            end_time = time.time()
            response_time = (end_time - start_time) * 1000
            success = False
            error = str(e)
            data = None
            cached = False
        
        result = {
            'timestamp': start_time,
            'qps_level': qps_level,
            'response_time_ms': response_time,
            'success': success,
            'error': error,
            'shipment_id': shipment_id,
            'address': address[:50] + '...' if len(address) > 50 else address,
            'cached': cached,
            'endpoint': 'single'
        }
        
        self.results.append(result)
        return result
    
    async def test_batch_prediction(self, addresses: List[str], shipment_ids: List[str], qps_level: int):
        """Test batch prediction endpoint"""
        start_time = time.time()
        
        payload = {
            "addresses": addresses,
            "shipment_ids": shipment_ids
        }
        
        try:
            async with self.session.post(f"{self.base_url}/predict/batch", json=payload) as response:
                end_time = time.time()
                response_time = (end_time - start_time) * 1000
                
                if response.status == 200:
                    data = await response.json()
                    success = True
                    error = None
                    batch_size = data.get('batch_size', 0)
                    total_time = data.get('total_processing_time_ms', 0)
                    
                    # Record batch performance
                    self.batch_performance[qps_level].append({
                        'batch_size': batch_size,
                        'total_time': total_time,
                        'avg_per_request': total_time / batch_size if batch_size > 0 else 0
                    })
                    
                else:
                    data = None
                    success = False
                    error = f"HTTP {response.status}"
                    batch_size = 0
                    total_time = 0
                    
        except Exception as e:
            end_time = time.time()
            response_time = (end_time - start_time) * 1000
            success = False
            error = str(e)
            data = None
            batch_size = 0
            total_time = 0
        
        result = {
            'timestamp': start_time,
            'qps_level': qps_level,
            'response_time_ms': response_time,
            'success': success,
            'error': error,
            'shipment_id': f"BATCH_{qps_level}_{int(start_time)}",
            'address': f"Batch of {len(addresses)} addresses",
            'cached': False,
            'endpoint': 'batch',
            'batch_size': batch_size,
            'total_processing_time_ms': total_time
        }
        
        self.results.append(result)
        return result
    
    async def test_cache_performance(self, qps_level: int, duration_seconds: int):
        """Test cache performance by repeating addresses"""
        print(f"🧪 Testing cache performance at {qps_level} QPS for {duration_seconds} seconds...")
        
        interval = 1.0 / qps_level
        start_time = time.time()
        end_time = start_time + duration_seconds
        
        tasks = []
        request_count = 0
        
        # Use a smaller set of addresses to test caching
        cache_test_addresses = TEST_ADDRESSES[:10]
        
        while time.time() < end_time:
            # Alternate between single and batch requests
            if request_count % 3 == 0:  # Every 3rd request is batch
                # Batch request
                batch_size = random.randint(5, 10)
                batch_addresses = random.sample(cache_test_addresses, batch_size)
                batch_shipment_ids = [f"CACHE_BATCH_{qps_level}_{i}_{int(time.time())}" for i in range(batch_size)]
                
                task = asyncio.create_task(
                    self.test_batch_prediction(batch_addresses, batch_shipment_ids, qps_level)
                )
            else:
                # Single request
                address = random.choice(cache_test_addresses)
                shipment_id = f"CACHE_SINGLE_{qps_level}_{request_count}_{int(time.time())}"
                
                task = asyncio.create_task(
                    self.test_single_prediction(address, shipment_id, qps_level)
                )
            
            tasks.append(task)
            request_count += 1
            
            # Wait for next request interval
            next_request_time = start_time + (request_count * interval)
            current_time = time.time()
            
            if next_request_time > current_time:
                await asyncio.sleep(next_request_time - current_time)
        
        # Wait for all pending requests to complete
        print(f"⏳ Waiting for {len(tasks)} cache test requests to complete...")
        await asyncio.gather(*tasks, return_exceptions=True)
        
        # Calculate cache performance metrics
        cache_results = [r for r in self.results if r['qps_level'] == qps_level and r['endpoint'] == 'single']
        cache_hits = len([r for r in cache_results if r['cached']])
        cache_misses = len(cache_results) - cache_hits
        
        if cache_results:
            response_times = [r['response_time_ms'] for r in cache_results if r['success']]
            if response_times:
                avg_response_time = np.mean(response_times)
                p95_response_time = np.percentile(response_times, 95)
                p99_response_time = np.percentile(response_times, 99)
            else:
                avg_response_time = p95_response_time = p99_response_time = 0
        else:
            avg_response_time = p95_response_time = p99_response_time = 0
        
        print(f"✅ Cache Test {qps_level} QPS: {cache_hits} hits, {cache_misses} misses, "
              f"avg: {avg_response_time:.1f}ms, p95: {p95_response_time:.1f}ms, p99: {p99_response_time:.1f}ms")
        
        return {
            'qps': qps_level,
            'total_requests': len(cache_results),
            'cache_hits': cache_hits,
            'cache_misses': cache_misses,
            'cache_hit_rate': cache_hits / len(cache_results) if cache_results else 0,
            'avg_response_time': avg_response_time,
            'p95_response_time': p95_response_time,
            'p99_response_time': p99_response_time
        }
    
    async def run_qps_level(self, qps: int, duration_seconds: int):
        """Run load test at specific QPS level"""
        print(f"🚀 Starting load test at {qps} QPS for {duration_seconds} seconds...")
        
        interval = 1.0 / qps
        start_time = time.time()
        end_time = start_time + duration_seconds
        
        tasks = []
        request_count = 0
        
        while time.time() < end_time:
            # Alternate between single and batch requests
            if request_count % 4 == 0:  # Every 4th request is batch
                # Batch request
                batch_size = random.randint(5, 15)
                batch_addresses = random.sample(TEST_ADDRESSES, batch_size)
                batch_shipment_ids = [f"LOAD_BATCH_{qps}_{i}_{int(time.time())}" for i in range(batch_size)]
                
                task = asyncio.create_task(
                    self.test_batch_prediction(batch_addresses, batch_shipment_ids, qps)
                )
            else:
                # Single request
                address = random.choice(TEST_ADDRESSES)
                shipment_id = f"LOAD_SINGLE_{qps}_{request_count}_{int(time.time())}"
                
                task = asyncio.create_task(
                    self.test_single_prediction(address, shipment_id, qps)
                )
            
            tasks.append(task)
            request_count += 1
            
            # Wait for next request interval
            next_request_time = start_time + (request_count * interval)
            current_time = time.time()
            
            if next_request_time > current_time:
                await asyncio.sleep(next_request_time - current_time)
        
        # Wait for all pending requests to complete
        print(f"⏳ Waiting for {len(tasks)} requests to complete...")
        await asyncio.gather(*tasks, return_exceptions=True)
        
        # Calculate metrics for this QPS level
        qps_results = [r for r in self.results if r['qps_level'] == qps]
        successful = len([r for r in qps_results if r['success']])
        failed = len(qps_results) - successful
        
        if qps_results:
            response_times = [r['response_time_ms'] for r in qps_results if r['success']]
            if response_times:
                avg_response_time = np.mean(response_times)
                p95_response_time = np.percentile(response_times, 95)
                p99_response_time = np.percentile(response_times, 99)
            else:
                avg_response_time = p95_response_time = p99_response_time = 0
        else:
            avg_response_time = p95_response_time = p99_response_time = 0
        
        print(f"✅ {qps} QPS: {successful} success, {failed} failed, "
              f"avg: {avg_response_time:.1f}ms, p95: {p95_response_time:.1f}ms, p99: {p99_response_time:.1f}ms")
        
        return {
            'qps': qps,
            'total_requests': len(qps_results),
            'successful_requests': successful,
            'failed_requests': failed,
            'success_rate': successful / len(qps_results) if qps_results else 0,
            'avg_response_time': avg_response_time,
            'p95_response_time': p95_response_time,
            'p99_response_time': p99_response_time
        }
    
    async def run_load_test(self, qps_levels: List[int], duration_per_level: int, cooldown: int):
        """Run complete load test across multiple QPS levels"""
        print(f"🎯 Starting optimized load test with QPS levels: {qps_levels}")
        print(f"📊 Duration per level: {duration_per_level}s, Cooldown: {cooldown}s")
        
        summary_results = []
        
        for i, qps in enumerate(qps_levels):
            # Run test at this QPS level
            summary = await self.run_qps_level(qps, duration_per_level)
            summary_results.append(summary)
            
            # Cooldown period between tests (except for last one)
            if i < len(qps_levels) - 1:
                print(f"😴 Cooling down for {cooldown} seconds...")
                await asyncio.sleep(cooldown)
        
        return summary_results
    
    async def run_cache_performance_test(self, qps_levels: List[int], duration_per_level: int, cooldown: int):
        """Run cache performance test"""
        print(f"🧪 Starting cache performance test with QPS levels: {qps_levels}")
        
        cache_results = []
        
        for i, qps in enumerate(qps_levels):
            # Run cache test at this QPS level
            summary = await self.test_cache_performance(qps, duration_per_level)
            cache_results.append(summary)
            
            # Cooldown period between tests (except for last one)
            if i < len(qps_levels) - 1:
                print(f"😴 Cooling down for {cooldown} seconds...")
                await asyncio.sleep(cooldown)
        
        return cache_results

def create_optimized_performance_graphs(results_df, summary_df, cache_results_df):
    """Create comprehensive performance visualization graphs"""
    print("📈 Creating optimized performance graphs...")
    
    plt.style.use('default')
    sns.set_palette("husl")
    
    fig = plt.figure(figsize=(24, 20))
    
    # 1. Response Time Distribution by QPS (Single vs Batch)
    plt.subplot(4, 4, 1)
    successful_results = results_df[results_df['success'] == True]
    
    qps_levels = sorted(successful_results['qps_level'].unique())
    colors = sns.color_palette("husl", len(qps_levels))
    
    for i, qps in enumerate(qps_levels):
        qps_data = successful_results[successful_results['qps_level'] == qps]
        plt.hist(qps_data['response_time_ms'], bins=30, alpha=0.7, 
                label=f'{qps} QPS', color=colors[i])
    
    plt.xlabel('Response Time (ms)')
    plt.ylabel('Frequency')
    plt.title('Response Time Distribution by QPS Level')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. Single vs Batch Performance Comparison
    plt.subplot(4, 4, 2)
    single_results = results_df[(results_df['success'] == True) & (results_df['endpoint'] == 'single')]
    batch_results = results_df[(results_df['success'] == True) & (results_df['endpoint'] == 'batch')]
    
    if not single_results.empty and not batch_results.empty:
        single_times = single_results.groupby('qps_level')['response_time_ms'].mean()
        batch_times = batch_results.groupby('qps_level')['response_time_ms'].mean()
        
        plt.plot(single_times.index, single_times.values, 'o-', label='Single Requests', linewidth=2)
        plt.plot(batch_times.index, batch_times.values, 's-', label='Batch Requests', linewidth=2)
        plt.xlabel('QPS Level')
        plt.ylabel('Average Response Time (ms)')
        plt.title('Single vs Batch Performance')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    # 3. Cache Performance Analysis
    plt.subplot(4, 4, 3)
    if not cache_results_df.empty:
        cache_hit_rates = cache_results_df['cache_hit_rate'] * 100
        plt.bar(cache_results_df['qps'], cache_hit_rates, alpha=0.7, color='green')
        plt.xlabel('QPS Level')
        plt.ylabel('Cache Hit Rate (%)')
        plt.title('Cache Performance by QPS')
        plt.grid(True, alpha=0.3)
    
    # 4. Response Time vs QPS (All Endpoints)
    plt.subplot(4, 4, 4)
    plt.plot(summary_df['qps'], summary_df['avg_response_time'], 
             'o-', linewidth=2, markersize=8, color='blue')
    plt.xlabel('QPS (Queries Per Second)')
    plt.ylabel('Average Response Time (ms)')
    plt.title('Average Response Time vs QPS')
    plt.grid(True, alpha=0.3)
    
    # 5. P95 and P99 Response Times
    plt.subplot(4, 4, 5)
    plt.plot(summary_df['qps'], summary_df['p95_response_time'], 
             'o-', linewidth=2, label='P95', color='orange')
    plt.plot(summary_df['qps'], summary_df['p99_response_time'], 
             'o-', linewidth=2, label='P99', color='red')
    plt.xlabel('QPS')
    plt.ylabel('Response Time (ms)')
    plt.title('P95 and P99 Response Times vs QPS')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 6. Success Rate vs QPS
    plt.subplot(4, 4, 6)
    plt.plot(summary_df['qps'], summary_df['success_rate'] * 100, 
             'o-', linewidth=2, color='green')
    plt.xlabel('QPS')
    plt.ylabel('Success Rate (%)')
    plt.ylim(0, 105)
    plt.title('Success Rate vs QPS')
    plt.grid(True, alpha=0.3)
    
    # 7. Cache Hit Rate vs Response Time
    plt.subplot(4, 4, 7)
    if not cache_results_df.empty:
        plt.scatter(cache_results_df['cache_hit_rate'] * 100, 
                   cache_results_df['avg_response_time'], 
                   s=100, alpha=0.7)
        plt.xlabel('Cache Hit Rate (%)')
        plt.ylabel('Average Response Time (ms)')
        plt.title('Cache Hit Rate vs Response Time')
        plt.grid(True, alpha=0.3)
    
    # 8. Batch Size vs Processing Time
    plt.subplot(4, 4, 8)
    batch_results = results_df[(results_df['success'] == True) & (results_df['endpoint'] == 'batch')]
    if not batch_results.empty:
        plt.scatter(batch_results['batch_size'], batch_results['total_processing_time_ms'], 
                   alpha=0.6, s=50)
        plt.xlabel('Batch Size')
        plt.ylabel('Total Processing Time (ms)')
        plt.title('Batch Size vs Processing Time')
        plt.grid(True, alpha=0.3)
    
    # 9. Throughput Analysis
    plt.subplot(4, 4, 9)
    actual_qps = []
    target_qps = summary_df['qps'].values
    
    for qps in target_qps:
        qps_data = results_df[results_df['qps_level'] == qps]
        if not qps_data.empty:
            time_span = qps_data['timestamp'].max() - qps_data['timestamp'].min()
            actual = len(qps_data) / time_span if time_span > 0 else 0
            actual_qps.append(actual)
        else:
            actual_qps.append(0)
    
    plt.plot(target_qps, target_qps, 'r--', label='Target QPS', linewidth=2)
    plt.plot(target_qps, actual_qps, 'o-', label='Actual QPS', linewidth=2, color='blue')
    plt.xlabel('Target QPS')
    plt.ylabel('Actual QPS')
    plt.title('Target vs Actual QPS')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 10. Memory and Resource Usage (if available)
    plt.subplot(4, 4, 10)
    # This would be populated with actual memory usage data from the service
    
    # 11. Error Rate Analysis
    plt.subplot(4, 4, 11)
    error_rates = []
    qps_for_errors = []
    
    for qps in sorted(results_df['qps_level'].unique()):
        qps_data = results_df[results_df['qps_level'] == qps]
        error_rate = (len(qps_data[qps_data['success'] == False]) / len(qps_data)) * 100
        error_rates.append(error_rate)
        qps_for_errors.append(qps)
    
    plt.bar(qps_for_errors, error_rates, alpha=0.7, color='red')
    plt.xlabel('QPS')
    plt.ylabel('Error Rate (%)')
    plt.title('Error Rate by QPS Level')
    plt.grid(True, alpha=0.3)
    
    # 12. Performance Summary Heatmap
    plt.subplot(4, 4, 12)
    # Create a summary heatmap of key metrics
    
    plt.tight_layout()
    
    # Save the plot
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'optimized_load_test_results_{timestamp}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"📊 Optimized performance graphs saved to: {filename}")
    
    plt.show()

def print_optimized_summary_table(summary_df, cache_results_df):
    """Print a formatted summary table with optimization metrics"""
    print("\n" + "="*120)
    print("🎯 OPTIMIZED LOAD TEST SUMMARY")
    print("="*120)
    
    print(f"{'QPS':<6} {'Total':<8} {'Success':<8} {'Failed':<7} {'Success%':<9} "
          f"{'Avg(ms)':<8} {'P95(ms)':<8} {'P99(ms)':<8} {'Cache%':<8}")
    print("-" * 120)
    
    for _, row in summary_df.iterrows():
        qps = row['qps']
        cache_row = cache_results_df[cache_results_df['qps'] == qps]
        cache_hit_rate = cache_row['cache_hit_rate'].iloc[0] * 100 if not cache_row.empty else 0
        
        print(f"{qps:<6} {row['total_requests']:<8} {row['successful_requests']:<8} "
              f"{row['failed_requests']:<7} {row['success_rate']*100:<8.1f}% "
              f"{row['avg_response_time']:<8.1f} {row['p95_response_time']:<8.1f} "
              f"{row['p99_response_time']:<8.1f} {cache_hit_rate:<8.1f}%")
    
    print("="*120)

async def main():
    parser = argparse.ArgumentParser(description='Optimized Load Test for CP Prediction Service')
    parser.add_argument('--max-qps', type=int, default=150, help='Maximum QPS to test (default: 150)')
    parser.add_argument('--duration', type=int, default=15, help='Duration per QPS level in seconds (default: 15)')
    parser.add_argument('--cooldown', type=int, default=3, help='Cooldown between tests in seconds (default: 3)')
    parser.add_argument('--steps', type=int, default=10, help='Number of QPS steps (default: 10)')
    parser.add_argument('--test-cache', action='store_true', help='Run cache performance test')
    
    args = parser.parse_args()
    
    # Generate QPS levels
    qps_levels = []
    qps_levels.extend([1, 2, 5, 10])
    
    if args.max_qps > 10:
        step_size = (args.max_qps - 10) // max(1, args.steps - 4)
        current_qps = 10
        while current_qps < args.max_qps:
            current_qps += step_size
            if current_qps <= args.max_qps:
                qps_levels.append(current_qps)
    
    if args.max_qps not in qps_levels:
        qps_levels.append(args.max_qps)
    
    qps_levels = sorted(list(set(qps_levels)))
    
    print(f"🚀 Optimized Load Testing CP Prediction Service")
    print(f"📊 QPS Levels: {qps_levels}")
    print(f"⏱️  Duration per level: {args.duration} seconds")
    print(f"😴 Cooldown between levels: {args.cooldown} seconds")
    print(f"🎯 Total estimated time: {len(qps_levels) * (args.duration + args.cooldown) - args.cooldown} seconds")
    
    # Test service availability
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{BASE_URL}/health") as response:
                if response.status == 200:
                    health_data = await response.json()
                    print(f"✅ Service is healthy: {health_data}")
                else:
                    print(f"❌ Service health check failed: {response.status}")
                    sys.exit(1)
    except Exception as e:
        print(f"❌ Cannot connect to service at {BASE_URL}: {e}")
        print("💡 Make sure the optimized service is running: uvicorn app.optimized_main:app --host 0.0.0.0 --port 8000")
        sys.exit(1)
    
    # Run load test
    async with OptimizedLoadTester() as tester:
        # Run main load test
        summary_results = await tester.run_load_test(qps_levels, args.duration, args.cooldown)
        
        # Run cache performance test if requested
        cache_results = []
        if args.test_cache:
            print("\n🧪 Running cache performance test...")
            cache_results = await tester.run_cache_performance_test(qps_levels, args.duration // 2, args.cooldown)
        
        # Convert results to DataFrames
        results_df = pd.DataFrame(tester.results)
        summary_df = pd.DataFrame(summary_results)
        cache_results_df = pd.DataFrame(cache_results) if cache_results else pd.DataFrame()
        
        # Print summary
        print_optimized_summary_table(summary_df, cache_results_df)
        
        # Save detailed results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = f'optimized_load_test_detailed_{timestamp}.json'
        summary_file = f'optimized_load_test_summary_{timestamp}.json'
        cache_file = f'optimized_load_test_cache_{timestamp}.json'
        
        # Save to JSON
        with open(results_file, 'w') as f:
            json.dump(tester.results, f, indent=2, default=str)
        
        with open(summary_file, 'w') as f:
            json.dump(summary_results, f, indent=2, default=str)
        
        if cache_results:
            with open(cache_file, 'w') as f:
                json.dump(cache_results, f, indent=2, default=str)
        
        # Save to CSV
        results_df.to_csv(f'optimized_load_test_detailed_{timestamp}.csv', index=False)
        summary_df.to_csv(f'optimized_load_test_summary_{timestamp}.csv', index=False)
        if not cache_results_df.empty:
            cache_results_df.to_csv(f'optimized_load_test_cache_{timestamp}.csv', index=False)
        
        print(f"\n💾 Results saved:")
        print(f"   Detailed: {results_file}")
        print(f"   Summary: {summary_file}")
        if cache_results:
            print(f"   Cache: {cache_file}")
        print(f"   CSV files: optimized_load_test_*.csv")
        
        # Create performance graphs
        if len(results_df) > 0:
            create_optimized_performance_graphs(results_df, summary_df, cache_results_df)
        else:
            print("❌ No results to graph")

if __name__ == "__main__":
    asyncio.run(main())
