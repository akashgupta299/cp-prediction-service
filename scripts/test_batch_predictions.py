#!/usr/bin/env python3
"""
Script to test batch predictions using real addresses from parquet file
"""

import pandas as pd
import requests
import json
import time
from datetime import datetime
import random

def load_addresses(file_path, sample_size=100):
    """Load addresses from parquet file"""
    print(f"📊 Loading addresses from {file_path}")
    df = pd.read_parquet(file_path)
    
    # Filter valid addresses with full_address column
    valid_addresses = df[df['full_address'].notna() & (df['full_address'].str.len() > 10)]
    
    # Sample random addresses
    sample_df = valid_addresses.sample(n=min(sample_size, len(valid_addresses)), random_state=42)
    
    addresses = []
    for idx, row in sample_df.iterrows():
        addresses.append({
            'address': row['full_address'],
            'shipment_id': f'BATCH_{idx}_{int(time.time())}',
            'original_cp': row.get('cp_code', 'Unknown')
        })
    
    print(f"✅ Loaded {len(addresses)} addresses")
    return addresses

def make_prediction_request(address_data, base_url="http://localhost:8000"):
    """Make a single prediction request"""
    try:
        payload = {
            "address": address_data['address'],
            "shipment_id": address_data['shipment_id']
        }
        
        response = requests.post(
            f"{base_url}/predict",
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            return {
                'success': True,
                'shipment_id': result['shipment_id'],
                'predicted_cp': result['predicted_cp'],
                'confidence': result['confidence'],
                'original_cp': address_data['original_cp'],
                'address': address_data['address'][:100] + '...' if len(address_data['address']) > 100 else address_data['address']
            }
        else:
            return {
                'success': False,
                'error': f"HTTP {response.status_code}: {response.text}",
                'shipment_id': address_data['shipment_id']
            }
            
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'shipment_id': address_data['shipment_id']
        }

def test_batch_predictions(file_path, sample_size=100, delay_ms=100):
    """Test batch predictions"""
    print("🚀 Starting Batch Prediction Test")
    print("=" * 50)
    
    # Load addresses
    addresses = load_addresses(file_path, sample_size)
    
    # Test server connectivity
    try:
        response = requests.get("http://localhost:8000/docs", timeout=5)
        if response.status_code != 200:
            print("❌ Server not responding. Please start the FastAPI service first.")
            return
    except:
        print("❌ Cannot connect to server. Please start the FastAPI service first.")
        return
    
    print(f"✅ Server is running")
    print(f"🎯 Making {len(addresses)} prediction requests...")
    print()
    
    results = []
    successful = 0
    failed = 0
    
    start_time = time.time()
    
    for i, address_data in enumerate(addresses, 1):
        print(f"[{i:3d}/{len(addresses)}] Processing: {address_data['address'][:50]}...")
        
        result = make_prediction_request(address_data)
        results.append(result)
        
        if result['success']:
            successful += 1
            print(f"    ✅ Predicted: {result['predicted_cp']} (confidence: {result['confidence']:.4f})")
        else:
            failed += 1
            print(f"    ❌ Failed: {result['error']}")
        
        # Add delay to avoid overwhelming the server
        if delay_ms > 0:
            time.sleep(delay_ms / 1000.0)
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Print summary
    print("\n" + "=" * 50)
    print("📊 BATCH PREDICTION SUMMARY")
    print("=" * 50)
    print(f"Total Requests: {len(addresses)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Success Rate: {successful/len(addresses)*100:.1f}%")
    print(f"Total Time: {total_time:.2f} seconds")
    print(f"Average Time per Request: {total_time/len(addresses)*1000:.2f}ms")
    
    # Show some sample results
    print(f"\n📝 Sample Results:")
    for result in results[:10]:
        if result['success']:
            print(f"  {result['shipment_id']}: {result['predicted_cp']} ({result['confidence']:.3f})")
    
    # Show prediction distribution
    if successful > 0:
        successful_results = [r for r in results if r['success']]
        cp_counts = {}
        for result in successful_results:
            cp = result['predicted_cp']
            cp_counts[cp] = cp_counts.get(cp, 0) + 1
        
        print(f"\n🎯 Prediction Distribution (Top 10):")
        sorted_cps = sorted(cp_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        for cp, count in sorted_cps:
            print(f"  {cp}: {count} ({count/successful*100:.1f}%)")
    
    # Save results to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"batch_prediction_results_{timestamp}.json"
    
    with open(output_file, 'w') as f:
        json.dump({
            'summary': {
                'total_requests': len(addresses),
                'successful': successful,
                'failed': failed,
                'success_rate': successful/len(addresses)*100,
                'total_time_seconds': total_time,
                'avg_time_per_request_ms': total_time/len(addresses)*1000
            },
            'results': results
        }, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_file}")
    
    return results

def main():
    """Main function"""
    file_path = "/Users/akash/Documents/cp_pred_github/cp-prediction-service/notebook/pan_India_test_data_2025-07-23.parquet"
    
    # Test with 100 addresses
    results = test_batch_predictions(
        file_path=file_path,
        sample_size=100,
        delay_ms=50  # 50ms delay between requests
    )
    
    print(f"\n🎉 Batch prediction test completed!")

if __name__ == "__main__":
    main()
