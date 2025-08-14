#!/usr/bin/env python3
"""
Script to calculate accuracy by comparing predictions with actual cp_code from parquet file
"""

import pandas as pd
import mysql.connector
import pymysql
from collections import defaultdict
import json

def load_actual_data(file_path):
    """Load actual cp_code data from parquet file"""
    print(f"📊 Loading actual data from {file_path}")
    df = pd.read_parquet(file_path)
    
    # Create a mapping from address to actual cp_code
    actual_mapping = {}
    for idx, row in df.iterrows():
        if pd.notna(row['full_address']) and pd.notna(row['cp_code']):
            actual_mapping[row['full_address']] = row['cp_code']
    
    print(f"✅ Loaded {len(actual_mapping)} address-to-cp mappings")
    return actual_mapping

def get_predictions_from_db():
    """Get all predictions from database"""
    print("🗄️  Fetching predictions from database...")
    
    try:
        connection = pymysql.connect(
            host='localhost',
            user='cp_service',
            password='$eRv!(e$@12',
            database='cp_predictions'
        )
        
        query = """
        SELECT shipment_id, address, predicted_cp, confidence, timestamp 
        FROM predictions 
        WHERE shipment_id LIKE 'BATCH_%'
        ORDER BY timestamp DESC
        """
        
        df = pd.read_sql(query, connection)
        connection.close()
        
        print(f"✅ Retrieved {len(df)} predictions from database")
        return df
        
    except Exception as e:
        print(f"❌ Error fetching predictions: {e}")
        return pd.DataFrame()

def calculate_accuracy(predictions_df, actual_mapping):
    """Calculate accuracy metrics"""
    print("🎯 Calculating accuracy metrics...")
    
    results = {
        'total_predictions': 0,
        'matched_addresses': 0,
        'exact_matches': 0,
        'invalid_predictions': 0,
        'accuracy_overall': 0.0,
        'accuracy_valid_only': 0.0,
        'confidence_stats': {},
        'detailed_results': []
    }
    
    exact_matches = 0
    invalid_predictions = 0
    matched_addresses = 0
    confidence_scores = []
    
    for idx, row in predictions_df.iterrows():
        address = row['address']
        predicted_cp = row['predicted_cp']
        confidence = row['confidence']
        shipment_id = row['shipment_id']
        
        # Check if we have actual data for this address
        if address in actual_mapping:
            matched_addresses += 1
            actual_cp = actual_mapping[address]
            
            is_exact_match = (predicted_cp == actual_cp)
            is_invalid = (predicted_cp == 'INVALID')
            
            if is_exact_match:
                exact_matches += 1
            
            if is_invalid:
                invalid_predictions += 1
            
            confidence_scores.append(confidence)
            
            results['detailed_results'].append({
                'shipment_id': shipment_id,
                'address': address[:100] + '...' if len(address) > 100 else address,
                'predicted_cp': predicted_cp,
                'actual_cp': actual_cp,
                'confidence': confidence,
                'exact_match': is_exact_match,
                'is_invalid': is_invalid
            })
    
    # Calculate metrics
    results['total_predictions'] = len(predictions_df)
    results['matched_addresses'] = matched_addresses
    results['exact_matches'] = exact_matches
    results['invalid_predictions'] = invalid_predictions
    
    if matched_addresses > 0:
        results['accuracy_overall'] = exact_matches / matched_addresses
        
        # Accuracy excluding INVALID predictions
        valid_predictions = matched_addresses - invalid_predictions
        if valid_predictions > 0:
            valid_exact_matches = sum(1 for r in results['detailed_results'] 
                                    if r['exact_match'] and not r['is_invalid'])
            results['accuracy_valid_only'] = valid_exact_matches / valid_predictions
    
    # Confidence statistics
    if confidence_scores:
        results['confidence_stats'] = {
            'mean': float(pd.Series(confidence_scores).mean()),
            'median': float(pd.Series(confidence_scores).median()),
            'std': float(pd.Series(confidence_scores).std()),
            'min': float(pd.Series(confidence_scores).min()),
            'max': float(pd.Series(confidence_scores).max())
        }
    
    return results

def analyze_prediction_patterns(results):
    """Analyze prediction patterns"""
    print("📈 Analyzing prediction patterns...")
    
    # Group by predicted CP
    cp_analysis = defaultdict(lambda: {
        'count': 0, 
        'correct': 0, 
        'avg_confidence': 0.0,
        'confidence_scores': []
    })
    
    # Group by actual CP
    actual_cp_analysis = defaultdict(lambda: {
        'count': 0, 
        'correct': 0, 
        'predicted_as': defaultdict(int)
    })
    
    for result in results['detailed_results']:
        predicted = result['predicted_cp']
        actual = result['actual_cp']
        confidence = result['confidence']
        is_correct = result['exact_match']
        
        # Predicted CP analysis
        cp_analysis[predicted]['count'] += 1
        cp_analysis[predicted]['confidence_scores'].append(confidence)
        if is_correct:
            cp_analysis[predicted]['correct'] += 1
        
        # Actual CP analysis
        actual_cp_analysis[actual]['count'] += 1
        actual_cp_analysis[actual]['predicted_as'][predicted] += 1
        if is_correct:
            actual_cp_analysis[actual]['correct'] += 1
    
    # Calculate averages
    for cp in cp_analysis:
        scores = cp_analysis[cp]['confidence_scores']
        cp_analysis[cp]['avg_confidence'] = sum(scores) / len(scores) if scores else 0.0
        cp_analysis[cp]['accuracy'] = cp_analysis[cp]['correct'] / cp_analysis[cp]['count']
    
    for cp in actual_cp_analysis:
        actual_cp_analysis[cp]['accuracy'] = (
            actual_cp_analysis[cp]['correct'] / actual_cp_analysis[cp]['count']
        )
    
    return dict(cp_analysis), dict(actual_cp_analysis)

def print_results(results, cp_analysis, actual_cp_analysis):
    """Print detailed results"""
    print("\n" + "="*60)
    print("🎯 ACCURACY ANALYSIS RESULTS")
    print("="*60)
    
    print(f"📊 Overall Metrics:")
    print(f"   Total Predictions: {results['total_predictions']}")
    print(f"   Matched Addresses: {results['matched_addresses']}")
    print(f"   Exact Matches: {results['exact_matches']}")
    print(f"   Invalid Predictions: {results['invalid_predictions']}")
    print(f"   Overall Accuracy: {results['accuracy_overall']:.4f} ({results['accuracy_overall']*100:.2f}%)")
    print(f"   Accuracy (Valid Only): {results['accuracy_valid_only']:.4f} ({results['accuracy_valid_only']*100:.2f}%)")
    
    print(f"\n📈 Confidence Statistics:")
    if results['confidence_stats']:
        stats = results['confidence_stats']
        print(f"   Mean Confidence: {stats['mean']:.4f}")
        print(f"   Median Confidence: {stats['median']:.4f}")
        print(f"   Std Deviation: {stats['std']:.4f}")
        print(f"   Min Confidence: {stats['min']:.4f}")
        print(f"   Max Confidence: {stats['max']:.4f}")
    
    print(f"\n🎯 Top Predicted CPs (by count):")
    sorted_predicted = sorted(cp_analysis.items(), key=lambda x: x[1]['count'], reverse=True)[:10]
    for cp, data in sorted_predicted:
        print(f"   {cp}: {data['count']} predictions, {data['accuracy']:.3f} accuracy, {data['avg_confidence']:.3f} avg conf")
    
    print(f"\n📋 Top Actual CPs (by count):")
    sorted_actual = sorted(actual_cp_analysis.items(), key=lambda x: x[1]['count'], reverse=True)[:10]
    for cp, data in sorted_actual:
        print(f"   {cp}: {data['count']} occurrences, {data['accuracy']:.3f} correctly predicted")
    
    print(f"\n✅ Sample Correct Predictions:")
    correct_samples = [r for r in results['detailed_results'] if r['exact_match'] and not r['is_invalid']][:5]
    for sample in correct_samples:
        print(f"   {sample['predicted_cp']} == {sample['actual_cp']} (conf: {sample['confidence']:.3f})")
    
    print(f"\n❌ Sample Incorrect Predictions:")
    incorrect_samples = [r for r in results['detailed_results'] if not r['exact_match'] and not r['is_invalid']][:5]
    for sample in incorrect_samples:
        print(f"   {sample['predicted_cp']} != {sample['actual_cp']} (conf: {sample['confidence']:.3f})")

def main():
    """Main function"""
    parquet_file = "/Users/akash/Documents/cp_pred_github/cp-prediction-service/notebook/pan_India_test_data_2025-07-23.parquet"
    
    # Load actual data
    actual_mapping = load_actual_data(parquet_file)
    
    # Get predictions from database
    predictions_df = get_predictions_from_db()
    
    if predictions_df.empty:
        print("❌ No predictions found in database")
        return
    
    # Calculate accuracy
    results = calculate_accuracy(predictions_df, actual_mapping)
    
    # Analyze patterns
    cp_analysis, actual_cp_analysis = analyze_prediction_patterns(results)
    
    # Print results
    print_results(results, cp_analysis, actual_cp_analysis)
    
    # Save detailed results
    output_file = f"accuracy_analysis_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump({
            'summary': {
                'total_predictions': results['total_predictions'],
                'matched_addresses': results['matched_addresses'],
                'exact_matches': results['exact_matches'],
                'overall_accuracy': results['accuracy_overall'],
                'valid_only_accuracy': results['accuracy_valid_only'],
                'confidence_stats': results['confidence_stats']
            },
            'cp_analysis': cp_analysis,
            'actual_cp_analysis': actual_cp_analysis,
            'detailed_results': results['detailed_results']
        }, f, indent=2)
    
    print(f"\n💾 Detailed results saved to: {output_file}")

if __name__ == "__main__":
    main()
