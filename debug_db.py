#!/usr/bin/env python3
"""
Simple script to debug database connection and insertion issues
"""

import pymysql
from datetime import datetime

def test_database_connection():
    """Test basic database connection"""
    try:
        # Test connection
        connection = pymysql.connect(
            host='localhost',
            user='cp_test',
            password='test123',
            database='cp_predictions'
        )
        print("✅ Database connection successful")
        
        # Test simple query
        with connection.cursor() as cursor:
            cursor.execute("SELECT COUNT(*) FROM predictions")
            result = cursor.fetchone()
            print(f"✅ Current predictions count: {result[0]}")
        
        # Test insertion
        test_data = (
            'TEST_DEBUG_001',
            'Test Address Debug',
            '110001',
            'TEST_CP',
            0.95,
            'v1_debug',
            0,
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            100.0
        )
        
        insert_query = """
        INSERT INTO predictions (shipment_id, address, pincode, predicted_cp, confidence, 
                              model_version, cached, timestamp, processing_time_ms)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        
        with connection.cursor() as cursor:
            cursor.execute(insert_query, test_data)
            connection.commit()
            print("✅ Test insertion successful")
        
        # Check count again
        with connection.cursor() as cursor:
            cursor.execute("SELECT COUNT(*) FROM predictions")
            result = cursor.fetchone()
            print(f"✅ New predictions count: {result[0]}")
        
        connection.close()
        print("✅ Database test completed successfully")
        
    except Exception as e:
        print(f"❌ Database test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_database_connection()
