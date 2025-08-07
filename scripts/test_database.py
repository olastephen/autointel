#!/usr/bin/env python3
"""
Database Connection Test Script
Tests the database connection and basic operations
"""

import os
import pandas as pd
import sys
from sqlalchemy import text

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config.config import DatabaseConfig
from src.data.data_loader import DataLoader

def test_database_connection():
    """Test basic database connection"""
    print("🔌 Testing Database Connection")
    print("=" * 40)
    
    db_config = DatabaseConfig()
    
    # Test connection setup
    if db_config.setup_from_env():
        print("✅ Database connection successful")
        
        # Test basic query
        try:
            with db_config.engine.connect() as conn:
                result = conn.execute(text("SELECT version();"))
                version = result.fetchone()[0]
                print(f"✅ PostgreSQL version: {version}")
        except Exception as e:
            print(f"❌ Error executing query: {e}")
            return False
        
        return True
    else:
        print("❌ Database connection failed")
        return False

def test_table_operations():
    """Test table creation and data operations"""
    print("\n📋 Testing Table Operations")
    print("=" * 40)
    
    db_config = DatabaseConfig()
    if not db_config.setup_from_env():
        return False
    
    try:
        # Create test table
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS test_table (
            id SERIAL PRIMARY KEY,
            name VARCHAR(100),
            value INTEGER
        );
        """
        
        with db_config.engine.connect() as conn:
            conn.execute(text(create_table_sql))
            conn.commit()
        
        print("✅ Test table created successfully")
        
        # Insert test data
        test_data = pd.DataFrame({
            'name': ['Test1', 'Test2', 'Test3'],
            'value': [1, 2, 3]
        })
        
        success = db_config.save_to_database(test_data, 'test_table')
        if success:
            print("✅ Test data inserted successfully")
        else:
            print("❌ Failed to insert test data")
            return False
        
        # Load test data
        loaded_data = db_config.load_table('test_table')
        if not loaded_data.empty:
            print(f"✅ Test data loaded successfully: {len(loaded_data)} rows")
            print(f"   Columns: {loaded_data.columns.tolist()}")
        else:
            print("❌ Failed to load test data")
            return False
        
        # Clean up test table
        with db_config.engine.connect() as conn:
            conn.execute(text("DROP TABLE IF EXISTS test_table;"))
            conn.commit()
        
        print("✅ Test table cleaned up")
        return True
        
    except Exception as e:
        print(f"❌ Error in table operations: {e}")
        return False

def test_data_loader():
    """Test DataLoader with database mode"""
    print("\n📊 Testing DataLoader with Database")
    print("=" * 40)
    
    # Check if tables exist
    db_config = DatabaseConfig()
    if not db_config.setup_from_env():
        return False
    
    try:
        # Check if car_news table exists
        with db_config.engine.connect() as conn:
            result = conn.execute(text("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'car_news'
                );
            """))
            news_exists = result.fetchone()[0]
            
            result = conn.execute(text("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'car_reviews'
                );
            """))
            reviews_exists = result.fetchone()[0]
        
        if not news_exists or not reviews_exists:
            print("⚠️  Database tables not found")
            print("   Run 'python migrate_to_database.py' to create tables and upload data")
            return False
        
        # Test DataLoader
        data_loader = DataLoader(
            news_file="car_news_dataset.csv",
            reviews_file="car_reviews_dataset.csv",
            use_database=True,
            news_table="car_news",
            reviews_table="car_reviews"
        )
        
        if data_loader.load_data():
            print("✅ DataLoader with database mode successful")
            
            if data_loader.car_news_df is not None:
                print(f"   News data: {data_loader.car_news_df.shape}")
            if data_loader.car_reviews_df is not None:
                print(f"   Reviews data: {data_loader.car_reviews_df.shape}")
            
            return True
        else:
            print("❌ DataLoader with database mode failed")
            return False
            
    except Exception as e:
        print(f"❌ Error testing DataLoader: {e}")
        return False

def main():
    """Main test function"""
    print("🧪 Database Connection and Operations Test")
    print("=" * 50)
    
    # Check if .env file exists
    if not os.path.exists('.env'):
        print("❌ .env file not found!")
        print("Please create a .env file with your database credentials")
        return
    
    # Test connection
    if not test_database_connection():
        return
    
    # Test table operations
    if not test_table_operations():
        return
    
    # Test DataLoader
    if not test_data_loader():
        return
    
    print("\n🎉 All database tests passed!")
    print("\n📝 Next steps:")
    print("1. Your database connection is working correctly")
    print("2. You can now run 'python migrate_to_database.py' to upload your data")
    print("3. Use 'USE_DATABASE=true python main.py' to run analysis with database")

if __name__ == "__main__":
    main()
