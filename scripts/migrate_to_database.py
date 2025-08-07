#!/usr/bin/env python3
"""
Database Migration Script
Uploads CSV data to online PostgreSQL database
"""

import pandas as pd
import os
import sys
from sqlalchemy import text

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config.config import DatabaseConfig
from src.data.data_loader import DataLoader

def create_tables(db_config):
    """Create the necessary tables in the database"""
    try:
        # Create car_news table with enhanced schema
        news_table_sql = """
        CREATE TABLE IF NOT EXISTS car_news (
            id SERIAL PRIMARY KEY,
            title TEXT,
            link TEXT,
            author TEXT,
            publication_date DATE,
            source TEXT,
            content TEXT,
            sentiment TEXT,
            sentiment_score FLOAT,
            topics TEXT,
            entities TEXT,
            keywords TEXT,
            analysis_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            top_ngrams TEXT
        );
        """
        
        # Create car_reviews table with enhanced schema
        reviews_table_sql = """
        CREATE TABLE IF NOT EXISTS car_reviews (
            id SERIAL PRIMARY KEY,
            title TEXT,
            link TEXT,
            author TEXT,
            publication_date DATE,
            source TEXT,
            verdict TEXT,
            rating FLOAT,
            price DECIMAL(10,2),
            sentiment TEXT,
            sentiment_score FLOAT,
            topics TEXT,
            entities TEXT,
            keywords TEXT,
            correlation_score FLOAT,
            review_length INTEGER,
            analysis_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            top_ngrams TEXT
        );
        """
        
        with db_config.engine.connect() as conn:
            conn.execute(text(news_table_sql))
            conn.execute(text(reviews_table_sql))
            conn.commit()
        
        print("✅ Database tables created successfully")
        return True
        
    except Exception as e:
        print(f"❌ Error creating tables: {e}")
        return False

def upload_csv_to_database(csv_file, table_name, db_config):
    """Upload CSV data to database table"""
    try:
        print(f"📁 Loading {csv_file}...")
        df = pd.read_csv(csv_file)
        
        print(f"📊 Data shape: {df.shape}")
        print(f"📋 Columns: {df.columns.tolist()}")
        
        # Transform data to match new schema
        if table_name == 'car_news':
            df = transform_news_data(df)
        elif table_name == 'car_reviews':
            df = transform_reviews_data(df)
        
        # Upload to database
        print(f"⬆️  Uploading to table: {table_name}")
        success = db_config.save_to_database(df, table_name)
        
        if success:
            print(f"✅ Successfully uploaded {len(df)} rows to {table_name}")
        else:
            print(f"❌ Failed to upload to {table_name}")
            
        return success
        
    except Exception as e:
        print(f"❌ Error uploading {csv_file}: {e}")
        return False

def transform_news_data(df):
    """Transform news CSV data to match new database schema"""
    # Create new DataFrame with required columns
    transformed_df = pd.DataFrame()
    
    # Map existing columns to new schema (using actual CSV columns)
    transformed_df['title'] = df.get('title', '')
    transformed_df['link'] = df.get('link', '')
    transformed_df['author'] = df.get('author', '')
    transformed_df['publication_date'] = pd.to_datetime(df.get('publication_date', ''), errors='coerce')
    transformed_df['source'] = df.get('source', 'CSV Import')
    transformed_df['content'] = df.get('content', '')
    
    # Add new columns with default values
    transformed_df['sentiment'] = None
    transformed_df['sentiment_score'] = None
    transformed_df['topics'] = None
    transformed_df['entities'] = None
    transformed_df['keywords'] = None
    transformed_df['top_ngrams'] = None
    
    return transformed_df

def transform_reviews_data(df):
    """Transform reviews CSV data to match new database schema"""
    # Create new DataFrame with required columns
    transformed_df = pd.DataFrame()
    
    # Map existing columns to new schema (using new CSV structure)
    transformed_df['title'] = df.get('title', '')
    transformed_df['link'] = df.get('link', '')
    transformed_df['author'] = df.get('author', '')
    transformed_df['publication_date'] = pd.to_datetime(df.get('publication_date', ''), errors='coerce')
    transformed_df['source'] = df.get('source', 'CSV Import')
    transformed_df['verdict'] = df.get('verdict', '')
    transformed_df['rating'] = pd.to_numeric(df.get('rating', 0), errors='coerce')
    transformed_df['price'] = pd.to_numeric(df.get('price', 0), errors='coerce')
    
    # Add new columns with default values
    transformed_df['sentiment'] = None
    transformed_df['sentiment_score'] = None
    transformed_df['topics'] = None
    transformed_df['entities'] = None
    transformed_df['keywords'] = None
    transformed_df['correlation_score'] = None
    transformed_df['review_length'] = transformed_df['verdict'].str.len()
    transformed_df['top_ngrams'] = None
    
    return transformed_df

def verify_database_connection():
    """Test database connection"""
    print("🔌 Testing database connection...")
    
    db_config = DatabaseConfig()
    if db_config.setup_from_env():
        print("✅ Database connection successful")
        return db_config
    else:
        print("❌ Database connection failed")
        return None

def main():
    """Main migration function"""
    print("🚀 Starting Database Migration")
    print("=" * 50)
    
    # Check if .env file exists
    if not os.path.exists('.env'):
        print("❌ .env file not found!")
        print("Please create a .env file with your database credentials:")
        print("""
DB_HOST=your-online-postgres-host
DB_PORT=5432
DB_NAME=your_database_name
DB_USER=your_username
DB_PASSWORD=your_password
NEWS_TABLE=car_news
REVIEWS_TABLE=car_reviews
        """)
        return
    
    # Test database connection
    db_config = verify_database_connection()
    if not db_config:
        return
    
    # Create tables
    print("\n📋 Creating database tables...")
    if not create_tables(db_config):
        return
    
    # Upload CSV files
    print("\n📤 Uploading CSV data...")
    
    # Upload news data
    if os.path.exists('datasets/car_news_dataset.csv'):
        upload_csv_to_database('datasets/car_news_dataset.csv', 'car_news', db_config)
    else:
        print("⚠️  datasets/car_news_dataset.csv not found")
    
    # Upload reviews data
    if os.path.exists('datasets/car_reviews_dataset.csv'):
        upload_csv_to_database('datasets/car_reviews_dataset.csv', 'car_reviews', db_config)
    else:
        print("⚠️  datasets/car_reviews_dataset.csv not found")
    
    # Test loading from database
    print("\n🧪 Testing database loading...")
    data_loader = DataLoader(
        news_file="datasets/car_news_dataset.csv",
        reviews_file="datasets/car_reviews_dataset.csv",
        use_database=True,
        news_table="car_news",
        reviews_table="car_reviews"
    )
    
    if data_loader.load_data():
        print("✅ Database migration completed successfully!")
        print("\n📝 Next steps:")
        print("1. Update your analysis scripts to use database mode")
        print("2. Set use_database=True in DataLoader")
        print("3. Your data is now available online!")
    else:
        print("❌ Database migration failed")
    
    # Cleanup
    db_config.cleanup()

if __name__ == "__main__":
    main()
