"""
Configuration file for Car Analysis Framework
Contains settings, constants, and file paths
"""

# File paths
NEWS_FILE = 'datasets/car_news_dataset.csv'
REVIEWS_FILE = 'datasets/car_reviews_dataset.csv'

# Dataset Structure Information
# Car News Dataset Columns:
# - title: News article titles
# - content: News article content

# Car Reviews Dataset Columns:
# - Unnamed: 0: Index column
# - Review_Date: Date of the review (format: "on MM/DD/YY HH:MM PM (PST)")
# - Author_Name: Name of the reviewer
# - Vehicle_Title: Vehicle information (e.g., "1997 Toyota Previa Minivan LE 3dr Minivan")
# - Review_Title: Title of the review
# - Review: Review text content
# - Rating: Numerical rating (1-5 scale)

# Analysis settings
N_TOPICS = 5
N_WORDS_PER_TOPIC = 10
TOP_N_BIGRAMS = 15
TOP_N_TRIGRAMS = 10
CORRELATION_THRESHOLD = 0.3
KEYWORD_CORRELATION_THRESHOLD = 0.2

# Car-related keywords for analysis
CAR_BRANDS = [
    'toyota', 'honda', 'ford', 'bmw', 'mercedes', 'audi', 'volkswagen',
    'nissan', 'hyundai', 'kia', 'chevrolet', 'dodge', 'jeep', 'tesla',
    'lexus', 'acura', 'infiniti', 'cadillac', 'buick', 'chrysler',
    'volvo', 'subaru', 'mazda', 'mitsubishi', 'porsche', 'ferrari',
    'lamborghini', 'bentley', 'rolls royce', 'aston martin'
]

ANALYSIS_KEYWORDS = [
    'price', 'quality', 'performance', 'safety', 'fuel', 'electric',
    'hybrid', 'luxury', 'affordable', 'reliable', 'comfort', 'technology',
    'mileage', 'warranty', 'maintenance', 'resale', 'efficiency', 'speed',
    'handling', 'interior', 'exterior', 'features', 'design', 'brand'
]

# Visualization settings
PLOT_STYLE = 'seaborn-v0_8'
COLOR_PALETTE = "husl"
FIGURE_SIZE = (12, 8)
HEATMAP_SIZE = (10, 8)

# NLP settings
MAX_FEATURES = 1000
MAX_DF = 0.95
MIN_DF = 2
LDA_MAX_ITER = 50
LDA_RANDOM_STATE = 42

# Sample size for NER analysis (for performance)
NER_SAMPLE_SIZE = 100

# Database Configuration
import os
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
import psycopg2
from dotenv import load_dotenv

class DatabaseConfig:
    """Database configuration and connection management"""
    
    def __init__(self):
        self.engine = None
        self.connection_string = None
        
    def setup_from_env(self):
        """Setup database connection from environment variables"""
        try:
            # Load environment variables
            load_dotenv()
            
            # Get database credentials from environment
            db_host = os.getenv('DB_HOST')
            db_port = os.getenv('DB_PORT', '5432')
            db_name = os.getenv('DB_NAME')
            db_user = os.getenv('DB_USER')
            db_password = os.getenv('DB_PASSWORD')
            
            # Validate required credentials
            if not all([db_host, db_name, db_user, db_password]):
                print("❌ Missing database credentials in environment variables")
                print("Please set DB_HOST, DB_NAME, DB_USER, and DB_PASSWORD")
                return False
            
            # Create connection string
            self.connection_string = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
            
            # Create engine with connection pooling
            self.engine = create_engine(
                self.connection_string,
                pool_size=5,
                max_overflow=10,
                pool_timeout=30,
                pool_recycle=3600
            )
            
            # Test connection
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            
            return True
            
        except Exception as e:
            print(f"❌ Database connection failed: {e}")
            return False
    
    def load_table(self, table_name):
        """Load data from a database table"""
        try:
            if self.engine is None:
                print("❌ Database not connected")
                return pd.DataFrame()
            
            query = f"SELECT * FROM {table_name}"
            df = pd.read_sql(query, self.engine)
            return df
            
        except Exception as e:
            print(f"❌ Error loading table {table_name}: {e}")
            return pd.DataFrame()
    
    def save_to_database(self, df, table_name, if_exists='replace'):
        """Save DataFrame to database table"""
        try:
            if self.engine is None:
                print("❌ Database not connected")
                return False
            
            # Save DataFrame to database
            df.to_sql(
                name=table_name,
                con=self.engine,
                if_exists=if_exists,
                index=False,
                method='multi'
            )
            
            return True
            
        except Exception as e:
            print(f"❌ Error saving to table {table_name}: {e}")
            return False
    
    def cleanup(self):
        """Clean up database connections"""
        if self.engine:
            self.engine.dispose()
            print("✅ Database connections closed") 