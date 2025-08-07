"""
Data Loader Module
Handles loading and basic preprocessing of car news and reviews datasets
"""

import pandas as pd
import numpy as np
import re
from src.config.config import NEWS_FILE, REVIEWS_FILE


class DataLoader:
    """Handles data loading and basic preprocessing"""
    
    def __init__(self, news_file=NEWS_FILE, reviews_file=REVIEWS_FILE, use_database=False, 
                 news_table="car_news", reviews_table="car_reviews"):
        """
        Initialize the data loader
        
        Args:
            news_file (str): Path to car news CSV file (fallback)
            reviews_file (str): Path to car reviews CSV file (fallback)
            use_database (bool): Whether to use database instead of CSV files
            news_table (str): Database table name for news data
            reviews_table (str): Database table name for reviews data
        """
        self.news_file = news_file
        self.reviews_file = reviews_file
        self.use_database = use_database
        self.news_table = news_table
        self.reviews_table = reviews_table
        self.car_news_df = None
        self.car_reviews_df = None
        
        # Initialize database config if using database
        if self.use_database:
            from src.config.config import DatabaseConfig
            self.db_config = DatabaseConfig()
        else:
            self.db_config = None
    
    def load_data(self):
        """Load car news and reviews datasets"""
        try:
            if self.use_database:
                return self._load_from_database()
            else:
                return self._load_from_csv()
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def _load_from_csv(self):
        """Load data from CSV files"""
        try:
            self.car_news_df = pd.read_csv(self.news_file)
            self.car_reviews_df = pd.read_csv(self.reviews_file)
            
            print(f"Car News Dataset: {self.car_news_df.shape}")
            print(f"Car Reviews Dataset: {self.car_reviews_df.shape}")
            
            # Display basic info
            print("\nCar News Columns:", self.car_news_df.columns.tolist())
            print("Car Reviews Columns:", self.car_reviews_df.columns.tolist())
            
            return True
        except FileNotFoundError as e:
            print(f"Error loading CSV files: {e}")
            print("Please ensure your CSV files are in the same directory as this script.")
            return False
    
    def _load_from_database(self):
        """Load data from database"""
        try:
            # Setup database connection if not already connected
            if self.db_config.engine is None:
                if not self.db_config.setup_from_env():
                    print("Failed to setup database connection")
                    return False
            
            # Load news data
            print(f"Loading news data from table: {self.news_table}")
            self.car_news_df = self.db_config.load_table(self.news_table)
            
            # Load reviews data
            print(f"Loading reviews data from table: {self.reviews_table}")
            self.car_reviews_df = self.db_config.load_table(self.reviews_table)
            
            if self.car_news_df.empty and self.car_reviews_df.empty:
                print("No data found in database tables")
                return False
            
            print(f"Car News Dataset: {self.car_news_df.shape}")
            print(f"Car Reviews Dataset: {self.car_reviews_df.shape}")
            
            # Display basic info
            if not self.car_news_df.empty:
                print("\nCar News Columns:", self.car_news_df.columns.tolist())
            if not self.car_reviews_df.empty:
                print("Car Reviews Columns:", self.car_reviews_df.columns.tolist())
            
            return True
            
        except Exception as e:
            print(f"Error loading from database: {e}")
            return False
    
    def preprocess_text(self, text_series):
        """
        Basic text preprocessing
        
        Args:
            text_series: Pandas series containing text data
            
        Returns:
            pd.Series: Preprocessed text series
        """
        def clean_text(text):
            if pd.isna(text):
                return ""
            # Convert to string and lowercase
            text = str(text).lower()
            # Remove special characters but keep spaces
            text = re.sub(r'[^a-zA-Z\s]', '', text)
            # Remove extra whitespace
            text = ' '.join(text.split())
            return text
        
        return text_series.apply(clean_text)
    
    def get_text_columns(self):
        """Get the text column names from both datasets"""
        text_columns = {}
        
        if self.car_news_df is not None:
            # Actual text column names for news
            news_text_cols = ['content', 'title', 'text', 'article', 'headline', 'body']
            for col in news_text_cols:
                if col in self.car_news_df.columns:
                    text_columns['news'] = col
                    break
        
        if self.car_reviews_df is not None:
            # Actual text column names for reviews
            review_text_cols = ['verdict', 'title', 'Review', 'Review_Title', 'text', 'review_text', 'comment', 'feedback', 'description']
            for col in review_text_cols:
                if col in self.car_reviews_df.columns:
                    text_columns['reviews'] = col
                    break
        
        return text_columns
    
    def get_date_columns(self):
        """Get the date column names from both datasets"""
        date_columns = {}
        
        if self.car_news_df is not None:
            date_cols_news = [col for col in self.car_news_df.columns if 'date' in col.lower()]
            if date_cols_news:
                date_columns['news'] = date_cols_news[0]
        
        if self.car_reviews_df is not None:
            # Actual date column name for reviews
            date_cols_reviews = ['Review_Date', 'date', 'created_date', 'published_date']
            for col in date_cols_reviews:
                if col in self.car_reviews_df.columns:
                    date_columns['reviews'] = col
                    break
        
        return date_columns
    
    def get_numerical_columns(self):
        """Get numerical column names from reviews dataset"""
        if self.car_reviews_df is not None:
            return self.car_reviews_df.select_dtypes(include=[np.number]).columns.tolist()
        return []
    
    def validate_data(self):
        """Validate that required columns exist in the datasets"""
        text_cols = self.get_text_columns()
        
        if not text_cols.get('news'):
            print("Warning: No text column found in news dataset")
        
        if not text_cols.get('reviews'):
            print("Warning: No text column found in reviews dataset")
        
        return len(text_cols) > 0
    
    def load_car_news(self):
        """
        Load car news dataset specifically
        
        Returns:
            pd.DataFrame: Car news dataframe or None if failed
        """
        try:
            if self.car_news_df is None:
                self.car_news_df = pd.read_csv(self.news_file)
            return self.car_news_df
        except FileNotFoundError as e:
            print(f"Error loading car news data: {e}")
            return None
    
    def load_car_reviews(self):
        """
        Load car reviews dataset specifically
        
        Returns:
            pd.DataFrame: Car reviews dataframe or None if failed
        """
        try:
            if self.car_reviews_df is None:
                self.car_reviews_df = pd.read_csv(self.reviews_file)
            return self.car_reviews_df
        except FileNotFoundError as e:
            print(f"Error loading car reviews data: {e}")
            return None


def main():
    """Test data loading functionality"""
    print("="*60)
    print("DATA LOADER TEST")
    print("="*60)
    
    try:
        # Initialize data loader
        data_loader = DataLoader()
        
        # Test loading car news dataset
        print("\n" + "="*40)
        print("TESTING CAR NEWS DATASET LOADING")
        print("="*40)
        
        car_news_df = data_loader.load_data()
        if car_news_df:
            print(f"✓ Car News Dataset loaded successfully!")
            print(f"  - Shape: {data_loader.car_news_df.shape}")
            print(f"  - Columns: {list(data_loader.car_news_df.columns)}")
            print(f"  - Sample data:")
            print(data_loader.car_news_df.head(2).to_string())
        else:
            print("❌ Failed to load car news dataset")
        
        # Test loading car reviews dataset
        print("\n" + "="*40)
        print("TESTING CAR REVIEWS DATASET LOADING")
        print("="*40)
        
        car_reviews_df = data_loader.load_data()
        if car_reviews_df:
            print(f"✓ Car Reviews Dataset loaded successfully!")
            print(f"  - Shape: {data_loader.car_reviews_df.shape}")
            print(f"  - Columns: {list(data_loader.car_reviews_df.columns)}")
            print(f"  - Sample data:")
            print(data_loader.car_reviews_df.head(2).to_string())
        else:
            print("❌ Failed to load car reviews dataset")
        
        # Test preprocessing
        print("\n" + "="*40)
        print("TESTING DATA PREPROCESSING")
        print("="*40)
        
        if data_loader.car_news_df is not None and data_loader.car_reviews_df is not None:
            processed_news = data_loader.preprocess_text(data_loader.car_news_df['content'])
            processed_reviews = data_loader.preprocess_text(data_loader.car_reviews_df['Review'])
            
            print(f"✓ News preprocessing completed!")
            print(f"  - Original shape: {data_loader.car_news_df.shape}")
            print(f"  - Processed shape: {processed_news.shape}")
            
            print(f"✓ Reviews preprocessing completed!")
            print(f"  - Original shape: {data_loader.car_reviews_df.shape}")
            print(f"  - Processed shape: {processed_reviews.shape}")
            
            # Show preprocessing results
            if 'brand' in data_loader.car_reviews_df.columns:
                print(f"\nBrand extraction results:")
                brand_counts = data_loader.car_reviews_df['brand'].value_counts().head(5)
                for brand, count in brand_counts.items():
                    print(f"  - {brand}: {count}")
        
        print("\n" + "="*60)
        print("DATA LOADER TEST COMPLETE!")
        print("="*60)
        
    except Exception as e:
        print(f"❌ Error during data loader test: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 