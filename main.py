#!/usr/bin/env python3
"""
Main Script for Car Analysis Framework
Orchestrates all analysis modules and generates comprehensive reports
"""

import warnings
warnings.filterwarnings('ignore')

# Import analysis modules
from src.data.data_loader import DataLoader
from src.features.ngram_analyzer import NgramAnalyzer
from src.features.sentiment_analyzer import SentimentAnalyzer
from src.features.topic_modeler import TopicModeler
from src.features.ner_analyzer import NERAnalyzer
from src.features.correlation_analyzer import CorrelationAnalyzer
from src.features.keyword_analyzer import KeywordAnalyzer
from src.features.time_series_analyzer import TimeSeriesAnalyzer

# Import configuration
from src.config.config import *

# Set up visualization
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
plt.style.use(PLOT_STYLE)
sns.set_palette(COLOR_PALETTE)


class CarAnalysisFramework:
    """Main framework that orchestrates all analysis components"""
    
    def __init__(self, news_file=NEWS_FILE, reviews_file=REVIEWS_FILE, use_database=False, 
                 news_table="car_news", reviews_table="car_reviews"):
        """
        Initialize the analysis framework
        
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
        
        # Initialize analysis modules
        self.data_loader = DataLoader(
            news_file=news_file, 
            reviews_file=reviews_file,
            use_database=use_database,
            news_table=news_table,
            reviews_table=reviews_table
        )
        self.ngram_analyzer = NgramAnalyzer()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.topic_modeler = TopicModeler()
        self.ner_analyzer = NERAnalyzer()
        self.correlation_analyzer = CorrelationAnalyzer()
        self.keyword_analyzer = KeywordAnalyzer()
        self.time_series_analyzer = TimeSeriesAnalyzer()
        
        # Store results
        self.analysis_results = {}
    
    def run_full_analysis(self):
        """Run the complete analysis pipeline"""
        print("="*60)
        print("CAR NEWS & REVIEWS ANALYSIS FRAMEWORK")
        print("="*60)
        
        # Step 1: Load and validate data
        if not self._load_and_validate_data():
            return
        
        # Step 2: Preprocess data
        print("\nPreprocessing data...")
        self._preprocess_news_data()
        self._preprocess_reviews_data()
        
        # Step 3: Get column information
        text_columns = self.data_loader.get_text_columns()
        date_columns = self.data_loader.get_date_columns()
        
        print(f"\nDetected text columns: {text_columns}")
        print(f"Detected date columns: {date_columns}")
        
        # Step 4: Run analyses
        self._run_ngram_analysis(text_columns)
        self._run_sentiment_analysis(text_columns)
        self._run_topic_modeling(text_columns)
        self._run_ner_analysis(text_columns)
        self._run_correlation_analysis()
        self._run_keyword_analysis(text_columns)
        self._run_time_series_analysis(text_columns, date_columns)
        self._run_brand_analysis()
        
        # Step 5: Generate comprehensive report
        self._generate_comprehensive_report()
    
    def _load_and_validate_data(self):
        """Load and validate the datasets"""
        print("\nLoading datasets...")
        
        if not self.data_loader.load_data():
            return False
        
        if not self.data_loader.validate_data():
            print("Warning: Data validation failed, but continuing with available data")
        
        return True
    
    def _run_ngram_analysis(self, text_columns):
        """Run n-gram analysis on available text data"""
        print("\n" + "="*40)
        print("RUNNING N-GRAM ANALYSIS")
        print("="*40)
        
        if 'news' in text_columns:
            news_text = self.data_loader.car_news_df[text_columns['news']]
            self.ngram_analyzer.analyze_ngrams(news_text, "Car News")
            self.analysis_results['news_ngrams'] = self.ngram_analyzer.get_ngram_insights(news_text, "Car News")
        
        if 'reviews' in text_columns:
            review_text = self.data_loader.car_reviews_df[text_columns['reviews']]
            self.ngram_analyzer.analyze_ngrams(review_text, "Car Reviews")
            self.analysis_results['review_ngrams'] = self.ngram_analyzer.get_ngram_insights(review_text, "Car Reviews")
    
    def _run_sentiment_analysis(self, text_columns):
        """Run sentiment analysis on available text data"""
        print("\n" + "="*40)
        print("RUNNING SENTIMENT ANALYSIS")
        print("="*40)
        
        if 'news' in text_columns:
            self.data_loader.car_news_df = self.sentiment_analyzer.analyze_sentiment(
                self.data_loader.car_news_df, text_columns['news'], "Car News"
            )
            self.analysis_results['news_sentiment'] = self.sentiment_analyzer.get_sentiment_insights(
                self.data_loader.car_news_df, "Car News"
            )
        
        if 'reviews' in text_columns:
            self.data_loader.car_reviews_df = self.sentiment_analyzer.analyze_sentiment(
                self.data_loader.car_reviews_df, text_columns['reviews'], "Car Reviews"
            )
            self.analysis_results['review_sentiment'] = self.sentiment_analyzer.get_sentiment_insights(
                self.data_loader.car_reviews_df, "Car Reviews"
            )
    
    def _run_topic_modeling(self, text_columns):
        """Run topic modeling on available text data"""
        print("\n" + "="*40)
        print("RUNNING TOPIC MODELING")
        print("="*40)
        
        if 'news' in text_columns:
            news_text = self.data_loader.car_news_df[text_columns['news']]
            self.topic_modeler.analyze_topics(news_text, "Car News")
            self.analysis_results['news_topics'] = self.topic_modeler.get_topic_insights(news_text, "Car News")
        
        if 'reviews' in text_columns:
            review_text = self.data_loader.car_reviews_df[text_columns['reviews']]
            self.topic_modeler.analyze_topics(review_text, "Car Reviews")
            self.analysis_results['review_topics'] = self.topic_modeler.get_topic_insights(review_text, "Car Reviews")
    
    def _run_ner_analysis(self, text_columns):
        """Run Named Entity Recognition on available text data"""
        print("\n" + "="*40)
        print("RUNNING NAMED ENTITY RECOGNITION")
        print("="*40)
        
        if 'news' in text_columns:
            news_text = self.data_loader.car_news_df[text_columns['news']]
            self.ner_analyzer.extract_entities(news_text, "Car News")
            self.analysis_results['news_ner'] = self.ner_analyzer.get_ner_insights(news_text, "Car News")
            self.ner_analyzer.analyze_brand_mentions(news_text, "Car News")
        
        if 'reviews' in text_columns:
            review_text = self.data_loader.car_reviews_df[text_columns['reviews']]
            self.ner_analyzer.extract_entities(review_text, "Car Reviews")
            self.analysis_results['review_ner'] = self.ner_analyzer.get_ner_insights(review_text, "Car Reviews")
            self.ner_analyzer.analyze_brand_mentions(review_text, "Car Reviews")
    
    def _run_correlation_analysis(self):
        """Run correlation analysis on numerical data"""
        print("\n" + "="*40)
        print("RUNNING CORRELATION ANALYSIS")
        print("="*40)
        
        # Analyze correlations in reviews data
        if self.data_loader.car_reviews_df is not None:
            self.analysis_results['review_correlations'] = self.correlation_analyzer.get_correlation_insights(
                self.data_loader.car_reviews_df, "Car Reviews"
            )
        
        # Analyze correlations in news data
        if self.data_loader.car_news_df is not None:
            self.analysis_results['news_correlations'] = self.correlation_analyzer.get_correlation_insights(
                self.data_loader.car_news_df, "Car News"
            )
    
    def _run_keyword_analysis(self, text_columns):
        """Run keyword analysis on available text data"""
        print("\n" + "="*40)
        print("RUNNING KEYWORD ANALYSIS")
        print("="*40)
        
        if 'news' in text_columns:
            news_text = self.data_loader.car_news_df[text_columns['news']]
            self.analysis_results['news_keywords'] = self.keyword_analyzer.get_keyword_insights(news_text, "Car News")
        
        if 'reviews' in text_columns:
            review_text = self.data_loader.car_reviews_df[text_columns['reviews']]
            self.analysis_results['review_keywords'] = self.keyword_analyzer.get_keyword_insights(review_text, "Car Reviews")
    
    def _run_time_series_analysis(self, text_columns, date_columns):
        """Run time series analysis on available data"""
        print("\n" + "="*40)
        print("RUNNING TIME SERIES ANALYSIS")
        print("="*40)
        
        # Handle reviews time series analysis
        if 'reviews' in text_columns and 'reviews' in date_columns:
            # Clean the date format for reviews
            self._clean_review_dates()
            
            review_text_col = text_columns['reviews']
            review_date_col = date_columns['reviews']
            
            self.analysis_results['review_time_series'] = self.time_series_analyzer.get_time_series_insights(
                self.data_loader.car_reviews_df, review_text_col, review_date_col, "Car Reviews"
            )
        
        # Handle news time series analysis if date column exists
        if 'news' in text_columns and 'news' in date_columns:
            news_text_col = text_columns['news']
            news_date_col = date_columns['news']
            
            self.analysis_results['news_time_series'] = self.time_series_analyzer.get_time_series_insights(
                self.data_loader.car_news_df, news_text_col, news_date_col, "Car News"
            )
    
    def _clean_review_dates(self):
        """Clean the review date format to make it parseable"""
        if 'Review_Date' in self.data_loader.car_reviews_df.columns:
            # The date format is "on MM/DD/YY HH:MM PM (PST)"
            # Extract just the date part and convert to standard format
            def clean_date(date_str):
                if pd.isna(date_str):
                    return None
                try:
                    # Extract date part from "on MM/DD/YY HH:MM PM (PST)"
                    date_part = str(date_str).split(' on ')[-1].split(' ')[0]
                    # Convert MM/DD/YY to YYYY-MM-DD
                    month, day, year = date_part.split('/')
                    year = '20' + year if int(year) < 50 else '19' + year
                    return f"{year}-{month.zfill(2)}-{day.zfill(2)}"
                except:
                    return None
            
            self.data_loader.car_reviews_df['Review_Date_Clean'] = self.data_loader.car_reviews_df['Review_Date'].apply(clean_date)
            # Update the date column reference
            self.data_loader.car_reviews_df['Review_Date'] = pd.to_datetime(self.data_loader.car_reviews_df['Review_Date_Clean'], errors='coerce')
    
    def _preprocess_reviews_data(self):
        """Preprocess reviews data for better analysis"""
        if self.data_loader.car_reviews_df is not None:
            # Clean Rating column - ensure it's numeric
            if 'Rating' in self.data_loader.car_reviews_df.columns:
                self.data_loader.car_reviews_df['Rating'] = pd.to_numeric(
                    self.data_loader.car_reviews_df['Rating'], errors='coerce'
                )
            
            # Extract brand and model from Vehicle_Title
            if 'Vehicle_Title' in self.data_loader.car_reviews_df.columns:
                def extract_brand_model(vehicle_title):
                    if pd.isna(vehicle_title):
                        return None, None
                    try:
                        # Example: "1997 Toyota Previa Minivan LE 3dr Minivan"
                        parts = str(vehicle_title).split()
                        if len(parts) >= 2:
                            brand = parts[1]  # Toyota
                            model = parts[2] if len(parts) > 2 else None  # Previa
                            return brand, model
                        return None, None
                    except:
                        return None, None
                
                brand_model = self.data_loader.car_reviews_df['Vehicle_Title'].apply(extract_brand_model)
                self.data_loader.car_reviews_df['Brand'] = [bm[0] for bm in brand_model]
                self.data_loader.car_reviews_df['Model'] = [bm[1] for bm in brand_model]
            
            # Clean Author_Name
            if 'Author_Name' in self.data_loader.car_reviews_df.columns:
                self.data_loader.car_reviews_df['Author_Name'] = self.data_loader.car_reviews_df['Author_Name'].str.strip()
    
    def _preprocess_news_data(self):
        """Preprocess news data for better analysis"""
        if self.data_loader.car_news_df is not None:
            # Clean title and content
            if 'title' in self.data_loader.car_news_df.columns:
                self.data_loader.car_news_df['title'] = self.data_loader.car_news_df['title'].str.strip()
            
            if 'content' in self.data_loader.car_news_df.columns:
                self.data_loader.car_news_df['content'] = self.data_loader.car_news_df['content'].str.strip()
    
    def _run_brand_analysis(self):
        """Run brand-specific analysis for car sellers"""
        print("\n" + "="*40)
        print("RUNNING BRAND-SPECIFIC ANALYSIS")
        print("="*40)
        
        if self.data_loader.car_reviews_df is not None and 'Brand' in self.data_loader.car_reviews_df.columns:
            # Brand sentiment analysis
            brand_sentiment = self.data_loader.car_reviews_df.groupby('Brand').agg({
                'vader_compound': ['mean', 'count'],
                'Rating': 'mean',
                'Review': 'count'
            }).round(3)
            
            # Flatten column names
            brand_sentiment.columns = ['_'.join(col).strip() for col in brand_sentiment.columns]
            
            # Filter brands with sufficient reviews
            brand_sentiment = brand_sentiment[brand_sentiment['vader_compound_count'] >= 5]
            
            if not brand_sentiment.empty:
                # Plot brand sentiment comparison
                plt.figure(figsize=(12, 8))
                
                # Create subplots
                fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
                
                # Brand sentiment scores
                brand_sentiment['vader_compound_mean'].sort_values(ascending=True).plot(
                    kind='barh', ax=ax1, color='skyblue'
                )
                ax1.set_title('Brand Sentiment Scores')
                ax1.set_xlabel('Average VADER Compound Score')
                
                # Brand ratings
                brand_sentiment['Rating_mean'].sort_values(ascending=True).plot(
                    kind='barh', ax=ax2, color='lightgreen'
                )
                ax2.set_title('Brand Average Ratings')
                ax2.set_xlabel('Average Rating')
                
                # Review counts
                brand_sentiment['Review_count'].sort_values(ascending=True).plot(
                    kind='barh', ax=ax3, color='orange'
                )
                ax3.set_title('Number of Reviews by Brand')
                ax3.set_xlabel('Number of Reviews')
                
                # Sentiment vs Rating correlation
                ax4.scatter(brand_sentiment['vader_compound_mean'], brand_sentiment['Rating_mean'])
                ax4.set_xlabel('Average Sentiment Score')
                ax4.set_ylabel('Average Rating')
                ax4.set_title('Sentiment vs Rating Correlation')
                
                # Add brand labels to scatter plot
                for brand in brand_sentiment.index:
                    ax4.annotate(brand, 
                               (brand_sentiment.loc[brand, 'vader_compound_mean'], 
                                brand_sentiment.loc[brand, 'Rating_mean']),
                               xytext=(5, 5), textcoords='offset points', fontsize=8)
                
                plt.tight_layout()
                plt.show()
                
                # Print brand insights
                print("\nTop Brands by Sentiment Score:")
                top_sentiment = brand_sentiment['vader_compound_mean'].sort_values(ascending=False).head(10)
                for brand, score in top_sentiment.items():
                    print(f"- {brand}: {score:.3f}")
                
                print("\nTop Brands by Average Rating:")
                top_rating = brand_sentiment['Rating_mean'].sort_values(ascending=False).head(10)
                for brand, rating in top_rating.items():
                    print(f"- {brand}: {rating:.2f}")
                
                # Store results
                self.analysis_results['brand_analysis'] = {
                    'brand_sentiment': brand_sentiment.to_dict(),
                    'top_sentiment_brands': top_sentiment.to_dict(),
                    'top_rated_brands': top_rating.to_dict()
                }
            else:
                print("No brands with sufficient reviews found for analysis")
    
    def _generate_comprehensive_report(self):
        """Generate a comprehensive analysis report"""
        print("\n" + "="*60)
        print("COMPREHENSIVE ANALYSIS REPORT")
        print("="*60)
        
        # Basic statistics
        print(f"\nDataset Overview:")
        if self.data_loader.car_news_df is not None:
            print(f"- Car News Articles: {len(self.data_loader.car_news_df)}")
        if self.data_loader.car_reviews_df is not None:
            print(f"- Car Reviews: {len(self.data_loader.car_reviews_df)}")
        
        # Sentiment summary
        if 'news_sentiment' in self.analysis_results:
            sentiment_data = self.analysis_results['news_sentiment']
            if 'sentiment_distribution' in sentiment_data:
                print(f"\nNews Sentiment Summary:")
                for sentiment, count in sentiment_data['sentiment_distribution'].items():
                    percentage = (count / sentiment_data['total_documents']) * 100
                    print(f"- {sentiment.capitalize()}: {count} ({percentage:.1f}%)")
        
        if 'review_sentiment' in self.analysis_results:
            sentiment_data = self.analysis_results['review_sentiment']
            if 'sentiment_distribution' in sentiment_data:
                print(f"\nReview Sentiment Summary:")
                for sentiment, count in sentiment_data['sentiment_distribution'].items():
                    percentage = (count / sentiment_data['total_documents']) * 100
                    print(f"- {sentiment.capitalize()}: {count} ({percentage:.1f}%)")
        
        # Key insights summary
        print(f"\nKey Insights Summary:")
        
        # Top topics
        if 'news_topics' in self.analysis_results and self.analysis_results['news_topics']:
            print(f"- News Topics Discovered: {self.analysis_results['news_topics'].get('n_topics', 0)}")
        
        if 'review_topics' in self.analysis_results and self.analysis_results['review_topics']:
            print(f"- Review Topics Discovered: {self.analysis_results['review_topics'].get('n_topics', 0)}")
        
        # Correlation insights
        if 'review_correlations' in self.analysis_results:
            corr_data = self.analysis_results['review_correlations']
            if 'significant_correlations_count' in corr_data:
                print(f"- Significant Correlations Found: {corr_data['significant_correlations_count']}")
        
        # Time series insights
        if 'news_time_series' in self.analysis_results:
            ts_data = self.analysis_results['news_time_series']
            if 'date_range' in ts_data and ts_data['date_range']['start']:
                print(f"- News Date Range: {ts_data['date_range']['start']} to {ts_data['date_range']['end']}")
        
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE!")
        print("="*60)
        print("\nAll visualizations and insights have been generated.")
        print("Check the plots above for detailed analysis results.")
    
    def save_results(self, filename="analysis_results.json"):
        """Save analysis results to a JSON file"""
        import json
        
        # Convert results to JSON-serializable format
        serializable_results = {}
        for key, value in self.analysis_results.items():
            try:
                # Convert pandas objects to basic Python types
                if hasattr(value, 'to_dict'):
                    serializable_results[key] = value.to_dict()
                elif isinstance(value, dict):
                    serializable_results[key] = value
                else:
                    serializable_results[key] = str(value)
            except:
                serializable_results[key] = str(value)
        
        with open(filename, 'w') as f:
            json.dump(serializable_results, f, indent=2, default=str)
        
        print(f"\nAnalysis results saved to {filename}")


def main():
    """Main function to run the analysis"""
    import os
    
    print("🔌 Using database mode")
    # Initialize the framework with database
    analyzer = CarAnalysisFramework(
        news_file=NEWS_FILE,
        reviews_file=REVIEWS_FILE,
        use_database=True,
        news_table=os.getenv('NEWS_TABLE', 'car_news'),
        reviews_table=os.getenv('REVIEWS_TABLE', 'car_reviews')
    )
    
    # Run the complete analysis
    analyzer.run_full_analysis()
    
    # Optionally save results
    # analyzer.save_results()


if __name__ == "__main__":
    main() 