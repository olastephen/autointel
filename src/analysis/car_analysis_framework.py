#!/usr/bin/env python3
"""
Car News & Reviews Analysis Framework

This script provides comprehensive analysis of car news and reviews data,
extracting hidden insights useful for car sellers.

Key Features:
- Text Analysis & NLP Processing
- Sentiment Analysis
- Topic Modeling
- Named Entity Recognition
- Correlation Analysis
- Time Series Analysis
- Visualization and Reporting
- Database Integration for Results Storage
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# NLP Libraries
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.util import ngrams
import spacy
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Machine Learning
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

# Visualization
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from wordcloud import WordCloud

# Additional
from collections import Counter, defaultdict
import re
from datetime import datetime, timedelta
import json

# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('averaged_perceptron_tagger')

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class CarAnalysisFramework:
    def __init__(self, news_file='car_news_dataset.csv', reviews_file='car_reviews_dataset.csv',
                 use_database=True, news_table="car_news", reviews_table="car_reviews"):
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
        self.car_news_df = None
        self.car_reviews_df = None
        
        # Initialize data loader
        from src.data.data_loader import DataLoader
        self.data_loader = DataLoader(
            news_file=news_file,
            reviews_file=reviews_file,
            use_database=use_database,
            news_table=news_table,
            reviews_table=reviews_table
        )
        
        # Initialize database config if using database
        if self.use_database:
            from src.config.config import DatabaseConfig
            self.db_config = DatabaseConfig()
        else:
            self.db_config = None
        
        # Load spaCy model
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("spaCy model not found. Please install with: python -m spacy download en_core_web_sm")
            self.nlp = None
    
    def load_data(self):
        """Load car news and reviews datasets"""
        try:
            success = self.data_loader.load_data()
            if success:
                self.car_news_df = self.data_loader.car_news_df
                self.car_reviews_df = self.data_loader.car_reviews_df
            return True
            return False
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def preprocess_text(self, text_series):
        """Basic text preprocessing"""
        return self.data_loader.preprocess_text(text_series)
    
    def get_text_columns(self):
        """Get the text column names from both datasets"""
        return self.data_loader.get_text_columns()
    
    def perform_sentiment_analysis(self):
        """Perform sentiment analysis and save results to database"""
        print("\n=== Performing Sentiment Analysis ===")
        
        def get_sentiment_scores(text):
            if pd.isna(text) or str(text).strip() == "":
                return pd.Series([0, 0, 0, 0, 0, 0, 'neutral'])
            
            # VADER sentiment
            vader = SentimentIntensityAnalyzer()
            vs = vader.polarity_scores(str(text))
            
            # TextBlob sentiment
            blob = TextBlob(str(text))
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity
            
            # Label based on VADER compound score
            if vs['compound'] >= 0.05:
                label = 'positive'
            elif vs['compound'] <= -0.05:
                label = 'negative'
            else:
                label = 'neutral'
            
            return pd.Series([vs['compound'], vs['pos'], vs['neu'], vs['neg'], 
                            polarity, subjectivity, label])
        
        # Get text columns
        text_cols = self.get_text_columns()
        
        # Apply sentiment analysis to news
        if 'news' in text_cols and self.car_news_df is not None:
            news_text_col = text_cols['news']
            if news_text_col in self.car_news_df.columns:
                print(f"Analyzing sentiment for news using column: {news_text_col}")
                sentiment_cols = ['sentiment_score', 'vader_pos', 'vader_neu', 'vader_neg', 
                                'tb_polarity', 'tb_subjectivity', 'sentiment']
                self.car_news_df[sentiment_cols] = self.car_news_df[news_text_col].apply(get_sentiment_scores)
        
        # Apply sentiment analysis to reviews
        if 'reviews' in text_cols and self.car_reviews_df is not None:
            reviews_text_col = text_cols['reviews']
            if reviews_text_col in self.car_reviews_df.columns:
                print(f"Analyzing sentiment for reviews using column: {reviews_text_col}")
                sentiment_cols = ['sentiment_score', 'vader_pos', 'vader_neu', 'vader_neg', 
                                'tb_polarity', 'tb_subjectivity', 'sentiment']
                self.car_reviews_df[sentiment_cols] = self.car_reviews_df[reviews_text_col].apply(get_sentiment_scores)
        
        print("✓ Sentiment analysis completed")
    
    def perform_topic_modeling(self):
        """Perform topic modeling and save results to database"""
        print("\n=== Performing Topic Modeling ===")
        
        def extract_topics(text_series, n_topics=5, n_words=10):
            """Extract topics using LDA"""
            # Clean and prepare text
            clean_texts = self.preprocess_text(text_series)
            clean_texts = clean_texts[clean_texts != ""]
            
            if len(clean_texts) == 0:
                return []
            
            # Vectorize
            vectorizer = CountVectorizer(
                stop_words='english', 
                max_df=0.95, 
                min_df=2,
                max_features=1000
            )
            
            try:
                dtm = vectorizer.fit_transform(clean_texts)
                
                # LDA
                lda = LatentDirichletAllocation(
                    n_components=n_topics, 
                    random_state=42,
                    max_iter=50
                )
                lda.fit(dtm)
                
                # Get topics
                feature_names = vectorizer.get_feature_names_out()
                topics = []
                
                for topic_idx, topic in enumerate(lda.components_):
                    top_words_idx = topic.argsort()[-n_words:][::-1]
                    top_words = [feature_names[i] for i in top_words_idx]
                    topics.append(top_words)
                
                return topics
                
            except Exception as e:
                print(f"Error in topic modeling: {e}")
                return []
        
        # Get text columns
        text_cols = self.get_text_columns()
        
        # Apply topic modeling to news
        if 'news' in text_cols and self.car_news_df is not None:
            news_text_col = text_cols['news']
            if news_text_col in self.car_news_df.columns:
                print(f"Extracting topics for news using column: {news_text_col}")
                topics = extract_topics(self.car_news_df[news_text_col])
                # Store topics as JSON string
                self.car_news_df['topics'] = json.dumps(topics) if topics else None
        
        # Apply topic modeling to reviews
        if 'reviews' in text_cols and self.car_reviews_df is not None:
            reviews_text_col = text_cols['reviews']
            if reviews_text_col in self.car_reviews_df.columns:
                print(f"Extracting topics for reviews using column: {reviews_text_col}")
                topics = extract_topics(self.car_reviews_df[reviews_text_col])
                # Store topics as JSON string
                self.car_reviews_df['topics'] = json.dumps(topics) if topics else None
        
        print("✓ Topic modeling completed")
    
    def perform_ner_analysis(self):
        """Perform Named Entity Recognition and save results to database"""
        print("\n=== Performing Named Entity Recognition ===")
        
        if self.nlp is None:
            print("spaCy model not available. Skipping NER analysis.")
            return
        
        def extract_entities(text_series):
            """Extract entities from text"""
            # Car-related entities to look for
            car_brands = ['toyota', 'honda', 'ford', 'bmw', 'mercedes', 'audi', 'volkswagen', 
                         'nissan', 'hyundai', 'kia', 'chevrolet', 'dodge', 'jeep', 'tesla',
                         'lexus', 'acura', 'infiniti', 'cadillac', 'buick', 'chrysler']
            
            entities = []
            
            # Process all texts
            for text in text_series:
                if pd.isna(text) or str(text).strip() == "":
                    continue
                    
                doc = self.nlp(str(text))
                text_entities = []
                
                for ent in doc.ents:
                    if ent.label_ in ['ORG', 'GPE', 'PRODUCT']:
                        text_entities.append(f"{ent.text.lower()}:{ent.label_}")
                
                # Also look for car brands in the text
                text_lower = str(text).lower()
                for brand in car_brands:
                    if brand in text_lower:
                        text_entities.append(f"{brand}:BRAND")
                
                if text_entities:  # Only append if we found entities
                    entities.append(text_entities)
            
            return entities
        
        # Get text columns
        text_cols = self.get_text_columns()
        
        # Apply NER to news
        if 'news' in text_cols and self.car_news_df is not None:
            news_text_col = text_cols['news']
            if news_text_col in self.car_news_df.columns:
                print(f"Extracting entities for news using column: {news_text_col}")
                entities = extract_entities(self.car_news_df[news_text_col])
                # Store entities as JSON string
                self.car_news_df['entities'] = json.dumps(entities) if entities else None
        
        # Apply NER to reviews
        if 'reviews' in text_cols and self.car_reviews_df is not None:
            reviews_text_col = text_cols['reviews']
            if reviews_text_col in self.car_reviews_df.columns:
                print(f"Extracting entities for reviews using column: {reviews_text_col}")
                entities = extract_entities(self.car_reviews_df[reviews_text_col])
                # Store entities as JSON string
                self.car_reviews_df['entities'] = json.dumps(entities) if entities else None
        
        print("✓ Named Entity Recognition completed")
    
    def perform_keyword_analysis(self):
        """Perform keyword analysis and save results to database"""
        print("\n=== Performing Keyword Analysis ===")
        
        def extract_keywords(text_series, top_n=20):
            """Extract top keywords from text"""
            # Clean texts
            clean_texts = self.preprocess_text(text_series)
            
            # Combine all texts
            all_text = ' '.join(clean_texts.dropna())
            
            # Tokenize and remove stopwords with fallback handling
            try:
                stop_words = set(stopwords.words('english'))
            except LookupError:
                print("⚠️  NLTK stopwords not available, using basic stopwords")
                stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'}
            
            try:
                tokens = word_tokenize(all_text.lower())
            except LookupError:
                print("⚠️  NLTK punkt tokenizer not available, using simple tokenization")
                import re
                tokens = re.findall(r'\b[a-zA-Z]+\b', all_text.lower())
            
            tokens = [token for token in tokens if token.isalpha() and token not in stop_words]
            
            # Count frequencies
            word_freq = Counter(tokens)
            
            # Get top keywords
            keywords = [word for word, freq in word_freq.most_common(top_n)]
            
            return keywords
        
        # Get text columns
        text_cols = self.get_text_columns()
        
        # Apply keyword analysis to news
        if 'news' in text_cols and self.car_news_df is not None:
            news_text_col = text_cols['news']
            if news_text_col in self.car_news_df.columns:
                print(f"Extracting keywords for news using column: {news_text_col}")
                keywords = extract_keywords(self.car_news_df[news_text_col])
                # Store keywords as JSON string
                self.car_news_df['keywords'] = json.dumps(keywords) if keywords else None
        
        # Apply keyword analysis to reviews
        if 'reviews' in text_cols and self.car_reviews_df is not None:
            reviews_text_col = text_cols['reviews']
            if reviews_text_col in self.car_reviews_df.columns:
                print(f"Extracting keywords for reviews using column: {reviews_text_col}")
                keywords = extract_keywords(self.car_reviews_df[reviews_text_col])
                # Store keywords as JSON string
                self.car_reviews_df['keywords'] = json.dumps(keywords) if keywords else None
        
        print("✓ Keyword analysis completed")
    
    def perform_ngram_analysis(self):
        """Perform n-gram analysis and save results to database"""
        print("\n=== Performing N-gram Analysis ===")
        
        def extract_ngrams(text_series, n=2, top_n=15):
            """Extract top n-grams from text with better stopword filtering"""
            # Clean texts
            clean_texts = self.preprocess_text(text_series)
            
            # Enhanced stopwords list with fallback handling
            try:
                stop_words = set(stopwords.words('english'))
            except LookupError:
                print("⚠️  NLTK stopwords not available, using basic stopwords")
                stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'}
            
            # Add common car-related stopwords that are not meaningful
            car_stopwords = {'car', 'vehicle', 'auto', 'automotive', 'review', 'test', 'drive', 'model', 'new', 'good', 'bad', 'great', 'nice', 'best', 'worst', 'really', 'very', 'much', 'get', 'go', 'like', 'would', 'could', 'should', 'one', 'two', 'also', 'even', 'well', 'way', 'say', 'said', 'make', 'made', 'take', 'come', 'see', 'know', 'think', 'want', 'need', 'use', 'used', 'look', 'looks', 'feel', 'feels', 'seem', 'seems'}
            stop_words.update(car_stopwords)
            
            # Get n-grams
            all_ngrams = []
            for text in clean_texts:
                if text.strip():
                    try:
                        tokens = word_tokenize(text.lower())
                    except LookupError:
                        # Fallback to simple tokenization
                        import re
                        tokens = re.findall(r'\b[a-zA-Z]+\b', text.lower())
                    
                    # Remove stopwords and short words
                    filtered_tokens = [token for token in tokens if token.isalpha() and len(token) > 2 and token not in stop_words]
                    
                    if len(filtered_tokens) >= n:
                        ngram_list = list(ngrams(filtered_tokens, n))
                        # Filter n-grams that contain stopwords
                        clean_ngrams = []
                        for ngram in ngram_list:
                            if not any(word in stop_words for word in ngram):
                                clean_ngrams.append(ngram)
                        all_ngrams.extend(clean_ngrams)
            
            # Count frequencies
            ngram_freq = Counter(all_ngrams)
            
            # Get top n-grams
            top_ngrams = [' '.join(ng) for ng, freq in ngram_freq.most_common(top_n)]
            
            return top_ngrams
        
        # Get text columns
        text_cols = self.get_text_columns()
        
        # Apply n-gram analysis to news
        if 'news' in text_cols and self.car_news_df is not None:
            news_text_col = text_cols['news']
            if news_text_col in self.car_news_df.columns:
                print(f"Extracting bigrams for news using column: {news_text_col}")
                bigram_results = extract_ngrams(self.car_news_df[news_text_col], n=2, top_n=15)
                # Store bigrams as JSON string
                self.car_news_df['top_bigrams'] = json.dumps(bigram_results) if bigram_results else None
                
                print(f"Extracting trigrams for news using column: {news_text_col}")
                trigram_results = extract_ngrams(self.car_news_df[news_text_col], n=3, top_n=15)
                # Store trigrams as JSON string
                self.car_news_df['top_trigrams'] = json.dumps(trigram_results) if trigram_results else None
                
                # Keep the old column for backward compatibility
                combined_ngrams = bigram_results + trigram_results
                self.car_news_df['top_ngrams'] = json.dumps(combined_ngrams) if combined_ngrams else None
        
        # Apply n-gram analysis to reviews
        if 'reviews' in text_cols and self.car_reviews_df is not None:
            reviews_text_col = text_cols['reviews']
            if reviews_text_col in self.car_reviews_df.columns:
                print(f"Extracting bigrams for reviews using column: {reviews_text_col}")
                bigram_results = extract_ngrams(self.car_reviews_df[reviews_text_col], n=2, top_n=15)
                # Store bigrams as JSON string
                self.car_reviews_df['top_bigrams'] = json.dumps(bigram_results) if bigram_results else None
                
                print(f"Extracting trigrams for reviews using column: {reviews_text_col}")
                trigram_results = extract_ngrams(self.car_reviews_df[reviews_text_col], n=3, top_n=15)
                # Store trigrams as JSON string
                self.car_reviews_df['top_trigrams'] = json.dumps(trigram_results) if trigram_results else None
                
                # Keep the old column for backward compatibility
                combined_ngrams = bigram_results + trigram_results
                self.car_reviews_df['top_ngrams'] = json.dumps(combined_ngrams) if combined_ngrams else None
        
        print("✓ N-gram analysis completed")
    
    def perform_correlation_analysis(self):
        """Perform correlation analysis for reviews and save results to database"""
        print("\n=== Performing Correlation Analysis ===")
        
        if self.car_reviews_df is None:
            print("No reviews data available for correlation analysis")
            return
        
        # Check for numerical columns in reviews
        numerical_cols = self.car_reviews_df.select_dtypes(include=[np.number]).columns.tolist()
        sentiment_cols = ['sentiment_score', 'tb_polarity', 'tb_subjectivity']
        
        # Add sentiment columns if they exist
        available_cols = []
        for col in sentiment_cols:
            if col in self.car_reviews_df.columns:
                available_cols.append(col)
        
        # Combine numerical and sentiment columns
        analysis_cols = numerical_cols + available_cols
        
        if len(analysis_cols) < 2:
            print("Not enough numerical columns for correlation analysis")
            return
        
        # Calculate correlation with rating if available
        if 'rating' in self.car_reviews_df.columns:
            for col in available_cols:
                if col in self.car_reviews_df.columns:
                    correlation = self.car_reviews_df['rating'].corr(self.car_reviews_df[col])
                    # Store correlation score
                    self.car_reviews_df['correlation_score'] = correlation
        
        print("✓ Correlation analysis completed")
    
    def calculate_review_length(self):
        """Calculate review length and save to database"""
        print("\n=== Calculating Review Length ===")
        
        # Get text columns
        text_cols = self.get_text_columns()
        
        # Calculate review length for reviews
        if 'reviews' in text_cols and self.car_reviews_df is not None:
            reviews_text_col = text_cols['reviews']
            if reviews_text_col in self.car_reviews_df.columns:
                print(f"Calculating review length using column: {reviews_text_col}")
                self.car_reviews_df['review_length'] = self.car_reviews_df[reviews_text_col].str.len()
        
        print("✓ Review length calculation completed")
    
    def save_results_to_database(self):
        """Save analysis results back to the database"""
        print("\n=== Saving Results to Database ===")
        
        if not self.use_database or self.db_config is None:
            print("Database mode not enabled. Skipping save to database.")
            return
        
        try:
            # Setup database connection if not already connected
            if self.db_config.engine is None:
                if not self.db_config.setup_from_env():
                    print("Failed to setup database connection")
                    return False
            
            # Save news results
            if self.car_news_df is not None and not self.car_news_df.empty:
                print(f"Saving news analysis results to table: {self.news_table}")
                self.db_config.save_to_database(self.car_news_df, self.news_table, if_exists='replace')
            
            # Save reviews results
            if self.car_reviews_df is not None and not self.car_reviews_df.empty:
                print(f"Saving reviews analysis results to table: {self.reviews_table}")
                self.db_config.save_to_database(self.car_reviews_df, self.reviews_table, if_exists='replace')
            
            print("✓ Results saved to database successfully")
            return True
                
        except Exception as e:
            print(f"Error saving results to database: {e}")
            return False
    
    def generate_analysis_summary(self):
        """Generate a summary of the analysis results"""
        print("\n" + "="*60)
        print("ANALYSIS SUMMARY")
        print("="*60)
        
        # Basic statistics
        if self.car_news_df is not None:
            print(f"\nCar News Dataset: {self.car_news_df.shape}")
            if 'sentiment' in self.car_news_df.columns:
                sentiment_dist = self.car_news_df['sentiment'].value_counts()
                print("News Sentiment Distribution:")
            for sentiment, count in sentiment_dist.items():
                percentage = (count / len(self.car_news_df)) * 100
                print(f"- {sentiment.capitalize()}: {count} ({percentage:.1f}%)")
        
        if self.car_reviews_df is not None:
            print(f"\nCar Reviews Dataset: {self.car_reviews_df.shape}")
            if 'sentiment' in self.car_reviews_df.columns:
                sentiment_dist = self.car_reviews_df['sentiment'].value_counts()
                print("Review Sentiment Distribution:")
            for sentiment, count in sentiment_dist.items():
                percentage = (count / len(self.car_reviews_df)) * 100
                print(f"- {sentiment.capitalize()}: {count} ({percentage:.1f}%)")
            
            if 'rating' in self.car_reviews_df.columns:
                print(f"\nReview Rating Statistics:")
                print(f"- Average Rating: {self.car_reviews_df['rating'].mean():.2f}")
                print(f"- Rating Range: {self.car_reviews_df['rating'].min():.1f} - {self.car_reviews_df['rating'].max():.1f}")
        
        print("\n" + "="*60)
        print("Analysis Complete!")
        print("="*60)
    
    def run_full_analysis(self):
        """Run the complete analysis pipeline"""
        print("🚀 Starting Car Analysis Framework")
        print("="*50)
        
        # Load data
        if not self.load_data():
            print("❌ Failed to load data. Exiting.")
            return False
        
        print("✓ Data loaded successfully")
        
        # Perform all analyses
        self.perform_sentiment_analysis()
        self.perform_topic_modeling()
        self.perform_ner_analysis()
        self.perform_keyword_analysis()
        self.perform_ngram_analysis()
        self.perform_correlation_analysis()
        self.calculate_review_length()
        
        # Save results to database
        if self.use_database:
            self.save_results_to_database()
        
        # Generate summary
        self.generate_analysis_summary()
        
        return True


def main():
    """Main function to run the analysis"""
    import os
    
    print("🔌 Using database mode")
    # Initialize the framework with database
    analyzer = CarAnalysisFramework(
        news_file='datasets/car_news_dataset.csv',
        reviews_file='datasets/car_reviews_dataset.csv',
        use_database=True,
        news_table=os.getenv('NEWS_TABLE', 'car_news'),
        reviews_table=os.getenv('REVIEWS_TABLE', 'car_reviews')
    )
    
    # Run the complete analysis
    analyzer.run_full_analysis()


if __name__ == "__main__":
    main() 