#!/usr/bin/env python3
"""
Main Entry Point for Car Analysis Framework
Run this script to execute the complete analysis pipeline for both car news and car reviews

This script:
1. Loads data from both car_news and car_reviews database tables
2. Performs comprehensive analysis on both datasets
3. Saves analysis results back to the database
4. Provides detailed reporting on the analysis process

Usage:
    python run_analysis.py
"""

import warnings
warnings.filterwarnings('ignore')

import os
import sys
import time
from datetime import datetime

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def main():
    """Main function to run the complete analysis pipeline for both datasets"""
    
    print("🚀 CAR ANALYSIS PIPELINE - NEWS & REVIEWS")
    print("=" * 70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    start_time = time.time()
    
    try:
        # Import the analysis framework
        from src.analysis.car_analysis_framework import CarAnalysisFramework
        
        # Initialize the framework with database mode for both tables
        print("\n📋 Initializing Analysis Framework")
        print("-" * 50)
        
        analyzer = CarAnalysisFramework(
            news_file='datasets/car_news_dataset.csv',
            reviews_file='datasets/car_reviews_dataset.csv',
            use_database=True,
            news_table=os.getenv('NEWS_TABLE', 'car_news'),
            reviews_table=os.getenv('REVIEWS_TABLE', 'car_reviews')
        )
        
        print("✓ Analysis framework initialized")
        print(f"  - News table: {analyzer.news_table}")
        print(f"  - Reviews table: {analyzer.reviews_table}")
        
        # Load data from both tables
        print("\n📊 Loading Data from Database")
        print("-" * 50)
        
        if not analyzer.load_data():
            print("❌ Failed to load data from database")
            return False
        
        print("✓ Data loaded successfully")
        
        # Display data overview
        if analyzer.car_news_df is not None:
            print(f"  📰 Car News: {analyzer.car_news_df.shape[0]} articles, {analyzer.car_news_df.shape[1]} columns")
        else:
            print("  ❌ Car News data not loaded")
            
        if analyzer.car_reviews_df is not None:
            print(f"  ⭐ Car Reviews: {analyzer.car_reviews_df.shape[0]} reviews, {analyzer.car_reviews_df.shape[1]} columns")
        else:
            print("  ❌ Car Reviews data not loaded")
        
        # Perform comprehensive analysis on both datasets
        print("\n🔍 PERFORMING COMPREHENSIVE ANALYSIS")
        print("=" * 70)
        
        # 1. Sentiment Analysis for both datasets
        print("\n😊 Step 1: Sentiment Analysis")
        print("-" * 30)
        analyzer.perform_sentiment_analysis()
        print("✓ Sentiment analysis completed for both news and reviews")
        
        # 2. Topic Modeling for both datasets
        print("\n📝 Step 2: Topic Modeling")
        print("-" * 30)
        analyzer.perform_topic_modeling()
        print("✓ Topic modeling completed for both news and reviews")
        
        # 3. Named Entity Recognition for both datasets
        print("\n🏷️ Step 3: Named Entity Recognition")
        print("-" * 30)
        analyzer.perform_ner_analysis()
        print("✓ NER completed for both news and reviews")
        
        # 4. Keyword Analysis for both datasets
        print("\n🔑 Step 4: Keyword Analysis")
        print("-" * 30)
        analyzer.perform_keyword_analysis()
        print("✓ Keyword analysis completed for both news and reviews")
        
        # 5. N-gram Analysis for both datasets
        print("\n📊 Step 5: N-gram Analysis")
        print("-" * 30)
        analyzer.perform_ngram_analysis()
        print("✓ N-gram analysis completed for both news and reviews")
        
        # 6. Correlation Analysis (primarily for reviews)
        print("\n📈 Step 6: Correlation Analysis")
        print("-" * 30)
        analyzer.perform_correlation_analysis()
        print("✓ Correlation analysis completed")
        
        # 7. Review Length Calculation (for reviews)
        print("\n📏 Step 7: Review Length Calculation")
        print("-" * 30)
        analyzer.calculate_review_length()
        print("✓ Review length calculation completed")
        
        # Save results back to database
        print("\n💾 Saving Results to Database")
        print("-" * 50)
        
        if analyzer.save_results_to_database():
            print("✓ Results saved to database successfully")
            print(f"  - Updated table: {analyzer.news_table}")
            print(f"  - Updated table: {analyzer.reviews_table}")
        else:
            print("❌ Failed to save results to database")
            return False
        
        # Generate comprehensive analysis summary
        print("\n📋 Generating Analysis Summary")
        print("-" * 50)
        analyzer.generate_analysis_summary()
        
        # Calculate execution time
        end_time = time.time()
        duration = end_time - start_time
        
        # Final summary
        print("\n" + "=" * 70)
        print("🎉 ANALYSIS PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        print(f"Total execution time: {duration:.2f} seconds")
        print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70)
        
        # Detailed results summary
        print("\n📊 DETAILED RESULTS SUMMARY")
        print("-" * 50)
        
        # News analysis results
        if analyzer.car_news_df is not None:
            print(f"\n📰 CAR NEWS ANALYSIS RESULTS:")
            print(f"  - Articles processed: {len(analyzer.car_news_df):,}")
            
            if 'sentiment' in analyzer.car_news_df.columns:
                sentiment_dist = analyzer.car_news_df['sentiment'].value_counts()
                print(f"  - Sentiment distribution:")
                for sentiment, count in sentiment_dist.items():
                    percentage = (count / len(analyzer.car_news_df)) * 100
                    print(f"    • {sentiment.capitalize()}: {count:,} ({percentage:.1f}%)")
            
            if 'topics' in analyzer.car_news_df.columns:
                topics_count = analyzer.car_news_df['topics'].notna().sum()
                print(f"  - Topics extracted: {topics_count:,} articles")
            
            if 'entities' in analyzer.car_news_df.columns:
                entities_count = analyzer.car_news_df['entities'].notna().sum()
                print(f"  - Entities identified: {entities_count:,} articles")
            
            if 'keywords' in analyzer.car_news_df.columns:
                keywords_count = analyzer.car_news_df['keywords'].notna().sum()
                print(f"  - Keywords extracted: {keywords_count:,} articles")
        
        # Reviews analysis results
        if analyzer.car_reviews_df is not None:
            print(f"\n⭐ CAR REVIEWS ANALYSIS RESULTS:")
            print(f"  - Reviews processed: {len(analyzer.car_reviews_df):,}")
            
            if 'sentiment' in analyzer.car_reviews_df.columns:
                sentiment_dist = analyzer.car_reviews_df['sentiment'].value_counts()
                print(f"  - Sentiment distribution:")
                for sentiment, count in sentiment_dist.items():
                    percentage = (count / len(analyzer.car_reviews_df)) * 100
                    print(f"    • {sentiment.capitalize()}: {count:,} ({percentage:.1f}%)")
            
            if 'rating' in analyzer.car_reviews_df.columns:
                avg_rating = analyzer.car_reviews_df['rating'].mean()
                print(f"  - Average rating: {avg_rating:.2f} stars")
                print(f"  - Rating range: {analyzer.car_reviews_df['rating'].min():.1f} - {analyzer.car_reviews_df['rating'].max():.1f}")
            
            if 'topics' in analyzer.car_reviews_df.columns:
                topics_count = analyzer.car_reviews_df['topics'].notna().sum()
                print(f"  - Topics extracted: {topics_count:,} reviews")
            
            if 'entities' in analyzer.car_reviews_df.columns:
                entities_count = analyzer.car_reviews_df['entities'].notna().sum()
                print(f"  - Entities identified: {entities_count:,} reviews")
            
            if 'keywords' in analyzer.car_reviews_df.columns:
                keywords_count = analyzer.car_reviews_df['keywords'].notna().sum()
                print(f"  - Keywords extracted: {keywords_count:,} reviews")
            
            if 'correlation_score' in analyzer.car_reviews_df.columns:
                corr_score = analyzer.car_reviews_df['correlation_score'].iloc[0]
                print(f"  - Rating-sentiment correlation: {corr_score:.3f}")
            
            if 'review_length' in analyzer.car_reviews_df.columns:
                avg_length = analyzer.car_reviews_df['review_length'].mean()
                print(f"  - Average review length: {avg_length:.0f} characters")
        
        print("\n💾 DATABASE TABLES UPDATED:")
        print(f"  - {analyzer.news_table}: Enhanced with sentiment, topics, entities, keywords, n-grams")
        print(f"  - {analyzer.reviews_table}: Enhanced with sentiment, topics, entities, keywords, correlations, review length, n-grams")
        
        print("\n🎯 NEXT STEPS:")
        print("  - Launch Streamlit dashboard: streamlit run streamlit_app.py")
        print("  - Query analysis results: python scripts/query_analysis_results.py")
        print("  - Explore insights in the interactive dashboard")
        
        return True
        
    except Exception as e:
        print(f"\n❌ PIPELINE FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
