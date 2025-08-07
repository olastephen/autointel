#!/usr/bin/env python3
"""
Car Analysis Pipeline Script

This script orchestrates the complete analysis pipeline:
1. Load data from database
2. Perform comprehensive analysis
3. Save results back to database
4. Generate reports and visualizations

Usage:
    python scripts/run_analysis_pipeline.py
"""

import os
import sys
import warnings
import time
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

warnings.filterwarnings('ignore')

def main():
    """Main pipeline function"""
    print("🚀 CAR ANALYSIS PIPELINE")
    print("=" * 60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    start_time = time.time()
    
    try:
        # Step 1: Import and initialize the analysis framework
        print("\n📋 Step 1: Initializing Analysis Framework")
        print("-" * 40)
        
        from src.analysis.car_analysis_framework import CarAnalysisFramework
        
        # Initialize with database mode
        analyzer = CarAnalysisFramework(
            news_file='datasets/car_news_dataset.csv',
            reviews_file='datasets/car_reviews_dataset.csv',
            use_database=True,
            news_table=os.getenv('NEWS_TABLE', 'car_news'),
            reviews_table=os.getenv('REVIEWS_TABLE', 'car_reviews')
        )
        
        print("✓ Analysis framework initialized")
        
        # Step 2: Load data from database
        print("\n📊 Step 2: Loading Data from Database")
        print("-" * 40)
        
        if not analyzer.load_data():
            print("❌ Failed to load data from database")
            return False
        
        print("✓ Data loaded successfully")
        print(f"  - News data: {analyzer.car_news_df.shape if analyzer.car_news_df is not None else 'None'}")
        print(f"  - Reviews data: {analyzer.car_reviews_df.shape if analyzer.car_reviews_df is not None else 'None'}")
        
        # Step 3: Perform sentiment analysis
        print("\n😊 Step 3: Performing Sentiment Analysis")
        print("-" * 40)
        
        analyzer.perform_sentiment_analysis()
        print("✓ Sentiment analysis completed")
        
        # Step 4: Perform topic modeling
        print("\n📝 Step 4: Performing Topic Modeling")
        print("-" * 40)
        
        analyzer.perform_topic_modeling()
        print("✓ Topic modeling completed")
        
        # Step 5: Perform Named Entity Recognition
        print("\n🏷️ Step 5: Performing Named Entity Recognition")
        print("-" * 40)
        
        analyzer.perform_ner_analysis()
        print("✓ Named Entity Recognition completed")
        
        # Step 6: Perform keyword analysis
        print("\n🔑 Step 6: Performing Keyword Analysis")
        print("-" * 40)
        
        analyzer.perform_keyword_analysis()
        print("✓ Keyword analysis completed")
        
        # Step 7: Perform n-gram analysis
        print("\n📊 Step 7: Performing N-gram Analysis")
        print("-" * 40)
        
        analyzer.perform_ngram_analysis()
        print("✓ N-gram analysis completed")
        
        # Step 8: Perform correlation analysis
        print("\n📈 Step 8: Performing Correlation Analysis")
        print("-" * 40)
        
        analyzer.perform_correlation_analysis()
        print("✓ Correlation analysis completed")
        
        # Step 9: Calculate review length
        print("\n📏 Step 9: Calculating Review Length")
        print("-" * 40)
        
        analyzer.calculate_review_length()
        print("✓ Review length calculation completed")
        
        # Step 10: Save results to database
        print("\n💾 Step 10: Saving Results to Database")
        print("-" * 40)
        
        if analyzer.save_results_to_database():
            print("✓ Results saved to database successfully")
        else:
            print("❌ Failed to save results to database")
            return False
        
        # Step 11: Generate analysis summary
        print("\n📋 Step 11: Generating Analysis Summary")
        print("-" * 40)
        
        analyzer.generate_analysis_summary()
        
        # Step 12: Pipeline completion
        end_time = time.time()
        duration = end_time - start_time
        
        print("\n" + "=" * 60)
        print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"Total execution time: {duration:.2f} seconds")
        print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)
        
        # Print summary of what was accomplished
        print("\n📊 ANALYSIS RESULTS SUMMARY:")
        print("-" * 40)
        
        if analyzer.car_news_df is not None:
            print(f"✓ Car News Analysis:")
            print(f"  - Articles processed: {len(analyzer.car_news_df)}")
            if 'sentiment' in analyzer.car_news_df.columns:
                sentiment_dist = analyzer.car_news_df['sentiment'].value_counts()
                print(f"  - Sentiment distribution: {dict(sentiment_dist)}")
            if 'topics' in analyzer.car_news_df.columns:
                print(f"  - Topics extracted: ✓")
            if 'entities' in analyzer.car_news_df.columns:
                print(f"  - Entities identified: ✓")
            if 'keywords' in analyzer.car_news_df.columns:
                print(f"  - Keywords extracted: ✓")
        
        if analyzer.car_reviews_df is not None:
            print(f"✓ Car Reviews Analysis:")
            print(f"  - Reviews processed: {len(analyzer.car_reviews_df)}")
            if 'sentiment' in analyzer.car_reviews_df.columns:
                sentiment_dist = analyzer.car_reviews_df['sentiment'].value_counts()
                print(f"  - Sentiment distribution: {dict(sentiment_dist)}")
            if 'rating' in analyzer.car_reviews_df.columns:
                avg_rating = analyzer.car_reviews_df['rating'].mean()
                print(f"  - Average rating: {avg_rating:.2f}")
            if 'topics' in analyzer.car_reviews_df.columns:
                print(f"  - Topics extracted: ✓")
            if 'entities' in analyzer.car_reviews_df.columns:
                print(f"  - Entities identified: ✓")
            if 'keywords' in analyzer.car_reviews_df.columns:
                print(f"  - Keywords extracted: ✓")
            if 'correlation_score' in analyzer.car_reviews_df.columns:
                print(f"  - Correlation analysis: ✓")
            if 'review_length' in analyzer.car_reviews_df.columns:
                print(f"  - Review length calculated: ✓")
        
        print("\n💾 Database Tables Updated:")
        print(f"  - {analyzer.news_table}: News analysis results")
        print(f"  - {analyzer.reviews_table}: Reviews analysis results")
        
        print("\n🎯 Next Steps:")
        print("  - Query the database to explore analysis results")
        print("  - Create visualizations using the analysis data")
        print("  - Build dashboards or reports based on insights")
        
        return True
        
    except Exception as e:
        print(f"\n❌ PIPELINE FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
