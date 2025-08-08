#!/usr/bin/env python3
"""
Test script for Car Analysis Framework with memory monitoring
"""

import sys
import os
import gc
import psutil
import traceback
from datetime import datetime

# Add current directory to path
sys.path.insert(0, '.')

def monitor_memory():
    """Monitor and log memory usage"""
    try:
        memory = psutil.virtual_memory()
        print(f"🧠 Memory: {memory.percent:.1f}% used ({memory.used // 1024 // 1024}MB / {memory.total // 1024 // 1024}MB)")
        if memory.percent > 80:
            print("⚠️  High memory usage detected!")
            gc.collect()
        return memory.percent
    except Exception as e:
        print(f"⚠️  Memory monitoring error: {e}")
        return 0

def main():
    print("🚀 Starting Car Analysis Framework Test")
    print("=" * 50)
    print(f"📅 Start time: {datetime.now()}")
    
    # Initial memory check
    print("\n📊 Initial memory status:")
    monitor_memory()
    
    try:
        print("\n📦 Importing CarAnalysisFramework...")
        monitor_memory()
        
        from src.analysis.car_analysis_framework import CarAnalysisFramework
        
        print("\n🔧 Creating framework instance...")
        monitor_memory()
        
        # Create framework with smaller batch size for testing
        framework = CarAnalysisFramework()
        framework.batch_size = 25  # Even smaller batches for testing
        framework.max_memory_usage = 0.6  # More conservative memory limit
        
        print(f"✅ Framework created with batch_size={framework.batch_size}, max_memory={framework.max_memory_usage*100}%")
        monitor_memory()
        
        print("\n📊 Loading data...")
        monitor_memory()
        
        # Load data
        framework.load_data()
        
        print(f"✅ Data loaded: News={len(framework.car_news_df) if framework.car_news_df is not None else 0} records, Reviews={len(framework.car_reviews_df) if framework.car_reviews_df is not None else 0} records")
        monitor_memory()
        
        print("\n🔍 Running analysis pipeline...")
        monitor_memory()
        
        # Run analysis step by step with memory monitoring
        print("\n1️⃣ Running sentiment analysis...")
        framework.perform_sentiment_analysis()
        monitor_memory()
        
        print("\n2️⃣ Running topic modeling...")
        framework.perform_topic_modeling()
        monitor_memory()
        
        print("\n3️⃣ Running NER analysis...")
        framework.perform_ner_analysis()
        monitor_memory()
        
        print("\n4️⃣ Running keyword analysis...")
        framework.perform_keyword_analysis()
        monitor_memory()
        
        print("\n5️⃣ Running n-gram analysis...")
        framework.perform_ngram_analysis()
        monitor_memory()
        
        print("\n6️⃣ Running correlation analysis...")
        framework.perform_correlation_analysis()
        monitor_memory()
        
        print("\n💾 Saving results to database...")
        framework.save_results_to_database()
        monitor_memory()
        
        print("\n✅ Analysis completed successfully!")
        print(f"📅 End time: {datetime.now()}")
        
    except MemoryError as e:
        print(f"\n💥 Memory error occurred: {e}")
        print("🧠 Forcing garbage collection...")
        gc.collect()
        monitor_memory()
        
    except ImportError as e:
        print(f"\n⚠️  Import error: {e}")
        print("📋 Check that all dependencies are installed")
        
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        print("📋 Error details:")
        traceback.print_exc()
        
    finally:
        print("\n🧹 Final cleanup...")
        gc.collect()
        monitor_memory()
        print("🏁 Test completed")

if __name__ == "__main__":
    main()
