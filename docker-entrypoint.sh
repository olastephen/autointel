#!/bin/bash

# Docker entrypoint script for Car Analysis Intelligence Platform
set -e

echo "🚀 Starting Car Analysis Intelligence Platform..."
echo "📊 Container: autointel-dashboard"
echo "🐳 Version: $(cat /etc/debian_version 2>/dev/null || echo 'Unknown')"
echo ""

# Function to check if datasets exist
check_datasets() {
    echo "🔍 Checking for required datasets..."
    
    if [ ! -f "/app/datasets/car_news_dataset.csv" ] || [ ! -f "/app/datasets/car_reviews_dataset.csv" ]; then
        echo "⚠️  Warning: Core datasets not found. Dashboard will run with limited functionality."
        return 1
    fi
    
    if [ ! -f "/app/datasets/car_work_data.csv" ]; then
        echo "⚠️  Warning: Car work data not found. Market analysis will be limited."
        return 1
    fi
    
    echo "✅ All datasets found!"
    return 0
}

# Function to ensure NLTK data is available
ensure_nltk_data() {
    echo "📚 Ensuring NLTK data is available..."
    
    python3 << 'EOF'
import nltk
import os

# Create NLTK data directory if it doesn't exist
nltk_data_dir = '/root/nltk_data'
if not os.path.exists(nltk_data_dir):
    os.makedirs(nltk_data_dir, exist_ok=True)

# Download required NLTK data
required_data = [
    'punkt',
    'punkt_tab', 
    'stopwords',
    'averaged_perceptron_tagger',
    'wordnet',
    'vader_lexicon'
]

for data_name in required_data:
    try:
        nltk.data.find(f'tokenizers/{data_name}')
        print(f"✅ {data_name} already available")
    except LookupError:
        try:
            print(f"📥 Downloading {data_name}...")
            nltk.download(data_name, quiet=True)
            print(f"✅ {data_name} downloaded successfully")
        except Exception as e:
            print(f"⚠️  Failed to download {data_name}: {e}")

print("📚 NLTK data check completed")
EOF
}

# Function to run analysis framework
run_analysis() {
    echo "🔄 Executing car analysis framework..."
    echo "📈 This may take a few minutes for initial analysis..."
    
    # Ensure NLTK data is available
    ensure_nltk_data
    
    # Set PYTHONPATH to ensure imports work
    export PYTHONPATH=/app:$PYTHONPATH
    
    # Run the analysis with error handling
    python3 << 'EOF'
import sys
import traceback
import os

# Add current directory to path
sys.path.insert(0, '/app')

try:
    print("📊 Initializing Car Analysis Framework...")
    from src.analysis.car_analysis_framework import CarAnalysisFramework
    
    print("🚀 Creating framework instance...")
    framework = CarAnalysisFramework()
    
    print("✅ Framework initialized successfully!")
    print("🔍 Running comprehensive analysis pipeline...")
    
    # Run the full analysis
    framework.run_full_analysis()
    
    print("✅ Analysis completed successfully!")
    print("📊 Results stored in database and ready for dashboard.")
    
except ImportError as e:
    print(f"⚠️  Import error: {e}")
    print("📊 Continuing with dashboard startup...")
    
except Exception as e:
    print(f"⚠️  Analysis error: {e}")
    print("📊 Error details:")
    traceback.print_exc()
    print("📊 Continuing with dashboard startup...")
    
finally:
    print("🔄 Analysis phase completed.")
EOF
}

# Function to start Streamlit
start_streamlit() {
    echo "🌐 Starting Streamlit dashboard..."
    echo "📱 Dashboard will be available at: http://localhost:8501"
    echo "🔗 Network URL: http://$(hostname -i):8501"
    echo ""
    
    # Start Streamlit with production settings
    exec streamlit run streamlit_app.py \
        --server.port=8501 \
        --server.address=0.0.0.0 \
        --server.headless=true \
        --server.runOnSave=false \
        --server.enableCORS=true \
        --server.enableXsrfProtection=false \
        --browser.gatherUsageStats=false
}

# Main execution flow
main() {
    echo "🏁 Starting main execution flow..."
    
    # Check datasets
    check_datasets || echo "📊 Proceeding with available data..."
    
    # Run analysis framework
    echo ""
    echo "🔬 Phase 1: Data Analysis"
    echo "========================="
    run_analysis
    
    # Start dashboard
    echo ""
    echo "🎯 Phase 2: Dashboard Launch"
    echo "============================"
    start_streamlit
}

# Handle signals gracefully
trap 'echo "🛑 Received shutdown signal, stopping..."; exit 0' SIGTERM SIGINT

# Execute main function
main
