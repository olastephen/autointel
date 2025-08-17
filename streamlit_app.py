#!/usr/bin/env python3
"""
Car Analysis Dashboard - Streamlit Application
Enhanced version with filtering, dropdown selection, and better visualizations
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os
import sys
from datetime import datetime, timedelta
from wordcloud import WordCloud
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for Streamlit
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
import networkx as nx

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import database configuration
from src.config.config import DatabaseConfig

# Page configuration
st.set_page_config(
    page_title="🚗 Car Analysis Dashboard",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        color: #1f77b4;
        text-align: center;
        font-size: 3rem;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .section-header {
        color: #2c3e50;
        border-bottom: 3px solid #3498db;
        padding-bottom: 10px;
        margin-bottom: 20px;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .filter-box {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #dee2e6;
        margin-bottom: 20px;
    }
    .stSelectbox > div > div {
        background-color: white;
    }
    .stDateInput > div > div {
        background-color: white;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data_from_database():
    """Load data from PostgreSQL database"""
    try:
        db_config = DatabaseConfig()
        db_config.setup_from_env()
        
        # Load both datasets
        news_df = db_config.load_table("car_news")
        reviews_df = db_config.load_table("car_reviews")
        
        if news_df is not None:
            print(f"✓ News data loaded: {news_df.shape}")
        if reviews_df is not None:
            print(f"✓ Reviews data loaded: {reviews_df.shape}")
        
        return news_df, reviews_df
        
    except Exception as e:
        st.error(f"Database connection error: {e}")
        return None, None

def parse_json_column(df, column_name):
    """Parse JSON strings from database columns with enhanced error handling"""
    if column_name not in df.columns:
        return []
    
    parsed_data = []
    for value in df[column_name]:
        if pd.isna(value) or value is None:
            parsed_data.append([])
        elif isinstance(value, str):
            try:
                # First try to parse as JSON
                parsed = json.loads(value)
                parsed_data.append(parsed if isinstance(parsed, list) else [])
            except json.JSONDecodeError:
                try:
                    # Fallback: try to evaluate as Python list (for old data)
                    import ast
                    parsed = ast.literal_eval(value)
                    parsed_data.append(parsed if isinstance(parsed, list) else [])
                except (ValueError, SyntaxError):
                    # If all else fails, try to split the string representation
                    if value.startswith('[') and value.endswith(']'):
                        # Remove brackets and quotes, split by comma
                        clean_value = value.strip('[]').replace("'", "").replace('"', '')
                        if clean_value:
                            items = [item.strip() for item in clean_value.split(',') if item.strip()]
                            parsed_data.append(items)
                        else:
                            parsed_data.append([])
                    else:
                        parsed_data.append([])
        else:
            parsed_data.append(value if isinstance(value, list) else [])
    
    return parsed_data

def parse_json_object_column(df, column_name):
    """Parse JSON object columns (like keyword_counts, bigram_counts) with enhanced error handling"""
    if column_name not in df.columns:
        return []
    
    parsed_data = []
    for value in df[column_name]:
        if pd.isna(value) or value is None:
            parsed_data.append({})
        elif isinstance(value, str):
            try:
                # Try to parse as JSON object
                parsed = json.loads(value)
                parsed_data.append(parsed if isinstance(parsed, dict) else {})
            except json.JSONDecodeError:
                try:
                    # Fallback: try to evaluate as Python dict
                    import ast
                    parsed = ast.literal_eval(value)
                    parsed_data.append(parsed if isinstance(parsed, dict) else {})
                except (ValueError, SyntaxError):
                    # If all else fails, return empty dict
                    parsed_data.append({})
        else:
            parsed_data.append(value if isinstance(value, dict) else {})
    
    return parsed_data

def extract_brands_from_entities(entities_data):
    """Extract car brands from entity data"""
    brands = []
    car_brands = [
        'toyota', 'honda', 'ford', 'bmw', 'mercedes', 'audi', 'volkswagen', 
        'nissan', 'hyundai', 'kia', 'chevrolet', 'dodge', 'jeep', 'tesla',
        'lexus', 'acura', 'infiniti', 'cadillac', 'buick', 'chrysler',
        'volvo', 'subaru', 'mazda', 'mitsubishi', 'porsche', 'ferrari',
        'lamborghini', 'bentley', 'rolls royce', 'aston martin', 'jaguar',
        'land rover', 'range rover', 'mini', 'fiat', 'alfa romeo', 'maserati'
    ]
    
    for entities_list in entities_data:
        if isinstance(entities_list, list):
            for entity in entities_list:
                if isinstance(entity, str):
                    entity_lower = entity.lower()
                    for brand in car_brands:
                        if brand in entity_lower:
                            brands.append(brand.title())
                            break
    
    return list(set(brands))

def create_wordcloud_from_frequencies(frequency_dict, title):
    """Create word cloud from frequency dictionary"""
    if not frequency_dict:
        print(f"Wordcloud: No frequency data provided for {title}")
        return None
    
    try:
        print(f"Wordcloud: Creating frequency-based wordcloud for {title} with {len(frequency_dict)} keywords")
        print(f"Wordcloud: Sample frequencies: {dict(list(frequency_dict.items())[:5])}")
        
        # Create word cloud from frequency dictionary
        wordcloud = WordCloud(
            width=800, 
            height=400, 
            background_color='white',
            colormap='viridis',
            max_words=100,
            min_font_size=8,
            max_font_size=120,
            relative_scaling=1.0,
            collocations=False,
            regexp=r'\b\w+\b',
            normalize_plurals=True,
            prefer_horizontal=0.7
        ).generate_from_frequencies(frequency_dict)
        
        # Create matplotlib figure
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        
        print(f"Wordcloud: Successfully created frequency-based wordcloud for {title}")
        return fig
        
    except Exception as e:
        print(f"Error creating frequency-based word cloud for {title}: {e}")
        import traceback
        traceback.print_exc()
        return None

def create_wordcloud(text_data, title):
    """Create word cloud from text data"""
    if not text_data:
        print(f"Wordcloud: No text data provided for {title}")
        return None
    
    try:
        # Handle both single strings and lists of strings
        if isinstance(text_data, str):
            # Single string - use it directly
            all_text = text_data
        else:
            # List of strings - clean and combine them
            cleaned_texts = []
            for text in text_data:
                if text and str(text).strip():
                    # Remove special characters and clean the text
                    clean_text = str(text).strip().replace(':', ' ').replace('-', ' ')
                    if clean_text:
                        cleaned_texts.append(clean_text)
            
            if not cleaned_texts:
                print(f"Wordcloud: No cleaned texts for {title}")
                return None
        
            # Join all text for word cloud
            all_text = ' '.join(cleaned_texts)
        
        if len(all_text.strip()) < 3:  # Need at least 3 characters
            print(f"Wordcloud: Text too short for {title}: '{all_text[:50]}...'")
            return None
        
        print(f"Wordcloud: Creating for {title} with {len(all_text)} characters")
        print(f"Wordcloud: Sample text: '{all_text[:100]}...'")
        print(f"Wordcloud: Text contains {len(all_text.split())} words")
        print(f"Wordcloud: First 10 words: {all_text.split()[:10]}")
        
        # Create word cloud with better settings
        wordcloud = WordCloud(
            width=800, 
            height=400, 
            background_color='white',
            colormap='viridis',
            max_words=100,  # Increased from 50
            min_font_size=8,  # Reduced from 10
            max_font_size=120,  # Increased from 100
            relative_scaling=1.0,  # Changed from 0.5 for better word separation
            collocations=False,  # Avoid pairing words
            regexp=r'\b\w+\b',  # Ensure we only match complete words
            normalize_plurals=True,  # Handle plural forms
            prefer_horizontal=0.7  # Mix of horizontal and vertical text
        ).generate(all_text)
        
        # Create matplotlib figure
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        
        print(f"Wordcloud: Successfully created for {title}")
        return fig
        
    except Exception as e:
        print(f"Error creating word cloud for {title}: {e}")
        import traceback
        traceback.print_exc()
        return None

def create_sentiment_distribution_chart(df, title):
    """Create sentiment distribution pie chart"""
    if 'sentiment' not in df.columns:
        return None
    
    # Filter out records with null content/verdict
    filtered_df = df.copy()
    if 'content' in filtered_df.columns:
        filtered_df = filtered_df.dropna(subset=['content'])
    if 'verdict' in filtered_df.columns:
        filtered_df = filtered_df.dropna(subset=['verdict'])
    
    if filtered_df.empty:
        return None
    
    sentiment_counts = filtered_df['sentiment'].value_counts()
    
    fig = px.pie(
        values=sentiment_counts.values,
        names=sentiment_counts.index,
        title=title,
        color=sentiment_counts.index,
        color_discrete_map={
            'positive': '#00ff00',  # Traffic Light Green
            'negative': '#ff0000',  # Traffic Light Red
            'neutral': '#ffff00'    # Traffic Light Yellow
        }
    )
    
    fig.update_traces(textposition='inside', textinfo='percent+label')
    
    # Force color update to ensure traffic light colors are applied
    fig.update_layout(
        showlegend=True,
        font=dict(size=12)
    )
    
    return fig

def create_sentiment_bar_chart(df, title):
    """Create sentiment distribution bar chart"""
    if 'sentiment' not in df.columns:
        return None
    
    # Filter out records with null content/verdict
    filtered_df = df.copy()
    if 'content' in filtered_df.columns:
        filtered_df = filtered_df.dropna(subset=['content'])
    if 'verdict' in filtered_df.columns:
        filtered_df = filtered_df.dropna(subset=['verdict'])
    
    if filtered_df.empty:
        return None
    
    sentiment_counts = filtered_df['sentiment'].value_counts()
    
    fig = px.bar(
        x=sentiment_counts.index,
        y=sentiment_counts.values,
        title=title,
        color=sentiment_counts.index,
        color_discrete_map={
            'positive': '#00ff00',  # Traffic Light Green
            'negative': '#ff0000',  # Traffic Light Red
            'neutral': '#ffff00'    # Traffic Light Yellow
        },
        labels={'x': 'Sentiment', 'y': 'Count'}
    )
    
    fig.update_layout(
        xaxis_title="Sentiment",
        yaxis_title="Count",
        showlegend=False
    )
    
    return fig

def create_sentiment_score_distribution(df, title):
    """Create sentiment score distribution histogram"""
    if 'sentiment_score' not in df.columns:
        return None
    
    # Filter out records with null content/verdict
    filtered_df = df.copy()
    if 'content' in filtered_df.columns:
        filtered_df = filtered_df.dropna(subset=['content'])
    if 'verdict' in filtered_df.columns:
        filtered_df = filtered_df.dropna(subset=['verdict'])
    
    if filtered_df.empty:
        return None
    
    fig = px.histogram(
        filtered_df,
        x='sentiment_score',
        nbins=20,
        title=title,
        color_discrete_sequence=['#3498db'],
        labels={'sentiment_score': 'Sentiment Score', 'count': 'Frequency'}
    )
    
    fig.update_layout(
        xaxis_title="Sentiment Score",
        yaxis_title="Frequency",
        showlegend=False
    )
    
    return fig

def create_sentiment_trend_chart(df, title):
    """Create sentiment trend over time"""
    if 'sentiment' not in df.columns or 'publication_date' not in df.columns:
        return None
    
    # Filter out records with null content/verdict
    filtered_df = df.copy()
    if 'content' in filtered_df.columns:
        filtered_df = filtered_df.dropna(subset=['content'])
    if 'verdict' in filtered_df.columns:
        filtered_df = filtered_df.dropna(subset=['verdict'])
    
    if filtered_df.empty:
        return None
    
    # Convert to datetime and group by date
    filtered_df['publication_date'] = pd.to_datetime(filtered_df['publication_date'], errors='coerce')
    filtered_df = filtered_df.dropna(subset=['publication_date'])
    
    if filtered_df.empty:
        return None
    
    # Group by date and sentiment
    daily_sentiment = filtered_df.groupby([filtered_df['publication_date'].dt.date, 'sentiment']).size().reset_index(name='count')
    daily_sentiment.columns = ['date', 'sentiment', 'count']
    
    fig = px.line(
        daily_sentiment,
        x='date',
        y='count',
        color='sentiment',
        title=title,
        color_discrete_map={
            'positive': '#00ff00',  # Traffic Light Green
            'negative': '#ff0000',  # Traffic Light Red
            'neutral': '#ffff00'    # Traffic Light Yellow
        }
    )
    
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Count",
        showlegend=True
    )
    
    return fig

def create_rating_distribution_chart(df):
    """Create rating distribution histogram"""
    if 'rating' not in df.columns:
        return None
    
    fig = px.histogram(
        df, 
        x='rating',
        nbins=20,
        title="Rating Distribution",
        labels={'rating': 'Rating (Stars)', 'count': 'Number of Reviews'},
        color_discrete_sequence=['#3498db']
    )
    
    fig.update_layout(
        xaxis_title="Rating (Stars)",
        yaxis_title="Number of Reviews",
        showlegend=False
    )
    
    return fig

def create_sentiment_vs_rating_scatter(df):
    """Create sentiment vs rating scatter plot"""
    if 'sentiment_score' not in df.columns or 'rating' not in df.columns:
        return None
    
    # Filter out records with null content/verdict
    filtered_df = df.copy()
    if 'content' in filtered_df.columns:
        filtered_df = filtered_df.dropna(subset=['content'])
    if 'verdict' in filtered_df.columns:
        filtered_df = filtered_df.dropna(subset=['verdict'])
    
    if filtered_df.empty:
        return None
    
    # Create color mapping for sentiment
    color_map = {'positive': '#00ff00', 'negative': '#ff0000', 'neutral': '#ffff00'}
    filtered_df['sentiment_color'] = filtered_df['sentiment'].map(color_map)
    
    fig = px.scatter(
        filtered_df,
        x='rating',
        y='sentiment_score',
        color='sentiment',
        title="Sentiment Score vs Rating",
        labels={'rating': 'Rating (Stars)', 'sentiment_score': 'Sentiment Score'},
        color_discrete_map=color_map,
        hover_data=['title']
    )
    
    return fig

def create_review_length_vs_rating_scatter(df):
    """Create review length vs rating scatter plot"""
    if 'review_length' not in df.columns or 'rating' not in df.columns:
        return None
    
    # Filter out records with null content/verdict
    filtered_df = df.copy()
    if 'content' in filtered_df.columns:
        filtered_df = filtered_df.dropna(subset=['content'])
    if 'verdict' in filtered_df.columns:
        filtered_df = filtered_df.dropna(subset=['verdict'])
    
    if filtered_df.empty:
        return None
    
    fig = px.scatter(
        filtered_df,
        x='rating',
        y='review_length',
        color='sentiment',
        title="Review Length vs Rating",
        labels={'rating': 'Rating (Stars)', 'review_length': 'Review Length (Characters)'},
        color_discrete_map={'positive': '#00ff00', 'negative': '#ff0000', 'neutral': '#ffff00'},
        hover_data=['title']
    )
    
    return fig

def create_sentiment_vs_review_length_scatter(df):
    """Create sentiment score vs review length scatter plot"""
    if 'sentiment_score' not in df.columns or 'review_length' not in df.columns:
        return None
    
    # Filter out records with null content/verdict
    filtered_df = df.copy()
    if 'content' in filtered_df.columns:
        filtered_df = filtered_df.dropna(subset=['content'])
    if 'verdict' in filtered_df.columns:
        filtered_df = filtered_df.dropna(subset=['verdict'])
    
    if filtered_df.empty:
        return None
    
    fig = px.scatter(
        filtered_df,
        x='review_length',
        y='sentiment_score',
        color='sentiment',
        title="Sentiment Score vs Review Length",
        labels={'review_length': 'Review Length (Characters)', 'sentiment_score': 'Sentiment Score'},
        color_discrete_map={'positive': '#00ff00', 'negative': '#ff0000', 'neutral': '#ffff00'},
        hover_data=['title']
    )
    
    return fig



def load_market_data():
    """Load and process car work market data"""
    try:
        df = pd.read_csv('datasets/car_work_data.csv')
        
        # Aggregate data by authority and car availability
        market_summary = df.groupby([
            'Lower tier local authorities',
            'Car or van availability (3 categories)',
            'Distance travelled to work (5 categories)'
        ])['Observation'].sum().reset_index()
        
        # Calculate car dependency metrics
        car_dependency = df[df['Car or van availability (3 categories)'] == '1 or more cars or vans in household']
        total_households = df[df['Car or van availability (3 categories)'] != 'Does not apply']
        
        authority_metrics = []
        for authority in df['Lower tier local authorities'].unique():
            if pd.notna(authority):
                auth_car = car_dependency[car_dependency['Lower tier local authorities'] == authority]['Observation'].sum()
                auth_total = total_households[total_households['Lower tier local authorities'] == authority]['Observation'].sum()
                
                if auth_total > 0:
                    car_dependency_rate = auth_car / auth_total
                    authority_metrics.append({
                        'Authority': authority,
                        'Car_Dependency_Rate': car_dependency_rate,
                        'Total_Households': auth_total
                    })
        
        return pd.DataFrame(authority_metrics)
    except Exception as e:
        print(f"Error loading market data: {e}")
        return None

def simulate_economic_indicators(start_date, end_date):
    """Simulate economic indicators for correlation analysis"""
    import numpy as np
    from datetime import datetime, timedelta
    
    # Create date range
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Simulate interest rates (with realistic fluctuations)
    np.random.seed(42)  # For reproducible results
    base_rate = 3.5
    interest_rates = base_rate + np.cumsum(np.random.normal(0, 0.01, len(dates)))
    interest_rates = np.clip(interest_rates, 0.5, 8.0)  # Keep realistic bounds
    
    # Simulate consumer confidence index (inverse correlation with interest rates)
    base_confidence = 75
    confidence_noise = np.random.normal(0, 2, len(dates))
    consumer_confidence = base_confidence - (interest_rates - base_rate) * 5 + confidence_noise
    consumer_confidence = np.clip(consumer_confidence, 30, 100)
    
    # Add seasonal effects
    seasonal_factor = 5 * np.sin(2 * np.pi * np.arange(len(dates)) / 365.25)
    consumer_confidence += seasonal_factor
    
    return pd.DataFrame({
        'date': dates,
        'interest_rate': interest_rates,
        'consumer_confidence': consumer_confidence
    })

def create_time_series_chart(df, date_column, value_column, title):
    """Create time series chart"""
    if date_column not in df.columns or value_column not in df.columns:
        return None
    
    # Convert to datetime and group by date
    df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
    df = df.dropna(subset=[date_column])
    
    if df.empty:
        return None
    
    # Group by date and count
    daily_counts = df.groupby(df[date_column].dt.date)[value_column].count().reset_index()
    daily_counts.columns = ['date', 'count']
    
    fig = px.line(
        daily_counts,
        x='date',
        y='count',
        title=title,
        labels={'date': 'Date', 'count': 'Count'}
    )
    
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Count",
        showlegend=False
    )
    
    return fig

def create_enhanced_time_series_analysis(df, market_data, title):
    """Create enhanced time series analysis with market correlation"""
    if 'publication_date' not in df.columns or 'sentiment_score' not in df.columns:
        return None
    
    # Convert dates and clean data
    df_clean = df.copy()
    df_clean['publication_date'] = pd.to_datetime(df_clean['publication_date'], errors='coerce')
    df_clean = df_clean.dropna(subset=['publication_date', 'sentiment_score'])
    
    if df_clean.empty:
        return None
    
    # Get date range for economic simulation
    start_date = df_clean['publication_date'].min()
    end_date = df_clean['publication_date'].max()
    
    # Generate economic indicators
    economic_data = simulate_economic_indicators(start_date, end_date)
    
    # Aggregate sentiment by date
    daily_sentiment = df_clean.groupby(df_clean['publication_date'].dt.date).agg({
        'sentiment_score': 'mean'
    }).reset_index()
    daily_sentiment.columns = ['date', 'avg_sentiment']
    daily_sentiment['date'] = pd.to_datetime(daily_sentiment['date'])
    
    # Merge with economic data
    economic_data['date'] = economic_data['date'].dt.date
    daily_sentiment['date'] = daily_sentiment['date'].dt.date
    
    merged_data = pd.merge(daily_sentiment, economic_data, on='date', how='inner')
    
    if merged_data.empty:
        return None
    
    # Convert date back to datetime for plotting
    merged_data['date'] = pd.to_datetime(merged_data['date'])
    
    # Create subplots - Removed interest rate, focus on sentiment vs consumer confidence
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=[
            f"{title} - Sentiment Trend",
            "Consumer Confidence Index",
            "Sentiment vs Consumer Confidence Correlation"
        ],
        vertical_spacing=0.1,
        specs=[[{"secondary_y": False}],
               [{"secondary_y": False}],
               [{"secondary_y": True}]]
    )
    
    # Plot sentiment
    fig.add_trace(
        go.Scatter(
            x=merged_data['date'],
            y=merged_data['avg_sentiment'],
            mode='lines',
            name='Average Sentiment',
            line=dict(color='#00ff00', width=2)
        ),
        row=1, col=1
    )
    
    # Plot consumer confidence
    fig.add_trace(
        go.Scatter(
            x=merged_data['date'],
            y=merged_data['consumer_confidence'],
            mode='lines',
            name='Consumer Confidence',
            line=dict(color='#ffff00', width=2)
        ),
        row=2, col=1
    )
    
    # Plot correlation (sentiment vs consumer confidence)
    fig.add_trace(
        go.Scatter(
            x=merged_data['date'],
            y=merged_data['avg_sentiment'],
            mode='lines',
            name='Sentiment',
            line=dict(color='#00ff00', width=2)
        ),
        row=3, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=merged_data['date'],
            y=merged_data['consumer_confidence'],
            mode='lines',
            name='Consumer Confidence',
            line=dict(color='#ffff00', width=2, dash='dash'),
            yaxis='y2'
        ),
        row=3, col=1,
        secondary_y=True
    )
    
    # Update layout
    fig.update_layout(
        height=700,
        title_text=f"{title} - Consumer Sentiment vs Consumer Confidence Analysis",
        showlegend=False
    )
    
    # Update y-axes
    fig.update_yaxes(title_text="Sentiment Score", row=1, col=1)
    fig.update_yaxes(title_text="Confidence Index", row=2, col=1)
    fig.update_yaxes(title_text="Sentiment Score", row=3, col=1)
    fig.update_yaxes(title_text="Consumer Confidence Index", row=3, col=1, secondary_y=True)
    
    # Update x-axes
    fig.update_xaxes(title_text="Date", row=3, col=1)
    
    return fig, merged_data

def display_keyword_analysis(df, title):
    """Display keyword analysis with word cloud, bar chart, and correlation"""
    if 'keywords' not in df.columns:
        return
    
    keywords_data = parse_json_column(df, 'keywords')
    if not keywords_data:
        return
    
    # Check if we have document-level keyword counts (new approach)
    has_doc_counts = 'keyword_counts' in df.columns
    
    if has_doc_counts:
        # Use document-level keyword counts for meaningful frequency analysis
        keyword_counts_data = parse_json_object_column(df, 'keyword_counts')
        
        # Aggregate keyword frequencies across documents
        keyword_frequencies = {}
        for doc_counts in keyword_counts_data:
            if isinstance(doc_counts, dict):
                for keyword, count in doc_counts.items():
                    keyword_frequencies[keyword] = keyword_frequencies.get(keyword, 0) + count
        
        # Get top keywords by frequency
        if keyword_frequencies:
            sorted_keywords = sorted(keyword_frequencies.items(), key=lambda x: x[1], reverse=True)
            top_keywords = sorted_keywords[:20]
            
            # Create meaningful frequency data
            plot_df = pd.DataFrame({
                'Keyword': [kw[0] for kw in top_keywords],
                'Frequency': [kw[1] for kw in top_keywords]
            })
            
            # Create word cloud from all keywords with their frequencies
            # Use WordCloud's built-in frequency support instead of text repetition
            wordcloud_frequencies = {kw[0]: kw[1] for kw in sorted_keywords[:50]}
            print(f"Keyword Analysis: Generated wordcloud frequencies for {len(wordcloud_frequencies)} keywords")
            print(f"Keyword Analysis: Sample frequencies: {dict(list(wordcloud_frequencies.items())[:5])}")
            
            # For fallback display, still create text
            wordcloud_text = ' '.join([kw[0] for kw in sorted_keywords[:50]])
            
        else:
            st.warning("No keyword frequency data found")
            return
    else:
        # Fallback to old approach (global keywords)
        st.info("⚠️ Using global keyword list. Run analysis pipeline for document-level frequencies.")
        
        # Flatten keywords and count frequencies
        all_keywords = []
        for keywords_list in keywords_data:
            if isinstance(keywords_list, list):
                all_keywords.extend(keywords_list)
        
        if not all_keywords:
            return
        
        keyword_counts = pd.Series(all_keywords).value_counts().head(20)
        
        # Create frequency data (this will show same frequency for all)
        plot_df = pd.DataFrame({
            'Keyword': keyword_counts.index,
            'Frequency': keyword_counts.values
        })
        
        # Create word cloud from all keywords
        wordcloud_text = ' '.join(all_keywords)
        print(f"Keyword Analysis (fallback): Generated wordcloud text with {len(wordcloud_text)} characters")
        print(f"Keyword Analysis (fallback): Sample text: '{wordcloud_text[:100]}...'")
        
        # Create frequency dict for fallback case too
        wordcloud_frequencies = {kw: 1 for kw in all_keywords[:50]}
    
    # Create tabs for different analyses
    tab1, tab2 = st.tabs(["🔑 Top Keywords", "🔗 Keyword Correlations"])
    
    with tab1:
        # Create word cloud
        print(f"Keyword Analysis: Creating wordcloud with text length: {len(wordcloud_text)}")
        
        try:
            # Always use frequency-based wordcloud for better results
            wordcloud_fig = create_wordcloud_from_frequencies(wordcloud_frequencies, f"{title} - Word Cloud")
                
            if wordcloud_fig:
                st.pyplot(wordcloud_fig)
            else:
                st.warning("Could not generate word cloud")
                print(f"Keyword Analysis: Wordcloud creation failed for {title}")
                
                # Fallback: show the top keywords as a simple list
                st.info("📝 Top Keywords (Wordcloud unavailable):")
                if has_doc_counts and keyword_frequencies:
                    top_10 = sorted(keyword_frequencies.items(), key=lambda x: x[1], reverse=True)[:10]
                    for i, (keyword, freq) in enumerate(top_10):
                        st.write(f"{i+1}. **{keyword}** - {freq} occurrences")
                else:
                    # Fallback for old approach
                    for i, keyword in enumerate(plot_df['Keyword'].head(10)):
                        st.write(f"{i+1}. **{keyword}**")
                        
        except Exception as e:
            st.error(f"Error creating wordcloud: {e}")
            print(f"Keyword Analysis: Exception in wordcloud creation: {e}")
            import traceback
            traceback.print_exc()
            
            # Fallback display
            st.info("📝 Top Keywords (Wordcloud error):")
            if has_doc_counts and keyword_frequencies:
                top_10 = sorted(keyword_frequencies.items(), key=lambda x: x[1], reverse=True)[:10]
                for i, (keyword, freq) in enumerate(top_10):
                    st.write(f"{i+1}. **{keyword}** - {freq} occurrences")
            else:
                for i, keyword in enumerate(plot_df['Keyword'].head(10)):
                    st.write(f"{i+1}. **{keyword}**")
    
    # Create bar chart
    fig = px.bar(
        plot_df,
        x='Frequency',
        y='Keyword',
        orientation='h',
        title=f"{title} - Top Keywords by Frequency",
        labels={'x': 'Frequency', 'y': 'Keywords'},
        color='Frequency',
        color_continuous_scale='blues'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Show frequency statistics
    if has_doc_counts and keyword_frequencies:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Keywords", len(keyword_frequencies))
        with col2:
            st.metric("Most Frequent", f"{plot_df.iloc[0]['Keyword']}")
        with col3:
            st.metric("Total Occurrences", sum(keyword_frequencies.values()))
    
    with tab2:
        st.subheader("🔗 Keyword Co-occurrence Analysis")
        
        if has_doc_counts and keyword_counts_data:
            # Calculate keyword co-occurrence using document-level data
            keyword_cooccurrence = {}
            top_keywords = [kw[0] for kw in sorted_keywords[:15]]
            
            for doc_counts in keyword_counts_data:
                if isinstance(doc_counts, dict):
                    # Get keywords that appear in this document
                    doc_keywords = [kw for kw in doc_counts.keys() if kw in top_keywords]
                    
                    # Calculate co-occurrence for this document
                    for i, kw1 in enumerate(doc_keywords):
                        for kw2 in doc_keywords[i+1:]:
                            pair = tuple(sorted([kw1, kw2]))
                            keyword_cooccurrence[pair] = keyword_cooccurrence.get(pair, 0) + 1
        else:
            # Fallback to old approach
            keyword_cooccurrence = {}
            top_keywords = plot_df['Keyword'].head(15).tolist()
            
            for keywords_list in keywords_data:
                if isinstance(keywords_list, list):
                    # Get intersection with top keywords
                    doc_keywords = [kw for kw in keywords_list if kw in top_keywords]
                    
                    # Calculate co-occurrence for this document
                    for i, kw1 in enumerate(doc_keywords):
                        for kw2 in doc_keywords[i+1:]:
                            pair = tuple(sorted([kw1, kw2]))
                            keyword_cooccurrence[pair] = keyword_cooccurrence.get(pair, 0) + 1
        
        if keyword_cooccurrence:
            # Create co-occurrence matrix
            cooccurrence_matrix = []
            for kw1 in top_keywords:
                row = []
                for kw2 in top_keywords:
                    if kw1 == kw2:
                        if has_doc_counts:
                            row.append(keyword_frequencies.get(kw1, 0))
                        else:
                            row.append(1)  # Fallback value
                    else:
                        pair = tuple(sorted([kw1, kw2]))
                        row.append(keyword_cooccurrence.get(pair, 0))
                cooccurrence_matrix.append(row)
            
            # Create heatmap
            fig = px.imshow(
                cooccurrence_matrix,
                x=top_keywords,
                y=top_keywords,
                title=f"{title} - Keyword Co-occurrence Matrix",
                color_continuous_scale='RdYlBu_r',
                aspect='auto'
            )
            
            fig.update_layout(
                xaxis_title="Keywords",
                yaxis_title="Keywords",
                height=600
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Show top co-occurring pairs
            st.subheader("🏆 Top Keyword Pairs")
            sorted_pairs = sorted(keyword_cooccurrence.items(), key=lambda x: x[1], reverse=True)
            for i, (pair, count) in enumerate(sorted_pairs[:10]):
                st.write(f"{i+1}. {' + '.join(pair)}: {count} documents")
        else:
            st.info("No keyword co-occurrence data available")

def display_topic_analysis(df, title):
    """Display topic analysis results with enhanced visualizations"""
    if 'topics' not in df.columns:
        st.warning(f"No topics column found in {title} data")
        return
    
    topics_data = parse_json_column(df, 'topics')
    if not topics_data:
        st.warning(f"No topics data found in {title}")
        return
    
    # Count how many records have topics
    topics_count = sum(1 for topics in topics_data if topics and isinstance(topics, list))
    st.info(f"Found topics in {topics_count} out of {len(df)} records")
    
    # Extract and process all topics
    all_topics = []
    topic_frequencies = {}
    topic_words = {}
    
    for topics_list in topics_data:
        if isinstance(topics_list, list) and topics_list:
            if isinstance(topics_list[0], list):
                # Nested topics
                for topic in topics_list:
                    if isinstance(topic, list) and topic:
                        topic_key = ' '.join([str(word) for word in topic[:3] if word is not None and str(word).strip()])
                        all_topics.append(topic_key)
                        topic_frequencies[topic_key] = topic_frequencies.get(topic_key, 0) + 1
                        topic_words[topic_key] = [str(word) for word in topic if word is not None and str(word).strip()]
            else:
                # Flat topics
                topic_key = ' '.join([str(word) for word in topics_list[:3] if word is not None and str(word).strip()])
                all_topics.append(topic_key)
                topic_frequencies[topic_key] = topic_frequencies.get(topic_key, 0) + 1
                topic_words[topic_key] = [str(word) for word in topics_list if word is not None and str(word).strip()]
    
    if not all_topics:
        st.warning("No valid topics found to display")
        return
    
    # Create tabs for different visualizations
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Topic Distribution", 
        "🔥 Topic Word Importance", 
        "🌐 Topic Network", 
        "📈 Topic Evolution", 
        "📊 Topic Coherence",
        "📝 Sample Topics"
    ])
    
    with tab1:
        st.subheader("📊 Topic Distribution Analysis")
        
        # Create topic frequency bar chart
        if topic_frequencies:
            # Sort by frequency
            sorted_topics = sorted(topic_frequencies.items(), key=lambda x: x[1], reverse=True)
            top_topics = sorted_topics[:15]  # Show top 15 topics
            
            topic_names = [topic[0] for topic in top_topics]
            frequencies = [topic[1] for topic in top_topics]
            
            fig = px.bar(
                x=frequencies,
                y=topic_names,
                orientation='h',
                title=f"{title} - Topic Frequency Distribution",
                labels={'x': 'Frequency', 'y': 'Topics'},
                color=frequencies,
                color_continuous_scale='viridis'
            )
            
            fig.update_layout(
                xaxis_title="Frequency",
                yaxis_title="Topics",
                height=600
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Topic statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Topics", len(topic_frequencies))
            with col2:
                st.metric("Most Frequent Topic", f"{topic_names[0] if topic_names else 'N/A'}")
            with col3:
                st.metric("Average Frequency", f"{np.mean(frequencies):.1f}")
    
    with tab2:
        st.subheader("🔥 Topic Word Importance Analysis")
        
        # Create word importance visualization for top topics
        if topic_words:
            # Select top 5 topics by frequency
            sorted_topics = sorted(topic_frequencies.items(), key=lambda x: x[1], reverse=True)
            top_topic_names = [topic[0] for topic in sorted_topics[:5]]
            
            # Create word importance chart
            word_importance_data = []
            for topic_name in top_topic_names:
                words = topic_words.get(topic_name, [])
                for i, word in enumerate(words[:10]):  # Top 10 words per topic
                    importance = 10 - i  # Higher importance for earlier words
                    word_importance_data.append({
                        'Topic': topic_name,
                        'Word': word,
                        'Importance': importance
                    })
            
            if word_importance_data:
                importance_df = pd.DataFrame(word_importance_data)
                
                # Create heatmap
                pivot_df = importance_df.pivot(index='Word', columns='Topic', values='Importance')
                
                fig = px.imshow(
                    pivot_df,
                    title=f"{title} - Word Importance Heatmap",
                    color_continuous_scale='viridis',
                    aspect='auto'
                )
                
                fig.update_layout(
                    xaxis_title="Topics",
                    yaxis_title="Words",
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Create bar chart for word importance
                fig2 = px.bar(
                    importance_df,
                    x='Word',
                    y='Importance',
                    color='Topic',
                    title=f"{title} - Word Importance by Topic",
                    barmode='group'
                )
                
                fig2.update_layout(
                    xaxis_title="Words",
                    yaxis_title="Importance Score",
                    height=500
                )
                
                st.plotly_chart(fig2, use_container_width=True)
    
    with tab3:
        st.subheader("🌐 Topic Network Analysis")
        
        # Create topic similarity network
        if len(topic_words) > 1:
            # Calculate topic similarities based on shared words
            topic_similarities = []
            topic_names = list(topic_words.keys())
            
            for i, topic1 in enumerate(topic_names):
                for j, topic2 in enumerate(topic_names[i+1:], i+1):
                    words1 = set(topic_words[topic1])
                    words2 = set(topic_words[topic2])
                    
                    # Jaccard similarity
                    intersection = len(words1.intersection(words2))
                    union = len(words1.union(words2))
                    similarity = intersection / union if union > 0 else 0
                    
                    if similarity > 0.1:  # Only show connections with similarity > 0.1
                        topic_similarities.append({
                            'Topic1': topic1,
                            'Topic2': topic2,
                            'Similarity': similarity
                        })
            
            if topic_similarities:
                # Create network visualization
                import networkx as nx
                
                G = nx.Graph()
                
                # Add nodes
                for topic_name in topic_names:
                    G.add_node(topic_name, size=topic_frequencies.get(topic_name, 1))
                
                # Add edges
                for sim in topic_similarities:
                    G.add_edge(sim['Topic1'], sim['Topic2'], weight=sim['Similarity'])
                
                # Create network layout
                pos = nx.spring_layout(G, k=1, iterations=50)
                
                # Create network plot
                fig = go.Figure()
                
                # Add edges
                edge_x = []
                edge_y = []
                edge_weights = []
                for edge in G.edges(data=True):
                    x0, y0 = pos[edge[0]]
                    x1, y1 = pos[edge[1]]
                    edge_x.extend([x0, x1, None])
                    edge_y.extend([y0, y1, None])
                    edge_weights.append(edge[2]['weight'])
                
                fig.add_trace(go.Scatter(
                    x=edge_x, y=edge_y,
                    line=dict(width=1, color='gray'),
                    hoverinfo='none',
                    mode='lines'))
                
                # Add nodes
                node_x = []
                node_y = []
                node_text = []
                node_sizes = []
                
                for node in G.nodes():
                    x, y = pos[node]
                    node_x.append(x)
                    node_y.append(y)
                    node_text.append(f"{node}<br>Frequency: {topic_frequencies.get(node, 0)}")
                    node_sizes.append(topic_frequencies.get(node, 1) * 10)
                
                fig.add_trace(go.Scatter(
                    x=node_x, y=node_y,
                    mode='markers+text',
                    hoverinfo='text',
                    text=node_text,
                    textposition="top center",
                    marker=dict(
                        size=node_sizes,
                        color=node_sizes,
                        colorscale='viridis',
                        showscale=True,
                        colorbar=dict(title="Topic Frequency")
                    )))
                
                fig.update_layout(
                    title=f"{title} - Topic Network Graph",
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=20,l=5,r=5,t=40),
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    height=600
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No significant topic similarities found for network visualization")
        else:
            st.info("Need at least 2 topics to create network visualization")
    
    with tab4:
        st.subheader("📈 Topic Evolution Over Time")
        
        # Create topic evolution timeline if date column exists
        if 'publication_date' in df.columns and topic_frequencies:
            # Convert dates and group by time periods
            df['publication_date'] = pd.to_datetime(df['publication_date'], errors='coerce')
            df = df.dropna(subset=['publication_date'])
            
            if not df.empty:
                # Create time periods (monthly)
                df['time_period'] = df['publication_date'].dt.to_period('M')
                
                # Count topics by time period
                topic_evolution = []
                for period in df['time_period'].unique():
                    period_df = df[df['time_period'] == period]
                    period_topics = []
                    
                    for idx in period_df.index:
                        if idx < len(topics_data) and topics_data[idx]:
                            topics_list = topics_data[idx]
                            if isinstance(topics_list, list) and topics_list:
                                if isinstance(topics_list[0], list):
                                    for topic in topics_list:
                                        if isinstance(topic, list) and topic:
                                            topic_key = ' '.join([str(word) for word in topic[:3] if word is not None and str(word).strip()])
                                            period_topics.append(topic_key)
                                else:
                                    topic_key = ' '.join([str(word) for word in topics_list[:3] if word is not None and str(word).strip()])
                                    period_topics.append(topic_key)
                    
                    # Count frequencies for this period
                    period_freq = Counter(period_topics)
                    for topic, freq in period_freq.items():
                        topic_evolution.append({
                            'Time_Period': str(period),
                            'Topic': topic,
                            'Frequency': freq
                        })
                
                if topic_evolution:
                    evolution_df = pd.DataFrame(topic_evolution)
                    
                    # Show top 5 topics over time
                    sorted_topics = sorted(topic_frequencies.items(), key=lambda x: x[1], reverse=True)
                    top_topics = [topic[0] for topic in sorted_topics[:5]]
                    evolution_filtered = evolution_df[evolution_df['Topic'].isin(top_topics)]
                    
                    if not evolution_filtered.empty:
                        fig = px.line(
                            evolution_filtered,
                            x='Time_Period',
                            y='Frequency',
                            color='Topic',
                            title=f"{title} - Topic Evolution Over Time",
                            markers=True
                        )
                        
                        fig.update_layout(
                            xaxis_title="Time Period",
                            yaxis_title="Frequency",
                            height=500
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Create heatmap of topic evolution
                        pivot_evolution = evolution_filtered.pivot(
                            index='Topic', 
                            columns='Time_Period', 
                            values='Frequency'
                        ).fillna(0)
                        
                        fig2 = px.imshow(
                            pivot_evolution,
                            title=f"{title} - Topic Evolution Heatmap",
                            color_continuous_scale='viridis',
                            aspect='auto'
                        )
                        
                        fig2.update_layout(
                            xaxis_title="Time Period",
                            yaxis_title="Topics",
                            height=400
                        )
                        
                        st.plotly_chart(fig2, use_container_width=True)
                    else:
                        st.info("No topic evolution data available for visualization")
                else:
                    st.info("No topic evolution data available")
            else:
                st.info("No valid dates found for evolution analysis")
        else:
            st.info("Date column not available for evolution analysis")
    
    with tab5:
        st.subheader("📊 Topic Coherence Analysis")
        
        # Calculate topic coherence metrics
        if topic_words and len(topic_words) > 1:
            # Calculate topic coherence using word overlap
            coherence_scores = []
            topic_names = list(topic_words.keys())
            
            for topic_name in topic_names:
                words = topic_words[topic_name]
                if len(words) > 1:
                    # Calculate average word similarity within topic
                    similarities = []
                    for i, word1 in enumerate(words):
                        for j, word2 in enumerate(words[i+1:], i+1):
                            # Simple similarity based on character overlap
                            common_chars = len(set(word1.lower()) & set(word2.lower()))
                            total_chars = len(set(word1.lower()) | set(word2.lower()))
                            similarity = common_chars / total_chars if total_chars > 0 else 0
                            similarities.append(similarity)
                    
                    avg_coherence = np.mean(similarities) if similarities else 0
                    coherence_scores.append({
                        'Topic': topic_name,
                        'Coherence': avg_coherence,
                        'Word_Count': len(words),
                        'Frequency': topic_frequencies.get(topic_name, 0)
                    })
            
            if coherence_scores:
                coherence_df = pd.DataFrame(coherence_scores)
                
                # Create coherence vs frequency scatter plot
                fig = px.scatter(
                    coherence_df,
                    x='Frequency',
                    y='Coherence',
                    size='Word_Count',
                    color='Coherence',
                    hover_name='Topic',
                    title=f"{title} - Topic Coherence vs Frequency",
                    labels={'Frequency': 'Topic Frequency', 'Coherence': 'Topic Coherence Score'},
                    color_continuous_scale='viridis'
                )
                
                fig.update_layout(
                    xaxis_title="Topic Frequency",
                    yaxis_title="Topic Coherence Score",
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Create coherence distribution histogram
                fig2 = px.histogram(
                    coherence_df,
                    x='Coherence',
                    nbins=10,
                    title=f"{title} - Topic Coherence Distribution",
                    labels={'Coherence': 'Coherence Score', 'count': 'Number of Topics'},
                    color_discrete_sequence=['#3498db']
                )
                
                fig2.update_layout(
                    xaxis_title="Coherence Score",
                    yaxis_title="Number of Topics",
                    height=400
                )
                
                st.plotly_chart(fig2, use_container_width=True)
                
                # Show top coherent topics
                st.subheader("🏆 Top Coherent Topics")
                top_coherent = coherence_df.nlargest(10, 'Coherence')
                
                fig3 = px.bar(
                    top_coherent,
                    x='Coherence',
                    y='Topic',
                    orientation='h',
                    title=f"{title} - Top 10 Most Coherent Topics",
                    color='Coherence',
                    color_continuous_scale='viridis'
                )
                
                fig3.update_layout(
                    xaxis_title="Coherence Score",
                    yaxis_title="Topics",
                    height=500
                )
                
                st.plotly_chart(fig3, use_container_width=True)
                
                # Coherence statistics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Average Coherence", f"{coherence_df['Coherence'].mean():.3f}")
                with col2:
                    st.metric("Max Coherence", f"{coherence_df['Coherence'].max():.3f}")
                with col3:
                    st.metric("Min Coherence", f"{coherence_df['Coherence'].min():.3f}")
                
                # Create topic similarity matrix
                if len(topic_names) > 1:
                    st.subheader("🔗 Topic Similarity Matrix")
                    
                    # Calculate similarity matrix
                    similarity_matrix = []
                    for topic1 in topic_names:
                        row = []
                        for topic2 in topic_names:
                            if topic1 == topic2:
                                row.append(1.0)
                            else:
                                words1 = set(topic_words[topic1])
                                words2 = set(topic_words[topic2])
                                intersection = len(words1.intersection(words2))
                                union = len(words1.union(words2))
                                similarity = intersection / union if union > 0 else 0
                                row.append(similarity)
                        similarity_matrix.append(row)
                    
                    # Create heatmap
                    fig4 = px.imshow(
                        similarity_matrix,
                        x=topic_names,
                        y=topic_names,
                        title=f"{title} - Topic Similarity Matrix",
                        color_continuous_scale='viridis',
                        aspect='auto'
                    )
                    
                    fig4.update_layout(
                        xaxis_title="Topics",
                        yaxis_title="Topics",
                        height=500
                    )
                    
                    st.plotly_chart(fig4, use_container_width=True)
            else:
                st.info("No coherence data available for visualization")
        else:
            st.info("Need at least 2 topics with multiple words to calculate coherence")
    
    with tab6:
        st.subheader("📝 Sample Topics")
        
        # Display sample topics
    displayed_count = 0
    for i, topics_list in enumerate(topics_data):
        if displayed_count >= 5:  # Only show first 5
            break
            
        if isinstance(topics_list, list) and topics_list:
            displayed_count += 1
            
            # Handle nested topic structure
            if isinstance(topics_list[0], list):
                # Topics are nested lists (multiple topics per record)
                for j, topic in enumerate(topics_list[:3]):  # Show first 3 topics per set
                    if isinstance(topic, list) and topic:
                        # Ensure all items are strings before joining
                        topic_words = [str(word) for word in topic[:5] if word is not None and str(word).strip()]
                        if topic_words:
                            st.write(f"**Topic Set {i+1}.{j+1}:** {', '.join(topic_words)}")
            else:
                # Topics are flat list (single topic per record)
                # Ensure all items are strings before joining
                topic_words = [str(word) for word in topics_list[:5] if word is not None and str(word).strip()]
                if topic_words:
                    st.write(f"**Topic Set {i+1}:** {', '.join(topic_words)}")
    
    if displayed_count == 0:
        st.warning("No valid topics found to display")
    else:
        # Create word cloud from all topics
        all_topic_words = []
        for topics_list in topics_data:
            if isinstance(topics_list, list) and topics_list:
                if isinstance(topics_list[0], list):
                    # Nested topics
                    for topic in topics_list:
                        if isinstance(topic, list):
                            all_topic_words.extend([str(word) for word in topic if word is not None and str(word).strip()])
                else:
                    # Flat topics
                    all_topic_words.extend([str(word) for word in topics_list if word is not None and str(word).strip()])
        
        if all_topic_words:
            # Create word cloud
            wordcloud_fig = create_wordcloud(all_topic_words, f"{title} - Topic Word Cloud")
            if wordcloud_fig:
                st.pyplot(wordcloud_fig)

def display_entity_analysis(df, title):
    """Display entity analysis results with clean, simple visualizations"""
    if 'entities' not in df.columns:
        st.warning(f"No entities column found in {title} data")
        return
    
    # Parse entities data
    entities_data = parse_json_column(df, 'entities')
    if not entities_data:
        st.warning(f"No entities data found in {title}")
        return
    
    # Process entities
    all_entities = []
    entity_types = {}
    
    for entities_list in entities_data:
        if isinstance(entities_list, list) and entities_list:
            for entity_item in entities_list:
                if isinstance(entity_item, list):
                    # Handle nested lists
                    for entity in entity_item:
                        if isinstance(entity, str) and entity.strip():
                            if ':' in entity:
                                name, etype = entity.split(':', 1)
                                all_entities.append(name.strip())
                                entity_types[name.strip()] = etype.strip()
                            else:
                                all_entities.append(entity.strip())
                                entity_types[entity.strip()] = 'UNKNOWN'
                elif isinstance(entity_item, str) and entity_item.strip():
                    # Handle flat strings
                    if ':' in entity_item:
                        name, etype = entity_item.split(':', 1)
                        all_entities.append(name.strip())
                        entity_types[name.strip()] = etype.strip()
                    else:
                        all_entities.append(entity_item.strip())
                        entity_types[entity_item.strip()] = 'UNKNOWN'
    
    if not all_entities:
        st.warning("No valid entities found to display")
        return
    
    # Display basic statistics
    st.info(f"Found {len(all_entities)} entity mentions ({len(set(all_entities))} unique entities)")
    
    st.subheader("📊 Most Frequent Entities")
    
    # Count frequencies
    entity_counts = pd.Series(all_entities).value_counts().head(20)
    
    if len(entity_counts) > 0:
        # Bar chart
        fig = px.bar(
            x=entity_counts.values,
            y=entity_counts.index,
            orientation='h',
            title=f"{title} - Top 20 Entities",
            labels={'x': 'Frequency', 'y': 'Entity'},
            color=entity_counts.values,
            color_continuous_scale='viridis'
        )
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
        
        # Statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Mentions", len(all_entities))
        with col2:
            st.metric("Unique Entities", len(set(all_entities)))
        with col3:
            st.metric("Most Frequent", entity_counts.index[0])
        
        # Word cloud
        if all_entities:
            st.subheader("🌟 Entity Word Cloud")
            try:
                # Clean entity names for word cloud
                cleaned_entities = []
                for entity in all_entities:
                    if isinstance(entity, str) and entity.strip():
                        # Remove any remaining colons and clean the entity name
                        clean_entity = entity.split(':')[0].strip()
                        if clean_entity:
                            cleaned_entities.append(clean_entity)
                
                if cleaned_entities:
                    st.write(f"Generating word cloud from {len(cleaned_entities)} entity mentions...")
                    wordcloud_fig = create_wordcloud(cleaned_entities, f"{title} - Entities")
                    if wordcloud_fig:
                        st.pyplot(wordcloud_fig)
                    else:
                        st.warning("Could not generate word cloud")
                else:
                    st.warning("No valid entities found for word cloud")
            except Exception as e:
                st.error(f"Error generating word cloud: {str(e)}")
                # Show some sample entities for debugging
                st.write("Sample entities:", all_entities[:10])
    else:
        st.info("No entities to display")

def display_ngram_analysis(df, title):
    """Display n-gram analysis results"""
    
    # Check for new separate columns first, fall back to old column
    all_bigrams = []
    all_trigrams = []
    
    # Check if we have document-level n-gram counts (new approach)
    has_doc_counts = 'bigram_counts' in df.columns or 'trigram_counts' in df.columns
    
    if has_doc_counts:
        # Use document-level n-gram counts for meaningful frequency analysis
        bigram_frequencies = {}
        trigram_frequencies = {}
        
        if 'bigram_counts' in df.columns:
            bigram_counts_data = parse_json_object_column(df, 'bigram_counts')
            for doc_counts in bigram_counts_data:
                if isinstance(doc_counts, dict):
                    for bigram, count in doc_counts.items():
                        bigram_frequencies[bigram] = bigram_frequencies.get(bigram, 0) + count
        
        if 'trigram_counts' in df.columns:
            trigram_counts_data = parse_json_object_column(df, 'trigram_counts')
            for doc_counts in trigram_counts_data:
                if isinstance(doc_counts, dict):
                    for trigram, count in doc_counts.items():
                        trigram_frequencies[trigram] = trigram_frequencies.get(trigram, 0) + count
        
        # Get top n-grams by frequency
        if bigram_frequencies:
            sorted_bigrams = sorted(bigram_frequencies.items(), key=lambda x: x[1], reverse=True)
            all_bigrams = [bg[0] for bg in sorted_bigrams[:15]]
        
        if trigram_frequencies:
            sorted_trigrams = sorted(trigram_frequencies.items(), key=lambda x: x[1], reverse=True)
            all_trigrams = [tg[0] for tg in sorted_trigrams[:15]]
    
    # Fallback to old approach if new columns don't exist
    if not all_bigrams and not all_trigrams:
        if 'top_bigrams' in df.columns:
            bigrams_data = parse_json_column(df, 'top_bigrams')
            for bigrams_list in bigrams_data:
                if isinstance(bigrams_list, list):
                    all_bigrams.extend([ngram for ngram in bigrams_list if isinstance(ngram, str)])
                elif isinstance(bigrams_list, str):
                    all_bigrams.append(bigrams_list)
        
        if 'top_trigrams' in df.columns:
            trigrams_data = parse_json_column(df, 'top_trigrams')
            for trigrams_list in trigrams_data:
                if isinstance(trigrams_list, list):
                    all_trigrams.extend([ngram for ngram in trigrams_list if isinstance(ngram, str)])
                elif isinstance(trigrams_list, str):
                    all_trigrams.append(trigrams_list)
        
        # Fall back to old combined column if new columns don't exist
        ngrams_data = []
        if not all_bigrams and not all_trigrams and 'top_ngrams' in df.columns:
            ngrams_data = parse_json_column(df, 'top_ngrams')
            for ngrams_list in ngrams_data:
                if isinstance(ngrams_list, list):
                    for ngram in ngrams_list:
                        if isinstance(ngram, str):
                            word_count = len(ngram.split())
                            if word_count == 2:
                                all_bigrams.append(ngram)
                            elif word_count == 3:
                                all_trigrams.append(ngram)
    
    # Debug information for troubleshooting (optional, controlled by environment variable)
    import os
    if os.getenv("DEBUG_MODE", "false").lower() == "true":
        st.write(f"Debug - Bigrams found: {len(all_bigrams)}")
        st.write(f"Debug - Trigrams found: {len(all_trigrams)}")
        if all_bigrams:
            st.write(f"Debug - Sample bigrams: {all_bigrams[:3]}")
        if all_trigrams:
            st.write(f"Debug - Sample trigrams: {all_trigrams[:3]}")
    
    if not all_bigrams and not all_trigrams:
        st.warning(f"No n-grams data found in {title}")
        st.info("💡 **Tip**: Run the analysis framework to generate n-gram data")
        return
    
    # Display statistics
    st.info(f"Found {len(all_bigrams)} bigrams and {len(all_trigrams)} trigrams")
    
    if has_doc_counts:
        st.success("✅ Using document-level frequency data for meaningful analysis")
    else:
        st.info("⚠️ Using global n-gram lists. Run analysis pipeline for document-level frequencies.")
    
    # Display Bigrams
    if all_bigrams:
        st.subheader(f"{title} - Top Bigrams")
        
        if has_doc_counts and bigram_frequencies:
            # Use document-level frequencies
            sorted_bigrams = sorted(bigram_frequencies.items(), key=lambda x: x[1], reverse=True)[:15]
            plot_df = pd.DataFrame({
                'Bigram': [bg[0] for bg in sorted_bigrams],
                'Frequency': [bg[1] for bg in sorted_bigrams]
            })
        else:
            # Fallback to old approach
            bigram_counts = pd.Series(all_bigrams).value_counts().head(15)
            plot_df = pd.DataFrame({
                'Bigram': bigram_counts.index,
                'Frequency': bigram_counts.values
            })
        
        fig = px.bar(
            plot_df,
            x='Frequency',
            y='Bigram',
            orientation='h',
            title=f"{title} - Top Bigrams by Frequency",
            labels={'x': 'Frequency', 'y': 'Bigrams'},
            color='Frequency',
            color_continuous_scale='viridis'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show frequency statistics for bigrams
        if has_doc_counts and bigram_frequencies:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Bigrams", len(bigram_frequencies))
            with col2:
                st.metric("Most Frequent", plot_df.iloc[0]['Bigram'])
            with col3:
                st.metric("Total Occurrences", sum(bigram_frequencies.values()))
    else:
        st.warning("No bigrams found")
    
    # Display Trigrams
    if all_trigrams:
        st.subheader(f"{title} - Top Trigrams")
        
        if has_doc_counts and trigram_frequencies:
            # Use document-level frequencies
            sorted_trigrams = sorted(trigram_frequencies.items(), key=lambda x: x[1], reverse=True)[:15]
            plot_df = pd.DataFrame({
                'Trigram': [tg[0] for tg in sorted_trigrams],
                'Frequency': [tg[1] for tg in sorted_trigrams]
            })
        else:
            # Fallback to old approach
            trigram_counts = pd.Series(all_trigrams).value_counts().head(15)
            plot_df = pd.DataFrame({
                'Trigram': trigram_counts.index,
                'Frequency': trigram_counts.values
            })
        
        fig = px.bar(
            plot_df,
            x='Frequency',
            y='Trigram',
            orientation='h',
            title=f"{title} - Top Trigrams by Frequency",
            labels={'x': 'Frequency', 'y': 'Trigrams'},
            color='Frequency',
            color_continuous_scale='plasma'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show frequency statistics for trigrams
        if has_doc_counts and trigram_frequencies:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Trigrams", len(trigram_frequencies))
            with col2:
                st.metric("Most Frequent", plot_df.iloc[0]['Trigram'])
            with col3:
                st.metric("Total Occurrences", sum(trigram_frequencies.values()))
    else:
        st.warning("No trigrams found")

def apply_filters(df, filters):
    """Apply filters to dataframe"""
    filtered_df = df.copy()
    
    # Date range filter
    if filters.get('date_range'):
        start_date, end_date = filters['date_range']
        if start_date and end_date:
            date_col = 'publication_date'
            if date_col in filtered_df.columns:
                filtered_df[date_col] = pd.to_datetime(filtered_df[date_col], errors='coerce')
                # Convert date objects to datetime for comparison
                start_datetime = pd.to_datetime(start_date)
                end_datetime = pd.to_datetime(end_date)
                filtered_df = filtered_df[
                    (filtered_df[date_col] >= start_datetime) & 
                    (filtered_df[date_col] <= end_datetime)
                ]
    
    # Sentiment filter
    if filters.get('sentiment') and filters['sentiment'] != 'All':
        if 'sentiment' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['sentiment'] == filters['sentiment']]
    
    # Brand filter
    if filters.get('brand') and filters['brand'] != 'All':
        # Search in title, content, and entities
        brand_keywords = [filters['brand'].lower()]
        mask = pd.Series([False] * len(filtered_df))
        
        for col in ['title', 'content', 'verdict']:
            if col in filtered_df.columns:
                col_mask = filtered_df[col].str.lower().str.contains('|'.join(brand_keywords), na=False)
                mask = mask | col_mask
        
        # Also check entities column
        if 'entities' in filtered_df.columns:
            entities_data = parse_json_column(filtered_df, 'entities')
            for i, entities_list in enumerate(entities_data):
                if isinstance(entities_list, list):
                    for entity in entities_list:
                        if isinstance(entity, str) and filters['brand'].lower() in entity.lower():
                            mask.iloc[i] = True
        
        filtered_df = filtered_df[mask]
    
    # Rating filter (for reviews)
    if filters.get('min_rating') and 'rating' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['rating'] >= filters['min_rating']]
    
    return filtered_df

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<h1 class="main-header">🚗 Car Analysis Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("### Comprehensive Analysis of Car News & Reviews")
    
    # Load data
    with st.spinner("Loading analysis results from database..."):
        news_df, reviews_df = load_data_from_database()
    
    if news_df is None or reviews_df is None:
        st.error("Failed to load data. Please check your database connection.")
        return
    
    # Data type selection
    st.markdown('<div class="filter-box">', unsafe_allow_html=True)
    st.subheader("📊 Data Selection")
    data_type = st.selectbox(
        "Select Data Type:",
        ["📰 News Articles", "⭐ Car Reviews", "📊 Both Datasets"],
        help="Choose which dataset to analyze"
    )
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Filtering options
    st.markdown('<div class="filter-box">', unsafe_allow_html=True)
    st.subheader("🔍 Filters")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # Date range filter - get actual date range from data
        if 'publication_date' in news_df.columns:
            news_df['publication_date'] = pd.to_datetime(news_df['publication_date'], errors='coerce')
            news_min_date = news_df['publication_date'].min()
            news_max_date = news_df['publication_date'].max()
        else:
            news_min_date = news_max_date = datetime.now()
            
        if 'publication_date' in reviews_df.columns:
            reviews_df['publication_date'] = pd.to_datetime(reviews_df['publication_date'], errors='coerce')
            reviews_min_date = reviews_df['publication_date'].min()
            reviews_max_date = reviews_df['publication_date'].max()
        else:
            reviews_min_date = reviews_max_date = datetime.now()
        
        # Use the overall min and max dates
        overall_min_date = min(news_min_date, reviews_min_date)
        overall_max_date = max(news_max_date, reviews_max_date)
        
        # Convert to date objects for the date_input
        if pd.notna(overall_min_date) and pd.notna(overall_max_date):
            default_start = overall_min_date.date()
            default_end = overall_max_date.date()
        else:
            default_start = (datetime.now() - timedelta(days=365)).date()
            default_end = datetime.now().date()
        
        date_range = st.date_input(
            "Date Range",
            value=(default_start, default_end),
            help="Filter by publication date range"
        )
    
    with col2:
        # Sentiment filter
        sentiment_options = ['All', 'positive', 'negative', 'neutral']
        sentiment_filter = st.selectbox(
            "Sentiment",
            sentiment_options,
            help="Filter by sentiment"
        )
    
    with col3:
        # Brand filter
        # Extract available brands from entities
        news_entities = parse_json_column(news_df, 'entities')
        reviews_entities = parse_json_column(reviews_df, 'entities')
        
        available_brands = list(set(
            extract_brands_from_entities(news_entities) + 
            extract_brands_from_entities(reviews_entities)
        ))
        
        brand_options = ['All'] + sorted(available_brands)
        brand_filter = st.selectbox(
            "Brand",
            brand_options,
            help="Filter by car brand"
        )
    
    with col4:
        # Rating filter (for reviews)
        if data_type in ["⭐ Car Reviews", "📊 Both Datasets"]:
            min_rating = st.slider(
                "Minimum Rating",
                min_value=1.0,
                max_value=5.0,
                value=1.0,
                step=0.1,
                help="Filter reviews by minimum rating"
            )
        else:
            min_rating = None
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Apply filters
    filters = {
        'date_range': date_range if len(date_range) == 2 else None,
        'sentiment': sentiment_filter if sentiment_filter != 'All' else None,
        'brand': brand_filter if brand_filter != 'All' else None,
        'min_rating': min_rating
    }
    
    # Filter data based on selection
    if data_type == "📰 News Articles":
        filtered_news = apply_filters(news_df, filters)
        filtered_reviews = pd.DataFrame()
        current_df = filtered_news
        df_name = "News Articles"
    elif data_type == "⭐ Car Reviews":
        filtered_news = pd.DataFrame()
        filtered_reviews = apply_filters(reviews_df, filters)
        current_df = filtered_reviews
        df_name = "Car Reviews"
    else:  # Both datasets
        filtered_news = apply_filters(news_df, filters)
        filtered_reviews = apply_filters(reviews_df, filters)
        current_df = pd.concat([filtered_news, filtered_reviews], ignore_index=True)
        df_name = "Combined Data"
    
    # Show filter results
    st.info(f"📊 Showing {len(current_df)} records after applying filters")
    
    # Create tabs for navigation
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
        "📈 Overview & Metrics",
        "😊 Sentiment Analysis", 
        "📝 Topic Modeling",
        "🏷️ Named Entity Recognition",
        "🔑 Keyword Analysis",
        "📊 N-gram Analysis",
        "📊 Correlation Analysis",
        "⏰ Time Series Analysis",
        "📋 Raw Data Explorer"
    ])
    
    # Overview & Metrics Section
    with tab1:
        st.markdown('<h2 class="section-header">📈 Overview & Key Metrics</h2>', unsafe_allow_html=True)
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if data_type in ["📰 News Articles", "📊 Both Datasets"]:
                st.metric("📰 News Articles", f"{len(filtered_news):,}")
            else:
                st.metric("📰 News Articles", "N/A")
        
        with col2:
            if data_type in ["⭐ Car Reviews", "📊 Both Datasets"]:
                st.metric("⭐ Reviews", f"{len(filtered_reviews):,}")
            else:
                st.metric("⭐ Reviews", "N/A")
        
        with col3:
            if data_type in ["⭐ Car Reviews", "📊 Both Datasets"] and 'rating' in filtered_reviews.columns:
                avg_rating = filtered_reviews['rating'].mean()
                st.metric("Average Rating", f"{avg_rating:.2f} ⭐")
            else:
                st.metric("Average Rating", "N/A")
        
        with col4:
            if 'sentiment' in current_df.columns:
                positive_count = (current_df['sentiment'] == 'positive').sum()
                st.metric("Positive Content", f"{positive_count:,}")
            else:
                st.metric("Positive Content", "N/A")
        
        # Data overview
        if data_type == "📊 Both Datasets":
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📰 News Data Overview")
                if len(filtered_news) > 0:
                    st.write(f"**Shape:** {filtered_news.shape}")
                    st.write(f"**Date Range:** {filtered_news['publication_date'].min()} to {filtered_news['publication_date'].max()}")
                else:
                    st.write("No news data after filtering")
            
            with col2:
                st.subheader("⭐ Reviews Data Overview")
                if len(filtered_reviews) > 0:
                    st.write(f"**Shape:** {filtered_reviews.shape}")
                    st.write(f"**Date Range:** {filtered_reviews['publication_date'].min()} to {filtered_reviews['publication_date'].max()}")
                else:
                    st.write("No reviews data after filtering")
        else:
            st.subheader(f"📊 {df_name} Overview")
            st.write(f"**Shape:** {current_df.shape}")
            if 'publication_date' in current_df.columns:
                st.write(f"**Date Range:** {current_df['publication_date'].min()} to {current_df['publication_date'].max()}")
            st.write(f"**Columns:** {', '.join(current_df.columns)}")
    
    # Sentiment Analysis Section
    with tab2:
        st.markdown('<h2 class="section-header">😊 Sentiment Analysis</h2>', unsafe_allow_html=True)
        
        # Sentiment Value Explanation
        st.info("📊 **Sentiment Value Ranges**: 🟢 Positive (0.1 to 1.0) | 🟡 Neutral (-0.1 to 0.1) | 🔴 Negative (-1.0 to -0.1)")
        
        # Sentiment Distribution Visualizations
        st.subheader("📊 Sentiment Distribution")
        
        if data_type == "📊 Both Datasets":
            col1, col2 = st.columns(2)
            
            with col1:
                if len(filtered_news) > 0:
                    # Pie chart
                    fig = create_sentiment_distribution_chart(filtered_news, "News Sentiment Distribution")
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Bar chart
                    fig = create_sentiment_bar_chart(filtered_news, "News Sentiment Distribution")
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Sentiment score distribution
                    fig = create_sentiment_score_distribution(filtered_news, "News Sentiment Score Distribution")
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                if len(filtered_reviews) > 0:
                    # Pie chart
                    fig = create_sentiment_distribution_chart(filtered_reviews, "Reviews Sentiment Distribution")
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Bar chart
                    fig = create_sentiment_bar_chart(filtered_reviews, "Reviews Sentiment Distribution")
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Sentiment score distribution
                    fig = create_sentiment_score_distribution(filtered_reviews, "Reviews Sentiment Score Distribution")
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
        else:
            # Pie chart
            fig = create_sentiment_distribution_chart(current_df, f"{df_name} Sentiment Distribution")
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            
            # Bar chart
            fig = create_sentiment_bar_chart(current_df, f"{df_name} Sentiment Distribution")
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            
            # Sentiment score distribution
            fig = create_sentiment_score_distribution(current_df, f"{df_name} Sentiment Score Distribution")
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        
        # Sentiment Trends Over Time
        st.subheader("📈 Sentiment Trends Over Time")
        
        if data_type == "📊 Both Datasets":
            col1, col2 = st.columns(2)
            
            with col1:
                if len(filtered_news) > 0:
                    fig = create_sentiment_trend_chart(filtered_news, "News Sentiment Trends")
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                if len(filtered_reviews) > 0:
                    fig = create_sentiment_trend_chart(filtered_reviews, "Reviews Sentiment Trends")
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
        else:
            fig = create_sentiment_trend_chart(current_df, f"{df_name} Sentiment Trends")
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        
        # Rating and Correlation Analysis for Reviews
        if data_type in ["⭐ Car Reviews", "📊 Both Datasets"] and len(filtered_reviews) > 0:
            st.subheader("⭐ Rating Analysis")
            
            # Rating distribution
            fig = create_rating_distribution_chart(filtered_reviews)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            
            # Correlation Analysis
            st.subheader("📊 Correlation Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Sentiment vs Rating correlation
                fig = create_sentiment_vs_rating_scatter(filtered_reviews)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                
                # Review length vs Rating
                fig = create_review_length_vs_rating_scatter(filtered_reviews)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Sentiment vs Review length
                fig = create_sentiment_vs_review_length_scatter(filtered_reviews)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                
                # Additional correlation analysis can be added here
                st.info("Additional correlation analysis available in the Correlation Analysis tab")
    
    # Topic Modeling Section
    with tab3:
        st.markdown('<h2 class="section-header">📝 Topic Modeling</h2>', unsafe_allow_html=True)
        
        if data_type == "📊 Both Datasets":
            col1, col2 = st.columns(2)
            
            with col1:
                if len(filtered_news) > 0:
                    display_topic_analysis(filtered_news, "News Topics")
            
            with col2:
                if len(filtered_reviews) > 0:
                    display_topic_analysis(filtered_reviews, "Review Topics")
        else:
            display_topic_analysis(current_df, f"{df_name} Topics")
    
    # Named Entity Recognition Section
    with tab4:
        st.markdown('<h2 class="section-header">🏷️ Named Entity Recognition</h2>', unsafe_allow_html=True)
        
        if data_type == "📊 Both Datasets":
            col1, col2 = st.columns(2)
            
            with col1:
                if len(filtered_news) > 0:
                    display_entity_analysis(filtered_news, "News Entities")
            
            with col2:
                if len(filtered_reviews) > 0:
                    display_entity_analysis(filtered_reviews, "Review Entities")
        else:
            display_entity_analysis(current_df, f"{df_name} Entities")
    
    # Keyword Analysis Section
    with tab5:
        st.markdown('<h2 class="section-header">🔑 Keyword Analysis</h2>', unsafe_allow_html=True)
        
        if data_type == "📊 Both Datasets":
            col1, col2 = st.columns(2)
            
            with col1:
                if len(filtered_news) > 0:
                    display_keyword_analysis(filtered_news, "News Keywords")
            
            with col2:
                if len(filtered_reviews) > 0:
                    display_keyword_analysis(filtered_reviews, "Review Keywords")
        else:
            display_keyword_analysis(current_df, f"{df_name} Keywords")
    
    # N-gram Analysis Section
    with tab6:
        st.markdown('<h2 class="section-header">📊 N-gram Analysis</h2>', unsafe_allow_html=True)
        
        if data_type == "📊 Both Datasets":
            col1, col2 = st.columns(2)
            
            with col1:
                if len(filtered_news) > 0:
                    display_ngram_analysis(filtered_news, "News N-grams")
            
            with col2:
                if len(filtered_reviews) > 0:
                    display_ngram_analysis(filtered_reviews, "Review N-grams")
        else:
            display_ngram_analysis(current_df, f"{df_name} N-grams")
    
    # Correlation Analysis Section
    with tab7:
        st.markdown('<h2 class="section-header">📊 Correlation Analysis</h2>', unsafe_allow_html=True)
        
        if data_type in ["⭐ Car Reviews", "📊 Both Datasets"] and len(filtered_reviews) > 0:
            # Show correlation metrics
            if 'correlation_score' in filtered_reviews.columns:
                corr_score = filtered_reviews['correlation_score'].iloc[0]
                if pd.notna(corr_score):
                    st.metric("Rating-Sentiment Correlation", f"{corr_score:.3f}")
            
            if 'review_length' in filtered_reviews.columns:
                avg_length = filtered_reviews['review_length'].mean()
                st.metric("Average Review Length", f"{avg_length:.0f} characters")
            
            # Correlation heatmap - Only show rating, price, and sentiment score
            target_cols = ['rating', 'price', 'sentiment_score']
            available_cols = [col for col in target_cols if col in filtered_reviews.columns]
            
            if len(available_cols) > 1:
                # Filter data to only include the target columns
                correlation_data = filtered_reviews[available_cols].copy()
                
                # Remove any rows with NaN values for clean correlation analysis
                correlation_data = correlation_data.dropna()
                
                if len(correlation_data) > 1:
                    corr_matrix = correlation_data.corr()
                    
                    # Create the correlation heatmap
                    fig = px.imshow(
                        corr_matrix,
                        title="Correlation Heatmap - Rating, Price & Sentiment",
                        color_continuous_scale='RdBu',
                        aspect='auto',
                        text_auto=True,  # Show correlation values on the heatmap
                        labels=dict(x="Features", y="Features", color="Correlation")
                    )
                    
                    # Update layout for better readability
                    fig.update_layout(
                        xaxis_title="Features",
                        yaxis_title="Features",
                        height=500
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show correlation insights
                    st.subheader("📊 Correlation Insights")
                    
                    # Display individual correlation pairs
                    for i, col1 in enumerate(available_cols):
                        for j, col2 in enumerate(available_cols):
                            if i < j:  # Only show upper triangle to avoid duplicates
                                corr_value = corr_matrix.loc[col1, col2]
                                st.write(f"**{col1.replace('_', ' ').title()} vs {col2.replace('_', ' ').title()}**: {corr_value:.3f}")
                else:
                    st.warning("Not enough data points for correlation analysis after removing NaN values.")
            else:
                st.warning("Need at least 2 of the following columns for correlation analysis: rating, price, sentiment_score")
                st.info("Available columns: " + ", ".join(filtered_reviews.columns.tolist()))
        else:
            st.info("Correlation analysis is available for reviews data only.")
    
    # Time Series Analysis Section
    with tab8:
        st.markdown('<h2 class="section-header">⏰ Time Series Analysis</h2>', unsafe_allow_html=True)
        
        # Load market data
        with st.spinner("Loading car market data..."):
            market_data = load_market_data()
        
        # Create tabs for different analyses
        time_tab1, time_tab2 = st.tabs([
            "📈 Basic Timeline", 
            "📊 Car Dependency Analysis"
        ])
        
        with time_tab1:
            st.subheader("📈 Publication Timeline")
        
        if data_type == "📊 Both Datasets":
            col1, col2 = st.columns(2)
            
            with col1:
                if len(filtered_news) > 0:
                    fig = create_time_series_chart(filtered_news, 'publication_date', 'title', "News Publication Timeline")
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                if len(filtered_reviews) > 0:
                    fig = create_time_series_chart(filtered_reviews, 'publication_date', 'title', "Reviews Publication Timeline")
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
        else:
            fig = create_time_series_chart(current_df, 'publication_date', 'title', f"{df_name} Publication Timeline")
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        
        with time_tab2:
            st.subheader("📊 Car Dependency Analysis from Census 2021")
            
            if market_data is not None and not market_data.empty:
                st.info("📈 **Data Source**: Census 2021 estimates for car/van availability by local authority")
                
                # Car dependency statistics
                avg_dependency = market_data['Car_Dependency_Rate'].mean()
                high_dependency_areas = market_data[market_data['Car_Dependency_Rate'] > 0.8]
                low_dependency_areas = market_data[market_data['Car_Dependency_Rate'] < 0.6]
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("📊 Average Car Dependency", f"{avg_dependency:.1%}")
                with col2:
                    st.metric("🚗 High Dependency Areas", f"{len(high_dependency_areas)}")
                with col3:
                    st.metric("🚇 Low Dependency Areas", f"{len(low_dependency_areas)}")
                
                # Top 20 car dependency areas - Arranged from lowest to largest
                top_dependency = market_data.nlargest(20, 'Car_Dependency_Rate').sort_values('Car_Dependency_Rate', ascending=True)
                
                fig = px.bar(
                    top_dependency,
                    x='Car_Dependency_Rate',
                    y='Authority',
                    orientation='h',
                    title="Top 20 Areas by Car Dependency Rate (Lowest to Largest)",
                    labels={'Car_Dependency_Rate': 'Car Dependency Rate (%)', 'Authority': 'Local Authority'},
                    color='Car_Dependency_Rate',
                    color_continuous_scale='Reds'
                )
                
                fig.update_layout(height=600)
                st.plotly_chart(fig, use_container_width=True)
                
                # Bottom 20 car dependency areas - Also flipped to show progression
                bottom_dependency = market_data.nsmallest(20, 'Car_Dependency_Rate').sort_values('Car_Dependency_Rate', ascending=True)
                
                fig2 = px.bar(
                    bottom_dependency,
                    x='Car_Dependency_Rate',
                    y='Authority',
                    orientation='h',
                    title="Top 20 Areas with Lowest Car Dependency (Lowest to Highest)",
                    labels={'Car_Dependency_Rate': 'Car Dependency Rate (%)', 'Authority': 'Local Authority'},
                    color='Car_Dependency_Rate',
                    color_continuous_scale='Greens'
                )
                
                fig2.update_layout(height=600)
                st.plotly_chart(fig2, use_container_width=True)
                
                # Market insights
                st.subheader("🎯 Market Insights")
                st.write(f"• **High car dependency** areas ({len(high_dependency_areas)} locations) represent potential markets for car-related services")
                st.write(f"• **Low car dependency** areas ({len(low_dependency_areas)} locations) may indicate better public transport or urban density")
                st.write(f"• **Average dependency rate** of {avg_dependency:.1%} suggests significant car reliance across England and Wales")
            else:
                st.error("Unable to load car market data for analysis")
    
    # Raw Data Explorer Section
    with tab9:
        st.markdown('<h2 class="section-header">📋 Raw Data Explorer</h2>', unsafe_allow_html=True)
        
        if data_type == "📊 Both Datasets":
            tab1, tab2 = st.tabs(["📰 News Data", "⭐ Reviews Data"])
            
            with tab1:
                if len(filtered_news) > 0:
                    st.dataframe(filtered_news, use_container_width=True)
                    
                    # Download button
                    csv = filtered_news.to_csv(index=False)
                    st.download_button(
                        label="Download News Data as CSV",
                        data=csv,
                        file_name=f"filtered_news_data_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )
                else:
                    st.info("No news data after filtering")
            
            with tab2:
                if len(filtered_reviews) > 0:
                    st.dataframe(filtered_reviews, use_container_width=True)
                    
                    # Download button
                    csv = filtered_reviews.to_csv(index=False)
                    st.download_button(
                        label="Download Reviews Data as CSV",
                        data=csv,
                        file_name=f"filtered_reviews_data_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )
                else:
                    st.info("No reviews data after filtering")
        else:
            st.dataframe(current_df, use_container_width=True)
            
            # Download button
            csv = current_df.to_csv(index=False)
            st.download_button(
                label=f"Download {df_name} as CSV",
                data=csv,
                file_name=f"filtered_{df_name.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )

if __name__ == "__main__":
    main()
