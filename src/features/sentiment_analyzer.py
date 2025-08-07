"""
Sentiment Analysis Module
Performs sentiment analysis using VADER and TextBlob
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from src.config.config import COLOR_PALETTE


class SentimentAnalyzer:
    """Handles sentiment analysis using multiple methods"""
    
    def __init__(self):
        """Initialize the sentiment analyzer"""
        self.vader_analyzer = SentimentIntensityAnalyzer()
    
    def get_sentiment_scores(self, text):
        """
        Calculate sentiment scores using VADER and TextBlob
        
        Args:
            text: Input text string
            
        Returns:
            pd.Series: Series containing sentiment scores and label
        """
        if pd.isna(text) or str(text).strip() == "":
            return pd.Series([0, 0, 0, 0, 0, 0, 'neutral'])
        
        # VADER sentiment analysis
        vs = self.vader_analyzer.polarity_scores(str(text))
        
        # TextBlob sentiment analysis
        blob = TextBlob(str(text))
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        
        # Determine sentiment label based on VADER compound score
        if vs['compound'] >= 0.05:
            label = 'positive'
        elif vs['compound'] <= -0.05:
            label = 'negative'
        else:
            label = 'neutral'
        
        return pd.Series([
            vs['compound'],  # VADER compound score
            vs['pos'],       # VADER positive score
            vs['neu'],       # VADER neutral score
            vs['neg'],       # VADER negative score
            polarity,        # TextBlob polarity
            subjectivity,    # TextBlob subjectivity
            label           # Sentiment label
        ])
    
    def analyze_sentiment(self, data, title_or_column="Text", title="Text"):
        """
        Perform sentiment analysis on data (handles both dataframe and series)
        
        Args:
            data: Pandas dataframe or series
            title_or_column: If data is dataframe, this is the text column name. If data is series, this is the title.
            title: Title for the analysis (only used if data is dataframe)
        """
        if isinstance(data, pd.DataFrame):
            # Handle dataframe input
            text_column = title_or_column
            return self._analyze_sentiment_dataframe(data, text_column, title)
        else:
            # Handle series input
            title = title_or_column
            return self._analyze_sentiment_series(data, title)
    
    def _analyze_sentiment_dataframe(self, df, text_column, title="Text"):
        """
        Perform sentiment analysis on a dataframe
        
        Args:
            df: Pandas dataframe
            text_column: Name of the text column
            title: Title for the analysis
        """
        print(f"\n=== {title} - Sentiment Analysis ===")
        
        # Define sentiment column names
        sentiment_cols = [
            'vader_compound', 'vader_pos', 'vader_neu', 'vader_neg',
            'tb_polarity', 'tb_subjectivity', 'sentiment_label'
        ]
        
        # Apply sentiment analysis
        df[sentiment_cols] = df[text_column].apply(self.get_sentiment_scores)
        
        # Visualize sentiment distribution
        self._plot_sentiment_distribution(df, title)
        
        # Print sentiment statistics
        self._print_sentiment_stats(df, title)
        
        return df
    
    def _analyze_sentiment_series(self, text_series, title="Text"):
        """
        Perform sentiment analysis on a text series
        
        Args:
            text_series: Pandas series containing text data
            title: Title for the analysis
        """
        print(f"\n=== {title} - Sentiment Analysis ===")
        
        # Convert series to dataframe for analysis
        df = pd.DataFrame({'text': text_series})
        
        # Define sentiment column names
        sentiment_cols = [
            'vader_compound', 'vader_pos', 'vader_neu', 'vader_neg',
            'tb_polarity', 'tb_subjectivity', 'sentiment_label'
        ]
        
        # Apply sentiment analysis
        df[sentiment_cols] = df['text'].apply(self.get_sentiment_scores)
        
        # Visualize sentiment distribution
        self._plot_sentiment_distribution(df, title)
        
        # Print sentiment statistics
        self._print_sentiment_stats(df, title)
        
        return df
    
    def _plot_sentiment_distribution(self, df, title):
        """
        Create sentiment distribution visualization
        
        Args:
            df: Dataframe with sentiment columns
            title: Title for the plot
        """
        if 'sentiment_label' not in df.columns:
            return
        
        plt.figure(figsize=(10, 6))
        sns.countplot(x='sentiment_label', data=df, palette='coolwarm')
        plt.title(f'Sentiment Distribution in {title}')
        plt.xlabel('Sentiment')
        plt.ylabel('Count')
        plt.show()
        
        # Additional plots for sentiment scores
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # VADER scores distribution
        if 'vader_compound' in df.columns:
            axes[0, 0].hist(df['vader_compound'], bins=30, alpha=0.7, color='skyblue')
            axes[0, 0].set_title('VADER Compound Score Distribution')
            axes[0, 0].set_xlabel('Compound Score')
            axes[0, 0].set_ylabel('Frequency')
        
        if 'tb_polarity' in df.columns:
            axes[0, 1].hist(df['tb_polarity'], bins=30, alpha=0.7, color='lightgreen')
            axes[0, 1].set_title('TextBlob Polarity Distribution')
            axes[0, 1].set_xlabel('Polarity Score')
            axes[0, 1].set_ylabel('Frequency')
        
        if 'tb_subjectivity' in df.columns:
            axes[1, 0].hist(df['tb_subjectivity'], bins=30, alpha=0.7, color='orange')
            axes[1, 0].set_title('TextBlob Subjectivity Distribution')
            axes[1, 0].set_xlabel('Subjectivity Score')
            axes[1, 0].set_ylabel('Frequency')
        
        # Sentiment comparison
        if 'vader_compound' in df.columns and 'tb_polarity' in df.columns:
            axes[1, 1].scatter(df['vader_compound'], df['tb_polarity'], alpha=0.6)
            axes[1, 1].set_title('VADER vs TextBlob Sentiment')
            axes[1, 1].set_xlabel('VADER Compound Score')
            axes[1, 1].set_ylabel('TextBlob Polarity')
        
        plt.tight_layout()
        plt.show()
    
    def _print_sentiment_stats(self, df, title):
        """
        Print sentiment analysis statistics
        
        Args:
            df: Dataframe with sentiment columns
            title: Title for the analysis
        """
        if 'sentiment_label' in df.columns:
            print(f"\n{title} Sentiment Distribution:")
            sentiment_dist = df['sentiment_label'].value_counts()
            for sentiment, count in sentiment_dist.items():
                percentage = (count / len(df)) * 100
                print(f"- {sentiment.capitalize()}: {count} ({percentage:.1f}%)")
        
        # Print score statistics
        if 'vader_compound' in df.columns:
            print(f"\n{title} VADER Compound Score Statistics:")
            print(f"- Mean: {df['vader_compound'].mean():.3f}")
            print(f"- Median: {df['vader_compound'].median():.3f}")
            print(f"- Std: {df['vader_compound'].std():.3f}")
            print(f"- Range: {df['vader_compound'].min():.3f} to {df['vader_compound'].max():.3f}")
        
        if 'tb_polarity' in df.columns:
            print(f"\n{title} TextBlob Polarity Statistics:")
            print(f"- Mean: {df['tb_polarity'].mean():.3f}")
            print(f"- Median: {df['tb_polarity'].median():.3f}")
            print(f"- Std: {df['tb_polarity'].std():.3f}")
    
    def get_sentiment_insights(self, data, title="Text"):
        """
        Extract insights from sentiment analysis (handles both dataframe and series)
        
        Args:
            data: Dataframe with sentiment columns or text series
            title: Title for the analysis
            
        Returns:
            dict: Dictionary containing sentiment insights
        """
        if isinstance(data, pd.DataFrame):
            # If dataframe already has sentiment columns, use them
            if 'sentiment_label' in data.columns:
                df = data
            else:
                # If dataframe doesn't have sentiment columns, analyze it
                df = self._analyze_sentiment_series(data.iloc[:, 0], title)
        else:
            # If series, analyze it
            df = self._analyze_sentiment_series(data, title)
        
        # Calculate insights
        total_documents = len(df)
        
        # Sentiment distribution
        if 'sentiment_label' in df.columns:
            sentiment_dist = df['sentiment_label'].value_counts()
            positive_count = sentiment_dist.get('positive', 0)
            negative_count = sentiment_dist.get('negative', 0)
            neutral_count = sentiment_dist.get('neutral', 0)
            
            positive_percentage = (positive_count / total_documents) * 100
            negative_percentage = (negative_count / total_documents) * 100
            neutral_percentage = (neutral_count / total_documents) * 100
        else:
            positive_count = negative_count = neutral_count = 0
            positive_percentage = negative_percentage = neutral_percentage = 0
        
        # Average sentiment scores
        avg_sentiment = df['vader_compound'].mean() if 'vader_compound' in df.columns else 0
        
        insights = {
            'title': title,
            'total_documents': total_documents,
            'avg_sentiment': avg_sentiment,
            'positive_count': positive_count,
            'negative_count': negative_count,
            'neutral_count': neutral_count,
            'positive_percentage': positive_percentage,
            'negative_percentage': negative_percentage,
            'neutral_percentage': neutral_percentage,
            'sentiment_distribution': sentiment_dist.to_dict() if 'sentiment_label' in df.columns else {},
            'dominant_sentiment': sentiment_dist.index[0] if 'sentiment_label' in df.columns and len(sentiment_dist) > 0 else None
        }
        
        # Add VADER stats if available
        if 'vader_compound' in df.columns:
            insights['vader_stats'] = {
                'mean': df['vader_compound'].mean(),
                'median': df['vader_compound'].median(),
                'std': df['vader_compound'].std(),
                'min': df['vader_compound'].min(),
                'max': df['vader_compound'].max()
            }
        
        # Add TextBlob stats if available
        if 'tb_polarity' in df.columns:
            insights['textblob_stats'] = {
                'mean_polarity': df['tb_polarity'].mean(),
                'mean_subjectivity': df['tb_subjectivity'].mean() if 'tb_subjectivity' in df.columns else None
            }
        
        return insights


def main():
    """Test sentiment analysis with actual datasets"""
    import pandas as pd
    
    print("="*60)
    print("SENTIMENT ANALYSIS TEST - CAR NEWS & REVIEWS")
    print("="*60)
    
    try:
        # Load datasets
        print("\nLoading datasets...")
        car_news_df = pd.read_csv('car_news_dataset.csv')
        car_reviews_df = pd.read_csv('car_reviews_dataset.csv')
        
        print(f"✓ Car News Dataset: {car_news_df.shape}")
        print(f"✓ Car Reviews Dataset: {car_reviews_df.shape}")
        
        # Initialize sentiment analyzer
        sentiment_analyzer = SentimentAnalyzer()
        
        # Analyze Car News Sentiment
        print("\n" + "="*40)
        print("ANALYZING CAR NEWS SENTIMENT")
        print("="*40)
        
        if 'content' in car_news_df.columns:
            news_text = car_news_df['content'].dropna()
            print(f"Processing {len(news_text)} news articles...")
            
            # Analyze sentiment
            news_sentiment = sentiment_analyzer.analyze_sentiment(news_text, "Car News")
            
            # Get insights
            news_insights = sentiment_analyzer.get_sentiment_insights(news_text, "Car News")
            print(f"\nNews Sentiment Summary:")
            print(f"- Average sentiment: {news_insights.get('avg_sentiment', 0):.3f}")
            print(f"- Positive texts: {news_insights.get('positive_count', 0)} ({news_insights.get('positive_percentage', 0):.1f}%)")
            print(f"- Negative texts: {news_insights.get('negative_count', 0)} ({news_insights.get('negative_percentage', 0):.1f}%)")
            print(f"- Neutral texts: {news_insights.get('neutral_count', 0)} ({news_insights.get('neutral_percentage', 0):.1f}%)")
        else:
            print("❌ 'content' column not found in news dataset")
        
        # Analyze Car Reviews Sentiment
        print("\n" + "="*40)
        print("ANALYZING CAR REVIEWS SENTIMENT")
        print("="*40)
        
        if 'Review' in car_reviews_df.columns:
            review_text = car_reviews_df['Review'].dropna()
            print(f"Processing {len(review_text)} reviews...")
            
            # Analyze sentiment
            review_sentiment = sentiment_analyzer.analyze_sentiment(review_text, "Car Reviews")
            
            # Get insights
            review_insights = sentiment_analyzer.get_sentiment_insights(review_text, "Car Reviews")
            print(f"\nReview Sentiment Summary:")
            print(f"- Average sentiment: {review_insights.get('avg_sentiment', 0):.3f}")
            print(f"- Positive reviews: {review_insights.get('positive_count', 0)} ({review_insights.get('positive_percentage', 0):.1f}%)")
            print(f"- Negative reviews: {review_insights.get('negative_count', 0)} ({review_insights.get('negative_percentage', 0):.1f}%)")
            print(f"- Neutral reviews: {review_insights.get('neutral_count', 0)} ({review_insights.get('neutral_percentage', 0):.1f}%)")
        else:
            print("❌ 'Review' column not found in reviews dataset")
        
        print("\n" + "="*60)
        print("SENTIMENT ANALYSIS TEST COMPLETE!")
        print("="*60)
        
    except Exception as e:
        print(f"❌ Error during sentiment analysis test: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 