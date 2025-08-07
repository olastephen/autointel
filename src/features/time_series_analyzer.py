"""
Time Series Analysis Module
Performs time series analysis of sentiment and trends over time
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from src.config.config import FIGURE_SIZE


class TimeSeriesAnalyzer:
    """Handles time series analysis of sentiment and trends"""
    
    def __init__(self):
        """Initialize the time series analyzer"""
        pass
    
    def analyze_sentiment_trends(self, df, text_column, date_column, title="Data"):
        """
        Analyze sentiment trends over time
        
        Args:
            df: Pandas dataframe
            text_column: Name of the text column
            date_column: Name of the date column
            title: Title for the analysis
        """
        print(f"\n=== {title} - Time Series Sentiment Analysis ===")
        
        # Check if required columns exist
        if date_column not in df.columns:
            print(f"Date column '{date_column}' not found")
            return {}
        
        if text_column not in df.columns:
            print(f"Text column '{text_column}' not found")
            return {}
        
        try:
            # Convert date column to datetime
            df_copy = df.copy()
            df_copy[date_column] = pd.to_datetime(df_copy[date_column])
            
            # Add sentiment analysis if not already present
            if 'vader_compound' not in df_copy.columns:
                from sentiment_analyzer import SentimentAnalyzer
                sentiment_analyzer = SentimentAnalyzer()
                df_copy = sentiment_analyzer.analyze_sentiment(df_copy, text_column, title)
            
            # Create time-based analysis
            results = self._perform_time_analysis(df_copy, date_column, title)
            
            return results
            
        except Exception as e:
            print(f"Error in time series analysis: {e}")
            return {}
    
    def _perform_time_analysis(self, df, date_column, title):
        """
        Perform time-based analysis
        
        Args:
            df: Dataframe with datetime column
            date_column: Name of the date column
            title: Title for the analysis
            
        Returns:
            dict: Dictionary containing time series results
        """
        # Add time-based columns
        df['year'] = df[date_column].dt.year
        df['month'] = df[date_column].dt.to_period('M')
        df['quarter'] = df[date_column].dt.to_period('Q')
        df['week'] = df[date_column].dt.to_period('W')
        
        results = {}
        
        # Monthly sentiment analysis
        if 'vader_compound' in df.columns:
            monthly_results = self._analyze_monthly_sentiment(df, title)
            results['monthly'] = monthly_results
        
        # Quarterly sentiment analysis
        if 'vader_compound' in df.columns:
            quarterly_results = self._analyze_quarterly_sentiment(df, title)
            results['quarterly'] = quarterly_results
        
        # Yearly sentiment analysis
        if 'vader_compound' in df.columns:
            yearly_results = self._analyze_yearly_sentiment(df, title)
            results['yearly'] = yearly_results
        
        # Document count trends
        count_results = self._analyze_document_counts(df, date_column, title)
        results['counts'] = count_results
        
        return results
    
    def _analyze_monthly_sentiment(self, df, title):
        """
        Analyze monthly sentiment trends
        
        Args:
            df: Dataframe with sentiment and month columns
            title: Title for the analysis
            
        Returns:
            dict: Monthly sentiment analysis results
        """
        monthly_sentiment = df.groupby('month').agg({
            'vader_compound': ['mean', 'std', 'count'],
            'tb_polarity': 'mean' if 'tb_polarity' in df.columns else None,
            'tb_subjectivity': 'mean' if 'tb_subjectivity' in df.columns else None
        }).dropna()
        
        # Flatten column names
        monthly_sentiment.columns = ['_'.join(col).strip() for col in monthly_sentiment.columns]
        
        # Plot monthly sentiment trends
        self._plot_monthly_sentiment(monthly_sentiment, title)
        
        # Print monthly insights
        self._print_monthly_insights(monthly_sentiment, title)
        
        return {
            'data': monthly_sentiment,
            'trend': self._calculate_trend(monthly_sentiment['vader_compound_mean'])
        }
    
    def _plot_monthly_sentiment(self, monthly_data, title):
        """
        Create monthly sentiment visualization
        
        Args:
            monthly_data: Monthly sentiment data
            title: Title for the plots
        """
        fig, axes = plt.subplots(2, 2, figsize=FIGURE_SIZE)
        
        # Monthly average sentiment
        monthly_data['vader_compound_mean'].plot(kind='line', marker='o', ax=axes[0, 0])
        axes[0, 0].set_title(f'{title} - Monthly Average Sentiment')
        axes[0, 0].set_ylabel('Average VADER Compound Score')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Monthly document count
        monthly_data['vader_compound_count'].plot(kind='bar', ax=axes[0, 1])
        axes[0, 1].set_title(f'{title} - Monthly Document Count')
        axes[0, 1].set_ylabel('Number of Documents')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Sentiment volatility (std)
        monthly_data['vader_compound_std'].plot(kind='line', marker='s', ax=axes[1, 0], color='orange')
        axes[1, 0].set_title(f'{title} - Monthly Sentiment Volatility')
        axes[1, 0].set_ylabel('Standard Deviation')
        axes[1, 0].grid(True, alpha=0.3)
        
        # TextBlob polarity if available
        if 'tb_polarity_mean' in monthly_data.columns:
            monthly_data['tb_polarity_mean'].plot(kind='line', marker='^', ax=axes[1, 1], color='green')
            axes[1, 1].set_title(f'{title} - Monthly TextBlob Polarity')
            axes[1, 1].set_ylabel('Average Polarity Score')
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].text(0.5, 0.5, 'TextBlob data not available', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title(f'{title} - TextBlob Polarity')
        
        plt.tight_layout()
        plt.show()
    
    def _print_monthly_insights(self, monthly_data, title):
        """
        Print monthly sentiment insights
        
        Args:
            monthly_data: Monthly sentiment data
            title: Title for the analysis
        """
        print(f"\n{title} Monthly Sentiment Insights:")
        
        # Overall statistics
        overall_mean = monthly_data['vader_compound_mean'].mean()
        overall_std = monthly_data['vader_compound_mean'].std()
        
        print(f"- Overall average sentiment: {overall_mean:.3f}")
        print(f"- Sentiment variability: {overall_std:.3f}")
        
        # Best and worst months
        best_month = monthly_data['vader_compound_mean'].idxmax()
        worst_month = monthly_data['vader_compound_mean'].idxmin()
        
        print(f"- Best sentiment month: {best_month} ({monthly_data['vader_compound_mean'].max():.3f})")
        print(f"- Worst sentiment month: {worst_month} ({monthly_data['vader_compound_mean'].min():.3f})")
        
        # Trend analysis
        trend = self._calculate_trend(monthly_data['vader_compound_mean'])
        print(f"- Sentiment trend: {trend}")
    
    def _analyze_quarterly_sentiment(self, df, title):
        """
        Analyze quarterly sentiment trends
        
        Args:
            df: Dataframe with sentiment and quarter columns
            title: Title for the analysis
            
        Returns:
            dict: Quarterly sentiment analysis results
        """
        quarterly_sentiment = df.groupby('quarter').agg({
            'vader_compound': ['mean', 'std', 'count']
        }).dropna()
        
        # Flatten column names
        quarterly_sentiment.columns = ['_'.join(col).strip() for col in quarterly_sentiment.columns]
        
        # Plot quarterly trends
        self._plot_quarterly_sentiment(quarterly_sentiment, title)
        
        return {
            'data': quarterly_sentiment,
            'trend': self._calculate_trend(quarterly_sentiment['vader_compound_mean'])
        }
    
    def _plot_quarterly_sentiment(self, quarterly_data, title):
        """
        Create quarterly sentiment visualization
        
        Args:
            quarterly_data: Quarterly sentiment data
            title: Title for the plots
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Quarterly average sentiment
        quarterly_data['vader_compound_mean'].plot(kind='line', marker='o', ax=ax1)
        ax1.set_title(f'{title} - Quarterly Average Sentiment')
        ax1.set_ylabel('Average VADER Compound Score')
        ax1.grid(True, alpha=0.3)
        
        # Quarterly document count
        quarterly_data['vader_compound_count'].plot(kind='bar', ax=ax2)
        ax2.set_title(f'{title} - Quarterly Document Count')
        ax2.set_ylabel('Number of Documents')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
    
    def _analyze_yearly_sentiment(self, df, title):
        """
        Analyze yearly sentiment trends
        
        Args:
            df: Dataframe with sentiment and year columns
            title: Title for the analysis
            
        Returns:
            dict: Yearly sentiment analysis results
        """
        yearly_sentiment = df.groupby('year').agg({
            'vader_compound': ['mean', 'std', 'count']
        }).dropna()
        
        # Flatten column names
        yearly_sentiment.columns = ['_'.join(col).strip() for col in yearly_sentiment.columns]
        
        # Plot yearly trends
        self._plot_yearly_sentiment(yearly_sentiment, title)
        
        return {
            'data': yearly_sentiment,
            'trend': self._calculate_trend(yearly_sentiment['vader_compound_mean'])
        }
    
    def _plot_yearly_sentiment(self, yearly_data, title):
        """
        Create yearly sentiment visualization
        
        Args:
            yearly_data: Yearly sentiment data
            title: Title for the plots
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Yearly average sentiment
        yearly_data['vader_compound_mean'].plot(kind='line', marker='o', ax=ax1)
        ax1.set_title(f'{title} - Yearly Average Sentiment')
        ax1.set_ylabel('Average VADER Compound Score')
        ax1.grid(True, alpha=0.3)
        
        # Yearly document count
        yearly_data['vader_compound_count'].plot(kind='bar', ax=ax2)
        ax2.set_title(f'{title} - Yearly Document Count')
        ax2.set_ylabel('Number of Documents')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
    
    def _analyze_document_counts(self, df, date_column, title):
        """
        Analyze document count trends
        
        Args:
            df: Dataframe with date column
            date_column: Name of the date column
            title: Title for the analysis
            
        Returns:
            dict: Document count analysis results
        """
        # Daily counts
        daily_counts = df.groupby(df[date_column].dt.date).size()
        
        # Weekly counts
        weekly_counts = df.groupby('week').size()
        
        # Monthly counts
        monthly_counts = df.groupby('month').size()
        
        # Plot count trends
        self._plot_count_trends(daily_counts, weekly_counts, monthly_counts, title)
        
        return {
            'daily_counts': daily_counts,
            'weekly_counts': weekly_counts,
            'monthly_counts': monthly_counts
        }
    
    def _plot_count_trends(self, daily_counts, weekly_counts, monthly_counts, title):
        """
        Create visualization for document count trends
        
        Args:
            daily_counts: Daily document counts
            weekly_counts: Weekly document counts
            monthly_counts: Monthly document counts
            title: Title for the plots
        """
        fig, axes = plt.subplots(2, 2, figsize=FIGURE_SIZE)
        
        # Daily counts
        daily_counts.plot(kind='line', ax=axes[0, 0], alpha=0.7)
        axes[0, 0].set_title(f'{title} - Daily Document Count')
        axes[0, 0].set_ylabel('Number of Documents')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Weekly counts
        weekly_counts.plot(kind='line', marker='o', ax=axes[0, 1])
        axes[0, 1].set_title(f'{title} - Weekly Document Count')
        axes[0, 1].set_ylabel('Number of Documents')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3)
        
        # Monthly counts
        monthly_counts.plot(kind='bar', ax=axes[1, 0])
        axes[1, 0].set_title(f'{title} - Monthly Document Count')
        axes[1, 0].set_ylabel('Number of Documents')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Monthly counts trend
        monthly_counts.plot(kind='line', marker='s', ax=axes[1, 1], color='orange')
        axes[1, 1].set_title(f'{title} - Monthly Document Count Trend')
        axes[1, 1].set_ylabel('Number of Documents')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def _calculate_trend(self, series):
        """
        Calculate trend direction for a time series
        
        Args:
            series: Time series data
            
        Returns:
            str: Trend description
        """
        if len(series) < 2:
            return "Insufficient data"
        
        # Simple linear trend calculation
        x = np.arange(len(series))
        y = series.values
        
        # Remove NaN values
        mask = ~np.isnan(y)
        if np.sum(mask) < 2:
            return "Insufficient valid data"
        
        x_clean = x[mask]
        y_clean = y[mask]
        
        # Calculate slope
        slope = np.polyfit(x_clean, y_clean, 1)[0]
        
        if slope > 0.01:
            return "Increasing"
        elif slope < -0.01:
            return "Decreasing"
        else:
            return "Stable"
    
    def get_time_series_insights(self, df, date_column, title="Data"):
        """
        Extract comprehensive time series insights
        
        Args:
            df: Pandas dataframe
            date_column: Name of the date column
            title: Title for the analysis
            
        Returns:
            dict: Dictionary containing time series insights
        """
        # Use the analyze_time_series method to get results
        time_series_results = self.analyze_time_series(df, date_column, title)
        
        # If no results, return empty insights
        if not time_series_results:
            return {
                'title': title,
                'date_range': 'N/A',
                'total_periods': 0,
                'avg_reviews_per_period': 0,
                'peak_period': 'N/A',
                'trend_direction': 'N/A',
                'seasonal_patterns': {}
            }
        
        # Return the insights from analyze_time_series
        return time_series_results
    
    def analyze_time_series(self, df, date_column, title="Data"):
        """
        Analyze time series patterns in the data (main method for testing)
        
        Args:
            df: Pandas dataframe
            date_column: Name of the date column
            title: Title for the analysis
            
        Returns:
            dict: Dictionary containing time series analysis results
        """
        print(f"\n=== {title} - Time Series Analysis ===")
        
        # Check if date column exists
        if date_column not in df.columns:
            print(f"Date column '{date_column}' not found")
            return {}
        
        try:
            # Convert date column to datetime with custom parsing
            df_copy = df.copy()
            
            # Check if dates are already parsed as datetime objects
            if df_copy[date_column].dtype == 'object':
                # Parse dates if they're still strings
                df_copy[date_column] = self._clean_and_parse_dates(df_copy[date_column])
            else:
                # Dates are already parsed, just ensure they're datetime
                df_copy[date_column] = pd.to_datetime(df_copy[date_column], errors='coerce')
            
            df_copy = df_copy.dropna(subset=[date_column])
            
            if len(df_copy) == 0:
                print("No valid dates found in the dataset")
                return {}
            
            # Add time-based columns
            df_copy['year'] = df_copy[date_column].dt.year
            df_copy['month'] = df_copy[date_column].dt.to_period('M')
            df_copy['quarter'] = df_copy[date_column].dt.to_period('Q')
            df_copy['week'] = df_copy[date_column].dt.to_period('W')
            
            # Analyze document counts over time
            count_results = self._analyze_document_counts(df_copy, date_column, title)
            
            # Calculate insights
            insights = {
                'date_range': f"{df_copy[date_column].min().strftime('%Y-%m-%d')} to {df_copy[date_column].max().strftime('%Y-%m-%d')}",
                'total_periods': len(df_copy[date_column].dt.to_period('M').unique()),
                'avg_reviews_per_period': len(df_copy) / len(df_copy[date_column].dt.to_period('M').unique()),
                'peak_period': df_copy.groupby(df_copy[date_column].dt.to_period('M')).size().idxmax().strftime('%Y-%m'),
                'trend_direction': self._calculate_trend(df_copy.groupby(df_copy[date_column].dt.to_period('M')).size()),
                'seasonal_patterns': {
                    'monthly': df_copy.groupby(df_copy[date_column].dt.month).size().to_dict(),
                    'quarterly': df_copy.groupby(df_copy[date_column].dt.quarter).size().to_dict()
                }
            }
            
            return insights
            
        except Exception as e:
            print(f"Error in time series analysis: {e}")
            return {}
    
    def _clean_and_parse_dates(self, date_series):
        """
        Clean and parse dates from the specific format in the dataset
        
        Args:
            date_series: Pandas series containing date strings
            
        Returns:
            pd.Series: Parsed datetime series
        """
        import re
        
        def clean_date(date_str):
            if pd.isna(date_str):
                return pd.NaT
            
            # Convert to string
            date_str = str(date_str)
            
            # Extract date part from format like ' on 02/02/17 19:53 PM (PST)'
            # Look for patterns like MM/DD/YY or MM/DD/YYYY
            date_patterns = [
                r'(\d{1,2}/\d{1,2}/\d{2,4})',  # MM/DD/YY or MM/DD/YYYY
                r'(\d{4}-\d{1,2}-\d{1,2})',    # YYYY-MM-DD
                r'(\d{1,2}-\d{1,2}-\d{2,4})',  # MM-DD-YY or MM-DD-YYYY
            ]
            
            for pattern in date_patterns:
                match = re.search(pattern, date_str)
                if match:
                    date_part = match.group(1)
                    try:
                        # Try to parse the date part
                        if len(date_part.split('/')[2]) == 2:
                            # Convert 2-digit year to 4-digit
                            parts = date_part.split('/')
                            if int(parts[2]) < 50:
                                parts[2] = '20' + parts[2]
                            else:
                                parts[2] = '19' + parts[2]
                            date_part = '/'.join(parts)
                        
                        return pd.to_datetime(date_part, errors='coerce')
                    except:
                        continue
            
            return pd.NaT
        
        # Apply cleaning to the series
        cleaned_dates = date_series.apply(clean_date)
        
        # Count valid dates
        valid_count = cleaned_dates.notna().sum()
        total_count = len(date_series)
        print(f"Date parsing: {valid_count} valid dates out of {total_count} ({valid_count/total_count*100:.1f}%)")
        
        return cleaned_dates


def main():
    """Test time series analysis with actual datasets"""
    import pandas as pd
    
    print("="*60)
    print("TIME SERIES ANALYSIS TEST - CAR REVIEWS")
    print("="*60)
    
    try:
        # Load datasets
        print("\nLoading datasets...")
        car_reviews_df = pd.read_csv('car_reviews_dataset.csv')
        
        print(f"✓ Car Reviews Dataset: {car_reviews_df.shape}")
        
        # Initialize time series analyzer
        time_analyzer = TimeSeriesAnalyzer()
        
        # Analyze time series in car reviews
        print("\n" + "="*40)
        print("ANALYZING TIME SERIES IN CAR REVIEWS")
        print("="*40)
        
        # Check for date column
        if 'Review_Date' in car_reviews_df.columns:
            print(f"Processing reviews with dates...")
            
            # Convert date column using the same method as analyze_time_series
            car_reviews_df['Review_Date'] = time_analyzer._clean_and_parse_dates(car_reviews_df['Review_Date'])
            reviews_with_dates = car_reviews_df.dropna(subset=['Review_Date'])
            
            print(f"Reviews with valid dates: {len(reviews_with_dates)}")
            
            if len(reviews_with_dates) > 0:
                # Analyze time series
                time_series = time_analyzer.analyze_time_series(reviews_with_dates, 'Review_Date', "Car Reviews")
                
                # Get insights
                insights = time_analyzer.get_time_series_insights(reviews_with_dates, 'Review_Date', "Car Reviews")
                print(f"\nTime Series Analysis Summary:")
                print(f"- Date range: {insights.get('date_range', 'N/A')}")
                print(f"- Total time periods: {insights.get('total_periods', 0)}")
                print(f"- Average reviews per period: {insights.get('avg_reviews_per_period', 0):.2f}")
                print(f"- Peak activity period: {insights.get('peak_period', 'N/A')}")
                print(f"- Trend direction: {insights.get('trend_direction', 'N/A')}")
                
                # Show seasonal patterns if available
                seasonal_patterns = insights.get('seasonal_patterns', {})
                if seasonal_patterns:
                    print(f"\nSeasonal Patterns:")
                    for period, count in seasonal_patterns.items():
                        print(f"  - {period}: {count} reviews")
            else:
                print("❌ No valid dates found in the dataset")
        else:
            print("❌ 'Review_Date' column not found in reviews dataset")
            print(f"Available columns: {list(car_reviews_df.columns)}")
        
        print("\n" + "="*60)
        print("TIME SERIES ANALYSIS TEST COMPLETE!")
        print("="*60)
        
    except Exception as e:
        print(f"❌ Error during time series analysis test: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 