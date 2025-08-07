"""
Keyword Analysis Module
Analyzes keyword correlations and co-occurrence patterns
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from src.config.config import ANALYSIS_KEYWORDS, KEYWORD_CORRELATION_THRESHOLD


class KeywordAnalyzer:
    """Handles keyword correlation and co-occurrence analysis"""
    
    def __init__(self, keywords=None):
        """
        Initialize the keyword analyzer
        
        Args:
            keywords: List of keywords to analyze (defaults to config keywords)
        """
        self.keywords = keywords if keywords else ANALYSIS_KEYWORDS
    
    def analyze_keyword_correlations(self, text_series, title="Text"):
        """
        Analyze keyword correlations and co-occurrence patterns
        
        Args:
            text_series: Pandas series containing text data
            title: Title for the analysis
        """
        print(f"\n=== {title} - Keyword Correlation Analysis ===")
        
        # Create keyword presence matrix
        keyword_matrix = self._create_keyword_matrix(text_series)
        
        if keyword_matrix.empty:
            print("No keywords found in the text data")
            return {}
        
        # Calculate correlation matrix
        corr_matrix = keyword_matrix.corr()
        
        # Visualize correlations
        self._plot_keyword_correlations(corr_matrix, title)
        
        # Find strong correlations
        strong_correlations = self._find_strong_correlations(corr_matrix)
        
        # Print correlation insights
        self._print_keyword_correlations(strong_correlations, title)
        
        return {
            'correlation_matrix': corr_matrix,
            'strong_correlations': strong_correlations,
            'keyword_matrix': keyword_matrix
        }
    
    def _create_keyword_matrix(self, text_series):
        """
        Create a matrix showing keyword presence in each document
        
        Args:
            text_series: Pandas series containing text data
            
        Returns:
            pd.DataFrame: Matrix with keyword presence (1/0)
        """
        keyword_matrix = []
        
        for text in text_series.dropna():
            text_lower = str(text).lower()
            presence = [1 if keyword in text_lower else 0 for keyword in self.keywords]
            keyword_matrix.append(presence)
        
        if not keyword_matrix:
            return pd.DataFrame()
        
        return pd.DataFrame(keyword_matrix, columns=self.keywords)
    
    def _plot_keyword_correlations(self, corr_matrix, title):
        """
        Create visualization for keyword correlations
        
        Args:
            corr_matrix: Correlation matrix
            title: Title for the plots
        """
        plt.figure(figsize=(12, 10))
        
        # Create mask for upper triangle
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        sns.heatmap(
            corr_matrix, 
            annot=True, 
            cmap='coolwarm', 
            center=0, 
            square=True, 
            fmt='.2f',
            mask=mask,
            cbar_kws={"shrink": .8}
        )
        plt.title(f'{title} - Keyword Correlation Matrix')
        plt.tight_layout()
        plt.show()
    
    def _find_strong_correlations(self, corr_matrix):
        """
        Find correlations above the threshold
        
        Args:
            corr_matrix: Correlation matrix
            
        Returns:
            list: List of strong correlation dictionaries
        """
        strong_correlations = []
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > KEYWORD_CORRELATION_THRESHOLD:
                    strong_correlations.append({
                        'keyword1': corr_matrix.columns[i],
                        'keyword2': corr_matrix.columns[j],
                        'correlation': corr_val,
                        'strength': self._get_correlation_strength(corr_val)
                    })
        
        # Sort by absolute correlation value
        strong_correlations.sort(key=lambda x: abs(x['correlation']), reverse=True)
        
        return strong_correlations
    
    def _get_correlation_strength(self, corr_val):
        """
        Get correlation strength description
        
        Args:
            corr_val: Correlation value
            
        Returns:
            str: Strength description
        """
        abs_val = abs(corr_val)
        if abs_val >= 0.7:
            return "Very Strong"
        elif abs_val >= 0.5:
            return "Strong"
        elif abs_val >= 0.3:
            return "Moderate"
        else:
            return "Weak"
    
    def _print_keyword_correlations(self, strong_correlations, title):
        """
        Print keyword correlation insights
        
        Args:
            strong_correlations: List of strong correlations
            title: Title for the analysis
        """
        print(f"\nStrong Keyword Correlations in {title} (|r| > {KEYWORD_CORRELATION_THRESHOLD}):")
        
        if strong_correlations:
            for corr in strong_correlations:
                direction = "positive" if corr['correlation'] > 0 else "negative"
                print(f"- {corr['keyword1']} ↔ {corr['keyword2']}: {corr['correlation']:.3f} ({corr['strength']} {direction})")
        else:
            print("No strong keyword correlations found")
    
    def analyze_keyword_frequency(self, text_series, title="Text"):
        """
        Analyze keyword frequency patterns
        
        Args:
            text_series: Pandas series containing text data
            title: Title for the analysis
        """
        print(f"\n=== {title} - Keyword Frequency Analysis ===")
        
        # Count keyword occurrences
        keyword_counts = Counter()
        total_documents = 0
        
        for text in text_series.dropna():
            if str(text).strip():
                total_documents += 1
                text_lower = str(text).lower()
                for keyword in self.keywords:
                    if keyword in text_lower:
                        keyword_counts[keyword] += 1
        
        if not keyword_counts:
            print("No keywords found in the text data")
            return {}
        
        # Calculate frequencies
        keyword_frequencies = {}
        for keyword, count in keyword_counts.items():
            frequency = count / total_documents
            keyword_frequencies[keyword] = {
                'count': count,
                'frequency': frequency,
                'percentage': frequency * 100
            }
        
        # Visualize frequencies
        self._plot_keyword_frequencies(keyword_frequencies, title)
        
        # Print frequency insights
        self._print_keyword_frequencies(keyword_frequencies, title)
        
        return keyword_frequencies
    
    def analyze_keywords(self, text_series, title="Text"):
        """
        Extract and analyze keywords from text data
        
        Args:
            text_series: Pandas series containing text data
            title: Title for the analysis
            
        Returns:
            list: List of (keyword, frequency) tuples sorted by frequency
        """
        print(f"\n=== {title} - Keyword Analysis ===")
        
        # Extract all words from text
        all_words = []
        for text in text_series.dropna():
            if str(text).strip():
                # Simple word extraction (can be enhanced with NLP)
                words = str(text).lower().split()
                # Filter out common stop words and short words
                filtered_words = [word for word in words if len(word) > 3 and word.isalpha()]
                all_words.extend(filtered_words)
        
        if not all_words:
            print("No words found in the text data")
            return []
        
        # Count word frequencies
        word_counts = Counter(all_words)
        
        # Get top keywords (excluding very common words)
        common_stop_words = {
            'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how', 'man', 'new', 'now', 'old', 'see', 'two', 'way', 'who', 'boy', 'did', 'its', 'let', 'put', 'say', 'she', 'too', 'use', 'that', 'this', 'with', 'have', 'will', 'your', 'from', 'they', 'know', 'want', 'been', 'good', 'much', 'some', 'time', 'very', 'when', 'come', 'just', 'like', 'long', 'make', 'many', 'over', 'such', 'take', 'than', 'them', 'well', 'were', 'what', 'work', 'year', 'also', 'around', 'another', 'could', 'first', 'great', 'should', 'through', 'where', 'world', 'after', 'before', 'between', 'during', 'little', 'might', 'never', 'often', 'other', 'people', 'right', 'still', 'think', 'under', 'while', 'would', 'always', 'because', 'before', 'better', 'during', 'enough', 'everything', 'family', 'father', 'friend', 'however', 'important', 'interest', 'mother', 'nothing', 'perhaps', 'picture', 'really', 'something', 'sometimes', 'together', 'without'
        }
        
        # Filter out stop words and get top keywords
        keywords = [(word, count) for word, count in word_counts.items() 
                   if word not in common_stop_words and count > 1]
        
        # Sort by frequency (descending)
        keywords.sort(key=lambda x: x[1], reverse=True)
        
        print(f"Found {len(keywords)} unique keywords")
        print(f"Total word occurrences: {sum(word_counts.values())}")
        
        return keywords
    
    def _plot_keyword_frequencies(self, keyword_frequencies, title):
        """
        Create visualization for keyword frequencies
        
        Args:
            keyword_frequencies: Dictionary of keyword frequency data
            title: Title for the plots
        """
        if not keyword_frequencies:
            return
        
        # Sort by frequency percentage
        sorted_keywords = sorted(keyword_frequencies.items(), 
                               key=lambda x: x[1]['frequency_pct'], reverse=True)
        
        keywords, freq_data = zip(*sorted_keywords)
        frequencies = [data['frequency_pct'] for data in freq_data]
        counts = [data['count'] for data in freq_data]
        
        # Create subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Frequency percentage plot
        ax1.barh(range(len(keywords)), frequencies, color='skyblue')
        ax1.set_yticks(range(len(keywords)))
        ax1.set_yticklabels(keywords)
        ax1.set_title(f'{title} - Keyword Frequency (%)')
        ax1.set_xlabel('Frequency (%)')
        
        # Count plot
        ax2.barh(range(len(keywords)), counts, color='lightgreen')
        ax2.set_yticks(range(len(keywords)))
        ax2.set_yticklabels(keywords)
        ax2.set_title(f'{title} - Keyword Count')
        ax2.set_xlabel('Count')
        
        plt.tight_layout()
        plt.show()
    
    def _print_keyword_frequencies(self, keyword_frequencies, title):
        """
        Print keyword frequency insights
        
        Args:
            keyword_frequencies: Dictionary of keyword frequency data
            title: Title for the analysis
        """
        if not keyword_frequencies:
            return
        
        # Sort by frequency percentage
        sorted_keywords = sorted(keyword_frequencies.items(), 
                               key=lambda x: x[1]['frequency_pct'], reverse=True)
        
        print(f"\nTop Keywords by Frequency in {title}:")
        for i, (keyword, data) in enumerate(sorted_keywords[:10], 1):
            print(f"{i}. {keyword}: {data['count']} occurrences ({data['frequency_pct']:.1f}%)")
    
    def analyze_keyword_co_occurrence(self, text_series, title="Text"):
        """
        Analyze keyword co-occurrence patterns
        
        Args:
            text_series: Pandas series containing text data
            title: Title for the analysis
        """
        print(f"\n=== {title} - Keyword Co-occurrence Analysis ===")
        
        co_occurrence_matrix = {}
        
        # Initialize co-occurrence matrix
        for keyword1 in self.keywords:
            co_occurrence_matrix[keyword1] = {}
            for keyword2 in self.keywords:
                co_occurrence_matrix[keyword1][keyword2] = 0
        
        # Count co-occurrences
        total_documents = 0
        for text in text_series.dropna():
            if str(text).strip():
                total_documents += 1
                text_lower = str(text).lower()
                
                # Find keywords present in this document
                present_keywords = [kw for kw in self.keywords if kw in text_lower]
                
                # Count co-occurrences
                for i, kw1 in enumerate(present_keywords):
                    for kw2 in present_keywords[i:]:  # Include self-co-occurrence
                        co_occurrence_matrix[kw1][kw2] += 1
                        if kw1 != kw2:  # Avoid double counting
                            co_occurrence_matrix[kw2][kw1] += 1
        
        # Convert to DataFrame
        co_occurrence_df = pd.DataFrame(co_occurrence_matrix)
        
        # Visualize co-occurrence matrix
        self._plot_co_occurrence_matrix(co_occurrence_df, title)
        
        # Find top co-occurring pairs
        top_co_occurrences = self._find_top_co_occurrences(co_occurrence_df)
        
        # Print co-occurrence insights
        self._print_co_occurrence_insights(top_co_occurrences, title)
        
        return {
            'co_occurrence_matrix': co_occurrence_df,
            'top_co_occurrences': top_co_occurrences,
            'total_documents': total_documents
        }
    
    def _plot_co_occurrence_matrix(self, co_occurrence_df, title):
        """
        Create visualization for co-occurrence matrix
        
        Args:
            co_occurrence_df: Co-occurrence matrix
            title: Title for the plot
        """
        plt.figure(figsize=(12, 10))
        
        sns.heatmap(
            co_occurrence_df, 
            annot=True, 
            cmap='YlOrRd', 
            fmt='d',
            cbar_kws={"shrink": .8}
        )
        plt.title(f'{title} - Keyword Co-occurrence Matrix')
        plt.tight_layout()
        plt.show()
    
    def _find_top_co_occurrences(self, co_occurrence_df):
        """
        Find top co-occurring keyword pairs
        
        Args:
            co_occurrence_df: Co-occurrence matrix
            
        Returns:
            list: List of top co-occurrence tuples
        """
        co_occurrences = []
        
        for i in range(len(co_occurrence_df.columns)):
            for j in range(i+1, len(co_occurrence_df.columns)):
                kw1 = co_occurrence_df.columns[i]
                kw2 = co_occurrence_df.columns[j]
                count = co_occurrence_df.iloc[i, j]
                co_occurrences.append((kw1, kw2, count))
        
        # Sort by count
        co_occurrences.sort(key=lambda x: x[2], reverse=True)
        
        return co_occurrences[:10]  # Top 10
    
    def _print_co_occurrence_insights(self, top_co_occurrences, title):
        """
        Print co-occurrence insights
        
        Args:
            top_co_occurrences: List of top co-occurrence tuples
            title: Title for the analysis
        """
        print(f"\nTop Keyword Co-occurrences in {title}:")
        
        for i, (kw1, kw2, count) in enumerate(top_co_occurrences, 1):
            print(f"{i}. {kw1} + {kw2}: {count} co-occurrences")
    
    def get_keyword_insights(self, text_series, title="Text"):
        """
        Extract comprehensive keyword insights
        
        Args:
            text_series: Pandas series containing text data
            title: Title for the analysis
            
        Returns:
            dict: Dictionary containing keyword insights
        """
        # Extract keywords using the new method
        keywords = self.analyze_keywords(text_series, title)
        
        # Calculate insights
        total_keywords = sum(count for _, count in keywords)
        unique_keywords = len(keywords)
        avg_keywords_per_text = total_keywords / len(text_series.dropna()) if len(text_series.dropna()) > 0 else 0
        
        # Get top keywords
        top_keywords = keywords[:50]  # Top 50 keywords
        
        insights = {
            'title': title,
            'total_keywords': total_keywords,
            'unique_keywords': unique_keywords,
            'avg_keywords_per_text': avg_keywords_per_text,
            'top_keywords': top_keywords,
            'keyword_distribution': dict(keywords[:20])  # Top 20 for distribution
        }
        
        return insights


def main():
    """Test keyword analysis with actual datasets"""
    import pandas as pd
    
    print("="*60)
    print("KEYWORD ANALYSIS TEST - CAR NEWS & REVIEWS")
    print("="*60)
    
    try:
        # Load datasets
        print("\nLoading datasets...")
        car_news_df = pd.read_csv('car_news_dataset.csv')
        car_reviews_df = pd.read_csv('car_reviews_dataset.csv')
        
        print(f"✓ Car News Dataset: {car_news_df.shape}")
        print(f"✓ Car Reviews Dataset: {car_reviews_df.shape}")
        
        # Initialize keyword analyzer
        keyword_analyzer = KeywordAnalyzer()
        
        # Analyze Car News Keywords
        print("\n" + "="*40)
        print("ANALYZING CAR NEWS KEYWORDS")
        print("="*40)
        
        if 'content' in car_news_df.columns:
            news_text = car_news_df['content'].dropna()
            print(f"Processing {len(news_text)} news articles...")
            
            # Analyze keywords
            news_keywords = keyword_analyzer.analyze_keywords(news_text, "Car News")
            
            # Get insights
            news_insights = keyword_analyzer.get_keyword_insights(news_text, "Car News")
            print(f"\nNews Keyword Summary:")
            print(f"- Total keywords found: {news_insights.get('total_keywords', 0)}")
            print(f"- Unique keywords: {news_insights.get('unique_keywords', 0)}")
            print(f"- Average keywords per text: {news_insights.get('avg_keywords_per_text', 0):.2f}")
            
            # Show top keywords
            top_keywords = news_insights.get('top_keywords', [])
            if top_keywords:
                print(f"\nTop 15 Keywords in Car News:")
                for i, (keyword, freq) in enumerate(top_keywords[:15], 1):
                    print(f"{i:2d}. {keyword}: {freq}")
        else:
            print("❌ 'content' column not found in news dataset")
        
        # Analyze Car Reviews Keywords
        print("\n" + "="*40)
        print("ANALYZING CAR REVIEWS KEYWORDS")
        print("="*40)
        
        if 'Review' in car_reviews_df.columns:
            review_text = car_reviews_df['Review'].dropna()
            print(f"Processing {len(review_text)} reviews...")
            
            # Analyze keywords
            review_keywords = keyword_analyzer.analyze_keywords(review_text, "Car Reviews")
            
            # Get insights
            review_insights = keyword_analyzer.get_keyword_insights(review_text, "Car Reviews")
            print(f"\nReview Keyword Summary:")
            print(f"- Total keywords found: {review_insights.get('total_keywords', 0)}")
            print(f"- Unique keywords: {review_insights.get('unique_keywords', 0)}")
            print(f"- Average keywords per review: {review_insights.get('avg_keywords_per_text', 0):.2f}")
            
            # Show top keywords
            top_keywords = review_insights.get('top_keywords', [])
            if top_keywords:
                print(f"\nTop 15 Keywords in Car Reviews:")
                for i, (keyword, freq) in enumerate(top_keywords[:15], 1):
                    print(f"{i:2d}. {keyword}: {freq}")
        else:
            print("❌ 'Review' column not found in reviews dataset")
        
        print("\n" + "="*60)
        print("KEYWORD ANALYSIS TEST COMPLETE!")
        print("="*60)
        
    except Exception as e:
        print(f"❌ Error during keyword analysis test: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 