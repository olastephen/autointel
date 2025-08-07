"""
N-gram Analysis Module
Performs bigram and trigram frequency analysis on text data
"""

import matplotlib.pyplot as plt
import seaborn as sns
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from collections import Counter
from src.config.config import TOP_N_BIGRAMS, TOP_N_TRIGRAMS, COLOR_PALETTE


class NgramAnalyzer:
    """Handles bigram and trigram frequency analysis"""
    
    def __init__(self):
        """Initialize the n-gram analyzer"""
        pass
    
    def get_ngrams(self, text_series, n=2, top_n=20):
        """
        Extract n-grams from text series
        
        Args:
            text_series: Pandas series containing text data
            n: N-gram size (2 for bigrams, 3 for trigrams)
            top_n: Number of top n-grams to return
            
        Returns:
            list: List of tuples (ngram, count) sorted by frequency
        """
        all_ngrams = []
        
        try:
            for text in text_series.dropna():
                if str(text).strip():
                    tokens = word_tokenize(str(text).lower())
                    ngram_list = list(ngrams(tokens, n))
                    all_ngrams.extend(ngram_list)
        except (ImportError, LookupError):
            # Fallback to simple tokenization if NLTK is not available
            print("NLTK not available, using simple tokenization")
            for text in text_series.dropna():
                if str(text).strip():
                    # Simple word splitting
                    words = str(text).lower().split()
                    if len(words) >= n:
                        for i in range(len(words) - n + 1):
                            ngram = tuple(words[i:i+n])
                            all_ngrams.append(ngram)
        
        return Counter(all_ngrams).most_common(top_n)
    
    def analyze_ngrams(self, text_series, title="Text Analysis"):
        """
        Perform complete n-gram analysis with visualization
        
        Args:
            text_series: Pandas series containing text data
            title: Title for the analysis
        """
        print(f"\n=== {title} - Bigram/Trigram Analysis ===")
        
        # Get bigrams and trigrams
        bigrams = self.get_ngrams(text_series, n=2, top_n=TOP_N_BIGRAMS)
        trigrams = self.get_ngrams(text_series, n=3, top_n=TOP_N_TRIGRAMS)
        
        # Plot results
        self._plot_ngrams(bigrams, trigrams, title)
        
        # Print top results
        self._print_ngram_results(bigrams, trigrams, title)
    
    def _plot_ngrams(self, bigrams, trigrams, title):
        """
        Create visualization for n-grams
        
        Args:
            bigrams: List of bigram tuples
            trigrams: List of trigram tuples
            title: Title for the plots
        """
        if not bigrams and not trigrams:
            print("No n-grams found to plot")
            return
        
        # Determine subplot layout
        if bigrams and trigrams:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        else:
            fig, ax1 = plt.subplots(1, 1, figsize=(10, 6))
            ax2 = None
        
        # Plot bigrams
        if bigrams:
            bigram_labels = [' '.join(ng) for ng, _ in bigrams]
            bigram_counts = [count for _, count in bigrams]
            
            sns.barplot(x=bigram_counts, y=bigram_labels, ax=ax1, palette='viridis')
            ax1.set_title(f'{title} - Top Bigrams')
            ax1.set_xlabel('Frequency')
        
        # Plot trigrams
        if trigrams and ax2:
            trigram_labels = [' '.join(ng) for ng, _ in trigrams]
            trigram_counts = [count for _, count in trigrams]
            
            sns.barplot(x=trigram_counts, y=trigram_labels, ax=ax2, palette='plasma')
            ax2.set_title(f'{title} - Top Trigrams')
            ax2.set_xlabel('Frequency')
        
        plt.tight_layout()
        plt.show()
    
    def _print_ngram_results(self, bigrams, trigrams, title):
        """
        Print n-gram analysis results
        
        Args:
            bigrams: List of bigram tuples
            trigrams: List of trigram tuples
            title: Title for the analysis
        """
        if bigrams:
            print(f"\nTop 10 Bigrams in {title}:")
            for i, (ngram, count) in enumerate(bigrams[:10], 1):
                print(f"{i}. {' '.join(ngram)}: {count}")
        
        if trigrams:
            print(f"\nTop 5 Trigrams in {title}:")
            for i, (ngram, count) in enumerate(trigrams[:5], 1):
                print(f"{i}. {' '.join(ngram)}: {count}")
    
    def get_ngram_insights(self, text_series, title="Text"):
        """
        Extract insights from n-gram analysis
        
        Args:
            text_series: Pandas series containing text data
            title: Title for the analysis
            
        Returns:
            dict: Dictionary containing n-gram insights
        """
        bigrams = self.get_ngrams(text_series, n=2, top_n=TOP_N_BIGRAMS)
        trigrams = self.get_ngrams(text_series, n=3, top_n=TOP_N_TRIGRAMS)
        
        insights = {
            'title': title,
            'total_bigrams': len(bigrams),
            'total_trigrams': len(trigrams),
            'top_bigrams': bigrams[:5],
            'top_trigrams': trigrams[:3],
            'bigram_diversity': len(set([ng[0] for ng in bigrams])),
            'trigram_diversity': len(set([ng[0] for ng in trigrams]))
        }
        
        return insights
    
    def analyze_bigrams(self, text_series, title="Text"):
        """
        Analyze bigrams in text data
        
        Args:
            text_series: Pandas series containing text data
            title: Title for the analysis
            
        Returns:
            list: List of (bigram, frequency) tuples sorted by frequency
        """
        print(f"\n=== {title} - Bigram Analysis ===")
        
        bigrams = self.get_ngrams(text_series, n=2, top_n=TOP_N_BIGRAMS)
        
        if bigrams:
            print(f"Found {len(bigrams)} unique bigrams")
            print(f"Top 10 Bigrams:")
            for i, (bigram, count) in enumerate(bigrams[:10], 1):
                print(f"{i:2d}. {' '.join(bigram)}: {count}")
        else:
            print("No bigrams found")
        
        return bigrams
    
    def analyze_trigrams(self, text_series, title="Text"):
        """
        Analyze trigrams in text data
        
        Args:
            text_series: Pandas series containing text data
            title: Title for the analysis
            
        Returns:
            list: List of (trigram, frequency) tuples sorted by frequency
        """
        print(f"\n=== {title} - Trigram Analysis ===")
        
        trigrams = self.get_ngrams(text_series, n=3, top_n=TOP_N_TRIGRAMS)
        
        if trigrams:
            print(f"Found {len(trigrams)} unique trigrams")
            print(f"Top 10 Trigrams:")
            for i, (trigram, count) in enumerate(trigrams[:10], 1):
                print(f"{i:2d}. {' '.join(trigram)}: {count}")
        else:
            print("No trigrams found")
        
        return trigrams


def main():
    """Test n-gram analysis with actual datasets"""
    import pandas as pd
    
    print("="*60)
    print("N-GRAM ANALYSIS TEST - CAR NEWS & REVIEWS")
    print("="*60)
    
    try:
        # Load datasets
        print("\nLoading datasets...")
        car_news_df = pd.read_csv('car_news_dataset.csv')
        car_reviews_df = pd.read_csv('car_reviews_dataset.csv')
        
        print(f"✓ Car News Dataset: {car_news_df.shape}")
        print(f"✓ Car Reviews Dataset: {car_reviews_df.shape}")
        
        # Initialize n-gram analyzer
        ngram_analyzer = NgramAnalyzer()
        
        # Analyze Car News N-grams
        print("\n" + "="*40)
        print("ANALYZING CAR NEWS N-GRAMS")
        print("="*40)
        
        if 'content' in car_news_df.columns:
            news_text = car_news_df['content'].dropna()
            print(f"Processing {len(news_text)} news articles...")
            
            # Analyze bigrams
            news_bigrams = ngram_analyzer.analyze_bigrams(news_text, "Car News")
            print(f"\nTop 10 Bigrams in Car News:")
            for i, (bigram, freq) in enumerate(news_bigrams[:10], 1):
                print(f"{i:2d}. {' '.join(bigram)}: {freq}")
            
            # Analyze trigrams
            news_trigrams = ngram_analyzer.analyze_trigrams(news_text, "Car News")
            print(f"\nTop 10 Trigrams in Car News:")
            for i, (trigram, freq) in enumerate(news_trigrams[:10], 1):
                print(f"{i:2d}. {' '.join(trigram)}: {freq}")
        else:
            print("❌ 'content' column not found in news dataset")
        
        # Analyze Car Reviews N-grams
        print("\n" + "="*40)
        print("ANALYZING CAR REVIEWS N-GRAMS")
        print("="*40)
        
        if 'Review' in car_reviews_df.columns:
            review_text = car_reviews_df['Review'].dropna()
            print(f"Processing {len(review_text)} reviews...")
            
            # Analyze bigrams
            review_bigrams = ngram_analyzer.analyze_bigrams(review_text, "Car Reviews")
            print(f"\nTop 10 Bigrams in Car Reviews:")
            for i, (bigram, freq) in enumerate(review_bigrams[:10], 1):
                print(f"{i:2d}. {' '.join(bigram)}: {freq}")
            
            # Analyze trigrams
            review_trigrams = ngram_analyzer.analyze_trigrams(review_text, "Car Reviews")
            print(f"\nTop 10 Trigrams in Car Reviews:")
            for i, (trigram, freq) in enumerate(review_trigrams[:10], 1):
                print(f"{i:2d}. {' '.join(trigram)}: {freq}")
        else:
            print("❌ 'Review' column not found in reviews dataset")
        
        print("\n" + "="*60)
        print("N-GRAM ANALYSIS TEST COMPLETE!")
        print("="*60)
        
    except Exception as e:
        print(f"❌ Error during n-gram analysis test: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 