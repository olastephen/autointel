"""
Topic Modeling Module
Performs topic modeling using Latent Dirichlet Allocation (LDA)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from src.config.config import N_TOPICS, N_WORDS_PER_TOPIC, MAX_FEATURES, MAX_DF, MIN_DF, LDA_MAX_ITER, LDA_RANDOM_STATE


class TopicModeler:
    """Handles topic modeling using LDA"""
    
    def __init__(self, n_topics=N_TOPICS, n_words=N_WORDS_PER_TOPIC):
        """
        Initialize the topic modeler
        
        Args:
            n_topics: Number of topics to extract
            n_words: Number of top words per topic
        """
        self.n_topics = n_topics
        self.n_words = n_words
        self.vectorizer = None
        self.lda_model = None
        self.feature_names = None
    
    def fit_lda(self, text_series):
        """
        Fit LDA model to text data
        
        Args:
            text_series: Pandas series containing text data
            
        Returns:
            bool: True if successful, False otherwise
        """
        # Clean and prepare text
        clean_texts = text_series.dropna()
        clean_texts = clean_texts[clean_texts != ""]
        
        if len(clean_texts) == 0:
            print("No valid text data found for topic modeling")
            return False
        
        # Initialize vectorizer
        self.vectorizer = CountVectorizer(
            stop_words='english',
            max_df=MAX_DF,
            min_df=MIN_DF,
            max_features=MAX_FEATURES
        )
        
        try:
            # Create document-term matrix
            dtm = self.vectorizer.fit_transform(clean_texts)
            self.feature_names = self.vectorizer.get_feature_names_out()
            
            # Initialize and fit LDA model
            self.lda_model = LatentDirichletAllocation(
                n_components=self.n_topics,
                random_state=LDA_RANDOM_STATE,
                max_iter=LDA_MAX_ITER
            )
            self.lda_model.fit(dtm)
            
            return True
            
        except Exception as e:
            print(f"Error in LDA fitting: {e}")
            return False
    
    def get_topics(self):
        """
        Extract topics from fitted LDA model
        
        Returns:
            list: List of tuples (topic_id, top_words)
        """
        if self.lda_model is None or self.feature_names is None:
            return []
        
        topics = []
        for topic_idx, topic in enumerate(self.lda_model.components_):
            top_words_idx = topic.argsort()[-self.n_words:][::-1]
            top_words = [self.feature_names[i] for i in top_words_idx]
            topics.append((topic_idx, top_words))
        
        return topics
    
    def analyze_topics(self, text_series, title="Text"):
        """
        Perform complete topic modeling analysis
        
        Args:
            text_series: Pandas series containing text data
            title: Title for the analysis
        """
        print(f"\n=== {title} - Topic Modeling (LDA) ===")
        
        # Fit LDA model
        if not self.fit_lda(text_series):
            return None
        
        # Get topics
        topics = self.get_topics()
        
        # Print topics
        self._print_topics(topics, title)
        
        # Visualize topics
        self._plot_topics(topics, title)
        
        return topics
    
    def _print_topics(self, topics, title):
        """
        Print discovered topics
        
        Args:
            topics: List of topic tuples
            title: Title for the analysis
        """
        print(f"\nDiscovered Topics in {title}:")
        for idx, words in topics:
            print(f"Topic {idx+1}: {', '.join(words)}")
    
    def _plot_topics(self, topics, title):
        """
        Create topic visualization
        
        Args:
            topics: List of topic tuples
            title: Title for the plots
        """
        if not topics:
            return
        
        # Create subplots for each topic
        n_topics = len(topics)
        cols = min(3, n_topics)
        rows = (n_topics + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
        if rows == 1:
            axes = [axes] if cols == 1 else axes
        else:
            axes = axes.flatten()
        
        for i, (topic_idx, words) in enumerate(topics):
            if i < len(axes):
                # Get topic weights for top words
                topic_weights = self.lda_model.components_[topic_idx]
                top_indices = topic_weights.argsort()[-self.n_words:][::-1]
                top_weights = topic_weights[top_indices]
                top_words = [self.feature_names[idx] for idx in top_indices]
                
                # Create horizontal bar plot
                axes[i].barh(range(len(top_words)), top_weights)
                axes[i].set_yticks(range(len(top_words)))
                axes[i].set_yticklabels(top_words)
                axes[i].set_title(f'Topic {topic_idx+1}')
                axes[i].set_xlabel('Weight')
        
        # Hide empty subplots
        for i in range(len(topics), len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle(f'{title} - Topic Word Weights', fontsize=16)
        plt.tight_layout()
        plt.show()
    
    def get_topic_insights(self, text_series, title="Text"):
        """
        Extract insights from topic modeling
        
        Args:
            text_series: Pandas series containing text data
            title: Title for the analysis
            
        Returns:
            dict: Dictionary containing topic insights
        """
        if not self.fit_lda(text_series):
            return {}
        
        topics = self.get_topics()
        
        insights = {
            'title': title,
            'n_topics': self.n_topics,
            'n_words_per_topic': self.n_words,
            'topics': topics,
            'topic_coherence': self._calculate_topic_coherence(),
            'vocabulary_size': len(self.feature_names) if self.feature_names is not None else 0
        }
        
        return insights
    
    def _calculate_topic_coherence(self):
        """
        Calculate a simple topic coherence score
        
        Returns:
            float: Average topic coherence score
        """
        if self.lda_model is None:
            return 0.0
        
        # Simple coherence based on topic weight distribution
        coherence_scores = []
        for topic in self.lda_model.components_:
            # Calculate how concentrated the weights are
            sorted_weights = np.sort(topic)[::-1]
            top_weights = sorted_weights[:self.n_words]
            coherence = np.mean(top_weights) / np.std(top_weights) if np.std(top_weights) > 0 else 0
            coherence_scores.append(coherence)
        
        return np.mean(coherence_scores)
    
    def assign_topics_to_documents(self, text_series):
        """
        Assign dominant topics to documents
        
        Args:
            text_series: Pandas series containing text data
            
        Returns:
            pd.Series: Series with assigned topic labels
        """
        if self.lda_model is None or self.vectorizer is None:
            return pd.Series([None] * len(text_series))
        
        # Transform text to document-term matrix
        dtm = self.vectorizer.transform(text_series.dropna())
        
        # Get topic distributions for documents
        topic_distributions = self.lda_model.transform(dtm)
        
        # Assign dominant topic to each document
        dominant_topics = topic_distributions.argmax(axis=1)
        
        # Create series with topic assignments
        topic_assignments = pd.Series([None] * len(text_series))
        topic_assignments[text_series.dropna().index] = dominant_topics
        
        return topic_assignments


def main():
    """Test topic modeling with actual datasets"""
    import pandas as pd
    
    print("="*60)
    print("TOPIC MODELING TEST - CAR NEWS & REVIEWS")
    print("="*60)
    
    try:
        # Load datasets
        print("\nLoading datasets...")
        car_news_df = pd.read_csv('car_news_dataset.csv')
        car_reviews_df = pd.read_csv('car_reviews_dataset.csv')
        
        print(f"✓ Car News Dataset: {car_news_df.shape}")
        print(f"✓ Car Reviews Dataset: {car_reviews_df.shape}")
        
        # Initialize topic modeler
        topic_modeler = TopicModeler(n_topics=5, n_words=10)
        
        # Analyze Car News Topics
        print("\n" + "="*40)
        print("ANALYZING CAR NEWS TOPICS")
        print("="*40)
        
        if 'content' in car_news_df.columns:
            news_text = car_news_df['content'].dropna()
            print(f"Processing {len(news_text)} news articles...")
            
            # Run topic modeling
            news_topics = topic_modeler.analyze_topics(news_text, "Car News")
            
            # Get insights
            news_insights = topic_modeler.get_topic_insights(news_text, "Car News")
            print(f"\nNews Analysis Summary:")
            print(f"- Topics discovered: {news_insights.get('n_topics', 0)}")
            print(f"- Vocabulary size: {news_insights.get('vocabulary_size', 0)}")
            print(f"- Topic coherence: {news_insights.get('topic_coherence', 0):.3f}")
        else:
            print("❌ 'content' column not found in news dataset")
        
        # Analyze Car Reviews Topics
        print("\n" + "="*40)
        print("ANALYZING CAR REVIEWS TOPICS")
        print("="*40)
        
        if 'Review' in car_reviews_df.columns:
            review_text = car_reviews_df['Review'].dropna()
            print(f"Processing {len(review_text)} reviews...")
            
            # Run topic modeling
            review_topics = topic_modeler.analyze_topics(review_text, "Car Reviews")
            
            # Get insights
            review_insights = topic_modeler.get_topic_insights(review_text, "Car Reviews")
            print(f"\nReview Analysis Summary:")
            print(f"- Topics discovered: {review_insights.get('n_topics', 0)}")
            print(f"- Vocabulary size: {review_insights.get('vocabulary_size', 0)}")
            print(f"- Topic coherence: {review_insights.get('topic_coherence', 0):.3f}")
        else:
            print("❌ 'Review' column not found in reviews dataset")
        
        print("\n" + "="*60)
        print("TOPIC MODELING TEST COMPLETE!")
        print("="*60)
        
    except Exception as e:
        print(f"❌ Error during topic modeling test: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 