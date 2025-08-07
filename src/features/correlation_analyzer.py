"""
Correlation Analysis Module
Analyzes correlations between review scores, sentiment, and other numerical features
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.config.config import CORRELATION_THRESHOLD, HEATMAP_SIZE


class CorrelationAnalyzer:
    """Handles correlation analysis between numerical features"""
    
    def __init__(self):
        """Initialize the correlation analyzer"""
        pass
    
    def analyze_correlations(self, df, title="Data"):
        """
        Analyze correlations between numerical features
        
        Args:
            df: Pandas dataframe with numerical columns
            title: Title for the analysis
        """
        print(f"\n=== {title} - Correlation Analysis ===")
        
        # Get numerical columns and filter out problematic ones
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Filter out unnamed index columns and other problematic columns
        filtered_cols = []
        for col in numerical_cols:
            # Skip unnamed index columns
            if 'Unnamed' in col or col.endswith('.1'):
                continue
            # Skip columns that are likely just row indices
            if col in ['index', 'id', 'row_id']:
                continue
            filtered_cols.append(col)
        
        # Define sentiment columns to look for
        sentiment_cols = ['vader_compound', 'tb_polarity', 'tb_subjectivity']
        
        # Add sentiment columns if they exist
        available_cols = []
        for col in sentiment_cols:
            if col in df.columns:
                available_cols.append(col)
        
        # Combine numerical and sentiment columns
        analysis_cols = filtered_cols + available_cols
        
        if len(analysis_cols) < 2:
            print(f"Not enough numerical columns for correlation analysis. Found: {analysis_cols}")
            return {}
        
        # Create correlation matrix
        corr_matrix = df[analysis_cols].corr()
        
        # Visualize correlations
        self._plot_correlation_heatmap(corr_matrix, title)
        
        # Find significant correlations
        significant_correlations = self._find_significant_correlations(corr_matrix)
        
        # Print correlation insights
        self._print_correlation_insights(corr_matrix, significant_correlations, title)
        
        return {
            'correlation_matrix': corr_matrix,
            'significant_correlations': significant_correlations,
            'analysis_columns': analysis_cols
        }
    
    def _plot_correlation_heatmap(self, corr_matrix, title):
        """
        Create correlation heatmap visualization
        
        Args:
            corr_matrix: Correlation matrix
            title: Title for the plot
        """
        plt.figure(figsize=HEATMAP_SIZE)
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
        plt.title(f'{title} - Correlation Matrix')
        plt.tight_layout()
        plt.show()
    
    def _find_significant_correlations(self, corr_matrix):
        """
        Find correlations above the threshold
        
        Args:
            corr_matrix: Correlation matrix
            
        Returns:
            list: List of significant correlation tuples
        """
        significant_correlations = []
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > CORRELATION_THRESHOLD:
                    significant_correlations.append({
                        'var1': corr_matrix.columns[i],
                        'var2': corr_matrix.columns[j],
                        'correlation': corr_val,
                        'strength': self._get_correlation_strength(corr_val)
                    })
        
        # Sort by absolute correlation value
        significant_correlations.sort(key=lambda x: abs(x['correlation']), reverse=True)
        
        return significant_correlations
    
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
    
    def _print_correlation_insights(self, corr_matrix, significant_correlations, title):
        """
        Print correlation analysis insights
        
        Args:
            corr_matrix: Correlation matrix
            significant_correlations: List of significant correlations
            title: Title for the analysis
        """
        print(f"\nSignificant Correlations in {title} (|r| > {CORRELATION_THRESHOLD}):")
        
        if significant_correlations:
            for corr in significant_correlations:
                direction = "positive" if corr['correlation'] > 0 else "negative"
                print(f"- {corr['var1']} ↔ {corr['var2']}: {corr['correlation']:.3f} ({corr['strength']} {direction})")
        else:
            print("No significant correlations found")
        
        # Additional insights
        print(f"\nCorrelation Matrix Shape: {corr_matrix.shape}")
        print(f"Variables Analyzed: {', '.join(corr_matrix.columns)}")
    
    def analyze_sentiment_correlations(self, df, title="Data"):
        """
        Specifically analyze sentiment correlations
        
        Args:
            df: Pandas dataframe with sentiment columns
            title: Title for the analysis
        """
        print(f"\n=== {title} - Sentiment Correlation Analysis ===")
        
        sentiment_cols = ['vader_compound', 'tb_polarity', 'tb_subjectivity']
        available_sentiment = [col for col in sentiment_cols if col in df.columns]
        
        if len(available_sentiment) < 2:
            print("Not enough sentiment columns for analysis")
            return
        
        # Get numerical columns excluding sentiment
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numerical_cols = [col for col in numerical_cols if col not in sentiment_cols]
        
        if not numerical_cols:
            print("No numerical columns found for sentiment correlation analysis")
            return
        
        # Analyze correlations between sentiment and numerical features
        sentiment_correlations = {}
        
        for sentiment_col in available_sentiment:
            correlations = {}
            for num_col in numerical_cols:
                corr_val = df[sentiment_col].corr(df[num_col])
                if not pd.isna(corr_val):
                    correlations[num_col] = corr_val
            
            sentiment_correlations[sentiment_col] = correlations
        
        # Visualize sentiment correlations
        self._plot_sentiment_correlations(sentiment_correlations, title)
        
        # Print sentiment correlation insights
        self._print_sentiment_correlations(sentiment_correlations, title)
        
        return sentiment_correlations
    
    def _plot_sentiment_correlations(self, sentiment_correlations, title):
        """
        Create visualization for sentiment correlations
        
        Args:
            sentiment_correlations: Dictionary of sentiment correlations
            title: Title for the plots
        """
        if not sentiment_correlations:
            return
        
        n_sentiment = len(sentiment_correlations)
        fig, axes = plt.subplots(1, n_sentiment, figsize=(5*n_sentiment, 6))
        
        if n_sentiment == 1:
            axes = [axes]
        
        for i, (sentiment_col, correlations) in enumerate(sentiment_correlations.items()):
            if correlations:
                variables = list(correlations.keys())
                corr_values = list(correlations.values())
                
                # Create bar plot
                colors = ['red' if x < 0 else 'green' for x in corr_values]
                axes[i].barh(variables, corr_values, color=colors, alpha=0.7)
                axes[i].axvline(x=0, color='black', linestyle='-', alpha=0.3)
                axes[i].set_title(f'{sentiment_col.replace("_", " ").title()}')
                axes[i].set_xlabel('Correlation Coefficient')
                axes[i].set_xlim(-1, 1)
        
        plt.suptitle(f'{title} - Sentiment Correlations', fontsize=14)
        plt.tight_layout()
        plt.show()
    
    def _print_sentiment_correlations(self, sentiment_correlations, title):
        """
        Print sentiment correlation insights
        
        Args:
            sentiment_correlations: Dictionary of sentiment correlations
            title: Title for the analysis
        """
        print(f"\nSentiment Correlation Insights for {title}:")
        
        for sentiment_col, correlations in sentiment_correlations.items():
            if correlations:
                print(f"\n{sentiment_col.replace('_', ' ').title()} Correlations:")
                
                # Sort by absolute correlation value
                sorted_correlations = sorted(correlations.items(), 
                                           key=lambda x: abs(x[1]), reverse=True)
                
                for variable, corr_val in sorted_correlations[:5]:  # Top 5
                    direction = "positive" if corr_val > 0 else "negative"
                    strength = self._get_correlation_strength(corr_val)
                    print(f"- {variable}: {corr_val:.3f} ({strength} {direction})")
    
    def get_correlation_insights(self, df, title="Data"):
        """
        Extract comprehensive correlation insights
        
        Args:
            df: Pandas dataframe
            title: Title for the analysis
            
        Returns:
            dict: Dictionary containing correlation insights
        """
        # General correlations
        general_results = self.analyze_correlations(df, title)
        
        # Sentiment correlations
        sentiment_results = self.analyze_sentiment_correlations(df, title)
        
        # Calculate correlation statistics
        significant_correlations = general_results.get('significant_correlations', [])
        
        strong_correlations = len([c for c in significant_correlations if abs(c['correlation']) > 0.7])
        moderate_correlations = len([c for c in significant_correlations if 0.3 < abs(c['correlation']) <= 0.7])
        weak_correlations = len([c for c in significant_correlations if abs(c['correlation']) <= 0.3])
        
        # Create top correlations list
        top_correlations = []
        for corr in significant_correlations:
            top_correlations.append((corr['var1'], corr['var2'], corr['correlation']))
        
        insights = {
            'title': title,
            'general_correlations': general_results,
            'sentiment_correlations': sentiment_results,
            'total_correlations': len(significant_correlations),
            'strong_correlations': strong_correlations,
            'moderate_correlations': moderate_correlations,
            'weak_correlations': weak_correlations,
            'top_correlations': top_correlations,
            'total_variables': len(general_results.get('analysis_columns', [])),
            'significant_correlations_count': len(significant_correlations)
        }
        
        return insights


def main():
    """Test correlation analysis with actual datasets"""
    import pandas as pd
    
    print("="*60)
    print("CORRELATION ANALYSIS TEST - CAR REVIEWS")
    print("="*60)
    
    try:
        # Load datasets
        print("\nLoading datasets...")
        car_reviews_df = pd.read_csv('car_reviews_dataset.csv')
        
        print(f"✓ Car Reviews Dataset: {car_reviews_df.shape}")
        
        # Initialize correlation analyzer
        correlation_analyzer = CorrelationAnalyzer()
        
        # Analyze correlations in car reviews
        print("\n" + "="*40)
        print("ANALYZING CORRELATIONS IN CAR REVIEWS")
        print("="*40)
        
        # Check for numerical columns
        numerical_cols = car_reviews_df.select_dtypes(include=['number']).columns.tolist()
        print(f"Numerical columns found: {numerical_cols}")
        
        if len(numerical_cols) >= 2:
            # Analyze correlations
            correlations = correlation_analyzer.analyze_correlations(car_reviews_df, "Car Reviews")
            
            # Get insights
            insights = correlation_analyzer.get_correlation_insights(car_reviews_df, "Car Reviews")
            print(f"\nCorrelation Analysis Summary:")
            print(f"- Total correlations analyzed: {insights.get('total_correlations', 0)}")
            print(f"- Strong correlations (|r| > 0.7): {insights.get('strong_correlations', 0)}")
            print(f"- Moderate correlations (0.3 < |r| ≤ 0.7): {insights.get('moderate_correlations', 0)}")
            print(f"- Weak correlations (|r| ≤ 0.3): {insights.get('weak_correlations', 0)}")
            
            # Show top correlations
            top_correlations = insights.get('top_correlations', [])
            if top_correlations:
                print(f"\nTop 10 Strongest Correlations:")
                for i, (var1, var2, corr) in enumerate(top_correlations[:10], 1):
                    print(f"{i:2d}. {var1} ↔ {var2}: {corr:.3f}")
        else:
            print("❌ Need at least 2 numerical columns for correlation analysis")
            print(f"Available columns: {list(car_reviews_df.columns)}")
        
        print("\n" + "="*60)
        print("CORRELATION ANALYSIS TEST COMPLETE!")
        print("="*60)
        
    except Exception as e:
        print(f"❌ Error during correlation analysis test: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 