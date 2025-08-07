"""
Named Entity Recognition Module
Performs NER analysis to identify car models, brands, locations, and organizations
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from collections import Counter
from src.config.config import CAR_BRANDS, NER_SAMPLE_SIZE


class NERAnalyzer:
    """Handles Named Entity Recognition analysis"""
    
    def __init__(self):
        """Initialize the NER analyzer"""
        self.nlp = None
        self._load_spacy_model()
    
    def _load_spacy_model(self):
        """Load spaCy model for NER (optional)"""
        try:
            import spacy
            self.nlp = spacy.load("en_core_web_sm")
            print("spaCy model loaded successfully")
        except (OSError, ImportError, Exception) as e:
            print(f"spaCy model not available: {e}")
            print("Using alternative NER methods (regex and keyword matching)")
            self.nlp = None
    
    def extract_entities_simple(self, text_series, title="Text"):
        """
        Extract named entities using simple regex and keyword matching
        
        Args:
            text_series: Pandas series containing text data
            title: Title for the analysis
        """
        print(f"\n=== {title} - Named Entity Recognition (Simple Method) ===")
        
        # Initialize entity counters
        organizations = []
        locations = []
        car_models = []
        car_brands_found = []
        
        # Common location patterns
        location_patterns = [
            r'\b[A-Z][a-z]+(?:[\s,]+[A-Z][a-z]+)*\s+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Drive|Dr|Lane|Ln|Place|Pl|Court|Ct|Way|Highway|Hwy)\b',
            r'\b[A-Z][a-z]+(?:[\s,]+[A-Z][a-z]+)*\s+(?:City|Town|Village|County|State|Province|Country)\b',
            r'\b[A-Z]{2}\b',  # State abbreviations
            r'\b[A-Z][a-z]+(?:[\s,]+[A-Z][a-z]+)*\b'  # Capitalized words (potential locations)
        ]
        
        # Car model patterns
        model_patterns = [
            r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:Sedan|SUV|Truck|Van|Coupe|Convertible|Hatchback|Wagon|Minivan)\b',
            r'\b[A-Z][a-z]+\s+\d{3,4}[A-Z]?\b',  # e.g., BMW 330i, Audi A4
            r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b'  # Two word models
        ]
        
        # Process a sample of texts for efficiency
        sample_texts = text_series.dropna().head(NER_SAMPLE_SIZE)
        
        print(f"Processing {len(sample_texts)} documents for NER analysis...")
        
        for text in sample_texts:
            if pd.isna(text) or str(text).strip() == "":
                continue
            
            text_str = str(text)
            text_lower = text_str.lower()
            
            # Look for car brands
            for brand in CAR_BRANDS:
                if brand in text_lower:
                    car_brands_found.append(brand)
            
            # Look for locations using regex
            for pattern in location_patterns:
                matches = re.findall(pattern, text_str)
                locations.extend([loc.strip() for loc in matches if len(loc.strip()) > 2])
            
            # Look for car models using regex
            for pattern in model_patterns:
                matches = re.findall(pattern, text_str)
                car_models.extend([model.strip() for model in matches if len(model.strip()) > 2])
            
            # Look for organizations (companies, dealerships, etc.)
            org_keywords = ['dealership', 'company', 'corporation', 'inc', 'llc', 'ltd', 'motors', 'automotive']
            for keyword in org_keywords:
                if keyword in text_lower:
                    # Extract potential organization names around the keyword
                    words = text_str.split()
                    for i, word in enumerate(words):
                        if keyword in word.lower():
                            # Get surrounding words as potential org name
                            start = max(0, i-2)
                            end = min(len(words), i+3)
                            org_name = ' '.join(words[start:end])
                            if len(org_name) > 3:
                                organizations.append(org_name)
        
        # Count entities
        entity_counts = {
            'organizations': Counter(organizations),
            'locations': Counter(locations),
            'car_models': Counter(car_models),
            'car_brands': Counter(car_brands_found)
        }
        
        # Display results
        self._display_entity_results(entity_counts, title)
        
        # Create visualizations
        self._plot_entities(entity_counts, title)
        
        return entity_counts
    
    def extract_entities(self, text_series, title="Text"):
        """
        Extract named entities from text data (uses spaCy if available, otherwise simple method)
        
        Args:
            text_series: Pandas series containing text data
            title: Title for the analysis
        """
        if self.nlp is not None:
            return self._extract_entities_spacy(text_series, title)
        else:
            return self.extract_entities_simple(text_series, title)
    
    def _extract_entities_spacy(self, text_series, title="Text"):
        """
        Extract named entities using spaCy (if available)
        
        Args:
            text_series: Pandas series containing text data
            title: Title for the analysis
        """
        print(f"\n=== {title} - Named Entity Recognition (spaCy) ===")
        
        # Initialize entity counters
        organizations = []
        locations = []
        car_models = []
        car_brands_found = []
        
        # Process a sample of texts for efficiency
        sample_texts = text_series.dropna().head(NER_SAMPLE_SIZE)
        
        print(f"Processing {len(sample_texts)} documents for NER analysis...")
        
        for text in sample_texts:
            if pd.isna(text) or str(text).strip() == "":
                continue
                
            doc = self.nlp(str(text))
            
            for ent in doc.ents:
                if ent.label_ == 'ORG':
                    organizations.append(ent.text.lower())
                elif ent.label_ == 'GPE':
                    locations.append(ent.text.lower())
                elif ent.label_ == 'PRODUCT':
                    car_models.append(ent.text.lower())
            
            # Also look for car brands in the text
            text_lower = str(text).lower()
            for brand in CAR_BRANDS:
                if brand in text_lower:
                    organizations.append(brand)
        
        # Count entities
        entity_counts = {
            'organizations': Counter(organizations),
            'locations': Counter(locations),
            'car_models': Counter(car_models),
            'car_brands': Counter(car_brands_found)
        }
        
        # Display results
        self._display_entity_results(entity_counts, title)
        
        # Create visualizations
        self._plot_entities(entity_counts, title)
        
        return entity_counts
    
    def _display_entity_results(self, entity_counts, title):
        """
        Display NER analysis results
        
        Args:
            entity_counts: Dictionary of entity counters
            title: Title for the analysis
        """
        if entity_counts['organizations']:
            print(f"\nTop Organizations/Brands in {title}:")
            for org, count in entity_counts['organizations'].most_common(10):
                print(f"- {org}: {count}")
        
        if entity_counts['locations']:
            print(f"\nTop Locations in {title}:")
            for loc, count in entity_counts['locations'].most_common(10):
                print(f"- {loc}: {count}")
        
        if entity_counts['car_models']:
            print(f"\nTop Car Models in {title}:")
            for model, count in entity_counts['car_models'].most_common(10):
                print(f"- {model}: {count}")
        
        if entity_counts['car_brands']:
            print(f"\nTop Car Brands in {title}:")
            for brand, count in entity_counts['car_brands'].most_common(10):
                print(f"- {brand}: {count}")
    
    def _plot_entities(self, entity_counts, title):
        """
        Create visualizations for entity analysis
        
        Args:
            entity_counts: Dictionary of entity counters
            title: Title for the plots
        """
        # Create subplots for different entity types
        entity_types = ['organizations', 'locations', 'car_models', 'car_brands']
        available_types = [et for et in entity_types if entity_counts[et]]
        
        if not available_types:
            print("No entities found to plot")
            return
        
        n_types = len(available_types)
        cols = min(2, n_types)
        rows = (n_types + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
        if rows == 1:
            axes = [axes] if cols == 1 else axes
        else:
            axes = axes.flatten()
        
        for i, entity_type in enumerate(available_types):
            if i < len(axes):
                entity_data = entity_counts[entity_type]
                if entity_data:
                    # Get top entities
                    top_entities = entity_data.most_common(10)
                    entities, counts = zip(*top_entities)
                    
                    # Create bar plot
                    axes[i].barh(range(len(entities)), counts)
                    axes[i].set_yticks(range(len(entities)))
                    axes[i].set_yticklabels(entities)
                    axes[i].set_title(f'{title} - Top {entity_type.replace("_", " ").title()}')
                    axes[i].set_xlabel('Count')
        
        # Hide empty subplots
        for i in range(len(available_types), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.show()
    
    def get_ner_insights(self, text_series, title="Text"):
        """
        Extract insights from NER analysis
        
        Args:
            text_series: Pandas series containing text data
            title: Title for the analysis
            
        Returns:
            dict: Dictionary containing NER insights
        """
        entity_counts = self.extract_entities(text_series, title)
        
        insights = {
            'title': title,
            'total_documents_analyzed': min(len(text_series.dropna()), NER_SAMPLE_SIZE),
            'entity_diversity': {},
            'top_entities': {}
        }
        
        for entity_type, counter in entity_counts.items():
            insights['entity_diversity'][entity_type] = len(counter)
            insights['top_entities'][entity_type] = counter.most_common(5)
        
        return insights
    
    def analyze_brand_mentions(self, text_series, title="Text"):
        """
        Analyze car brand mentions specifically
        
        Args:
            text_series: Pandas series containing text data
            title: Title for the analysis
        """
        print(f"\n=== {title} - Car Brand Analysis ===")
        
        brand_mentions = Counter()
        
        for text in text_series.dropna():
            if pd.isna(text):
                continue
            
            text_lower = str(text).lower()
            for brand in CAR_BRANDS:
                if brand in text_lower:
                    brand_mentions[brand] += 1
        
        if brand_mentions:
            print(f"\nCar Brand Mentions in {title}:")
            for brand, count in brand_mentions.most_common():
                print(f"- {brand.title()}: {count}")
            
            # Plot brand mentions
            top_brands = brand_mentions.most_common(10)
            if top_brands:
                brands, counts = zip(*top_brands)
                
                plt.figure(figsize=(12, 6))
                sns.barplot(x=list(counts), y=list(brands), palette='viridis')
                plt.title(f'{title} - Car Brand Mentions')
                plt.xlabel('Number of Mentions')
                plt.ylabel('Brand')
                plt.show()
        else:
            print("No car brand mentions found")
        
        return brand_mentions
    
    def analyze_entities(self, text_series, title="Text"):
        """
        Analyze named entities in text data (main method for testing)
        
        Args:
            text_series: Pandas series containing text data
            title: Title for the analysis
            
        Returns:
            dict: Dictionary containing entity analysis results
        """
        print(f"\n=== {title} - Named Entity Analysis ===")
        
        # Use the appropriate extraction method
        if self.nlp is not None:
            entity_counts = self._extract_entities_spacy(text_series, title)
        else:
            entity_counts = self.extract_entities_simple(text_series, title)
        
        # Also analyze brand mentions
        brand_mentions = self.analyze_brand_mentions(text_series, title)
        
        # Combine results
        results = {
            'entity_counts': entity_counts,
            'brand_mentions': brand_mentions,
            'total_documents': len(text_series.dropna())
        }
        
        return results
    
    def get_entity_insights(self, text_series, title="Text"):
        """
        Extract insights from NER analysis (alias for get_ner_insights)
        
        Args:
            text_series: Pandas series containing text data
            title: Title for the analysis
            
        Returns:
            dict: Dictionary containing NER insights
        """
        # Use the existing get_ner_insights method
        insights = self.get_ner_insights(text_series, title)
        
        # Add additional insights for compatibility
        entity_counts = self.extract_entities(text_series, title)
        
        # Calculate total entities
        total_entities = sum(sum(counter.values()) for counter in entity_counts.values())
        unique_entities = sum(len(counter) for counter in entity_counts.values())
        
        # Get entity types
        entity_types = {entity_type: len(counter) for entity_type, counter in entity_counts.items() if counter}
        
        # Get top entities by type
        top_entities_by_type = {}
        for entity_type, counter in entity_counts.items():
            if counter:
                top_entities_by_type[entity_type] = counter.most_common(10)
        
        # Update insights
        insights.update({
            'total_entities': total_entities,
            'unique_entities': unique_entities,
            'entity_types': entity_types,
            'top_entities_by_type': top_entities_by_type
        })
        
        return insights


def main():
    """Test NER analysis with actual datasets"""
    import pandas as pd
    
    print("="*60)
    print("NER ANALYSIS TEST - CAR NEWS & REVIEWS")
    print("="*60)
    
    try:
        # Load datasets
        print("\nLoading datasets...")
        car_news_df = pd.read_csv('car_news_dataset.csv')
        car_reviews_df = pd.read_csv('car_reviews_dataset.csv')
        
        print(f"✓ Car News Dataset: {car_news_df.shape}")
        print(f"✓ Car Reviews Dataset: {car_reviews_df.shape}")
        
        # Initialize NER analyzer
        ner_analyzer = NERAnalyzer()
        
        # Analyze Car News Entities
        print("\n" + "="*40)
        print("ANALYZING CAR NEWS ENTITIES")
        print("="*40)
        
        if 'content' in car_news_df.columns:
            news_text = car_news_df['content'].dropna()
            print(f"Processing {len(news_text)} news articles...")
            
            # Analyze entities
            news_entities = ner_analyzer.analyze_entities(news_text, "Car News")
            
            # Get insights
            news_insights = ner_analyzer.get_entity_insights(news_text, "Car News")
            print(f"\nNews Entity Summary:")
            print(f"- Total entities found: {news_insights.get('total_entities', 0)}")
            print(f"- Unique entities: {news_insights.get('unique_entities', 0)}")
            print(f"- Entity types: {list(news_insights.get('entity_types', {}).keys())}")
            
            # Show top entities by type
            for entity_type, entities in news_insights.get('top_entities_by_type', {}).items():
                print(f"\nTop {entity_type} entities:")
                for entity, count in entities[:5]:
                    print(f"  - {entity}: {count}")
        else:
            print("❌ 'content' column not found in news dataset")
        
        # Analyze Car Reviews Entities
        print("\n" + "="*40)
        print("ANALYZING CAR REVIEWS ENTITIES")
        print("="*40)
        
        if 'Review' in car_reviews_df.columns:
            review_text = car_reviews_df['Review'].dropna()
            print(f"Processing {len(review_text)} reviews...")
            
            # Analyze entities
            review_entities = ner_analyzer.analyze_entities(review_text, "Car Reviews")
            
            # Get insights
            review_insights = ner_analyzer.get_entity_insights(review_text, "Car Reviews")
            print(f"\nReview Entity Summary:")
            print(f"- Total entities found: {review_insights.get('total_entities', 0)}")
            print(f"- Unique entities: {review_insights.get('unique_entities', 0)}")
            print(f"- Entity types: {list(review_insights.get('entity_types', {}).keys())}")
            
            # Show top entities by type
            for entity_type, entities in review_insights.get('top_entities_by_type', {}).items():
                print(f"\nTop {entity_type} entities:")
                for entity, count in entities[:5]:
                    print(f"  - {entity}: {count}")
        else:
            print("❌ 'Review' column not found in reviews dataset")
        
        print("\n" + "="*60)
        print("NER ANALYSIS TEST COMPLETE!")
        print("="*60)
        
    except Exception as e:
        print(f"❌ Error during NER analysis test: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 