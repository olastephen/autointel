# Car Analysis Framework

A comprehensive analysis framework for car news and reviews data using advanced NLP techniques and machine learning.

## 📁 Project Structure

```
demo_project/
├── src/                          # Source code
│   ├── config/                   # Configuration files
│   │   ├── __init__.py
│   │   └── config.py            # Main configuration
│   ├── data/                     # Data handling
│   │   ├── __init__.py
│   │   └── data_loader.py       # Data loading and preprocessing
│   ├── features/                 # Analysis features
│   │   ├── __init__.py
│   │   ├── sentiment_analyzer.py
│   │   ├── ner_analyzer.py
│   │   ├── ngram_analyzer.py
│   │   ├── keyword_analyzer.py
│   │   ├── topic_modeler.py
│   │   ├── correlation_analyzer.py
│   │   └── time_series_analyzer.py
│   └── analysis/                 # Main analysis framework
│       ├── __init__.py
│       └── car_analysis_framework.py
├── datasets/                     # Data files
│   ├── car_news_dataset.csv
│   └── car_reviews_dataset.csv
├── scripts/                      # Utility scripts
│   ├── migrate_to_database.py
│   ├── test_database.py
│   ├── test_all_features.py
│   └── test_analysis.py
├── docs/                         # Documentation
│   └── DATABASE_SETUP.md
├── main.py                       # Legacy main script
├── run_analysis.py              # New main entry point
├── requirements.txt
└── README.md
```

## 🚀 Quick Start

### Option 1: Using Online PostgreSQL Database (Recommended)

1. **Set up your database credentials** in a `.env` file:
   ```env
   DB_HOST=your-online-postgres-host.com
   DB_PORT=5432
   DB_NAME=your_database_name
   DB_USER=your_username
   DB_PASSWORD=your_password
   NEWS_TABLE=car_news
   REVIEWS_TABLE=car_reviews
   ```

2. **Migrate your data to the database**:
   ```bash
   python scripts/migrate_to_database.py
   ```

3. **Run the analysis**:
   ```bash
   python run_analysis.py
   ```

### Option 2: Using Local CSV Files

1. **Place your CSV files** in the `datasets/` directory:
   - `datasets/car_news_dataset.csv`
   - `datasets/car_reviews_dataset.csv`

2. **Run the analysis**:
   ```bash
   python main.py
   ```

## 📊 Features

### Analysis Modules

- **Sentiment Analysis**: VADER and TextBlob sentiment scoring
- **Named Entity Recognition**: Extract car brands, models, and features
- **N-gram Analysis**: Bigram and trigram frequency analysis
- **Keyword Analysis**: Car-related keyword extraction and correlation
- **Topic Modeling**: LDA-based topic discovery
- **Correlation Analysis**: Feature correlation analysis
- **Time Series Analysis**: Temporal trend analysis

### Data Sources

- **Car News**: News articles about cars and automotive industry
- **Car Reviews**: User reviews and ratings of vehicles

## 🧠 NLP Techniques & Tools

### Core NLP Libraries

#### **Natural Language Processing (NLTK)**
- **`nltk`** - Core NLP library for text processing
- **`stopwords`** - Remove common words (the, and, or, etc.)
- **`word_tokenize`** - Split text into individual words
- **`sent_tokenize`** - Split text into sentences
- **`WordNetLemmatizer`** - Reduce words to root form (running → run)
- **`ngrams`** - Extract word sequences (bigrams, trigrams)

#### **Advanced NLP (spaCy)**
- **`spacy`** - Industrial-strength NLP library
- **`en_core_web_sm`** - English language model
- **Named Entity Recognition (NER)** - Identify organizations, locations, products
- **Part-of-speech tagging** - Identify word types (noun, verb, adjective)

#### **Sentiment Analysis**
- **`TextBlob`** - Polarity and subjectivity scoring
- **`VADER`** - Valence Aware Dictionary and sEntiment Reasoner
  - Compound sentiment score (-1 to +1)
  - Positive, negative, neutral probabilities

### Machine Learning & Text Processing

#### **Text Vectorization**
- **`CountVectorizer`** - Convert text to word frequency matrix
- **`TfidfVectorizer`** - Term frequency-inverse document frequency
- **Feature extraction** - Transform text into numerical features

#### **Topic Modeling**
- **`LatentDirichletAllocation (LDA)`** - Discover hidden topics in documents
- **`NMF`** - Non-negative Matrix Factorization for topic extraction
- **Document-topic assignment** - Map each document to dominant topics

#### **Clustering & Similarity**
- **`KMeans`** - Group similar documents together
- **`cosine_similarity`** - Measure text similarity between documents

### Text Analysis Techniques

#### **Keyword Extraction**
- **Frequency-based analysis** - Count word occurrences
- **Stopword removal** - Filter out common, non-meaningful words
- **Document-level keyword counting** - Track keywords per document
- **Global keyword ranking** - Identify most important terms across dataset

#### **N-gram Analysis**
- **Bigrams** - Two-word sequences (e.g., "electric vehicle")
- **Trigrams** - Three-word sequences (e.g., "new car model")
- **Frequency counting** - Track phrase occurrences
- **Document-level n-gram tracking** - Monitor phrase usage per document

#### **Named Entity Recognition**
- **Organization detection** - Car companies, brands
- **Geographic entities** - Locations, cities
- **Product identification** - Car models, features
- **Custom entity extraction** - Car brand recognition

### Visualization & Output

#### **Text Visualization**
- **`WordCloud`** - Visual representation of word frequencies
- **Interactive charts** - Plotly-based visualizations
- **Frequency distributions** - Bar charts of most common terms
- **Topic visualization** - Display discovered topics

### Data Processing & Storage

#### **Text Preprocessing**
- **Text cleaning** - Remove special characters, normalize
- **Tokenization** - Split into words/sentences
- **Lemmatization** - Reduce words to base form
- **Case normalization** - Convert to lowercase

#### **Database Integration**
- **JSON serialization** - Store complex NLP results
- **PostgreSQL storage** - Persistent storage of analysis results
- **Structured data** - Organized storage of entities, topics, keywords

### Complete Package Requirements

```bash
# Core NLP
nltk>=3.6.0          # Natural language processing
spacy>=3.1.0         # Advanced NLP with pre-trained models
textblob>=0.15.0     # Sentiment analysis
vaderSentiment>=3.3.0 # VADER sentiment analysis

# Machine Learning
scikit-learn>=1.0.0  # Text vectorization, topic modeling, clustering
gensim>=4.0.0        # Additional NLP algorithms

# Visualization
wordcloud>=1.8.0     # Word cloud generation
plotly>=5.0.0        # Interactive charts

# Data Processing
pandas>=1.3.0        # Data manipulation
numpy>=1.21.0        # Numerical operations
```

### NLP Pipeline Summary

The framework implements a **comprehensive NLP pipeline**:

1. **Text Input** → 2. **Preprocessing** → 3. **Feature Extraction** → 4. **Analysis** → 5. **Results Storage**

This provides **enterprise-grade NLP capabilities** for automotive text analysis, combining multiple approaches for robust sentiment analysis, topic discovery, and entity recognition.

## 🛠️ Installation

1. **Clone the repository**:
```bash
   git clone <repository-url>
   cd demo_project
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables** (for database mode):
   ```bash
   cp .env.example .env
   # Edit .env with your database credentials
   ```

## 📈 Usage Examples

### Basic Analysis
```python
from src.analysis.car_analysis_framework import CarAnalysisFramework

# Initialize with database
analyzer = CarAnalysisFramework(use_database=True)
analyzer.run_full_analysis()
```

### Individual Feature Analysis
```python
from src.features.sentiment_analyzer import SentimentAnalyzer
from src.data.data_loader import DataLoader

# Load data
loader = DataLoader(use_database=True)
loader.load_data()

# Analyze sentiment
sentiment = SentimentAnalyzer()
sentiment.analyze_sentiment(loader.car_news_df, 'content', 'Car News')
```

## 🔧 Configuration

Edit `src/config/config.py` to customize:
- Analysis parameters (N_TOPICS, MAX_FEATURES, etc.)
- Visualization settings
- Car brands and keywords
- Database settings

## 📝 Database Schema

### Car News Table
- `title`, `content` (from CSV)
- `link`, `author`, `publication_date`, `source` (metadata)
- `sentiment`, `sentiment_score` (analysis results)
- `topics`, `entities`, `keywords`, `top_ngrams` (NLP analysis)

### Car Reviews Table
- `title`, `author`, `rating`, `verdict` (from CSV)
- `publication_date`, `link`, `source`, `price` (metadata)
- `sentiment`, `sentiment_score` (analysis results)
- `topics`, `entities`, `keywords`, `top_ngrams` (NLP analysis)
- `correlation_score`, `review_length` (additional metrics)

## 🧪 Testing

### Test Database Connection
```bash
python scripts/test_database.py
```

### Test Individual Features
```bash
python scripts/test_all_features.py
```

### Test Complete Analysis
   ```bash
python scripts/test_analysis.py
```

## 📊 Output

The framework generates:
- Comprehensive analysis reports
- Visualizations and charts
- Statistical summaries
- JSON results files (optional)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🆘 Support

For issues and questions:
1. Check the documentation in `docs/`
2. Review the test scripts in `scripts/`
3. Open an issue on GitHub 