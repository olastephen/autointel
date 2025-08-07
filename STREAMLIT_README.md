# 🚗 Car Analysis Dashboard - Streamlit Application

## Overview

The Car Analysis Dashboard is a comprehensive Streamlit application that displays advanced analytics results from car news and reviews data. The dashboard provides interactive visualizations and insights for sentiment analysis, topic modeling, named entity recognition, keyword analysis, correlations, and time-series analysis.

## 🎯 Features

### 📈 Overview & Metrics
- **Key Performance Indicators**: Total news articles, reviews, average ratings, and sentiment distribution
- **Data Overview**: Dataset shapes, column information, and date ranges
- **Quick Insights**: High-level summary of the analysis results

### 😊 Sentiment Analysis
- **Sentiment Distribution**: Interactive pie charts showing positive, negative, and neutral sentiment
- **Sentiment Insights**: Detailed breakdown of sentiment percentages and counts
- **Comparative Analysis**: Side-by-side comparison of news vs reviews sentiment

### 📝 Topic Modeling
- **LDA Topic Extraction**: Display of extracted topics using Latent Dirichlet Allocation
- **Topic Visualization**: Sample topics from both news and reviews datasets
- **Theme Identification**: Key themes and discussions in the automotive industry

### 🏷️ Named Entity Recognition (NER)
- **Entity Extraction**: Identification of car brands, models, locations, and organizations
- **Entity Frequency**: Bar charts showing most frequently mentioned entities
- **Brand Tracking**: Monitor which brands and models are most discussed

### 🔑 Keyword Analysis
- **Keyword Frequency**: Horizontal bar charts of most frequent keywords
- **Trending Topics**: Identification of popular terms and phrases
- **Content Themes**: Understanding what aspects of cars are most discussed

### 📊 Correlation Analysis
- **Rating Distribution**: Histogram of review ratings
- **Sentiment vs Rating**: Scatter plot showing correlation between sentiment and ratings
- **Correlation Insights**: Statistical analysis of relationships between variables
- **Review Length Analysis**: Correlation between review length and ratings

### ⏰ Time Series Analysis
- **Sentiment Trends**: Line charts showing sentiment changes over time
- **Monthly Patterns**: Temporal analysis of sentiment and engagement
- **Event Impact**: Monitor how major events affect automotive sentiment

### 📋 Raw Data Explorer
- **Data Tables**: Interactive tables showing sample data
- **Data Download**: Export functionality for analysis results
- **Column Information**: Detailed view of all analysis columns

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8 or higher
- PostgreSQL database with analysis results
- All required Python packages (see requirements.txt)

### Installation Steps

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Install spaCy Model** (if not already installed):
   ```bash
   python -m spacy download en_core_web_sm
   ```

3. **Configure Environment Variables**:
   Create a `.env` file with your database credentials:
   ```env
   DB_HOST=your-database-host
   DB_PORT=5432
   DB_NAME=your-database-name
   DB_USER=your-username
   DB_PASSWORD=your-password
   NEWS_TABLE=car_news
   REVIEWS_TABLE=car_reviews
   ```

4. **Run the Analysis Pipeline** (if not already done):
   ```bash
   python run_analysis.py
   ```

5. **Launch Streamlit Dashboard**:
   ```bash
   streamlit run streamlit_app.py
   ```

## 📊 Dashboard Sections

### 1. Overview & Metrics
- **Purpose**: High-level summary and key metrics
- **Key Visualizations**: 
  - Total counts (news articles, reviews)
  - Average ratings
  - Positive sentiment counts
  - Data shape and date range information

### 2. Sentiment Analysis
- **Purpose**: Analyze emotional tone of content
- **Key Visualizations**:
  - Pie charts for sentiment distribution
  - Percentage breakdowns
  - Comparative analysis between news and reviews

### 3. Topic Modeling
- **Purpose**: Identify key themes and discussions
- **Key Visualizations**:
  - Sample topic lists
  - Topic word groups
  - Theme identification

### 4. Named Entity Recognition
- **Purpose**: Identify brands, models, and locations
- **Key Visualizations**:
  - Entity frequency bar charts
  - Brand and model mentions
  - Geographic patterns

### 5. Keyword Analysis
- **Purpose**: Identify trending terms and phrases
- **Key Visualizations**:
  - Keyword frequency charts
  - Popular terms identification
  - Content theme analysis

### 6. Correlation Analysis
- **Purpose**: Understand relationships between variables
- **Key Visualizations**:
  - Rating distribution histograms
  - Sentiment vs rating scatter plots
  - Correlation coefficients

### 7. Time Series Analysis
- **Purpose**: Track trends over time
- **Key Visualizations**:
  - Sentiment trend lines
  - Monthly patterns
  - Temporal analysis

### 8. Raw Data Explorer
- **Purpose**: Access and export raw data
- **Key Features**:
  - Interactive data tables
  - CSV download functionality
  - Column information display

## 🔧 Technical Details

### Data Sources
- **News Data**: Car news articles with sentiment, topics, entities, and keywords
- **Reviews Data**: Car reviews with ratings, sentiment, topics, entities, and correlations

### Analysis Methods
- **Sentiment Analysis**: VADER and TextBlob for sentiment scoring
- **Topic Modeling**: Latent Dirichlet Allocation (LDA) for theme extraction
- **NER**: spaCy for entity recognition
- **Keyword Analysis**: TF-IDF and frequency analysis
- **Correlation Analysis**: Pearson correlation coefficients
- **Time Series**: Monthly aggregation and trend analysis

### Database Schema
The dashboard expects the following columns in the database tables:

**car_news table**:
- `title`, `content`, `link`, `author`, `publication_date`, `source`
- `sentiment`, `sentiment_score`, `topics`, `entities`, `keywords`, `top_ngrams`

**car_reviews table**:
- `title`, `verdict`, `rating`, `price`, `link`, `author`, `publication_date`, `source`
- `sentiment`, `sentiment_score`, `topics`, `entities`, `keywords`, `correlation_score`, `review_length`, `top_ngrams`

## 🎨 Customization

### Styling
The dashboard uses custom CSS for styling. You can modify the styles in the `streamlit_app.py` file:

```python
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    /* Add more custom styles here */
</style>
""", unsafe_allow_html=True)
```

### Adding New Visualizations
To add new visualizations, create new functions and add them to the appropriate section:

```python
def create_new_chart(df, title):
    """Create a new chart"""
    fig = px.bar(df, x='column1', y='column2', title=title)
    return fig

# Then use it in the main function:
if section == "New Section":
    fig = create_new_chart(df, "New Chart")
    st.plotly_chart(fig, use_container_width=True)
```

## 🚨 Troubleshooting

### Common Issues

1. **Database Connection Error**:
   - Check your `.env` file configuration
   - Verify database credentials
   - Ensure database is accessible

2. **Missing spaCy Model**:
   ```bash
   python -m spacy download en_core_web_sm
   ```

3. **Missing Analysis Data**:
   - Run the analysis pipeline first: `python run_analysis.py`
   - Check if database tables exist and contain data

4. **Streamlit Not Starting**:
   - Check if all dependencies are installed
   - Verify Python version compatibility
   - Check for port conflicts

### Performance Tips

1. **Data Caching**: The dashboard uses `@st.cache_data` for efficient data loading
2. **Lazy Loading**: Visualizations are created only when needed
3. **Efficient Queries**: Database queries are optimized for performance

## 📈 Future Enhancements

### Planned Features
- **Real-time Updates**: Live data refresh capabilities
- **Advanced Filtering**: Date range, brand, and sentiment filters
- **Export Functionality**: PDF reports and advanced export options
- **User Authentication**: Multi-user support with role-based access
- **Mobile Optimization**: Responsive design for mobile devices

### Analytics Enhancements
- **Predictive Analytics**: Sentiment trend forecasting
- **Brand Comparison**: Side-by-side brand analysis
- **Market Intelligence**: Competitive analysis features
- **Alert System**: Automated insights and notifications

## 🤝 Contributing

To contribute to the dashboard:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📞 Support

For support and questions:
- Create an issue in the repository
- Check the troubleshooting section
- Review the documentation

---

**🚗 Car Analysis Dashboard** - Transforming automotive data into actionable insights!
