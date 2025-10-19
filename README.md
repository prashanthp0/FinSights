🚀 FinSights – AI-Powered Financial News Analysis

FinSights is an advanced AI-powered financial news platform that leverages state-of-the-art NLP to provide real-time sentiment analysis, investment insights, and actionable intelligence. The application automatically scrapes financial news from multiple sources, analyzes sentiment using financial models, and generates clear insights for investors.

✨ Key Features
🔍 Smart News Aggregation

➤ Multi-source RSS Integration: Real-time news from 15+ financial sources including Bloomberg, Reuters, Yahoo Finance, and Economic Times

➤ Intelligent Keyword Filtering: Collect news based on company names, stock symbols, or financial topics

➤ Parallel Web Scraping: Optimized scraping with concurrent processing for faster collection

🤖 AI-Powered Analysis

➤ Sentiment Analysis: Using FinBERT (financial domain-specific BERT)

➤ 5-Point Sentiment Scale: Very Positive → Very Negative

➤ Smart Summarization: Automatic article summaries with BART-large-CNN

➤ Confidence Scoring: AI confidence metrics for each analysis

📊 Advanced Analytics & Visualization

➤ Interactive Dashboards: Real-time sentiment distribution with pie charts & histograms

➤ Confidence Analysis: Visual representation of AI confidence scores

➤ Sentiment-Confidence Correlation: Scatter plots showing relationship between sentiment & confidence

➤ Key Insights Extraction: Automated extraction of critical investment insights

💼 Investment Intelligence

➤ AI-Generated Recommendations: Buy/Hold/Sell signals based on sentiment analysis

➤ Risk Assessment: Comprehensive sentiment scoring and reasoning

➤ Comparative Analysis: Side-by-side positive and negative insights

➤ Professional Reporting: Exportable in PDF, CSV, and JSON formats

🎨 Professional User Experience

➤ Real-time Progress Tracking: Live progress bars during scraping & analysis

➤ Export Capabilities: PDF, CSV, and JSON report generation

🛠️ Technical Architecture
Core Technologies

Frontend: Streamlit 1.28+

Backend: Python 3.8+

AI/ML: Hugging Face Transformers, PyTorch

Data Processing: Pandas, NumPy

Visualization: Plotly, Streamlit Components

Web Scraping: BeautifulSoup4, Feedparser, Requests

AI Models Used

Sentiment Analysis: ProsusAI/finbert – Financial domain-specific BERT model

Text Summarization: facebook/bart-large-cnn – State-of-the-art summarization

Performance Optimizations

➤ Batch Processing: Parallel article analysis (2-3x speed improvement)

➤ Memory Optimization: Model quantization & efficient memory management

➤ Content Truncation: Smart content length optimization

➤ Caching: Model & result caching for faster subsequent runs

📦 Installation & Setup
Prerequisites

Python 3.8+

Minimum 4GB RAM (8GB recommended)

Stable internet connection

Step-by-Step Installation

Clone the Repository

git clone https://github.com/yourusername/finsights.git
cd finsights


Create Virtual Environment (Recommended)

python -m venv finsights_env
# On Mac/Linux
source finsights_env/bin/activate
# On Windows
finsights_env\Scripts\activate


Install Dependencies

pip install -r requirements.txt


Run the Application

streamlit run app.py
