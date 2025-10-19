import streamlit as st
import requests
import feedparser
from bs4 import BeautifulSoup
import pandas as pd
import urllib.request
from urllib.parse import quote
import re
import time
from transformers import pipeline
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import plotly.express as px
from fpdf import FPDF
from datetime import datetime

# ==============================================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="FinSights - Financial News Analyzer",
    layout="wide",
    page_icon="📈"
)

# ==============================================================================

@st.cache_resource(show_spinner=False)
def load_models():
    """Loads and caches the AI models with memory optimization."""
    with st.spinner("Loading AI models for the first time... This can take a few minutes. 🧠"):
        device = 0 if torch.cuda.is_available() else -1
        model_kwargs = {
            "torch_dtype": torch.float16 if device == 0 else torch.float32,
            "low_cpu_mem_usage": True
        }
        try:
            sentiment_analyzer = pipeline(
                "text-classification",
                model="ProsusAI/finbert",
                device=device,
                model_kwargs=model_kwargs,
                truncation=True
            )

            summarizer = pipeline(
                "summarization",
                model="facebook/bart-large-cnn",
                device=device,
                model_kwargs=model_kwargs
            )

            return sentiment_analyzer, summarizer

        except Exception as e:
            logger.exception("Failed to load models")
            st.error(f"❌ Failed to load models: {str(e)}")
            return None, None

# ==============================================================================

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    )
}

def parse_feed(rss_url):
    try:
        req = urllib.request.Request(rss_url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=15) as response:
            return feedparser.parse(response.read())
    except Exception:
        logger.debug(f"Failed to parse feed {rss_url}", exc_info=True)
        return None

def fetch_article_content(url):
    try:
        time.sleep(0.25)
        resp = requests.get(url, headers=HEADERS, timeout=20)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.content, "lxml")
        title = soup.title.string.strip() if soup.title and soup.title.string else "No title found"

        for tag in soup(["script", "style", "nav", "header", "footer", "aside"]):
            tag.decompose()

        content_selectors = [
            "article",
            ".story-content", ".article-body", ".main-content",
            ".content", ".post-content", ".entry-content",
            "[class*='article']", "[class*='content']",
            "div.story", "div.article", "div.main"
        ]

        main_content = None
        for sel in content_selectors:
            main_content = soup.select_one(sel)
            if main_content and len(main_content.get_text(strip=True)) > 200:
                break

        target = main_content if main_content else soup.find("body") or soup

        paragraphs = target.find_all("p")
        parts = []
        for p in paragraphs:
            txt = p.get_text(separator=" ", strip=True)
            wc = len(txt.split())
            if wc > 15 and wc < 400:
                parts.append(txt)

        content = "\n\n".join(parts[:20])

        if len(content) < 100:
            return None

        return {"title": title, "content": content, "link": url, "content_length": len(content)}

    except Exception:
        logger.debug(f"Failed to fetch article {url}", exc_info=True)
        return None

def get_news_rss_urls(keywords):
    search_query = "+".join([quote(k) for k in keywords])
    return [
        f"https://news.search.yahoo.com/rss?p={search_query}+business+finance&ei=UTF-8",
        f"https://www.bing.com/news/search?q={search_query}+financial&format=rss",
    ]

def scrape_feed_articles(feed_url, keywords, processed_links):
    try:
        feed = parse_feed(feed_url)
        if not feed or not getattr(feed, "entries", None):
            return []

        results = []
        for entry in feed.entries[:10]:
            link = entry.get("link") or entry.get("url") or entry.get("id")
            if not link or link in processed_links:
                continue

            title = (entry.get("title") or "").lower()
            summary = (entry.get("summary") or "").lower()

            if any(k in title or k in summary for k in keywords):
                article = fetch_article_content(link)
                if article:
                    results.append(article)
                    processed_links.add(link)

        return results

    except Exception:
        logger.debug(f"Error processing feed {feed_url}", exc_info=True)
        return []

def scrape_all_news(keywords, progress_obj=None):
    rss_urls = [
        "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
        "https://www.livemint.com/rss/companies",
        "https://feeds.feedburner.com/ndtvprofit-latest",
        "https://www.moneycontrol.com/rss/latestnews.xml",
        "https://www.moneycontrol.com/rss/business.xml",
        "https://feeds.bloomberg.com/markets/news.rss",
        "https://feeds.reuters.com/reuters/businessNews",
        "https://finance.yahoo.com/news/rssindex",
        "http://feeds.bbci.co.uk/news/business/rss.xml",
        "https://www.cnbc.com/id/10000664/device/rss/rss.html",
        "https://www.business-standard.com/rss/home_page_top_stories.rss",
        "https://www.thehindubusinessline.com/news/feeder/default.rss",
    ]
    dynamic = get_news_rss_urls(keywords)
    all_feeds = rss_urls + dynamic

    all_articles = []
    processed = set()

    with ThreadPoolExecutor(max_workers=5) as ex:
        futures = {ex.submit(scrape_feed_articles, f, keywords, processed): f for f in all_feeds}
        completed = 0
        total = len(all_feeds)
        for future in as_completed(futures):
            feed_url = futures[future]
            try:
                articles = future.result(timeout=30)
                if articles:
                    all_articles.extend(articles)
            except Exception:
                logger.debug(f"Feed {feed_url} failed", exc_info=True)
            completed += 1
            if progress_obj:
                try:
                    progress_obj.progress(min(1.0, completed / total))
                except Exception:
                    pass

    return all_articles

def get_five_point_sentiment(label, score, threshold=0.85):
    label = label.capitalize()
    if label == "Positive" and score > threshold:
        return "Very Positive"
    if label == "Negative" and score > threshold:
        return "Very Negative"
    if label == "Neutral" and score > threshold:
        return "Strongly Neutral"
    return label

# ==============================================================================

def analyze_article_batch(articles_batch, sentiment_analyzer, summarizer, confidence_threshold):
    results = []

    def process_single(article):
        try:
            content = str(article.get("content", "")).strip()
            if len(content) < 100:
                return None

            if len(content) > 800:
                content = content[:800]

            sentiment_result = sentiment_analyzer(content, truncation=True, top_k=1)[0]
            score = float(sentiment_result.get("score", 0.0))
            label = sentiment_result.get("label", "Neutral")

            if score < confidence_threshold:
                return None

            five_label = get_five_point_sentiment(label, score)

            if score > 0.7:
                summ = summarizer(content, max_length=80, min_length=20, do_sample=False, truncation=True)[0]
                summary_text = summ.get("summary_text") or summ.get("summary") or ""
                insight = summary_text.split(".")[0].strip()
                if insight and not insight.endswith("."):
                    insight += "."
            else:
                summary_text = "Summary omitted due to low confidence"
                insight = "Quick analysis completed."

            return {
                "Title": article.get("title", "No title"),
                "Sentiment": five_label,
                "Confidence": round(score, 3),
                "Key Insight": insight,
                "Summary": summary_text,
                "Link": article.get("link")
            }
        except Exception:
            logger.debug("Error processing article", exc_info=True)
            return None

    with ThreadPoolExecutor(max_workers=2) as ex:
        futures = [ex.submit(process_single, art) for art in articles_batch]
        for f in as_completed(futures):
            try:
                r = f.result(timeout=20)
                if r:
                    results.append(r)
            except Exception:
                continue

    return results

# ==============================================================================

def create_sentiment_pie_chart(df):
    counts = df["Sentiment"].value_counts()
    color_map = {
        "Very Positive": "#00FF00",
        "Positive": "#90EE90",
        "Neutral": "#FFA500",
        "Strongly Neutral": "#FFD700",
        "Negative": "#FF6B6B",
        "Very Negative": "#FF0000"
    }
    fig = px.pie(
        names=counts.index,
        values=counts.values,
        title="📊 Sentiment Distribution",
        color=counts.index,
        color_discrete_map=color_map
    )
    fig.update_traces(textposition="inside", textinfo="percent+label")
    fig.update_layout(showlegend=True)
    return fig

def create_confidence_histogram(df):
    fig = px.histogram(df, x="Confidence", nbins=20, title="📈 Confidence Score Distribution", color_discrete_sequence=["#1f77b4"])
    fig.update_layout(xaxis_title="Confidence Score", yaxis_title="Number of Articles", showlegend=False)
    return fig

def create_sentiment_confidence_scatter(df):
    order = ["Very Negative", "Negative", "Neutral", "Strongly Neutral", "Positive", "Very Positive"]
    df = df.copy()
    df["Sentiment_Num"] = df["Sentiment"].apply(lambda x: order.index(x) if x in order else -1)
    fig = px.scatter(df, x="Confidence", y="Sentiment_Num", color="Sentiment", hover_data=["Title"], size="Confidence", size_max=15, title="🎯 Sentiment vs Confidence Correlation")
    fig.update_layout(yaxis=dict(title="Sentiment", tickmode="array", tickvals=list(range(len(order))), ticktext=order), xaxis_title="Confidence Score")
    return fig

def create_stock_recommendations(df):
    mapping = {"Very Positive": 2, "Positive": 1, "Neutral": 0, "Strongly Neutral": 0, "Negative": -1, "Very Negative": -2}
    df = df.copy()
    df["Sentiment_Score"] = df["Sentiment"].map(mapping).fillna(0)
    avg = df["Sentiment_Score"].mean() if len(df) > 0 else 0.0
    total = len(df)
    if avg >= 1.5:
        rec = "STRONG BUY 🟢"
        reasoning = "Overwhelmingly positive sentiment across multiple news sources"
    elif avg >= 0.5:
        rec = "BUY 🟢"
        reasoning = "Generally positive sentiment with strong confidence"
    elif avg >= -0.5:
        rec = "HOLD 🟡"
        reasoning = "Mixed or neutral sentiment - monitor closely"
    elif avg >= -1.5:
        rec = "SELL 🔴"
        reasoning = "Negative sentiment trending across news sources"
    else:
        rec = "STRONG SELL 🔴"
        reasoning = "Overwhelmingly negative sentiment with high confidence"

    return {"recommendation": rec, "score": avg, "reasoning": reasoning, "total_articles": total}

# ==============================================================================

class PDFReport(FPDF):
    def header(self):
        self.set_font("Arial", "B", 16)
        self.cell(0, 10, "FinSights - Financial News Analysis Report", 0, 1, "C")
        self.ln(5)

    def chapter_title(self, title):
        self.set_font("Arial", "B", 12)
        self.cell(0, 10, title, 0, 1, "L")
        self.ln(2)

    def chapter_body(self, body):
        self.set_font("Arial", "", 10)
        self.multi_cell(0, 8, body)
        self.ln()

def sanitize_for_pdf(text):
    """Remove characters that can't be encoded with latin-1 (drops emojis & other non-latin1)."""
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    return text.encode("latin-1", errors="ignore").decode("latin-1")


def create_pdf_report(df, search_term, recommendation):
    pdf = PDFReport()
    pdf.add_page()
    pdf.set_font("Arial", "B", 14)

    safe_search = sanitize_for_pdf(search_term)
    pdf.cell(0, 10, f"Analysis for: {safe_search}", 0, 1, "C")
    pdf.ln(10)

    # Recommendation section (sanitize all written text)
    pdf.chapter_title(sanitize_for_pdf("Investment Recommendation"))
    pdf.chapter_body(sanitize_for_pdf(f"Recommendation: {recommendation.get('recommendation', '')}"))
    pdf.chapter_body(sanitize_for_pdf(f"Sentiment Score: {recommendation.get('score', 0.0):.2f}"))
    pdf.chapter_body(sanitize_for_pdf(f"Reasoning: {recommendation.get('reasoning', '')}"))
    pdf.chapter_body(sanitize_for_pdf(f"Articles Analyzed: {recommendation.get('total_articles', 0)}"))
    pdf.ln(10)

    # Sentiment summary
    pdf.chapter_title(sanitize_for_pdf("Sentiment Summary"))
    try:
        sentiment_summary = df["Sentiment"].value_counts().to_string()
    except Exception:
        sentiment_summary = ""
    pdf.chapter_body(sanitize_for_pdf(sentiment_summary))
    pdf.ln(10)

    # Detailed article analysis (sanitize each field)
    pdf.chapter_title(sanitize_for_pdf("Detailed Article Analysis"))
    for i, (_, row) in enumerate(df.iterrows()):
        if i >= 10:
            break
        title = sanitize_for_pdf(row.get("Title", "No title"))
        sentiment = sanitize_for_pdf(row.get("Sentiment", ""))
        confidence = row.get("Confidence", "")
        insight = sanitize_for_pdf(row.get("Key Insight", ""))
        link = sanitize_for_pdf(row.get("Link", ""))

        pdf.set_font("Arial", "B", 10)
        pdf.multi_cell(0, 8, f"Article {i+1}: {title}")
        pdf.set_font("Arial", "", 9)
        pdf.multi_cell(0, 6, sanitize_for_pdf(f"Sentiment: {sentiment} (Confidence: {confidence})"))
        if insight:
            pdf.multi_cell(0, 6, sanitize_for_pdf(f"Insight: {insight}"))
        if link:
            pdf.multi_cell(0, 6, sanitize_for_pdf(f"Link: {link}"))
        pdf.ln(5)

    pdf.chapter_title(sanitize_for_pdf("Disclaimer"))
    pdf.chapter_body(
        sanitize_for_pdf(
            "This platform is for educational and informational purposes only and does not constitute financial or investment advice. Always do your own research or consult a qualified advisor before making investment decisions."
        )
    )

    # Make filename safe (remove characters that could be problematic)
    safe_term = re.sub(r"[^\w\-_\. ]", "", sanitize_for_pdf(search_term)).strip()
    if not safe_term:
        safe_term = "report"
    safe_term = safe_term[:50].replace(" ", "_")
    filename = f"finsights_report_{safe_term}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"

    pdf.output(filename)
    return filename

# ==============================================================================

def apply_custom_css(dark_mode=False):
    if dark_mode:
        background_color = "#0E1117"
        text_color = "#FFFFFF"
        secondary_bg = "#262730"
        border_color = "#505050"
    else:
        background_color = "#FFFFFF"
        text_color = "#000000"
        secondary_bg = "#f0f2f6"
        border_color = "#d0d0d0"

    st.markdown(
        f"""
    <style>
    .main-header {{
        font-size: 2.5rem;
        color: {text_color};
        text-align: center;
        margin-bottom: 2rem;
    }}
    .metric-card {{
        background-color: {secondary_bg};
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        color: {text_color};
    }}
    .positive-sentiment {{
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        padding: 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
        color: #000000;
    }}
    .negative-sentiment {{
        background-color: #f8d7da;
        border-left: 5px solid #dc3545;
        padding: 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
        color: #000000;
    }}
    .neutral-sentiment {{
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
        padding: 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
        color: #000000;
    }}
    .dataframe {{
        color: {text_color} !important;
    }}
    .disclaimer {{
        background-color: {secondary_bg};
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #ff6b6b;
        margin: 1rem 0;
        color: {text_color};
        font-size: 0.9rem;
    }}
    </style>
    """,
        unsafe_allow_html=True,
    )

# ==============================================================================

def main():
    if "dark_mode" not in st.session_state:
        st.session_state.dark_mode = False

    apply_custom_css(st.session_state.dark_mode)

    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.markdown('<h1 class="main-header">📈 FinSights - AI Financial News Analyzer</h1>', unsafe_allow_html=True)
    with col3:
        dark_mode = st.checkbox("🌙 Dark Mode", value=st.session_state.dark_mode)
        if dark_mode != st.session_state.dark_mode:
            st.session_state.dark_mode = dark_mode
            st.experimental_rerun()

    st.markdown(
        """
    <div style='text-align: center; margin-bottom: 2rem;'>
    Enter a company name, stock symbol, or financial topic to get AI-powered news analysis.
    The app will scrape recent news, analyze sentiment, and provide investment recommendations.
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
    <div class="disclaimer">
    <strong>Disclaimer:</strong> This platform is for educational and informational purposes only and does not constitute financial or investment advice. Always do your own research or consult a qualified advisor before making investment decisions.
    </div>
    """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("⚙️ Settings")
        st.subheader("🚀 Performance")
        max_articles = st.slider("Maximum Articles to Analyze", 5, 30, 15)
        confidence_threshold = st.slider("Confidence Threshold", 0.5, 0.95, 0.7)
        batch_size = st.select_slider("Processing Speed", options=["Fast", "Balanced", "Thorough"], value="Balanced")
        st.markdown("---")
        st.markdown("### 📊 Visualization Options")
        show_pie_chart = st.checkbox("Show Sentiment Pie Chart", value=True)
        show_histogram = st.checkbox("Show Confidence Histogram", value=True)
        show_scatter = st.checkbox("Show Sentiment-Confidence Correlation", value=True)
        st.markdown("---")
        st.markdown("### 💡 How to use:")
        st.markdown(
            """
        1. Enter company/topic (e.g., 'TCS', 'Bitcoin', 'Stock Market')
        2. Click 'Analyze News'
        3. View sentiment analysis & investment recommendations
        4. Export results as PDF
        """
        )

    sentiment_analyzer, summarizer = load_models()

    if sentiment_analyzer is None or summarizer is None:
        st.error("Failed to load AI models. Please refresh the page and try again.")
        return

    with st.form("search_form"):
        c1, c2 = st.columns([3, 1])
        with c1:
            search_term = st.text_input(
                "🔍 Enter Company or Keyword",
                placeholder="e.g., Reliance, Gold prices, Nifty 50",
                help="You can enter multiple keywords separated by commas",
            )
        with c2:
            st.markdown("<br>", unsafe_allow_html=True)
            submitted = st.form_submit_button("⚡ Analyze News", use_container_width=True)

    if submitted and search_term:
        if not search_term.strip():
            st.warning("Please enter a valid search term.")
            return

        raw_keywords = [k.strip() for k in search_term.split(",")]
        keywords = [k.lower() for k in raw_keywords if k.strip()]

        with st.spinner(f"🔍 Searching for news about: {', '.join(raw_keywords)}..."):
            scrape_progress = st.progress(0)
            scraped_articles = scrape_all_news(keywords, scrape_progress)
            scrape_progress.progress(1.0)

            if not scraped_articles:
                st.warning(
                    """
                ❌ No articles found for the given keywords. Try:
                - Using different keywords
                - Checking spelling
                - Using broader terms (e.g., 'stocks' instead of specific company)
                """
                )
                return

            raw_df = pd.DataFrame(scraped_articles).drop_duplicates(subset=["link"])
            raw_df = raw_df.head(max_articles)

            st.success(f"📰 Found {len(raw_df)} articles. Starting AI analysis...")

            analyzed_results = []
            analysis_progress = st.progress(0)
            analysis_status = st.empty()

            batch_config = {"Fast": 2, "Balanced": 3, "Thorough": 1}
            actual_batch_size = batch_config.get(batch_size, 3)
            total_batches = (len(raw_df) + actual_batch_size - 1) // actual_batch_size

            for batch_num in range(total_batches):
                start_idx = batch_num * actual_batch_size
                end_idx = min((batch_num + 1) * actual_batch_size, len(raw_df))
                batch_articles = raw_df.iloc[start_idx:end_idx].to_dict("records")

                batch_results = analyze_article_batch(batch_articles, sentiment_analyzer, summarizer, confidence_threshold)
                analyzed_results.extend(batch_results)

                progress = (batch_num + 1) / total_batches
                analysis_progress.progress(min(1.0, progress))
                analysis_status.text(f"Processed batch {batch_num+1}/{total_batches} ({len(batch_results)} articles)")

            analysis_progress.progress(1.0)
            analysis_status.empty()

            if not analyzed_results:
                st.warning("No articles passed the confidence threshold. Try lowering the threshold in sidebar.")
                return

            final_df = pd.DataFrame(analyzed_results)

            recommendation = create_stock_recommendations(final_df)

            st.subheader("🎯 Investment Recommendation")

            # prepare colors
            if "BUY" in recommendation["recommendation"]:
                bg = "#d4edda"
                border = "#28a745"
                text = "#155724"
                rec_color = "🟢"
            elif "SELL" in recommendation["recommendation"]:
                bg = "#f8d7da"
                border = "#dc3545"
                text = "#721c24"
                rec_color = "🔴"
            else:
                bg = "#fff3cd"
                border = "#ffc107"
                text = "#856404"
                rec_color = "🟡"

            st.markdown(
                f"""
            <div style='background-color: {bg}; padding: 1.5rem; border-radius: 10px; border-left: 5px solid {border};'>
                <h3 style='margin: 0; color: {text};'>{rec_color} {recommendation['recommendation']}</h3>
                <p style='margin: 0.5rem 0 0 0; color: {text};'>{recommendation['reasoning']}</p>
                <p style='margin: 0.5rem 0 0 0; color: {text};'>Sentiment Score: {recommendation['score']:.2f} | Articles Analyzed: {recommendation['total_articles']}</p>
            </div>
            """,
                unsafe_allow_html=True,
            )

            st.subheader("💡 Key Insights Summary")
            insights_col1, insights_col2 = st.columns(2)

            with insights_col1:
                positive_articles = final_df[final_df["Sentiment"].str.contains("Positive")]
                if len(positive_articles) > 0:
                    st.write("**✅ Positive Highlights:**")
                    for _, article in positive_articles.head(3).iterrows():
                        st.write(f"• {article['Key Insight']}")

            with insights_col2:
                negative_articles = final_df[final_df["Sentiment"].str.contains("Negative")]
                if len(negative_articles) > 0:
                    st.write("**❌ Concerns:**")
                    for _, article in negative_articles.head(3).iterrows():
                        st.write(f"• {article['Key Insight']}")

            st.subheader("📊 Analysis Overview")
            st.info(f"Analyzed {len(final_df)} articles with confidence ≥ {confidence_threshold}")

            col1, col2, col3, col4, col5 = st.columns(5)

            with col1:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Total Articles", len(final_df))
                st.markdown("</div>", unsafe_allow_html=True)

            with col2:
                positive = len(final_df[final_df["Sentiment"].str.contains("Positive")])
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("✅ Positive", positive)
                st.markdown("</div>", unsafe_allow_html=True)

            with col3:
                negative = len(final_df[final_df["Sentiment"].str.contains("Negative")])
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("❌ Negative", negative)
                st.markdown("</div>", unsafe_allow_html=True)

            with col4:
                neutral = len(final_df[final_df["Sentiment"].str.contains("Neutral")])
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("⚪ Neutral", neutral)
                st.markdown("</div>", unsafe_allow_html=True)

            with col5:
                avg_confidence = final_df["Confidence"].mean()
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("🎯 Avg Confidence", f"{avg_confidence:.2f}")
                st.markdown("</div>", unsafe_allow_html=True)

            if show_pie_chart or show_histogram or show_scatter:
                st.subheader("📈 Visual Analytics")
                viz_col1, viz_col2 = st.columns(2)
                with viz_col1:
                    if show_pie_chart:
                        st.plotly_chart(create_sentiment_pie_chart(final_df), use_container_width=True)
                    if show_histogram:
                        st.plotly_chart(create_confidence_histogram(final_df), use_container_width=True)
                with viz_col2:
                    if show_scatter:
                        st.plotly_chart(create_sentiment_confidence_scatter(final_df), use_container_width=True)

            st.subheader("📋 Detailed Analysis")
            display_columns = ["Title", "Sentiment", "Confidence", "Key Insight", "Link"]
            st.dataframe(final_df[display_columns], use_container_width=True, hide_index=True, height=400)

            st.subheader("📤 Export Results")
            export_col1, export_col2, export_col3 = st.columns(3)

            with export_col1:
                csv = final_df.to_csv(index=False)
                st.download_button(label="📥 Download CSV", data=csv, file_name=f"finsights_analysis_{search_term.replace(' ', '_')}.csv", mime="text/csv", use_container_width=True)

            with export_col2:
                pdf_filename = create_pdf_report(final_df, search_term, recommendation)
                with open(pdf_filename, "rb") as pdf_file:
                    PDFbyte = pdf_file.read()
                st.download_button(label="📄 Download PDF Report", data=PDFbyte, file_name=pdf_filename, mime="application/pdf", use_container_width=True)

            with export_col3:
                json_data = final_df.to_json(orient="records", indent=2)
                st.download_button(label="📊 Download JSON", data=json_data, file_name=f"finsights_analysis_{search_term.replace(' ', '_')}.json", mime="application/json", use_container_width=True)

            with st.expander("📖 Read Detailed News Analysis"):
                for _, article in final_df.iterrows():
                    sentiment_class = "positive-sentiment" if "Positive" in article["Sentiment"] else "negative-sentiment" if "Negative" in article["Sentiment"] else "neutral-sentiment"
                    title = article["Title"]
                    sent = article["Sentiment"]
                    conf = article["Confidence"]
                    insight = article.get("Key Insight", "")
                    link = article.get("Link", "")
                    st.markdown(
                        f"""
                    <div class="{sentiment_class}">
                        <h4>{title}</h4>
                        <p><strong>Sentiment:</strong> {sent} (Confidence: {conf})</p>
                        <p><strong>Key Insight:</strong> {insight}</p>
                        <p><a href="{link}" target="_blank">Read original article</a></p>
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )

if __name__ == "__main__":
    main()