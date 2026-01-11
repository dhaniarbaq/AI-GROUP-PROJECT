import streamlit as st
import pandas as pd
import numpy as np

# ---------------- NLP Setup ----------------
from textblob import TextBlob
import spacy
from spacy.cli import download as spacy_download

# Download spaCy model if not present
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    st.info("Downloading spaCy English model...")
    spacy_download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# ----------------- Sidebar -----------------
st.sidebar.title("📰 Fake News Detection System")

# Quick Analysis Button
st.sidebar.subheader("⚡ Quick Analysis")
if st.sidebar.button("Run Quick Analysis"):
    st.sidebar.success("Quick analysis triggered!")

# Analysis Dashboard Section
st.sidebar.subheader("📊 Analysis Dashboard")
st.sidebar.info("No analyses yet. Start by analyzing text above!")

# System Info Section
st.sidebar.subheader("ℹ️ System Info")
st.sidebar.text("Version: 1.0 Fake News Detection")
st.sidebar.text("Algorithm: NLP + Sentiment + Heuristics")
st.sidebar.text("Patterns: Automatic detection")

# ----------------- Main Tabs -----------------
tabs = ["Home", "Analyzer", "Dashboard", "Reports", "Settings"]
selected_tab = st.tabs(tabs)

# ----- Tab 1: Home -----
with selected_tab[0]:
    st.header("Welcome to the Fake News Detection System")
    st.write("""
        This system automatically detects whether a piece of text is likely to be **fake news**.
        You don’t need to choose patterns manually.  
        Simply enter text in the Analyzer tab to get results.  
    """)
    
    st.subheader("How the system detects fake news")
    st.write("""
    1. **Emotional Amplification:** Detects exaggerated emotions or sensational words.  
    2. **False Urgency:** Identifies clickbait or time-sensitive manipulations.  
    3. **Source Obfuscation:** Checks for vague or suspicious sources.  
    4. **Binary Narrative:** Looks for extreme ‘us vs them’ statements.  
    """)
    
    st.image("https://cdn-icons-png.flaticon.com/512/2910/2910762.png", width=200)

# ----- Tab 2: Analyzer -----
with selected_tab[1]:
    st.header("Fake News Analyzer")
    text_input = st.text_area("Enter text to analyze for fake news:", "")
    
    if st.button("Analyze Text"):
        if text_input.strip() != "":
            # ---- NLP Detection Logic ----
            doc = nlp(text_input)
            blob = TextBlob(text_input)
            
            sentiment_score = blob.sentiment.polarity
            fake_news_score = 0
            
            # Heuristic rules for fake news
            if any(word.lower() in text_input.lower() for word in ["shocking", "unbelievable", "exclusive", "alert"]):
                fake_news_score += 1
            if sentiment_score > 0.5 or sentiment_score < -0.5:
                fake_news_score += 1
            if len(doc.ents) == 0:
                fake_news_score += 1
            
            # Determine result
            if fake_news_score >= 2:
                result = "⚠️ Likely Fake News"
            else:
                result = "✅ Likely Authentic"
            
            st.success(result)
            st.write(f"Fake News Score: {fake_news_score} / 3")
        else:
            st.warning("Please enter some text to analyze!")

# ----- Tab 3: Dashboard -----
with selected_tab[2]:
    st.header("📊 Analysis Dashboard")
    st.write("Visualizations and analysis summary (Dummy Data)")
    
    data = pd.DataFrame({
        'Category': ["Likely Fake", "Likely Authentic"],
        'Count': np.random.randint(0, 10, 2)
    })
    
    st.bar_chart(data.set_index('Category'))

# ----- Tab 4: Reports -----
with selected_tab[3]:
    st.header("Reports")
    st.write("Download previous analysis results (Dummy Data)")
    
    dummy_report = pd.DataFrame({
        "Text Sample": ["Sample 1", "Sample 2", "Sample 3", "Sample 4"],
        "Result": ["Likely Fake", "Likely Authentic", "Likely Fake", "Likely Authentic"],
        "Score": np.random.randint(0, 4, 4)
    })
    
    st.dataframe(dummy_report)
    
    csv = dummy_report.to_csv(index=False).encode('utf-8')
    st.download_button("Download Report as CSV", csv, "fake_news_report.csv", "text/csv")

# ----- Tab 5: Settings -----
with selected_tab[4]:
    st.header("Settings")
    st.write("Configure system preferences")
    
    version = st.text_input("System Version", "1.0 Fake News Detection")
    algorithm = st.text_input("Detection Algorithm", "NLP + Sentiment + Heuristics")
    
    if st.button("Save Settings"):
        st.success("Settings saved successfully!")
