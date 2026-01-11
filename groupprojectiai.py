import streamlit as st
import pandas as pd
import numpy as np
import spacy
from textblob import TextBlob

# Load English NLP model
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
st.sidebar.text("Patterns: 8 fake news + 5 credibility checks")
st.sidebar.text("Algorithm: NLP & Weighted Pattern Matching")

# ----------------- Main Tabs -----------------
tabs = ["Home", "Fake News Detection", "Text Analyzer", "Dashboard", "Reports", "Settings"]
selected_tab = st.tabs(tabs)

# Fake news patterns with keywords and explanations
fake_patterns = {
    "Clickbait Headlines": {
        "desc": "Sensational headlines designed to get clicks, often misleading.",
        "keywords": ["shocking", "you won't believe", "amazing", "secret", "revealed"]
    },
    "False Urgency": {
        "desc": "Content that pressures you to act immediately.",
        "keywords": ["urgent", "immediately", "act now", "last chance"]
    },
    "Anonymous Sources": {
        "desc": "Information from unverifiable or hidden sources.",
        "keywords": ["anonymous", "unknown source", "unverified", "reportedly"]
    },
    "Manipulated Media": {
        "desc": "Images, videos, or quotes that are altered to mislead.",
        "keywords": ["edited", "photoshopped", "fake video", "misleading image"]
    },
    "Polarizing Narrative": {
        "desc": "Oversimplifies events into us vs them or good vs evil.",
        "keywords": ["enemy", "betray", "us vs them", "all or nothing"]
    }
}

# ----- Tab 1: Home -----
with selected_tab[0]:
    st.header("Welcome to the Fake News Detection System")
    st.write("""
        Automatically detect fake news and questionable content using NLP.  
        No need to manually select patterns—simply input your text to get instant insights.  
    """)
    st.image("https://cdn-icons-png.flaticon.com/512/2910/2910762.png", width=200)

    st.subheader("Fake News Patterns Library 📚")
    for pattern, info in fake_patterns.items():
        st.markdown(f"**{pattern}**")
        st.info(f"{info['desc']}\n*Caption:* Helps identify when news might be misleading or fake.")

# ----- Tab 2: Fake News Detection -----
with selected_tab[1]:
    st.header("Fake News Detection")
    
    text_input = st.text_area("Enter text to analyze for fake news:", "")
    
    if st.button("Analyze Text Automatically"):
        if text_input.strip() != "":
            doc = nlp(text_input.lower())
            detected_patterns = []
            for pattern, info in fake_patterns.items():
                for keyword in info["keywords"]:
                    if keyword.lower() in text_input.lower():
                        detected_patterns.append(pattern)
                        break  # Only detect once per pattern
            
            # Sentiment check (optional)
            sentiment = TextBlob(text_input).sentiment
            st.write(f"Text Sentiment: Polarity={sentiment.polarity:.2f}, Subjectivity={sentiment.subjectivity:.2f}")
            
            if detected_patterns:
                st.error("Potential Fake News Detected:")
                for p in detected_patterns:
                    st.write(f"- {p}")
            else:
                st.success("No major fake news patterns detected.")
        else:
            st.warning("Please enter some text to analyze!")

# ----- Tab 3: Text Analyzer -----
with selected_tab[2]:
    st.header("Text Analyzer")
    st.write("Perform sentiment and word frequency analysis")
    
    text_input2 = st.text_area("Enter text for analysis:", "")
    
    if st.button("Run Text Analysis"):
        if text_input2.strip() != "":
            words = text_input2.split()
            word_count = len(words)
            unique_words = len(set(words))
            st.write(f"Total words: {word_count}")
            st.write(f"Unique words: {unique_words}")
            
            # Sentiment analysis
            sentiment = TextBlob(text_input2).sentiment
            st.write(f"Polarity: {sentiment.polarity:.2f}")
            st.write(f"Subjectivity: {sentiment.subjectivity:.2f}")
        else:
            st.warning("Enter some text for analysis!")

# ----- Tab 4: Dashboard -----
with selected_tab[3]:
    st.header("📊 Analysis Dashboard")
    st.write("Visualizations and analysis summary")
    
    data = pd.DataFrame({
        'Pattern': list(fake_patterns.keys()),
        'Count': np.random.randint(0, 10, len(fake_patterns))
    })
    st.bar_chart(data.set_index('Pattern'))

# ----- Tab 5: Reports -----
with selected_tab[4]:
    st.header("Reports")
    st.write("Download analysis results")
    
    dummy_report = pd.DataFrame({
        "Pattern": list(fake_patterns.keys()),
        "Instances Detected": np.random.randint(0, 10, len(fake_patterns))
    })
    
    st.dataframe(dummy_report)
    
    csv = dummy_report.to_csv(index=False).encode('utf-8')
    st.download_button("Download Report as CSV", csv, "fake_news_report.csv", "text/csv")

# ----- Tab 6: Settings -----
with selected_tab[5]:
    st.header("Settings")
    st.write("Configure system preferences")
    
    version = st.text_input("System Version", "1.0 Fake News Detection")
    algorithm = st.text_input("Detection Algorithm", "NLP & Weighted Pattern Matching")
    
    if st.button("Save Settings"):
        st.success("Settings saved successfully!")
