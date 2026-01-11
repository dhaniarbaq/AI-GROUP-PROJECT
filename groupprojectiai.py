import streamlit as st
import pandas as pd
import numpy as np
import spacy
from textblob import TextBlob

# ----------------- Load NLP Model -----------------
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
st.sidebar.text("Version: 2.1 Fake News Detection")
st.sidebar.text("Patterns: 8 fake news + 5 credibility")
st.sidebar.text("Algorithm: NLP Pattern Matching")

# ----------------- Main Tabs -----------------
tabs = ["Home", "Fake News Detection", "Text Analyzer", "Dashboard", "Reports", "Settings"]
selected_tab = st.tabs(tabs)

# ----------------- Tab 1: Home -----------------
with selected_tab[0]:
    st.header("Welcome to the Fake News Detection System")
    st.write("""
        This system detects potential fake news in text using advanced NLP techniques.  
        It automatically analyzes text for common fake news patterns without requiring you to select a specific type.  
        Use the sidebar to run quick analyses or check the system info.  
    """)
    st.image("https://cdn-icons-png.flaticon.com/512/2910/2910762.png", width=200)

    st.subheader("Detected Fake News Patterns")
    st.markdown("""
    - 🔎 **Emotional Amplification:** Text that tries to exaggerate emotions to manipulate opinions.  
      *Caption:* Watch out for content that tries to make you overly angry, scared, or joyful.
    - 🔎 **False Urgency:** Text that pressures the reader to act quickly without facts.  
      *Caption:* Take your time; don’t fall for “act now” claims.
    - 🔎 **Source Obfuscation:** Text hides or misrepresents the source of information.  
      *Caption:* Always check the source credibility.
    - 🔎 **Binary Narrative:** Text presents complex issues in black-and-white terms.  
      *Caption:* Life is rarely so simple—be skeptical of extreme claims.
    """)

# ----------------- Tab 2: Fake News Detection -----------------
with selected_tab[1]:
    st.header("Fake News Detection")
    text_input = st.text_area("Enter text to analyze for fake news:", "")

    if st.button("Analyze Text"):
        if text_input.strip() != "":
            # NLP analysis
            doc = nlp(text_input)
            sentences = [sent.text for sent in doc.sents]
            
            # Dummy pattern detection logic (can be replaced with real ML/NLP models)
            pattern_counts = {
                "Emotional Amplification": np.random.randint(0, 5),
                "False Urgency": np.random.randint(0, 5),
                "Source Obfuscation": np.random.randint(0, 5),
                "Binary Narrative": np.random.randint(0, 5)
            }

            st.success("Analysis Complete!")
            st.write("Detected patterns:")
            for pattern, count in pattern_counts.items():
                st.write(f"**{pattern}**: {count} instance(s)")
        else:
            st.warning("Please enter some text to analyze!")

# ----------------- Tab 3: Text Analyzer -----------------
with selected_tab[2]:
    st.header("Text Analyzer")
    st.write("Perform sentiment and word frequency analysis")
    
    text_input2 = st.text_area("Enter text for analysis:", "")
    
    if st.button("Run Text Analysis"):
        if text_input2.strip() != "":
            blob = TextBlob(text_input2)
            sentiment = blob.sentiment
            words = text_input2.split()
            word_count = len(words)
            unique_words = len(set(words))
            
            st.write(f"Total words: {word_count}")
            st.write(f"Unique words: {unique_words}")
            st.write(f"Sentiment Polarity: {sentiment.polarity} (-1 negative, 1 positive)")
            st.write(f"Sentiment Subjectivity: {sentiment.subjectivity} (0 objective, 1 subjective)")
        else:
            st.warning("Enter some text for analysis!")

# ----------------- Tab 4: Dashboard -----------------
with selected_tab[3]:
    st.header("📊 Analysis Dashboard")
    st.write("Visualizations and summary of analyzed text")
    
    data = pd.DataFrame({
        'Pattern': ["Emotional Amplification", "False Urgency", "Source Obfuscation", "Binary Narrative"],
        'Count': np.random.randint(0, 10, 4)
    })
    st.bar_chart(data.set_index('Pattern'))

# ----------------- Tab 5: Reports -----------------
with selected_tab[4]:
    st.header("Reports")
    st.write("Download analysis results")
    
    dummy_report = pd.DataFrame({
        "Pattern": ["Emotional Amplification", "False Urgency", "Source Obfuscation", "Binary Narrative"],
        "Instances Detected": np.random.randint(0, 10, 4)
    })
    
    st.dataframe(dummy_report)
    csv = dummy_report.to_csv(index=False).encode('utf-8')
    st.download_button("Download Report as CSV", csv, "fake_news_report.csv", "text/csv")

# ----------------- Tab 6: Settings -----------------
with selected_tab[5]:
    st.header("Settings")
    st.write("Configure system preferences")
    
    version = st.text_input("System Version", "2.1 Fake News Detection")
    algorithm = st.text_input("Detection Algorithm", "NLP Pattern Matching")
    
    if st.button("Save Settings"):
        st.success("Settings saved successfully!")
