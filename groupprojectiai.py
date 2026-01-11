import streamlit as st
import pandas as pd
import numpy as np
from textblob import TextBlob
import spacy

# ---------------- NLP Setup ----------------
# Use lightweight blank English model to avoid download delays
nlp = spacy.blank("en")

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
st.sidebar.text("Algorithm: NLP + Sentiment Heuristics")

# ----------------- Main Tabs -----------------
tabs = ["Home", "Analyzer", "Dashboard", "Reports", "Settings"]
selected_tab = st.tabs(tabs)

# ----- Tab 1: Home -----
with selected_tab[0]:
    st.header("Welcome to the Fake News Detection System")
    st.write("""
        This system automatically detects potential fake news and analyzes text for credibility and sentiment.
        Use the Analyzer tab to input text and get results.
        Navigate through the Dashboard and Reports tabs to explore your analysis.
    """)
    st.image("https://cdn-icons-png.flaticon.com/512/2910/2910762.png", width=200)

# ----- Tab 2: Analyzer -----
with selected_tab[1]:
    st.header("Fake News Analyzer")
    text_input = st.text_area("Enter text to analyze:", "")

    if st.button("Analyze Text"):
        if text_input.strip() != "":
            doc = nlp(text_input)
            # Simple heuristic: if sentiment polarity is extremely negative or positive with strong claims -> fake news flag
            sentiment = TextBlob(text_input).sentiment
            polarity = sentiment.polarity
            subjectivity = sentiment.subjectivity

            # Fake news heuristic (can improve with ML models later)
            if abs(polarity) > 0.6 and subjectivity > 0.5:
                result = "⚠️ Likely Fake News"
            else:
                result = "✅ Likely Genuine News"

            st.subheader("Analysis Result")
            st.write(result)
            st.write(f"**Sentiment Polarity:** {polarity:.2f}")
            st.write(f"**Subjectivity:** {subjectivity:.2f}")

            # Word count and keyword check
            words = [token.text for token in doc]
            st.write(f"**Total words:** {len(words)}")
            st.write(f"**Unique words:** {len(set(words))}")
        else:
            st.warning("Please enter some text to analyze!")

# ----- Tab 3: Dashboard -----
with selected_tab[2]:
    st.header("📊 Analysis Dashboard")
    st.write("Visualizations and summary of analyzed texts")

    # Dummy chart data (replace with real analytics in production)
    patterns = ["Likely Fake News", "Likely Genuine News"]
    counts = np.random.randint(0, 10, 2)
    data = pd.DataFrame({"Result": patterns, "Count": counts})
    st.bar_chart(data.set_index("Result"))

# ----- Tab 4: Reports -----
with selected_tab[3]:
    st.header("Reports")
    st.write("Download previous analysis results")

    # Dummy report
    dummy_report = pd.DataFrame({
        "Text Sample": ["Example 1", "Example 2", "Example 3"],
        "Result": np.random.choice(["Likely Fake News", "Likely Genuine News"], 3),
        "Polarity": np.round(np.random.uniform(-1, 1, 3), 2),
        "Subjectivity": np.round(np.random.uniform(0, 1, 3), 2)
    })

    st.dataframe(dummy_report)
    csv = dummy_report.to_csv(index=False).encode('utf-8')
    st.download_button("Download Report as CSV", csv, "fake_news_report.csv", "text/csv")

# ----- Tab 5: Settings -----
with selected_tab[4]:
    st.header("Settings")
    st.write("Configure system preferences")
    version = st.text_input("System Version", "1.0 Fake News Detection")
    algorithm = st.text_input("Detection Algorithm", "NLP + Sentiment Heuristics")

    if st.button("Save Settings"):
        st.success("Settings saved successfully!")
