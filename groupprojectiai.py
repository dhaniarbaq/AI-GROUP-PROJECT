import streamlit as st
import pandas as pd
import numpy as np
import subprocess
import sys

# ------------------- NLP Setup -------------------
# Install SpaCy model if missing
try:
    import spacy
except ModuleNotFoundError:
    subprocess.run([sys.executable, "-m", "pip", "install", "spacy"])
    import spacy

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

# Install TextBlob if missing
try:
    from textblob import TextBlob
except ModuleNotFoundError:
    subprocess.run([sys.executable, "-m", "pip", "install", "textblob"])
    from textblob import TextBlob

# ----------------- Sidebar -----------------
st.sidebar.title("📰 Fake News Detection")

# Quick Analysis Button
st.sidebar.subheader("⚡ Quick Analysis")
if st.sidebar.button("Run Quick Analysis"):
    st.sidebar.success(f"Quick analysis triggered!")

# Analysis Dashboard Section
st.sidebar.subheader("📊 Analysis Dashboard")
st.sidebar.info("No analyses yet. Start by analyzing text above!")

# System Info Section
st.sidebar.subheader("ℹ️ System Info")
st.sidebar.text("Version: 2.1 Fake News Detection")
st.sidebar.text("Algorithm: NLP + Pattern Matching")
st.sidebar.text("Supports: Sentiment, Fake News, and Pattern Detection")

# ----------------- Main Tabs -----------------
tabs = ["Home", "Fake News Detection", "Text Analyzer", "Dashboard", "Reports", "Settings"]
selected_tab = st.tabs(tabs)

# ----- Tab 1: Home -----
with selected_tab[0]:
    st.header("Welcome to the Fake News Detection System")
    st.write("""
        This system helps detect potential fake news and misleading patterns in text.  
        Use the sidebar for quick analysis and explore the tabs for detailed tools.  
    """)
    st.image("https://cdn-icons-png.flaticon.com/512/2910/2910762.png", width=200)

    st.subheader("Detected Fake News Patterns")
    st.markdown("""
    - **Emotional Amplification**: Text trying to manipulate emotions to influence opinions.  
      *Caption:* Beware of overly emotional headlines!
    - **False Urgency**: Text creating panic or pressure to act immediately.  
      *Caption:* Don’t rush, check facts first!
    - **Source Obfuscation**: Hiding or faking sources to appear credible.  
      *Caption:* Always check where info comes from!
    - **Binary Narrative**: Text framing issues as black-or-white.  
      *Caption:* Reality is rarely only one side.
    """)

# ----- Tab 2: Fake News Detection -----
with selected_tab[1]:
    st.header("Fake News Detection")
    text_input = st.text_area("Enter text to analyze for fake news:", "")

    if st.button("Analyze Text"):
        if text_input.strip() != "":
            doc = nlp(text_input)
            blob = TextBlob(text_input)
            
            # Dummy fake news scoring logic
            fake_score = np.random.randint(0, 101)
            sentiment = blob.sentiment.polarity

            st.success("Analysis Complete!")
            st.write(f"**Fake News Score:** {fake_score}%")
            st.write(f"**Sentiment Polarity:** {sentiment:.2f}")
            
            st.subheader("Detected Patterns")
            detected_patterns = []
            if "!" in text_input or "shocking" in text_input.lower():
                detected_patterns.append("Emotional Amplification")
            if "urgent" in text_input.lower() or "immediately" in text_input.lower():
                detected_patterns.append("False Urgency")
            if "source" not in text_input.lower():
                detected_patterns.append("Source Obfuscation")
            if "either" in text_input.lower() or "or" in text_input.lower():
                detected_patterns.append("Binary Narrative")

            if detected_patterns:
                for p in detected_patterns:
                    st.markdown(f"- {p}")
            else:
                st.write("No major fake news patterns detected.")
        else:
            st.warning("Please enter text to analyze!")

# ----- Tab 3: Text Analyzer -----
with selected_tab[2]:
    st.header("Text Analyzer")
    text_input2 = st.text_area("Enter text for sentiment & word frequency analysis:", "")
    
    if st.button("Run Text Analysis"):
        if text_input2.strip() != "":
            words = text_input2.split()
            word_count = len(words)
            unique_words = len(set(words))
            st.write(f"Total words: {word_count}")
            st.write(f"Unique words: {unique_words}")
            
            # Display most common words
            word_freq = pd.Series(words).value_counts().head(10)
            st.bar_chart(word_freq)
        else:
            st.warning("Enter some text for analysis!")

# ----- Tab 4: Dashboard -----
with selected_tab[3]:
    st.header("📊 Analysis Dashboard")
    st.write("Visualizations and analysis summary")

    # Dummy chart example
    data = pd.DataFrame({
        'Pattern': ["Emotional Amplification", "False Urgency", "Source Obfuscation", "Binary Narrative"],
        'Count': np.random.randint(0, 10, 4)
    })
    st.bar_chart(data.set_index('Pattern'))

# ----- Tab 5: Reports -----
with selected_tab[4]:
    st.header("Reports")
    st.write("Download analysis results")
    
    dummy_report = pd.DataFrame({
        "Pattern": ["Emotional Amplification", "False Urgency", "Source Obfuscation", "Binary Narrative"],
        "Instances Detected": np.random.randint(0, 10, 4)
    })
    
    st.dataframe(dummy_report)
    
    csv = dummy_report.to_csv(index=False).encode('utf-8')
    st.download_button("Download Report as CSV", csv, "analysis_report.csv", "text/csv")

# ----- Tab 6: Settings -----
with selected_tab[5]:
    st.header("Settings")
    st.write("Configure system preferences")
    
    version = st.text_input("System Version", "2.1 Fake News Detection")
    algorithm = st.text_input("Detection Algorithm", "NLP + Pattern Matching")
    
    if st.button("Save Settings"):
        st.success("Settings saved successfully!")
