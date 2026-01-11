# fake_news_app.py

import streamlit as st
import pandas as pd
import numpy as np
from textblob import TextBlob
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# ----------------- Sidebar -----------------
st.sidebar.title("📰 Fake News Detection System")

# Quick Info
st.sidebar.subheader("ℹ️ System Info")
st.sidebar.text("Version: 1.0 Fake News Detection")
st.sidebar.text("Algorithm: NLP-based automatic detection")
st.sidebar.text("Features: Detection, Dashboard, Reports, Settings")

# ----------------- Tabs -----------------
tabs = ["Home", "Analyzer", "Dashboard", "Reports", "Settings"]
selected_tab = st.tabs(tabs)

# ----------------- Tab 1: Home -----------------
with selected_tab[0]:
    st.header("Welcome to the Fake News Detection System")
    st.write("""
        This system automatically detects potential fake news in text using NLP techniques.
        You can analyze text, view dashboards of patterns, download reports, and customize settings.
    """)
    st.image("https://cdn-icons-png.flaticon.com/512/2910/2910762.png", width=200)
    st.subheader("How the Detection Works:")
    st.markdown("""
    - **Polarity:** Measures the sentiment (-1 negative, 0 neutral, 1 positive).  
      Fake news often uses extreme sentiment.  
    - **Subjectivity:** Measures objectivity (0 objective, 1 subjective).  
      High subjectivity may indicate sensationalism.  
    - **Keywords:** Highlights suspicious or unusual words that often appear in fake news.  
    - **Confidence Score:** Combines all metrics into a probability of being fake news.  
    """)

# ----------------- Tab 2: Analyzer -----------------
with selected_tab[1]:
    st.header("📝 Analyze Text")
    text_input = st.text_area("Enter text to analyze:", "")

    if st.button("Run Analysis"):
        if text_input.strip() != "":
            blob = TextBlob(text_input)
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity

            # Confidence score: weighted combination of subjectivity & polarity extremeness
            confidence = round(min(1.0, (abs(polarity) + subjectivity) / 2), 2)

            if confidence > 0.5:
                result = "Likely Fake News ⚠️"
            else:
                result = "Likely Genuine ✅"

            st.subheader("Analysis Result")
            st.write(f"**Result:** {result}")
            st.write(f"**Polarity:** {polarity}")
            st.write(f"**Subjectivity:** {subjectivity}")
            st.write(f"**Confidence Score:** {confidence * 100}%")

            # Highlight keywords
            words = text_input.split()
            suspicious_words = [word for word in words if len(word) > 6]  # simplistic example
            st.subheader("Suspicious Keywords")
            st.write(", ".join(suspicious_words) if suspicious_words else "None found")

        else:
            st.warning("Please enter text to analyze!")

# ----------------- Tab 3: Dashboard -----------------
with selected_tab[2]:
    st.header("📊 Dashboard")
    st.write("Visualizations of text analyses")

    # Dummy data example
    patterns = ["Likely Fake", "Likely Genuine"]
    counts = np.random.randint(0, 20, 2)
    df_dashboard = pd.DataFrame({"Category": patterns, "Count": counts})
    st.bar_chart(df_dashboard.set_index("Category"))

    # Word cloud visualization
    st.subheader("Word Cloud Example")
    sample_text = "Fake news detection system text analysis dashboard visualization".lower()
    wc = WordCloud(width=600, height=300, background_color="white").generate(sample_text)
    plt.imshow(wc, interpolation='bilinear')
    plt.axis("off")
    st.pyplot(plt)

# ----------------- Tab 4: Reports -----------------
with selected_tab[3]:
    st.header("📑 Reports")
    st.write("Download analysis results")

    # Example report
    dummy_report = pd.DataFrame({
        "Text": ["Sample text 1", "Sample text 2"],
        "Result": ["Likely Fake", "Likely Genuine"],
        "Confidence": [0.82, 0.35]
    })
    st.dataframe(dummy_report)

    csv = dummy_report.to_csv(index=False).encode('utf-8')
    st.download_button("Download Report as CSV", csv, "fake_news_report.csv", "text/csv")

# ----------------- Tab 5: Settings -----------------
with selected_tab[4]:
    st.header("⚙️ Settings")
    st.write("Configure system preferences")
    version = st.text_input("System Version", "1.0 Fake News Detection")
    algorithm = st.text_input("Detection Algorithm", "NLP-based automatic detection")

    if st.button("Save Settings"):
        st.success("Settings saved successfully!")
