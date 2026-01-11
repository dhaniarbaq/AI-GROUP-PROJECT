import streamlit as st
import pandas as pd
import numpy as np
import re
from datetime import datetime

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Fake News Detection", layout="wide")

# ---------------- SAMPLE DATASET ----------------
data = {
    "text": [
        "Breaking news miracle cure discovered overnight",
        "Government announces new education policy today",
        "Shocking secret revealed about vaccines",
        "The prime minister met foreign leaders yesterday",
        "Experts warn phones will explode tonight",
        "Economic growth increased by three percent this year"
    ],
    "label": [1, 0, 1, 0, 1, 0]  # 1 = Fake, 0 = Genuine
}

df = pd.DataFrame(data)

# ---------------- NLP PIPELINE ----------------
model = Pipeline([
    ("tfidf", TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2)
    )),
    ("classifier", LogisticRegression())
])

X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.2, random_state=42
)

model.fit(X_train, y_train)

# ---------------- SESSION STATE ----------------
if "history" not in st.session_state:
    st.session_state.history = []

# ---------------- SIDEBAR ----------------
st.sidebar.title("📰 Fake News Detection System")
st.sidebar.write("NLP based Machine Learning Model")
st.sidebar.text("Method: TF-IDF + Logistic Regression")

# ---------------- TABS ----------------
tabs = ["Home", "Analyzer", "Dashboard", "Reports"]
tab_home, tab_analyzer, tab_dashboard, tab_reports = st.tabs(tabs)

# ---------------- HOME ----------------
with tab_home:
    st.header("Fake News Detection using NLP")
    st.write("""
    This system applies **Natural Language Processing (NLP)** and
    **Machine Learning** to identify potential fake news.

    The model learns linguistic patterns from text using TF-IDF
    and predicts authenticity using Logistic Regression.
    """)

    st.info("This tool provides probabilistic predictions, not factual verification.")

# ---------------- ANALYZER ----------------
with tab_analyzer:
    st.header("🧠 News Analyzer")

    text_input = st.text_area(
        "Enter news text for analysis",
        height=180
    )

    if st.button("🔍 Analyze News"):
        if text_input.strip() == "":
            st.warning("Please enter text to analyze.")
        else:
            prediction = model.predict([text_input])[0]
            probability = model.predict_proba([text_input])[0][prediction]

            label = "Likely Fake News" if prediction == 1 else "Likely Genuine News"

            st.subheader("Result")
            st.write(f"**Prediction:** {label}")
            st.progress(probability)
            st.write(f"**Confidence:** {probability * 100:.2f}%")

            st.caption("Prediction based on learned linguistic patterns.")

            st.session_state.history.append({
                "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Text": text_input[:70] + "...",
                "Prediction": label,
                "Confidence (%)": round(probability * 100, 2)
            })

# ---------------- DASHBOARD ----------------
with tab_dashboard:
    st.header("📊 Analysis Dashboard")

    if st.session_state.history:
        df_hist = pd.DataFrame(st.session_state.history)

        st.bar_chart(df_hist["Prediction"].value_counts())
        st.metric("Average Confidence", f"{df_hist['Confidence (%)'].mean():.2f}%")
    else:
        st.info("No analysis data yet.")

# ---------------- REPORTS ----------------
with tab_reports:
    st.header("📄 Reports")

    if st.session_state.history:
        report_df = pd.DataFrame(st.session_state.history)
        st.dataframe(report_df)

        csv = report_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download CSV Report",
            csv,
            "fake_news_report.csv",
            "text/csv"
        )
    else:
        st.info("Run analysis to generate reports.")
