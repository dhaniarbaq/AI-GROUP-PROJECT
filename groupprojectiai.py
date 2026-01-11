import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Fake News Detection System",
    layout="wide"
)

# ---------------- SAMPLE TRAINING DATA ----------------
data = {
    "text": [
        "Breaking news miracle cure discovered overnight",
        "Government announces new education policy today",
        "Shocking secret revealed about vaccines",
        "The prime minister met foreign leaders yesterday",
        "Experts warn phones will explode tonight",
        "Economic growth increased by three percent this year",
        "Celebrity claims salt water cures all diseases",
        "New tax regulation introduced for small businesses"
    ],
    "label": [1, 0, 1, 0, 1, 0, 1, 0]  # 1 = Fake, 0 = Genuine
}

df = pd.DataFrame(data)

# ---------------- NLP PIPELINE ----------------
model = Pipeline([
    ("tfidf", TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2)
    )),
    ("classifier", LogisticRegression(max_iter=1000))
])

X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"],
    test_size=0.2,
    random_state=42
)

model.fit(X_train, y_train)

# ---------------- SESSION STATE ----------------
if "history" not in st.session_state:
    st.session_state.history = []

# ---------------- SIDEBAR ----------------
st.sidebar.title("📰 Fake News Detection System")
st.sidebar.subheader("System Overview")
st.sidebar.write("Natural Language Processing based classification")
st.sidebar.text("Model: TF-IDF + Logistic Regression")
st.sidebar.text("Output: Probabilistic Prediction")

# ---------------- TABS ----------------
tabs = ["Home", "Analyzer", "Dashboard", "Reports"]
tab_home, tab_analyzer, tab_dashboard, tab_reports = st.tabs(tabs)

# ---------------- HOME ----------------
with tab_home:
    st.header("Welcome")
    st.write("""
    This system detects **potential fake news** using Natural Language Processing
    and Machine Learning techniques.

    The model learns linguistic patterns from text data and produces
    probability based predictions.
    """)

    st.info(
        "This application provides an early warning indicator. "
        "It does not verify factual accuracy."
    )

# ---------------- ANALYZER ----------------
with tab_analyzer:
    st.header("🧠 News Analyzer")

    text_input = st.text_area(
        "Enter news text for analysis",
        height=180
    )

    if st.button("🔍 Analyze News"):
        if text_input.strip() == "":
            st.warning("Please enter text before analysis.")
        else:
            prediction = model.predict([text_input])[0]
            probability = model.predict_proba([text_input])[0][prediction]

            label = "Likely Fake News" if prediction == 1 else "Likely Genuine News"

            st.subheader("Result")
            st.write(f"**Prediction:** {label}")
            st.progress(probability)
            st.write(f"**Confidence:** {probability * 100:.2f}%")

            # ---------------- Explanation ----------------
            if prediction == 1:
                explanation = (
                    "The text is classified as Likely Fake News because the NLP model "
                    "identified linguistic patterns commonly associated with misleading "
                    "or sensational reporting. TF-IDF weighting emphasized terms that "
                    "frequently appear in deceptive content."
                )
            else:
                explanation = (
                    "The text is classified as Likely Genuine News because the language "
                    "structure and vocabulary align with factual reporting styles. "
                    "The model detected neutral and informative linguistic features."
                )

            confidence_explanation = (
                f"The confidence value of {probability * 100:.2f}% represents the "
                "model’s certainty based on TF-IDF feature weighting and the learned "
                "decision boundary of the Logistic Regression classifier."
            )

            st.markdown("### 🔍 Justification of Result")
            st.write(explanation)
            st.write(confidence_explanation)

            st.caption(
                "This prediction is generated using NLP and Machine Learning. "
                "It is a probabilistic assessment, not factual verification."
            )

            # ---------------- Save to History ----------------
            st.session_state.history.append({
                "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Text": text_input[:80] + "...",
                "Prediction": label,
                "Confidence (%)": round(probability * 100, 2)
            })

# ---------------- DASHBOARD ----------------
with tab_dashboard:
    st.header("📊 Analysis Dashboard")

    if st.session_state.history:
        hist_df = pd.DataFrame(st.session_state.history)

        st.subheader("Prediction Distribution")
        st.bar_chart(hist_df["Prediction"].value_counts())

        st.subheader("Model Confidence Overview")
        st.metric(
            "Average Confidence Score",
            f"{hist_df['Confidence (%)'].mean():.2f}%"
        )
    else:
        st.info("No analysis data available yet.")

# ---------------- REPORTS ----------------
with tab_reports:
    st.header("📄 Reports")

    if st.session_state.history:
        report_df = pd.DataFrame(st.session_state.history)
        st.dataframe(report_df)

        csv = report_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download CSV Report",
            csv,
            "fake_news_report.csv",
            "text/csv"
        )
    else:
        st.info("Run analysis to generate reports.")
