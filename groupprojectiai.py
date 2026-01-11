import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Fake News Detection System",
    layout="wide"
)

# ---------------- TRAINING DATA ----------------
# IMPORTANT
# For better performance you should replace this with a much larger
# dataset of both real and fake news articles.
# Load from a CSV if available: pd.read_csv("your_dataset.csv")

data = {
    "text": [
        "Breaking news miracle cure discovered overnight",
        "Government announces new education policy today",
        "Shocking secret revealed about vaccines",
        "Prime minister met foreign leaders yesterday",
        "Experts warn phones will explode tonight",
        "Economic growth increased by three percent this year",
        "Celebrity claims salt water cures all diseases",
        "New tax regulation introduced for small businesses",
        "Renowned scientist explains climate change findings",
        "Local hospital invests in new cancer treatment tech"
    ],
    "label": [1, 0, 1, 0, 1, 0, 1, 0, 0, 0]  # 1 = Fake, 0 = Genuine
}

df = pd.DataFrame(data)

# we can balance if needed
# df = pd.concat([df[df.label == 0].sample(50), df[df.label == 1].sample(50)])

# ---------------- MODEL TRAINING ----------------
@st.cache_resource
def train_model(dataframe):
    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 3),
            min_df=2,
            max_df=0.85
        )),
        ("classifier", LogisticRegression(
            max_iter=3000,
            class_weight="balanced"
        ))
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        dataframe["text"],
        dataframe["label"],
        test_size=0.25,
        random_state=42,
        stratify=dataframe["label"]
    )

    model.fit(X_train, y_train)

    # Optional evaluation report in logs
    preds = model.predict(X_test)
    print(classification_report(y_test, preds))

    return model

model = train_model(df)

# ---------------- SESSION STATE ----------------
if "history" not in st.session_state:
    st.session_state.history = []

# ---------------- SIDEBAR ----------------
st.sidebar.title("📰 Fake News Detection System")
st.sidebar.subheader("System Overview")
st.sidebar.write("Natural Language Processing classification")
st.sidebar.text("Model: TF-IDF + Logistic Regression")
st.sidebar.text("Prediction Type: Probabilistic with Thresholding")
st.sidebar.success("System Status Operational")
st.sidebar.caption("Version 2.0 | 2026")

# ---------------- TABS ----------------
tab_home, tab_analyzer, tab_dashboard, tab_reports = st.tabs(
    ["Home", "Analyzer", "Dashboard", "Reports"]
)

with tab_home:
    st.header("Welcome")
    st.write(
        """
        This improved system helps you check if a piece of news is likely fake.
        It uses better features and a balanced model so it is more reliable
        when you feed it real published news.
        """
    )
    st.info("This model is a supporting tool and does not perform factual verification.")

with tab_analyzer:
    st.header("🧠 News Analyzer")

    text_input = st.text_area(
        "Enter news text for analysis",
        height=200
    )

    if st.button("Analyze"):
        if not text_input.strip():
            st.warning("Please enter some text before analysis.")
        else:
            proba = model.predict_proba([text_input])[0]
            proba_fake = proba[1]
            proba_real = proba[0]

            # smarter threshold
            threshold = 0.60

            if proba_fake > threshold and proba_fake > proba_real:
                prediction = 1
                label = "Likely Fake News"
                confidence = proba_fake
            else:
                prediction = 0
                label = "Likely Genuine News"
                confidence = proba_real

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Prediction", label)
            with col2:
                st.metric("Confidence", f"{confidence * 100:.2f}%")

            st.progress(int(confidence * 100))

            # Explainability features
            tfidf = model.named_steps["tfidf"]
            classifier = model.named_steps["classifier"]

            feature_names = tfidf.get_feature_names_out()
            coefs = classifier.coef_[0]

            sorted_features = sorted(
                zip(feature_names, coefs),
                key=lambda x: abs(x[1]),
                reverse=True
            )[:8]

            features_str = ", ".join([f"{w}" for w, c in sorted_features])

            st.markdown("### Why the model thinks so")
            st.write(
                "Important keywords influencing the decision include:"
            )
            st.write(features_str)

            st.caption(
                "Confidence is based on model’s internal text pattern recognition"
            )

            st.session_state.history.append({
                "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Text Sample": text_input[:50] + "...",
                "Prediction": label,
                "Confidence": round(confidence * 100, 2)
            })

with tab_dashboard:
    st.header("📊 Dashboard Summary")

    if st.session_state.history:
        history_df = pd.DataFrame(st.session_state.history)
        history_df["Time"] = pd.to_datetime(history_df["Time"])

        st.subheader("Predictions Overview")
        st.bar_chart(history_df["Prediction"].value_counts())

        st.subheader("Mean Confidence")
        st.metric("Average Confidence", f"{history_df['Confidence'].mean():.2f}%")

        st.subheader("Confidence Trend")
        st.line_chart(history_df.set_index("Time")["Confidence"])
    else:
        st.info("No analyses run yet.")

with tab_reports:
    st.header("📄 Reports")
    if st.session_state.history:
        reports_df = pd.DataFrame(st.session_state.history)
        st.dataframe(reports_df, use_container_width=True)

        csv = reports_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download Report as CSV", csv, "fake_news_report.csv", "text/csv")
    else:
        st.info("Please analyze news to generate reports.")
