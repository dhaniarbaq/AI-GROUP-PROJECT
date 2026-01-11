import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Fake News Detection System",
    layout="wide"
)

# ---------------- LOAD DATASET ----------------
@st.cache_data
def load_data():
    df = pd.read_csv("FakeNewsNet.csv")

    # Rename for consistency
    df = df.rename(columns={
        "title": "text",
        "real": "label"
    })

    # Drop missing values
    df = df.dropna(subset=["text", "label", "source_domain"])

    return df

df = load_data()

# ---------------- MODEL TRAINING ----------------
@st.cache_resource
def train_model(dataframe):

    preprocessor = ColumnTransformer(
        transformers=[
            ("text", TfidfVectorizer(
                lowercase=True,
                stop_words="english",
                ngram_range=(1,2),
                min_df=5,
                max_df=0.9,
                sublinear_tf=True
            ), "text"),

            ("domain", TfidfVectorizer(), "source_domain")
        ]
    )

    model = Pipeline([
        ("features", preprocessor),
        ("classifier", LogisticRegression(
            max_iter=4000,
            class_weight="balanced"
        ))
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        dataframe[["text", "source_domain"]],
        dataframe["label"],
        test_size=0.25,
        random_state=42,
        stratify=dataframe["label"]
    )

    model.fit(X_train, y_train)

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
st.sidebar.write("NLP + Source Domain Classification")
st.sidebar.text("Model: TF-IDF + Logistic Regression")
st.sidebar.success("System Operational")

# ---------------- TABS ----------------
tab_home, tab_analyzer, tab_dashboard, tab_reports = st.tabs(
    ["Home", "Analyzer", "Dashboard", "Reports"]
)

# ---------------- HOME ----------------
with tab_home:
    st.header("Welcome")
    st.write("""
    This system uses **23,000+ real news headlines** from FakeNewsNet.
    It analyzes both **text content** and **news source domain**
    to detect misinformation.
    """)
    st.info("This tool predicts likelihood — it does not fact-check.")

# ---------------- ANALYZER ----------------
with tab_analyzer:
    st.header("🧠 News Analyzer")

    text_input = st.text_area("Enter news headline", height=120)
    domain_input = st.text_input("Enter source domain (e.g. cnn.com, bbc.com)")

    if st.button("Analyze"):
        if not text_input.strip() or not domain_input.strip():
            st.warning("Please enter both headline and source domain.")
        else:
            X_input = pd.DataFrame([{
                "text": text_input,
                "source_domain": domain_input
            }])

            proba = model.predict_proba(X_input)[0]

            # FakeNewsNet: 1 = Real, 0 = Fake
            proba_real = proba[1]
            proba_fake = proba[0]

            threshold = 0.60

            if proba_fake > threshold:
                label = "Likely Fake News"
                confidence = proba_fake
            else:
                label = "Likely Genuine News"
                confidence = proba_real

            col1, col2 = st.columns(2)
            col1.metric("Prediction", label)
            col2.metric("Confidence", f"{confidence*100:.2f}%")

            st.progress(int(confidence * 100))

            st.session_state.history.append({
                "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Headline": text_input[:50] + "...",
                "Source": domain_input,
                "Prediction": label,
                "Confidence": round(confidence * 100, 2)
            })

# ---------------- DASHBOARD ----------------
with tab_dashboard:
    st.header("📊 Dashboard")

    if st.session_state.history:
        history_df = pd.DataFrame(st.session_state.history)
        history_df["Time"] = pd.to_datetime(history_df["Time"])

        st.subheader("Prediction Distribution")
        st.bar_chart(history_df["Prediction"].value_counts())

        st.subheader("Average Confidence")
        st.metric("Mean", f"{history_df['Confidence'].mean():.2f}%")

        st.subheader("Confidence Over Time")
        st.line_chart(history_df.set_index("Time")["Confidence"])
    else:
        st.info("Run some analyses first.")

# ---------------- REPORTS ----------------
with tab_reports:
    st.header("📄 Reports")

    if st.session_state.history:
        reports_df = pd.DataFrame(st.session_state.history)
        st.dataframe(reports_df, use_container_width=True)

        csv = reports_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV", csv, "fake_news_report.csv", "text/csv")
    else:
        st.info("No data yet.")
