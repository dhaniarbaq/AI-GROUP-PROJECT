import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Fake News Detection System", layout="wide")

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    df = pd.read_csv("FakeNewsNet.csv")
    df = df.rename(columns={"title":"text", "real":"label"})
    df = df.dropna(subset=["text","label","source_domain"])
    return df

df = load_data()

# ---------------- MODEL ----------------
@st.cache_resource
def train_model(data):

    preprocessor = ColumnTransformer([
        ("text", TfidfVectorizer(
            stop_words="english",
            ngram_range=(1,2),
            min_df=5,
            max_df=0.9
        ), "text"),

        ("domain", TfidfVectorizer(), "source_domain")
    ])

    model = Pipeline([
        ("features", preprocessor),
        ("clf", LogisticRegression(
            max_iter=4000,
            class_weight="balanced"
        ))
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        data[["text","source_domain"]],
        data["label"],
        test_size=0.25,
        random_state=42,
        stratify=data["label"]
    )

    model.fit(X_train, y_train)
    return model

model = train_model(df)

# ---------------- CONFIDENCE INTERPRETATION ----------------
def interpret_confidence(prob):
    if prob > 0.85:
        return "Very High"
    elif prob > 0.70:
        return "High"
    elif prob > 0.55:
        return "Moderate"
    else:
        return "Low"

# ---------------- STREAMLIT UI ----------------
st.title("📰 Fake News Detection with Explainable AI")

headline = st.text_area("Enter news headline")
domain = st.text_input("Enter source domain (e.g. bbc.com, cnn.com)")

if st.button("Analyze"):

    X = pd.DataFrame([{"text":headline, "source_domain":domain}])
    proba = model.predict_proba(X)[0]

    # FakeNewsNet: 1 = Real, 0 = Fake
    p_real = proba[1]
    p_fake = proba[0]

    if p_fake > 0.60:
        label = "Fake News"
        confidence = p_fake
    else:
        label = "Genuine News"
        confidence = p_real

    level = interpret_confidence(confidence)

    # ---------------- RESULTS ----------------
    st.subheader("Prediction Result")
    st.metric("Classification", label)
    st.metric("Confidence", f"{confidence*100:.2f}%")
    st.progress(int(confidence*100))
    st.write(f"Confidence Level: **{level}**")

    # ---------------- EXPLANATION ----------------
    tfidf_text = model.named_steps["features"].transformers_[0][1]
    clf = model.named_steps["clf"]

    feature_names = tfidf_text.get_feature_names_out()
    coefs = clf.coef_[0][:len(feature_names)]

    top_words = sorted(zip(feature_names, coefs), key=lambda x: abs(x[1]), reverse=True)[:6]

    st.subheader("Why the model made this decision")

    for word, weight in top_words:
        if weight > 0:
            st.write(f"• **'{word}'** pushes the model toward **Real News**")
        else:
            st.write(f"• **'{word}'** pushes the model toward **Fake News**")

    # ---------------- DOMAIN EFFECT ----------------
    st.subheader("Source credibility analysis")

    domain_weight = model.named_steps["features"].transformers_[1][1]
    if domain.lower() in domain_weight.get_feature_names_out():
        st.write(f"The source **{domain}** has learned reliability patterns from historical data.")
    else:
        st.write(f"The source **{domain}** is unknown, so the model relies more on wording.")

    # ---------------- DECISION INTERPRETATION ----------------
    st.subheader("Final Explanation")

    if label == "Fake News":
        st.write(
            f"The system classified this as **Fake News** because the writing style and source resemble patterns "
            f"found in known misinformation. With a confidence of **{confidence*100:.2f}%**, "
            f"the prediction reliability is **{level}**."
        )
    else:
        st.write(
            f"The system classified this as **Genuine News** because the language and source match those from "
            f"trusted news articles. With a confidence of **{confidence*100:.2f}%**, "
            f"the prediction reliability is **{level}**."
        )
