# ===============================
# FAKE NEWS DETECTION SYSTEM
# AI-Based News Verification
# ===============================

import streamlit as st
import pandas as pd
import numpy as np
import re
import requests
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib
import random
import time

# -------------------------------
# APP CONFIGURATION
# -------------------------------
st.set_page_config(
    page_title="AI Fake News Detector",
    page_icon="📰",
    layout="wide"
)

# -------------------------------
# SAMPLE DATASET (for demonstration)
# -------------------------------
# This is minimal; in practice, replace with larger dataset
data = pd.DataFrame({
    "text": [
        "BREAKING!!! Government hides shocking truth about vaccines!",
        "Study shows eating apples daily improves health.",
        "Experts claim that aliens live among us.",
        "Local school implements new reading program successfully."
    ],
    "label": ["fake", "real", "fake", "real"]
})

# -------------------------------
# TRAIN SIMPLE MODELS
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(data["text"], data["label"], test_size=0.2, random_state=42)

# Naive Bayes Pipeline
nb_model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("nb", MultinomialNB())
])
nb_model.fit(X_train, y_train)

# Logistic Regression Pipeline
lr_model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("lr", LogisticRegression())
])
lr_model.fit(X_train, y_train)

# -------------------------------
# SYSTEM OVERVIEW
# -------------------------------
st.markdown("# 📰 AI Fake News Detection System")
st.markdown(
    """
This system uses **machine learning models** (Naive Bayes and Logistic Regression) 
to predict whether a news headline or article is **real or fake**.
- Enter a headline or paste a website URL to analyze.
- The system fetches the text (if URL) and predicts using both models.
- A sample fake news headline is included for testing.
"""
)

# -------------------------------
# INPUT SECTION
# -------------------------------
st.markdown("## Enter Headline or URL")
input_type = st.radio("Select input type:", ["Text", "URL"])

user_input = ""
if input_type == "Text":
    user_input = st.text_area("Enter the news headline or article text:", height=150)
elif input_type == "URL":
    url = st.text_input("Enter the website URL:")
    if url:
        try:
            response = requests.get(url, timeout=5)
            soup = BeautifulSoup(response.content, "lxml")
            paragraphs = soup.find_all("p")
            user_input = " ".join([p.get_text() for p in paragraphs])
        except Exception as e:
            st.error(f"⚠️ Failed to fetch URL content: {e}")

# -------------------------------
# SAMPLE HEADLINE
# -------------------------------
if st.button("Load Sample Fake News"):
    user_input = "SHOCKING BREAKING NEWS!!! Government hides the real truth about vaccines!"
    st.text_area("News text loaded:", user_input, height=150)

# -------------------------------
# ANALYSIS
# -------------------------------
if st.button("Analyze News") and user_input.strip():
    with st.spinner("Analyzing..."):
        time.sleep(0.5)

        nb_pred = nb_model.predict([user_input])[0]
        nb_prob = nb_model.predict_proba([user_input])[0]
        lr_pred = lr_model.predict([user_input])[0]
        lr_prob = lr_model.predict_proba([user_input])[0]

        st.markdown("### 📊 Predictions")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Naive Bayes Model**")
            st.metric("Prediction", nb_pred.title())
            st.progress(int(max(nb_prob) * 100))
        with col2:
            st.markdown("**Logistic Regression Model**")
            st.metric("Prediction", lr_pred.title())
            st.progress(int(max(lr_prob) * 100))

        # Combined Result
        final_result = "fake" if nb_pred=="fake" and lr_pred=="fake" else "real"
        st.markdown("### 🏆 Final Assessment")
        if final_result == "fake":
            st.error("⚠️ The news is likely **FAKE**")
        else:
            st.success("✅ The news is likely **REAL**")

# -------------------------------
# ADDITIONAL INFO
# -------------------------------
st.markdown("---")
st.markdown("### ℹ️ Note")
st.markdown(
    """
- This system is a **demo** and uses a minimal dataset.
- For real-world use, train the models on a large, diverse dataset.
- URL fetching only extracts `<p>` text; some websites may not work properly.
"""
)
