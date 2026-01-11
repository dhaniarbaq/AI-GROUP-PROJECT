# ===============================
# FAKE NEWS DETECTION SYSTEM
# AI-Based News Verification
# ===============================

import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
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
- Enter a headline or article text to analyze.
- The system predicts using both models.
- A sample fake news headline is included for testing.
"""
)

# -------------------------------
# INPUT SECTION
# -------------------------------
st.markdown("## Enter News Headline or Text")
user_input = st.text_area("Enter the news text here:", height=150)

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
"""
)
