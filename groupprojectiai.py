# ===============================
# FAKE NEWS DETECTION SYSTEM
# AI-Powered Verification Platform
# ===============================

import streamlit as st
import pandas as pd
import numpy as np
import re
import random
import time
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from urllib.request import urlopen
from bs4 import BeautifulSoup

# -------------------------------
# APP CONFIG
# -------------------------------
st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.main-title {
    font-size: 2.8rem;
    color: #1E3A8A;
    font-weight: 800;
    text-align: center;
    margin-bottom: 1rem;
}
.stButton>button {
    background: linear-gradient(135deg,#1E40AF,#3B82F6);
    color: white;
    font-weight: 600;
    border-radius: 8px;
}
.stButton>button:hover {
    transform: translateY(-2px);
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">📰 FAKE NEWS DETECTION SYSTEM</div>', unsafe_allow_html=True)

# -------------------------------
# SYSTEM OVERVIEW
# -------------------------------
st.markdown("""
### System Overview
This platform uses **AI models (Naive Bayes & Logistic Regression)** to detect fake news.
Users can either:
1. Type a headline or article text.
2. Provide a website link.

The system processes the text, evaluates its authenticity, and outputs whether it is likely **REAL** or **FAKE**.  

Key Features:
- NLP preprocessing (TF-IDF)
- ML prediction with NB & Logistic Regression
- Sample fake news for testing
- Interactive, user-friendly interface
""")

# -------------------------------
# SAMPLE DATASET
# -------------------------------
# Minimal example for demonstration
data = {
    'text': [
        "BREAKING: Scientists discovered a miracle cure for cancer overnight!",
        "The government announced new policies for economic growth.",
        "Celebrity endorses secret method to become rich in 7 days!",
        "Local school wins award for sustainable environment program."
    ],
    'label': [1, 0, 1, 0]  # 1 = Fake, 0 = Real
}

df = pd.DataFrame(data)

# -------------------------------
# MODEL TRAINING
# -------------------------------
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['text'])
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Naive Bayes
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)

# Logistic Regression
lr_model = LogisticRegression(max_iter=500)
lr_model.fit(X_train, y_train)

# -------------------------------
# USER INPUT
# -------------------------------
tab1, tab2 = st.tabs(["📝 Text Input", "🌐 URL Input"])

with tab1:
    user_text = st.text_area("Enter headline or article text:", height=150, placeholder="Type or paste text here...")
    if st.button("Analyze Text"):
        if user_text.strip() == "":
            st.error("Please enter some text to analyze.")
        else:
            # Preprocess
            text_vector = vectorizer.transform([user_text])
            nb_pred = nb_model.predict(text_vector)[0]
            lr_pred = lr_model.predict(text_vector)[0]

            st.markdown("### 🔍 Analysis Results")
            st.markdown(f"**Naive Bayes Prediction:** {'FAKE' if nb_pred else 'REAL'}")
            st.markdown(f"**Logistic Regression Prediction:** {'FAKE' if lr_pred else 'REAL'}")

            confidence_nb = nb_model.predict_proba(text_vector)[0][nb_pred]
            confidence_lr = lr_model.predict_proba(text_vector)[0][lr_pred]

            st.markdown(f"**Naive Bayes Confidence:** {confidence_nb:.1%}")
            st.markdown(f"**Logistic Regression Confidence:** {confidence_lr:.1%}")

with tab2:
    user_url = st.text_input("Enter website URL to analyze:")
    if st.button("Analyze URL"):
        if user_url.strip() == "":
            st.error("Please enter a URL.")
        else:
            try:
                page = urlopen(user_url)
                soup = BeautifulSoup(page, 'html.parser')
                text = ' '.join([p.get_text() for p in soup.find_all('p')])
                if len(text.strip()) == 0:
                    st.warning("No textual content detected on this page.")
                else:
                    text_vector = vectorizer.transform([text])
                    nb_pred = nb_model.predict(text_vector)[0]
                    lr_pred = lr_model.predict(text_vector)[0]

                    st.markdown("### 🔍 Analysis Results")
                    st.markdown(f"**Naive Bayes Prediction:** {'FAKE' if nb_pred else 'REAL'}")
                    st.markdown(f"**Logistic Regression Prediction:** {'FAKE' if lr_pred else 'REAL'}")

                    confidence_nb = nb_model.predict_proba(text_vector)[0][nb_pred]
                    confidence_lr = lr_model.predict_proba(text_vector)[0][lr_pred]

                    st.markdown(f"**Naive Bayes Confidence:** {confidence_nb:.1%}")
                    st.markdown(f"**Logistic Regression Confidence:** {confidence_lr:.1%}")
            except Exception as e:
                st.error(f"Error fetching URL: {e}")

# -------------------------------
# SAMPLE FAKE NEWS
# -------------------------------
st.markdown("---")
st.markdown("### 💡 Sample Fake News for Testing")
sample_fake = random.choice(df[df['label']==1]['text'].tolist())
st.info(sample_fake)
st.caption("This example is marked as FAKE in the dataset.")

# -------------------------------
# END OF SYSTEM
# -------------------------------
st.markdown("---")
st.markdown("© 2026 AI News Lab")
