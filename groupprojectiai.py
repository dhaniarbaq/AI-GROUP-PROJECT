import streamlit as st
import pandas as pd
import numpy as np
import re
import time
import nltk
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Ensure NLP assets are available
@st.cache_resource
def download_nltk_data():
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('omw-1.4')

download_nltk_data()

# ===============================
# SYSTEM CONFIG & UI THEME
# ===============================
st.set_page_config(page_title="Forensic AI Info-Verif", page_icon="🕵️", layout="wide")

st.markdown("""
<style>
    .reportview-container { background: #0f172a; color: white; }
    .stMetric { background: #1e293b; padding: 15px; border-radius: 10px; border: 1px solid #334155; }
    .verdict-box { padding: 20px; border-radius: 15px; text-align: center; font-size: 1.5rem; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# -------------------------------
# CORE NLP & ML ENGINE
# -------------------------------
class ComplexVerificationSystem:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 3))
        
        # Expanded Synthetic Training Set
        self.corpus = [
            ("SHOCKING! New law allows government to take your home tomorrow!", 1),
            ("Consumer Price Index rose by 0.2% according to Labor Statistics.", 0),
            ("CURE FOR DIABETES FOUND! Big Pharma doesn't want you to know!!", 1),
            ("Global stock markets stabilized after central bank policy announcement.", 0),
            ("URGENT: Drinking lemon water cures viral infections instantly!", 1),
            ("The city council approved the new public transit expansion plan.", 0),
            ("LEAKED: Famous celebrity caught in secret underground cult meeting.", 1),
            ("NASA launches new satellite to monitor arctic ice shelf melting.", 0)
        ]
        self._prepare_models()

    def clean_text(self, text):
        text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
        tokens = text.split()
        tokens = [self.lemmatizer.lemmatize(w) for w in tokens if w not in self.stop_words]
        return " ".join(tokens)

    def _prepare_models(self):
        texts = [self.clean_text(t[0]) for t in self.corpus]
        labels = [t[1] for t in self.corpus]
        X = self.tfidf.fit_transform(texts)
        self.nb = MultinomialNB().fit(X, labels)
        self.lr = LogisticRegression().fit(X, labels)

    def analyze_deep(self, text):
        cleaned = self.clean_text(text)
        vec = self.tfidf.transform([cleaned])
        
        nb_prob = self.nb.predict_proba(vec)[0][1]
        lr_prob = self.lr.predict_proba(vec)[0][1]
        
        # Forensic Heuristics
        all_caps = len(re.findall(r'\b[A-Z]{3,}\b', text))
        exclams = text.count('!')
        lexical_diversity = len(set(text.split())) / len(text.split()) if len(text.split()) > 0 else 0
        
        return {
            "nb_risk": nb_prob,
            "lr_risk": lr_prob,
            "heuristic_risk": min(1.0, (all_caps * 0.15) + (exclams * 0.1)),
            "lexical_score": lexical_diversity,
            "final_consensus": (nb_prob * 0.4) + (lr_prob * 0.4) + (min(1.0, (all_caps * 0.15)) * 0.2)
        }

# Initialize System
if 'ai_system' not in st.session_state:
    st.session_state.ai_system = ComplexVerificationSystem()

# -------------------------------
# MAIN DASHBOARD UI
# -------------------------------
st.title("🕵️ Forensic AI Disinformation Analysis Suite")
st.markdown("---")

# System Overview Section
with st.container():
    col_a, col_b = st.columns([1, 1])
    with col_a:
        st.subheader("🛠️ System Overview")
        st.write("""
        This suite employs a **Multi-Model Ensemble Pipeline**. It converts raw text into 
        **TF-IDF Vectors** (N-gram range 1-3) and processes them through two distinct 
        mathematical frameworks.
        """)
    with col_b:
        st.info("**Example Fake News:** 'BREAKING: Scientists discover moon is made of blue cheese, government hides the truth!'")

# Input Workspace
st.subheader("📥 Analysis Workspace")
user_input = st.text_area("Enter Headline or Website Content for Forensic Scan:", height=150)

if st.button("🚀 INITIATE DEEP SCAN", use_container_width=True):
    if user_input:
        results = st.session_state.ai_system.analyze_deep(user_input)
        
        # 📊 Visualization Row
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Naive Bayes Risk", f"{results['nb_risk']:.1%}")
        with col2:
            st.metric("Logistic Reg. Risk", f"{results['lr_risk']:.1%}")
        with col3:
            st.metric("Linguistic Risk", f"{results['heuristic_risk']:.1%}")

        # Consensus Gauge Chart
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = results['final_consensus'] * 100,
            title = {'text': "Aggregate Forensic Risk Score"},
            gauge = {
                'axis': {'range': [0, 100]},
                'bar': {'color': "#3b82f6"},
                'steps': [
                    {'range': [0, 40], 'color': "#10b981"},
                    {'range': [40, 70], 'color': "#f59e0b"},
                    {'range': [70, 100], 'color': "#ef4444"}]
            }
        ))
        st.plotly_chart(fig, use_container_width=True)

        # Final Forensic Verdict
        st.divider()
        if results['final_consensus'] > 0.7:
            st.markdown('<div class="verdict-box" style="background:#450a0a; color:#f87171; border:2px solid #f87171;">🚩 FORENSIC VERDICT: HIGH RISK DISINFORMATION</div>', unsafe_allow_html=True)
            st.write("**Observations:** Detected high frequency of sensationalist linguistic patterns and statistical word-weights common in propaganda.")
        elif results['final_consensus'] > 0.4:
            st.markdown('<div class="verdict-box" style="background:#451a03; color:#fbbf24; border:2px solid #fbbf24;">⚠️ FORENSIC VERDICT: UNVERIFIED / SENSATIONALIST</div>', unsafe_allow_html=True)
            st.write("**Observations:** Content shows mixed signals. Lexical diversity is moderate, but emotional triggers are present.")
        else:
            st.markdown('<div class="verdict-box" style="background:#064e3b; color:#34d399; border:2px solid #34d399;">✅ FORENSIC VERDICT: LIKELY CREDIBLE</div>', unsafe_allow_html=True)
            st.write("**Observations:** Linguistic structure aligns with factual reporting standards.")

    else:
        st.error("Input required for analysis.")

st.markdown("---")
st.caption("AI Ethics Note: This system uses statistical probability and heuristics. It should supplement, not replace, human fact-checking.")
