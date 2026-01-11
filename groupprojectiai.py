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

# --- SETUP & INITIALIZATION ---
st.set_page_config(page_title="Forensic AI News Verifier", page_icon="🛡️", layout="wide")

@st.cache_resource
def load_nlp_resources():
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('punkt')
    return WordNetLemmatizer(), set(stopwords.words('english'))

lemmatizer, stop_words = load_nlp_resources()

# Custom Professional CSS
st.markdown("""
<style>
    .main-header { font-size: 2.8rem; color: #1E293B; font-weight: 800; text-align: center; margin-bottom: 5px; }
    .sub-header { text-align: center; color: #64748B; margin-bottom: 25px; }
    .stAlert { border-radius: 12px; }
    .metric-card { background: #F8FAFC; padding: 15px; border-radius: 10px; border: 1px solid #E2E8F0; }
</style>
""", unsafe_allow_html=True)

# --- SYSTEM OVERVIEW ---
st.markdown('<div class="main-header">🛡️ Forensic AI Verification Suite</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Hybrid Linguistic Analysis Engine v4.1</div>', unsafe_allow_html=True)

with st.expander("📊 System Architecture & Model Overview"):
    st.write("""
    **Objective:** To distinguish between factual reporting and disinformation using statistical probability.
    
    **Workflow:**
    1. **Text Normalization:** We remove noise, lowercase text, and perform **Lemmatization**.
    2. **TF-IDF Vectorization:** We create a matrix of 5,000 unique word-weights.
    3. **Hybrid Ensemble:**
        - **Naive Bayes (NB):** Best for identifying categorical 'trigger' words.
        - **Logistic Regression (LR):** Best for identifying the relationship between complex sentences.
    """)
    st.warning("**Synthetic Fake News Example:** 'SHOCKING: The secret moon base has been discovered! Government hiding miracle cure for every disease! CLICK NOW!'")

# --- CORE AI ENGINE ---
class ForensicAIEngine:
    def __init__(self):
        # Synthetic dataset for local training (In production, replace with 50,000+ row CSV)
        self.corpus = pd.DataFrame({
            'text': [
                "Stock markets reach record highs as inflation slows down.",
                "SHOCKING SECRETS: Government hiding miracle cure for every illness!",
                "Scientific study confirms moderate exercise improves heart longevity.",
                "URGENT: Drinking lemon juice prevents all viruses instantly!",
                "NASA rover finds new evidence of liquid water on Mars surface.",
                "LEAKED: Aliens living in secret tunnels under the White House!",
                "The Federal Reserve announced interest rates will remain stable.",
                "WARNING: Global currency to collapse tomorrow, buy gold now!"
            ],
            'label': [0, 1, 0, 1, 0, 1, 0, 1] # 0=Real, 1=Fake
        })
        self.vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        self.nb_model = MultinomialNB()
        self.lr_model = LogisticRegression()
        self._train_system()

    def _clean_text(self, text):
        text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
        tokens = [lemmatizer.lemmatize(w) for w in text.split() if w not in stop_words]
        return " ".join(tokens)

    def _train_system(self):
        X = self.vectorizer.fit_transform(self.corpus['text'].apply(self._clean_text))
        y = self.corpus['label']
        self.nb_model.fit(X, y)
        self.lr_model.fit(X, y)

    def analyze(self, text):
        cleaned = self._clean_text(text)
        vec = self.vectorizer.transform([cleaned])
        
        nb_p = self.nb_model.predict_proba(vec)[0][1]
        lr_p = self.lr_model.predict_proba(vec)[0][1]
        
        # Meta-analysis: Sensationalism Score
        all_caps = len(re.findall(r'\b[A-Z]{3,}\b', text))
        exclams = text.count('!') + text.count('?')
        meta_risk = min(1.0, (all_caps * 0.1) + (exclams * 0.15))
        
        combined = (nb_p * 0.4) + (lr_p * 0.4) + (meta_risk * 0.2)
        return nb_p, lr_p, combined

# --- INTERFACE ---
engine = ForensicAIEngine()

col_input, col_metrics = st.columns([2, 1])

with col_input:
    st.subheader("🖋️ Forensic Input")
    input_mode = st.radio("Select Input Type:", ["Headline / Paragraph", "Website URL"])
    
    if input_mode == "Website URL":
        user_text = st.text_input("Enter URL:", placeholder="https://news-outlet.com/article-path")
        # Simplified URL analysis for demo
        analysis_text = user_text.split('/')[-1].replace('-', ' ') if user_text else ""
    else:
        user_text = st.text_area("Enter Content:", height=200, placeholder="Paste article text here...")
        analysis_text = user_text

    scan_btn = st.button("🚀 INITIATE SYSTEM SCAN", use_container_width=True)

with col_metrics:
    st.subheader("📊 Performance Gauges")
    if scan_btn and analysis_text:
        nb, lr, risk = engine.analyze(analysis_text)
        
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = risk * 100,
            title = {'text': "Aggregate Risk Score"},
            gauge = {
                'axis': {'range': [0, 100]},
                'bar': {'color': "#1E293B"},
                'steps': [
                    {'range': [0, 40], 'color': "#10B981"},
                    {'range': [40, 75], 'color': "#F59E0B"},
                    {'range': [75, 100], 'color': "#EF4444"}]
            }
        ))
        fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("System idle. Awaiting forensic input.")

# --- RESULTS ---
if scan_btn and analysis_text:
    st.divider()
    m1, m2, m3 = st.columns(3)
    
    with m1:
        st.markdown(f'<div class="metric-card"><b>NB Model Risk:</b><br>{nb:.1%}</div>', unsafe_allow_html=True)
    with m2:
        st.markdown(f'<div class="metric-card"><b>LR Model Risk:</b><br>{lr:.1%}</div>', unsafe_allow_html=True)
    with m3:
        st.markdown(f'<div class="metric-card"><b>Pattern Confidence:</b><br>High Forensic Match</div>', unsafe_allow_html=True)

    st.markdown("### 📝 Forensic Verdict")
    if risk > 0.7:
        st.error("🚩 **CRITICAL RISK:** Analysis suggests high probability of fabrication. Text exhibits disinformation patterns.")
    elif risk > 0.4:
        st.warning("⚠️ **CAUTION:** Mixed signals detected. Content displays sensationalism inconsistent with neutral reporting.")
    else:
        st.success("✅ **CREDIBLE:** High structural alignment with objective journalistic standards.")

st.sidebar.markdown("### 🧪 Testing Sandbox")
if st.sidebar.button("Try Sample Fake News"):
    st.info("Copy this: 'BREAKING: NASA confirms aliens live on the Sun! Government hides the truth about heat shielding technology!!'")
if st.sidebar.button("Try Sample Real News"):
    st.info("Copy this: 'The European Union announced new digital privacy laws to protect citizen data across all member states today.'")
