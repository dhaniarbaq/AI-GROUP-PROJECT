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

# ===============================
# 1. INITIALIZATION & NLP SETUP
# ===============================
@st.cache_resource
def initialize_nlp():
    # Downloads necessary data for the server
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    nltk.download('punkt')
    return WordNetLemmatizer(), set(stopwords.words('english'))

lemmatizer, stop_words = initialize_nlp()

# Custom CSS for Professional UI
st.markdown("""
<style>
    .main-header { font-size: 2.5rem; color: #1E293B; font-weight: 800; text-align: center; margin-bottom: 10px; }
    .stAlert { border-radius: 12px; }
    .card { background: white; padding: 20px; border-radius: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); border: 1px solid #E2E8F0; color: #1E293B; }
    .verdict-high { background-color: #FEE2E2; color: #991B1B; border: 2px solid #EF4444; padding: 20px; border-radius: 12px; text-align: center; font-weight: bold; font-size: 1.2rem; }
    .verdict-low { background-color: #D1FAE5; color: #065F46; border: 2px solid #10B981; padding: 20px; border-radius: 12px; text-align: center; font-weight: bold; font-size: 1.2rem; }
</style>
""", unsafe_allow_html=True)

# ===============================
# 2. SYSTEM OVERVIEW
# ===============================
st.markdown('<div class="main-header">🔍 AI Forensic Information Verifier</div>', unsafe_allow_html=True)

with st.expander("📖 System Overview & Methodology", expanded=True):
    st.write("""
    This advanced system uses a **Hybrid AI Ensemble** to detect disinformation. 
    1. **Text Pre-processing**: Input is cleaned, tokenized, and lemmatized to its root form.
    2. **TF-IDF Vectorization**: Words are converted into numerical importance scores.
    3. **Machine Learning Models**: 
        - **Naive Bayes**: Uses probability to find 'spammy' word associations.
        - **Logistic Regression**: Analyzes the structural relationship between words.
    """)
    st.info("**Example of Fake News (for testing):** 'SHOCKING: The government is hiding a secret alien base under the city and plans to seize all bank accounts tomorrow!!'")

# ===============================
# 3. CORE ANALYTICS ENGINE
# ===============================
class ForensicEngine:
    def __init__(self):
        # Sample Training Data (Representative of real-world patterns)
        self.train_data = [
            ("The central bank raised interest rates by 25 basis points.", 0),
            ("SHOCKING: Miracle water cures cancer in 2 hours! Doctors are FURIOUS!", 1),
            ("New studies show coffee improves heart health in older adults.", 0),
            ("BREAKING: Alien invasion starting in New York, police are fleeing!", 1),
            ("The local mayor announced a new budget for public parks today.", 0),
            ("URGENT: Your bank account will be deleted unless you click here now!", 1),
            ("NASA launched a satellite to study the atmospheric changes in Mars.", 0),
            ("LEAKED: Secret document proves the moon is actually a hollow base.", 1)
        ]
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2))
        self.nb_model = MultinomialNB()
        self.lr_model = LogisticRegression()
        self._train()

    def _clean(self, text):
        text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
        tokens = [lemmatizer.lemmatize(w) for w in text.split() if w not in stop_words]
        return " ".join(tokens)

    def _train(self):
        texts = [self._clean(t[0]) for t in self.train_data]
        labels = [t[1] for t in self.train_data]
        X = self.vectorizer.fit_transform(texts)
        self.nb_model.fit(X, labels)
        self.lr_model.fit(X, labels)

    def predict(self, text):
        cleaned = self._clean(text)
        vec = self.vectorizer.transform([cleaned])
        nb_score = self.nb_model.predict_proba(vec)[0][1]
        lr_score = self.lr_model.predict_proba(vec)[0][1]
        
        # Meta-analysis (all caps and punctuation)
        cap_ratio = sum(1 for c in text if c.isupper()) / len(text) if len(text) > 0 else 0
        punc_score = (text.count('!') + text.count('?')) / 10
        
        combined = (nb_score * 0.4) + (lr_score * 0.4) + (min(1.0, cap_ratio + punc_score) * 0.2)
        return nb_score, lr_score, combined

# Init Engine
engine = ForensicEngine()

# ===============================
# 4. USER INTERFACE & INPUT
# ===============================
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("🖋️ Input Terminal")
    input_type = st.segmented_control("Selection Mode", ["Text/Headline", "Website Link"])
    
    if input_type == "Website Link":
        user_input = st.text_input("Enter Website URL:", placeholder="https://news-portal.com/article-123")
        # Simulate scraping by extracting keywords from the URL
        if user_input:
            processed_input = user_input.split('/')[-1].replace('-', ' ')
        else: processed_input = ""
    else:
        user_input = st.text_area("Enter News Headline or Content:", height=150, placeholder="Paste here...")
        processed_input = user_input

    analyze_btn = st.button("🚀 EXECUTE FORENSIC SCAN", use_container_width=True)

with col2:
    st.subheader("📊 Live Metrics")
    if analyze_btn and processed_input:
        nb, lr, combined = engine.predict(processed_input)
        
        # Risk Gauge
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = combined * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Aggregate Risk %"},
            gauge = {
                'axis': {'range': [0, 100]},
                'bar': {'color': "#1E293B"},
                'steps': [
                    {'range': [0, 40], 'color': "#10B981"},
                    {'range': [40, 70], 'color': "#F59E0B"},
                    {'range': [70, 100], 'color': "#EF4444"}]
            }
        ))
        fig.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Awaiting input for real-time analysis.")

# ===============================
# 5. RESULTS DISPLAY
# ===============================
if analyze_btn and processed_input:
    st.divider()
    res1, res2, res3 = st.columns(3)
    
    res1.metric("Naive Bayes Score", f"{nb:.1%}")
    res2.metric("Logistic Regression", f"{lr:.1%}")
    res3.metric("Pattern Confidence", "High" if combined > 0.7 or combined < 0.3 else "Medium")
    
    st.markdown("### 📋 Forensic Verdict")
    if combined > 0.65:
        st.markdown(f'<div class="verdict-high">🚩 HIGH RISK: Potential Disinformation Detected</div>', unsafe_allow_html=True)
        st.write("The AI ensemble has identified linguistic markers (Sensationalism, specific word-weights) typically found in unverified news sources.")
    elif combined > 0.35:
        st.warning("⚠️ MIXED SIGNALS: The system is uncertain. This text contains some objective language but exhibits sensationalist traits.")
    else:
        st.markdown(f'<div class="verdict-low">✅ LOW RISK: Likely Credible Information</div>', unsafe_allow_html=True)
        st.write("Linguistic analysis shows patterns consistent with factual reporting and standard journalistic ethics.")

elif analyze_btn:
    st.error("Please enter text or a link to proceed.")

st.sidebar.markdown("### 🛠️ Quick Sandbox")
if st.sidebar.button("Load Fake News Sample"):
    st.info("Copy this: 'BREAKING: Drinking bleach cures every disease known to man, scientists are shocked!'")
if st.sidebar.button("Load Real News Sample"):
    st.info("Copy this: 'The European Union announced new regulations regarding data privacy for tech companies today.'")
