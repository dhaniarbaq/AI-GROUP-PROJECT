import streamlit as st
import pandas as pd
import numpy as np
import random
from textblob import TextBlob
from datetime import datetime

# ---------------- Session State ----------------
if "history" not in st.session_state:
    st.session_state.history = []

if "threshold" not in st.session_state:
    st.session_state.threshold = 0.6

if "text_input" not in st.session_state:
    st.session_state.text_input = ""

# ---------------- Sidebar ----------------
st.sidebar.title("📰 Fake News Detection System")

st.sidebar.subheader("⚡ Quick Analysis")
st.sidebar.write("Automatic NLP-based fake news indicator")

st.sidebar.subheader("ℹ️ System Info")
st.sidebar.text("Version: 1.1")
st.sidebar.text("Method: NLP + Heuristics")
st.sidebar.text("Mode: Explainable Detection")

# ---------------- Tabs ----------------
tabs = ["Home", "Analyzer", "Dashboard", "Reports", "Settings"]
tab_home, tab_analyzer, tab_dashboard, tab_reports, tab_settings = st.tabs(tabs)

# ---------------- Home ----------------
with tab_home:
    st.header("Welcome")
    st.write("""
    This application identifies **potential fake news** using linguistic cues.

    The system evaluates:
    - Emotional extremeness
    - Subjectivity level
    - Sensational wording

    ⚠️ This tool provides an **early warning indicator**, not factual verification.
    """)

    st.info("Designed for academic demonstration and awareness purposes.")

# ---------------- Analyzer ----------------
with tab_analyzer:
    st.header("🧠 Fake News Analyzer")

    if st.button("🎲 Generate Sample Text"):
        samples = [
            "Breaking news scientists discover miracle cure overnight",
            "Government secretly approves law banning cash next month",
            "Experts warn phones will explode if not turned off tonight",
            "Celebrity claims salt water cures all diseases",
            "Shocking report reveals humans no longer need sleep",
            "New study guarantees weight loss in two days"
        ]
        st.session_state.text_input = "\n".join(random.sample(samples, 3))

    text_input = st.text_area(
        "Step 1: Enter text",
        value=st.session_state.text_input,
        height=180
    )

    if st.button("🔍 Analyze"):
        if text_input.strip() == "":
            st.warning("Please enter text before analysis.")
        else:
            blob = TextBlob(text_input)
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity

            sensational_words = [
                "breaking", "shocking", "secret", "miracle",
                "guarantee", "overnight", "warn", "explode"
            ]

            detected = [
                word for word in sensational_words
                if word in text_input.lower()
            ]

            score = round((abs(polarity) + subjectivity) / 2, 2)

            prediction = (
                "Likely Fake News"
                if score >= st.session_state.threshold
                else "Likely Genuine News"
            )

            reason = []
            if abs(polarity) > 0.5:
                reason.append("Extreme emotional polarity")
            if subjectivity > 0.5:
                reason.append("High subjectivity")
            if detected:
                reason.append("Sensational keywords detected")

            reason_text = ", ".join(reason) if reason else "Neutral linguistic tone"

            # Save history
            st.session_state.history.append({
                "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Text": text_input[:80] + "...",
                "Prediction": prediction,
                "Confidence (%)": int(score * 100),
                "Reason": reason_text
            })

            st.subheader("Result")
            st.write(f"**Prediction:** {prediction}")
            st.progress(score)
            st.write(f"**Confidence Score:** {score * 100:.0f}%")
            st.write(f"Polarity: {polarity:.2f}")
            st.write(f"Subjectivity: {subjectivity:.2f}")
            st.write(f"Key Reason: {reason_text}")

            st.caption("Result is probabilistic and not a factual judgment.")

# ---------------- Dashboard ----------------
with tab_dashboard:
    st.header("📊 Analysis Dashboard")

    if st.session_state.history:
        df = pd.DataFrame(st.session_state.history)

        counts = df["Prediction"].value_counts()
        st.bar_chart(counts)

        avg_conf = df["Confidence (%)"].mean()
        st.metric("Average Confidence Score", f"{avg_conf:.1f}%")
    else:
        st.info("No analysis data available yet.")

# ---------------- Reports ----------------
with tab_reports:
    st.header("📄 Reports")

    if st.session_state.history:
        report_df = pd.DataFrame(st.session_state.history)
        st.dataframe(report_df)

        csv = report_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download Report",
            csv,
            "fake_news_report.csv",
            "text/csv"
        )
    else:
        st.info("Run analysis to generate reports.")

# ---------------- Settings ----------------
with tab_settings:
    st.header("⚙️ Settings")

    threshold = st.slider(
        "Fake News Confidence Threshold",
        min_value=0.1,
        max_value=0.9,
        value=st.session_state.threshold,
        step=0.05
    )

    if st.button("💾 Save Settings"):
        st.session_state.threshold = threshold
        st.success("Settings updated successfully.")
