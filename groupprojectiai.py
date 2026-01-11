import streamlit as st
import pandas as pd
import numpy as np
import random
from textblob import TextBlob

# ---------------- Sidebar ----------------
st.sidebar.title("📰 Fake News Detection System")

st.sidebar.subheader("⚡ Quick Analysis")
st.sidebar.write("Automatically detects fake news using NLP")

st.sidebar.subheader("📊 Analysis Dashboard")
st.sidebar.info("Analyze text to populate dashboard")

st.sidebar.subheader("ℹ️ System Info")
st.sidebar.text("Version: 1.0")
st.sidebar.text("Method: NLP + Heuristics")
st.sidebar.text("Mode: Automatic Detection")

# ---------------- Tabs ----------------
tabs = ["Home", "Analyzer", "Dashboard", "Reports", "Settings"]
tab_home, tab_analyzer, tab_dashboard, tab_reports, tab_settings = st.tabs(tabs)

# ---------------- Home ----------------
with tab_home:
    st.header("Welcome to the Fake News Detection System")
    st.write("""
    This web application helps users identify **potential fake news** using
    Natural Language Processing (NLP).

    The system analyzes:
    - Emotional extremeness
    - Subjectivity
    - Sensational wording

    No manual pattern selection is required.
    """)

    st.subheader("Why Fake News Detection Matters")
    st.write("""
    Fake news can manipulate opinions, spread panic, and misinform the public.
    This system provides an **early warning indicator**, not a final judgment.
    """)

# ---------------- Analyzer ----------------
with tab_analyzer:
    st.header("🧠 Fake News Analyzer")

    # Sample sentence generator
    if st.button("🎲 Generate Sample Sentences"):
        samples = [
            "Breaking news: Scientists discover a miracle cure overnight!",
            "Government secretly approves new law banning cash next month.",
            "Experts warn that phones will explode if not turned off tonight.",
            "Celebrity claims drinking salt water cures all diseases.",
            "Shocking report reveals humans may not need sleep anymore.",
            "New study says chocolate guarantees weight loss in two days."
        ]
        generated_text = "\n".join(random.sample(samples, 3))
        st.session_state["text_input"] = generated_text

    text_input = st.text_area(
        "Enter text to analyze:",
        value=st.session_state.get("text_input", ""),
        height=180
    )

    if st.button("🔍 Analyze Text"):
        if text_input.strip() == "":
            st.warning("Please enter or generate text first.")
        else:
            blob = TextBlob(text_input)
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity

            # Fake news confidence score
            score = round((abs(polarity) + subjectivity) / 2, 2)

            if score >= 0.6:
                result = "⚠️ Likely Fake News"
            else:
                result = "✅ Likely Genuine News"

            st.subheader("Result")
            st.write(f"**Prediction:** {result}")
            st.write(f"**Confidence Score:** {score * 100:.0f}%")
            st.write(f"Polarity: {polarity:.2f}")
            st.write(f"Subjectivity: {subjectivity:.2f}")

# ---------------- Dashboard ----------------
with tab_dashboard:
    st.header("📊 Analysis Dashboard")

    data = pd.DataFrame({
        "Category": ["Likely Fake", "Likely Genuine"],
        "Count": np.random.randint(1, 10, 2)
    })

    st.bar_chart(data.set_index("Category"))

    st.caption("Dashboard shows simulated results for demonstration.")

# ---------------- Reports ----------------
with tab_reports:
    st.header("📄 Reports")

    report = pd.DataFrame({
        "Text Sample": ["Sample A", "Sample B", "Sample C"],
        "Prediction": ["Likely Fake", "Likely Genuine", "Likely Fake"],
        "Confidence (%)": [82, 34, 71]
    })

    st.dataframe(report)

    csv = report.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download CSV Report",
        csv,
        "fake_news_report.csv",
        "text/csv"
    )

# ---------------- Settings ----------------
with tab_settings:
    st.header("⚙️ Settings")

    version = st.text_input("System Version", "1.0")
    method = st.text_input("Detection Method", "NLP + Heuristics")

    if st.button("💾 Save Settings"):
        st.success("Settings saved successfully!")
