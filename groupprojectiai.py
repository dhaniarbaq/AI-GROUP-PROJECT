import streamlit as st
import pandas as pd
import numpy as np

# ----------------- Sidebar -----------------
st.sidebar.title("🧠 Disinformation Pattern Recognition")

# Pattern Library Section
st.sidebar.subheader("📚 Pattern Library")
pattern_options = [
    "🔎 Emotional Amplification",
    "🔎 False Urgency",
    "🔎 Source Obfuscation",
    "🔎 Binary Narrative"
]
selected_pattern = st.sidebar.radio("Select a Pattern", pattern_options)

# Quick Analysis Button
st.sidebar.subheader("⚡ Quick Analysis")
if st.sidebar.button("Run Quick Analysis"):
    st.sidebar.success(f"Quick analysis triggered for {selected_pattern}!")

# Analysis Dashboard Section
st.sidebar.subheader("📊 Analysis Dashboard")
st.sidebar.info("No analyses yet. Start by analyzing text above!")

# System Info Section
st.sidebar.subheader("ℹ️ System Info")
st.sidebar.text("Version: 2.1 Pattern Recognition")
st.sidebar.text("Patterns: 8 disinformation + 5 authenticity")
st.sidebar.text("Algorithm: Weighted pattern matching")

# ----------------- Main Tabs -----------------
tabs = ["Home", "Pattern Recognition", "Text Analyzer", "Dashboard", "Reports", "Settings"]
selected_tab = st.tabs(tabs)

# ----- Tab 1: Home -----
with selected_tab[0]:
    st.header("Welcome to the Disinformation Pattern Recognition System")
    st.write("""
        This system helps detect and analyze disinformation patterns in text.  
        Use the sidebar to select patterns or run quick analyses.  
        Navigate through the tabs to access the full features.  
    """)
    st.image("https://cdn-icons-png.flaticon.com/512/2910/2910762.png", width=200)

# ----- Tab 2: Pattern Recognition -----
with selected_tab[1]:
    st.header("Pattern Recognition")
    st.write(f"You selected: **{selected_pattern}**")
    
    text_input = st.text_area("Enter text to analyze for disinformation patterns:", "")
    
    if st.button("Analyze Text"):
        if text_input.strip() != "":
            # Dummy pattern detection logic
            st.success(f"Analysis complete for pattern: {selected_pattern}")
            st.write(f"Detected {np.random.randint(0, 5)} instances of {selected_pattern}")
        else:
            st.warning("Please enter some text to analyze!")

# ----- Tab 3: Text Analyzer -----
with selected_tab[2]:
    st.header("Text Analyzer")
    st.write("Perform sentiment and word frequency analysis")
    
    text_input2 = st.text_area("Enter text for analysis:", "")
    
    if st.button("Run Text Analysis"):
        if text_input2.strip() != "":
            words = text_input2.split()
            word_count = len(words)
            unique_words = len(set(words))
            st.write(f"Total words: {word_count}")
            st.write(f"Unique words: {unique_words}")
        else:
            st.warning("Enter some text for analysis!")

# ----- Tab 4: Dashboard -----
with selected_tab[3]:
    st.header("📊 Analysis Dashboard")
    st.write("Visualizations and analysis summary")
    
    # Dummy chart example
    data = pd.DataFrame({
        'Pattern': ["Emotional Amplification", "False Urgency", "Source Obfuscation", "Binary Narrative"],
        'Count': np.random.randint(0, 10, 4)
    })
    st.bar_chart(data.set_index('Pattern'))

# ----- Tab 5: Reports -----
with selected_tab[4]:
    st.header("Reports")
    st.write("Download analysis results")
    
    dummy_report = pd.DataFrame({
        "Pattern": pattern_options,
        "Instances Detected": np.random.randint(0, 10, 4)
    })
    
    st.dataframe(dummy_report)
    
    csv = dummy_report.to_csv(index=False).encode('utf-8')
    st.download_button("Download Report as CSV", csv, "analysis_report.csv", "text/csv")

# ----- Tab 6: Settings -----
with selected_tab[5]:
    st.header("Settings")
    st.write("Configure system preferences")
    
    version = st.text_input("System Version", "2.1 Pattern Recognition")
    algorithm = st.text_input("Detection Algorithm", "Weighted pattern matching")
    
    if st.button("Save Settings"):
        st.success("Settings saved successfully!")
