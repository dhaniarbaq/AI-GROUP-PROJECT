import streamlit as st

# -------------------------
# System Info
# -------------------------
SYSTEM_VERSION = "2.1 Pattern Recognition"
TOTAL_DISINFO_PATTERNS = 8
TOTAL_AUTH_PATTERNS = 5
ALGORITHM = "Weighted pattern matching"

# -------------------------
# Sidebar
# -------------------------
st.sidebar.title("Disinformation Pattern Recognition")
sidebar_options = [
    "📚 Pattern Library",
    "⚡ Quick Analysis",
    "📊 Analysis Dashboard",
    "ℹ️ System Info"
]
selected_tab = st.sidebar.radio("Navigate", sidebar_options)

# -------------------------
# Tabs / Pages
# -------------------------
# ----- Tab 1: Home / Pattern Library -----
if selected_tab == "📚 Pattern Library":
    st.header("Welcome to the Disinformation Pattern Recognition System")
    st.write("""
        This system helps detect and analyze disinformation patterns in text.  
        Use the sidebar to select patterns or run quick analyses.  
        Navigate through the tabs to access the full features.  
    """)
    st.image("https://cdn-icons-png.flaticon.com/512/2910/2910762.png", width=200)
    
    st.subheader("Disinformation Pattern Library 📚")
    
    # Define patterns with descriptions and captions
    patterns_info = {
        "🔎 Emotional Amplification": "Texts that exaggerate emotions to provoke anger, fear, or excitement. \n*Caption:* Recognize when content is trying to push your emotional buttons.",
        "🔎 False Urgency": "Messages that create a sense of immediate threat or opportunity. \n*Caption:* Spot the rush tactics before reacting impulsively.",
        "🔎 Source Obfuscation": "Information that hides or misrepresents the source to seem credible. \n*Caption:* Check if the source is trustworthy.",
        "🔎 Binary Narrative": "Content that oversimplifies complex issues into 'good vs. evil' or 'us vs. them'. \n*Caption:* Beware of black-and-white thinking in media."
    }
    
    for pattern, description in patterns_info.items():
        st.markdown(f"**{pattern}**")
        st.info(description)

# ----- Tab 2: Quick Analysis -----
elif selected_tab == "⚡ Quick Analysis":
    st.header("Quick Analysis ⚡")
    st.write("Paste your text below and the system will detect potential disinformation patterns.")
    user_input = st.text_area("Enter text to analyze:", height=200)
    
    if st.button("Analyze"):
        if user_input.strip() == "":
            st.warning("Please enter some text to analyze.")
        else:
            # Fake analysis logic for demonstration
            st.success("Analysis complete! ⚡")
            detected_patterns = ["Emotional Amplification", "False Urgency"]  # Example
            st.write("Detected patterns:")
            for p in detected_patterns:
                st.write(f"- {p}")

# ----- Tab 3: Analysis Dashboard -----
elif selected_tab == "📊 Analysis Dashboard":
    st.header("Analysis Dashboard 📊")
    st.write("No analyses yet. Start by analyzing text above!")

# ----- Tab 4: System Info -----
elif selected_tab == "ℹ️ System Info":
    st.header("System Info ℹ️")
    st.markdown(f"- **Version:** {SYSTEM_VERSION}")
    st.markdown(f"- **Patterns:** {TOTAL_DISINFO_PATTERNS} disinformation + {TOTAL_AUTH_PATTERNS} authenticity")
    st.markdown(f"- **Algorithm:** {ALGORITHM}")
