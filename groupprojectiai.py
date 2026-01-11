import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
import requests
import re

# --------------------------
# Load and prepare dataset
# --------------------------
@st.cache_data
def load_data():
    # Example dataset (can replace with larger dataset later)
    data = {
        "text": [
            "Breaking: Alien spaceship lands in New York City!",
            "The government will provide free education to all citizens",
            "Scientists discovered a cure for common cold",
            "Celebrity dies in suspicious accident",
            "Study shows that eating chocolate improves intelligence"
        ],
        "label": [1, 0, 0, 1, 1]  # 1 = Fake, 0 = Real
    }
    df = pd.DataFrame(data)
    return df

df = load_data()

# --------------------------
# Preprocessing and Model Training
# --------------------------
@st.cache_resource
def train_models(df):
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(df['text'])
    y = df['label']

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Naive Bayes
    nb_model = MultinomialNB()
    nb_model.fit(X_train, y_train)

    # Logistic Regression
    lr_model = LogisticRegression()
    lr_model.fit(X_train, y_train)

    # Accuracy check
    nb_acc = accuracy_score(y_test, nb_model.predict(X_test))
    lr_acc = accuracy_score(y_test, lr_model.predict(X_test))

    return vectorizer, nb_model, lr_model, nb_acc, lr_acc

vectorizer, nb_model, lr_model, nb_acc, lr_acc = train_models(df)

# --------------------------
# Streamlit Layout
# --------------------------
st.set_page_config(page_title="Fake News Detection AI", layout="wide")

st.title("📰 Fake News Detection System")
st.markdown("""
This system uses **AI models** to detect whether a news headline or article is fake or real.
We use **Naive Bayes** and **Logistic Regression** models trained on example news data.
""")

# Sidebar for input
st.sidebar.header("Input Options")
input_choice = st.sidebar.radio("Choose input type:", ["Type Headline", "Enter Website Link"])

# --------------------------
# Process User Input
# --------------------------
user_input = ""
if input_choice == "Type Headline":
    user_input = st.text_area("Enter the news headline:")
elif input_choice == "Enter Website Link":
    url = st.text_input("Enter website link:")
    if url:
        try:
            response = requests.get(url)
            text_content = re.sub(r'<[^>]+>', '', response.text)  # strip HTML tags
            user_input = ' '.join(text_content.split()[:50])  # Take first 50 words
        except:
            st.error("Unable to fetch the website content.")

# --------------------------
# Prediction
# --------------------------
if st.button("Check News"):
    if user_input:
        X_input = vectorizer.transform([user_input])
        nb_pred = nb_model.predict(X_input)[0]
        lr_pred = lr_model.predict(X_input)[0]

        st.markdown("### Results")
        st.write(f"**Naive Bayes Prediction:** {'Fake' if nb_pred == 1 else 'Real'}")
        st.write(f"**Logistic Regression Prediction:** {'Fake' if lr_pred == 1 else 'Real'}")
        st.write(f"**Naive Bayes Model Accuracy (example dataset):** {nb_acc*100:.2f}%")
        st.write(f"**Logistic Regression Model Accuracy (example dataset):** {lr_acc*100:.2f}%")
    else:
        st.warning("Please enter a headline or provide a website link!")

# --------------------------
# Example Fake News
# --------------------------
st.markdown("---")
st.subheader("💡 Example Fake News:")
st.info("Breaking: Alien spaceship lands in New York City!")
st.markdown("Try typing this headline above to see how the system predicts it.")

# --------------------------
# System Overview
# --------------------------
st.markdown("---")
st.subheader("🔍 System Overview")
st.markdown("""
1. **Input**: Users can type a news headline or provide a website link.
2. **Processing**: The system cleans and vectorizes the text using TF-IDF.
3. **AI Models**: Naive Bayes and Logistic Regression models classify the text as Fake or Real.
4. **Output**: Predictions are shown along with example model accuracy.
""")
