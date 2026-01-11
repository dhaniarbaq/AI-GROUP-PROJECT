import streamlit as st
import pandas as pd
import numpy as np
import requests, re
import joblib
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string

nltk.download('stopwords')
nltk.download('wordnet')

# --------------------------
# Preprocessing Function
# --------------------------
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    text = text.lower()
    text = re.sub(r'https?://\S+', '', text)  # remove urls
    text = re.sub(r'<.*?>', '', text)  # remove html tags
    text = re.sub(r'[^a-z\s]', '', text)  # remove punctuation
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return ' '.join(words)

# --------------------------
# Load Dataset (Example Fake News Dataset)
# --------------------------
@st.cache_data
def load_dataset():
    data = {
        "text": [
            "Breaking: Alien spaceship lands in New York City!",
            "Government will provide free education to all citizens",
            "Scientists discovered a cure for common cold",
            "Celebrity dies in suspicious accident",
            "Study shows that eating chocolate improves intelligence"
        ],
        "label": [1, 0, 0, 1, 1]  # 1=Fake, 0=Real
    }
    df = pd.DataFrame(data)
    df['text'] = df['text'].apply(preprocess)
    return df

df = load_dataset()

# --------------------------
# Train Models
# --------------------------
@st.cache_resource
def train_models(df):
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1,2))
    X = vectorizer.fit_transform(df['text'])
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    nb_model = MultinomialNB()
    nb_model.fit(X_train, y_train)

    lr_model = LogisticRegression()
    lr_model.fit(X_train, y_train)

    nb_acc = accuracy_score(y_test, nb_model.predict(X_test))
    lr_acc = accuracy_score(y_test, lr_model.predict(X_test))

    return vectorizer, nb_model, lr_model, nb_acc, lr_acc

vectorizer, nb_model, lr_model, nb_acc, lr_acc = train_models(df)

# --------------------------
# Streamlit Layout
# --------------------------
st.set_page_config(page_title="Advanced Fake News Detection AI", layout="wide")
st.title("📰 Advanced Fake News Detection System")

# Sidebar for overview
st.sidebar.header("System Overview")
st.sidebar.markdown("""
1. **Input**: User types a headline or enters a website link.  
2. **Preprocessing**: Text cleaned, stopwords removed, lemmatized.  
3. **Feature Extraction**: TF-IDF with n-grams.  
4. **AI Models**: Naive Bayes & Logistic Regression.  
5. **Output**: Prediction (Fake/Real) + Confidence Score.
""")

# Input choice
st.sidebar.header("Input Options")
input_choice = st.sidebar.radio("Choose input type:", ["Type Headline", "Enter Website Link"])

user_input = ""
if input_choice == "Type Headline":
    user_input = st.text_area("Enter the news headline:")
elif input_choice == "Enter Website Link":
    url = st.text_input("Enter website link:")
    if url:
        try:
            r = requests.get(url)
            soup = BeautifulSoup(r.text, 'html.parser')
            text_content = soup.get_text()
            user_input = ' '.join(text_content.split()[:100])  # first 100 words
        except:
            st.error("Failed to fetch website content.")

# Prediction
if st.button("Analyze News"):
    if user_input:
        processed = preprocess(user_input)
        X_input = vectorizer.transform([processed])

        nb_pred = nb_model.predict(X_input)[0]
        nb_prob = np.max(nb_model.predict_proba(X_input))

        lr_pred = lr_model.predict(X_input)[0]
        lr_prob = np.max(lr_model.predict_proba(X_input))

        st.markdown("### 🔍 Results")
        st.write(f"**Naive Bayes Prediction:** {'Fake' if nb_pred==1 else 'Real'} (Confidence: {nb_prob*100:.2f}%)")
        st.write(f"**Logistic Regression Prediction:** {'Fake' if lr_pred==1 else 'Real'} (Confidence: {lr_prob*100:.2f}%)")
        st.write(f"**Naive Bayes Model Accuracy:** {nb_acc*100:.2f}%")
        st.write(f"**Logistic Regression Model Accuracy:** {lr_acc*100:.2f}%")
    else:
        st.warning("Please provide a headline or website link!")

# Example fake news
st.markdown("---")
st.subheader("💡 Example Fake News:")
st.info("Breaking: Alien spaceship lands in New York City!")

# AI workflow diagram
st.markdown("---")
st.subheader("🤖 AI System Workflow")
st.markdown("""
**Step 1:** Input headline or URL  
**Step 2:** Preprocessing → clean text, remove stopwords, lemmatize  
**Step 3:** Feature extraction → TF-IDF with n-grams  
**Step 4:** AI model prediction → Naive Bayes & Logistic Regression  
**Step 5:** Output → Fake or Real with confidence score
""")
