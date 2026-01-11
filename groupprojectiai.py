import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

st.set_page_config(page_title="Student Data Dashboard", layout="wide")

# ------------------- Load Data -------------------
@st.cache_data
def load_data(file=None):
    if file:
        df = pd.read_csv(file)
    else:
        df = pd.read_csv("student_data.csv")
    return df

uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
data = load_data(uploaded_file)

# ------------------- Tabs -------------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Dashboard", "Data Table", "Visualization", "Prediction", "Insights", "ML Model"
])

# ------------------- Dashboard -------------------
with tab1:
    st.title("📊 Student Data Dashboard")
    col1, col2, col3 = st.columns(3)
    col1.metric("Average Study Hours", round(data["StudyHours"].mean(), 2))
    col2.metric("Avg Social Media Hours", round(data["SocialMediaHours"].mean(), 2))
    col3.metric("Students Using Transport", f"{(data['Transport'] == 'Yes').mean()*100:.2f}%")
    
    st.subheader("Gender Distribution")
    st.bar_chart(data['Gender'].value_counts())

    st.subheader("Program Distribution")
    st.bar_chart(data['Program'].value_counts())

# ------------------- Data Table -------------------
with tab2:
    st.title("📋 Data Table / Exploration")
    selected_columns = st.multiselect("Select columns", data.columns.tolist(), default=data.columns.tolist())
    filtered_data = data[selected_columns]
    
    st.dataframe(filtered_data, use_container_width=True)
    
    st.download_button(
        "Download Filtered CSV",
        filtered_data.to_csv(index=False),
        "filtered_student_data.csv"
    )

# ------------------- Visualization -------------------
with tab3:
    st.title("📈 Data Visualization")
    chart_type = st.selectbox("Select chart type", ["Bar", "Line", "Pie", "Scatter", "Histogram", "Boxplot", "Heatmap"])
    
    if chart_type in ["Bar", "Line", "Scatter"]:
        x_col = st.selectbox("X-axis", data.columns)
        y_col = st.selectbox("Y-axis", data.columns)
    
    if chart_type == "Bar":
        st.bar_chart(data.groupby(x_col)[y_col].mean())
    elif chart_type == "Line":
        st.line_chart(data.groupby(x_col)[y_col].mean())
    elif chart_type == "Pie":
        st.pyplot(plt.pie(data[y_col].value_counts(), labels=data[y_col].value_counts().index, autopct="%1.1f%%"))
    elif chart_type == "Scatter":
        fig, ax = plt.subplots()
        sns.scatterplot(x=data[x_col], y=data[y_col], hue=data.get("Gender", None), ax=ax)
        st.pyplot(fig)
    elif chart_type == "Histogram":
        col = st.selectbox("Select Column", data.columns)
        bins = st.slider("Number of Bins", 5, 50, 10)
        fig, ax = plt.subplots()
        sns.histplot(data[col], bins=bins, kde=True, ax=ax)
        st.pyplot(fig)
    elif chart_type == "Boxplot":
        col = st.selectbox("Select Column", data.columns)
        fig, ax = plt.subplots()
        sns.boxplot(y=data[col], ax=ax)
        st.pyplot(fig)
    elif chart_type == "Heatmap":
        st.subheader("Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(10,6))
        sns.heatmap(data.corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

# ------------------- Prediction -------------------
with tab4:
    st.title("🤖 Predict Study Efficiency")
    age = st.number_input("Age", min_value=15, max_value=50, value=20)
    study_hours = st.slider("Daily Study Hours", 0, 12, 3)
    social_hours = st.slider("Social Media Hours", 0, 12, 2)
    
    # Dummy prediction formula
    pred_score = 50 + study_hours*5 - social_hours*2 + (age-18)*1.5
    if st.button("Predict"):
        st.success(f"Predicted Study Efficiency Score: {pred_score:.2f}/100")

# ------------------- Insights -------------------
with tab5:
    st.title("💡 Insights & Recommendations")
    st.write("- Study >5 hrs/day for better results")
    st.write("- Limit social media usage <3 hrs/day")
    st.write("- Consider university transport for time efficiency")
    st.write("- Top performing programs: ", data.groupby("Program")["StudyHours"].mean().idxmax())

# ------------------- ML Model -------------------
with tab6:
    st.title("⚡ ML Model Training")
    target_col = st.selectbox("Select target column", data.columns)
    feature_cols = st.multiselect("Select features", [col for col in data.columns if col != target_col])

    if st.button("Train Model") and feature_cols:
        X = data[feature_cols]
        y = data[target_col]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)

        st.success(f"Model trained! R² Score on Test Data: {score:.2f}")

        # Feature importance
        fi = pd.DataFrame({'Feature': feature_cols, 'Importance': model.feature_importances_}).sort_values(by="Importance", ascending=False)
        st.subheader("Feature Importance")
        st.bar_chart(fi.set_index("Feature"))

        # Save model
        joblib.dump(model, "rf_model.pkl")
        st.write("Model saved as rf_model.pkl")
