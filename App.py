import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

# Title and Description
st.title("Twitter Sentiment Analysis App 🎯")
st.markdown("This app analyzes the sentiment of tweets as **Positive** 😊 or **Negative** 😞 using Machine Learning models.")

# Load and Prepare Data
st.subheader("Load Dataset 📂")
try:
    twitter_data = pd.read_csv("training.1600000.processed.noemoticon.csv", encoding="ISO-8859-1")
    twitter_data.columns = ['target', 'ids', 'date', 'flag', 'user', 'text']
    st.success("Data loaded successfully!")
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# Test with Your Own Tweet
st.subheader("Test with Your Own Tweet ✍️")
user_input = st.text_input("Enter a tweet to analyze sentiment:")
if user_input:
    try:
        vectorizer = TfidfVectorizer()
        X = twitter_data['text']
        y = twitter_data['target']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train_tfidf = vectorizer.fit_transform(X_train)

        model = LogisticRegression()
        model.fit(X_train_tfidf, y_train)

        user_input_tfidf = vectorizer.transform([user_input])
        prediction = model.predict(user_input_tfidf)[0]
        sentiment = "😊 Positive" if prediction == 4 else "😞 Negative"
        st.success(f"Predicted Sentiment: {sentiment}")

        # Emoji Visualization
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, sentiment, fontsize=30, ha='center', va='center', color='green' if prediction == 4 else 'red')
        ax.axis('off')
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error analyzing sentiment: {e}")

# Show Raw Data
if st.checkbox("Show raw data 🗃️"):
    st.write(twitter_data.head())

# Data Visualization
st.subheader("Dataset Overview 📊")
sentiment_counts = twitter_data['target'].value_counts()
sentiment_labels = ["Negative (0)", "Positive (4)"]
fig, ax = plt.subplots()
ax.pie(sentiment_counts, labels=sentiment_labels, autopct='%1.1f%%', startangle=90, colors=['#ff9999','#66b3ff'])
ax.axis('equal')
st.pyplot(fig)

# Prepare Data for Modeling
X_train_tfidf = vectorizer.transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Model Evaluation
st.subheader("Model Performance ✅")
y_pred = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)
st.metric(label="Accuracy", value=f"{accuracy:.2f}")

# Classification Report
st.text("Classification Report:")
st.text(classification_report(y_test, y_pred))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=["Negative", "Positive"], yticklabels=["Negative", "Positive"])
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
st.pyplot(fig)

# Positive input - 
#"Embracing every challenge with a positive mindset! 🌟 Growth happens when we step out of our comfort zones. Let’s keep pushing forward, one step at a time. 💪 #PositiveVibes #KeepGrowing #StayMotivated"

# Negative input -
#"Some days feel tougher than others, but it's okay to take a step back and recharge. 💭 Remember, it's okay to not be okay sometimes. #Struggling #MentalHealthMatters"