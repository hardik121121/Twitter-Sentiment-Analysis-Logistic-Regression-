# Twitter - Tweets Sentiment Analysis Model 🧠

This project is a **Twitter Sentiment Analysis** model that predicts the sentiment of a tweet as either **positive** or **negative**. Leveraging Natural Language Processing (NLP) techniques and a **Logistic Regression** classifier, the model analyzes tweet content and classifies the sentiment. A Streamlit web application is also created for real-time prediction and visualization of model performance.

---

## 📊 **Project Overview**
The goal of this project is to develop a sentiment analysis model that can classify tweets into positive and negative sentiments. The model uses a dataset of **1.6 million tweets** and applies various NLP techniques for preprocessing, feature extraction, and classification.

---

## 🧩 **How It Works**

### 1. **Data Collection** 📥
- The dataset contains **1.6 million tweets** scraped from Twitter, providing a diverse set of inputs for training the model.

### 2. **Preprocessing** 🔄
- Tweets are cleaned by:
  - Removing **special characters** and **stopwords** ❌.
  - Converting the text to **lowercase**.
  - **Stemming** the words using the **Porter Stemmer** 💬 to reduce words to their root forms.

### 3. **Vectorization** 🔠
- The text data is transformed into a **numerical form** using **TF-IDF Vectorizer** 📈, which calculates the importance of each word based on frequency and relevance.

### 4. **Modeling** 🤖
- The **Logistic Regression** model is trained on the cleaned and vectorized data to predict the sentiment of tweets.

### 5. **Model Accuracy** 📏
- **Training Accuracy**: XX%
- **Testing Accuracy**: XX%

---

## ⚙️ **Features**
- **Text Input** ✍️: Users can input their own tweets, and the model will predict whether the sentiment is positive or negative.
- **Model Performance** 📈: The app visualizes the performance of the model, displaying both the **training accuracy** and **testing accuracy**.

---

## 🚀 **Technologies Used**
- **Python** 🐍: The primary programming language for implementing the model.
- **Streamlit** 🌐: Framework used to build the web application for real-time predictions.
- **Scikit-learn** 🧪: For building the logistic regression model.
- **TF-IDF** 🔢: For converting text data into numerical format based on word importance.
- **Logistic Regression** 📉: For sentiment classification.
- **Matplotlib** 📊 and **Seaborn** 📈: Used for visualizations in the app.

---

## 📝 **How to Run the Application**

### Prerequisites
Make sure you have the following installed:
- **Python 3.x**
- **Streamlit**
- **Scikit-learn**
- **Matplotlib**
- **Seaborn**

You can install the necessary libraries using the following commands:

```bash
pip install streamlit scikit-learn matplotlib seaborn
```

### Running the App

1. Clone the repository:

```bash
git clone https://github.com/your-username/twitter-sentiment-analysis.git
```

2. Navigate to the project directory:

```bash
cd twitter-sentiment-analysis
```

3. Run the Streamlit app:

```bash
streamlit run app.py
```

4. The app will launch in your default browser, allowing you to input tweets and visualize their predicted sentiment in real time.

---

## 🧑‍💻 **Model Evaluation**
Once the model is trained, the performance is evaluated using **accuracy** as the metric. The model is evaluated on a separate test dataset to assess how well it generalizes to new, unseen data.

---

## ❤️ **Contributors**
- **Hardik Arora**: Creator and developer of the sentiment analysis model and Streamlit app.

---

## 🔧 **Future Enhancements**
- Extend the model to handle **multi-class sentiment analysis** (e.g., positive, negative, neutral).
- Implement **deep learning models** such as LSTM or BERT for improved accuracy.
- Add **real-time data collection** from Twitter to provide predictions on live tweets.

---

## 📄 **License**
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
