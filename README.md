# 📧 Email Classifier App

A Streamlit-based machine learning application that classifies email content into meaningful categories such as **Work**, **HR**, **Finance**, **Bills**, and **Promotions**. The app helps users quickly understand the nature of their emails and suggests possible actions or replies.

---

## 🚀 Features

- 🔍 **Email Text Classification** using Logistic Regression & TF-IDF
- 🧠 Categories include: `Work`, `HR`, `Finance`, `Bills`, `Promotions`
- 📎 Upload support for `.pdf` and `.docx` email content
- 📊 Display of prediction confidence and top keywords
- 💬 Suggested reply messages based on category
- 🖥️ Easy-to-use Streamlit UI

---

## 🛠️ Tech Stack

- Python 3
- Scikit-learn
- Pandas / NumPy
- SpaCy (for lemmatization)
- NLTK (for stopword removal)
- Streamlit (for deployment)
- TF-IDF for feature extraction

---

## 🧪 How It Works

1. **Text Preprocessing**: Removes HTML tags, punctuation, and stopwords.
2. **Lemmatization**: Normalizes words using SpaCy.
3. **TF-IDF Vectorization**: Transforms text into numeric features.
4. **Model Prediction**: Classifies the email using Logistic Regression.
5. **Result Display**: Shows predicted label, confidence, keywords, and reply suggestion.

---
