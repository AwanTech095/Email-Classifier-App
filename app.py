import streamlit as st
import pandas as pd
import numpy as np
import fitz  # PyMuPDF
import docx
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

# Load model and vectorizer
model = joblib.load("email_classifier_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

label_map = model.classes_

# --- Styling ---
st.markdown("""
    <style>
    /* Main app background and text */
    .stApp {
        background-color: #1f2937;
        color: #f3f4f6;
        font-family: 'Segoe UI', sans-serif;
    }

    /* Headings */
    h1, h2, h3, h4, h5, h6 {
        color: #e5e7eb;
    }

    /* Paragraphs & base text */
    p, label, .stMarkdown {
        color: #d1d5db;
    }

    /* Text inputs & text areas */
    .stTextInput > div > div > input,
    .stTextArea textarea {
        background-color: #374151;
        color: #f9fafb;
        border-radius: 6px;
        border: 1px solid #4b5563;
        font-size: 15px;
        padding: 10px;
    }

    /* Buttons */
    .stButton > button {
        background-color: #3b82f6;
        color: #ffffff;
        border: none;
        border-radius: 6px;
        font-weight: 600;
        font-size: 15px;
        padding: 0.6em 1.3em;
        transition: 0.3s ease;
    }

    .stButton > button:hover {
        background-color: #2563eb;
        transform: scale(1.03);
    }

    /* Radio buttons, file uploaders, etc. */
    .stRadio label, .stFileUploader label, .stTextArea label {
        color: #f3f4f6;
        font-weight: 500;
    }

    /* Download button */
    .stDownloadButton > button {
        background-color: #374151;
        color: #f3f4f6;
        border: 1px solid #6b7280;
        border-radius: 6px;
        font-weight: 500;
    }

    .stDownloadButton > button:hover {
        background-color: #4b5563;
    }
    </style>
""", unsafe_allow_html=True)

# --- App Title ---
st.title("📧 Smart Email Classifier")
st.write("Classify your emails into categories like HR, Bills, Finance, Work, Promotions.")

# --- Helper Functions ---
def get_top_keywords(vector, feature_names, top_n=5):
    indices = np.argsort(vector.toarray()[0])[::-1]
    return [feature_names[i] for i in indices[:top_n]]

def suggest_reply(label):
    suggestions = {
        "Work": "Got it. I’ll get back to you shortly with an update.",
        "Bills": "Acknowledged. Will check and pay on time.",
        "Finance": "I'll review the financial report and respond soon.",
        "HR": "Thanks! I’ll follow up with HR accordingly.",
        "Promotions": "Noted. Will consider the offer.",
    }
    return suggestions.get(label, "No suggestion available.")

def extract_text_from_pdf(file):
    text = ""
    with fitz.open(stream=file.read(), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return text

def extract_text_from_docx(file):
    doc = docx.Document(file)
    return "\n".join([para.text for para in doc.paragraphs])

# --- Email Input ---
input_method = st.radio("Choose input method:", ["Paste Email Text", "Upload PDF or DOCX File"])

email_text = ""

if input_method == "Paste Email Text":
    email_text = st.text_area("Paste your email here:", height=200)
    trigger = st.button("Classify Email")

elif input_method == "Upload PDF or DOCX File":
    uploaded_file = st.file_uploader("Upload PDF or DOCX", type=["pdf", "docx"])
    trigger = st.button("Classify Extracted Email")

    if uploaded_file:
        ext = uploaded_file.name.split(".")[-1]
        if ext == "pdf":
            email_text = extract_text_from_pdf(uploaded_file)
        elif ext == "docx":
            email_text = extract_text_from_docx(uploaded_file)
        else:
            st.error("Unsupported file type.")
            email_text = ""

        st.text_area("📄 Extracted Email Text", email_text[:1000], height=150)

# --- Classification Logic ---
if trigger and email_text.strip():
    tfidf_vec = vectorizer.transform([email_text])
    prediction = model.predict(tfidf_vec)[0]
    confidence = np.max(model.predict_proba(tfidf_vec)) * 100
    top_keywords = get_top_keywords(tfidf_vec, vectorizer.get_feature_names_out())

    # Display result
    st.success(f"🧠 **Prediction:** {prediction}")
    st.write(f"📊 **Confidence:** {confidence:.2f}%")

    if confidence < 60:
        st.warning("⚠️ Low confidence prediction. Please verify manually.")

    st.write(f"🔍 **Top Keywords:** {', '.join(top_keywords)}")
    st.info(f"✉️ **Suggested Reply:** {suggest_reply(prediction)}")

    # Download result
    result_df = pd.DataFrame({
        "Email": [email_text],
        "Predicted Label": [prediction],
        "Confidence (%)": [f"{confidence:.2f}"]
    })

    st.download_button(
        label="📥 Download Result as CSV",
        data=result_df.to_csv(index=False),
        file_name="email_prediction.csv",
        mime="text/csv"
    )
