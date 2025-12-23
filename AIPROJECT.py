import os
import re
import string
import joblib

import pandas as pd
import streamlit as st
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from newspaper import Article

# Streamlit Config
st.set_page_config(page_title="Fake News Detector (BERT)", page_icon="📰", layout="centered")
st.title("📰 Fake News Detection System (BERT Version)")
st.caption("Detect whether a news article is **Real** or **Fake** using semantic embeddings.")
st.divider()

# Text Cleaning Function
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(f'[{re.escape(string.punctuation)}]', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Constants
MODEL_PATH = "fake_news_model_bert.joblib"
LOCAL_EMBEDDING_PATH = "all-MiniLM-L6-v2" 
EMBEDDING_MODEL_SOURCE = LOCAL_EMBEDDING_PATH if os.path.isdir(LOCAL_EMBEDDING_PATH) else "all-MiniLM-L6-v2"

# Cache Model and Embedder
@st.cache_resource
def load_embedder(model_source):
    try:
        # This will load from the local path or try to download if model_source is the HF name
        return SentenceTransformer(model_source)
    except Exception as e:
        st.error(f"❌ Could not load or download embedding model. Error: {e}")
        st.info("💡 **Action Needed:** Please ensure you have the model folder named 'all-MiniLM-L6-v2' in this script's directory, or fix your network connection to download it.")
        st.stop()
        
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)

# Load the model and embedder
embedder = load_embedder(EMBEDDING_MODEL_SOURCE)
model = load_model()

# Train Model (if not already trained)
if model is None:
    st.info("Training model with BERT embeddings... please wait ⏳")

    if not os.path.exists("Fake.csv") or not os.path.exists("True.csv"):
        st.error("❌ Missing `Fake.csv` or `True.csv` dataset files in the current directory.")
        st.stop()

    df_fake = pd.read_csv("Fake.csv")
    df_true = pd.read_csv("True.csv")
    df_fake["class"], df_true["class"] = 0, 1
    df = pd.concat([df_fake, df_true], ignore_index=True)

    for col in ["title", "subject", "date"]:
        if col in df.columns:
            df.drop(columns=col, inplace=True)

    df["text"] = df["text"].astype(str).apply(clean_text)

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["class"], test_size=0.25, random_state=42, stratify=df["class"]
    )

    with st.spinner("Generating BERT embeddings... This may take a long time on first run."):
        X_train_emb = embedder.encode(X_train.tolist(), show_progress_bar=True)
        X_test_emb = embedder.encode(X_test.tolist(), show_progress_bar=True)

    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X_train_emb, y_train)

    joblib.dump(model, MODEL_PATH)
    st.success("✅ Model training complete and saved using BERT embeddings!")

# Input Section
input_type = st.radio("Choose Input Type:", ("📝 Text Input", "🌐 Article URL"))
article_text = ""

if input_type == "📝 Text Input":
    article_text = st.text_area("Enter News Article Here", height=200)
else:
    url = st.text_input("Enter Article URL")
    if url:
        try:
            with st.spinner("Extracting article text..."):
                article = Article(url)
                article.download()
                article.parse()
                article_text = article.text
            if len(article_text.strip()) > 0:
                st.success("✅ Extracted article text successfully.")
            else:
                st.warning("⚠️ Could not extract text from this URL. Try another.")
        except Exception as e:
            st.error(f"Error fetching article: {e}")

# Optionally show extracted text
if article_text.strip():
    if st.checkbox("📄 Show extracted text"):
        st.write(article_text[:2000])

st.divider()

# Prediction Section
if st.button("🔍 Check if News is Real or Fake"):
    if not article_text.strip():
        st.warning("⚠️ Please enter or load some text first.")
    else:
        with st.spinner("Analyzing article..."):
            cleaned = clean_text(article_text)
            cleaned = " ".join(cleaned.split()[:500])  # Limit to first 500 words
            emb = embedder.encode([cleaned])
            pred = model.predict(emb)[0]
            prob = model.predict_proba(emb)[0][pred]

        st.markdown("---")
        if prob < 0.55:
            st.warning(f"🤔 The model is uncertain (Confidence: {prob:.2%})")
        elif pred == 1:
            st.success(f"🟢 The news appears **REAL** (Confidence: {prob:.2%})")
        else:
            st.error(f"🔴 The news appears **FAKE** (Confidence: {prob:.2%})")
        st.markdown("---")
        st.caption("Predictions are based on linguistic meaning, not external fact-checking.")
