# 📰 Fake News Detection System (BERT + Logistic Regression)

This project detects whether a news article is **Real** or **Fake** using
`Sentence-BERT embeddings` and a `Logistic Regression` classifier.

The system supports:
- 📝 Manual text input
- 🌐 Automatic news extraction via URL
- 🎯 Confidence percentage
- ⚡ Fast prediction using semantic embeddings
- 🧠 Auto training if model not found

Built using **Python**, **Streamlit**, **Sentence Transformers**, and **Scikit-Learn**.

---

## 🚀 Demo Screenshot
(Add screenshot here)

---

## 📌 Features
✔ Detects Fake vs Real news  
✔ Confidence Percentage  
✔ URL support – auto extracts text using Newspaper3k  
✔ Cleans text using NLP preprocessing  
✔ Uses BERT semantic embeddings (`all-MiniLM-L6-v2`)  
✔ Automatically trains model if not available  
✔ Cached embedding & model loading for speed  

---

## 🛠️ Tech Stack
- Python
- Streamlit
- Sentence Transformers (`all-MiniLM-L6-v2`)
- Logistic Regression
- Newspaper3k
- Pandas
- Scikit-Learn
- Joblib

---

## 📂 Dataset Requirement
Place the following files in the **same directory** as the script:

