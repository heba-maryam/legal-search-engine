import os
import gdown
import pickle

share_url = "https://drive.google.com/file/d/1IwN7j0OdA8OXLYolBdaYE7t1alwYU7uG/view?usp=sharing"

file_id = share_url.split("/d/")[1].split("/")[0]
url = f"https://drive.google.com/uc?id={file_id}"

output_path = "bm25_index.pkl"

if not os.path.exists(output_path):
    print("Downloading from Google Drive...")
    gdown.download(url, output_path, quiet=False)
    print("Download complete.")
else:
    print("File already exists.")

# Try loading the pickle file
try:
    with open(output_path, "rb") as f:
        bm25_index = pickle.load(f)
        print("Pickle file loaded successfully.")
except Exception as e:
    print("Failed to load pickle:", e)


import streamlit as st
import os
import torch
import json
import pickle
import pandas as pd
from rank_bm25 import BM25Okapi
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re
import nltk
import matplotlib.pyplot as plt
st.set_page_config(page_title="Legal Document Search", layout="wide")

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('punkt_tab', quiet=True)

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")
model_path = "./legalbert_finetuned"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if os.path.exists(model_path):
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
else:
    model = AutoModelForSequenceClassification.from_pretrained("nlpaueb/legal-bert-base-uncased", num_labels=2).to(device)

# --- Helper functions ---
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    stemmer = PorterStemmer()
    return [stemmer.stem(token) for token in tokens]

def search_bm25(query, bm25, documents, metadata, top_k=10):
    tokenized_query = preprocess_text(query)
    scores = bm25.get_scores(tokenized_query)
    top_k_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    results = []
    for idx in top_k_indices:
        results.append({
            "section": documents[idx],
            "section_id": metadata[idx]["section_id"],
            "score": scores[idx],
            "case_name": metadata[idx]["case_name"],
            "year": metadata[idx]["year"],
            "file_name": metadata[idx]["file_name"]
        })
    return results

def rerank_with_legalbert(query, bm25_results, model, tokenizer):
    model.eval()
    scores = []
    with torch.no_grad():
        for result in bm25_results:
            encoding = tokenizer(
                query,
                result["section"],
                truncation=True,
                max_length=256,
                padding="max_length",
                return_tensors="pt"
            )
            encoding = {k: v.to(device) for k, v in encoding.items()}
            outputs = model(**encoding)
            score = outputs.logits.softmax(dim=1)[0][1].item()
            scores.append(score)
    reranked = sorted(
        zip(scores, bm25_results),
        key=lambda x: x[0],
        reverse=True
    )
    return [{"score": s, **r} for s, r in reranked]

def load_bm25_index(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def load_json_dataset(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    documents = []
    metadata = []
    for entry in data:
        doc = entry.get("Document", "")
        file_name = entry.get("FileName", "unknown")
        petitioner = entry.get("Petitioner", "Unknown")
        respondent = entry.get("Respondent", "Unknown")
        date = entry.get("Date of Judgment", "Unknown")
        bench = entry.get("Bench", "Unknown")
        case_number = entry.get("Case Number", "Unknown")
        case_name = f"{petitioner} vs {respondent}"
        year = date.split("/")[-1] if "/" in date else "Unknown"

        # Split into sections
        words = doc.split()
        for i in range(0, len(words), 250):
            section = " ".join(words[i:i + 250])
            documents.append(section)
            metadata.append({
                "section": section,
                "section_id": f"{file_name}_sec{i}",
                "file_name": file_name,
                "case_name": case_name,
                "year": year,
                "Case Number": case_number,
                "Petitioner": petitioner,
                "Respondent": respondent,
                "Date of Judgment": date,
                "Bench": bench,
                "FileData": doc
            })
    return documents, metadata

# --- Load once ---
@st.cache_resource(show_spinner=True)
def initialize():
    documents, metadata = load_json_dataset("data/train.json")
    bm25 = load_bm25_index("bm25_index.pkl")
    return bm25, documents, metadata

def search(query, top_k=5):
    tokenized_query = preprocess_text(query)
    bm25 = load_bm25_index("bm25_index.pkl")
    scores = bm25.get_scores(tokenized_query)
    documents, metadata = load_json_dataset("data/train.json")
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    results = [(metadata[i], scores[i]) for i in top_indices]
    return results

# --- Streamlit UI ---


st.image("/Users/tanvipundir/Downloads/search_engine/image1.png", width=200, use_container_width=False)
st.title("Legal Document Retrieval using BM25 + LegalBERT")

query = st.text_area(
    "Enter a legal question/query:", 
    "", 
    height=100,  # You can increase if you want more vertical space
    placeholder="Type your legal query here..."
)

if query:
    with st.spinner("Processing..."):
        bm25, documents, metadata = initialize()
        results = search(query, top_k=5)
    

    for i, (entry, score) in enumerate(results):
        st.markdown(f"### 🔹 Result {i + 1} — Score: {score:.2f}")
        with st.expander("📄 Document Information", expanded=True):
            st.write(f"**Case Number:** {entry.get('Case Number', 'N/A')}")
            st.write(f"**Petitioner:** {entry.get('Petitioner', 'N/A')}")
            st.write(f"**Respondent:** {entry.get('Respondent', 'N/A')}")
            st.write(f"**Date of Judgment:** {entry.get('Date of Judgment', 'N/A')}")
            st.write(f"**Bench:** {entry.get('Bench', 'N/A')}")
        with st.expander("📘 Document Preview"):
            st.text(entry.get("FileData", "")[:1500] + "...")

