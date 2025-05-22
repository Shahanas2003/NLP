import nltk
import os

# Set custom NLTK data directory
NLTK_DATA_DIR = os.path.join(os.path.dirname(__file__), "nltk_data")
nltk.data.path.append(NLTK_DATA_DIR)

# Try importing required packages and handle missing ones
try:
    nltk.data.find("tokenizers/punkt")
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("punkt", download_dir=NLTK_DATA_DIR)
    nltk.download("stopwords", download_dir=NLTK_DATA_DIR)

import streamlit as st
import re
import numpy as np
import pandas as pd
from io import StringIO
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi
from textblob import TextBlob
from sklearn.metrics import precision_score, recall_score
from PyPDF2 import PdfReader
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# ==== Preprocessing Components ====
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess(text):
    tokens = nltk.word_tokenize(text.lower())
    tokens = [stemmer.stem(t) for t in tokens if t.isalnum() and t not in stop_words]
    return tokens

def chunk_text(text, chunk_size=100):
    words = nltk.word_tokenize(text)
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

# ==== Document Ingestion ====
def extract_text_from_pdf(uploaded_file):
    reader = PdfReader(uploaded_file)
    return "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])

def extract_text_from_txt(uploaded_file):
    return StringIO(uploaded_file.getvalue().decode("utf-8")).read()

# ==== Sentiment Analysis ====
def get_sentiment(text):
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0: return ("Positive", "green")
    elif polarity < 0: return ("Negative", "red")
    return ("Neutral", "gray")

# ==== Summarization (simple) ====
def summarize_text(text):
    try:
        sentences = nltk.sent_tokenize(text)
        return " ".join(sentences[:2]) if len(sentences) >= 2 else "Summary Unavailable"
    except Exception:
        return "Summary Unavailable"

# ==== IR Models ====
def build_tfidf_index(corpus):
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(corpus)
    return vectorizer, matrix

def retrieve_tfidf(query, vectorizer, matrix, k):
    q_vec = vectorizer.transform([query])
    scores = (matrix * q_vec.T).toarray().ravel()
    return np.argsort(scores)[::-1][:k], scores

def retrieve_bm25(query, bm25_model, tokenized_corpus, k):
    tokens = preprocess(query)
    scores = bm25_model.get_scores(tokens)
    return np.argsort(scores)[::-1][:k], scores

# ==== Evaluation ====
def evaluate(retrieved_ids, relevant_ids, k):
    y_true = [1 if i in relevant_ids else 0 for i in retrieved_ids[:k]]
    y_pred = [1]*k
    return precision_score(y_true, y_pred, zero_division=0), recall_score(y_true, y_pred, zero_division=0)

# ==== Streamlit UI ====
st.set_page_config(page_title="Chat with Documents - Classic IR", layout="wide")
st.title("Chat with Documents (Classical IR Edition)")

uploaded_files = st.file_uploader("Upload PDF or TXT documents", type=["pdf", "txt"], accept_multiple_files=True)
k = st.slider("Top-K Results", 1, 10, 3)
method = st.radio("Choose Retrieval Method", ["TF-IDF", "BM25"], horizontal=True)
query = st.text_input("Ask a question:")

# ==== Processing Uploaded Docs ====
if uploaded_files:
    all_chunks = []
    chunk_map = {}
    doc_index = 0

    for f in uploaded_files:
        if f.name.endswith(".pdf"):
            text = extract_text_from_pdf(f)
        else:
            text = extract_text_from_txt(f)

        chunks = chunk_text(text)
        for c in chunks:
            chunk_map[len(all_chunks)] = (f.name, c)
            all_chunks.append(c)
        doc_index += 1

    # Build IR Models
    tokenized_chunks = [preprocess(c) for c in all_chunks]
    bm25 = BM25Okapi(tokenized_chunks)
    tfidf_vec, tfidf_mat = build_tfidf_index(all_chunks)

    # ==== Query Handling ====
    if query:
        if method == "TF-IDF":
            top_ids, scores = retrieve_tfidf(query, tfidf_vec, tfidf_mat, k)
        else:
            top_ids, scores = retrieve_bm25(query, bm25, tokenized_chunks, k)

        st.subheader("Top-K Results")
        for i in top_ids:
            fname, chunk = chunk_map[i]
            sentiment, color = get_sentiment(chunk)
            highlighted = re.sub(f"(?i)({re.escape(query)})", r"**\\1**", chunk)
            summary = summarize_text(chunk)

            st.markdown(f"**Document:** {fname}")
            st.markdown(f":{color}[Sentiment: {sentiment}]")
            st.markdown(f"**Chunk:** {highlighted}")
            st.markdown(f"**Summary:** {summary}")
            st.markdown("---")

        # Dummy relevance assumption (for demo only)
        relevant_ids = [i for i in range(len(all_chunks)) if query.lower() in all_chunks[i].lower()]
        precision, recall = evaluate(top_ids, relevant_ids, k)
        st.metric("Precision@K", f"{precision:.2f}")
        st.metric("Recall@K", f"{recall:.2f}")

        with st.expander("View Analysis Logs"):
            st.json({
                "Query": query,
                "Method": method,
                "Retrieved IDs": list(map(int, top_ids)),
                "Relevant IDs": relevant_ids
            })

st.caption("Built with classical IR techniques (no LLMs, no RAG). Upload PDFs or TXT files to search!")
