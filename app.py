"""
SageAlpha.ai v12 - Enhanced Accuracy + Auto-YoY + Blob Storage RAG
Flask version (for Azure App Service with Gunicorn)
"""

import os
import atexit
import io
import re
import traceback
from datetime import datetime
from typing import Optional

import numpy as np
from flask import Flask, render_template, request, jsonify, session
from flask_cors import CORS
from dotenv import load_dotenv
from waitress import serve
from openai import AzureOpenAI
from azure.storage.blob import BlobServiceClient
from sentence_transformers import SentenceTransformer
import PyPDF2
import faiss

# Load .env if running locally
load_dotenv()

# -------------------- ENV + CONFIG --------------------

AZURE_ENDPOINT     = os.environ.get("AZURE_OPENAI_ENDPOINT", "")
AZURE_KEY          = os.environ.get("AZURE_OPENAI_API_KEY", "")
AZURE_DEPLOYMENT   = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "")
AZURE_VERSION      = os.environ.get("AZURE_OPENAI_API_VERSION", "")
AZURE_BLOB_CONN    = os.environ.get("AZURE_BLOB_CONN", "")
AZURE_CONTAINER    = os.environ.get("AZURE_CONTAINER", "finance-docs")

COMPANY_NAME       = os.environ.get("COMPANY_NAME", "Sanghvi Movers Limited")
PORT               = int(os.environ.get("PORT", 5000))

GPT_CUTOFF_DATE    = "June 2024"
EMBEDDING_MODEL    = "all-MiniLM-L6-v2"
CHUNK_SIZE         = 500
TOP_K              = 5

DATA_ONLY_KEYWORDS = [
    "shareholding pattern", "balance sheet", "corporate actions",
    "stock prices", "financial statements", "earnings report",
    "dividend history"
]

# -------------------- FLASK APP --------------------

app = Flask(__name__, template_folder="templates")
app.secret_key = os.getenv("FLASK_SECRET_KEY", "super_secret_key_change_this")
CORS(app)

# -------------------- GPT CLIENT --------------------

client = AzureOpenAI(
    api_key=AZURE_KEY,
    azure_endpoint=AZURE_ENDPOINT,
    api_version=AZURE_VERSION
)

# -------------------- BLOB CLIENT --------------------

blob_container_client = None
if AZURE_BLOB_CONN:
    try:
        blob_service_client = BlobServiceClient.from_connection_string(AZURE_BLOB_CONN)
        blob_container_client = blob_service_client.get_container_client(AZURE_CONTAINER)
    except Exception as e:
        print(f"[ERROR] Blob init failed: {e}")

# -------------------- RAG GLOBALS --------------------

vector_index = None
document_chunks = []
embedder = None

DOCUMENT_TIMESTAMPS = {
    "Annual FY24 (ended Mar 2024).pdf": datetime(2024, 3, 31),
    "Annual FY25 (ended Mar 2025).pdf": datetime(2025, 3, 31),
    "Q1 FY26 (Apr-Jun 2025).pdf": datetime(2025, 6, 30),
    "Q2 FY25 (Jul-Sep 2024).pdf": datetime(2024, 9, 30),
    "Q2 FY26 (Jul-Sep 2025).pdf": datetime(2025, 9, 30),
    "Q3 FY25 (Oct-Dec 2024).pdf": datetime(2024, 12, 31)
}

# -------------------- DATE HELPERS --------------------

def extract_query_date(query: str):
    q = query.lower()
    if re.search(r"\bfy2[0-3]\b", q) or re.search(r"\b20(?:0[0-3]|3\d|4[0-3])\b", q):
        return "pre", True
    if re.search(r"\b20(?:24|25)\b", q) or re.search(r"\bfy2[4-6]\b", q):
        return "post", True
    return "unknown", False

def is_data_only_query(q):
    return any(x in q.lower() for x in DATA_ONLY_KEYWORDS)

def is_reasoning_query(q):
    return any(x in q.lower() for x in ["predict", "impact", "analyze", "forecast", "compare"])

def get_max_tokens(q):
    return 1000 if len(q) > 100 else 400

# -------------------- PDF LOADING + INDEX --------------------

def parse_timestamp_from_filename(name: str) -> Optional[datetime]:
    # (same logic – unchanged)
    try:
        m = re.search(r"(\d{4})", name)
        if m:
            return datetime(int(m.group(1)), 12, 31)
    except:
        pass
    return None

def load_pdfs_and_build_index():
    global vector_index, embedder, document_chunks

    if not blob_container_client:
        print("[RAG] No blob client available.")
        return

    print("[RAG] Loading PDFs...")

    embedder = SentenceTransformer(EMBEDDING_MODEL)

    docs = []
    for blob in blob_container_client.list_blobs():
        if not blob.name.endswith(".pdf"):
            continue

        print(f"[RAG] Reading {blob.name}")

        stream = blob_container_client.download_blob(blob.name)
        reader = PyPDF2.PdfReader(io.BytesIO(stream.readall()))
        text = "\n".join((page.extract_text() or "") for page in reader.pages)

        ts = (
            DOCUMENT_TIMESTAMPS.get(blob.name)
            or parse_timestamp_from_filename(blob.name)
            or blob.last_modified.replace(tzinfo=None)
        )

        paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 50]
        for p in paragraphs:
            docs.append({"text": p, "metadata": {"file": blob.name, "timestamp": ts.isoformat()}})

    document_chunks = docs
    if not docs:
        print("[RAG] No docs found.")
        return

    embeddings = embedder.encode([d["text"] for d in docs]).astype("float32")
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    vector_index = index
    print(f"[RAG] Loaded {len(docs)} chunks.")

# -------------------- RAG SEARCH --------------------

def retrieve_relevant_chunks(query, k=TOP_K):
    if not vector_index:
        return []
    q_emb = embedder.encode([query]).astype("float32")
    _, idx = vector_index.search(q_emb, k)
    return [document_chunks[i] for i in idx[0] if i < len(document_chunks)]

# -------------------- GPT --------------------

def ask_gpt(messages, temp=0.5, max_tokens=600):
    try:
        r = client.chat.completions.create(
            model=AZURE_DEPLOYMENT,
            messages=messages,
            temperature=temp,
            max_completion_tokens=max_tokens
        )
        return r.choices[0].message.content.strip()
    except Exception as e:
        print("[GPT ERROR]", e)
        return "GPT error."

# -------------------- ROUTES --------------------

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()
        user_msg = (data.get("message") or "").strip()

        if not user_msg:
            return jsonify({"error": "Empty message"}), 400

        # Basic Q type detection
        date_class, has_date = extract_query_date(user_msg)
        is_data = is_data_only_query(user_msg)
        is_reason = is_reasoning_query(user_msg)

        # PRE 2024 (no RAG)
        if date_class == "pre":
            sys = (
                f"Financial analyst up to {GPT_CUTOFF_DATE}. "
                f"FY23 revenue ₹485.6 Cr (+30.4% YoY from ₹372.3 Cr)."
            )
            reply = ask_gpt(
                [{"role": "system", "content": sys},
                 {"role": "user", "content": user_msg}],
                temp=0.3,
                max_tokens=get_max_tokens(user_msg)
            )
            return jsonify({"response": reply})

        # POST 2024 (RAG)
        chunks = retrieve_relevant_chunks(user_msg)
        if chunks:
            docs = "\n\n".join([f"From {c['metadata']['file']}:\n{c['text']}" for c in chunks])
            sys = f"Financial analyst for {COMPANY_NAME}. Use ONLY provided document facts. No emojis/bold."
            reply = ask_gpt(
                [{"role": "system", "content": sys},
                 {"role": "user", "content": f"Documents:\n{docs}\n\nQuestion: {user_msg}"}],
                temp=0.2,
                max_tokens=get_max_tokens(user_msg)
            )
            return jsonify({"response": reply})

        return jsonify({"response": "No data found."})

    except Exception as e:
        print("[CHAT ERROR]", e)
        traceback.print_exc()
        return jsonify({"error": "Server error"}), 500

@app.route("/health")
def health():
    return jsonify({"status": "ok", "rag_loaded": vector_index is not None})

# -------------------- APP START (local only) --------------------

if __name__ == "__main__":
    try:
        load_pdfs_and_build_index()
    except Exception as e:
        print("[RAG INIT FAILED]", e)

    print(f"Running locally at http://127.0.0.1:{PORT}")
    serve(app, host="0.0.0.0", port=PORT)

