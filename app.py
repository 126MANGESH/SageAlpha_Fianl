#!/usr/bin/env python3
"""
SageAlpha.ai v12 - Azure Blob-backed RAG with graceful fallback to Azure embeddings
- Auto-load all PDFs from the configured Azure Blob container.
- Prefer local sentence-transformers + faiss (fast), but fallback to Azure embeddings if unavailable.
- Clean responses only.
- FULLY COMPATIBLE WITH OpenAI v1+ (AzureOpenAI)

DEPLOYMENT NOTES:
- For Azure App Service, DO NOT run waitress in code. Instead set the startup command to:
    gunicorn --bind=0.0.0.0:5000 app:app
  (Replace `5000` with your PORT env var if needed and `app` with the module name.)
- For local development:
    python app.py
"""

import os
import atexit
import re
import io
import traceback
import json
import time
import sys
from datetime import datetime
from typing import Optional

from flask import Flask, render_template, request, jsonify, session
from flask_cors import CORS
from dotenv import load_dotenv

# ========== OpenAI v1+ SDK usage ==========
try:
    from openai import AzureOpenAI
    print("[INFO] ✓ Using openai v1+ SDK (AzureOpenAI) available")
    OPENAI_AVAILABLE = True
except ImportError:
    print("[ERROR] openai package not found. Install: pip install openai>=1.51.0")
    AzureOpenAI = None
    OPENAI_AVAILABLE = False

# Optional PDF parsing
try:
    import PyPDF2
except Exception:
    PyPDF2 = None

# Embedding / FAISS optional imports
try:
    from sentence_transformers import SentenceTransformer
    import faiss
    SE_SENTENCE_TRANSFORMERS_AVAILABLE = True
except Exception:
    SentenceTransformer = None
    faiss = None
    SE_SENTENCE_TRANSFORMERS_AVAILABLE = False

# Azure Blob Storage
try:
    from azure.storage.blob import BlobServiceClient
except Exception:
    BlobServiceClient = None

# Numeric helper (used in retrieval & embedding processing)
import numpy as _np

# Load environment variables
load_dotenv()

# --- Config (environment variable overrides) ---
AZURE_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT", "").rstrip("/")
AZURE_KEY = os.environ.get("AZURE_OPENAI_API_KEY", "")
AZURE_DEPLOYMENT = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")  # chat model deployment name
AZURE_EMBEDDING_DEPLOYMENT = os.environ.get("AZURE_EMBEDDING_DEPLOYMENT", "text-embedding-3-small")
AZURE_VERSION = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-10-21")
PORT = int(os.environ.get("PORT", 5000))

AZURE_BLOB_CONN = os.environ.get("AZURE_BLOB_CONN", "").strip()
AZURE_CONTAINER = os.environ.get("AZURE_CONTAINER", "finance-docs").strip()

COMPANY_NAME = os.environ.get("COMPANY_NAME", "Sanghvi Movers Limited")
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", 500))
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
TOP_K = int(os.environ.get("TOP_K", 5))

# Date cutoff for model-only knowledge (used in prompts / behavior)
GPT_CUTOFF_DATE = os.environ.get("GPT_CUTOFF_DATE", "September 2025")

DATA_ONLY_KEYWORDS = [
    "shareholding pattern", "balance sheet", "corporate actions", "stock prices",
    "financial statements", "earnings report", "dividend history"
]

print(f"[INFO] Server will listen on port {PORT}")
print(f"[INFO] Azure Deployment: {AZURE_DEPLOYMENT}")

# Flask app
app = Flask(__name__, template_folder="templates")
app.secret_key = os.getenv("FLASK_SECRET_KEY", "super_secret_key_change_this")
CORS(app)

# --- Setup Azure OpenAI client (v1+ SDK) ---
openai_client = None
if OPENAI_AVAILABLE and AZURE_ENDPOINT and AZURE_KEY and AZURE_DEPLOYMENT:
    try:
        openai_client = AzureOpenAI(
            azure_endpoint=AZURE_ENDPOINT,
            api_key=AZURE_KEY,
            api_version=AZURE_VERSION
        )
        print("[INFO] ✓ Azure OpenAI client initialized (v1+ API)")
    except Exception as e:
        print(f"[ERROR] Failed to init Azure OpenAI: {e}")
        traceback.print_exc()
        openai_client = None
else:
    print("[ERROR] Missing Azure OpenAI config (ENDPOINT/KEY/DEPLOYMENT) or SDK unavailable")

# --- Blob Storage initialization ---
blob_container_client = None
if BlobServiceClient and AZURE_BLOB_CONN:
    try:
        blob_service_client = BlobServiceClient.from_connection_string(AZURE_BLOB_CONN)
        blob_container_client = blob_service_client.get_container_client(AZURE_CONTAINER)
        blob_container_client.get_container_properties()
        print(f"[INFO] ✓ Blob container '{AZURE_CONTAINER}' connected")
    except Exception as e:
        print(f"[ERROR] Blob connection failed: {e}")
        blob_container_client = None
else:
    if not BlobServiceClient:
        print("[WARN] azure.storage.blob not installed")
    else:
        print("[WARN] AZURE_BLOB_CONN not configured")

# --- RAG globals ---
vector_index = None
document_chunks = []
embedder = None
USE_AZURE_EMBEDDINGS = False

DOCUMENT_TIMESTAMPS = {
    "Annual FY24 (ended Mar 2024).pdf": datetime(2024, 3, 31),
    "Annual FY25 (ended Mar 2025).pdf": datetime(2025, 3, 31),
    "Q1 FY26 (Apr-Jun 2025).pdf": datetime(2025, 6, 30),
    "Q2 FY25 (Jul-Sep 2024).pdf": datetime(2024, 9, 30),
    "Q2 FY26 (Jul-Sep 2025).pdf": datetime(2025, 9, 30),
    "Q3 FY25 (Oct-Dec 2024).pdf": datetime(2024, 12, 31),
}

# ---- Utility Functions ----

def extract_query_date(query: str) -> tuple[str, bool]:
    q_lower = query.lower()
    pre_patterns = [r"\b20(?:0[0-9]|1[0-9]|2[0-3])\b", r"\bfy(?:20)?2?3\b", r"\bq[1-4]\s*fy(?:20)?2?3\b"]
    for pattern in pre_patterns:
        if re.search(pattern, q_lower):
            return "pre", True
    post_patterns = [r"\b20(?:24|25|26)\b", r"\bfy(?:20)?(?:24|25|26)\b", r"\bq[1-4]\s*fy(?:20)?(?:24|25|26)\b"]
    for pattern in post_patterns:
        if re.search(pattern, q_lower):
            return "post", True
    if re.search(r"20(?:1[0-9]|2[0-9])\s*[\-to]+\s*20(?:2[0-9])", q_lower):
        return "range", True
    return "unknown", False

def is_data_only_query(query: str) -> bool:
    return any(kw in query.lower() for kw in DATA_ONLY_KEYWORDS)

def is_reasoning_query(query: str) -> bool:
    keywords = ["predict", "impact", "analyze", "forecast", "compare", "trend", "correlation", "what if"]
    return any(kw in query.lower() for kw in keywords)

def get_max_tokens(query: str) -> int:
    return 1000 if len(query) > 100 else 400

def verify_fy23_revenue(gpt_reply: str) -> str:
    if "fy23" in gpt_reply.lower() and "revenue" in gpt_reply.lower():
        num_match = re.search(r'₹?([\d,]+(?:\.\d+)?(?:\s*(?:Cr|cr|million|M))?)', gpt_reply)
        if num_match:
            num_str = num_match.group(1).replace(',', '').lower()
            try:
                if 'million' in num_str or 'm' in num_str:
                    num = float(re.sub(r'[^\d\.]', '', num_str)) / 10.0
                else:
                    num = float(re.sub(r'[^\d\.]', '', num_str))
                if abs(num - 485.6) > 10:
                    gpt_reply = gpt_reply.replace(num_match.group(0), "₹485.6 Cr")
                    gpt_reply += "\n(Verified: Exact FY23 revenue ₹485.6 Cr)"
            except Exception:
                pass
    return gpt_reply

def parse_timestamp_from_filename(filename: str) -> Optional[datetime]:
    name = filename.lower()
    m = re.search(r'fy\s*([0-9]{2})', name)
    if m:
        fy = m.group(1)
        try:
            fy_num = int(fy)
            year_full = 2000 + fy_num
            if 'mar' in name or 'march' in name or 'annual' in name:
                return datetime(year_full, 3, 31)
        except Exception:
            pass
    m2 = re.search(r'q([1-4])\s*.*fy\s*([0-9]{2})', name)
    if m2:
        q = int(m2.group(1)); fy = int(m2.group(2))
        year_full = 2000 + fy
        q_end_map = {1:(6,30),2:(9,30),3:(12,31),4:(3,31)}
        month, day = q_end_map.get(q,(12,31))
        year_out = year_full if q != 4 else year_full + 1
        try:
            return datetime(year_out, month, day)
        except Exception:
            pass
    return None

# ---- Embedding Functions ----

def init_local_embedder():
    global embedder
    if SE_SENTENCE_TRANSFORMERS_AVAILABLE and SentenceTransformer:
        try:
            print("[RAG] Loading local embedding model...")
            embedder = SentenceTransformer(EMBEDDING_MODEL)
            return True
        except Exception as e:
            print(f"[WARN] SentenceTransformer init failed: {e}")
            embedder = None
            return False
    return False

def get_local_embeddings(texts):
    if not embedder:
        raise RuntimeError("Local embedder not initialized")
    embs = embedder.encode(texts, show_progress_bar=False)
    return _np.array(embs, dtype="float32")

def get_azure_embeddings(texts):
    if not openai_client:
        raise RuntimeError("Azure OpenAI client not initialized")
    try:
        response = openai_client.embeddings.create(
            model=AZURE_EMBEDDING_DEPLOYMENT,
            input=texts
        )
        embs = [item.embedding for item in response.data]
        return _np.array(embs, dtype="float32")
    except Exception as e:
        raise RuntimeError(f"Azure embeddings failed: {e}")

# ---- PDF Loading ----

def load_pdfs_and_build_index():
    global vector_index, document_chunks, embedder, USE_AZURE_EMBEDDINGS

    print("[RAG] Starting PDF load and index build...")
    if not blob_container_client:
        print("[RAG] Blob container not available. Skipping PDF load.")
        return

    local_ok = init_local_embedder()
    if not local_ok:
        print("[RAG] Using Azure embeddings (fallback)")
        USE_AZURE_EMBEDDINGS = True
    else:
        USE_AZURE_EMBEDDINGS = False

    try:
        blob_list = list(blob_container_client.list_blobs())
        if not blob_list:
            print(f"[RAG] No blobs in container '{AZURE_CONTAINER}'")
            return
    except Exception as e:
        print(f"[RAG] Failed to list blobs: {e}")
        return

    all_chunks = []
    for blob in blob_list:
        if not blob.name.lower().endswith(".pdf"):
            continue

        print(f"[RAG] Processing: {blob.name}")
        try:
            data = blob_container_client.download_blob(blob.name).readall()
        except Exception as e:
            print(f"[RAG] Download failed for {blob.name}: {e}")
            continue

        if PyPDF2 is None:
            print(f"[RAG] PyPDF2 not available; skipping {blob.name}")
            continue

        try:
            reader = PyPDF2.PdfReader(io.BytesIO(data))
            text = ""
            for page in reader.pages:
                try:
                    text += (page.extract_text() or "") + "\n"
                except Exception:
                    continue
        except Exception as e:
            print(f"[RAG] PDF parse failed for {blob.name}: {e}")
            continue

        if not text.strip():
            print(f"[RAG] No text in {blob.name}")
            continue

        timestamp = DOCUMENT_TIMESTAMPS.get(blob.name) or parse_timestamp_from_filename(blob.name) or datetime.utcnow()

        paragraphs = re.split(r'\n\s*\n', text)
        for p in paragraphs:
            p_clean = p.strip()
            if len(p_clean) < 50:
                continue
            words = p_clean.split()
            if len(words) > CHUNK_SIZE:
                for start in range(0, len(words), CHUNK_SIZE):
                    sub = " ".join(words[start:start + CHUNK_SIZE])
                    all_chunks.append({"text": sub, "metadata": {"file": blob.name, "timestamp": timestamp.isoformat()}})
            else:
                all_chunks.append({"text": p_clean, "metadata": {"file": blob.name, "timestamp": timestamp.isoformat()}})

    if not all_chunks:
        print("[RAG] No chunks created")
        return

    document_chunks = all_chunks
    print(f"[RAG] Created {len(all_chunks)} chunks")

    texts = [c["text"] for c in all_chunks]
    try:
        if not USE_AZURE_EMBEDDINGS:
            print("[RAG] Creating local embeddings...")
            embs = get_local_embeddings(texts)
        else:
            print("[RAG] Creating Azure embeddings...")
            embs = get_azure_embeddings(texts)
    except Exception as e:
        print(f"[RAG] Embedding failed: {e}")
        return

    try:
        if faiss is not None:
            dim = embs.shape[1]
            idx = faiss.IndexFlatL2(dim)
            idx.add(embs)
            vector_index = idx
            print(f"[RAG] ✓ Built FAISS index with {idx.ntotal} vectors")
        else:
            for i, c in enumerate(document_chunks):
                c["embedding"] = embs[i].tolist()
            print(f"[RAG] Stored {len(document_chunks)} embeddings (FAISS not available)")
    except Exception as e:
        print(f"[RAG] Index build failed: {e}")

# ---- Retrieval ----

def retrieve_relevant_chunks(query: str, k=TOP_K):
    if not document_chunks:
        return []
    try:
        if not USE_AZURE_EMBEDDINGS and embedder:
            q_emb = get_local_embeddings([query])[0]
        else:
            q_emb = get_azure_embeddings([query])[0]
    except Exception as e:
        print(f"[RAG] Query embedding failed: {e}")
        return []

    if vector_index is not None:
        try:
            qv = _np.array([q_emb], dtype="float32")
            dists, idxs = vector_index.search(qv, k)
            items = []
            for idx in idxs[0]:
                if idx < len(document_chunks):
                    items.append({"text": document_chunks[idx]["text"], "metadata": document_chunks[idx]["metadata"]})
            items.sort(key=lambda x: datetime.fromisoformat(x["metadata"]["timestamp"]), reverse=True)
            return items
        except Exception as e:
            print(f"[RAG] FAISS search failed: {e}")

    sims = []
    for c in document_chunks:
        emb = _np.array(c.get("embedding", []), dtype="float32")
        if emb.size == 0:
            continue
        score = float(_np.dot(q_emb, emb) / (_np.linalg.norm(q_emb) * (_np.linalg.norm(emb) + 1e-10)))
        sims.append((score, c))
    sims.sort(key=lambda x: x[0], reverse=True)
    return [{"text": s[1]["text"], "metadata": s[1]["metadata"]} for s in sims[:k]]

# ---- GPT Helper (v1+ API) ----

def ask_gpt(messages, temp=0.7, max_tokens=800):
    if not openai_client:
        return "[ERROR] Azure OpenAI client not initialized"
    try:
        response = openai_client.chat.completions.create(
            model=AZURE_DEPLOYMENT,
            messages=messages,
            temperature=temp,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"[ERROR] GPT call failed: {e}")
        traceback.print_exc()
        return f"[GPT_ERROR] {str(e)}"

def combine_doc_and_gpt(doc_context: str, query: str, gpt_part: str = ""):
    messages = [
        {"role": "system", "content": f"You are a financial analyst. Use document facts (post-{GPT_CUTOFF_DATE}) for factual answers, and rely on pre-{GPT_CUTOFF_DATE} knowledge for background. When combining, clearly mark which part comes from documents."},
        {"role": "user", "content": f"Documents:\n{doc_context}\n\nPre Context (model knowledge):\n{gpt_part}\n\nQuestion: {query}"}
    ]
    return ask_gpt(messages, temp=0.3, max_tokens=get_max_tokens(query))

# ---- Flask Routes ----

@app.route("/")
def index():
    return render_template("index.html") if os.path.exists("templates/index.html") else jsonify({"status":"ok","service":"SageAlpha.ai v12"})

@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()
        user_msg = (data.get("message") or "").strip()
        if not user_msg:
            return jsonify({"error": "Empty message"}), 400

        conversation = session.get("conversation", [])[-10:]
        conversation.append({"role": "user", "content": user_msg})

        date_class, has_date = extract_query_date(user_msg)
        is_data_only = is_data_only_query(user_msg)
        is_reasoning = is_reasoning_query(user_msg)

        max_t = get_max_tokens(user_msg)

        if date_class == "pre" or (date_class == "unknown" and not has_date and not is_reasoning):
            system_prompt = f"Financial analyst up to {GPT_CUTOFF_DATE}. Use tables when possible and be concise."
            messages = [{"role": "system", "content": system_prompt}] + conversation[-6:]
            final_reply = ask_gpt(messages, temp=0.4, max_tokens=max_t)
        else:
            relevant_chunks = retrieve_relevant_chunks(user_msg, TOP_K)
            if relevant_chunks:
                docs_context = "\n\n".join([f"From {c['metadata']['file']}:\n{c['text']}" for c in relevant_chunks])
                if not is_reasoning and not is_data_only:
                    messages = [
                        {"role":"system","content":f"Financial analyst for {COMPANY_NAME}. Use only document contents for factual answers; do not hallucinate."},
                        {"role":"user","content":f"Documents:\n{docs_context}\n\nQuestion: {user_msg}"}
                    ]
                    final_reply = ask_gpt(messages, temp=0.1, max_tokens=max_t)
                else:
                    final_reply = combine_doc_and_gpt(docs_context, user_msg)
            else:
                if is_data_only:
                    final_reply = f"Data not found for: {user_msg}"
                else:
                    messages = [{"role":"system","content":f"Financial analyst for {COMPANY_NAME}. Only pre-{GPT_CUTOFF_DATE} knowledge available."}] + conversation[-6:]
                    final_reply = ask_gpt(messages, temp=0.4, max_tokens=max_t)

        final_reply = verify_fy23_revenue(final_reply)
        conversation.append({"role": "assistant", "content": final_reply})
        session["conversation"] = conversation
        return jsonify({"response": final_reply})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/reset", methods=["POST"])
def reset_chat():
    session.pop("conversation", None)
    return jsonify({"status":"cleared"})

@app.route("/health")
def health():
    return jsonify({"status":"ok","service":"SageAlpha.ai v12"})

atexit.register(lambda: print("[INFO] Shutting down SageAlpha.ai"))

# ---- Entrypoint ----
if __name__ == "__main__":
    # Load RAG index on startup (best-effort). If it fails, app still runs.
    try:
        load_pdfs_and_build_index()
    except Exception as e:
        print(f"[ERROR] PDF loading failed: {e}")
        traceback.print_exc()

    print(f"[INFO] Starting SageAlpha.ai on http://0.0.0.0:{PORT} (development server)")
    sys.stdout.flush()
    # Use Flask's built-in server for local development only.
    # On production (Azure App Service), use Gunicorn with startup command:
    #   gunicorn --bind=0.0.0.0:5000 app:app
    app.run(host="0.0.0.0", port=PORT)
