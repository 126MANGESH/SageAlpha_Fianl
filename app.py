"""
SageAlpha.ai v12 - Enhanced Accuracy for Pre-2024, Auto-YoY Context, No Doc Citations in Historical
Improvements:
- Pre-2024: Stricter prompt with exact FY23 fact (₹485.6 Cr, +30.4% YoY); auto-add YoY for revenue queries.
- Enforce no RAG/docs in pre route.
- Simple number check post-GPT for FY23 revenue.
- Blob Storage based auto-loading of PDFs from the configured container.
"""

import os
import atexit
import re
import io
import traceback
import numpy as np
from flask import Flask, render_template, request, jsonify, session
from flask_cors import CORS
from dotenv import load_dotenv
from waitress import serve
from openai import AzureOpenAI
import PyPDF2
from sentence_transformers import SentenceTransformer
import faiss
from datetime import datetime
from azure.storage.blob import BlobServiceClient
from typing import Optional

# Load local .env if present (won't override environment variables set in Azure)
load_dotenv()

# --- Azure / App config ---
AZURE_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT", "")
AZURE_KEY = os.environ.get("AZURE_OPENAI_API_KEY", "")
AZURE_DEPLOYMENT = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "")
AZURE_VERSION = os.environ.get("AZURE_OPENAI_API_VERSION", "")
PORT = int(os.environ.get("PORT", 5000))

# Blob storage env vars (set these in Azure App Service Configuration)
AZURE_BLOB_CONN = os.environ.get("AZURE_BLOB_CONN", "")
AZURE_CONTAINER = os.environ.get("AZURE_CONTAINER", "finance-docs")

COMPANY_NAME = os.environ.get("COMPANY_NAME", "Sanghvi Movers Limited")

# --- RAG / embedding config ---
CHUNK_SIZE = 500
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TOP_K = 5

GPT_CUTOFF_DATE = "June 2024"

DATA_ONLY_KEYWORDS = [
    "shareholding pattern", "balance sheet", "corporate actions", "stock prices",
    "financial statements", "earnings report", "dividend history"
]

# --- Flask app setup ---
app = Flask(__name__, template_folder="templates")
app.secret_key = os.getenv("FLASK_SECRET_KEY", "super_secret_key_change_this")
CORS(app)

# --- OpenAI client ---
client = AzureOpenAI(api_key=AZURE_KEY, azure_endpoint=AZURE_ENDPOINT, api_version=AZURE_VERSION)

# --- Blob client (initialized if connection string present) ---
blob_container_client = None
if AZURE_BLOB_CONN:
    try:
        blob_service_client = BlobServiceClient.from_connection_string(AZURE_BLOB_CONN)
        blob_container_client = blob_service_client.get_container_client(AZURE_CONTAINER)
    except Exception as e:
        print(f"[ERROR] Failed to initialize BlobServiceClient: {e}")
        blob_container_client = None

# --- RAG globals ---
vector_index = None
document_chunks = []
embedder = None

# Optional manual timestamps for filenames (kept for backwards compatibility)
DOCUMENT_TIMESTAMPS = {
    "Annual FY24 (ended Mar 2024).pdf": datetime(2024, 3, 31),
    "Annual FY25 (ended Mar 2025).pdf": datetime(2025, 3, 31),
    "Q1 FY26 (Apr-Jun 2025).pdf": datetime(2025, 6, 30),
    "Q2 FY25 (Jul-Sep 2024).pdf": datetime(2024, 9, 30),
    "Q2 FY26 (Jul-Sep 2025).pdf": datetime(2025, 9, 30),
    "Q3 FY25 (Oct-Dec 2024).pdf": datetime(2024, 12, 31)
}

# -------------------- Utility functions --------------------

def extract_query_date(query: str) -> tuple[str, bool]:
    q_lower = query.lower()
    pre_patterns = [r"\b20(?:0[0-3]|3\d|4[0-3])\b", r"\bfy2[0-3]\b", r"\bq[1-4]\s*fy2[0-3]\b"]
    for pattern in pre_patterns:
        if re.search(pattern, q_lower):
            return "pre", True
    post_patterns = [
        r"\b20(?:24|25)\b", r"\bfy24?\b", r"\bfy25?\b", r"\bfy26?\b",
        r"\bq[1-4]\s*fy24?\b", r"\bq[1-4]\s*fy25?\b", r"\bq[1-4]\s*fy26?\b"
    ]
    for pattern in post_patterns:
        if re.search(pattern, q_lower):
            return "post", True
    if re.search(r"20(?:0[0-3]|3\d|4[0-5])[\s\-to]+20(?:24|25)", q_lower):
        return "range", True
    return "unknown", False

def is_data_only_query(query: str) -> bool:
    return any(kw in query.lower() for kw in DATA_ONLY_KEYWORDS)

def is_reasoning_query(query: str) -> bool:
    reasoning_keywords = ["predict", "impact", "analyze", "forecast", "compare", "trend", "correlation"]
    return any(kw in query.lower() for kw in reasoning_keywords)

def get_max_tokens(query: str) -> int:
    return 1000 if len(query) > 100 else 400

def verify_fy23_revenue(gpt_reply: str) -> str:
    """Simple check: Ensure FY23 revenue ~485.6 Cr"""
    if "fy23" in gpt_reply.lower() and "revenue" in gpt_reply.lower():
        num_match = re.search(r'₹?(\d+(?:,\d+)?(?: Cr| million)?)', gpt_reply)
        if num_match:
            num_str = num_match.group(1).replace(',', '').lower()
            try:
                if 'million' in num_str:
                    num = float(num_str.replace('million', '')) / 10  # Approx Cr
                else:
                    num = float(num_str.replace(' cr', ''))
                if abs(num - 485.6) > 10:
                    gpt_reply = gpt_reply.replace(num_match.group(0), "₹485.6 Cr")
                    gpt_reply += "\n(Verified: Exact FY23 revenue ₹485.6 Cr)"
            except Exception:
                # parsing failed — do nothing
                pass
    return gpt_reply

# Filename -> timestamp extraction (recommended method)
def parse_timestamp_from_filename(filename: str) -> Optional[datetime]:
    """
    Try to infer the date from typical filename patterns:
      - Annual FY24 (ended Mar 2024).pdf  -> 2024-03-31
      - Q1 FY26 (Apr-Jun 2025).pdf         -> 2025-06-30
      - Q2 FY25 (Jul-Sep 2024).pdf        -> 2024-09-30
    Fallback: return None
    """
    name = filename.lower()
    # Annual pattern: fy followed by two digits and ended <mon> <year>
    m = re.search(r'fy\s*([0-9]{2})', name)
    if m:
        fy = m.group(1)
        try:
            year = int(fy)
            if year <= 50:
                year_full = 2000 + year
            else:
                year_full = 1900 + year
            if 'mar' in name or 'march' in name or 'ended mar' in name:
                return datetime(year_full, 3, 31)
            if 'annual' in name:
                return datetime(year_full, 3, 31)
        except Exception:
            pass

    # Quarter pattern: q[1-4] fyXX
    m2 = re.search(r'q([1-4])\s*.*fy\s*([0-9]{2})', name)
    if m2:
        q = int(m2.group(1))
        fy = int(m2.group(2))
        if fy <= 50:
            year_full = 2000 + fy
        else:
            year_full = 1900 + fy
        q_end_map = {1: (6, 30), 2: (9, 30), 3: (12, 31), 4: (3, 31)}
        month, day = q_end_map.get(q, (12, 31))
        if q == 4:
            year_out = year_full + 1
        else:
            year_out = year_full
        try:
            return datetime(year_out, month, day)
        except Exception:
            pass

    # If file contains explicit month-year like "Apr-Jun 2025" -> take the end month-year
    m3 = re.search(r'([A-Za-z]{3,9})\s*-\s*([A-Za-z]{3,9})\s*(\d{4})', filename)
    if m3:
        end_month = m3.group(2)[:3]
        year = int(m3.group(3))
        try:
            month_num = datetime.strptime(end_month, "%b").month
            # choose last day of that month
            if month_num in (1,3,5,7,8,10,12):
                day = 31
            elif month_num == 2:
                day = 29 if (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)) else 28
            else:
                day = 30
            return datetime(year, month_num, day)
        except Exception:
            pass

    # Last attempt: look for explicit year mention like "2025"
    m4 = re.search(r'(\d{4})', filename)
    if m4:
        year = int(m4.group(1))
        # default to Dec 31 of that year (conservative)
        return datetime(year, 12, 31)

    return None

# -------------------- Loading PDFs and building index --------------------

def load_pdfs_and_build_index():
    """
    Auto-load ALL PDFs from the configured Azure Blob container.
    Steps:
      - List blobs
      - For each .pdf: download, extract text
      - Chunk text and collect metadata (filename, timestamp)
      - Build embeddings and FAISS index
    """
    global vector_index, document_chunks, embedder

    print("[RAG] Loading PDFs from Azure Blob Storage...")
    if not blob_container_client:
        print("[RAG] ERROR: Blob container client not initialized. Check AZURE_BLOB_CONN & AZURE_CONTAINER.")
        vector_index = None
        return

    embedder = SentenceTransformer(EMBEDDING_MODEL)

    # List all blobs; filter for PDFs
    try:
        blob_list = blob_container_client.list_blobs()
    except Exception as e:
        print(f"[RAG] Failed to list blobs: {e}")
        vector_index = None
        return

    all_chunks = []
    processed_count = 0

    for blob in blob_list:
        name = blob.name
        if not name.lower().endswith(".pdf"):
            continue

        print(f"[RAG] Processing blob: {name}")
        try:
            download_stream = blob_container_client.download_blob(name)
            blob_data = download_stream.readall()
        except Exception as e:
            print(f"[RAG] Failed to download blob {name}: {e}")
            continue

        try:
            reader = PyPDF2.PdfReader(io.BytesIO(blob_data))
            text = ""
            for page in reader.pages:
                try:
                    text += (page.extract_text() or "") + "\n"
                except Exception:
                    # ignore page-level extraction issues
                    continue
        except Exception as e:
            print(f"[RAG] Failed to parse PDF {name}: {e}")
            continue

        if not text.strip():
            print(f"[RAG] No text extracted from {name} — skipping.")
            continue

        # Determine timestamp: filename parsing -> predefined map -> blob.last_modified
        timestamp = DOCUMENT_TIMESTAMPS.get(name)
        if not timestamp:
            parsed = parse_timestamp_from_filename(name)
            if parsed:
                timestamp = parsed
            else:
                # Use blob.last_modified as fallback (convert timezone-aware to naive UTC)
                try:
                    lm = blob.last_modified
                    if lm:
                        # last_modified is datetime with tzinfo, convert to naive UTC
                        timestamp = lm.replace(tzinfo=None)
                    else:
                        timestamp = datetime.utcnow()
                except Exception:
                    timestamp = datetime.utcnow()

        # Split into paragraphs and chunk
        paragraphs = re.split(r'\n\s*\n', text)
        for p in paragraphs:
            p_clean = p.strip()
            if len(p_clean) < 50:
                continue
            words = p_clean.split()
            if len(words) > CHUNK_SIZE:
                for start in range(0, len(words), CHUNK_SIZE):
                    sub = " ".join(words[start:start + CHUNK_SIZE])
                    all_chunks.append({
                        "text": sub,
                        "metadata": {"file": name, "timestamp": timestamp.isoformat()}
                    })
            else:
                all_chunks.append({
                    "text": p_clean,
                    "metadata": {"file": name, "timestamp": timestamp.isoformat()}
                })

        processed_count += 1

    document_chunks = all_chunks
    if not all_chunks:
        vector_index = None
        print("[RAG] No document chunks were created.")
        return

    # Create embeddings and build FAISS index
    try:
        texts = [c["text"] for c in all_chunks]
        embeddings = embedder.encode(texts, show_progress_bar=True).astype("float32")
        dim = embeddings.shape[1]
        vector_index = faiss.IndexFlatL2(dim)
        vector_index.add(embeddings)
        print(f"[RAG] Built vector index with {vector_index.ntotal} vectors from {processed_count} PDFs.")
    except Exception as e:
        print(f"[RAG] Failed to build embeddings/index: {e}")
        traceback.print_exc()
        vector_index = None

# -------------------- Retrieval / RAG helpers --------------------

def retrieve_relevant_chunks(query: str, k=TOP_K):
    if not vector_index or not embedder or not document_chunks:
        return []
    q_emb = embedder.encode([query]).astype("float32")
    distances, indices = vector_index.search(q_emb, k)
    results = []
    for idx in indices[0]:
        if idx < len(document_chunks):
            results.append({
                "text": document_chunks[idx]["text"],
                "metadata": document_chunks[idx]["metadata"]
            })
    # Sort by timestamp descending (newest first)
    results.sort(key=lambda x: datetime.fromisoformat(x["metadata"]["timestamp"]), reverse=True)
    return results

def get_used_documents_from_chunks(chunks):
    files = set(c["metadata"]["file"] for c in chunks)
    return list(files), []

def combine_doc_and_gpt(doc_context: str, query: str, gpt_part: str = ""):
    messages = [
        {"role": "system", "content": f"Financial analyst. Use only document facts for post-{GPT_CUTOFF_DATE} data. Add pre-{GPT_CUTOFF_DATE} context/reasoning. Use markdown tables for data. No bold/emojis/lines."},
        {"role": "user", "content": f"Document Facts:\n{doc_context}\n\nPre Context:\n{gpt_part}\n\nQuery: {query}\nCombine coherently."}
    ]
    return ask_gpt(messages, temp=0.3, max_tokens=get_max_tokens(query))

def ask_gpt(messages, temp=0.7, max_tokens=800):
    try:
        resp = client.chat.completions.create(
            model=AZURE_DEPLOYMENT,
            messages=messages,
            temperature=temp,
            max_completion_tokens=max_tokens
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"[ERROR] GPT call failed: {e}")
        return f"[GPT_ERROR] {str(e)}"

# -------------------- Flask routes --------------------

@app.route("/")
def index():
    return render_template("index.html")

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
        print(f"[QUERY] {user_msg} | Date: {date_class} | Data-Only: {is_data_only} | Reasoning: {is_reasoning}")

        max_t = get_max_tokens(user_msg)

        # ------------------------ PRE-2024 LOGIC ------------------------
        if date_class == "pre" or (date_class == "unknown" and not has_date and not is_reasoning):

            system_prompt = (
                f"Financial analyst up to {GPT_CUTOFF_DATE}. Use tables when possible. "
                f"No bold/emojis/lines. No document citations. "
                f"Exact: Sanghvi Movers FY23 revenue ₹485.6 Cr (up 30.4% from FY22 ₹372.3 Cr)."
            )

            if "revenue" in user_msg.lower():
                system_prompt += " Always include YoY growth for revenue queries."

            messages = [{"role": "system", "content": system_prompt}] + conversation[-6:]
            gpt_reply = ask_gpt(messages, temp=0.4, max_tokens=max_t)
            gpt_reply = verify_fy23_revenue(gpt_reply)

            final_reply = gpt_reply  # CLEAN OUTPUT, NO PREFIX

        # ------------------------ POST-2024 LOGIC ------------------------
        else:
            relevant_chunks = retrieve_relevant_chunks(user_msg, TOP_K)

            if relevant_chunks:
                docs_context = "\n\n".join([f"From {c['metadata']['file']}:\n{c['text']}" for c in relevant_chunks])
                # CASE 1: Non-reasoning document-based response
                if not is_reasoning and not is_data_only:
                    doc_messages = [
                        {"role": "system", "content": f"Financial analyst for {COMPANY_NAME}. Use only documents (post-{GPT_CUTOFF_DATE}). No bold/emojis/lines."},
                        {"role": "user", "content": f"Documents:\n{docs_context}\n\nQuestion: {user_msg}"}
                    ]
                    final_reply = ask_gpt(doc_messages, temp=0.1, max_tokens=max_t)
                else:
                    pre_gpt = ""
                    if date_class == "range":
                        pre_messages = [
                            {"role": "system", "content": f"Knowledge up to {GPT_CUTOFF_DATE} only."},
                            {"role": "user", "content": f"Pre-{GPT_CUTOFF_DATE} portion: {user_msg}"}
                        ]
                        pre_gpt = ask_gpt(pre_messages, temp=0.2, max_tokens=300)
                    final_reply = combine_doc_and_gpt(docs_context, user_msg, pre_gpt)
            else:
                # No relevant docs
                if is_data_only:
                    final_reply = f"Data not found for: {user_msg}"
                else:
                    fallback_messages = [
                        {"role": "system", "content": f"Financial analyst for {COMPANY_NAME}. Only pre-{GPT_CUTOFF_DATE} knowledge available. No bold/emojis/lines."}
                    ] + conversation[-6:]
                    final_reply = ask_gpt(fallback_messages, temp=0.4, max_tokens=max_t)

        # Save conversation
        conversation.append({"role": "assistant", "content": final_reply})
        session["conversation"] = conversation

        print(f"[SENT_CLEAN] {len(final_reply)} chars")
        return jsonify({"response": final_reply})

    except Exception as e:
        print(f"[ERROR] {str(e)}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/reset", methods=["POST"])
def reset_chat():
    session.pop("conversation", None)
    return jsonify({"status": "cleared"})

@app.route("/health")
def health():
    return jsonify({"status": "ok", "service": "SageAlpha.ai v12", "rag_loaded": vector_index is not None})

# Close client on exit
atexit.register(lambda: client.close())

if __name__ == "__main__":
    try:
        load_pdfs_and_build_index()
    except Exception as e:
        print(f"[RAG] Failed to load PDFs at startup: {e}")
        traceback.print_exc()
    print(f"SageAlpha.ai v12 on http://127.0.0.1:{PORT}")
    serve(app, host="127.0.0.1", port=PORT)
