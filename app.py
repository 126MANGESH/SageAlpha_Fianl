"""
SageAlpha.ai v12 - Enhanced Accuracy + Auto-YoY + Local PDF RAG
Flask version (for Azure App Service with Gunicorn)
IMPROVED: Better PDF chunking strategy for financial documents
"""

import os
import atexit
import io
import re
import traceback
from datetime import datetime
from typing import Optional
from pathlib import Path

import numpy as np
from flask import Flask, render_template, request, jsonify, session
from flask_cors import CORS
from dotenv import load_dotenv
from waitress import serve
from openai import AzureOpenAI
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

COMPANY_NAME       = os.environ.get("COMPANY_NAME", "Sanghvi Movers Limited")
PORT               = int(os.environ.get("PORT", 5000))

# -------- LOCAL PDF PATH --------
PDF_FOLDER_PATH = os.environ.get("PDF_FOLDER_PATH", "assets/Sanghvi Movers Limited")

GPT_CUTOFF_DATE    = "June 2024"
EMBEDDING_MODEL    = "all-MiniLM-L6-v2"
CHUNK_SIZE         = 300  # Max characters per chunk (optimized for financial documents)
MIN_CHUNK_LENGTH   = 30   # Minimum characters to keep a chunk
TOP_K              = 5

DATA_ONLY_KEYWORDS = [
    "shareholding pattern", "balance sheet", "corporate actions",
    "stock prices", "financial statements", "earnings report",
    "dividend history", "revenue", "profit", "ebitda", "margin"
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

# -------------------- RAG GLOBALS --------------------

vector_index = None
document_chunks = []
embedder = None
rag_status = {
    "initialized": False,
    "pdf_count": 0,
    "chunk_count": 0,
    "error_message": None
}

DOCUMENT_TIMESTAMPS = {
    "Annual FY24 (ended Mar 2024).pdf": datetime(2024, 3, 31),
    "Annual FY25 (ended Mar 2025).pdf": datetime(2025, 3, 31),
    "Q1 FY26 (Apr-Jun 2025).pdf": datetime(2025, 6, 30),
    "Q2 FY25 (Jul-Sep 2024).pdf": datetime(2024, 9, 30),
    "Q2 FY26 (Jul-Sep 2025).pdf": datetime(2025, 9, 30),
    "Q3 FY25 (Oct-Dec 2024).pdf": datetime(2024, 12, 31)
}

# -------------------- CHUNKING STRATEGY --------------------

def smart_chunk_text(text: str, file_name: str, max_size: int = CHUNK_SIZE) -> list:
    """
    Intelligently chunk financial document text
    Tries to keep financial metrics and data together
    """
    chunks = []
    
    # Split by common section breaks in financial documents
    sections = re.split(r'\n(?=\d+\.|[A-Z][A-Z\s]+\n|.*(?:Revenue|Profit|EBITDA|Balance Sheet))', text)
    
    for section in sections:
        section = section.strip()
        if len(section) < MIN_CHUNK_LENGTH:
            continue
            
        # If section is small enough, add as-is
        if len(section) <= max_size:
            chunks.append(section)
            continue
        
        # Otherwise, split by sentences
        sentences = re.split(r'(?<=[.!?])\s+', section)
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # Add sentence to current chunk if it fits
            if len(current_chunk) + len(sentence) + 1 <= max_size:
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
            else:
                # Save current chunk and start new one
                if len(current_chunk) >= MIN_CHUNK_LENGTH:
                    chunks.append(current_chunk)
                current_chunk = sentence
        
        # Add remaining chunk
        if len(current_chunk) >= MIN_CHUNK_LENGTH:
            chunks.append(current_chunk)
    
    return chunks

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
    try:
        m = re.search(r"(\d{4})", name)
        if m:
            return datetime(int(m.group(1)), 12, 31)
    except:
        pass
    return None

def load_pdfs_from_local_folder():
    """Load PDFs from local folder and build FAISS index with embeddings"""
    global vector_index, embedder, document_chunks, rag_status

    print("\n[RAG] ========== STARTING RAG INITIALIZATION ==========")

    # Check if path exists
    pdf_path = Path(PDF_FOLDER_PATH)
    
    if not pdf_path.exists():
        msg = f"❌ PDF folder not found: {pdf_path.absolute()}"
        print(f"[RAG] {msg}")
        rag_status["error_message"] = msg
        rag_status["initialized"] = False
        return False

    print(f"[RAG] 📂 Looking for PDFs in: {pdf_path.absolute()}")

    try:
        # Step 1: Find all PDF files
        print("[RAG] 📂 Searching for PDF files...")
        pdf_files = list(pdf_path.glob("*.pdf"))
        pdf_files.sort()
        
        print(f"[RAG] Found {len(pdf_files)} PDF files")
        
        rag_status["pdf_count"] = len(pdf_files)
        
        for pdf_file in pdf_files:
            print(f"[RAG]   - {pdf_file.name} ({pdf_file.stat().st_size / 1024 / 1024:.2f} MB)")

        if len(pdf_files) == 0:
            msg = f"⚠️ No PDF files found in {pdf_path}. Please add PDF files to continue."
            print(f"[RAG] {msg}")
            rag_status["error_message"] = msg
            rag_status["initialized"] = False
            # Still initialize embedder for future use
            print("[RAG] 🤖 Loading embedding model anyway...")
            embedder = SentenceTransformer(EMBEDDING_MODEL)
            return False

        # Step 2: Initialize embedder
        print("[RAG] 🤖 Loading embedding model...")
        embedder = SentenceTransformer(EMBEDDING_MODEL)
        print("[RAG] ✅ Embedding model loaded")

        # Step 3: Extract text from PDFs
        print("[RAG] 📄 Extracting text from PDFs...")
        docs = []
        total_chunks_before = 0
        
        for pdf_file in pdf_files:
            try:
                print(f"[RAG]   Reading {pdf_file.name}...")
                
                with open(pdf_file, "rb") as f:
                    reader = PyPDF2.PdfReader(f)
                    
                    page_count = len(reader.pages)
                    text = "\n".join((page.extract_text() or "") for page in reader.pages)
                    
                    print(f"[RAG]     - {page_count} pages, {len(text)} chars")
                    
                    ts = (
                        DOCUMENT_TIMESTAMPS.get(pdf_file.name)
                        or parse_timestamp_from_filename(pdf_file.name)
                        or datetime.fromtimestamp(pdf_file.stat().st_mtime)
                    )

                    # Use smart chunking strategy
                    chunks = smart_chunk_text(text, pdf_file.name)
                    
                    for chunk in chunks:
                        docs.append({
                            "text": chunk,
                            "metadata": {
                                "file": pdf_file.name,
                                "timestamp": ts.isoformat()
                            }
                        })
                    
                    chunks_created = len(chunks)
                    print(f"[RAG]     - Created {chunks_created} chunks")
                    total_chunks_before += chunks_created
                    
            except Exception as e:
                print(f"[RAG] ❌ Error reading {pdf_file.name}: {e}")
                traceback.print_exc()
                continue

        if len(docs) == 0:
            msg = "⚠️ No document chunks created from PDFs."
            print(f"[RAG] {msg}")
            rag_status["error_message"] = msg
            rag_status["initialized"] = False
            return False

        print(f"[RAG] ✅ Total chunks created: {len(docs)}")

        # Step 4: Create embeddings
        print("[RAG] 🧮 Creating embeddings...")
        texts = [d["text"] for d in docs]
        embeddings = embedder.encode(texts, show_progress_bar=False).astype("float32")
        print(f"[RAG] ✅ Embeddings created: {embeddings.shape}")

        # Step 5: Build FAISS index
        print("[RAG] 🔍 Building FAISS index...")
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)
        print(f"[RAG] ✅ FAISS index created with {index.ntotal} vectors")

        # Step 6: Assign globals
        document_chunks = docs
        vector_index = index

        rag_status["chunk_count"] = len(document_chunks)
        rag_status["initialized"] = True
        rag_status["error_message"] = None

        print("[RAG] ========== RAG INITIALIZATION COMPLETE ==========")
        print(f"[RAG] ✅ RAG is ready! {len(document_chunks)} chunks indexed from {len(pdf_files)} PDFs")
        return True

    except Exception as e:
        msg = f"FATAL ERROR: {str(e)}"
        print(f"[RAG] ❌ {msg}")
        traceback.print_exc()
        rag_status["error_message"] = msg
        rag_status["initialized"] = False
        return False

# -------------------- RAG SEARCH --------------------

def retrieve_relevant_chunks(query, k=TOP_K):
    """Retrieve top-k relevant chunks using vector similarity"""
    if not vector_index or not embedder:
        print("[SEARCH] ⚠️ RAG not ready (index or embedder is None)")
        return []
    
    try:
        q_emb = embedder.encode([query]).astype("float32")
        distances, idx = vector_index.search(q_emb, min(k, len(document_chunks)))
        
        results = [document_chunks[i] for i in idx[0] if i < len(document_chunks)]
        print(f"[SEARCH] Found {len(results)} relevant chunks for query")
        return results
    except Exception as e:
        print(f"[SEARCH] ❌ Error retrieving chunks: {e}")
        return []

# -------------------- GPT --------------------

def ask_gpt(messages, temp=0.5, max_tokens=600):
    """Call Azure OpenAI API"""
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
    """Main chat endpoint with RAG support"""
    try:
        data = request.get_json()
        user_msg = (data.get("message") or "").strip()

        if not user_msg:
            return jsonify({"error": "Empty message"}), 400

        print(f"\n[CHAT] User query: {user_msg}")
        print(f"[CHAT] RAG status - initialized: {rag_status['initialized']}, chunks: {len(document_chunks)}")

        # Basic Q type detection
        date_class, has_date = extract_query_date(user_msg)
        is_data = is_data_only_query(user_msg)
        is_reason = is_reasoning_query(user_msg)

        print(f"[CHAT] Query type - date_class: {date_class}, data_only: {is_data}, reasoning: {is_reason}")

        # PRE 2024 (no RAG needed - use knowledge cutoff)
        if date_class == "pre":
            print("[CHAT] Using pre-2024 mode (knowledge cutoff)")
            sys = (
                f"Financial analyst up to {GPT_CUTOFF_DATE}. "
                f"FY23 revenue ₹485.6 Cr (+30.4% YoY from ₹372.3 Cr). "
                f"Provide accurate historical financial information for {COMPANY_NAME}."
            )
            reply = ask_gpt(
                [{"role": "system", "content": sys},
                 {"role": "user", "content": user_msg}],
                temp=0.3,
                max_tokens=get_max_tokens(user_msg)
            )
            print(f"[CHAT] Response: {reply[:100]}...")
            return jsonify({"response": reply})

        # POST 2024 (RAG if available, else fallback)
        print("[CHAT] Using post-2024 mode")
        
        # Check if RAG is ready
        if not rag_status["initialized"] or vector_index is None or len(document_chunks) == 0:
            print("[CHAT] ⚠️ RAG not available, using general knowledge")
            sys = (
                f"Financial analyst for {COMPANY_NAME}. "
                f"Note: Document database is currently empty. "
                f"Provide general financial analysis based on your knowledge up to {GPT_CUTOFF_DATE}."
            )
            reply = ask_gpt(
                [{"role": "system", "content": sys},
                 {"role": "user", "content": user_msg}],
                temp=0.3,
                max_tokens=get_max_tokens(user_msg)
            )
            return jsonify({"response": reply})

        # RAG available - use document-based search
        chunks = retrieve_relevant_chunks(user_msg)
        
        if not chunks:
            print("[CHAT] ⚠️ No relevant chunks found")
            sys = (
                f"Financial analyst for {COMPANY_NAME}. "
                f"The query didn't match any documents. Provide general guidance if possible."
            )
            reply = ask_gpt(
                [{"role": "system", "content": sys},
                 {"role": "user", "content": user_msg}],
                temp=0.3,
                max_tokens=get_max_tokens(user_msg)
            )
            return jsonify({"response": reply})

        docs = "\n\n".join([f"From {c['metadata']['file']}:\n{c['text']}" for c in chunks])
        sys = f"Financial analyst for {COMPANY_NAME}. Use ONLY provided document facts. Be precise and cite sources. No emojis/bold."
        
        reply = ask_gpt(
            [{"role": "system", "content": sys},
             {"role": "user", "content": f"Documents:\n{docs}\n\nQuestion: {user_msg}"}],
            temp=0.2,
            max_tokens=get_max_tokens(user_msg)
        )
        
        print(f"[CHAT] Response: {reply[:100]}...")
        return jsonify({"response": reply})

    except Exception as e:
        print(f"[CHAT ERROR] {e}")
        traceback.print_exc()
        return jsonify({"error": "Server error"}), 500

@app.route("/health")
def health():
    """Health check endpoint"""
    status = {
        "status": "ok",
        "rag_initialized": rag_status["initialized"],
        "chunks_count": len(document_chunks),
        "embedder_loaded": embedder is not None,
        "pdf_count": rag_status["pdf_count"],
        "error": rag_status["error_message"],
        "pdf_folder": str(Path(PDF_FOLDER_PATH).absolute())
    }
    print(f"[HEALTH] {status}")
    return jsonify(status)

@app.route("/debug/rag-status")
def debug_rag_status():
    """Debug endpoint to check RAG status"""
    return jsonify({
        "rag_initialized": rag_status["initialized"],
        "vector_index_loaded": vector_index is not None,
        "document_chunks": len(document_chunks),
        "embedder_loaded": embedder is not None,
        "pdf_count": rag_status["pdf_count"],
        "error_message": rag_status["error_message"],
        "pdf_folder": str(Path(PDF_FOLDER_PATH).absolute()),
        "sample_chunks": [
            {
                "file": c["metadata"]["file"],
                "timestamp": c["metadata"]["timestamp"],
                "text_preview": c["text"][:150]
            }
            for c in document_chunks[:5]
        ] if document_chunks else []
    })

@app.route("/debug/reinit-rag", methods=["POST"])
def debug_reinit_rag():
    """Manually reinitialize RAG"""
    print("[DEBUG] Manual RAG reinitialization requested")
    success = load_pdfs_from_local_folder()
    return jsonify({
        "success": success,
        "status": rag_status
    })

# -------------------- APP INITIALIZATION --------------------

def init_rag_on_startup():
    """Initialize RAG when app starts - CRITICAL FOR PRODUCTION"""
    print("\n" + "="*70)
    print("🚀 SAGEALPHA.AI STARTUP - INITIALIZING RAG FROM LOCAL PDFS")
    print("="*70 + "\n")
    
    success = load_pdfs_from_local_folder()
    
    if success:
        print("\n✅ RAG INITIALIZATION SUCCESSFUL")
    else:
        print(f"\n⚠️ RAG INITIALIZATION WARNING: {rag_status['error_message']}")
        print(f"💡 Expected PDF folder: {Path(PDF_FOLDER_PATH).absolute()}")
    
    print("="*70 + "\n")

# Initialize RAG on app startup (works in both local and production)
init_rag_on_startup()

# -------------------- MAIN ENTRY POINT --------------------

if __name__ == "__main__":
    print(f"🌐 Running on http://127.0.0.1:{PORT}")
    serve(app, host="0.0.0.0", port=PORT)