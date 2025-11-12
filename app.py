"""
SageAlpha.ai – Flask + Azure OpenAI Chat App with
Google Web Search Fallback + Session Memory (v3)
"""

import os
import atexit
import re
import requests
from flask import Flask, render_template, request, jsonify, session
from flask_cors import CORS
from dotenv import load_dotenv
from waitress import serve
from openai import AzureOpenAI

# -----------------------------------------------------------
# Load environment variables
# -----------------------------------------------------------
load_dotenv()

AZURE_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT", "")
AZURE_KEY = os.environ.get("AZURE_OPENAI_API_KEY", "")
AZURE_DEPLOYMENT = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "")
AZURE_VERSION = os.environ.get("AZURE_OPENAI_API_VERSION", "")
PORT = int(os.environ.get("PORT", 5000))

GOOGLE_API_KEY = os.environ.get("GOOGLE_SEARCH_API_KEY", "")
GOOGLE_CX = os.environ.get("GOOGLE_SEARCH_CX", "")

# -----------------------------------------------------------
# Flask setup
# -----------------------------------------------------------
app = Flask(__name__, template_folder="templates")
app.secret_key = os.getenv("FLASK_SECRET_KEY", "super_secret_key_change_this")
CORS(app)

# -----------------------------------------------------------
# Azure OpenAI client
# -----------------------------------------------------------
client = AzureOpenAI(
    api_key=AZURE_KEY,
    azure_endpoint=AZURE_ENDPOINT,
    api_version=AZURE_VERSION
)

# -----------------------------------------------------------
# Google Custom Search helper
# -----------------------------------------------------------
def google_web_search(query, num_results=5):
    """Perform Google Custom Search and return text snippets."""
    try:
        url = "https://www.googleapis.com/customsearch/v1"
        params = {
            "key": GOOGLE_API_KEY,
            "cx": GOOGLE_CX,
            "q": query,
            "num": num_results,
            "hl": "en",
        }
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        items = data.get("items", [])
        if not items:
            return "No recent search results found."
        snippets = []
        for it in items:
            title = it.get("title", "")
            snippet = it.get("snippet", "")
            link = it.get("link", "")
            snippets.append(f"{title}: {snippet} (Source: {link})")
        return "\n".join(snippets)
    except Exception as e:
        return f"⚠️ Web search failed: {e}"

# -----------------------------------------------------------
# Azure GPT helper
# -----------------------------------------------------------
def ask_gpt(messages, temp=0.7, max_tokens=800):
    """Send chat messages list to Azure GPT."""
    resp = client.chat.completions.create(
        model=AZURE_DEPLOYMENT,
        messages=messages,
        temperature=temp,
        max_completion_tokens=max_tokens
    )
    return resp.choices[0].message.content.strip()

# -----------------------------------------------------------
# Heuristic: does GPT’s answer look stale / outdated?
# -----------------------------------------------------------
def looks_stale(answer):
    stale_patterns = [
        r"\bas of\b", r"not available", r"no official", r"not yet (launched|filed)",
        r"rumor", r"speculat", r"\b202[0-4]\b"
    ]
    return any(re.search(p, answer.lower()) for p in stale_patterns)

# -----------------------------------------------------------
# Routes
# -----------------------------------------------------------
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

        # --- Retrieve chat history (session memory) ---
        conversation = session.get("conversation", [])
        if len(conversation) > 10:  # limit context
            conversation = conversation[-10:]

        # Add user message to conversation
        conversation.append({"role": "user", "content": user_msg})

        # ---- Step 1: Ask GPT normally with history ----
        system_prompt = "You are SageAlpha AI, a finance assistant. Answer accurately, clearly, and concisely."
        base_reply = ask_gpt(
            [{"role": "system", "content": system_prompt}] + conversation
        )

        # ---- Step 2: Detect if answer looks outdated ----
        trigger_words = ["latest", "now", "current", "today", "update"]
        need_live = looks_stale(base_reply) or any(w in user_msg.lower() for w in trigger_words)

        print(f"[DEBUG] Stale={looks_stale(base_reply)}, LiveTrigger={need_live}, Msg='{user_msg[:60]}'")

        if need_live:
            # ---- Step 3: Fetch targeted finance results ----
            search_query = (
                f"{user_msg} latest financial update site:moneycontrol.com OR site:chittorgarh.com OR "
                f"site:economictimes.indiatimes.com OR site:livemint.com OR site:reuters.com OR site:business-standard.com"
            )
            web_data = google_web_search(search_query)

            # ---- Step 4: Summarize the search results ----
            summary_prompt = (
                f"Summarize these finance search results. Focus strictly on company, IPO date, price band, "
                f"GMP, quarterly or annual results, and relevant financial metrics. Ignore unrelated topics.\n\n"
                f"Search Results:\n{web_data}\n\n"
                f"User question: {user_msg}"
            )
            messages = [
                {"role": "system", "content": "You are SageAlpha AI, a financial analyst summarizing real-time search results."},
                {"role": "user", "content": summary_prompt}
            ]
            live_summary = ask_gpt(messages, temp=0.5)
            base_reply = f"📈 **Live Web Summary:**\n\n{live_summary}"

        # --- Save reply to chat history ---
        conversation.append({"role": "assistant", "content": base_reply})
        session["conversation"] = conversation

        return jsonify({"response": base_reply})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/reset", methods=["POST"])
def reset_chat():
    """Clear chat session."""
    session.pop("conversation", None)
    return jsonify({"status": "cleared"})

@app.route("/health")
def health():
    return jsonify({"status": "ok", "service": "SageAlpha.ai"})

# -----------------------------------------------------------
# Graceful shutdown
# -----------------------------------------------------------
atexit.register(lambda: client.close())

# -----------------------------------------------------------
# Main
# -----------------------------------------------------------
if __name__ == "__main__":
    print(f"🚀 SageAlpha.ai running on http://localhost:{PORT}")
    serve(app, host="0.0.0.0", port=PORT)
