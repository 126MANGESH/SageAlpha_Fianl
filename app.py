"""
SageAlpha.ai – Flask + Azure OpenAI Chat App
"""

import os
import atexit
from flask import Flask, render_template, request, jsonify
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

# -----------------------------------------------------------
# Initialize Flask
# -----------------------------------------------------------
app = Flask(__name__, template_folder="templates")
CORS(app)

# -----------------------------------------------------------
# Initialize Azure OpenAI client
# -----------------------------------------------------------
client = AzureOpenAI(
    api_key=AZURE_KEY,
    azure_endpoint=AZURE_ENDPOINT,
    api_version=AZURE_VERSION
)

# -----------------------------------------------------------
# Routes
# -----------------------------------------------------------
@app.route("/")
def index():
    """Render main chat interface."""
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    """Receive message from user and return AI-generated response."""
    try:
        data = request.get_json()
        user_message = (data.get("message") or "").strip()
        if not user_message:
            return jsonify({"error": "Empty message"}), 400

        # Send request to Azure OpenAI
        response = client.chat.completions.create(
            model=AZURE_DEPLOYMENT,
            messages=[
                {"role": "system", "content": "You are SageAlpha AI, a helpful finance assistant."},
                {"role": "user", "content": user_message}
            ],
            max_completion_tokens=1024,
            temperature=0.7,
        )

        # Extract reply text
        reply = response.choices[0].message.content.strip()
        return jsonify({"response": reply})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health")
def health():
    """Simple health check route."""
    return jsonify({"status": "ok", "service": "SageAlpha.ai"})


# -----------------------------------------------------------
# Graceful shutdown – close Azure client when exiting
# -----------------------------------------------------------
atexit.register(lambda: client.close())


# -----------------------------------------------------------
# Main entry
# -----------------------------------------------------------
if __name__ == "__main__":
    print(f"🚀 SageAlpha.ai running on http://localhost:{PORT}")
    serve(app, host="0.0.0.0", port=PORT)
