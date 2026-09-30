"""
Flask Backend Application for OpenRouter AI Chatbot
---------------------------------------------------
Provides REST API endpoints for:
1. Serving static frontend assets (HTML, CSS, JS).
2. Testing OpenRouter API key validity.
3. Discovering available AI models with pricing and context data.
4. Sending and managing chat conversations.
"""

import os
from pathlib import Path
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv

# Import the modular OpenRouter service
from services.openrouter_service import OpenRouterService

# Find and load .env file from backend directory or project root
current_dir = Path(__file__).resolve().parent
env_path = current_dir / ".env"
if not env_path.exists():
    env_path = current_dir.parent / ".env"
load_dotenv(dotenv_path=env_path)

# Initialize Flask App with frontend as static folder
frontend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend"))
app = Flask(__name__, static_folder=frontend_dir, static_url_path="")
CORS(app)

# Instantiate OpenRouter service
service = OpenRouterService()


@app.route("/")
def serve_index():
    """Serves the main single-page application frontend."""
    return send_from_directory(frontend_dir, "index.html")


@app.route("/<path:filename>")
def serve_static_files(filename):
    """Serves static files (CSS, JS, assets) from the frontend folder."""
    return send_from_directory(frontend_dir, filename)


@app.route("/api/test-key", methods=["GET"])
def test_key():
    """
    Tests whether the OpenRouter API key configured in .env is valid.
    
    Response format:
    {
        "success": true | false,
        "message": "...",
        "details": { "label": "...", "usage": "...", "limit": "..." }
    }
    """
    # Reload environment to pick up any recent updates to .env
    load_dotenv(dotenv_path=env_path, override=True)
    
    is_valid, message, details = service.test_api_key()

    if is_valid:
        return jsonify({
            "success": True,
            "message": message,
            "details": details
        }), 200
    else:
        return jsonify({
            "success": False,
            "error": message
        }), 400


@app.route("/api/models", methods=["GET"])
def get_models():
    """
    Fetches the dynamically available models from OpenRouter.
    
    Response format:
    {
        "success": true | false,
        "models": [ { id, name, provider, context_length, input_price, output_price, is_free }, ... ],
        "count": int
    }
    """
    success, models, error_message = service.get_available_models()

    if success:
        return jsonify({
            "success": True,
            "models": models,
            "count": len(models)
        }), 200
    else:
        return jsonify({
            "success": False,
            "error": error_message,
            "models": []
        }), 502


@app.route("/api/chat", methods=["POST"])
def chat():
    """
    Processes a chat completion request with multi-turn conversation memory.
    
    Expected JSON payload:
    {
        "model": "openai/gpt-4o-mini",
        "messages": [
            { "role": "user", "content": "Hello" },
            { "role": "assistant", "content": "Hi there!" },
            { "role": "user", "content": "How are you?" }
        ]
    }
    """
    data = request.get_json(silent=True)
    if not data:
        return jsonify({
            "success": False,
            "error": "Invalid request body. Expected JSON payload."
        }), 400

    model = data.get("model", "").strip()
    messages = data.get("messages", [])

    if not model:
        return jsonify({
            "success": False,
            "error": "Please select an AI model before sending a message."
        }), 400

    if not messages or not isinstance(messages, list):
        return jsonify({
            "success": False,
            "error": "Messages list is required and cannot be empty."
        }), 400

    # Ensure last message has non-empty content
    last_msg = messages[-1]
    if not isinstance(last_msg, dict) or not last_msg.get("content", "").strip():
        return jsonify({
            "success": False,
            "error": "The last message content cannot be empty."
        }), 400

    success, response_data, error_message = service.send_chat_completion(
        model=model,
        messages=messages
    )

    if success:
        return jsonify({
            "success": True,
            "message": {
                "role": response_data.get("role", "assistant"),
                "content": response_data.get("content", "")
            },
            "usage": response_data.get("usage", {}),
            "model": response_data.get("model", model)
        }), 200
    else:
        return jsonify({
            "success": False,
            "error": error_message
        }), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug_mode = os.environ.get("FLASK_DEBUG", "True").lower() in ("true", "1", "yes")
    print(f"\n=======================================================")
    print(f"🚀 OpenRouter Chatbot Server is starting...")
    print(f"🌐 Access Web Interface: http://127.0.0.1:{port}")
    print(f"📁 Frontend Directory:   {frontend_dir}")
    print(f"🔒 API Key Protected:    Backend Only")
    print(f"=======================================================\n")
    app.run(host="0.0.0.0", port=port, debug=debug_mode)
