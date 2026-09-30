"""
FastAPI Backend Server for OpenRouter AI Chatbot
------------------------------------------------
Provides secure proxy endpoints:
- GET  /api/health       -> Quick backend and config health check
- GET  /api/status       -> Validates live connection with OpenRouter (no key exposed)
- GET  /api/models       -> Returns discovered chat models & auto-selected model
- POST /api/model/select -> Updates the active model
- POST /api/chat         -> Chat completion proxy
- POST /api/chat/stream  -> Progressive SSE token streaming

Also serves the static frontend directly from ../frontend.
"""

import os
import logging
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv

# Load .env from backend directory or project root
backend_dir = Path(__file__).resolve().parent
load_dotenv(backend_dir / ".env")
load_dotenv(backend_dir.parent / ".env")

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from openrouter import openrouter_service

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("api")

app = FastAPI(
    title="OpenRouter AI Chatbot API",
    description="Secure intermediate full-stack chatbot backend powered by OpenRouter",
    version="1.0.0"
)

# Enable CORS for local cross-origin development (e.g. VSCode Live Server)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------
# Request & Response Data Models
# ---------------------------------------------------------

class ChatMessage(BaseModel):
    role: str = Field(..., description="Role of the sender: system, user, or assistant")
    content: str = Field(..., min_length=1, description="Message text content")


class ChatRequest(BaseModel):
    model: Optional[str] = Field(None, description="OpenRouter model ID to use")
    messages: List[ChatMessage] = Field(..., min_length=1, description="Conversation messages array")
    temperature: Optional[float] = Field(0.7, ge=0.0, le=2.0)


class ChatResponse(BaseModel):
    message: str
    model: str


class ModelSelectRequest(BaseModel):
    model: str = Field(..., min_length=1)


# ---------------------------------------------------------
# Application Startup Event
# ---------------------------------------------------------

@app.on_event("startup")
async def on_startup():
    logger.info("Chatbot server initializing...")
    if openrouter_service.is_key_configured:
        logger.info("OPENROUTER_API_KEY detected in environment.")
        # Attempt background warm-up of models
        try:
            models = await openrouter_service.fetch_available_models()
            selected = openrouter_service.get_selected_model()
            logger.info("Model discovery completed. Discovered %d chat models. Auto-selected default: %s", len(models), selected)
        except Exception as exc:
            logger.warning("Initial model fetch warning: %s", exc)
    else:
        logger.warning("No OPENROUTER_API_KEY found in .env. API calls will fail until the key is configured.")


# ---------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------

@app.get("/api/health")
async def health_check():
    """Simple backend health and environment check."""
    return {
        "status": "ok",
        "key_configured": openrouter_service.is_key_configured,
        "provider": "OpenRouter"
    }


@app.get("/api/status")
async def get_connection_status():
    """
    Step 1: Tests whether the OpenRouter API key is valid.
    Never exposes the API key to the client.
    """
    is_connected, message = await openrouter_service.verify_connection()
    selected_model = openrouter_service.get_selected_model()

    return {
        "connected": is_connected,
        "provider": "OpenRouter",
        "model": selected_model,
        "key_configured": openrouter_service.is_key_configured,
        "message": message
    }


@app.get("/api/models")
async def list_models():
    """
    Step 2 & 3: Retrieves available chat models and automatic default selection.
    """
    models = await openrouter_service.fetch_available_models()
    selected_model = openrouter_service.get_selected_model()

    return {
        "models": models,
        "selected_model": selected_model,
        "total_count": len(models)
    }


@app.post("/api/model/select")
async def select_model(payload: ModelSelectRequest):
    """Allows manual selection of a model from the UI."""
    openrouter_service.set_selected_model(payload.model)
    return {
        "success": True,
        "selected_model": payload.model
    }


@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Step 4 & 5: Proxies chat completion requests with conversation history.
    """
    # Validation
    if not request.messages:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Message list cannot be empty."
        )

    last_message = request.messages[-1]
    if last_message.role != "user" or not last_message.content.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The latest message must be a non-empty user message."
        )

    try:
        messages_dict = [{"role": m.role, "content": m.content} for m in request.messages]
        result = await openrouter_service.send_chat_completion(
            messages=messages_dict,
            model=request.model,
            temperature=request.temperature or 0.7
        )
        return ChatResponse(
            message=result["message"],
            model=result["model"]
        )

    except RuntimeError as err:
        err_msg = str(err)
        # Determine appropriate status code
        status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        if "authentication" in err_msg.lower():
            status_code = status.HTTP_401_UNAUTHORIZED
        elif "rate limit" in err_msg.lower():
            status_code = status.HTTP_429_TOO_MANY_REQUESTS
        elif "not found" in err_msg.lower() or "offline" in err_msg.lower():
            status_code = status.HTTP_404_NOT_FOUND
        elif "timed out" in err_msg.lower():
            status_code = status.HTTP_504_GATEWAY_TIMEOUT

        raise HTTPException(status_code=status_code, detail=err_msg)


@app.post("/api/chat/stream")
async def chat_stream_endpoint(request: ChatRequest):
    """
    Progressive token streaming endpoint via Server-Sent Events (SSE).
    """
    messages_dict = [{"role": m.role, "content": m.content} for m in request.messages]
    
    return StreamingResponse(
        openrouter_service.stream_chat_completion(
            messages=messages_dict,
            model=request.model,
            temperature=request.temperature or 0.7
        ),
        media_type="text/event-stream"
    )


# ---------------------------------------------------------
# Static Frontend Serving
# ---------------------------------------------------------

frontend_dir = backend_dir.parent / "frontend"
if frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")

    @app.get("/")
    async def serve_index():
        return FileResponse(frontend_dir / "index.html")
