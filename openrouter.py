"""
OpenRouter API Service Module
-----------------------------
Handles all communication between the local backend and OpenRouter's API:
- Secure API key validation
- Dynamic chat model discovery and filtering
- Intelligent automatic model selection
- Chat completion forwarding with robust error mapping

Security Note:
This file reads the API key from environment variables.
Never log, print, or return the raw API key to the frontend.
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional, Tuple, AsyncGenerator
import httpx

logger = logging.getLogger("openrouter")

# Base URLs and Defaults
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful, accurate, friendly AI assistant. "
    "Explain concepts clearly and adapt your answer to the user's level. "
    "For programming questions, provide practical examples and explain important parts of the code. "
    "Do not claim to have performed actions you did not perform. "
    "If information is uncertain, say so."
)

# Preferred default models to pick automatically, ordered by preference
PREFERRED_DEFAULT_MODELS = [
    "meta-llama/llama-3.3-70b-instruct:free",
    "google/gemini-2.0-flash-exp:free",
    "meta-llama/llama-3.1-8b-instruct:free",
    "qwen/qwen-2.5-72b-instruct:free",
    "mistralai/mistral-7b-instruct:free",
    "deepseek/deepseek-chat",
    "openai/gpt-4o-mini",
    "google/gemini-flash-1.5",
    "anthropic/claude-3-haiku",
]


class OpenRouterService:
    def __init__(self):
        self._cached_models: List[Dict[str, Any]] = []
        self._selected_model: Optional[str] = None

    @property
    def api_key(self) -> Optional[str]:
        """Retrieves the API key dynamically from the environment."""
        key = os.getenv("OPENROUTER_API_KEY", "").strip()
        # Filter out placeholder text
        if not key or key in ("YOUR_NEW_KEY_HERE", "YOUR_OPENROUTER_API_KEY", "your-key-here"):
            return None
        return key

    @property
    def is_key_configured(self) -> bool:
        """Returns True if a non-placeholder API key is set."""
        return self.api_key is not None

    def get_headers(self) -> Dict[str, str]:
        """Generates standard headers for OpenRouter API calls."""
        key = self.api_key or ""
        return {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:8000",
            "X-Title": "OpenRouter AI Chatbot",
        }

    async def verify_connection(self) -> Tuple[bool, str]:
        """
        Tests whether the OpenRouter API key is valid without exposing it.
        Uses the OpenRouter /auth/key endpoint.
        Returns: (is_connected: bool, status_message: str)
        """
        if not self.is_key_configured:
            return False, "OpenRouter API key is not configured. Please add OPENROUTER_API_KEY to backend/.env."

        url = f"{OPENROUTER_BASE_URL}/auth/key"
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(url, headers=self.get_headers())

            if response.status_code == 200:
                data = response.json().get("data", {})
                label = data.get("label") or "Active Key"
                limit = data.get("limit")
                limit_info = f" (Credit Limit: ${limit})" if limit is not None else ""
                return True, f"OpenRouter API connected successfully [{label}]{limit_info}"
            elif response.status_code == 401:
                return False, "OpenRouter authentication failed: The provided API key is invalid or revoked."
            else:
                return False, f"OpenRouter responded with status {response.status_code}."

        except httpx.TimeoutException:
            return False, "Connection timed out while reaching OpenRouter."
        except httpx.RequestError as exc:
            logger.error("Connection error during API key verification: %s", str(exc))
            return False, "Network error: Unable to reach OpenRouter API."

    async def fetch_available_models(self) -> List[Dict[str, Any]]:
        """
        Retrieves models from OpenRouter and filters them for standard text chat.
        Extracts: id, name, context_length, pricing, is_free, and provider.
        """
        url = f"{OPENROUTER_BASE_URL}/models"
        headers = self.get_headers() if self.is_key_configured else {"Content-Type": "application/json"}

        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.get(url, headers=headers)

            if response.status_code != 200:
                logger.error("Failed to fetch models: %s", response.text)
                return self._cached_models

            raw_models = response.json().get("data", [])
            filtered_models = []

            for item in raw_models:
                model_id = item.get("id", "")
                name = item.get("name", model_id)
                context_length = item.get("context_length", 4096)
                pricing = item.get("pricing", {})
                architecture = item.get("architecture", {})

                # Exclude non-text/chat modalities
                modality = architecture.get("modality", "text->text")
                lower_id = model_id.lower()
                if any(x in lower_id for x in ["embed", "whisper", "tts", "moderation", "flux", "stable-diffusion"]):
                    continue

                # Determine if free
                is_free = ":free" in lower_id or (
                    float(pricing.get("prompt", 0)) == 0 and float(pricing.get("completion", 0)) == 0
                )

                # Provider extracted from model ID (e.g., 'meta-llama/...' -> 'meta-llama')
                provider = model_id.split("/")[0] if "/" in model_id else "openrouter"

                filtered_models.append({
                    "id": model_id,
                    "name": name,
                    "provider": provider,
                    "context_length": context_length,
                    "is_free": is_free,
                    "pricing": {
                        "prompt": pricing.get("prompt", "0"),
                        "completion": pricing.get("completion", "0"),
                    },
                    "description": item.get("description", "")[:120] if item.get("description") else ""
                })

            # Sort: free models first, then alphabetical by name
            filtered_models.sort(key=lambda m: (not m["is_free"], m["name"].lower()))
            self._cached_models = filtered_models

            # Automatically select default model if not set yet
            if not self._selected_model or not any(m["id"] == self._selected_model for m in filtered_models):
                self._selected_model = self._pick_default_model(filtered_models)

            return filtered_models

        except Exception as exc:
            logger.error("Error during model discovery: %s", str(exc))
            return self._cached_models

    def _pick_default_model(self, models: List[Dict[str, Any]]) -> str:
        """Intelligently picks a suitable starting model from discovered models."""
        available_ids = {m["id"]: m for m in models}

        # Check preferred default list
        for preferred in PREFERRED_DEFAULT_MODELS:
            if preferred in available_ids:
                return preferred

        # Fallback 1: Any free model
        for m in models:
            if m.get("is_free"):
                return m["id"]

        # Fallback 2: Any model with reasonable context length (>= 4096)
        for m in models:
            if m.get("context_length", 0) >= 4096:
                return m["id"]

        # Fallback 3: First available or static default
        if models:
            return models[0]["id"]
        return os.getenv("DEFAULT_FALLBACK_MODEL", "meta-llama/llama-3.3-70b-instruct:free")

    def get_selected_model(self) -> str:
        """Returns currently selected model id."""
        if not self._selected_model:
            self._selected_model = os.getenv("DEFAULT_FALLBACK_MODEL", "meta-llama/llama-3.3-70b-instruct:free")
        return self._selected_model

    def set_selected_model(self, model_id: str) -> bool:
        """Updates currently selected model id."""
        self._selected_model = model_id
        return True

    def format_messages(self, messages: List[Dict[str, str]], system_prompt: Optional[str] = None) -> List[Dict[str, str]]:
        """Ensures messages conform to OpenRouter chat standards and prepends system prompt."""
        formatted = []
        has_system = any(m.get("role") == "system" for m in messages)

        if not has_system:
            prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
            formatted.append({"role": "system", "content": prompt})

        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            if role in ("system", "user", "assistant") and content.strip():
                formatted.append({"role": role, "content": content})

        return formatted

    async def send_chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = 1024,
    ) -> Dict[str, Any]:
        """
        Sends chat completion request to OpenRouter.
        Returns: {"message": str, "model": str} or raises sanitized RuntimeError.
        """
        if not self.is_key_configured:
            raise RuntimeError("OpenRouter API key is not configured. Please add OPENROUTER_API_KEY to backend/.env.")

        target_model = model or self.get_selected_model()
        formatted_messages = self.format_messages(messages)

        payload: Dict[str, Any] = {
            "model": target_model,
            "messages": formatted_messages,
            "temperature": temperature,
        }
        if max_tokens:
            payload["max_tokens"] = max_tokens

        url = f"{OPENROUTER_BASE_URL}/chat/completions"

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(url, headers=self.get_headers(), json=payload)

            if response.status_code == 200:
                data = response.json()
                choices = data.get("choices", [])
                if not choices:
                    raise RuntimeError("Received an empty response from OpenRouter.")
                
                content = choices[0].get("message", {}).get("content", "")
                resolved_model = data.get("model", target_model)
                return {
                    "message": content,
                    "model": resolved_model
                }
            elif response.status_code == 401:
                raise RuntimeError("OpenRouter authentication failed. Please check your API key in backend/.env.")
            elif response.status_code == 402:
                raise RuntimeError("Insufficient OpenRouter credits for this model. Please select a free model (:free) or top up credits.")
            elif response.status_code == 404:
                raise RuntimeError(f"Model '{target_model}' was not found or is currently offline on OpenRouter. Please select another model.")
            elif response.status_code == 429:
                raise RuntimeError("OpenRouter rate limit reached. Please wait a moment before sending another message.")
            elif response.status_code >= 500:
                raise RuntimeError("OpenRouter upstream server error. Please try again in a few moments or switch to a different model.")
            else:
                error_msg = "OpenRouter request error."
                try:
                    err_json = response.json()
                    if "error" in err_json:
                        error_msg = f"OpenRouter: {err_json['error'].get('message', 'Request failed')}"
                except Exception:
                    pass
                raise RuntimeError(error_msg)

        except httpx.TimeoutException:
            raise RuntimeError("The request to OpenRouter timed out. The model may be under heavy load. Please try again.")
        except httpx.RequestError as exc:
            logger.error("Network error during chat completion: %s", str(exc))
            raise RuntimeError("Network error connecting to OpenRouter. Please check your internet connection.")

    async def stream_chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.7,
    ) -> AsyncGenerator[str, None]:
        """
        Streams chat completion tokens from OpenRouter using Server-Sent Events (SSE).
        Yields raw text chunks.
        """
        if not self.is_key_configured:
            yield "data: " + json.dumps({"error": "OpenRouter API key is not configured in backend/.env."}) + "\n\n"
            return

        target_model = model or self.get_selected_model()
        formatted_messages = self.format_messages(messages)

        payload = {
            "model": target_model,
            "messages": formatted_messages,
            "temperature": temperature,
            "stream": True,
        }

        url = f"{OPENROUTER_BASE_URL}/chat/completions"

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                async with client.stream("POST", url, headers=self.get_headers(), json=payload) as response:
                    if response.status_code != 200:
                        yield f"data: {json.dumps({'error': f'OpenRouter returned status {response.status_code}'})}\n\n"
                        return

                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            content = line[6:].strip()
                            if content == "[DONE]":
                                yield "data: [DONE]\n\n"
                                break
                            try:
                                chunk = json.loads(content)
                                delta = chunk.get("choices", [{}])[0].get("delta", {})
                                text_piece = delta.get("content", "")
                                if text_piece:
                                    yield f"data: {json.dumps({'token': text_piece, 'model': target_model})}\n\n"
                            except json.JSONDecodeError:
                                continue

        except Exception as exc:
            yield f"data: {json.dumps({'error': f'Streaming error: {str(exc)}'})}\n\n"


# Singleton instance for the application
openrouter_service = OpenRouterService()
