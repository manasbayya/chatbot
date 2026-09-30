"""
OpenRouter Service Module
-------------------------
Handles all direct interactions with the OpenRouter API:
1. Validating API Key credentials.
2. Fetching and formatting available AI models.
3. Sending chat completion requests with conversation history.
"""

import os
import requests
from typing import Dict, Any, Tuple, List, Optional


class OpenRouterService:
    """Service class encapsulating OpenRouter HTTP communication."""

    BASE_URL = "https://openrouter.ai/api/v1"
    DEFAULT_TIMEOUT = 45  # seconds

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the service. Reads OPENROUTER_API_KEY from environment
        if not explicitly provided.
        """
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY", "").strip()

    def _get_headers(self) -> Dict[str, str]:
        """Generate standard OpenRouter HTTP headers."""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:5000",
            "X-Title": "OpenRouter AI Chatbot",
        }

    def is_key_configured(self) -> bool:
        """Check if an API key has been provided and is not a placeholder."""
        if not self.api_key:
            return False
        placeholder_values = [
            "your_openrouter_api_key_here",
            "your_api_key_here",
            "sk-or-v1-placeholder",
        ]
        return self.api_key not in placeholder_values

    def test_api_key(self) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Tests whether the configured API key is valid.
        Uses OpenRouter's key inspection endpoint (/auth/key).

        Returns:
            Tuple: (is_valid: bool, message: str, details: dict)
        """
        # Re-read environment in case .env was modified at runtime
        self.api_key = os.getenv("OPENROUTER_API_KEY", "").strip()

        if not self.is_key_configured():
            return (
                False,
                "API Key is missing. Please add OPENROUTER_API_KEY to backend/.env",
                {},
            )

        endpoint = f"{self.BASE_URL}/auth/key"

        try:
            response = requests.get(
                endpoint,
                headers=self._get_headers(),
                timeout=15,
            )

            if response.status_code == 200:
                data = response.json().get("data", {})
                label = data.get("label", "Default Key")
                usage = data.get("usage", 0.0)
                limit = data.get("limit")
                limit_text = f"${limit:.2f}" if limit is not None else "Unlimited"

                return (
                    True,
                    "OpenRouter API key is valid",
                    {
                        "label": label,
                        "usage": f"${usage:.4f}",
                        "limit": limit_text,
                    },
                )

            elif response.status_code == 401:
                return (
                    False,
                    "OpenRouter API key is invalid. Please verify your key.",
                    {},
                )

            elif response.status_code == 403:
                return (
                    False,
                    "API key access is forbidden. Check your OpenRouter account permissions.",
                    {},
                )

            else:
                return (
                    False,
                    f"OpenRouter returned HTTP {response.status_code}: {response.text[:150]}",
                    {},
                )

        except requests.exceptions.Timeout:
            return (
                False,
                "Connection timed out while reaching OpenRouter. Please try again.",
                {},
            )
        except requests.exceptions.ConnectionError:
            return (
                False,
                "Unable to connect to OpenRouter. Please check your internet connection.",
                {},
            )
        except Exception as e:
            return (
                False,
                f"Unexpected error testing API key: {str(e)}",
                {},
            )

    def get_available_models(self) -> Tuple[bool, List[Dict[str, Any]], str]:
        """
        Fetches the complete catalog of models available from OpenRouter
        and normalizes their details for the frontend UI.

        Returns:
            Tuple: (success: bool, models_list: list, error_message: str)
        """
        self.api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
        endpoint = f"{self.BASE_URL}/models"

        try:
            # Models list can be fetched with or without an API key,
            # but providing the key ensures user-tier availability.
            headers = self._get_headers() if self.is_key_configured() else {}
            response = requests.get(endpoint, headers=headers, timeout=20)

            if response.status_code != 200:
                return (
                    False,
                    [],
                    f"Failed to fetch models from OpenRouter (HTTP {response.status_code})",
                )

            raw_data = response.json()
            raw_models = raw_data.get("data", [])

            formatted_models: List[Dict[str, Any]] = []

            for item in raw_models:
                model_id = item.get("id", "")
                if not model_id:
                    continue

                # Extract provider from ID prefix (e.g., 'openai/gpt-4o' -> 'openai')
                parts = model_id.split("/")
                provider = parts[0] if len(parts) > 1 else "other"

                # Extract context length
                context_length = item.get("context_length") or item.get(
                    "top_provider", {}
                ).get("context_length", 4096)

                # Pricing calculation (OpenRouter pricing is per 1 token)
                pricing_raw = item.get("pricing", {})
                prompt_cost = float(pricing_raw.get("prompt", 0) or 0)
                completion_cost = float(pricing_raw.get("completion", 0) or 0)

                # Format human-readable pricing per 1 Million tokens
                if prompt_cost == 0 and completion_cost == 0:
                    input_price_str = "Free"
                    output_price_str = "Free"
                    is_free = True
                else:
                    input_price_str = f"${prompt_cost * 1_000_000:.2f} / 1M tokens"
                    output_price_str = f"${completion_cost * 1_000_000:.2f} / 1M tokens"
                    is_free = False

                formatted_models.append(
                    {
                        "id": model_id,
                        "name": item.get("name") or model_id,
                        "provider": provider.capitalize(),
                        "context_length": context_length,
                        "context_length_formatted": f"{context_length:,} tokens"
                        if context_length
                        else "N/A",
                        "input_price": input_price_str,
                        "output_price": output_price_str,
                        "is_free": is_free,
                        "description": item.get("description", ""),
                    }
                )

                # Priority ordering: popular providers first
            priority_providers = [
                "openai",
                "anthropic",
                "google",
                "meta-llama",
                "deepseek",
                "mistralai",
                "qwen",
            ]

            def sort_key(m):
                prov = m["provider"].lower()
                for idx, p in enumerate(priority_providers):
                    if p in prov:
                        return (0, idx, m["name"].lower())
                return (1, 0, m["name"].lower())

            formatted_models.sort(key=sort_key)

            return (True, formatted_models, "")

        except requests.exceptions.Timeout:
            return (
                False,
                [],
                "Request timed out while retrieving models from OpenRouter.",
            )
        except requests.exceptions.ConnectionError:
            return (
                False,
                [],
                "Network error: Unable to connect to OpenRouter to fetch models.",
            )
        except Exception as e:
            return (
                False,
                [],
                f"Error retrieving models: {str(e)}",
            )

    def send_chat_completion(
        self,
        model: str,
        messages: List[Dict[str, str]],
    ) -> Tuple[bool, Dict[str, Any], str]:
        """
        Sends a conversation history to the selected OpenRouter model.

        Parameters:
            model (str): Model identifier (e.g. 'openai/gpt-4o-mini')
            messages (list): List of message dictionaries [{'role': 'user', 'content': '...'}]

        Returns:
            Tuple: (success: bool, response_dict: dict, error_message: str)
        """
        self.api_key = os.getenv("OPENROUTER_API_KEY", "").strip()

        if not self.is_key_configured():
            return (
                False,
                {},
                "API Key is not configured. Please set OPENROUTER_API_KEY in backend/.env",
            )

        if not model:
            return (False, {}, "No AI model selected. Please select a model.")

        if not messages or not isinstance(messages, list):
            return (
                False,
                {},
                "Invalid messages format. At least one message is required.",
            )

        endpoint = f"{self.BASE_URL}/chat/completions"
        payload = {
            "model": model,
            "messages": messages,
        }

        try:
            response = requests.post(
                endpoint,
                headers=self._get_headers(),
                json=payload,
                timeout=self.DEFAULT_TIMEOUT,
            )

            # Handle successful completion
            if response.status_code == 200:
                data = response.json()
                choices = data.get("choices", [])
                if not choices:
                    return (
                        False,
                        {},
                        "OpenRouter returned an empty choices list.",
                    )

                message = choices[0].get("message", {})
                content = message.get("content", "")
                role = message.get("role", "assistant")
                usage = data.get("usage", {})

                return (
                    True,
                    {
                        "role": role,
                        "content": content,
                        "usage": usage,
                        "model": data.get("model", model),
                    },
                    "",
                )

            # Specific HTTP Error Handling with user-friendly messages
            elif response.status_code == 401:
                return (
                    False,
                    {},
                    "Authentication failed: Invalid OpenRouter API key. Please check your .env file.",
                )

            elif response.status_code == 402:
                return (
                    False,
                    {},
                    "Insufficient credits on OpenRouter. Please top up your account or choose a free model (:free).",
                )

            elif response.status_code == 404:
                return (
                    False,
                    {},
                    f"Selected model '{model}' is currently unavailable or retired on OpenRouter.",
                )

            elif response.status_code == 429:
                return (
                    False,
                    {},
                    "Rate limit reached on OpenRouter. Please wait a moment before sending another message.",
                )

            elif response.status_code in [500, 502, 503, 504]:
                return (
                    False,
                    {},
                    f"Upstream provider error for model '{model}'. Please try another model or retry later.",
                )

            else:
                error_msg = "Unknown error from OpenRouter."
                try:
                    err_json = response.json()
                    error_msg = err_json.get("error", {}).get("message", error_msg)
                except Exception:
                    error_msg = response.text[:200]
                return (
                    False,
                    {},
                    f"OpenRouter Error ({response.status_code}): {error_msg}",
                )

        except requests.exceptions.Timeout:
            return (
                False,
                {},
                "Request timed out waiting for the AI model to respond. The model might be overloaded.",
            )
        except requests.exceptions.ConnectionError:
            return (
                False,
                {},
                "Network connection failed. Could not communicate with OpenRouter.",
            )
        except Exception as e:
            return (
                False,
                {},
                f"An unexpected error occurred: {str(e)}",
            )
