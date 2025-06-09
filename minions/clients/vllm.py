import logging
import os
from typing import Optional

from minions.clients.openai import OpenAIClient


class VLLMClient(OpenAIClient):
    """Client for a vLLM server using the OpenAI-compatible API."""

    def __init__(
        self,
        model_name: str = "llama-3",
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        base_url: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Initialize the vLLM client.

        Args:
            model_name: Name of the model to use (default: "llama-3").
            api_key: API key if required by the server. Defaults to the
                ``VLLM_API_KEY`` environment variable or ``"EMPTY"``.
            temperature: Sampling temperature (default: 0.0).
            max_tokens: Maximum number of tokens to generate (default: 4096).
            base_url: Base URL of the vLLM server. Defaults to the
                ``VLLM_BASE_URL`` environment variable or
                ``"http://localhost:8000/v1"``.
            **kwargs: Additional parameters passed to :class:`OpenAIClient`.
        """
        base_url = base_url or os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
        api_key = api_key or os.getenv("VLLM_API_KEY", "EMPTY")

        super().__init__(
            model_name=model_name,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            base_url=base_url,
            **kwargs,
        )

        # vLLM typically runs locally and may not require an API key, so we keep
        # logging minimal to avoid noise.
        self.logger.setLevel(logging.INFO)

