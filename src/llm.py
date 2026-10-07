import os
import time
import logging
from typing import Dict, Any, Optional, List
from dotenv import load_dotenv

load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("RAG_LLM")

try:
    from openai import OpenAI, APIError, RateLimitError, APITimeoutError
except ImportError:
    OpenAI = None

class LLMClient:
    """
    Wrapper for LLM API providers using OpenAI-compatible endpoints (OpenRouter, Groq, OpenAI).
    Includes robust error handling, retries with exponential backoff, and fallback mock support.
    """
    def __init__(
        self,
        provider: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        max_retries: int = 3,
        timeout: float = 30.0
    ):
        self.provider = provider or os.getenv("LLM_PROVIDER", "openrouter").lower()
        self.model = model or os.getenv("LLM_MODEL", "meta-llama/llama-3.2-3b-instruct:free")
        self.max_retries = max_retries
        self.timeout = timeout
        
        # Determine API key & base URL based on provider
        if self.provider == "openrouter":
            self.api_key = api_key or os.getenv("OPENROUTER_API_KEY", "")
            self.base_url = "https://openrouter.ai/api/v1"
        elif self.provider == "groq":
            self.api_key = api_key or os.getenv("GROQ_API_KEY", "")
            self.base_url = "https://api.groq.com/openai/v1"
        else:
            self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
            self.base_url = "https://api.openai.com/v1"
            
        self.client = None
        if self.api_key and OpenAI is not None:
            extra_headers = {}
            if self.provider == "openrouter":
                extra_headers = {
                    "HTTP-Referer": "https://github.com/amaljemli347-ctrl/ai-pipeline-project",
                    "X-Title": "Enterprise RAG Assistant"
                }
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                default_headers=extra_headers if extra_headers else None,
                timeout=self.timeout
            )
        else:
            if not self.api_key:
                logger.warning(
                    f"No API key found for provider '{self.provider}'. "
                    "LLMClient will operate in Fallback/Mock mode until an API key is supplied in .env."
                )

    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 500
    ) -> Dict[str, Any]:
        """
        Generates a completion with retry logic for rate limits and network timeouts.

        Returns:
            Dict containing:
                - "text": Generated response string
                - "latency": Time taken in seconds
                - "model": Model used
                - "status": "success", "error", or "mock"
                - "error": Error message if failed
        """
        start_time = time.time()
        
        # Fallback/Mock mode if no client configured
        if not self.client:
            latency = time.time() - start_time
            # Generate a context-aware mock answer based on prompt
            last_msg = messages[-1]["content"] if messages else ""
            if "NOT_FOUND" in last_msg or "does not contain" in last_msg.lower():
                mock_text = "I don't have enough information in the provided context to answer this query."
            else:
                mock_text = (
                    "[Mock LLM Response] Based on the retrieved context, "
                    "the requested topic is covered in the Wikipedia corpus."
                )
            return {
                "text": mock_text,
                "latency": latency,
                "model": f"{self.model} (mock)",
                "status": "mock",
                "error": None
            }

        delay = 2.0
        last_exception = None

        for attempt in range(1, self.max_retries + 1):
            try:
                logger.info(f"Sending request to LLM ({self.model}), attempt {attempt}/{self.max_retries}...")
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                latency = time.time() - start_time
                text = response.choices[0].message.content.strip()
                
                return {
                    "text": text,
                    "latency": latency,
                    "model": self.model,
                    "status": "success",
                    "error": None
                }

            except RateLimitError as e:
                last_exception = e
                logger.warning(f"Rate limit hit (429). Retrying in {delay}s... Error: {e}")
                time.sleep(delay)
                delay *= 2

            except APITimeoutError as e:
                last_exception = e
                logger.warning(f"API Timeout. Retrying in {delay}s... Error: {e}")
                time.sleep(delay)
                delay *= 1.5

            except APIError as e:
                last_exception = e
                # Check for 404 Model Not Found or invalid model ID
                if getattr(e, 'status_code', None) == 404 or "not found" in str(e).lower():
                    logger.error(f"Model 404 Not Found: '{self.model}'. Please update LLM_MODEL in .env with an active model ID.")
                    break
                logger.warning(f"API Error ({getattr(e, 'status_code', 'unknown')}): {e}. Retrying in {delay}s...")
                time.sleep(delay)

            except Exception as e:
                last_exception = e
                logger.error(f"Unexpected error calling LLM API: {e}")
                break

        latency = time.time() - start_time
        return {
            "text": "Error: Failed to obtain response from LLM API due to rate limits or connection errors.",
            "latency": latency,
            "model": self.model,
            "status": "error",
            "error": str(last_exception)
        }
