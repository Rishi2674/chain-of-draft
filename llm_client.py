import os
from dotenv import load_dotenv
from groq import Groq
import tiktoken
import random
import time

load_dotenv()

class LLMClient:
    def __init__(self, model: str = "llama3-70b-8192", temperature: float = 0.1, max_tokens: int = 4096):
        # Load multiple API keys
        self.api_keys = [
            os.getenv("GROQ_API_KEY_1"),
            os.getenv("GROQ_API_KEY_2"),
            os.getenv("GROQ_API_KEY_3"),
            os.getenv("GROQ_API_KEY_4"),
        ]
        self.api_keys = [key for key in self.api_keys if key]  # Filter out None
        if not self.api_keys:
            raise ValueError("No Groq API keys found in environment variables.")

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.tokenizer = tiktoken.encoding_for_model("gpt-4")

        self.current_key_index = 0
        self._set_client()

    def _set_client(self):
        self.client = Groq(api_key=self.api_keys[self.current_key_index])

    def _rotate_key(self):
        self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
        print(f"Rotating to API key {self.current_key_index + 1}")
        self._set_client()

    def request(self, prompt: str, retries: int = 4) -> tuple[str, int]:
        messages = [{"role": "user", "content": prompt}]
        for attempt in range(retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
                content = response.choices[0].message.content
                token_count = len(self.tokenizer.encode(content))
                return content, token_count
            except Exception as e:
                if "503" in str(e):
                    print(f"🚫 API error: {e}")
                    self._rotate_key()
                    time.sleep(1)  # brief wait before retrying
                else:
                    raise  # raise non-503 errors
        raise RuntimeError("All retries failed. Groq API seems unavailable.")

