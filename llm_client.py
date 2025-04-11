import os
import requests
from dotenv import load_dotenv
import tiktoken

load_dotenv()

class LLMClient:
    def __init__(self, base_url: str = "https://openrouter.ai/api/v1", api_key: str = None):
        self.base_url = base_url
        self.api_key = os.getenv("OPENROUTER_KEY")

        if not self.api_key:
            raise ValueError("OpenRouter API key is missing. Please set OPENROUTER_API_KEY in your environment variables.")

    def request(self, payload: str, model: str, temperature: float = 0.1, max_tokens: int = 4096) -> tuple[str, int]:
        enc = tiktoken.encoding_for_model("gpt-4")
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        data = {
            "model": model,
            "messages": [{"role": "user", "content": payload}],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "tool_choice": "none",
        }

        response = requests.post(f"{self.base_url}/chat/completions", json=data, headers=headers)
        # print("response received: ", response)
        if response.status_code == 200:
            response_json = response.json()
            # print(response_json)
            message_content = response_json["choices"][0]["message"]["content"]
            # print(message_content)
            # token_count = response_json["usage"]["completion_tokens"]
            tokens = enc.encode(message_content)
            token_count = len(tokens)
            # Some models might not return token count
            return message_content, token_count
        else:
            print("Error:", response.json())
            return "Error in LLM request", 0


# if __name__ == "__main__":
#     api_key = os.getenv("OPENROUTER_KEY")
#     enc = tiktoken.encoding_for_model("gpt-4")
#     llm = LLMClient(api_key=api_key)
#     response, count = llm.request("A coin is heads up. Alejandro flips the coin. Carrie does not flip the coin. Darrell does not flip the coin. Lucas does not flip the coin. Is the coin still heads up?", "deepseek/deepseek-r1:free")
    
#     tokens = enc.encode("A coin is heads up. Alejandro flips the coin. Carrie does not flip the coin. Darrell does not flip the coin. Lucas does not flip the coin. Is the coin still heads up?")
#     print(f"Token count: {len(tokens)}")
#     print(response, count)  
