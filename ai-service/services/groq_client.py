import os
import requests
import time
import hashlib
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("GROQ_API_KEY")


class GroqClient:

    def __init__(self):
        self.url = "https://api.groq.com/openai/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        }
        self.model = "llama-3.3-70b-versatile"

    def generate(self, prompt):
        start_time = time.time()

        for attempt in range(3):  # retry logic
            try:
                data = {
                    "model": self.model,
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.3
                }

                response = requests.post(self.url, headers=self.headers, json=data)

                if response.status_code != 200:
                    raise Exception(f"API Error: {response.text}")

                result = response.json()

                content = result["choices"][0]["message"]["content"]

                end_time = time.time()

                return {
                    "result": content,
                    "meta": {
                        "model_used": self.model,
                        "response_time_ms": int((end_time - start_time) * 1000),
                        "tokens_used": result.get("usage", {}).get("total_tokens", 0),
                        "cached": False
                    }
                }

            except Exception as e:
                print(f"[Retry {attempt+1}] Error:", e)
                time.sleep(2)

        # fallback response
        return {
            "result": "AI temporarily unavailable. Please try again later.",
            "meta": {
                "model_used": self.model,
                "response_time_ms": 0,
                "tokens_used": 0,
                "cached": False,
                "is_fallback": True
            }
        }