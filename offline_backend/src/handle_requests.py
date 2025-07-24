import requests
import time
import json
from typing import Optional, Dict, Any

from dotenv import load_dotenv
from pathlib import Path
import os
from openai import OpenAI

# Find .env two levels up
env_path = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(dotenv_path=env_path)

# Now use your variables
api_key = os.getenv("API_KEY")
print("API_KEY:", api_key)

SCHEMA_DIR = '../data/schema'
model_request_config = {
    "gemini": {
        "endpoint": "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent",
        "headers": {
            "Content-Type": "application/json",
            "x-goog-api-key": api_key
        },
        "model": "models/gemini-2.5-flash"
    }
}

def call_llm_api(
    endpoint: str,
    payload: Dict[str, Any],
    headers: Optional[Dict[str, str]] = None,
    max_retries: int = 3,
    initial_retry_delay: float = 1.0
) -> Optional[str]:
    retry_delay = initial_retry_delay
    headers = headers or {"Content-Type": "application/json"}

    for attempt in range(max_retries + 1):
        try:
            print("\n--- REQUEST PAYLOAD ---")
            print(json.dumps(payload, indent=2))
            print("\n--- HEADERS ---")
            print(headers)

            response = requests.post(
                endpoint,
                json=payload,
                headers=headers,
                timeout=30
            )
            response.raise_for_status()
            resp_json_body = response.json()

            return resp_json_body["candidates"][0]["content"]["parts"][0]["text"]

        except requests.exceptions.HTTPError as e:
            print(f"Attempt {attempt + 1} failed with HTTPError:")
            print("Status Code:", response.status_code)
            print("Response:", response.text)
            if attempt == max_retries:
                return None
        except Exception as e:
            print(f"Attempt {attempt + 1} failed. Retrying in {retry_delay:.1f}s... Error: {str(e)}")
            if attempt == max_retries:
                return None
        time.sleep(retry_delay)
        retry_delay *= 2

    return None


def get_llm_response(
    system_prompt: Optional[str],
    user_prompt: str,
    model: str = "gemini",
) -> Optional[str]:
    model_config = model_request_config.get(model)
    if not model_config:
        raise ValueError(f"Model '{model}' is not supported.")
    model_endpoint = model_config["endpoint"]
    model_headers = model_config["headers"]

    full_prompt = f"{system_prompt.strip()}\n\n{user_prompt}" if system_prompt else user_prompt

    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": full_prompt}]
            }
        ],
        "generationConfig": {
            "temperature": 0.01
        }
    }

    return call_llm_api(
        endpoint=model_endpoint,
        payload=payload,
        headers=model_headers
    )


if __name__ == "__main__":
    # Example usage
    system_prompt = "You are a helpful assistant."
    user_prompt = "What is the capital of France?"
    model = "gemini"

    response = get_llm_response(system_prompt, user_prompt, model)
    if response:
        print("LLM Response:", response)
    else:
        print("Failed to get a response from the LLM.")