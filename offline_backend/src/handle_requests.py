import requests
import time
import json
from typing import Optional, Dict, Any

from dotenv import load_dotenv
from pathlib import Path
import os
from openai import OpenAI

env_path = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(dotenv_path=env_path)

google_api_key = os.getenv("GOOGLE_API_KEY")
v3_api_key = os.getenv("DEEPSEEK_API_KEY")

SCHEMA_DIR = '../data/schema'

model_request_config = {
    "gemini": {
        "endpoint": "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent",
        "headers": {
            "Content-Type": "application/json",
            "x-goog-api-key": google_api_key
        },
        "model": "models/gemini-2.5-flash"
    },
    "v3": {
        "endpoint": "https://openrouter.ai/api/v1/chat/completions",
        "headers": {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {v3_api_key}"
        },
        "model": "deepseek/deepseek-chat"
    }
}

def call_llm_api(
    endpoint: str,
    payload: Dict[str, Any],
    headers: Optional[Dict[str, str]] = None,
    max_retries: int = 3,
    initial_retry_delay: float = 1.0,
    model: str = "gemini"
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

            if model == "gemini":
                return resp_json_body["candidates"][0]["content"]["parts"][0]["text"]
            elif model == "v3":
                return resp_json_body["choices"][0]["message"]["content"]
            else:
                raise ValueError(f"Unsupported model in response handler: {model}")

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

    if model == "gemini":
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

    elif model == "v3":
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt.strip()})
        messages.append({"role": "user", "content": user_prompt})
        payload = {
            "model": model_config["model"],
            "messages": messages,
            "temperature": 0.01
        }

    else:
        raise ValueError(f"Model '{model}' is not supported.")

    return call_llm_api(
        endpoint=model_endpoint,
        payload=payload,
        headers=model_headers,
        model=model
    )



if __name__ == "__main__":
    # Example usage
    system_prompt = "You are a helpful assistant."
    user_prompt = "What is the capital of France?"
    model = "v3"

    response = get_llm_response(system_prompt, user_prompt, model)
    if response:
        print("LLM Response:", response)
    else:
        print("Failed to get a response from the LLM.")