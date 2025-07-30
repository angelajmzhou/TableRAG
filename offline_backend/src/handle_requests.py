import requests
import time
import json
from typing import Optional, Dict, Any
import sys

from dotenv import load_dotenv
from pathlib import Path
import os
from openai import OpenAI

env_path = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(dotenv_path=env_path)
# Load environment variables
google_api_keys = [os.getenv("GOOGLE_API_KEY1"), os.getenv("GOOGLE_API_KEY2"), os.getenv("GOOGLE_API_KEY3") , os.getenv("GOOGLE_API_KEY4") , os.getenv("GOOGLE_API_KEY5")]
v3_api_key = os.getenv("DEEPSEEK_API_KEY")
ind = 4  # Default to the first key

SCHEMA_DIR =get_llm_response '../data/schema'
#gemini-2.5-pro
model_request_config = {
    "gemini": {
        "endpoint": "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-pro:generateContent",
        "headers": {
            "Content-Type": "application/json",
            "x-goog-api-key": google_api_keys[ind]
        },
        "model": "models/gemini-2.5-pro"
    },
    "v3": {
        "endpoint": "https://openrouter.ai/api/v1/chat/completions",
        "headers": {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {v3_api_key}"
        },
        "model": "deepseek/deepseek-chat"
    }
    "claude": {
        "endpoint": "https://api.anthropic.com/v1/messages",
        "headers": {
            "Content-Type": "application/json",
            "x-api-key": os.getenv('CLAUDE_API_KEY'),
            "anthropic-version": "2023-06-01"
        },
        "model": "claude-3.5-sonnet-latest"
        "max_tokens": 8192
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
    global ind
    retry_delay = initial_retry_delay
    headers = headers or {"Content-Type": "application/json"}

    for attempt in range(max_retries + 1):
        try:

            response = requests.post(
                endpoint,
                json=payload,
                headers=headers,
                timeout=90
            )
            response.raise_for_status()
            resp_json_body = response.json()

            if model == "gemini":
                return resp_json_body["candidates"][0]["content"]["parts"][0]["text"]
            elif model == "v3":
                return resp_json_body["choices"][0]["message"]["content"]
            elif model == "claude":
                return resp_json_body["content"]
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
                print("Max retries reached. Exiting application.")
                sys.exit(1)
        time.sleep(retry_delay)
        retry_delay *= 2
        if model == "gemini":
            ind = (ind + 1) % len(google_api_keys)
            headers["x-goog-api-key"] = google_api_keys[ind]
           

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
    if not user_prompt.strip():
        raise ValueError("User prompt must not be empty.")
        return None
    print("using model:", model)
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
            messages.append({"role": "assistant", "content": system_prompt.strip()})
        messages.append({"role": "user", "content": user_prompt})
        print("Messages:", messages)
        payload = {
            "model": model_config["model"],
            "messages": messages,
            "temperature": 0.01
        }
    elif model == "claude":
        messages = []
        if system_prompt:
            messages.append({"role": "assistant", "content": system_prompt.strip()})
        messages.append({"role": "user", "content": user_prompt})
        payload = {
            "model": model_config["model"],
            "max_tokens": model_config.get("max_tokens", 8192),
            "messages": messages,
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