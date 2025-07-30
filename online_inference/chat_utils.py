import json
import requests
from openai import OpenAI
import logging
from functools import wraps
from typing import Dict, Any, Optional
import os
import hashlib
import time
from config import *
import httpx
import google.generativeai as genai
from typing import Dict, Any, Optional
import anthropic


def init_logger(name='my_logger', level=logging.DEBUG, log_file='app.log') :
    """
    Initialize a logger with console and file handlers.

    Args:
        level(int): logging level(DEBUG, INFO, WARINING...)
    """
    logger = logging.getLogger(name)

    if logger.hasHandlers() :
        logger.handlers.clear()

    logger.setLevel(level=level)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)

    return logger

def convert_to_gemini_format(openai_messages: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
    gemini_messages = []
    for msg in openai_messages:
        role = msg["role"]
        content = msg["content"]
        if isinstance(content, str):
            parts = [{"text": content}]
        elif isinstance(content, list):
            parts = [{"text": part} if isinstance(part, str) else part for part in content]
        else:
            raise ValueError(f"Unexpected content type: {type(content)}")
        # Gemini expects role to be "user" or "model"
        if role == "assistant":
            role = "model"
        gemini_messages.append({"role": role, "parts": parts})
    return gemini_messages

def get_chat_result(messages: list[Dict[str, Any]], tools: Any = None, llm_config: Dict = gemini_config):
    """
    Get LLM result using Gemini.
    """
    if "gemini" in llm_config.get("model"):
        try:
            genai.configure(api_key=gemini_config.get("api_key", ""))
            model = genai.GenerativeModel(
                model_name=llm_config.get("model", "gemini-2.5-pro"),
                tools=tools
            )

            gemini_messages = convert_to_gemini_format(messages)

            response = model.generate_content(
                contents=gemini_messages,
                generation_config={"temperature": 0.1}
            )
            return response
        except Exception as e:
            raise RuntimeError(f"Gemini error: {e}")
    elif "v3" in llm_config.get("model"):
        client = OpenAI(
            api_key=llm_config.get('api_key', ''),
            base_url=llm_config.get('url', '')
        )
        try:
            chat_completion = client.chat.completions.create(
                messages=messages,
                model=llm_config.get('model', ''),
                temperature=0.01
            )

            if not chat_completion or not chat_completion.choices:
                print("LLM response is None or contains no choices.")
                print(chat_completion)
                return None
            print(chat_completion)
            return chat_completion.choices[0].message
        except Exception as e:
            print(f"LLM call failed: {e}")
            return None
    elif "claude" in llm_config.get("model"):
        try:
            client = anthropic.Anthropic(api_key=llm_config.get('api_key', ''))
            if tools is not None:
                chat_completion = client.messages.create(
                    model='claude-3-5-sonnet-20240620',
                    max_tokens=8192,
                    messages = messages, 
                    tools = tools, 
                    temperature = 0.01
                )
            else:
                chat_completion = client.messages.create(
                model='claude-3-5-sonnet-20240620',
                max_tokens=8192,
                messages = messages, 
                temperature = 0.01
            )
            if not chat_completion:
                print("LLM response is None.")
                return None
            return chat_completion
        except Exception as e:
            print(f"LLM call failed: {e}")
            return None


      