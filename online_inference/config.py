from dotenv import load_dotenv
import os

load_dotenv()
gemini_key = os.getenv("VERTEX_API_KEY", "")
deepseek_key = os.getenv("DEEPSEEK_API_KEY", "")

gemini_config = {
    "url": "https://generativelanguage.googleapis.com/v1beta/openai/",
    "model": "gemini-2.0-flash",
    "api_key": gemini_key
}

sql_service_url = 'http://localhost:5000/get_tablerag_response' 


config_mapping = {
    "gemini": gemini_config
}

v3_config = {
    "url": "https://openrouter.ai/api/v1",
    "model": "deepseek/deepseek-chat-v3-0324:free",
    "api_key": deepseek_key
}

config_mapping = {
    "v3": v3_config,
    "gemini": gemini_config
}