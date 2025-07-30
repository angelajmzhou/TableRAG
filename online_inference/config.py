from dotenv import load_dotenv
import os

load_dotenv()
gemini_key = os.getenv("VERTEX_API_KEY", "")
deepseek_key = os.getenv("DEEPSEEK_API_KEY", "")

gemini_config = {
    "url": "https://generativelanguage.googleapis.com/v1beta/openai/",
    "model": "gemini-2.5-pro",
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

claude_config = {
    "url": "https://api.anthropic.com/v1/messages",
    "model": "claude-3-5-sonnet-20240620",
    "api_key": os.getenv('CLAUDE_API_KEY', '')
}


config_mapping = {
    "v3": v3_config,
    "gemini": gemini_config,
    "claude": claude_config
}