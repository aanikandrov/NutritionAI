from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # Настройки приложения
    app_name: str = "Nutrition AI"

    api_url: str = "http://127.0.0.1:8000/calculate"

    # Настройки Ollama
    ollama_url: str = "http://localhost:11434/v1"
    # Ollama не проверяет ключ
    ollama_api_key: str = "ollama_api_key"
    # model_name: str = "llama3.2"
    model_name: str = "mistral"
    max_tokens: int = 500
    default_temperature: float = 1.0

    # Настройки логирования
    log_level: str = "INFO"

    # Настройки ретривера
    retriever_enabled: bool = True
    retriever_data_path: str = "../data/products.json"
    retriever_embedding_model: str = "all-MiniLM-L6-v2"
    retriever_top_k: int = 5

    class Config:
        env_file = ".env"
        case_sensitive = False

@lru_cache()
def get_settings():
    return Settings()