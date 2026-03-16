from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # Настройки приложения
    app_name: str = "Nutrition AI"

    # Настройки Ollama
    ollama_url: str = "http://localhost:11434/v1"
    # Ollama не проверяет ключ
    ollama_api_key: str = "ollama_api_key"
    model_name: str = "llama3.2"
    max_tokens: int = 500
    default_temperature: float = 1.0

    # Настройки логирования
    log_level: str = "INFO"

    class Config:
        env_file = ".env"
        case_sensitive = False

@lru_cache()
def get_settings():
    return Settings()