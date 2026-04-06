import uvicorn
import logging
from fastapi import FastAPI

from app.config import get_settings
from app.api.endpoints import router
from app.retriever.retriever import init_retriever

settings = get_settings()

logging.basicConfig(level=settings.log_level)
logger = logging.getLogger(__name__)

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("watchfiles").setLevel(logging.WARNING)

if settings.retriever_enabled:
    try:
        init_retriever(
            data_path=settings.retriever_data_path,
            embedding_model=settings.retriever_embedding_model,
            top_k=settings.retriever_top_k
        )
        logger.info("Ретривер успешно инициализирован")
    except Exception as e:
        logger.error(f"Ошибка инициализации ретривера: {e}")


# Создание приложения
app = FastAPI(
    title=settings.app_name
)
app.include_router(router)

if __name__ == "__main__":
    print("Запуск...")
    print(f"Swagger: http://127.0.0.1:8000/docs")
    uvicorn.run(
        "app.main:app",
        host="127.0.0.1",
        port=8000,
        reload=True
    )