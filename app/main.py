import uvicorn
import logging
from fastapi import FastAPI

from app.config import get_settings
from app.api.endpoints import router

settings = get_settings()
logging.basicConfig(level=settings.log_level)
logger = logging.getLogger(__name__)

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