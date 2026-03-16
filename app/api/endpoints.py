import logging
from fastapi import APIRouter, HTTPException

from app.schemas.schemas import (
    ProductQuery,
    NutritionResponse,
    HealthResponse,
    RootResponse
)
from app.llm.llm_service import call_llm

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/", response_model=RootResponse)
async def root():
    """ Корневой эндпоинт для проверки работоспособности API """
    return {
        "message": "API подсчёта БЖУ работает!",
        "docs": "/docs",
        "how_to_use": "Отправьте POST запрос на /calculate с JSON вида {'text': 'яблоко 1 шт', 'temperature': 0.7}"
    }


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Эндпоинт для проверки здоровья сервиса и доступности Ollama.
    """
    try:
        # Проверяем, доступна ли Ollama
        from app.llm.llm_service import client
        response = client.models.list()
        return {
            "status": "healthy",
            "ollama": "connected",
            "available_models": [model.id for model in response.data]
        }
    except Exception as e:
        return {
            "status": "degraded",
            "ollama": "disconnected",
            "error": str(e)
        }


@router.post("/calculate", response_model=NutritionResponse)
async def calculate_macros(query: ProductQuery):
    """
    Основной эндпоинт для расчёта БЖУ.
    Принимает запрос с текстом продукта и температурой,
    возвращает структурированную информацию о БЖУ.
    """
    logger.info(f"запрос: \"{query.text}\", t={query.temperature}")

    try:
        # Вызываем LLM
        llm_response = await call_llm(query.text, query.temperature)
        # Возвращаем результат
        return llm_response

    except HTTPException:
        # Пробрасываем HTTP исключения дальше
        raise
    except Exception as e:
        logger.error(f"Неожиданная ошибка: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Внутренняя ошибка сервера: {str(e)}"
        )