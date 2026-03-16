import json
import logging
from pathlib import Path
from typing import Dict, Any

from openai import OpenAI, APIConnectionError, APIError
from fastapi import HTTPException
from openai.types.chat import ChatCompletionSystemMessageParam, ChatCompletionUserMessageParam
from openai.types.shared_params import ResponseFormatJSONSchema
from openai.types.shared_params.response_format_json_schema import JSONSchema

from app.config import get_settings
from app.schemas.schemas import NutritionResponse

logger = logging.getLogger(__name__)
settings = get_settings()

# Инициализация клиента
client = OpenAI(
    base_url=settings.ollama_url,
    api_key=settings.ollama_api_key
)

def load_prompt() -> str:
    """ Промпт из файла"""
    prompt_path = Path(__file__).parent / "prompts" / "nutrition_prompt.txt"
    with open(prompt_path, 'r', encoding='utf-8') as file:
        return file.read()

def load_examples() -> str:
    """ Примеры из файла"""
    examples_path = Path(__file__).parent / "prompts" / "examples_prompt.txt"
    with open(examples_path, 'r', encoding='utf-8') as file:
        return file.read()

# Кэшируем промпты при загрузке модуля
PROMPT = load_prompt()
EXAMPLES = load_examples()

def create_prompt(user_query: str) -> str:
    """ Создаёт промпт с подстановкой переменных """
    return PROMPT.format(
        examples=EXAMPLES,
        user_query=user_query
    )

def create_fallback_response(query_text: str) -> Dict[str, Any]:
    """ fallback-ответ при ошибках модели """
    return {
        "product": "unknown",
        "amount": 0,
        "unit": "грамм",
        "nutrition": {
            "weight": 0,
            "cal": 0,
            "protein": 0,
            "fat": 0,
            "carbs": 0
        }
    }

def clean_json_response(response_text: str) -> str:
    """ Очищает ответ от markdown """
    # Пробелы в начале и конце
    response_text = response_text.strip()
    # JSON-блок
    if response_text.startswith("```json"):
        response_text = response_text.replace("```json", "").replace("```", "").strip()
    # markdown-блок
    elif response_text.startswith("```"):
        response_text = response_text.replace("```", "").strip()
    return response_text


def validate_and_convert_types(llm_response: Dict[str, Any]) -> Dict[str, Any]:
    """Валидирует и преобразует типы данных"""

    main_fields = ["product", "amount", "unit", "nutrition"]
    for field in main_fields:
        if field not in llm_response:
            raise ValueError(f"Ответ не содержит поле '{field}'")

    nutrition_fields = ["weight", "cal", "protein", "fat", "carbs"]
    for field in nutrition_fields:
        if field not in llm_response["nutrition"]:
            raise ValueError(f"Ответ не содержит поле nutrition.{field}")

    # Преобразование типов
    try:
        llm_response["amount"] = float(llm_response["amount"])
        llm_response["nutrition"]["weight"] = float(llm_response["nutrition"]["weight"])
        llm_response["nutrition"]["cal"] = int(round(float(llm_response["nutrition"]["cal"])))
        llm_response["nutrition"]["protein"] = int(round(float(llm_response["nutrition"]["protein"])))
        llm_response["nutrition"]["fat"] = int(round(float(llm_response["nutrition"]["fat"])))
        llm_response["nutrition"]["carbs"] = int(round(float(llm_response["nutrition"]["carbs"])))

    except (ValueError, TypeError) as e:
        raise ValueError(f"Ошибка преобразования типов: {e}")

    return llm_response

async def call_llm(query_text: str, temperature: float) -> Dict[str, Any]:
    """ Отправляет запрос к LLM и возвращает JSON """
    try:
        # Создаем промпт
        prompt = create_prompt(query_text)

        system_message = "Ты помощник, который всегда отвечает только валидным JSON без пояснений."

        logger.info(f"запрос=\"{query_text}\" , model={settings.model_name}, t={temperature}")

        # JSON-схема
        json_schema = JSONSchema(
            name="nutrition_response",
            schema=NutritionResponse.model_json_schema()
        )

        # формат ответа
        response_format = ResponseFormatJSONSchema(
            type="json_schema",
            json_schema=json_schema
        )

        # Отправляем запрос к LLM
        response = client.chat.completions.create(
            # модель
            model=settings.model_name,
            messages=[
                # системное сообщение
                ChatCompletionSystemMessageParam(
                    role="system",
                    content=system_message
                ),
                # промпт
                ChatCompletionUserMessageParam(
                    role="user",
                    content=prompt
                )
            ],
            temperature=temperature,
            max_tokens=settings.max_tokens,
            response_format=response_format
        )

        # Получаем и очищаем ответ
        response_text = response.choices[0].message.content.strip()
        cleaned_text = clean_json_response(response_text)

        # logger.info(f"ответ: {response_text[:100]}...")
        # logger.info(f"ответ: {cleaned_text}")

        # Парсим JSON
        try:
            result = json.loads(cleaned_text)
            result = validate_and_convert_types(result)

            logger.info(
                f"результат: {result['product']}, {result['amount']}, {result['unit']}, "
                f"w={result['nutrition']['weight']}, cal={result['nutrition']['cal']}, "
                f"p={result['nutrition']['protein']}, f={result['nutrition']['fat']}, "
                f"carbs={result['nutrition']['carbs']}"
            )

            return result
        except json.JSONDecodeError as e:
            logger.error(f"Не удалось распарсить JSON: {e}")
            logger.error(f"Ответ модели: {response_text}")

            fallback = create_fallback_response(query_text)
            logger.info(f"fallback-ответ из-за ошибки парсинга JSON: {fallback}")
            # raise HTTPException(
            #     status_code=500,
            #     detail="Модель вернула некорректный JSON"
            # )
            return fallback
        except ValueError as e:
            logger.error(f"Ошибка валидации: {e}")

            fallback = create_fallback_response(query_text)
            logger.info(f"fallback-ответ из-за ошибки валидации: {fallback}")
            # raise HTTPException(
            #     status_code=500,
            #     detail=f"Модель вернула некорректные данные: {str(e)}"
            # )
            return fallback

    except APIConnectionError as e:
        logger.error(f"Ошибка подключения к LLM: {e}")
        raise HTTPException(
            status_code=503,
            detail="Сервис LLM недоступен"
        )
    except APIError as e:
        logger.error(f"Ошибка API LLM: {e}")
        raise HTTPException(
            status_code=502,
            detail=f"Ошибка при обработке запроса LLM: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Неожиданная ошибка: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Неожиданная ошибка"
        )