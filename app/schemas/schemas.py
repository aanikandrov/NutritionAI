from pydantic import BaseModel, Field

class ProductQuery(BaseModel):
    """ Входящий запрос """
    text: str = Field(..., description="Описание продукта")
    temperature: float = Field(
        1.0,     # значение по умолчанию
        ge=0.0,  # нижняя граница
        le=2.0,  # верхняя граница
        description="Температура для LLM (0.0 - 2.0)"
    )

class NutritionInfo(BaseModel):
    """ Вес и КБЖУ """
    weight: int = Field(..., description="Вес в граммах")
    cal: int = Field(..., description="Калории")
    protein: int = Field(..., description="Белки в граммах")
    fat: int = Field(..., description="Жиры в граммах")
    carbs: int = Field(..., description="Углеводы в граммах")

class NutritionResponse(BaseModel):
    """ Ответ LLM """
    product: str = Field(..., description="Название продукта")
    amount: float = Field(..., description="Количество")
    unit: str = Field(..., description="Единица измерения")
    nutrition: NutritionInfo = Field(..., description="Питательная ценность")

class HealthResponse(BaseModel):
    """ Ответ health check """
    status: str
    ollama: str
    available_models: list[str] | None = None
    error: str | None = None

class RootResponse(BaseModel):
    """ root """
    message: str
    docs: str
    how_to_use: str