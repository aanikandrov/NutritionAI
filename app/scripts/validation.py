import csv
import json
import httpx
from pathlib import Path

from app.config import get_settings
settings = get_settings()


def load_queries(csv_path: Path):
    """ Загружает запросы из CSV-файла """
    queries = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = row.get("query")
            if not text:
                continue
            queries.append({"text": text, "temperature": settings.default_temperature})
    return queries


def call_api(query_text: str, temperature: float):
    """ Отправляет запрос к API и возвращает результат """
    payload = {"text": query_text, "temperature": temperature}
    with httpx.Client() as client:
        try:
            response = client.post(settings.api_url, json=payload, timeout=30.0)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e), "status_code": getattr(response, "status_code", None)}


def save_results(queries, results, output_path):
    """ Сохраняет результаты в CSV """
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["запрос", "результат"])
        for q, r in zip(queries, results):
            writer.writerow([q["text"], json.dumps(r, ensure_ascii=False)])


def main():

    dataset_path = Path("../data/validation_data.csv")

    if not dataset_path.exists():
        print(f"Файл {dataset_path} не найден")

    queries = load_queries(dataset_path)
    if not queries:
        print("Нет данных для обработки")

    print(f"Загружено {len(queries)} запросов")

    results = []
    for index, query in enumerate(queries, 1):
        print(f"[{index}/{len(queries)}] {query['text']} ... ", end="")
        response = call_api(query["text"], query["temperature"])
        print("готово")
        results.append(response)

    output_path = Path(str(dataset_path.with_suffix("")) + "_results.csv")
    save_results(queries, results, output_path)
    print(f"Результаты сохранены в {output_path}")


if __name__ == "__main__":
    main()