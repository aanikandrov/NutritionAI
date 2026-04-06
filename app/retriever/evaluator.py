import json
import logging
from dataclasses import dataclass, asdict
from typing import List, Dict, Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """Результат одного запроса"""
    query: str
    expected_product: str
    expected_category: str
    retrieved_products: List[str]
    retrieved_categories: List[str]
    similarities: List[float]
    recall_at_1: bool
    recall_at_3: bool
    recall_at_5: bool
    reciprocal_rank: float

@dataclass
class EvaluationMetrics:
    """Агрегированные метрики оценки"""
    total_queries: int
    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    mean_reciprocal_rank: float
    detailed_results: List[Dict[str, Any]]


class RetrieverEvaluator:
    def __init__(self, retriever, eval_data_path: str):
        self.retriever = retriever
        self.eval_data = self._load_eval_data(eval_data_path)

    def _load_eval_data(self, path: str) -> List[Dict]:
        """Загружает данные для оценки"""
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def evaluate_single_query(self, query_data: Dict) -> RetrievalResult:
        """Оценивает один запрос"""
        query = query_data["query"]
        expected_product = query_data["expected_product"]
        expected_category = query_data["expected_category"]

        retrieved = self.retriever.retrieve(query)

        retrieved_products = [item["name"] for item in retrieved]
        retrieved_categories = [item.get("category", "") for item in retrieved]
        similarities = [item.get("similarity", 0.0) for item in retrieved]

        recall_at_1 = len(retrieved_products) > 0 and expected_product in retrieved_products[:1]
        recall_at_3 = len(retrieved_products) > 0 and expected_product in retrieved_products[:3]
        recall_at_5 = len(retrieved_products) > 0 and expected_product in retrieved_products[:5]

        reciprocal_rank = 0.0
        for i, prod in enumerate(retrieved_products[:10], start=1):
            if prod == expected_product:
                reciprocal_rank = 1.0 / i
                break

        return RetrievalResult(
            query=query,
            expected_product=expected_product,
            expected_category=expected_category,
            retrieved_products=retrieved_products,
            retrieved_categories=retrieved_categories,
            similarities=similarities,
            recall_at_1=recall_at_1,
            recall_at_3=recall_at_3,
            recall_at_5=recall_at_5,
            reciprocal_rank=reciprocal_rank
        )

    def evaluate(self, verbose: bool = True) -> EvaluationMetrics:
        """Запускает полную оценку"""
        results = []

        for query_data in self.eval_data:
            result = self.evaluate_single_query(query_data)
            results.append(result)

            if verbose:
                logger.info(f"\nЗапрос: '{result.query}'")
                logger.info(f"  Ожидаемый продукт: {result.expected_product}")
                logger.info(f"  Найденные продукты: {result.retrieved_products[:3]}")
                logger.info(f"  Recall@1: {result.recall_at_1}, Recall@3: {result.recall_at_3}, Recall@5: {result.recall_at_5}")
                logger.info(f"  MRR: {result.reciprocal_rank:.3f}")

        total = len(results)
        metrics = EvaluationMetrics(
            total_queries=total,
            recall_at_1=np.mean([r.recall_at_1 for r in results]),
            recall_at_3=np.mean([r.recall_at_3 for r in results]),
            recall_at_5=np.mean([r.recall_at_5 for r in results]),
            mean_reciprocal_rank=np.mean([r.reciprocal_rank for r in results]),
            detailed_results=[asdict(r) for r in results]
        )

        return metrics

    def print_summary(self, metrics: EvaluationMetrics):
        """Выводит сводку результатов"""
        print(f"Всего запросов: {metrics.total_queries}")
        print(f"\nМетрики:")
        print(f"  Recall@1:  {metrics.recall_at_1:.2%}")
        print(f"  Recall@3:  {metrics.recall_at_3:.2%}")
        print(f"  Recall@5:  {metrics.recall_at_5:.2%}")
        print(f"  MRR:       {metrics.mean_reciprocal_rank:.3f}")

        print(f"\nПроблемные запросы (Recall@3 = False):")
        for result in metrics.detailed_results:
            if not result['recall_at_3']:
                print(f"  - '{result['query']}': ожидался '{result['expected_product']}', "
                      f"найдены {result['retrieved_products'][:3]}")

    def save_results(self, metrics: EvaluationMetrics, output_path: str):
        """Сохраняет результаты в JSON файл"""
        output = {
            "summary": {
                "total_queries": metrics.total_queries,
                "recall_at_1": float(metrics.recall_at_1),
                "recall_at_3": float(metrics.recall_at_3),
                "recall_at_5": float(metrics.recall_at_5),
                "mean_reciprocal_rank": float(metrics.mean_reciprocal_rank)
            },
            "detailed_results": metrics.detailed_results,
            "config": {
                "top_k": self.retriever.top_k,
                "embedding_model": self.retriever.model._modules['0'].auto_model.config.name_or_path
            }
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)


def run_evaluation(
        data_path: str = "app/data/products.json",
        eval_data_path: str = "app/retriever/retriever_eval_data.json",
        embedding_model: str = "all-MiniLM-L6-v2",
        top_k: int = 5,
        output_path: str = "retriever_eval_results.json"
):
    """Запускает полную оценку ретривера"""
    from app.retriever.retriever import SemanticRetriever

    retriever = SemanticRetriever(data_path, embedding_model, top_k)
    evaluator = RetrieverEvaluator(retriever, eval_data_path)
    metrics = evaluator.evaluate(verbose=True)

    evaluator.print_summary(metrics)
    evaluator.save_results(metrics, output_path)

    return metrics


if __name__ == "__main__":
    run_evaluation()