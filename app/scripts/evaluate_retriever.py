import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.retriever.evaluator import run_evaluation
from app.config import get_settings

if __name__ == "__main__":
    settings = get_settings()

    metrics = run_evaluation(
        data_path=settings.retriever_data_path,
        eval_data_path="../data/retriever_eval_data.json",
        embedding_model=settings.retriever_embedding_model,
        top_k=settings.retriever_top_k,
        output_path="../data/retriever_eval_results.json"
    )
