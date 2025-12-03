from typing import Any

import yaml


def read_yaml(path: str) -> dict[str, Any]:
    with open(path, 'r') as yml:
        config = yaml.safe_load(yml)
    return config


def calculate_top1_accuracy(num_corrects: int, num_golds: int) -> dict[str, float]:
    return {
        "top1_accuracy": num_corrects / num_golds if num_golds > 0 else float("nan")
    }


def calculate_recall_mrr(predictions: list[dict[str, Any]]) -> dict[str, int | float]:
    true, tp_1, tp_10, tp_50, tp_100, reciprocal_rank = 0, 0, 0, 0, 0, 0.
    for prediction in predictions:
        true += 1
        best_rank = 0
        indices = [result['id'] for result in prediction['predict']]
        for label in prediction['gold']:
            if label in indices:
                rank = indices.index(label) + 1
                if rank < best_rank or best_rank == 0:
                    best_rank = rank
        if best_rank > 0:
            if best_rank == 1:
                tp_1 += 1
            if best_rank <= 10:
                tp_10 += 1
            if best_rank <= 50:
                tp_50 += 1
            if best_rank <= 100:
                tp_100 += 1
            reciprocal_rank += 1 / best_rank

    return {
        "recall@1": tp_1 / true if true > 0 else 0.,
        "recall@10": tp_10 / true if true > 0 else 0.,
        "recall@50": tp_50 / true if true > 0 else 0.,
        "recall@100": tp_100 / true if true > 0 else 0.,
        "mrr@100": reciprocal_rank / true if true > 0. else 0.
    }
